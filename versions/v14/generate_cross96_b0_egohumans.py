#!/usr/bin/env python3
"""Generate strictly causal cross96 B0 boundaries for a frozen EgoHumans manifest.

This program deliberately contains no DA3, SLAM, ReID, GT camera, or GT human
dependency.  For each boundary it uses only the five pre-cut RGB frames and
the first post-cut RGB frame:

    clean previous-shot frames + corrected first post frame -> shadow camera
    fresh raw first post frame                          -> raw camera
    B0 = C_shadow @ inverse(C_raw)

The shadow state and every shadow human/scene output are discarded.  The JSON
records are consumed later by the BRTC evaluator, where ground truth first
appears only after all B0 and person-refinement actions have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from versions.v13.gt_id_consensus import prepare_full_square_input  # noqa: E402
from versions.v14.run_v14_2_multihuman_sequence import run_rollout  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6"
    / "checkpoint-final.pth"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "config/manifests/v14_cross96_brtc_egohumans_confirmation_20260803.json"
)
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble")
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/cross96_b0_egohumans_confirmation"
)
CROSS96_CHECKPOINT_SHA256 = "05274f7b4841f6ebc73f2f5bdb419d63d272396724db886b6e10987d7210a144"
OLD_B0_CHECKPOINT = REPO_ROOT / "checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth"
OLD_B0_CHECKPOINT_SHA256 = "8379243216775adbc886d00e6f93b6492f7d8f1bd67adb4e8ad6fbdd84e47123"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(path: Path) -> str:
    """Use only previously frozen digests; hash any noncanonical override."""
    if path.resolve() == DEFAULT_CHECKPOINT.resolve():
        return CROSS96_CHECKPOINT_SHA256
    if path.resolve() == OLD_B0_CHECKPOINT.resolve():
        return OLD_B0_CHECKPOINT_SHA256
    return sha256(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chains = payload.get("chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError("Manifest must contain a nonempty chains list")
    for chain_index, chain in enumerate(chains):
        shots = chain.get("shots", [])
        if len(shots) != 3:
            raise ValueError(f"chain {chain_index} must have exactly three shots")
        for shot_index, shot in enumerate(shots):
            camera, frames = shot.get("camera"), shot.get("frames")
            if not isinstance(camera, str) or not isinstance(frames, list) or len(frames) < 2:
                raise ValueError(f"invalid shot {chain_index}:{shot_index}")
            if frames != list(range(int(frames[0]), int(frames[0]) + len(frames))):
                raise ValueError(f"shot {chain_index}:{shot_index} frames must be consecutive")
        if int(shots[0]["frames"][-1]) != int(shots[1]["frames"][0]):
            raise ValueError(f"chain {chain_index} first cut must repeat timestamp")
        if int(shots[1]["frames"][-1]) != int(shots[2]["frames"][0]):
            raise ValueError(f"chain {chain_index} second cut must repeat timestamp")
    return payload


def image_paths(data_root: Path, shot: dict[str, Any]) -> list[Path]:
    paths = [
        data_root / f"exo/{shot['camera']}/images/{int(frame):05d}.jpg"
        for frame in shot["frames"]
    ]
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    return paths


def ensure_workspace(path: Path) -> None:
    if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"Output must stay in Movie3R workspace: {path}")


def generate_case(
    model: ARCroco3DStereo,
    args: argparse.Namespace,
    chain_index: int,
    cut_index: int,
    pre: dict[str, Any],
    post: dict[str, Any],
) -> dict[str, Any]:
    pre_paths, post_paths = image_paths(args.data_root, pre), image_paths(args.data_root, post)
    shadow_views = set_event_indices(
        prepare_full_square_input(model, pre_paths + post_paths[:1], SimpleNamespace(size=args.size)),
        {len(pre_paths)},
    )
    raw_views = set_event_indices(
        prepare_full_square_input(model, post_paths[:1], SimpleNamespace(size=args.size)), set()
    )
    shadow, _, _, shadow_seconds = run_rollout(
        model, shadow_views, str(args.device), f"cross96_b0_chain{chain_index}_cut{cut_index}_shadow"
    )
    raw, _, _, raw_seconds = run_rollout(
        model, raw_views, str(args.device), f"cross96_b0_chain{chain_index}_cut{cut_index}_raw"
    )
    boundary = (
        boundary_from_camera_predictions(shadow[-1], raw[0])[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    shadow_camera = camera_matrix(shadow[-1]).astype(np.float64)
    raw_camera = camera_matrix(raw[0]).astype(np.float64)
    b0_camera = boundary @ raw_camera
    max_difference = float(np.max(np.abs(shadow_camera - b0_camera)))
    if max_difference > 1e-5:
        raise RuntimeError(f"B0 camera replay mismatch {max_difference:.3e}")
    return {
        "case_key": f"chain{chain_index}_cut{cut_index}_{pre['camera']}_{post['camera']}",
        "chain_index": int(chain_index),
        "cut_index": int(cut_index),
        "runtime_input_audit": {
            "pre_rgb_frames": [str(path) for path in pre_paths],
            "first_post_rgb": str(post_paths[0]),
            "future_post_frames_used": 0,
            "gt_used": False,
            "external_pretrained_models": [],
            "shadow_state_committed": False,
        },
        "boundaries": {"b0": boundary},
        "camera_replay": {
            "shadow_camera": shadow_camera,
            "raw_first_camera": raw_camera,
            "b0_raw_first_camera": b0_camera,
            "max_abs_difference": max_difference,
        },
        "timing_seconds": {"shadow": float(shadow_seconds), "raw_first": float(raw_seconds)},
    }


def main() -> None:
    args = parse_args()
    for path in (args.output_dir, args.manifest):
        ensure_workspace(path)
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    manifest = load_manifest(args.manifest)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
    flags = configure_model(model)
    reports = []
    try:
        for chain_index, chain in enumerate(manifest["chains"]):
            shots = chain["shots"]
            for cut_index in (0, 1):
                pre, post = shots[cut_index], shots[cut_index + 1]
                destination = case_dir / (
                    f"chain{chain_index}_cut{cut_index}_{pre['camera']}_{post['camera']}.json"
                )
                if destination.is_file() and not args.overwrite:
                    report = json.loads(destination.read_text(encoding="utf-8"))
                    print(f">> reuse {destination.name}", flush=True)
                else:
                    report = generate_case(model, args, chain_index, cut_index, pre, post)
                    destination.write_text(
                        json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    print(f">> wrote {destination.name}", flush=True)
                reports.append(report)
    finally:
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    index = {
        "title": "cross96 causal B0 boundaries for frozen EgoHumans confirmation",
        "checkpoint": {"path": args.model_path, "sha256": checkpoint_sha256(args.model_path)},
        "manifest": {"path": args.manifest, "sha256": sha256(args.manifest)},
        "execution": {
            "device": str(args.device),
            "da3_used": False,
            "external_pretrained_models": [],
            "model_flags": flags,
            "wall_seconds": time.perf_counter() - started,
        },
        "runtime_contract": {
            "first_post_cut_only": True,
            "future_post_frames_used": 0,
            "shadow_state_committed": False,
            "gt_used": False,
        },
        "case_count": len(reports),
        "camera_replay_max_abs": float(max(row["camera_replay"]["max_abs_difference"] for row in reports)),
        "case_reports": [row["case_key"] for row in reports],
    }
    index_path = args.output_dir / "index.json"
    index_path.write_text(
        json.dumps(jsonable(index), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f">> wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
