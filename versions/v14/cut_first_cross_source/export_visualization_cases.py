#!/usr/bin/env python3
"""Export viewer-ready examples for the frozen cross-source cut-first model.

The shadow branch is used only to estimate a camera-derived B0. Its scene,
human and recurrent state are discarded. The exported payload instead applies
that one rigid B0 to every post-cut frame of an independent Human3R raw-reset
payload, matching the deployable runtime path evaluated by this experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
THIS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT, THIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from evaluate_cut_events import add_mhmr_inputs  # noqa: E402
from evaluate_four_source_b0 import load_views  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    model_batch_from_gt,
    set_event_indices,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/"
    "v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/eval_cross96_180/"
    "four_source_b0_evaluation.json"
)
DEFAULT_RAW_ROOT = (
    REPO_ROOT / "output/v10_candidate_selection/oracle_gt_4source/cases"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14_cut_first_cross_source/visualization_cross96"
)
DEFAULT_DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
DEFAULT_CASES = (
    "avatarrex_060_090_lbn1_1836_22070935_22053912",
    "thuman_150_180_thuman02_2691_cam02_cam14",
    "mvhuman100_150_180_100004_374_CC32871A017_CC32871A008",
    "mvhuman200_090_120_200001_442_22236222_22327117",
)
CUT_INDEX = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_case_rows(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in report["cases"]:
        if row.get("status") != "ok":
            continue
        rows[str(row["record"]["pattern_id"])] = row
    return rows


def homogeneous(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected 3x4/4x4 matrix, got {matrix.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3] = matrix
    return result


def compute_b0(
    model: ARCroco3DStereo,
    views: list[dict],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    add_mhmr_inputs(views)
    clean = todevice(model_batch_from_gt(views), device)
    shadow_views = set_event_indices(clean, {CUT_INDEX})
    raw_views = set_event_indices(clean[CUT_INDEX : CUT_INDEX + 1], set())
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow, _ = model.forward_recurrent_lighter(
            shadow_views, str(device), ret_state=False, use_ttt3r=False
        )
        raw, _ = model.forward_recurrent_lighter(
            raw_views, str(device), ret_state=False, use_ttt3r=False
        )

    boundary = (
        boundary_from_camera_predictions(shadow[CUT_INDEX], raw[0])[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    shadow_camera = homogeneous(camera_matrix(shadow[CUT_INDEX]))
    raw_camera = homogeneous(camera_matrix(raw[0]))
    reconstructed = boundary @ raw_camera
    parity = {
        "camera_translation_m": float(
            np.linalg.norm(reconstructed[:3, 3] - shadow_camera[:3, 3])
        ),
        "camera_matrix_max_abs": float(
            np.max(np.abs(reconstructed - shadow_camera))
        ),
    }
    if parity["camera_matrix_max_abs"] > 1e-5:
        raise RuntimeError(f"B0 camera parity failed: {parity}")
    return boundary, parity


def replace_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".new")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def transform_cached_vertices(path: Path, boundary: np.ndarray) -> None:
    with np.load(path, allow_pickle=True) as source:
        if "verts_world" not in source.files:
            return
        values = {key: source[key] for key in source.files}
    vertices = np.asarray(values["verts_world"], dtype=np.float32)
    values["verts_world"] = (
        np.einsum("ij,...j->...i", boundary[:3, :3], vertices)
        + boundary[:3, 3]
    ).astype(np.float32)
    replace_npz(path, values)


def export_corrected_payload(
    raw_payload: Path,
    destination: Path,
    boundary: np.ndarray,
    overwrite: bool,
) -> int:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        if destination.parent == destination or str(destination) in ("/", "/data"):
            raise ValueError(f"Refusing broad deletion target: {destination}")
        shutil.rmtree(destination)
    shutil.copytree(raw_payload, destination, copy_function=os.link)

    camera_files = sorted((destination / "camera").glob("*.npz"))
    if len(camera_files) < CUT_INDEX + 1:
        raise RuntimeError(f"Too few camera frames under {destination}")
    for index in range(CUT_INDEX, len(camera_files)):
        camera_path = destination / "camera" / f"{index:06d}.npz"
        with np.load(camera_path) as source:
            values = {key: source[key] for key in source.files}
        values["pose"] = (
            boundary @ np.asarray(values["pose"], dtype=np.float64)
        ).astype(np.float32)
        replace_npz(camera_path, values)
        transform_cached_vertices(
            destination / "smpl" / f"{index:06d}.npz", boundary
        )
    return len(camera_files)


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.report):
        if not path.is_file():
            raise FileNotFoundError(path)
    case_rows = read_case_rows(args.report)
    missing = sorted(set(args.cases) - set(case_rows))
    if missing:
        raise KeyError(f"Cases absent from evaluation report: {missing}")

    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.checkpoint)).to(device)
    model.eval()
    model_flags = configure_model(model)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    for index, pattern_id in enumerate(args.cases, start=1):
        row = case_rows[pattern_id]
        raw_payload = args.raw_root / pattern_id / "human3r_local_reset"
        if not raw_payload.is_dir():
            raise FileNotFoundError(raw_payload)
        views = load_views(row["record"], args)
        boundary, parity = compute_b0(model, views, device)
        corrected = args.output_root / pattern_id / "b0_corrected"
        frame_count = export_corrected_payload(
            raw_payload, corrected, boundary, bool(args.overwrite)
        )
        case_manifest = {
            "pattern_id": pattern_id,
            "source": row["source"],
            "record": row["record"],
            "cut_index": CUT_INDEX,
            "frame_count": frame_count,
            "raw_payload": str(raw_payload.resolve()),
            "corrected_payload": str(corrected.resolve()),
            "b0": boundary.tolist(),
            "parity": parity,
            "evaluation": row["methods"],
        }
        manifest_cases.append(case_manifest)
        case_path = corrected.parent / "manifest.json"
        case_path.write_text(
            json.dumps(case_manifest, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        metrics = row["methods"]["b0_runtime"]
        print(
            f"[{index}/{len(args.cases)}] {pattern_id}: "
            f"composite={metrics['camera_composite']:.4f}, "
            f"frames={frame_count} -> {corrected}",
            flush=True,
        )

    manifest = {
        "experiment": "v14_cross96_visualization_cases",
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_report": str(args.report.resolve()),
        "model_flags": model_flags,
        "cases": manifest_cases,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
