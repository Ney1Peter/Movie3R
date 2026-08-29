#!/usr/bin/env python3
"""Probe whether the learned human-latent head reaches the published output.

This is an audit-only program.  It runs the frozen Bridge3R checkpoint twice
on the same causal prefix, changing only ``enable_v8_human_latent_corr``.  For
the first post-cut frame it exports the decoder human token before/after the
residual, the residual reported by the model, and the decoded SMPL outputs.
It also binds each probe to the already completed formal-90 caches so that the
model-internal effect can be compared with the final transaction artifact.

No GT, calibration, evaluator identity, or future post-cut frame is read.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model, set_event_indices  # noqa: E402
from versions.v15.harmony4d.run_harmony_case import frame_image_paths  # noqa: E402


SCHEMA = "Bridge3R-human-latent-head-audit-v1"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/"
    "v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth"
)
DEFAULT_MANIFEST = (
    REPO_ROOT.parents[0]
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
DEFAULT_FORMAL_ROOT = REPO_ROOT / "output/bridge3r_egohumans_ablation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--capture", default="002_fencing")
    parser.add_argument("--strata", nargs="+", default=("small", "medium", "extreme"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def selected_records(manifest: Path, capture: str, strata: list[str]) -> list[dict[str, Any]]:
    wanted = set(strata)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("capture")) != capture:
            continue
        stratum = str(row.get("angle_stratum"))
        if stratum in wanted and stratum not in selected:
            selected[stratum] = row
    missing = wanted.difference(selected)
    if missing:
        raise ValueError(f"Capture {capture} misses requested strata: {sorted(missing)}")
    return [selected[name] for name in strata]


def formal_runtime(root: Path, route: str, record: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    case_id = str(record["case_id"])
    matches = list((root / route / "test/predictions").glob(f"*/{case_id}.runtime.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {route} runtime for {case_id}, got {matches}")
    return matches[0], json.loads(matches[0].read_text(encoding="utf-8"))


def run_prefix(
    model: ARCroco3DStereo,
    views: list[dict[str, Any]],
    boundary: int,
    device: torch.device,
    enabled: bool,
) -> tuple[dict[str, np.ndarray], float]:
    model.enable_v8_human_latent_corr = bool(enabled)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        predictions, returned, debug = model.forward_recurrent_lighter(
            copy.deepcopy(views),
            str(device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    prediction = predictions[boundary]
    token_debug = debug[boundary]
    values = {
        "human_head_token": tensor_numpy(token_debug.get("human_head_tokens")),
        "human_token_delta_reported": tensor_numpy(
            prediction.get("v8_human_latent_corr_delta_applied")
        ),
        "human_token_delta_raw": tensor_numpy(
            prediction.get("v8_human_latent_corr_delta_raw")
        ),
        "human_token_gate": tensor_numpy(prediction.get("v8_human_latent_corr_gate")),
        "smpl_transl": tensor_numpy(prediction.get("smpl_transl")),
        "smpl_rotmat": tensor_numpy(prediction.get("smpl_rotmat")),
        "smpl_shape": tensor_numpy(prediction.get("smpl_shape")),
        "smpl_expression": tensor_numpy(prediction.get("smpl_expression")),
        "camera_pose": tensor_numpy(prediction.get("camera_pose")),
        "route_append": tensor_numpy(prediction.get("v9_pre_decoder_append")),
        "route_gate": tensor_numpy(prediction.get("v9_pre_decoder_effective_gate")),
    }
    del predictions, returned, debug, prediction, token_debug
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values, seconds


def max_abs(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or not a.size:
        return None
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def l2(a: np.ndarray) -> float | None:
    return None if not a.size else float(np.linalg.norm(a.astype(np.float64)))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    extracted_root = args.extracted_root.resolve()
    formal_root = args.formal_root.resolve()
    for path in (manifest, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    flags_before_mask = configure_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    reports: list[dict[str, Any]] = []
    for record in selected_records(manifest, str(args.capture), list(args.strata)):
        case_id = str(record["case_id"])
        sequence_root = extracted_root / str(record["capture_relative"])
        pre_paths, post_paths = frame_image_paths(sequence_root, record)
        boundary = int(record["boundary_index"])
        # Causal prefix only: all pre-cut context plus the first post-cut frame.
        prefix_paths = pre_paths + post_paths[:1]
        views = set_event_indices(
            gt_helpers.prepare_full_square_input(
                model, prefix_paths, SimpleNamespace(size=int(args.size))
            ),
            {boundary},
        )
        on, on_seconds = run_prefix(model, views, boundary, device, True)
        off, off_seconds = run_prefix(model, views, boundary, device, False)

        token_delta_observed = on["human_head_token"] - off["human_head_token"]
        reported = on["human_token_delta_reported"]
        if token_delta_observed.shape != reported.shape:
            residual_parity = None
        else:
            residual_parity = max_abs(token_delta_observed, reported)

        full_path, full_runtime = formal_runtime(formal_root, "formal90_full_replay", record)
        off_path, off_runtime = formal_runtime(
            formal_root, "formal90_human_residual_off", record
        )
        full_cache = full_path.with_name(full_path.name.removesuffix(".runtime.json") + ".npz")
        off_cache = off_path.with_name(off_path.name.removesuffix(".runtime.json") + ".npz")
        full_hash = str(full_runtime.get("cache_sha256") or sha256(full_cache))
        off_hash = str(off_runtime.get("cache_sha256") or sha256(off_cache))

        arrays: dict[str, np.ndarray] = {}
        for key, value in on.items():
            arrays[f"on__{key}"] = value
        for key, value in off.items():
            arrays[f"off__{key}"] = value
        arrays["observed__human_token_delta"] = token_delta_observed
        arrays["formal__full_cache_sha256"] = np.asarray(full_hash)
        arrays["formal__head_off_cache_sha256"] = np.asarray(off_hash)
        arrays["formal__cache_byte_identical"] = np.asarray(full_hash == off_hash, dtype=np.uint8)
        npz_path = output_dir / f"{case_id}.tensor_probe.npz"
        partial = npz_path.with_suffix(npz_path.suffix + ".partial")
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(partial, npz_path)

        report = {
            "schema_version": SCHEMA,
            "case_id": case_id,
            "capture": record["capture"],
            "angle_stratum": record["angle_stratum"],
            "angle_deg": record.get("camera_rotation_span_deg_evaluator_only"),
            "causal_prefix_frames": len(prefix_paths),
            "boundary_index": boundary,
            "future_post_frames_read": 0,
            "gt_or_calibration_read": False,
            "head_on_seconds": on_seconds,
            "head_off_seconds": off_seconds,
            "human_count": int(on["human_head_token"].shape[1])
            if on["human_head_token"].ndim >= 2
            else 0,
            "reported_delta_l2": l2(reported),
            "observed_token_delta_l2": l2(token_delta_observed),
            "reported_vs_observed_max_abs": residual_parity,
            "decoded_output_max_abs_on_vs_off": {
                key: max_abs(on[key], off[key])
                for key in (
                    "smpl_transl",
                    "smpl_rotmat",
                    "smpl_shape",
                    "smpl_expression",
                    "camera_pose",
                )
            },
            "event_route_append": on["route_append"],
            "event_route_gate": on["route_gate"],
            "formal_cache": {
                "full": str(full_cache),
                "head_off": str(off_cache),
                "full_sha256": full_hash,
                "head_off_sha256": off_hash,
                "byte_identical": full_hash == off_hash,
            },
            "npz": str(npz_path),
            "npz_sha256": sha256(npz_path),
        }
        json_path = output_dir / f"{case_id}.tensor_probe.json"
        json_path.write_text(
            json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(json.dumps(jsonable(report), ensure_ascii=False), flush=True)

    summary = {
        "schema_version": SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "capture": args.capture,
        "device": str(device),
        "model_flags_before_mask": flags_before_mask,
        "case_count": len(reports),
        "all_internal_token_deltas_nonzero": all(
            float(row["observed_token_delta_l2"] or 0.0) > 0.0 for row in reports
        ),
        "all_reported_residuals_match_injected_tokens": all(
            float(row["reported_vs_observed_max_abs"] or 0.0) <= 1e-6 for row in reports
        ),
        "all_formal_caches_byte_identical": all(
            bool(row["formal_cache"]["byte_identical"]) for row in reports
        ),
        "cases": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "cases": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
