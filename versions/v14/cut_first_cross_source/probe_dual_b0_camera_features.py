#!/usr/bin/env python3
"""Cache causal features for cross96/old-B0 camera safety selection.

This is deliberately a *two-proposal cost ablation*, not the Movie3R main
runtime.  Both proposals use only last-pre and first-post frames; no future
frame, GT, or external pretrained network is used for proposal construction.
GT is read only after all candidate cameras and observable features are fixed.

The old candidate is converted into the cross96 raw/pre gauge before comparing
or selecting it.  Thus a later selector cannot accidentally compare camera
matrices expressed in different recurrent gauges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from evaluate_cut_events import add_mhmr_inputs, homogeneous, read_jsonl  # noqa: E402
from evaluate_four_source_b0 import load_views, safe_name  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    model_batch_from_gt,
    rotation_error_deg,
    set_event_indices,
)


CURRENT = REPO_ROOT / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth"
OLD = Path("/dev/shm/movie3r_v14_1/v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth")
DEFAULT_RECORDS = REPO_ROOT / "config/manifests/v14_vsp_pair_disjoint_20260802/dev_all.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14_cut_first_cross_source/dual_b0_camera_features_dev96"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-model", type=Path, default=CURRENT)
    parser.add_argument("--old-model", type=Path, default=OLD)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--max-cases-per-source", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95")} | {"count": 0}
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.quantile(array, .90)), "p95": float(np.quantile(array, .95)),
    }


def load_event_views(record: dict[str, Any], args: argparse.Namespace) -> list[dict]:
    loader_args = SimpleNamespace(
        data_root=args.data_root, resolution=(512, 288), resize_mode="human3r_demo", boundary=2,
    )
    views = load_views(record, loader_args)
    if len(views) < 3:
        raise RuntimeError(f"Expected 3 views, got {len(views)}")
    return views[:3]


def transform_camera(transform: np.ndarray, camera: np.ndarray) -> np.ndarray:
    return homogeneous(transform) @ homogeneous(camera)


def transform_points_torch(transform: np.ndarray, points: torch.Tensor) -> torch.Tensor:
    rotation = torch.as_tensor(transform[:3, :3], device=points.device, dtype=points.dtype)
    translation = torch.as_tensor(transform[:3, 3], device=points.device, dtype=points.dtype)
    return points @ rotation.T + translation


def scalar_tensor(prediction: dict[str, Any], key: str) -> float:
    value = prediction.get(key)
    if value is None or not torch.is_tensor(value):
        return float("nan")
    value = value.detach().float()
    return float(value.mean().cpu()) if value.numel() else float("nan")


def geometry_self_consistency(shadow: dict[str, Any], raw: dict[str, Any], boundary: np.ndarray) -> dict[str, float]:
    shadow_points = shadow.get("pts3d_in_other_view")
    raw_points = raw.get("pts3d_in_other_view")
    if shadow_points is None or raw_points is None:
        return {key: float("nan") for key in ("mean_m", "median_m", "p90_m", "weighted_mean_m", "confidence_mean")}
    mapped = transform_points_torch(boundary, raw_points.detach().float())
    difference = torch.linalg.vector_norm(mapped - shadow_points.detach().float(), dim=-1)
    valid = torch.isfinite(difference)
    values = difference[valid]
    if not values.numel():
        return {key: float("nan") for key in ("mean_m", "median_m", "p90_m", "weighted_mean_m", "confidence_mean")}
    confidence = torch.minimum(
        raw.get("conf", torch.ones_like(difference)).detach().float(),
        shadow.get("conf", torch.ones_like(difference)).detach().float(),
    )
    weights = confidence[valid].clamp_min(0.0)
    return {
        "mean_m": float(values.mean().cpu()),
        "median_m": float(values.median().cpu()),
        "p90_m": float(torch.quantile(values, .90).cpu()),
        "weighted_mean_m": float((values * weights).sum().cpu() / weights.sum().clamp_min(1e-8).cpu()),
        "confidence_mean": float(weights.mean().cpu()),
    }


def proposal_features(shadow: dict[str, Any], raw: dict[str, Any], boundary: np.ndarray) -> dict[str, Any]:
    identity = np.eye(4, dtype=np.float64)
    return {
        "boundary_translation_norm_m": float(np.linalg.norm(boundary[:3, 3])),
        "boundary_rotation_deg": rotation_error_deg(boundary, identity),
        "geometry_self_consistency": geometry_self_consistency(shadow, raw, boundary),
        "token": {
            key: scalar_tensor(shadow, key)
            for key in (
                "v8_pose_prompt_delta_norm", "v8_pose_prompt_learned_gate", "v8_pose_prompt_gate",
                "v8_pose_prompt_drift_logit", "v8_human_latent_corr_delta_norm",
                "v8_human_latent_corr_learned_gate",
            )
        },
    }


def camera_metrics(camera: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    translation = float(np.linalg.norm(camera[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(camera, target)
    return {
        "translation_m": translation,
        "rotation_deg": rotation,
        "composite": translation + .02 * rotation,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
    }


def forward(model: ARCroco3DStereo, gt_views: list[dict], device: torch.device) -> dict[str, Any]:
    clean = todevice(model_batch_from_gt(gt_views), device)
    shadow_views = set_event_indices(clean, {2})
    raw_views = set_event_indices(clean[2:3], set())
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow, _ = model.forward_recurrent_lighter(shadow_views, str(device), ret_state=False, use_ttt3r=False)
        raw, _ = model.forward_recurrent_lighter(raw_views, str(device), ret_state=False, use_ttt3r=False)
    pre = homogeneous(camera_matrix(shadow[1]))
    raw_camera = homogeneous(camera_matrix(raw[0]))
    shadow_camera = homogeneous(camera_matrix(shadow[2]))
    boundary = boundary_from_camera_predictions(shadow[2], raw[0])[0].detach().float().cpu().numpy().astype(np.float64)
    b0 = boundary @ raw_camera
    parity = float(np.max(np.abs(b0 - shadow_camera)))
    if parity > 1e-5:
        raise RuntimeError(f"B0 parity failed ({parity})")
    return {
        "pre_camera": pre, "raw_camera": raw_camera, "shadow_camera": shadow_camera,
        "boundary": boundary, "features": proposal_features(shadow[2], raw[0], boundary),
    }


def evaluate_case(current: ARCroco3DStereo, old: ARCroco3DStereo, views: list[dict], device: torch.device) -> dict[str, Any]:
    add_mhmr_inputs(views)
    started = time.perf_counter()
    current_out = forward(current, views, device)
    old_out = forward(old, views, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # Current is the only commit gauge.  Convert the old B0 candidate from
    # old-raw -> old-pre into current-raw -> current-pre before comparison.
    old_in_current = (
        current_out["pre_camera"] @ np.linalg.inv(old_out["pre_camera"])
        @ old_out["boundary"] @ old_out["raw_camera"] @ np.linalg.inv(current_out["raw_camera"])
    )
    gt_pre = homogeneous(gt_pose_from_view(views[1]).detach().float().cpu().numpy())
    gt_post = homogeneous(gt_pose_from_view(views[2]).detach().float().cpu().numpy())
    target = current_out["pre_camera"] @ np.linalg.inv(gt_pre) @ gt_post
    disagreement = {
        "translation_m": float(np.linalg.norm(current_out["shadow_camera"][:3, 3] - old_in_current[:3, 3])),
        "rotation_deg": rotation_error_deg(current_out["shadow_camera"], old_in_current),
    }
    old_own_target = old_out["pre_camera"] @ np.linalg.inv(gt_pre) @ gt_post
    return json_ready({
        "timing_seconds": time.perf_counter() - started,
        "current": {
            "features": current_out["features"],
            "metrics_in_current_gauge": camera_metrics(current_out["shadow_camera"], target),
        },
        "old": {
            "features": old_out["features"],
            "metrics_adapted_to_current_gauge": camera_metrics(old_in_current, target),
            "metrics_own_gauge_parity": camera_metrics(old_out["shadow_camera"], old_own_target),
        },
        "proposal_disagreement": disagreement,
        # Small matrices are retained solely for a later CPU-only fixed SE(3)
        # interpolation probe.  They are never GT-derived runtime inputs.
        "cameras_in_current_gauge": {
            "cross96_b0": current_out["shadow_camera"],
            "old_b0_adapted": old_in_current,
            "target_evaluation_only": target,
        },
        "raw_camera_cross_model_difference": {
            "translation_m": float(np.linalg.norm(current_out["raw_camera"][:3, 3] - old_out["raw_camera"][:3, 3])),
            "rotation_deg": rotation_error_deg(current_out["raw_camera"], old_out["raw_camera"]),
        },
        "pre_camera_cross_model_difference": {
            "translation_m": float(np.linalg.norm(current_out["pre_camera"][:3, 3] - old_out["pre_camera"][:3, 3])),
            "rotation_deg": rotation_error_deg(current_out["pre_camera"], old_out["pre_camera"]),
        },
    })


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"case_count": len(rows), "methods": {}}
    for name, metric_key in (("cross96", "metrics_in_current_gauge"), ("old_adapted", "metrics_adapted_to_current_gauge")):
        values = [row["current"][metric_key] if name == "cross96" else row["old"][metric_key] for row in rows]
        result["methods"][name] = {
            key: stats([float(row[key]) for row in values]) for key in ("translation_m", "rotation_deg", "composite")
        }
        result["methods"][name]["catastrophic_count"] = int(sum(row["catastrophic"] for row in values))
    result["disagreement"] = {
        key: stats([float(row["proposal_disagreement"][key]) for row in rows])
        for key in ("translation_m", "rotation_deg")
    }
    return result


def main() -> None:
    args = parse_args()
    for path in (args.current_model, args.old_model, args.records):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"; cases_dir.mkdir(parents=True, exist_ok=True)
    selected, counts = [], defaultdict(int)
    for record in read_jsonl(args.records):
        source = str(record["source"])
        if source not in args.sources or (args.max_cases_per_source and counts[source] >= args.max_cases_per_source):
            continue
        selected.append(record); counts[source] += 1
    device = torch.device(args.device)
    current = ARCroco3DStereo.from_pretrained(str(args.current_model)).to(device)
    current_flags = configure_model(current)
    old = ARCroco3DStereo.from_pretrained(str(args.old_model)).to(device)
    old_flags = configure_model(old)
    rows, failures = [], []
    for index, record in enumerate(selected, 1):
        path = cases_dir / f"{safe_name(record['pattern_id'])}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not args.overwrite else None
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                row = evaluate_case(current, old, load_event_views(record, args), device)
                row.update({"status": "ok", "source": record["source"], "record": record})
            except Exception as error:
                row = {"status": "failed", "source": record["source"], "record": record, "error": repr(error), "traceback": traceback.format_exc()}
            path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        if row["status"] == "ok":
            rows.append(row)
            print(f"[{index:03d}/{len(selected):03d}] {record['source']} current={row['current']['metrics_in_current_gauge']['composite']:.3f} old={row['old']['metrics_adapted_to_current_gauge']['composite']:.3f}", flush=True)
        else:
            failures.append(row); print(f"[{index:03d}/{len(selected):03d}] FAILED {row['error']}", flush=True)
        if device.type == "cuda" and index % 5 == 0:
            torch.cuda.empty_cache()
    report = json_ready({
        "experiment": "dual_b0_camera_features", "records": str(args.records),
        "checkpoints": {
            "cross96": {"path": str(args.current_model), "sha256": sha256(args.current_model), "flags": current_flags},
            "old": {"path": str(args.old_model), "sha256": sha256(args.old_model), "flags": old_flags},
        },
        "summary": summarize(rows),
        "by_source": {source: summarize([row for row in rows if row["source"] == source]) for source in args.sources},
        "failures": failures, "cases": rows,
    })
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
