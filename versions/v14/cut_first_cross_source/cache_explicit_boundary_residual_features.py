#!/usr/bin/env python3
"""Cache frozen causal latents for a directly supervised explicit B0 residual.

The cache is intentionally separate from Human3R training.  It treats the
frozen cross96 model as a causal proposal encoder and stores only the first
post-cut quantities available to a runtime head:

  last-pre + first-post -> read-only shadow rollout / clean raw rollout
  -> B0 and frozen pose/correction tokens -> feature vector.

The supervised target is retained only in this offline cache:

  Delta* = inverse(B0) @ B_gt,

so a later residual head can emit a *right-composed* update
``B = B0 @ Delta`` in the new shot's local gauge.  This choice is gauge
equivariant: a common change of the old world gauge left-multiplies B0 and
B_gt but leaves Delta* unchanged.  The shadow recurrent state is never saved
or committed.
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


CHECKPOINT = REPO_ROOT / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/explicit_boundary_residual_features"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
# All tensors are pooled only over non-channel axes.  Their order is a frozen
# contract for the subsequent small residual head.
TOKEN_KEYS = (
    "v8_pose_prompt_corr_token",
    "v8_pose_prompt_pose_token_raw",
    "v8_pose_prompt_pose_token_corrected",
    "v8_pose_prompt_delta_raw",
    "v8_pose_prompt_delta_applied",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="One or more JSONL manifests")
    parser.add_argument("--split", required=True, help="Audit label only, e.g. train96 or vsp_dev")
    parser.add_argument("--model-path", type=Path, default=CHECKPOINT)
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


def so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * .5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * .5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-8:
            for index in range(3):
                if index != largest:
                    axis[index] = (rotation[largest, index] + rotation[index, largest]) / (4.0 * axis[largest])
        return angle * axis / max(float(np.linalg.norm(axis)), 1e-12)
    return angle / (2.0 * np.sin(angle)) * np.asarray(
        (rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]),
        dtype=np.float64,
    )


def camera_metrics(camera: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    translation = float(np.linalg.norm(camera[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(camera, target)
    return {
        "translation_m": translation,
        "rotation_deg": rotation,
        "composite": translation + .02 * rotation,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
    }


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {"count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)), "p95": float(np.quantile(array, .95))}


def load_event_views(record: dict[str, Any], args: argparse.Namespace) -> list[dict]:
    loader_args = SimpleNamespace(
        data_root=args.data_root, resolution=(512, 288), resize_mode="human3r_demo", boundary=2,
    )
    views = load_views(record, loader_args)
    if len(views) < 3:
        raise RuntimeError(f"Expected 3 views, got {len(views)}")
    return views[:3]


def pooled_token(prediction: dict[str, Any], key: str) -> np.ndarray:
    value = prediction.get(key)
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"Missing causal latent `{key}` in shadow prediction")
    value = value.detach().float().cpu().numpy()
    if value.ndim < 1:
        raise RuntimeError(f"Unexpected scalar latent `{key}`")
    # D=768 is the only unpooled semantic dimension; no GT quantity enters.
    return value.reshape(-1, value.shape[-1]).mean(axis=0).astype(np.float32, copy=False)


def geometry_scalars(shadow: dict[str, Any], raw: dict[str, Any], boundary: np.ndarray) -> np.ndarray:
    raw_points, shadow_points = raw.get("pts3d_in_other_view"), shadow.get("pts3d_in_other_view")
    values = [
        float(np.linalg.norm(boundary[:3, 3])),
        float(rotation_error_deg(boundary, np.eye(4, dtype=np.float64))),
    ]
    for key in ("v8_pose_prompt_gate", "v8_pose_prompt_learned_gate", "v8_pose_prompt_drift_logit"):
        tensor = shadow.get(key)
        values.append(float(tensor.detach().float().mean().cpu()) if torch.is_tensor(tensor) and tensor.numel() else 0.0)
    if raw_points is None or shadow_points is None:
        values.extend((0.0, 0.0, 0.0))
    else:
        rotation = torch.as_tensor(boundary[:3, :3], device=raw_points.device, dtype=raw_points.dtype)
        translation = torch.as_tensor(boundary[:3, 3], device=raw_points.device, dtype=raw_points.dtype)
        distance = torch.linalg.vector_norm(raw_points.detach().float() @ rotation.float().T + translation.float() - shadow_points.detach().float(), dim=-1)
        finite = distance[torch.isfinite(distance)]
        if finite.numel():
            values.extend((float(finite.mean().cpu()), float(finite.median().cpu()), float(torch.quantile(finite, .90).cpu())))
        else:
            values.extend((0.0, 0.0, 0.0))
    return np.asarray(values, dtype=np.float32)


def forward(model: ARCroco3DStereo, views: list[dict], device: torch.device) -> dict[str, Any]:
    clean = todevice(model_batch_from_gt(views), device)
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
    features = np.concatenate([*(pooled_token(shadow[2], key) for key in TOKEN_KEYS), geometry_scalars(shadow[2], raw[0], boundary)])
    return {"pre_camera": pre, "b0": b0, "feature": features, "parity": parity}


def evaluate_case(model: ARCroco3DStereo, record: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    views = load_event_views(record, args)
    add_mhmr_inputs(views)
    started = time.perf_counter()
    output = forward(model, views, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gt_pre = homogeneous(gt_pose_from_view(views[1]).detach().float().cpu().numpy())
    gt_post = homogeneous(gt_pose_from_view(views[2]).detach().float().cpu().numpy())
    target = output["pre_camera"] @ np.linalg.inv(gt_pre) @ gt_post
    # Right action: the explicit residual lives in the raw post-shot gauge.
    target_right = np.linalg.inv(output["b0"]) @ target
    target_se3 = np.concatenate((target_right[:3, 3], so3_log(target_right[:3, :3]))).astype(np.float32)
    return json_ready({
        "feature": output["feature"],
        "target_right_se3_training_only": target_se3,
        "b0_camera": output["b0"],
        "target_camera_evaluation_only": target,
        "b0_metrics": camera_metrics(output["b0"], target),
        "b0_parity_max_abs": output["parity"],
        "timing_seconds": time.perf_counter() - started,
    })


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    for path in args.records:
        if not path.is_file():
            raise FileNotFoundError(path)
    selected, seen, counts = [], set(), defaultdict(int)
    for path in args.records:
        for record in read_jsonl(path):
            source, identifier = str(record["source"]), str(record["pattern_id"])
            if source not in args.sources or identifier in seen:
                continue
            if args.max_cases_per_source and counts[source] >= args.max_cases_per_source:
                continue
            selected.append(record); seen.add(identifier); counts[source] += 1
    if not selected:
        raise RuntimeError("No selected records")
    root = args.output_dir / args.split
    cases_dir = root / "cases"; cases_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    rows, failures = [], []
    for index, record in enumerate(selected, 1):
        path = cases_dir / f"{safe_name(record['pattern_id'])}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not args.overwrite else None
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                row = evaluate_case(model, record, args, device)
                row |= {"status": "ok", "source": record["source"], "record": record}
            except Exception as error:
                row = {"status": "failed", "source": record.get("source"), "record": record, "error": repr(error), "traceback": traceback.format_exc()}
            path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        if row["status"] == "ok":
            rows.append(row)
            print(f"[{index:03d}/{len(selected):03d}] {record['source']} b0={row['b0_metrics']['composite']:.3f}", flush=True)
        else:
            failures.append(row); print(f"[{index:03d}/{len(selected):03d}] FAILED {row['error']}", flush=True)
        if device.type == "cuda" and index % 10 == 0:
            torch.cuda.empty_cache()
    report = {
        "experiment": "frozen_cross96_causal_latents_for_explicit_boundary_residual",
        "split": args.split, "records": [str(path) for path in args.records],
        "checkpoint": {"path": str(args.model_path), "sha256": sha256(args.model_path), "flags": flags},
        "feature_schema": {"token_keys": list(TOKEN_KEYS), "pooling": "mean over every axis except channel", "geometry_scalars": ["boundary_translation_norm_m", "boundary_rotation_deg", "pose_gate", "learned_pose_gate", "drift_logit", "raw_shadow_pointmap_mean_m", "median_m", "p90_m"]},
        "feature_dimension": int(len(rows[0]["feature"])) if rows else 0,
        "case_count": len(rows), "failures": failures,
        "b0_composite": stats([row["b0_metrics"]["composite"] for row in rows]),
        "b0_catastrophic_count": int(sum(row["b0_metrics"]["catastrophic"] for row in rows)),
        "target_right_translation_norm_m": stats([float(np.linalg.norm(row["target_right_se3_training_only"][:3])) for row in rows]),
        "target_right_rotation_deg": stats([float(np.degrees(np.linalg.norm(row["target_right_se3_training_only"][3:]))) for row in rows]),
        "cases": rows,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(root / "report.json")


if __name__ == "__main__":
    main()
