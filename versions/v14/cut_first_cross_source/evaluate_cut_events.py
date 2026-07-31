#!/usr/bin/env python3
"""Evaluate first-post-cut correction and runtime B0 on cut-event manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    gt_head_world,
    model_batch_from_gt,
    predicted_head_world,
    rotation_error_deg,
    set_event_indices,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "config/manifests/v14_1_cut_event/ten"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14_cut_first_cross_source/eval_current_single_frozen10"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
SOURCE_ORDER = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
SOURCE_SPECS = {
    "avatarrex": ("avatarrex.jsonl", "Training"),
    "thuman": ("thuman.jsonl", "Training"),
    "mvhuman100": ("mvhuman100.jsonl", "Training/mvhuman"),
    "mvhuman200": ("mvhuman200.jsonl", "Training/mvhuman"),
}
METHODS = ("raw_reset", "shadow_event", "b0_runtime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_ORDER, default=SOURCE_ORDER)
    parser.add_argument("--max-cases-per-source", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def raw_calibration_roots() -> dict[str, str]:
    root = DATA_ROOT / "AvatarReX_raw_meta"
    return {name: str(root / name) for name in ("lbn1", "lbn2", "zzr", "zxc")}


def add_mhmr_inputs(views: list[dict], model_resolution: int = 896) -> None:
    from dust3r.utils.geometry import resize_camera_intrinsics
    from dust3r.utils.image import pad_image

    images = torch.stack([view["img"] for view in views], dim=0)
    images = images.view(-1, *images.shape[2:])
    intrinsics = torch.stack([view["camera_intrinsics"] for view in views], dim=0)
    intrinsics = intrinsics.view(-1, *intrinsics.shape[2:])
    intrinsics_mhmr = resize_camera_intrinsics(
        intrinsics, *images.shape[2:], model_resolution
    )
    images_mhmr = pad_image(images, model_resolution)
    for view, image_mhmr, intrinsic_mhmr in zip(
        views,
        images_mhmr.chunk(len(views), dim=0),
        intrinsics_mhmr.chunk(len(views), dim=0),
    ):
        view["img_mhmr"] = image_mhmr
        view["K_mhmr"] = intrinsic_mhmr


def make_loader(args: argparse.Namespace, source: str) -> tuple[DataLoader, list[dict]]:
    from dust3r.datasets.avatarrex import AvatarReX_Pattern

    filename, split = SOURCE_SPECS[source]
    path = args.manifest_root / filename
    records = read_jsonl(path)
    if args.max_cases_per_source:
        records = records[: args.max_cases_per_source]
    dataset = AvatarReX_Pattern(
        allow_repeat=True,
        split=split,
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=512,
        resize_mode="human3r_demo",
        num_views=3,
        seed=14900 + SOURCE_ORDER.index(source),
        n_corres=0,
        fixed_samples=records,
        load_da3_depth=False,
        raw_calibration_root=raw_calibration_roots() if source in ("avatarrex", "thuman") else None,
        max_humans=1,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    if len(dataset) != len(records):
        raise RuntimeError(f"{source}: dataset has {len(dataset)} rows, expected {len(records)}")
    return loader, records


def homogeneous(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected 3x4/4x4 camera matrix, got {matrix.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3] = matrix
    return result


def evaluate_pose(
    camera: np.ndarray,
    prediction: dict,
    target_camera: np.ndarray,
    target_head: np.ndarray | None,
) -> dict[str, Any]:
    camera = homogeneous(camera)
    target_camera = homogeneous(target_camera)
    translation = float(np.linalg.norm(camera[:3, 3] - target_camera[:3, 3]))
    rotation = rotation_error_deg(camera, target_camera)
    head = predicted_head_world(prediction, camera)
    human = (
        float(np.linalg.norm(head - target_head))
        if head is not None and target_head is not None
        else float("nan")
    )
    return {
        "camera_translation_m": translation,
        "camera_rotation_deg": rotation,
        "camera_composite": translation + 0.02 * rotation,
        "human_head_m": human,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
    }


def evaluate_batch(
    model: ARCroco3DStereo,
    smpl_layer: SMPL_Layer,
    gt_views: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    add_mhmr_inputs(gt_views)
    clean = todevice(model_batch_from_gt(gt_views), device)
    shadow_views = set_event_indices(clean, {2})
    raw_views = set_event_indices(clean[2:3], set())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow, _ = model.forward_recurrent_lighter(
            shadow_views, str(device), ret_state=False, use_ttt3r=False
        )
        raw, _ = model.forward_recurrent_lighter(
            raw_views, str(device), ret_state=False, use_ttt3r=False
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    pre_camera = homogeneous(camera_matrix(shadow[1]))
    shadow_camera = homogeneous(camera_matrix(shadow[2]))
    raw_camera = homogeneous(camera_matrix(raw[0]))
    boundary = (
        boundary_from_camera_predictions(shadow[2], raw[0])[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    b0_camera = boundary @ raw_camera

    gt_pre = homogeneous(gt_pose_from_view(gt_views[1]).cpu().numpy())
    gt_post = homogeneous(gt_pose_from_view(gt_views[2]).cpu().numpy())
    evaluation_gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_camera = evaluation_gauge @ gt_post
    target_head = gt_head_world(gt_views[2], evaluation_gauge, smpl_layer)

    shadow_metrics = evaluate_pose(
        shadow_camera, shadow[2], target_camera, target_head
    )
    b0_metrics = evaluate_pose(b0_camera, raw[0], target_camera, target_head)
    camera_parity = {
        "translation_m": float(
            np.linalg.norm(shadow_camera[:3, 3] - b0_camera[:3, 3])
        ),
        "matrix_max_abs": float(np.max(np.abs(shadow_camera - b0_camera))),
    }
    if camera_parity["matrix_max_abs"] > 1e-5:
        raise RuntimeError(f"B0 camera parity failed: {camera_parity}")
    return {
        "timing_seconds": elapsed,
        "methods": {
            "raw_reset": evaluate_pose(raw_camera, raw[0], target_camera, target_head),
            "shadow_event": shadow_metrics,
            "b0_runtime": b0_metrics,
        },
        "b0_camera_parity": camera_parity,
    }


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "camera_translation_m",
        "camera_rotation_deg",
        "camera_composite",
        "human_head_m",
    )
    result: dict[str, Any] = {"case_count": len(rows), "methods": {}}
    for method in METHODS:
        summary = {
            metric: stats([row["methods"][method][metric] for row in rows])
            for metric in metrics
        }
        summary["catastrophic_count"] = int(
            sum(row["methods"][method]["catastrophic"] for row in rows)
        )
        summary["catastrophic_rate"] = (
            float(summary["catastrophic_count"] / len(rows)) if rows else float("nan")
        )
        if method != "raw_reset":
            summary["paired_vs_raw"] = {
                metric: {
                    "mean_delta": float(
                        np.nanmean(
                            [
                                row["methods"][method][metric]
                                - row["methods"]["raw_reset"][metric]
                                for row in rows
                            ]
                        )
                    ),
                    "improvement_rate": float(
                        np.mean(
                            [
                                row["methods"][method][metric]
                                < row["methods"]["raw_reset"][metric]
                                for row in rows
                            ]
                        )
                    ),
                }
                for metric in metrics
            }
        result["methods"][method] = summary
    result["timing_seconds"] = stats([row["timing_seconds"] for row in rows])
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V14 Cut-Event Evaluation",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "`shadow_event` is a diagnostic corrected first frame. `b0_runtime` discards its human/scene/state and applies only its camera-derived B0 to the independent raw-reset output.",
        "",
        "| Split | N | Method | Camera T | Camera R | Composite | P90 comp. | P95 comp. | Human head | Catastrophic |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("overall", report["summary"])] + [
        (source, report["by_source"][source])
        for source in SOURCE_ORDER
        if source in report["by_source"]
    ]
    for split, summary in groups:
        for method in METHODS:
            row = summary["methods"][method]
            lines.append(
                f"| {split} | {summary['case_count']} | {method} | "
                f"{row['camera_translation_m']['mean']:.4f} | "
                f"{row['camera_rotation_deg']['mean']:.3f} | "
                f"{row['camera_composite']['mean']:.4f} | "
                f"{row['camera_composite']['p90']:.4f} | "
                f"{row['camera_composite']['p95']:.4f} | "
                f"{row['human_head_m']['mean']:.4f} | "
                f"{row['catastrophic_count']} |"
            )
    lines.extend(
        [
            "",
            f"Maximum B0-vs-shadow camera translation parity error: `{report['parity']['max_translation_m']:.3e} m`.",
            f"Maximum B0-vs-shadow 4x4 matrix element error: `{report['parity']['max_matrix_abs']:.3e}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    smpl_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()

    rows = []
    counts: defaultdict[str, int] = defaultdict(int)
    for source in args.sources:
        loader, records = make_loader(args, source)
        for local_index, (gt_views, record) in enumerate(zip(loader, records), start=1):
            row = evaluate_batch(model, smpl_layer, gt_views, device)
            row.update({"source": source, "record": record})
            rows.append(row)
            counts[source] += 1
            metrics = row["methods"]["b0_runtime"]
            print(
                f"[{len(rows):03d}] {source} {local_index}/{len(records)} "
                f"comp={metrics['camera_composite']:.4f} "
                f"cat={metrics['catastrophic']}",
                flush=True,
            )

    report = {
        "experiment": "v14_cut_first_cross_source_cut_event_eval",
        "checkpoint": str(args.model_path),
        "manifest_root": str(args.manifest_root),
        "model_flags": flags,
        "summary": summarize(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in args.sources
        },
        "parity": {
            "max_translation_m": max(
                row["b0_camera_parity"]["translation_m"] for row in rows
            ),
            "max_matrix_abs": max(
                row["b0_camera_parity"]["matrix_max_abs"] for row in rows
            ),
        },
        "cases": rows,
    }
    json_path = args.output_dir / "cut_event_evaluation.json"
    md_path = args.output_dir / "cut_event_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
