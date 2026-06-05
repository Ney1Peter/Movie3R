#!/usr/bin/env python3
"""Evaluate a V8.4 pose-correction checkpoint on a fixed benchmark.

This reuses the training criterion so benchmark numbers match the training
logs: corrected pose error, raw pose error, drift/gate loss, residual norm, and
improvement margin are computed with V82PoseRelationLoss.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import AvatarReX_AABB, AvatarReX_Video
from dust3r.inference import loss_of_one_batch
from dust3r.losses import V82PoseRelationLoss
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import todevice


DEFAULT_RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1",
    "zzr": "/data/wangzheng/iJCV-CODE/data/avatarrex_zzr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=Path("output/v8_4_pose_benchmark"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/eval"),
    )
    parser.add_argument(
        "--subsets",
        default="test_aabb,test_aaaa,train_sanity_aabb,train_sanity_aaaa",
        help="Comma-separated benchmark subset names.",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--test_split", default="Test/v8_4_mixed_aabb_aaaa")
    parser.add_argument("--raw_roots", default=json.dumps(DEFAULT_RAW_ROOTS, sort_keys=True))
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def parse_raw_roots(text: str):
    value = ast.literal_eval(text) if text.strip().startswith("{") else text
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return str(value)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_dataset(args: argparse.Namespace, subset: str, manifest_path: Path):
    split = "Training" if subset.startswith("train_sanity") else args.test_split
    common = dict(
        allow_repeat=True,
        split=split,
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=tuple(args.resolution),
        num_views=4,
        seed=int(args.seed),
        n_corres=0,
        manifest_path=str(manifest_path),
        load_da3_depth=False,
        raw_calibration_root=parse_raw_roots(args.raw_roots),
    )
    if subset.endswith("aabb"):
        return AvatarReX_AABB(**common)
    if subset.endswith("aaaa"):
        return AvatarReX_Video(**common)
    raise ValueError(f"Cannot infer dataset type from subset name: {subset}")


def make_criterion(device: torch.device):
    return V82PoseRelationLoss(
        translation_weight=1.0,
        rotation_weight=5.0,
        residual_weight=1.0e-4,
        drift_weight=0.2,
        improvement_weight=0.1,
        pose_key="raw_camera_pose",
        drift_trans_scale=0.5,
        drift_rot_scale_deg=45.0,
        improvement_margin=0.0,
    ).to(device)


def safe_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def compact_details(details: dict) -> dict:
    keys = [
        "v82_pose_relation_loss",
        "v82_pose_loss",
        "v82_trans_err",
        "v82_rot_err_deg",
        "v82_raw_trans_err",
        "v82_raw_rot_err_deg",
        "v82_gate_mean",
        "v82_drift_target_mean",
        "v82_drift_loss",
        "v82_delta_norm",
        "v82_norm_error_improvement",
        "v82_residual_small_loss",
        "v82_improvement_margin_loss",
    ]
    out = {}
    for key in keys:
        if key in details:
            value = safe_float(details[key])
            if value is not None:
                out[key] = value
    if "v82_raw_trans_err" in out and "v82_trans_err" in out:
        out["v82_trans_improvement"] = out["v82_raw_trans_err"] - out["v82_trans_err"]
    if "v82_raw_rot_err_deg" in out and "v82_rot_err_deg" in out:
        out["v82_rot_improvement_deg"] = out["v82_raw_rot_err_deg"] - out["v82_rot_err_deg"]
    return out


def summarize_rows(rows: list[dict]) -> dict:
    summary = {"count": len(rows)}
    numeric = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                numeric[key].append(float(value))
    for key, values in sorted(numeric.items()):
        if key in {"benchmark_index"}:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_median"] = float(np.median(arr))
    return summary


def row_identity(record: dict) -> dict:
    keys = [
        "benchmark_subset",
        "benchmark_index",
        "clip_type",
        "group",
        "angle_bucket",
        "view_angle_deg",
        "seq",
        "seqA",
        "seqB",
        "start_frame",
    ]
    return {key: record[key] for key in keys if key in record}


@torch.no_grad()
def eval_subset(args, model, criterion, smpl_model, device, subset: str) -> tuple[list[dict], dict]:
    manifest_path = args.benchmark_dir / f"{subset}.jsonl"
    records = load_jsonl(manifest_path)
    dataset = make_dataset(args, subset, manifest_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        drop_last=False,
    )

    rows = []
    cursor = 0
    for batch in loader:
        batch_records = records[cursor: cursor + len(batch[0]["img"])]
        cursor += len(batch_records)
        batch = todevice(batch, device)
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator=None,
            symmetrize_batch=False,
            inference=False,
            smpl_model=smpl_model,
        )
        loss, details = result["loss"]
        compact = compact_details(details)
        loss_value = safe_float(loss)
        if loss_value is not None:
            compact["loss"] = loss_value

        if int(args.batch_size) == 1 and batch_records:
            row = row_identity(batch_records[0])
            row.update(compact)
            rows.append(row)
        else:
            row = {
                "benchmark_subset": subset,
                "benchmark_index": len(rows),
                "batch_size": len(batch_records),
            }
            row.update(compact)
            rows.append(row)

    return rows, summarize_rows(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    run_name = args.name or args.model_path.stem
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).float().eval()
    criterion = make_criterion(device)
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )

    all_rows = []
    summary = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "subsets": {},
    }
    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        rows, subset_summary = eval_subset(args, model, criterion, smpl_model, device, subset)
        all_rows.extend(rows)
        summary["subsets"][subset] = subset_summary
        write_csv(output_dir / f"{subset}.csv", rows)
        (output_dir / f"{subset}.json").write_text(
            json.dumps({"summary": subset_summary, "rows": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary["overall"] = summarize_rows(all_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "all_rows.csv", all_rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote evaluation to {output_dir}")


if __name__ == "__main__":
    main()
