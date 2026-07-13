#!/usr/bin/env python3
"""Probe Human3R raw-output signals for V10 shot-boundary detection.

This complements ``v10_detector_feature_probe.py``.  The first detector round
uses only cheap image features.  This script runs frozen original Human3R on the
same short/long pattern manifests and evaluates whether adjacent-frame raw
camera and SMPL-output changes are useful boundary cues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from dust3r.inference import loss_of_one_batch
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.camera import pose_encoding_to_camera
from dust3r.utils.device import to_cpu, todevice
from scripts.v10_detector_feature_probe import (
    ORACLE_FEATURES,
    SOURCE_ORDER,
    group_metrics,
    leave_source_model,
    leave_source_threshold,
    metrics,
    predict_leave_source_model,
    predict_leave_source_threshold,
    read_jsonl,
    summarize_counts,
    write_csv,
    write_markdown,
)
from scripts.v8_4_view_pose_benchmark_scene import make_single_record_dataset, write_one_record_manifest
from scripts.v9_learned_stream_alignment_overfit import DEFAULT_RAW_ROOTS
from scripts.v9_online_stream_human3r_segment_align import strict_original_model


POSE_FEATURES = [
    "raw_cam_rot_step_deg",
    "raw_cam_trans_step_m",
    "corr_cam_rot_step_deg",
    "corr_cam_trans_step_m",
]

HUMAN_FEATURES = [
    "smpl_transl_step_m",
    "smpl_root_rot_step_deg",
    "smpl_detect_prev",
    "smpl_detect_cur",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_pattern_probe",
    )
    parser.add_argument(
        "--long12_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_long12_pattern_probe",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_detector_probe" / "human3r_pose_round1",
    )
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--limit_per_source_set", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "case"


def load_records(args: argparse.Namespace) -> list[dict]:
    records = []
    for manifest_set, root in (("short4", args.pattern_root), ("long12", args.long12_root)):
        for source in SOURCE_ORDER:
            path = root / source / "train_all_patterns.jsonl"
            source_records = read_jsonl(path)
            if int(args.limit_per_source_set) > 0:
                source_records = source_records[: int(args.limit_per_source_set)]
            for local_idx, record in enumerate(source_records):
                row = dict(record)
                row["source"] = source
                row["manifest_set"] = manifest_set
                row["source_local_index"] = local_idx
                records.append(row)
    return records


def dataset_args_for_record(args: argparse.Namespace, record: dict, num_views: int) -> argparse.Namespace:
    group = str(record.get("group", ""))
    first_seq = str(record.get("seqs", [""])[0])
    is_mvhuman = str(record.get("source", "")).startswith("mvhuman") or group.isdigit() or first_seq.split("/", 1)[0].isdigit()
    raw_roots = "null" if is_mvhuman else json.dumps({k: str(v) for k, v in DEFAULT_RAW_ROOTS.items()}, sort_keys=True)
    return argparse.Namespace(
        data_root=args.data_root,
        test_split="Test/v8_4_mixed_aabb_aaaa",
        resolution=tuple(args.resolution),
        resize_mode=str(args.resize_mode),
        raw_roots=raw_roots,
        num_views=int(num_views),
    )


def prepare_views(args: argparse.Namespace, record: dict, manifest_path: Path, device: torch.device) -> list[dict]:
    num_views = len(record["seqs"])
    dataset_args = dataset_args_for_record(args, record, num_views)
    write_one_record_manifest(manifest_path, record)
    dataset = make_single_record_dataset(dataset_args, record, manifest_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    views = next(iter(loader))
    for view in views:
        view["img_mask"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["ray_mask"] = torch.zeros_like(view["ray_mask"], dtype=torch.bool)
        view["update"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_state"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_mem"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_v8_history"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
    return todevice(views, device)


def pose_to_matrix(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    pose = pose_encoding_to_camera(value.detach().float()).detach().cpu().numpy().astype(np.float32)
    return pose


def rotation_angle_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    rel = R0.T @ R1
    cos = float((np.trace(rel) - 1.0) * 0.5)
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def pose_step_features(poses: np.ndarray | None, idx: int, prefix: str) -> dict:
    if poses is None or idx <= 0 or idx >= poses.shape[0]:
        return {
            f"{prefix}_cam_rot_step_deg": 0.0,
            f"{prefix}_cam_trans_step_m": 0.0,
        }
    p0 = poses[idx - 1]
    p1 = poses[idx]
    return {
        f"{prefix}_cam_rot_step_deg": rotation_angle_deg(p0[:3, :3], p1[:3, :3]),
        f"{prefix}_cam_trans_step_m": float(np.linalg.norm(p1[:3, 3] - p0[:3, 3])),
    }


def first_human_transl(pred: dict) -> tuple[np.ndarray, float]:
    value = pred.get("smpl_transl", None)
    if value is None:
        return np.zeros(3, dtype=np.float32), 0.0
    arr = value.detach().float().cpu().numpy()
    if arr.ndim < 3 or arr.shape[1] < 1:
        return np.zeros(3, dtype=np.float32), 0.0
    return arr[0, 0].astype(np.float32), 1.0


def first_human_root_rot(pred: dict) -> tuple[np.ndarray, float]:
    value = pred.get("smpl_rotmat", None)
    if value is None:
        return np.eye(3, dtype=np.float32), 0.0
    arr = value.detach().float().cpu().numpy()
    if arr.ndim < 5 or arr.shape[1] < 1:
        return np.eye(3, dtype=np.float32), 0.0
    return arr[0, 0, 0].astype(np.float32), 1.0


def human_step_features(preds: list[dict], idx: int) -> dict:
    if idx <= 0 or idx >= len(preds):
        return {
            "smpl_transl_step_m": 0.0,
            "smpl_root_rot_step_deg": 0.0,
            "smpl_detect_prev": 0.0,
            "smpl_detect_cur": 0.0,
        }
    t0, d0 = first_human_transl(preds[idx - 1])
    t1, d1 = first_human_transl(preds[idx])
    R0, r0 = first_human_root_rot(preds[idx - 1])
    R1, r1 = first_human_root_rot(preds[idx])
    if d0 <= 0.0 or d1 <= 0.0:
        transl_step = 0.0
    else:
        transl_step = float(np.linalg.norm(t1 - t0))
    if r0 <= 0.0 or r1 <= 0.0:
        root_step = 0.0
    else:
        root_step = rotation_angle_deg(R0, R1)
    return {
        "smpl_transl_step_m": transl_step,
        "smpl_root_rot_step_deg": root_step,
        "smpl_detect_prev": float(d0),
        "smpl_detect_cur": float(d1),
    }


def run_record(
    model: ARCroco3DStereo,
    smpl_model: SMPLModel,
    args: argparse.Namespace,
    record: dict,
    device: torch.device,
) -> list[dict]:
    pattern_id = str(record.get("pattern_id", "unknown"))
    manifest_path = args.output_dir / "one_record_manifests" / f"{record['manifest_set']}_{record['source']}_{safe_name(pattern_id)}.jsonl"
    views = prepare_views(args, record, manifest_path, device)
    with torch.no_grad():
        result = loss_of_one_batch(
            views,
            model,
            criterion=None,
            accelerator=None,
            symmetrize_batch=False,
            inference=False,
            smpl_model=smpl_model,
        )
    outputs = to_cpu({"views": result["views"], "pred": result["pred"]})
    preds = list(outputs["pred"])
    raw_poses = []
    corr_poses = []
    for pred in preds:
        raw_pose = pose_to_matrix(pred.get("v8_raw_camera_pose", pred.get("camera_pose")))
        corr_pose = pose_to_matrix(pred.get("camera_pose"))
        raw_poses.append(raw_pose[0] if raw_pose is not None else np.eye(4, dtype=np.float32))
        corr_poses.append(corr_pose[0] if corr_pose is not None else np.eye(4, dtype=np.float32))
    raw_poses_np = np.stack(raw_poses).astype(np.float32)
    corr_poses_np = np.stack(corr_poses).astype(np.float32)

    rows = []
    seqs = list(record["seqs"])
    frames = list(record["frames"])
    labels = list(record["shot_labels"])
    angles = list(record.get("transition_angles_deg", [0.0] * len(seqs)))
    pattern = str(record.get("clip_type", "unknown"))
    for idx in range(1, len(seqs)):
        row = {
            "manifest_set": record["manifest_set"],
            "source": record["source"],
            "pattern": pattern,
            "pattern_id": pattern_id,
            "pair_idx": idx,
            "seq_prev": seqs[idx - 1],
            "seq_cur": seqs[idx],
            "frame_prev": int(frames[idx - 1]),
            "frame_cur": int(frames[idx]),
            "label": int(labels[idx]),
            "transition_angle_deg": float(angles[idx]),
        }
        row.update(pose_step_features(raw_poses_np, idx, "raw"))
        row.update(pose_step_features(corr_poses_np, idx, "corr"))
        row.update(human_step_features(preds, idx))
        rows.append(row)
    return rows


def selected_group_metrics(rows: list[dict], method: str, pred: np.ndarray) -> list[dict]:
    out = []
    base_rows = [{**row, "pred": int(p)} for row, p in zip(rows, pred)]
    for group_key in ("source", "pattern", "manifest_set"):
        for group in sorted({str(row[group_key]) for row in base_rows}):
            subset = [row for row in base_rows if str(row[group_key]) == group]
            y = np.asarray([int(row["label"]) for row in subset], dtype=np.int64)
            p = np.asarray([int(row["pred"]) for row in subset], dtype=np.int64)
            row = metrics(y, p)
            row.update(
                {
                    "method": method,
                    "group_type": group_key,
                    "group": group,
                    "pairs": len(subset),
                    "positives": int(y.sum()),
                    "negatives": int(len(y) - y.sum()),
                }
            )
            out.append(row)
    return out


def selected_predictions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    specs = [
        ("threshold:smpl_root_rot_step_deg", "threshold", ["smpl_root_rot_step_deg"]),
        ("threshold:raw_cam_rot_step_deg", "threshold", ["raw_cam_rot_step_deg"]),
        ("threshold:transition_angle_deg", "threshold", ["transition_angle_deg"]),
        ("logreg:pose_camera", "model", POSE_FEATURES),
        ("logreg:human_smpl", "model", HUMAN_FEATURES),
        ("logreg:pose_plus_human", "model", POSE_FEATURES + HUMAN_FEATURES),
    ]
    pred_rows = []
    metric_rows = []
    for method, kind, features in specs:
        if kind == "threshold":
            _, _, _, method_rows = predict_leave_source_threshold(rows, features[0])
        else:
            _, _, _, method_rows = predict_leave_source_model(rows, features, method)
        compact_rows = []
        for row in method_rows:
            compact = {
                "method": method,
                "manifest_set": row["manifest_set"],
                "source": row["source"],
                "pattern": row["pattern"],
                "pattern_id": row["pattern_id"],
                "pair_idx": row["pair_idx"],
                "label": row["label"],
                "pred": row["pred"],
                "is_error": int(row["label"] != row["pred"]),
                "seq_prev": row["seq_prev"],
                "seq_cur": row["seq_cur"],
                "frame_prev": row["frame_prev"],
                "frame_cur": row["frame_cur"],
            }
            if "prob" in row:
                compact["prob"] = row["prob"]
            compact_rows.append(compact)
        pred_rows.extend(compact_rows)
        metric_rows.extend(group_metrics(compact_rows, method, ["source", "pattern", "manifest_set"]))
    return pred_rows, metric_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = args.output_dir / "detector_human3r_pose_pair_features.csv"
    if pair_csv.is_file() and not args.overwrite:
        rows = list(csv.DictReader(pair_csv.open("r", encoding="utf-8")))
        for row in rows:
            for key in [
                "pair_idx",
                "frame_prev",
                "frame_cur",
                "label",
                *POSE_FEATURES,
                *HUMAN_FEATURES,
                *ORACLE_FEATURES,
            ]:
                if key in row:
                    row[key] = float(row[key]) if "." in str(row[key]) else int(row[key])
    else:
        device = torch.device(args.device)
        add_path_to_dust3r(str(args.model_path))
        model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
        strict_original_model(model)
        smpl_model = SMPLModel(
            device,
            model_args={
                "patch_size": model.croco_args["patch_size"],
                "mhmr_img_res": model.mhmr_img_res,
                "bb_patch_size": model.bb_patch_size,
            },
        )
        records = load_records(args)
        rows = []
        for index, record in enumerate(records, start=1):
            print(
                f"pose probe {index}/{len(records)} {record['manifest_set']} {record['source']} {record.get('pattern_id')}",
                flush=True,
            )
            rows.extend(run_record(model, smpl_model, args, record, device))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        write_csv(pair_csv, rows, fieldnames)

    results = []
    for feature in POSE_FEATURES + HUMAN_FEATURES + ORACLE_FEATURES:
        results.append(leave_source_threshold(rows, feature))
    results.append(leave_source_model(rows, POSE_FEATURES, "logreg:pose_camera"))
    results.append(leave_source_model(rows, HUMAN_FEATURES, "logreg:human_smpl"))
    results.append(leave_source_model(rows, POSE_FEATURES + HUMAN_FEATURES, "logreg:pose_plus_human"))
    results.append(leave_source_model(rows, ORACLE_FEATURES, "logreg:oracle_gt_angle"))
    results.append(
        leave_source_model(
            rows,
            POSE_FEATURES + HUMAN_FEATURES + ORACLE_FEATURES,
            "logreg:pose_human_plus_oracle",
        )
    )

    serializable = []
    for row in results:
        clean = {k: v for k, v in row.items() if k != "thresholds"}
        clean["thresholds_json"] = json.dumps(row.get("thresholds", {}), sort_keys=True)
        serializable.append(clean)
    serializable = sorted(serializable, key=lambda x: x["f1"], reverse=True)
    write_csv(args.output_dir / "detector_human3r_pose_method_results.csv", serializable)
    (args.output_dir / "detector_human3r_pose_method_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    counts = summarize_counts(rows)
    (args.output_dir / "dataset_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.output_dir / "detector_human3r_pose_summary.md", counts, results)
    pred_rows, group_rows = selected_predictions(rows)
    write_csv(args.output_dir / "detector_human3r_pose_selected_predictions.csv", pred_rows)
    write_csv(
        args.output_dir / "detector_human3r_pose_selected_group_metrics.csv",
        sorted(group_rows, key=lambda x: (x["method"], x["group_type"], x["group"])),
    )
    print(json.dumps({"counts": counts, "top": serializable[:8]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
