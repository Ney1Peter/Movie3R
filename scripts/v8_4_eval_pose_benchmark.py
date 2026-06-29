#!/usr/bin/env python3
"""Evaluate a V8.4 pose-correction checkpoint on a fixed benchmark.

This reuses the training criterion so benchmark numbers match the training
logs: corrected pose error, raw pose error, drift/gate loss, residual norm, and
improvement margin are computed with V82PoseRelationLoss. It also reports
Human3R-style SMPL metrics such as Metric-MPJPE, MPJPE, PA-MPJPE, RootError,
and PVE from the predicted SMPL parameters.
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
import roma

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
from dust3r.utils import SMPL_Layer
from dust3r.utils.camera import pose_encoding_to_camera
from dust3r.utils.device import todevice
from dust3r.utils.geometry import inv


DEFAULT_RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
    "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
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
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512), metavar=("W", "H"))
    parser.add_argument(
        "--resize_mode",
        default="human3r_demo",
        help="AvatarReX image preprocessing mode; human3r_demo matches demo.py load_images(size=512).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default=None)
    parser.add_argument("--translation_weight", type=float, default=1.0)
    parser.add_argument("--rotation_weight", type=float, default=5.0)
    parser.add_argument("--residual_weight", type=float, default=1.0e-4)
    parser.add_argument("--drift_weight", type=float, default=0.2)
    parser.add_argument("--improvement_weight", type=float, default=0.1)
    parser.add_argument("--improvement_margin", type=float, default=0.0)
    parser.add_argument("--drift_trans_scale", type=float, default=0.5)
    parser.add_argument("--drift_rot_scale_deg", type=float, default=45.0)
    parser.add_argument("--drift_target_deadzone", type=float, default=0.0)
    parser.add_argument("--drift_target_scale", type=float, default=1.0)
    parser.add_argument("--human_trans_weight", type=float, default=0.0)
    parser.add_argument("--human_trans_delta_weight", type=float, default=0.0)
    parser.add_argument("--human_cam_ref_weight", type=float, default=0.0)
    parser.add_argument("--human_pairwise_ref_weight", type=float, default=0.0)
    parser.add_argument(
        "--dump_poses",
        action="store_true",
        help="Also save GT/raw/corrected 4x4 camera matrices for viewer comparisons.",
    )
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
        resize_mode=str(args.resize_mode),
    )
    if subset.endswith("aabb"):
        return AvatarReX_AABB(**common)
    if subset.endswith("aaaa"):
        return AvatarReX_Video(**common)
    raise ValueError(f"Cannot infer dataset type from subset name: {subset}")


def make_criterion(args: argparse.Namespace, device: torch.device):
    return V82PoseRelationLoss(
        translation_weight=float(args.translation_weight),
        rotation_weight=float(args.rotation_weight),
        residual_weight=float(args.residual_weight),
        drift_weight=float(args.drift_weight),
        improvement_weight=float(args.improvement_weight),
        pose_key="raw_camera_pose",
        drift_trans_scale=float(args.drift_trans_scale),
        drift_rot_scale_deg=float(args.drift_rot_scale_deg),
        drift_target_deadzone=float(args.drift_target_deadzone),
        drift_target_scale=float(args.drift_target_scale),
        improvement_margin=float(args.improvement_margin),
        human_trans_weight=float(args.human_trans_weight),
        human_trans_delta_weight=float(args.human_trans_delta_weight),
        human_cam_ref_weight=float(args.human_cam_ref_weight),
        human_pairwise_ref_weight=float(args.human_pairwise_ref_weight),
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
        "v82_human_trans_err",
        "v82_raw_human_trans_err",
        "v82_human_trans_loss",
        "v82_human_trans_delta_small_loss",
    ]
    out = {}
    for key in keys:
        if key in details:
            value = safe_float(details[key])
            if value is not None:
                out[key] = value
    for key, value in details.items():
        if key.startswith(("v82_human_trans_err/", "v82_raw_human_trans_err/", "v82_human_trans_delta_norm/")):
            value = safe_float(value)
            if value is not None:
                out[key.replace("/", "_view")] = value
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


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def stack_pose_field(preds: list[dict], key: str, batch_size: int, device: torch.device) -> torch.Tensor:
    fields = []
    for pred in preds:
        value = pred.get(key, None)
        if value is None:
            fields.append(torch.full((batch_size, 7), np.nan, device=device))
        else:
            fields.append(value.detach().float())
    return torch.stack(fields, dim=1)


def stack_optional_field(preds: list[dict], key: str, batch_size: int, device: torch.device) -> torch.Tensor:
    fields = []
    for pred in preds:
        value = pred.get(key, None)
        if value is None:
            fields.append(torch.full((batch_size, 1), np.nan, device=device))
            continue
        value = value.detach().float().reshape(batch_size, -1)
        fields.append(value)
    return torch.stack(fields, dim=1)


def stack_view(items: list[dict], key: str) -> torch.Tensor:
    value = torch.stack([item[key] for item in items], dim=0)
    return value.view(-1, *value.shape[2:])


def make_eval_smpl_layers(device: torch.device) -> torch.nn.ModuleDict:
    layers = torch.nn.ModuleDict(
        {
            "neutral_10": SMPL_Layer(
                type="smplx",
                gender="neutral",
                num_betas=10,
                kid=False,
                person_center="head",
            ),
            "neutral_11": SMPL_Layer(
                type="smplx",
                gender="neutral",
                num_betas=11,
                kid=False,
                person_center="head",
            ),
        }
    )
    return layers.to(device).eval()


def batch_compute_similarity_transform_torch(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Human3R-style batched Procrustes alignment from source points to target."""
    source_t = source.permute(0, 2, 1)
    target_t = target.permute(0, 2, 1)

    mu_source = source_t.mean(dim=-1, keepdim=True)
    mu_target = target_t.mean(dim=-1, keepdim=True)
    centered_source = source_t - mu_source
    centered_target = target_t - mu_target

    var_source = torch.sum(centered_source**2, dim=(1, 2)).clamp_min(1e-8)
    covariance = centered_source.bmm(centered_target.permute(0, 2, 1))

    u, singular_values, v = torch.svd(covariance)
    z = torch.eye(u.shape[1], device=source.device, dtype=source.dtype).unsqueeze(0).repeat(u.shape[0], 1, 1)
    z[:, -1, -1] *= torch.sign(torch.det(u.bmm(v.permute(0, 2, 1)))).clamp(min=-1.0, max=1.0)

    rotation = v.bmm(z.bmm(u.permute(0, 2, 1)))
    scale = torch.stack([torch.trace(mat) for mat in rotation.bmm(covariance)]) / var_source
    translation = mu_target - scale[:, None, None] * rotation.bmm(mu_source)

    aligned = scale[:, None, None] * rotation.bmm(source_t) + translation
    return aligned.permute(0, 2, 1)


def mean_point_error_mm(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - gt, dim=-1).mean() * 1000.0


def compute_smpl_benchmark_metrics(
    batch: list[dict],
    preds: list[dict],
    smpl_layers: torch.nn.ModuleDict,
) -> dict:
    required_gt = {"smpl_mask", "smpl_j3d", "smpl_v3d", "camera_intrinsics"}
    required_pred = {"smpl_rotmat", "smpl_shape", "smpl_transl", "smpl_expression"}
    if not all(required_gt.issubset(view.keys()) for view in batch):
        return {}
    if not all(required_pred.issubset(pred.keys()) for pred in preds):
        return {}

    pred_rotmat = stack_view(preds, "smpl_rotmat")
    pred_shape = stack_view(preds, "smpl_shape")
    pred_transl = stack_view(preds, "smpl_transl")
    pred_expression = stack_view(preds, "smpl_expression")
    intrinsics = stack_view(batch, "camera_intrinsics")
    gt_j3d_all = stack_view(batch, "smpl_j3d")
    gt_v3d_all = stack_view(batch, "smpl_v3d")

    num_humans = min(
        stack_view(batch, "smpl_mask").shape[1],
        pred_rotmat.shape[1],
        pred_shape.shape[1],
        pred_transl.shape[1],
        pred_expression.shape[1],
        gt_j3d_all.shape[1],
        gt_v3d_all.shape[1],
    )
    smpl_mask = stack_view(batch, "smpl_mask").bool()[:, :num_humans]
    pred_rotmat = pred_rotmat[:, :num_humans]
    pred_shape = pred_shape[:, :num_humans]
    pred_transl = pred_transl[:, :num_humans]
    pred_expression = pred_expression[:, :num_humans]
    gt_j3d_all = gt_j3d_all[:, :num_humans]
    gt_v3d_all = gt_v3d_all[:, :num_humans]

    img_mask = stack_view(batch, "img_mask").bool() if "img_mask" in batch[0] else None
    if img_mask is not None:
        smpl_mask = smpl_mask & img_mask[:, None]
    if int(smpl_mask.sum()) == 0:
        return {}
    view_indices = torch.where(smpl_mask)[0]

    shape_dim = int(pred_shape.shape[-1])
    layer_key = f"neutral_{shape_dim}"
    if layer_key not in smpl_layers:
        return {}
    layer = smpl_layers[layer_key]

    pred_rotvec = roma.rotmat_to_rotvec(pred_rotmat[smpl_mask])
    smpl_out = layer(
        pred_rotvec,
        pred_shape[smpl_mask],
        pred_transl[smpl_mask],
        None,
        None,
        K=intrinsics[view_indices],
        expression=pred_expression[smpl_mask],
    )
    if "smpl_j3d" not in smpl_out or "smpl_v3d" not in smpl_out:
        return {}

    pred_j3d = smpl_out["smpl_j3d"].float()
    pred_v3d = smpl_out["smpl_v3d"].float()
    gt_j3d = gt_j3d_all[smpl_mask].to(device=pred_j3d.device, dtype=pred_j3d.dtype)
    gt_v3d = gt_v3d_all[smpl_mask].to(device=pred_v3d.device, dtype=pred_v3d.dtype)

    joint_count = min(pred_j3d.shape[1], gt_j3d.shape[1])
    vert_count = min(pred_v3d.shape[1], gt_v3d.shape[1])
    pred_j3d = pred_j3d[:, :joint_count]
    gt_j3d = gt_j3d[:, :joint_count]
    pred_v3d = pred_v3d[:, :vert_count]
    gt_v3d = gt_v3d[:, :vert_count]

    pelvis_idxs = [1, 2] if joint_count > 2 else [0]
    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdim=True)
    gt_pelvis = gt_j3d[:, pelvis_idxs].mean(dim=1, keepdim=True)

    pred_j3d_centered = pred_j3d - pred_pelvis
    gt_j3d_centered = gt_j3d - gt_pelvis
    pred_v3d_centered = pred_v3d - pred_pelvis
    gt_v3d_centered = gt_v3d - gt_pelvis

    pa_pred_j3d = batch_compute_similarity_transform_torch(pred_j3d_centered, gt_j3d_centered)
    pa_pred_v3d = batch_compute_similarity_transform_torch(pred_v3d_centered, gt_v3d_centered)

    metrics = {
        "v82_metric_mpjpe_mm": mean_point_error_mm(pred_j3d, gt_j3d),
        "v82_mpjpe_mm": mean_point_error_mm(pred_j3d_centered, gt_j3d_centered),
        "v82_pa_mpjpe_mm": mean_point_error_mm(pa_pred_j3d, gt_j3d_centered),
        "v82_root_error_mm": mean_point_error_mm(pred_pelvis, gt_pelvis),
        "v82_metric_pve_mm": mean_point_error_mm(pred_v3d, gt_v3d),
        "v82_pve_mm": mean_point_error_mm(pred_v3d_centered, gt_v3d_centered),
        "v82_pa_pve_mm": mean_point_error_mm(pa_pred_v3d, gt_v3d_centered),
        "v82_smpl_metric_count": torch.as_tensor(float(pred_j3d.shape[0]), device=pred_j3d.device),
    }
    out = {}
    for key, value in metrics.items():
        value = safe_float(value)
        if value is not None:
            out[key] = value
    return out


def dump_pose_matrices(
    output_dir: Path,
    subset: str,
    records: list[dict],
    batch: list[dict],
    preds: list[dict],
) -> list[str]:
    """Save one compressed pose bundle per clip.

    The model predicts relative camera pose encodings in the same coordinate
    system used by V82PoseRelationLoss. For visualization we save both relative
    matrices and GT-anchored absolute matrices. The absolute raw/corrected
    matrices use GT view-0 only as an evaluation anchor; GT is not model input.
    """
    pose_dir = output_dir / "poses" / subset
    pose_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(batch[0]["raw_camera_pose"].shape[0])
    device = batch[0]["raw_camera_pose"].device
    gt_abs = torch.stack([view["raw_camera_pose"].detach().float() for view in batch], dim=1)
    gt0_inv = inv(gt_abs[:, 0])
    gt_rel = torch.matmul(gt0_inv[:, None], gt_abs)

    corrected_enc = stack_pose_field(preds, "camera_pose", batch_size, device)
    raw_enc = stack_pose_field(preds, "v8_raw_camera_pose", batch_size, device)
    gate = stack_optional_field(preds, "v8_pose_prompt_gate", batch_size, device)
    drift_logit = stack_optional_field(preds, "v8_pose_prompt_drift_logit", batch_size, device)
    delta_norm = stack_optional_field(preds, "v8_pose_prompt_delta_norm", batch_size, device)

    saved_paths = []
    for batch_idx, record in enumerate(records):
        stem = f"{int(record.get('benchmark_index', batch_idx)):04d}_{record.get('clip_type', 'clip')}_{record.get('group', 'group')}"
        if record.get("angle_bucket"):
            stem += f"_{record['angle_bucket']}"
        path = pose_dir / f"{stem}.npz"

        corrected_rel = pose_encoding_to_camera(corrected_enc[batch_idx]).detach().float()
        raw_rel = pose_encoding_to_camera(raw_enc[batch_idx]).detach().float()
        gt0_abs = gt_abs[batch_idx, 0]
        corrected_abs = torch.matmul(gt0_abs[None], corrected_rel)
        raw_abs = torch.matmul(gt0_abs[None], raw_rel)

        metadata = json.dumps(row_identity(record), sort_keys=True)
        np.savez_compressed(
            path,
            metadata=np.asarray(metadata),
            gt_c2w_abs=tensor_to_numpy(gt_abs[batch_idx]),
            gt_c2w_rel=tensor_to_numpy(gt_rel[batch_idx]),
            raw_pose_encoding=tensor_to_numpy(raw_enc[batch_idx]),
            corrected_pose_encoding=tensor_to_numpy(corrected_enc[batch_idx]),
            raw_c2w_rel=tensor_to_numpy(raw_rel),
            corrected_c2w_rel=tensor_to_numpy(corrected_rel),
            raw_c2w_abs_gt0=tensor_to_numpy(raw_abs),
            corrected_c2w_abs_gt0=tensor_to_numpy(corrected_abs),
            gate=tensor_to_numpy(gate[batch_idx]),
            drift_logit=tensor_to_numpy(drift_logit[batch_idx]),
            delta_norm=tensor_to_numpy(delta_norm[batch_idx]),
        )
        saved_paths.append(str(path.relative_to(output_dir)))
    return saved_paths


@torch.no_grad()
def eval_subset(args, model, criterion, smpl_model, device, subset: str, output_dir: Path) -> tuple[list[dict], dict]:
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
        pose_paths = []
        if args.dump_poses:
            pose_paths = dump_pose_matrices(output_dir, subset, batch_records, batch, result["pred"])
        loss, details = result["loss"]
        compact = compact_details(details)
        compact.update(compute_smpl_benchmark_metrics(batch, result["pred"], smpl_model.eval_smpl_layers))
        loss_value = safe_float(loss)
        if loss_value is not None:
            compact["loss"] = loss_value

        if int(args.batch_size) == 1 and batch_records:
            row = row_identity(batch_records[0])
            row.update(compact)
            if pose_paths:
                row["pose_npz"] = pose_paths[0]
            rows.append(row)
        else:
            for batch_idx, record in enumerate(batch_records):
                row = row_identity(record)
                row["batch_size"] = len(batch_records)
                row.update(compact)
                if batch_idx < len(pose_paths):
                    row["pose_npz"] = pose_paths[batch_idx]
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
    criterion = make_criterion(args, device)
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    smpl_model.eval_smpl_layers = make_eval_smpl_layers(device)

    all_rows = []
    summary = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "subsets": {},
    }
    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        rows, subset_summary = eval_subset(args, model, criterion, smpl_model, device, subset, output_dir)
        all_rows.extend(rows)
        summary["subsets"][subset] = subset_summary
        write_csv(output_dir / f"{subset}.csv", rows)
        (output_dir / f"{subset}.json").write_text(
            json.dumps({"summary": subset_summary, "rows": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary["overall"] = summarize_rows(all_rows)
    if args.dump_poses:
        pose_index = [row for row in all_rows if "pose_npz" in row]
        with (output_dir / "poses_index.jsonl").open("w", encoding="utf-8") as f:
            for row in pose_index:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        summary["pose_dump"] = {
            "count": len(pose_index),
            "index": "poses_index.jsonl",
            "coordinate_note": "raw/corrected absolute matrices are anchored by GT view-0 for evaluation visualization only.",
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "all_rows.csv", all_rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote evaluation to {output_dir}")


if __name__ == "__main__":
    main()
