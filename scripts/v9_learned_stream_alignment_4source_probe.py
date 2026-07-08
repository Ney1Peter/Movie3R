#!/usr/bin/env python3
"""Train the learned streaming segment-alignment probe on a tiny 4-source set.

This extends ``v9_learned_stream_alignment_overfit.py`` from one AABB clip to a
shared small probe: by default two AABB clips from each source
{AvatarReX, THUman, MVHuman100, MVHuman200}.  Human3R itself stays frozen and
strictly original.  The only learnable part is a small streaming alignment MLP
that predicts one transform for the post-boundary segment from A-history human
anchors and the current B1 human anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
ARCHIVE_V7 = SCRIPTS_ROOT / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from demo import prepare_output
from dust3r.datasets.avatarrex import AvatarReX_AABB
from dust3r.inference import inference_recurrent_lighter
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import to_cpu, todevice
from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v9_learned_stream_alignment_overfit import (
    DEFAULT_RAW_ROOTS,
    StreamingAlignmentMLP,
    build_feature,
    camera_metrics,
    extract_gt_world,
    human_metrics,
    output_complete,
    rotation_geodesic,
    so3_exp,
    write_aligned_output,
)
from v9_online_stream_human3r_segment_align import strict_original_model


SOURCE_ORDER = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


@dataclass
class CachedSample:
    index: int
    source: str
    pattern_id: str
    record: dict
    local_dir: Path
    aligned_dir: Path
    pred_poses: np.ndarray
    pred_joints: np.ndarray
    target_poses: np.ndarray
    target_joints: np.ndarray
    feature: torch.Tensor
    bridge_debug: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v9_learned_stream_alignment_4source_probe" / "aabb_2per_source",
    )
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_pattern_probe",
    )
    parser.add_argument("--manifest_name", default="train_aabb.jsonl")
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_ORDER), choices=list(SOURCE_ORDER))
    parser.add_argument("--samples_per_source", type=int, default=2)
    parser.add_argument("--boundary", type=int, default=2, help="First frame index of the post-boundary segment.")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_rot_deg", type=float, default=180.0)
    parser.add_argument("--max_trans", type=float, default=12.0)
    parser.add_argument("--human_weight", type=float, default=5.0)
    parser.add_argument("--camera_t_weight", type=float, default=2.0)
    parser.add_argument("--camera_r_weight", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--eval_checkpoint", type=Path, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def select_records(args: argparse.Namespace) -> list[dict]:
    selected = []
    for source in args.sources:
        path = args.manifest_root / source / str(args.manifest_name)
        records = read_jsonl(path)
        if len(records) < int(args.samples_per_source):
            raise RuntimeError(f"{source} only has {len(records)} AABB records in {path}")
        for local_idx, record in enumerate(records[: int(args.samples_per_source)]):
            record = dict(record)
            record["source"] = source
            record["source_local_index"] = local_idx
            record.setdefault(
                "pattern_id",
                f"{source}_{record.get('group', 'group')}_{record.get('start_frame', 'start')}_{local_idx}",
            )
            selected.append(record)
    return selected


def source_split_and_scope(record: dict) -> tuple[str, str]:
    source = str(record.get("source", ""))
    group = str(record.get("group", ""))
    first_seq = str(record.get("seqs", [record.get("seqA", "")])[0])
    is_mvhuman = source.startswith("mvhuman") or group.isdigit() or first_seq.split("/", 1)[0].isdigit()
    if is_mvhuman:
        return "Training/mvhuman", "same_parent"
    return "Training", "all"


def raw_roots_for_record(record: dict):
    split, _ = source_split_and_scope(record)
    if split == "Training/mvhuman":
        return None
    return {k: str(v) for k, v in DEFAULT_RAW_ROOTS.items()}


def aabb_tuple_from_record(record: dict) -> tuple[str, str, int]:
    if "seqs" in record and "frames" in record:
        seqs = record["seqs"]
        frames = record["frames"]
        return str(seqs[0]), str(seqs[2]), int(frames[0])
    return str(record["seqA"]), str(record["seqB"]), int(record["start_frame"])


def load_aabb_views_for_record(record: dict, args: argparse.Namespace, device: torch.device) -> list[dict]:
    split, pair_scope = source_split_and_scope(record)
    seq_a, seq_b, start_frame = aabb_tuple_from_record(record)
    dataset = AvatarReX_AABB(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=[(seq_a, seq_b, int(start_frame))],
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(record),
        resize_mode=str(args.resize_mode),
        max_humans=1,
        pair_scope=pair_scope,
    )
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
    views[int(args.boundary) - 1]["reset"] = torch.ones_like(views[int(args.boundary) - 1]["img_mask"], dtype=torch.bool)
    return todevice(views, device)


def run_local_reset_human3r(
    model: ARCroco3DStereo,
    views: list[dict],
    local_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if output_complete(local_dir) and not args.overwrite:
        return
    if local_dir.exists() and args.overwrite:
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, str(device), use_ttt3r=False)
    outputs_cpu = to_cpu(outputs)
    outputs_to_save = {"pred": outputs_cpu["pred"], "views": [dict(v) for v in outputs_cpu["views"]]}
    for view in outputs_to_save["views"]:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    prepare_output(
        outputs_to_save,
        str(local_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=getattr(model, "mhmr_img_res", None),
        subsample=1,
    )


def cache_samples(records: list[dict], args: argparse.Namespace, device: torch.device) -> list[CachedSample]:
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
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    cached = []

    for index, record in enumerate(records):
        pattern_id = str(record.get("pattern_id") or f"{record['source']}_{index:02d}")
        sample_dir = args.output_dir / "samples" / f"{index:02d}_{safe_name(pattern_id)}"
        local_dir = sample_dir / "original_human3r_local_reset"
        aligned_dir = sample_dir / "learned_stream_aligned"
        print(f">> [{index + 1}/{len(records)}] cache {record['source']} {pattern_id}", flush=True)
        views = load_aabb_views_for_record(record, args, device)
        smpl_model.update_smpl_gt(views)
        if not args.skip_inference:
            run_local_reset_human3r(model, views, local_dir, args, device)
        if not output_complete(local_dir):
            raise FileNotFoundError(f"Local Human3R output is incomplete: {local_dir}")

        pred_data = load_sequence(local_dir, 4, device)
        target_poses, target_joints, bridge_debug = extract_gt_world(
            views,
            pred_data.poses,
            pred_data.joints_world,
            int(args.boundary),
            joint_ids_np,
            device,
        )
        pred_joints_t = torch.from_numpy(pred_data.joints_world).to(device=device, dtype=torch.float32)
        feature = build_feature(pred_joints_t, int(args.boundary), joint_ids).detach().cpu()
        cached.append(
            CachedSample(
                index=index,
                source=str(record["source"]),
                pattern_id=pattern_id,
                record=record,
                local_dir=local_dir,
                aligned_dir=aligned_dir,
                pred_poses=pred_data.poses.astype(np.float32),
                pred_joints=pred_data.joints_world.astype(np.float32),
                target_poses=target_poses,
                target_joints=target_joints,
                feature=feature,
                bridge_debug=bridge_debug,
            )
        )
        del views
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return cached


def apply_transform_batch(
    pred_poses: torch.Tensor,
    pred_joints: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    boundary: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_poses = pred_poses.clone()
    out_joints = pred_joints.clone()
    out_joints[:, boundary:] = torch.einsum("nij,nfkj->nfki", R, pred_joints[:, boundary:]) + t[:, None, None, :]
    out_poses[:, boundary:, :3, :3] = torch.einsum("nij,nfjk->nfik", R, pred_poses[:, boundary:, :3, :3])
    out_poses[:, boundary:, :3, 3] = torch.einsum("nij,nfj->nfi", R, pred_poses[:, boundary:, :3, 3]) + t[:, None, :]
    return out_poses, out_joints


def train_shared_alignment(
    samples: list[CachedSample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    features = torch.cat([sample.feature for sample in samples], dim=0).to(device)
    pred_poses = torch.from_numpy(np.stack([sample.pred_poses for sample in samples])).to(device=device, dtype=torch.float32)
    pred_joints = torch.from_numpy(np.stack([sample.pred_joints for sample in samples])).to(device=device, dtype=torch.float32)
    target_poses = torch.from_numpy(np.stack([sample.target_poses for sample in samples])).to(device=device, dtype=torch.float32)
    target_joints = torch.from_numpy(np.stack([sample.target_joints for sample in samples])).to(device=device, dtype=torch.float32)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    boundary = int(args.boundary)

    aligner = StreamingAlignmentMLP(
        in_dim=features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        max_rot_deg=float(args.max_rot_deg),
        max_trans=float(args.max_trans),
    ).to(device)
    if args.eval_checkpoint is not None:
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        aligner.load_state_dict(checkpoint["model"], strict=True)
    optim = torch.optim.AdamW(aligner.parameters(), lr=float(args.lr), weight_decay=1e-4)
    post = slice(boundary, pred_poses.shape[1])
    history = []
    log_path = args.output_dir / "alignment_train_steps.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    train_steps = 0 if args.eval_only else int(args.steps)
    for step in range(train_steps + 1):
        optim.zero_grad(set_to_none=True)
        rotvec, trans = aligner(features)
        R = so3_exp(rotvec)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R, trans, boundary)
        human_loss = F.smooth_l1_loss(
            aligned_joints[:, post][:, :, joint_ids],
            target_joints[:, post][:, :, joint_ids],
            beta=0.05,
        )
        camera_t_loss = F.smooth_l1_loss(aligned_poses[:, post, :3, 3], target_poses[:, post, :3, 3], beta=0.05)
        camera_r_loss = rotation_geodesic(
            aligned_poses[:, post, :3, :3].reshape(-1, 3, 3),
            target_poses[:, post, :3, :3].reshape(-1, 3, 3),
        ).mean()
        prior_loss = rotvec.pow(2).mean() + trans.pow(2).mean()
        loss = (
            float(args.human_weight) * human_loss
            + float(args.camera_t_weight) * camera_t_loss
            + float(args.camera_r_weight) * camera_r_loss
            + float(args.prior_weight) * prior_loss
        )
        if not args.eval_only:
            loss.backward()
            optim.step()

        if step % int(args.log_every) == 0 or step == train_steps:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "human_loss": float(human_loss.detach().cpu()),
                "camera_t_loss": float(camera_t_loss.detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(camera_r_loss.detach()).cpu()),
                "rotvec_deg_mean": float(torch.rad2deg(rotvec.norm(dim=-1).detach()).mean().cpu()),
                "trans_norm_mean": float(trans.norm(dim=-1).detach().mean().cpu()),
            }
            row["mode"] = "eval_only" if args.eval_only else "train"
            print(json.dumps(row, sort_keys=True), flush=True)
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        rotvec, trans = aligner(features)
        R = so3_exp(rotvec)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R, trans, boundary)
    checkpoint = {
        "model": aligner.state_dict(),
        "features": features.detach().cpu(),
        "joint_ids": joint_ids.detach().cpu(),
        "rotvec": rotvec.detach().cpu(),
        "trans": trans.detach().cpu(),
        "args": vars(args),
        "samples": [
            {
                "source": sample.source,
                "pattern_id": sample.pattern_id,
                "record": sample.record,
            }
            for sample in samples
        ],
    }
    torch.save(checkpoint, args.output_dir / "alignment_head_4source_probe.pth")
    debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "learned_rotvec_deg_norm": torch.rad2deg(rotvec.norm(dim=-1)).detach().cpu().numpy().astype(float).tolist(),
        "learned_trans_norm": trans.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
    }
    return (
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        debug,
    )


def segment_anchor_metrics(joints: np.ndarray, boundary: int, joint_ids: np.ndarray) -> dict:
    hist = joints[:boundary, joint_ids].mean(axis=0)
    b0 = joints[boundary, joint_ids]
    b1 = joints[boundary + 1, joint_ids]
    return {
        "Amean_B0_m": float(np.linalg.norm(hist - b0, axis=-1).mean()),
        "Amean_B1_m": float(np.linalg.norm(hist - b1, axis=-1).mean()),
        "BB_m": float(np.linalg.norm(b0 - b1, axis=-1).mean()),
    }


def nested_get(row: dict, path: str) -> float:
    cur = row
    for part in path.split("."):
        cur = cur[part]
    return float(cur)


def evaluate_and_write_outputs(
    samples: list[CachedSample],
    aligned_poses: np.ndarray,
    aligned_joints: np.ndarray,
    debug: dict,
    args: argparse.Namespace,
) -> dict:
    joint_ids = np.asarray(debug["joint_ids"], dtype=np.int64)
    boundary = int(args.boundary)
    rows = []
    for i, sample in enumerate(samples):
        write_aligned_output(sample.local_dir, sample.aligned_dir, aligned_poses[i], boundary, bool(args.overwrite))
        raw = {
            "camera_post": camera_metrics(sample.pred_poses, sample.target_poses, list(range(boundary, 4))),
            "human_post": human_metrics(sample.pred_joints, sample.target_joints, list(range(boundary, 4)), joint_ids),
            "human_AA": human_metrics(sample.pred_joints, sample.target_joints, list(range(0, boundary)), joint_ids),
            "segment_anchor": segment_anchor_metrics(sample.pred_joints, boundary, joint_ids),
        }
        aligned = {
            "camera_post": camera_metrics(aligned_poses[i], sample.target_poses, list(range(boundary, 4))),
            "human_post": human_metrics(aligned_joints[i], sample.target_joints, list(range(boundary, 4)), joint_ids),
            "human_AA": human_metrics(aligned_joints[i], sample.target_joints, list(range(0, boundary)), joint_ids),
            "segment_anchor": segment_anchor_metrics(aligned_joints[i], boundary, joint_ids),
        }
        rows.append(
            {
                "index": sample.index,
                "source": sample.source,
                "pattern_id": sample.pattern_id,
                "record": sample.record,
                "local_reset": str(sample.local_dir),
                "learned_aligned": str(sample.aligned_dir),
                "raw_metrics": raw,
                "aligned_metrics": aligned,
                "gt_bridge_debug": sample.bridge_debug,
            }
        )

    metric_paths = {
        "cam_rot_deg": "camera_post.mean_r_deg",
        "cam_trans_m": "camera_post.mean_t_m",
        "human_post_m": "human_post.mean_m",
        "Amean_B0_m": "segment_anchor.Amean_B0_m",
        "Amean_B1_m": "segment_anchor.Amean_B1_m",
        "BB_m": "segment_anchor.BB_m",
    }
    flat_rows = []
    for row in rows:
        flat = {
            "index": row["index"],
            "source": row["source"],
            "pattern_id": row["pattern_id"],
        }
        for label, path in metric_paths.items():
            flat[f"raw_{label}"] = nested_get(row["raw_metrics"], path)
            flat[f"aligned_{label}"] = nested_get(row["aligned_metrics"], path)
            flat[f"gain_{label}"] = flat[f"raw_{label}"] - flat[f"aligned_{label}"]
        flat_rows.append(flat)

    csv_path = args.output_dir / "metrics_flat.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    aggregate = {"overall": {}, "by_source": {}}
    for source in ["overall"] + list(args.sources):
        source_rows = flat_rows if source == "overall" else [row for row in flat_rows if row["source"] == source]
        if not source_rows:
            continue
        target = aggregate["overall"] if source == "overall" else aggregate["by_source"].setdefault(source, {})
        for label in metric_paths:
            raw_vals = [row[f"raw_{label}"] for row in source_rows]
            aligned_vals = [row[f"aligned_{label}"] for row in source_rows]
            target[label] = {
                "raw_mean": float(np.mean(raw_vals)),
                "aligned_mean": float(np.mean(aligned_vals)),
                "gain_mean": float(np.mean(np.asarray(raw_vals) - np.asarray(aligned_vals))),
            }

    summary = {
        "method": "shared learned streaming segment alignment; strict original Human3R local-reset; oracle AABB boundary; no hand-written yaw/translation rule",
        "samples": rows,
        "aggregate": aggregate,
        "training_debug": debug,
        "outputs": {
            "checkpoint": str(args.output_dir / "alignment_head_4source_probe.pth"),
            "train_log": str(args.output_dir / "alignment_train_steps.jsonl"),
            "metrics_csv": str(csv_path),
        },
        "streaming_semantics": {
            "alignment_head_runs_on_frame": boundary,
            "later_segment_frames_use_cached_transform": True,
            "uses_future_frames_as_input": False,
            "boundary_is_oracle_for_this_probe": True,
        },
    }
    (args.output_dir / "learned_stream_alignment_4source_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_summary(args.output_dir / "metrics_summary.md", flat_rows, aggregate)
    return summary


def write_markdown_summary(path: Path, flat_rows: list[dict], aggregate: dict) -> None:
    lines = [
        "# Learned Streaming Alignment 4-Source Probe",
        "",
        "Lower is better for all metrics. Gain = raw local-reset - learned aligned.",
        "",
        "## Overall",
        "",
        "| Metric | Raw | Aligned | Gain |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in aggregate["overall"].items():
        lines.append(
            f"| {metric} | {values['raw_mean']:.4f} | {values['aligned_mean']:.4f} | {values['gain_mean']:.4f} |"
        )
    lines += ["", "## By Source", ""]
    for source, metrics in aggregate["by_source"].items():
        lines += [f"### {source}", "", "| Metric | Raw | Aligned | Gain |", "|---|---:|---:|---:|"]
        for metric, values in metrics.items():
            lines.append(
                f"| {metric} | {values['raw_mean']:.4f} | {values['aligned_mean']:.4f} | {values['gain_mean']:.4f} |"
            )
        lines.append("")
    lines += [
        "## Per Sample",
        "",
        "| Source | Pattern | Cam Rot raw->aligned | Cam Trans raw->aligned | Human raw->aligned | Amean-B0 raw->aligned | Amean-B1 raw->aligned | BB raw->aligned |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat_rows:
        lines.append(
            "| {source} | {pattern_id} | {raw_cam_rot_deg:.2f}->{aligned_cam_rot_deg:.2f} | "
            "{raw_cam_trans_m:.3f}->{aligned_cam_trans_m:.3f} | {raw_human_post_m:.3f}->{aligned_human_post_m:.3f} | "
            "{raw_Amean_B0_m:.3f}->{aligned_Amean_B0_m:.3f} | {raw_Amean_B1_m:.3f}->{aligned_Amean_B1_m:.3f} | "
            "{raw_BB_m:.3f}->{aligned_BB_m:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(17)
    np.random.seed(17)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    records = select_records(args)
    (args.output_dir / "selected_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    selected_text = ", ".join(f"{r['source']}:{r['pattern_id']}" for r in records)
    print(f">> selected {len(records)} samples: {selected_text}")
    samples = cache_samples(records, args, device)
    aligned_poses, aligned_joints, debug = train_shared_alignment(samples, args, device)
    summary = evaluate_and_write_outputs(samples, aligned_poses, aligned_joints, debug, args)
    print(json.dumps({"aggregate": summary["aggregate"], "outputs": summary["outputs"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
