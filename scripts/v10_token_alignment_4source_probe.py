#!/usr/bin/env python3
"""Probe whether Human3R tokens carry segment-alignment coordinates.

This is a V10 side branch.  It freezes strict original Human3R, extracts compact
pose/human/state token summaries during local-reset streaming inference, and
trains small alignment heads from A-segment tokens to the B-segment transform.
The purpose is diagnostic: determine whether token-level features can learn
shot segment-to-global alignment without an explicit SMPLX Procrustes proposal.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import to_cpu, todevice
from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v9_learned_stream_alignment_4source_probe import (
    DEFAULT_BAD_SAMPLE_REGISTRY,
    SOURCE_ORDER,
    apply_transform_batch,
    evaluate_and_write_outputs,
    select_records,
)
from v9_learned_stream_alignment_overfit import (
    DEFAULT_RAW_ROOTS,
    StreamingAlignmentMLP,
    extract_gt_world,
    output_complete,
    rotation_geodesic,
    so3_exp,
)
from v9_online_stream_human3r_segment_align import strict_original_model


TOKEN_FEATURE_SETS = {
    "pose_only": ("pose_token_out",),
    "human_only": ("human_token_out",),
    "state_only": ("state_summary_after", "pose_memory_summary_after"),
    "pose_human": ("pose_token_out", "human_token_out"),
    "pose_human_state": (
        "pose_token_out",
        "human_token_out",
        "state_summary_after",
        "pose_memory_summary_after",
    ),
}


@dataclass
class TokenCachedSample:
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
    token_features: dict[str, np.ndarray]
    bridge_debug: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_token_alignment_probe" / "4source_s2",
    )
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_pattern_probe",
    )
    parser.add_argument("--manifest_map", type=Path, default=None)
    parser.add_argument("--manifest_name", default="train_aabb.jsonl")
    parser.add_argument("--bad_sample_registry", type=Path, default=DEFAULT_BAD_SAMPLE_REGISTRY)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_ORDER), choices=list(SOURCE_ORDER))
    parser.add_argument("--samples_per_source", type=int, default=2)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_rot_deg", type=float, default=180.0)
    parser.add_argument("--max_trans", type=float, default=12.0)
    parser.add_argument("--human_weight", type=float, default=5.0)
    parser.add_argument("--camera_t_weight", type=float, default=2.0)
    parser.add_argument("--camera_r_weight", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=list(TOKEN_FEATURE_SETS.keys()),
        choices=list(TOKEN_FEATURE_SETS.keys()),
    )
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_bad_samples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        return str(record["seqs"][0]), str(record["seqs"][2]), int(record["frames"][0])
    return str(record.get("seqA", "")), str(record.get("seqB", "")), int(record.get("start_frame", -1))


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


def token_npz_to_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key].astype(np.float32) for key in data.files}


def token_debug_to_arrays(token_debug: list[dict]) -> dict[str, np.ndarray]:
    keys = [
        "pose_token_in",
        "pose_token_out",
        "human_token_in",
        "human_token_out",
        "state_summary_before",
        "state_summary_new",
        "state_summary_after",
        "pose_memory_summary_before",
        "pose_memory_summary_new",
        "pose_memory_summary_after",
    ]
    arrays = {}
    for key in keys:
        vals = []
        for row in token_debug:
            value = row.get(key)
            if value is None:
                vals.append(None)
            else:
                vals.append(value.detach().cpu().numpy().astype(np.float32)[0])
        if any(v is None for v in vals):
            if all(v is None for v in vals):
                continue
            dim = next(v.shape[0] for v in vals if v is not None)
            vals = [np.zeros((dim,), dtype=np.float32) if v is None else v for v in vals]
        arrays[key] = np.stack(vals).astype(np.float32)
    return arrays


def run_local_reset_human3r_with_tokens(
    model: ARCroco3DStereo,
    views: list[dict],
    local_dir: Path,
    token_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if output_complete(local_dir) and token_path.is_file() and not args.overwrite:
        return token_npz_to_dict(token_path)
    if local_dir.exists() and args.overwrite:
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        preds, batch, _, token_debug = model.forward_recurrent_lighter(
            views,
            str(device),
            ret_state=True,
            use_ttt3r=False,
            return_token_debug=True,
        )
    outputs_cpu = to_cpu({"pred": preds, "views": batch})
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
    token_arrays = token_debug_to_arrays(token_debug)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(token_path, **token_arrays)
    return token_arrays


def cache_samples(records: list[dict], args: argparse.Namespace, device: torch.device) -> list[TokenCachedSample]:
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
    cached = []
    for index, record in enumerate(records):
        pattern_id = str(record.get("pattern_id") or f"{record['source']}_{index:02d}")
        sample_dir = args.output_dir / "samples" / f"{index:02d}_{safe_name(pattern_id)}"
        local_dir = sample_dir / "original_human3r_local_reset"
        token_path = sample_dir / "token_features.npz"
        print(f">> [{index + 1}/{len(records)}] cache tokens {record['source']} {pattern_id}", flush=True)
        views = load_aabb_views_for_record(record, args, device)
        smpl_model.update_smpl_gt(views)
        try:
            if args.skip_inference and token_path.is_file():
                token_features = token_npz_to_dict(token_path)
            else:
                token_features = run_local_reset_human3r_with_tokens(model, views, local_dir, token_path, args, device)
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
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            if getattr(args, "skip_bad_samples", False):
                print(f"!! skip bad token sample {record['source']} {pattern_id}: {exc}", flush=True)
                continue
            raise
        cached.append(
            TokenCachedSample(
                index=index,
                source=str(record["source"]),
                pattern_id=pattern_id,
                record=record,
                local_dir=local_dir,
                aligned_dir=sample_dir / "token_aligned",
                pred_poses=pred_data.poses.astype(np.float32),
                pred_joints=pred_data.joints_world.astype(np.float32),
                target_poses=target_poses,
                target_joints=target_joints,
                token_features=token_features,
                bridge_debug=bridge_debug,
            )
        )
        del views
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cached


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def build_pair_feature(token_features: dict[str, np.ndarray], feature_set: str, boundary: int) -> np.ndarray:
    pieces = []
    for key in TOKEN_FEATURE_SETS[feature_set]:
        if key not in token_features:
            raise KeyError(f"Missing token feature {key}; available={sorted(token_features)}")
        frames = token_features[key].astype(np.float32)
        hist = frames[:boundary].mean(axis=0)
        cur = frames[boundary]
        pieces += [hist, cur, cur - hist, np.abs(cur - hist)]
    feat = np.concatenate(pieces, axis=0).astype(np.float32)
    return feat


def train_token_alignment_variant(
    samples: list[TokenCachedSample],
    args: argparse.Namespace,
    feature_set: str,
    variant_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    features_np = np.stack(
        [build_pair_feature(sample.token_features, feature_set, int(args.boundary)) for sample in samples]
    ).astype(np.float32)
    features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
    pred_poses = torch.from_numpy(np.stack([sample.pred_poses for sample in samples])).to(device=device, dtype=torch.float32)
    pred_joints = torch.from_numpy(np.stack([sample.pred_joints for sample in samples])).to(device=device, dtype=torch.float32)
    target_poses = torch.from_numpy(np.stack([sample.target_poses for sample in samples])).to(device=device, dtype=torch.float32)
    target_joints = torch.from_numpy(np.stack([sample.target_joints for sample in samples])).to(device=device, dtype=torch.float32)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    boundary = int(args.boundary)
    post = slice(boundary, pred_poses.shape[1])

    model = StreamingAlignmentMLP(
        in_dim=features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        max_rot_deg=float(args.max_rot_deg),
        max_trans=float(args.max_trans),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    history = []
    log_path = variant_dir / "alignment_train_steps.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    for step in range(int(args.steps) + 1):
        optim.zero_grad(set_to_none=True)
        rotvec, trans = model(features)
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
        loss.backward()
        optim.step()
        if step % int(args.log_every) == 0 or step == int(args.steps):
            row = {
                "feature_set": feature_set,
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "human_loss": float(human_loss.detach().cpu()),
                "camera_t_loss": float(camera_t_loss.detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(camera_r_loss.detach()).cpu()),
                "rotvec_deg_mean": float(torch.rad2deg(rotvec.norm(dim=-1).detach()).mean().cpu()),
                "trans_norm_mean": float(trans.norm(dim=-1).detach().mean().cpu()),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        rotvec, trans = model(features)
        R = so3_exp(rotvec)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R, trans, boundary)
    checkpoint = {
        "model": model.state_dict(),
        "features": features.detach().cpu(),
        "feature_set": feature_set,
        "feature_keys": TOKEN_FEATURE_SETS[feature_set],
        "joint_ids": joint_ids.detach().cpu(),
        "rotvec": rotvec.detach().cpu(),
        "trans": trans.detach().cpu(),
        "args": vars(args),
        "samples": [
            {"source": sample.source, "pattern_id": sample.pattern_id, "record": sample.record}
            for sample in samples
        ],
    }
    torch.save(checkpoint, variant_dir / "alignment_head_token_probe.pth")
    debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "feature_set": feature_set,
        "feature_keys": list(TOKEN_FEATURE_SETS[feature_set]),
        "feature_dim": int(features.shape[-1]),
        "learned_rotvec_deg_norm": torch.rad2deg(rotvec.norm(dim=-1)).detach().cpu().numpy().astype(float).tolist(),
        "learned_trans_norm": trans.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
    }
    return (
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        debug,
    )


def evaluate_variant(
    samples: list[TokenCachedSample],
    aligned_poses: np.ndarray,
    aligned_joints: np.ndarray,
    debug: dict,
    args: argparse.Namespace,
    feature_set: str,
    variant_dir: Path,
) -> dict:
    variant_samples = []
    for sample in samples:
        variant_samples.append(
            SimpleNamespace(
                index=sample.index,
                source=sample.source,
                pattern_id=sample.pattern_id,
                record=sample.record,
                local_dir=sample.local_dir,
                aligned_dir=variant_dir / "samples" / f"{sample.index:02d}_{safe_name(sample.pattern_id)}" / "token_aligned",
                pred_poses=sample.pred_poses,
                pred_joints=sample.pred_joints,
                target_poses=sample.target_poses,
                target_joints=sample.target_joints,
                bridge_debug=sample.bridge_debug,
            )
        )
    eval_args = SimpleNamespace(**vars(args))
    eval_args.output_dir = variant_dir
    summary = evaluate_and_write_outputs(variant_samples, aligned_poses, aligned_joints, debug, eval_args)
    summary["method"] = f"token-level segment alignment probe: {feature_set}"
    summary["feature_set"] = feature_set
    summary["feature_keys"] = list(TOKEN_FEATURE_SETS[feature_set])
    (variant_dir / "learned_stream_alignment_4source_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return summary


def write_probe_comparison(output_dir: Path, summaries: dict[str, dict]) -> None:
    rows = []
    for feature_set, summary in summaries.items():
        overall = summary["aggregate"]["overall"]
        row = {"feature_set": feature_set}
        for metric, values in overall.items():
            row[f"raw_{metric}"] = values["raw_mean"]
            row[f"aligned_{metric}"] = values["aligned_mean"]
            row[f"gain_{metric}"] = values["gain_mean"]
        rows.append(row)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "token_probe_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# V10 Token-Level Alignment Probe",
        "",
        "Lower is better. Gain = raw local-reset - token aligned.",
        "",
        "| Feature set | Cam Rot | Cam Trans | Human | Amean-B0 | Amean-B1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature_set} | {raw_cam_rot_deg:.2f}->{aligned_cam_rot_deg:.2f} ({gain_cam_rot_deg:+.2f}) | "
            "{raw_cam_trans_m:.3f}->{aligned_cam_trans_m:.3f} ({gain_cam_trans_m:+.3f}) | "
            "{raw_human_post_m:.3f}->{aligned_human_post_m:.3f} ({gain_human_post_m:+.3f}) | "
            "{raw_Amean_B0_m:.3f}->{aligned_Amean_B0_m:.3f} ({gain_Amean_B0_m:+.3f}) | "
            "{raw_Amean_B1_m:.3f}->{aligned_Amean_B1_m:.3f} ({gain_Amean_B1_m:+.3f}) |".format(**row)
        )
    (output_dir / "token_probe_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(19)
    np.random.seed(19)
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
    print(f">> selected {len(records)} samples")
    samples = cache_samples(records, args, device)
    summaries = {}
    for feature_set in args.feature_sets:
        print(f">> train token probe variant: {feature_set}", flush=True)
        variant_dir = args.output_dir / "variants" / feature_set
        aligned_poses, aligned_joints, debug = train_token_alignment_variant(
            samples,
            args,
            feature_set,
            variant_dir,
            device,
        )
        summaries[feature_set] = evaluate_variant(
            samples,
            aligned_poses,
            aligned_joints,
            debug,
            args,
            feature_set,
            variant_dir,
        )
    write_probe_comparison(args.output_dir, summaries)
    print(json.dumps({"output_dir": str(args.output_dir), "feature_sets": list(summaries)}, indent=2))


if __name__ == "__main__":
    main()
