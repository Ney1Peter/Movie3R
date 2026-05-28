#!/usr/bin/env python3
"""Build online human-motion pose-correction diagnostics.

This V8.1 baseline is intentionally causal.  It does not use GT camera pose,
GT SMPL trajectory, background matching, or future frames for correction.

For frame 0, it keeps the raw Human3R camera pose.  For each later frame:

1. read the current Human3R camera-space SMPL anchors
   (pelvis, torso, left foot, right foot);
2. read the previous corrected world anchors from history;
3. fit a camera pose that maps current camera-space anchors to the previous
   corrected world anchors;
4. optionally gate/blend this correction based on the raw anchor jump.

This directly tests whether the token-aligned human motion cue can correct the
same pose drift that it detects.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.v8_1_build_human_only_pose_correction import (  # noqa: E402
    TOKEN_BODY_ANCHOR_SPECS,
    fit_rigid,
    load_camera_poses,
    load_human3r_smpl_joints,
    transform_points,
    write_camera_poses,
)


ANCHOR_NAMES = [name for name, _ in TOKEN_BODY_ANCHOR_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_root", type=Path, default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare")
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--output_prefix", default="online_human_motion")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--history_mode",
        choices=["previous", "constant_velocity"],
        default="previous",
        help="previous freezes the four anchors to the last corrected frame; constant_velocity extrapolates one-step anchor motion.",
    )
    parser.add_argument("--gate_low", type=float, default=0.08)
    parser.add_argument("--gate_high", type=float, default=0.35)
    return parser.parse_args()


def read_case_manifest(compare_root: Path) -> list[dict]:
    manifest_path = compare_root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data["input_frames"]


def copy_saved_output_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    copied_video = dst / "output_video.mp4"
    if copied_video.exists():
        copied_video.unlink()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(x * x, axis=-1))))


def rotation_step_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    rel = pose_b[:3, :3] @ pose_a[:3, :3].T
    value = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def so3_log(R: np.ndarray) -> np.ndarray:
    value = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(value))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = w / theta
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def blend_pose(raw_pose: np.ndarray, fit_pose: np.ndarray, gate: float) -> np.ndarray:
    gate = float(np.clip(gate, 0.0, 1.0))
    if gate <= 1e-8:
        return raw_pose.copy()
    if gate >= 1.0 - 1e-8:
        return fit_pose.copy()
    out = raw_pose.copy()
    R_delta = fit_pose[:3, :3] @ raw_pose[:3, :3].T
    out[:3, :3] = so3_exp(gate * so3_log(R_delta)) @ raw_pose[:3, :3]
    out[:3, 3] = raw_pose[:3, 3] + gate * (fit_pose[:3, 3] - raw_pose[:3, 3])
    return out


def gate_from_residual(raw_residual: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("--gate_high must be larger than --gate_low")
    return float(np.clip((raw_residual - low) / (high - low), 0.0, 1.0))


def correction_delta_metrics(raw_pose: np.ndarray, corr_pose: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(corr_pose[:3, 3] - raw_pose[:3, 3])),
        rotation_step_deg(raw_pose, corr_pose),
    )


def target_from_history(
    prev_anchors: np.ndarray,
    prevprev_anchors: np.ndarray | None,
    history_mode: str,
) -> np.ndarray:
    if history_mode == "constant_velocity" and prevprev_anchors is not None:
        return prev_anchors + (prev_anchors - prevprev_anchors)
    return prev_anchors


def build_online_variant(
    variant: str,
    raw_poses: list[np.ndarray],
    pred_cam_anchors: list[np.ndarray],
    specs: list[dict],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[dict], np.ndarray]:
    corrected_poses: list[np.ndarray] = []
    corrected_anchors: list[np.ndarray] = []
    rows: list[dict] = []

    for i, (raw_pose, anchors_cam) in enumerate(zip(raw_poses, pred_cam_anchors)):
        raw_world = transform_points(raw_pose, anchors_cam)

        if i == 0:
            gate = 0.0
            fit_pose = raw_pose.copy()
            corrected_pose = raw_pose.copy()
            target = raw_world.copy()
            fit_world = raw_world.copy()
            raw_residual = 0.0
            fit_residual = 0.0
        else:
            prev = corrected_anchors[-1]
            prevprev = corrected_anchors[-2] if len(corrected_anchors) >= 2 else None
            target = target_from_history(prev, prevprev, args.history_mode)
            fit_pose = fit_rigid(anchors_cam, target)
            fit_world = transform_points(fit_pose, anchors_cam)
            raw_residual = rms(raw_world - target)
            fit_residual = rms(fit_world - target)

            if variant == "always":
                gate = 1.0
            elif variant == "gated":
                gate = gate_from_residual(raw_residual, args.gate_low, args.gate_high)
            else:
                raise ValueError(f"Unknown variant: {variant}")
            corrected_pose = blend_pose(raw_pose, fit_pose, gate)

        corrected_world = transform_points(corrected_pose, anchors_cam)
        corr_delta_t, corr_delta_r = correction_delta_metrics(raw_pose, corrected_pose)
        fit_delta_t, fit_delta_r = correction_delta_metrics(raw_pose, fit_pose)
        if i > 0:
            corrected_step = np.linalg.norm(corrected_world - corrected_anchors[-1], axis=-1)
            raw_step = np.linalg.norm(raw_world - corrected_anchors[-1], axis=-1)
        else:
            corrected_step = np.zeros(len(ANCHOR_NAMES), dtype=np.float64)
            raw_step = np.zeros(len(ANCHOR_NAMES), dtype=np.float64)

        corrected_poses.append(corrected_pose.astype(np.float64))
        corrected_anchors.append(corrected_world.astype(np.float64))
        rows.append(
            {
                "idx": i,
                "label": specs[i]["label"],
                "seq": specs[i]["seq"],
                "frame": int(specs[i]["frame"]),
                "variant": variant,
                "gate": float(gate),
                "raw_history_anchor_rmse": float(raw_residual),
                "fit_history_anchor_rmse": float(fit_residual),
                "corrected_history_anchor_rmse": rms(corrected_world - target),
                "raw_step_mean_to_history": float(raw_step.mean()),
                "corrected_step_mean_to_history": float(corrected_step.mean()),
                "correction_delta_t": corr_delta_t,
                "correction_delta_rot_deg": corr_delta_r,
                "fit_delta_t": fit_delta_t,
                "fit_delta_rot_deg": fit_delta_r,
                "raw_camera_center": raw_pose[:3, 3].tolist(),
                "corrected_camera_center": corrected_pose[:3, 3].tolist(),
            }
        )

    return corrected_poses, rows, np.stack(corrected_anchors, axis=0)


def write_variant_output(
    raw_dir: Path,
    output_dir: Path,
    poses: list[np.ndarray],
    intrinsics: list[np.ndarray],
    rows: list[dict],
    args: argparse.Namespace,
) -> None:
    copy_saved_output_tree(raw_dir, output_dir)
    write_camera_poses(output_dir, poses, intrinsics)
    summary = {
        "correction_type": "online_human_motion_history",
        "note": "Causal diagnostic only. Uses current Human3R camera-space pelvis/torso/feet anchors and previous corrected anchors.",
        "variant": rows[0]["variant"] if rows else None,
        "history_mode": args.history_mode,
        "gate_low": args.gate_low,
        "gate_high": args.gate_high,
        "used_information": [
            "Human3R predicted SMPL camera-space pelvis token anchor",
            "Human3R predicted SMPL camera-space torso token anchor",
            "Human3R predicted SMPL camera-space left-foot token anchor",
            "Human3R predicted SMPL camera-space right-foot token anchor",
            "previous corrected world anchors maintained online",
            "raw Human3R camera pose for gate and residual blend",
        ],
        "not_used_information": [
            "GT camera pose",
            "GT SMPL world trajectory",
            "future frames",
            "background or scene matching",
            "non-token body anchors",
        ],
        "anchor_names": ANCHOR_NAMES,
        "per_frame": rows,
    }
    with open(output_dir / "online_human_motion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def plot_metrics(compare_root: Path, all_rows: list[dict], all_anchors: dict[str, np.ndarray]) -> None:
    out_dir = compare_root / "online_human_motion_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = sorted({row["variant"] for row in all_rows})
    labels = [row["label"] for row in all_rows if row["variant"] == variants[0]]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8), sharex=True)
    for variant in variants:
        rows = [row for row in all_rows if row["variant"] == variant]
        axes[0].plot(x, [r["raw_history_anchor_rmse"] for r in rows], "--o", label=f"{variant} raw->history")
        axes[0].plot(x, [r["corrected_history_anchor_rmse"] for r in rows], "-o", label=f"{variant} corrected->history")
        axes[1].plot(x, [r["gate"] for r in rows], "-o", label=f"{variant} gate")
        axes[2].plot(x, [r["correction_delta_t"] for r in rows], "-o", label=f"{variant} delta t")
    axes[0].set_ylabel("anchor RMSE")
    axes[1].set_ylabel("gate")
    axes[2].set_ylabel("correction delta t")
    for ax in axes:
        ax.axvline(2, color="orange", linestyle="--", alpha=0.75)
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=18, ha="right")
    fig.suptitle("Online human-motion correction from token-aligned anchors")
    fig.tight_layout()
    fig.savefig(out_dir / "online_human_motion_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for variant, anchors in all_anchors.items():
        pelvis = anchors[:, 0]
        torso = anchors[:, 1]
        axes[0].plot(pelvis[:, 0], pelvis[:, 2], "-o", label=f"{variant} pelvis")
        axes[1].plot(torso[:, 0], torso[:, 2], "-o", label=f"{variant} torso")
        for i, point in enumerate(pelvis):
            axes[0].text(point[0], point[2], str(i), fontsize=8)
        for i, point in enumerate(torso):
            axes[1].text(point[0], point[2], str(i), fontsize=8)
    for ax, title in zip(axes, ["Pelvis XZ trajectory", "Torso XZ trajectory"]):
        ax.set_title(title)
        ax.set_xlabel("world X")
        ax.set_ylabel("world Z")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "online_human_motion_anchor_trajectory_xz.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or (args.compare_root / "raw")
    specs = read_case_manifest(args.compare_root)
    raw_poses, intrinsics = load_camera_poses(raw_dir)
    pred_cam_anchors = load_human3r_smpl_joints(raw_dir, intrinsics, TOKEN_BODY_ANCHOR_SPECS, args.device)

    if len(raw_poses) != len(specs) or len(pred_cam_anchors) != len(specs):
        raise RuntimeError(
            f"Mismatched frame counts: specs={len(specs)}, poses={len(raw_poses)}, anchors={len(pred_cam_anchors)}"
        )

    all_rows: list[dict] = []
    all_anchors: dict[str, np.ndarray] = {}
    outputs = {}
    for variant in ["always", "gated"]:
        poses, rows, anchors = build_online_variant(variant, raw_poses, pred_cam_anchors, specs, args)
        out_dir = args.compare_root / f"{args.output_prefix}_{variant}"
        write_variant_output(raw_dir, out_dir, poses, intrinsics, rows, args)
        all_rows.extend(rows)
        all_anchors[variant] = anchors
        outputs[variant] = str(out_dir)

    raw_world_anchors = np.stack(
        [transform_points(pose, anchors) for pose, anchors in zip(raw_poses, pred_cam_anchors)],
        axis=0,
    )
    all_anchors["raw"] = raw_world_anchors
    plot_metrics(args.compare_root, all_rows, all_anchors)

    summary = {
        "outputs": outputs,
        "diagnostics": str(args.compare_root / "online_human_motion_diagnostics"),
        "history_mode": args.history_mode,
        "gate_low": args.gate_low,
        "gate_high": args.gate_high,
        "anchor_names": ANCHOR_NAMES,
        "per_frame": all_rows,
    }
    with open(args.compare_root / f"{args.output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
