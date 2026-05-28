#!/usr/bin/env python3
"""Build V7 pseudo-GT smoke-test outputs for two H36M clips.

This script organizes the existing Human3R saved outputs and runs the current
explicit human-scene teacher to create corrected camera-pose pseudo labels. The
teacher may use decoded SMPL and future stable frames; these labels are intended
to supervise a later causal implicit token student, not to be used directly at
inference.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SmokeCase:
    name: str
    source_video: Path
    raw_output_dir: Path
    boundary: int
    target_count: int
    stable_start: int
    stable_end: int
    subset_start: int
    subset_count: int


CASES = [
    SmokeCase(
        name="h36m_test_boundary63",
        source_video=Path("data-V7-test/h36m/h36_new.mp4"),
        raw_output_dir=Path("output/human3r_h36m_test"),
        boundary=63,
        target_count=3,
        stable_start=66,
        stable_end=100,
        subset_start=62,
        subset_count=40,
    ),
    SmokeCase(
        name="h36m_18s_boundary91",
        source_video=Path("data-V7-test/h36m/h36m_ms_000020_18s_25s.mp4"),
        raw_output_dir=Path("output/human3r_h36m_18s"),
        boundary=91,
        target_count=3,
        stable_start=94,
        stable_end=120,
        subset_start=90,
        subset_count=31,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("output/v7_h36m_pseudo_gt_smoke"),
        help="Directory that will contain case manifests, corrected outputs, and pseudo labels.",
    )
    parser.add_argument("--case", choices=[case.name for case in CASES] + ["all"], default="all")
    parser.add_argument("--device", type=str, default=None, help="Teacher optimization device; defaults to teacher script default.")
    parser.add_argument("--steps_per_frame", type=int, default=800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_teacher", action="store_true", help="Only rebuild pseudo labels from existing teacher outputs.")
    return parser.parse_args()


def resolve_case(case: SmokeCase, repo_root: Path) -> SmokeCase:
    return SmokeCase(
        name=case.name,
        source_video=(repo_root / case.source_video).resolve(),
        raw_output_dir=(repo_root / case.raw_output_dir).resolve(),
        boundary=case.boundary,
        target_count=case.target_count,
        stable_start=case.stable_start,
        stable_end=case.stable_end,
        subset_start=case.subset_start,
        subset_count=case.subset_count,
    )


def run_teacher(case: SmokeCase, case_dir: Path, args: argparse.Namespace, repo_root: Path) -> None:
    # **========== 原始代码：脚本归档前路径 ==========**
    # teacher_script = repo_root / "scripts" / "build_post_shot_local_gauge_teacher.py"
    # **========== 新代码：脚本已归档到 archive_v7 ==========**
    teacher_script = repo_root / "scripts" / "archive_v7" / "build_post_shot_local_gauge_teacher.py"
    # **========== 结束 ==========**
    corrected_dir = case_dir / "teacher_corrected"
    subset_dir = case_dir / "teacher_subset"
    raw_subset_dir = case_dir / "raw_subset"
    metrics_path = corrected_dir / "post_shot_local_gauge_teacher_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        return

    cmd = [
        sys.executable,
        str(teacher_script),
        "--input_dir",
        str(case.raw_output_dir),
        "--output_dir",
        str(corrected_dir),
        "--source_video",
        str(case.source_video),
        "--boundary",
        str(case.boundary),
        "--target_count",
        str(case.target_count),
        "--stable_start",
        str(case.stable_start),
        "--stable_end",
        str(case.stable_end),
        "--subset_output_dir",
        str(subset_dir),
        "--raw_subset_output_dir",
        str(raw_subset_dir),
        "--subset_start",
        str(case.subset_start),
        "--subset_count",
        str(case.subset_count),
        "--steps_per_frame",
        str(args.steps_per_frame),
        "--overwrite",
    ]
    if args.device is not None:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd, cwd=repo_root, check=True)


def load_pose(camera_dir: Path, frame_id: int) -> np.ndarray:
    path = camera_dir / f"{frame_id:06d}.npz"
    data = np.load(path)
    return data["pose"].astype(np.float32)


def so3_log_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    skew = (rotation - rotation.T) / (2.0 * math.sin(theta))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float32)
    return (axis * theta).astype(np.float32)


def teacher_frame_reliability(metrics: dict) -> dict[int, dict[str, float]]:
    reliability = {}
    for frame in metrics.get("frames", []):
        frame_id = int(frame["frame"])
        if frame.get("skipped", False):
            reliability[frame_id] = {"r_scene": 0.0, "r_human": 0.0}
            continue
        matches = frame.get("matches", [])
        if matches:
            r_scene = float(np.mean([float(item.get("dot_abs", 0.0)) for item in matches]))
        else:
            r_scene = 0.0
        reliability[frame_id] = {"r_scene": r_scene, "r_human": 1.0}
    return reliability


def build_pseudo_labels(case: SmokeCase, case_dir: Path) -> dict:
    corrected_dir = case_dir / "teacher_corrected"
    metrics_path = corrected_dir / "post_shot_local_gauge_teacher_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    target_frames = [int(frame) for frame in metrics["target_frames"]]
    reliability = teacher_frame_reliability(metrics)
    raw_poses = []
    teacher_poses = []
    delta_transforms = []
    delta_t = []
    delta_rotvec = []
    alpha = []
    r_human = []
    r_scene = []

    for frame_id in target_frames:
        raw_pose = load_pose(case.raw_output_dir / "camera", frame_id)
        teacher_pose = load_pose(corrected_dir / "camera", frame_id)
        delta = teacher_pose @ np.linalg.inv(raw_pose)
        rotvec = so3_log_np(delta[:3, :3])
        trans = delta[:3, 3].astype(np.float32)
        correction_norm = float(np.linalg.norm(trans) + np.linalg.norm(rotvec))
        rel = reliability.get(frame_id, {"r_human": 0.0, "r_scene": 0.0})

        raw_poses.append(raw_pose)
        teacher_poses.append(teacher_pose)
        delta_transforms.append(delta.astype(np.float32))
        delta_t.append(trans)
        delta_rotvec.append(rotvec)
        alpha.append(1.0 if correction_norm > 1e-5 else 0.0)
        r_human.append(float(rel["r_human"]))
        r_scene.append(float(rel["r_scene"]))

    labels_path = case_dir / "pseudo_gt_labels.npz"
    np.savez_compressed(
        labels_path,
        frame_ids=np.asarray(target_frames, dtype=np.int32),
        raw_pose=np.stack(raw_poses).astype(np.float32),
        teacher_pose=np.stack(teacher_poses).astype(np.float32),
        delta_transform=np.stack(delta_transforms).astype(np.float32),
        delta_t=np.stack(delta_t).astype(np.float32),
        delta_rotvec=np.stack(delta_rotvec).astype(np.float32),
        alpha=np.asarray(alpha, dtype=np.float32),
        r_human=np.asarray(r_human, dtype=np.float32),
        r_scene=np.asarray(r_scene, dtype=np.float32),
    )

    summary = {
        "case": case.name,
        "labels_path": str(labels_path),
        "target_frames": target_frames,
        "boundary": case.boundary,
        "source_video": str(case.source_video),
        "raw_output_dir": str(case.raw_output_dir),
        "teacher_corrected_dir": str(corrected_dir),
        "teacher_subset_dir": str(case_dir / "teacher_subset"),
        "raw_subset_dir": str(case_dir / "raw_subset"),
        "delta_t_norm": [float(np.linalg.norm(v)) for v in delta_t],
        "delta_rotvec_deg": [float(np.linalg.norm(v) * 180.0 / math.pi) for v in delta_rotvec],
        "alpha": alpha,
        "r_human": r_human,
        "r_scene": r_scene,
        "raw_transition": metrics.get("raw_transition", {}),
        "corrected_transition": metrics.get("corrected_transition", {}),
    }
    with open(case_dir / "pseudo_gt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    selected = CASES if args.case == "all" else [case for case in CASES if case.name == args.case]
    output_root = (repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for raw_case in selected:
        case = resolve_case(raw_case, repo_root)
        if not case.source_video.is_file():
            raise FileNotFoundError(case.source_video)
        if not case.raw_output_dir.is_dir():
            raise FileNotFoundError(case.raw_output_dir)
        case_dir = output_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        with open(case_dir / "case_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(case), f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        if not args.skip_teacher:
            run_teacher(case, case_dir, args, repo_root)
        summaries.append(build_pseudo_labels(case, case_dir))

    manifest = {
        "output_root": str(output_root),
        "teacher_type": "explicit_human_scene_post_shot_local_gauge",
        "causal_inference_allowed": False,
        "cases": summaries,
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
