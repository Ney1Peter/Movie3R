#!/usr/bin/env python3
"""Regenerate V7 Stage-A labels with floor-locked human alignment.

This script intentionally does not use the older local-gauge teacher labels.  For
each 30-frame refined clip it rebuilds Human3R saved outputs, hard-levels the
post-shot floor normals to the last pre-shot frame, then aligns post-shot humans
with a yaw rotation around the floor normal plus translation.  The resulting
poses are written as pseudo labels and the existing token dumps are copied with
their target label fields replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


LABEL_KEYS = {
    "label_frame_ids",
    "target_mask",
    "target_delta_t",
    "target_delta_rotvec",
    "target_alpha",
    "target_r_human",
    "target_r_scene",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/data-V7-stage-a/ms-aist/shot2_30f_partial35/usable_cases_35.json"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/data-V7-stage-a/ms-aist/shot2_30f_floor_locked_human35"),
    )
    parser.add_argument("--model_path", type=Path, default=Path("src/human3r_896L.pth"))
    parser.add_argument("--target_count", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align_device", default="cuda")
    parser.add_argument("--token_device", default=None)
    parser.add_argument("--no_pool_large_tokens", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--keep_saved_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_command(cmd: list[str], cwd: Path) -> None:
    print("RUN", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def run_floor_normal_estimation(repo: Path, raw_dir: Path, normals_json: Path, frames_for_floor: list[int]) -> None:
    base_cmd = [
        sys.executable,
        # **========== 原始代码：脚本归档前路径 ==========**
        # repo / "scripts" / "estimate_saved_output_floor_normals.py",
        # **========== 新代码：脚本已归档到 archive_v7 ==========**
        repo / "scripts" / "archive_v7" / "estimate_saved_output_floor_normals.py",
        # **========== 结束 ==========**
        "--output_dir",
        raw_dir,
        "--json_out",
        normals_json,
        "--frames",
        *[str(frame) for frame in frames_for_floor],
    ]
    fallbacks = [
        [],
        ["--conf_threshold", "1.0", "--mask_threshold", "0.9", "--bottom_start", "0.45"],
    ]
    last_error = None
    for extra_args in fallbacks:
        try:
            run_command([*base_cmd, *extra_args], repo)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"floor normal estimation failed, trying fallback: {exc}", flush=True)
    raise last_error


def completed_case_result(case: dict, case_dir: Path, labels_path: Path, tokens_path: Path, metrics_path: Path, target_frames: list[int]) -> dict:
    return {
        "name": case["name"],
        "status": "skipped",
        "case_dir": str(case_dir),
        "source_video": case["source_video"],
        "boundary": int(case["boundary"]),
        "target_frames": target_frames,
        "labels_npz": str(labels_path),
        "tokens_npz": str(tokens_path),
        "teacher_metrics": str(metrics_path),
    }


def so3_log_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    skew = (rotation - rotation.T) / (2.0 * math.sin(theta))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float32)
    return (axis * theta).astype(np.float32)


def load_pose(camera_dir: Path, frame_id: int) -> np.ndarray:
    return np.load(camera_dir / f"{frame_id:06d}.npz")["pose"].astype(np.float32)


def build_labels(case: dict, raw_dir: Path, corrected_dir: Path, labels_path: Path, target_frames: list[int]) -> dict:
    raw_poses = []
    teacher_poses = []
    delta_transforms = []
    delta_t = []
    delta_rotvec = []
    alpha = []

    for frame_id in target_frames:
        raw_pose = load_pose(raw_dir / "camera", frame_id)
        teacher_pose = load_pose(corrected_dir / "camera", frame_id)
        delta = teacher_pose @ np.linalg.inv(raw_pose)
        rotvec = so3_log_np(delta[:3, :3])
        trans = delta[:3, 3].astype(np.float32)
        raw_poses.append(raw_pose)
        teacher_poses.append(teacher_pose)
        delta_transforms.append(delta.astype(np.float32))
        delta_t.append(trans)
        delta_rotvec.append(rotvec)
        alpha.append(1.0 if float(np.linalg.norm(trans) + np.linalg.norm(rotvec)) > 1e-5 else 0.0)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        labels_path,
        frame_ids=np.asarray(target_frames, dtype=np.int32),
        raw_pose=np.stack(raw_poses).astype(np.float32),
        teacher_pose=np.stack(teacher_poses).astype(np.float32),
        delta_transform=np.stack(delta_transforms).astype(np.float32),
        delta_t=np.stack(delta_t).astype(np.float32),
        delta_rotvec=np.stack(delta_rotvec).astype(np.float32),
        alpha=np.asarray(alpha, dtype=np.float32),
        r_human=np.ones((len(target_frames),), dtype=np.float32),
        r_scene=np.zeros((len(target_frames),), dtype=np.float32),
    )

    return {
        "case": case["name"],
        "teacher_type": "floor_locked_human_yaw_translation",
        "labels_path": str(labels_path),
        "source_video": case["source_video"],
        "boundary": int(case["boundary"]),
        "target_frames": target_frames,
        "delta_t_norm": [float(np.linalg.norm(v)) for v in delta_t],
        "delta_rotvec_deg": [float(np.linalg.norm(v) * 180.0 / math.pi) for v in delta_rotvec],
        "alpha": alpha,
        "r_human": [1.0 for _ in target_frames],
        "r_scene": [0.0 for _ in target_frames],
    }


def update_token_labels(old_tokens: Path, new_tokens: Path, labels_path: Path) -> None:
    data = np.load(old_tokens)
    arrays = {key: data[key] for key in data.files if key not in LABEL_KEYS}
    labels = np.load(labels_path)
    frame_ids = arrays["frame_ids"].astype(np.int64)
    label_frames = labels["frame_ids"].astype(np.int64)
    label_index = {int(frame): i for i, frame in enumerate(label_frames.tolist())}

    target_mask = np.zeros((len(frame_ids),), dtype=np.bool_)
    target_delta_t = np.zeros((len(frame_ids), 3), dtype=np.float32)
    target_delta_rotvec = np.zeros((len(frame_ids), 3), dtype=np.float32)
    target_alpha = np.zeros((len(frame_ids),), dtype=np.float32)
    target_r_human = np.zeros((len(frame_ids),), dtype=np.float32)
    target_r_scene = np.zeros((len(frame_ids),), dtype=np.float32)
    for i, frame in enumerate(frame_ids.tolist()):
        j = label_index.get(int(frame))
        if j is None:
            continue
        target_mask[i] = True
        target_delta_t[i] = labels["delta_t"][j]
        target_delta_rotvec[i] = labels["delta_rotvec"][j]
        target_alpha[i] = labels["alpha"][j]
        target_r_human[i] = labels["r_human"][j]
        target_r_scene[i] = labels["r_scene"][j]

    arrays.update(
        {
            "label_frame_ids": label_frames.astype(np.int32),
            "target_mask": target_mask,
            "target_delta_t": target_delta_t,
            "target_delta_rotvec": target_delta_rotvec,
            "target_alpha": target_alpha,
            "target_r_human": target_r_human,
            "target_r_scene": target_r_scene,
        }
    )
    new_tokens.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(new_tokens, **arrays)


def dump_tokens(repo: Path, case: dict, args: argparse.Namespace, tokens_path: Path, labels_path: Path) -> None:
    cmd = [
        sys.executable,
        repo / "scripts" / "archive_v7" / "dump_v7_implicit_tokens.py",
        "--model_path",
        args.model_path if args.model_path.is_absolute() else repo / args.model_path,
        "--seq_path",
        case["source_video"],
        "--pseudo_labels",
        labels_path,
        "--output_npz",
        tokens_path,
        "--device",
        args.token_device or args.device,
        "--overwrite",
    ]
    if not args.no_pool_large_tokens:
        cmd.extend(["--pool_scene_tokens", "--pool_memory_tokens"])
    run_command(cmd, repo)


def selected_cases(manifest: dict, args: argparse.Namespace) -> list[dict]:
    cases = manifest["cases"]
    start = max(0, int(args.start_index))
    cases = cases[start:]
    if args.max_cases > 0:
        cases = cases[: int(args.max_cases)]
    return cases


def process_case(case: dict, args: argparse.Namespace, root: Path, repo: Path) -> dict:
    case_name = case["name"]
    case_dir = args.output_root / "cases" / case_name
    labels_path = case_dir / "pseudo_gt_labels.npz"
    tokens_path = case_dir / "v7_tokens.npz"
    metrics_path = case_dir / "teacher_metrics.json"
    boundary = int(case["boundary"])
    reference_frame = boundary - 1
    target_frames = list(range(boundary, boundary + int(args.target_count)))
    if labels_path.is_file() and tokens_path.is_file() and metrics_path.is_file() and not args.overwrite:
        # **========== 原始代码 ==========**
        # return {"name": case_name, "status": "skipped", "case_dir": str(case_dir)}
        # **========== 新代码 ==========**
        return completed_case_result(case, case_dir, labels_path, tokens_path, metrics_path, target_frames)
        # **========== 结束 ==========**
    if case_dir.exists() and args.overwrite:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    frames_for_floor = [reference_frame] + target_frames
    work_dir = args.output_root / "_tmp_saved_outputs" / case_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "human3r_raw"
    normals_json = work_dir / "floor_normals.json"
    leveled_dir = work_dir / "floor_leveled"
    corrected_dir = work_dir / "floor_locked_human"

    model_path = args.model_path if args.model_path.is_absolute() else repo / args.model_path
    run_command(
        [
            sys.executable,
            repo / "scripts" / "run_human3r_save_output.py",
            "--model_path",
            model_path,
            "--seq_path",
            case["source_video"],
            "--output_dir",
            raw_dir,
            "--device",
            args.device,
            "--overwrite",
        ],
        repo,
    )
    # **========== 原始代码 ==========**
    # run_command(
    #     [
    #         sys.executable,
    #         repo / "scripts" / "estimate_saved_output_floor_normals.py",
    #         "--output_dir",
    #         raw_dir,
    #         "--json_out",
    #         normals_json,
    #         "--frames",
    #         *[str(frame) for frame in frames_for_floor],
    #     ],
    #     repo,
    # )
    # **========== 新代码 ==========**
    run_floor_normal_estimation(repo, raw_dir, normals_json, frames_for_floor)
    # **========== 结束 ==========**
    run_command(
        [
            sys.executable,
            # **========== 原始代码：脚本归档前路径 ==========**
            # repo / "scripts" / "align_saved_output_floor_normals.py",
            # **========== 新代码：脚本已归档到 archive_v7 ==========**
            repo / "scripts" / "archive_v7" / "align_saved_output_floor_normals.py",
            # **========== 结束 ==========**
            "--input_dir",
            raw_dir,
            "--output_dir",
            leveled_dir,
            "--normal_debug_json",
            normals_json,
            "--reference_viewer_frame",
            str(reference_frame),
            "--align_viewer_frames",
            *[str(frame) for frame in target_frames],
            "--overwrite",
        ],
        repo,
    )
    run_command(
        [
            sys.executable,
            # **========== 原始代码：脚本归档前路径 ==========**
            # repo / "scripts" / "align_saved_output_floor_human.py",
            # **========== 新代码：脚本已归档到 archive_v7 ==========**
            repo / "scripts" / "archive_v7" / "align_saved_output_floor_human.py",
            # **========== 结束 ==========**
            "--input_dir",
            leveled_dir,
            "--output_dir",
            corrected_dir,
            "--normal_debug_json",
            leveled_dir / "floor_normal_alignment_debug.json",
            "--reference_viewer_frame",
            str(reference_frame),
            "--align_viewer_frames",
            *[str(frame) for frame in target_frames],
            "--normal_translation_source",
            "human_centroid",
            "--device",
            args.align_device,
            "--overwrite",
        ],
        repo,
    )

    pseudo_summary = build_labels(case, raw_dir, corrected_dir, labels_path, target_frames)
    # **========== 原始代码 ==========**
    # update_token_labels(Path(case["tokens_npz"]), tokens_path, labels_path)
    # **========== 新代码 ==========**
    old_tokens = Path(case["tokens_npz"]) if case.get("tokens_npz") else None
    if old_tokens is not None and old_tokens.is_file():
        update_token_labels(old_tokens, tokens_path, labels_path)
    else:
        dump_tokens(repo, case, args, tokens_path, labels_path)
    # **========== 结束 ==========**

    human_metrics = json.loads((corrected_dir / "floor_human_alignment_metrics.json").read_text())
    floor_metrics = json.loads((leveled_dir / "floor_normal_alignment_metrics.json").read_text())
    metrics = {
        "teacher_type": "floor_locked_human_yaw_translation",
        "case": case_name,
        "source_video": case["source_video"],
        "boundary": boundary,
        "reference_frame": reference_frame,
        "target_frames": target_frames,
        "pseudo_summary": pseudo_summary,
        "floor_alignment": floor_metrics,
        "human_alignment": human_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    case_config = dict(case)
    case_config.update(
        {
            "status": "ok",
            "teacher_type": "floor_locked_human_yaw_translation",
            "case_dir": str(case_dir),
            "labels_npz": str(labels_path),
            "tokens_npz": str(tokens_path),
            "teacher_metrics": str(metrics_path),
            "pseudo_summary": pseudo_summary,
        }
    )
    (case_dir / "case_config.json").write_text(json.dumps(case_config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (case_dir / "pseudo_gt_summary.json").write_text(json.dumps(pseudo_summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if args.keep_saved_outputs:
        saved_dir = case_dir / "saved_outputs"
        if saved_dir.exists():
            shutil.rmtree(saved_dir)
        shutil.move(str(work_dir), str(saved_dir))
    else:
        shutil.rmtree(work_dir)

    return {
        "name": case_name,
        "status": "ok",
        "case_dir": str(case_dir),
        "source_video": case["source_video"],
        "boundary": boundary,
        "target_frames": target_frames,
        "labels_npz": str(labels_path),
        "tokens_npz": str(tokens_path),
        "teacher_metrics": str(metrics_path),
    }


def write_manifest(args: argparse.Namespace, results: list[dict], failures: list[dict]) -> None:
    ok_cases = [item for item in results if item.get("status") in {"ok", "skipped"}]
    manifest = {
        "description": "V7 MS-AIST shot2_30f labels regenerated with floor-locked human yaw/translation alignment.",
        "teacher_type": "floor_locked_human_yaw_translation",
        "input_manifest": str(args.input_manifest),
        "output_root": str(args.output_root),
        "num_cases": len(ok_cases),
        "num_failures": len(failures),
        "cases": ok_cases,
        "failures": failures,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in ["usable_cases_floor_locked_human.json", "stage_a_manifest_floor_locked_human.json"]:
        (args.output_root / name).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (args.output_root / "usable_cases_floor_locked_human.txt").write_text("\n".join(item["name"] for item in ok_cases) + "\n", encoding="utf-8")

    single_human_cases = []
    dropped_multi_human_cases = []
    for item in ok_cases:
        rec = dict(item)
        try:
            tokens = np.load(item["tokens_npz"])
            human_mask = tokens["human_token_mask"].astype(np.bool_)
            max_valid_humans = int(human_mask.sum(axis=1).max()) if human_mask.size else 0
            human_token_slots = int(human_mask.shape[1]) if human_mask.ndim == 2 else 0
            rec["max_valid_humans"] = max_valid_humans
            rec["human_token_slots"] = human_token_slots
            if max_valid_humans <= 1 and human_token_slots <= 1:
                single_human_cases.append(rec)
            else:
                dropped_multi_human_cases.append(rec)
        except Exception as exc:
            rec["single_human_filter_error"] = f"{type(exc).__name__}: {exc}"
            dropped_multi_human_cases.append(rec)
    single_human_manifest = {
        **manifest,
        "description": manifest["description"] + " Filtered to cases with at most one valid human token in every frame.",
        "num_cases": len(single_human_cases),
        "cases": single_human_cases,
        "dropped_multi_human_cases": dropped_multi_human_cases,
        "num_dropped_multi_human_cases": len(dropped_multi_human_cases),
    }
    (args.output_root / "usable_cases_floor_locked_human_single_human.json").write_text(
        json.dumps(single_human_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "usable_cases_floor_locked_human_single_human.txt").write_text(
        "\n".join(item["name"] for item in single_human_cases) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo = repo_root()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.input_manifest.read_text())
    cases = selected_cases(manifest, args)
    results = []
    failures = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case['name']}", flush=True)
        try:
            results.append(process_case(case, args, args.output_root, repo))
        except Exception as exc:
            failure = {"name": case.get("name"), "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            print(json.dumps(failure, sort_keys=True), flush=True)
            # **========== 原始代码 ==========**
            # if args.strict:
            #     raise
            # **========== 新代码 ==========**
            if not args.keep_saved_outputs and case.get("name"):
                failed_work_dir = args.output_root / "_tmp_saved_outputs" / str(case["name"])
                if failed_work_dir.exists():
                    shutil.rmtree(failed_work_dir)
            if args.strict:
                raise
            # **========== 结束 ==========**
        write_manifest(args, results, failures)
    print(json.dumps({"ok": len([r for r in results if r.get("status") in {"ok", "skipped"}]), "failed": len(failures), "output_root": str(args.output_root)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
