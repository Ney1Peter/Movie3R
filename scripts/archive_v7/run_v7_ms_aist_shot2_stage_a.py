#!/usr/bin/env python3
"""Run the V7 MS-AIST shot2 Stage-A pilot pipeline.

The pipeline is intentionally small and storage-aware:

1. run Human3R raw saved output;
2. build offline human-scene teacher pseudo labels;
3. dump causal implicit tokens with pooled scene/memory tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2

from build_v7_h36m_pseudo_gt_smoke import SmokeCase, build_pseudo_labels, run_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/data-V7-shot-change-clips/ms-aist/videos/shot2"),
    )
    parser.add_argument(
        "--source_manifest",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/data-V7-shot-change-clips/ms-aist/manifest.json"),
    )
    parser.add_argument(
        "--refined_manifest",
        type=Path,
        default=None,
        help="Optional refined 30-frame manifest. If set, use its accepted clips and local boundary frames.",
    )
    parser.add_argument("--output_root", type=Path, default=Path("output/v7_ms_aist_shot2_stage_a"))
    parser.add_argument("--model_path", type=Path, default=Path("src/human3r_896L.pth"))
    parser.add_argument("--num_clips", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--min_detection_score",
        type=float,
        default=0.2,
        help="Skip extracted clips whose shot-change score is below this value. Use a negative value to disable.",
    )
    parser.add_argument("--target_count", type=int, default=3)
    parser.add_argument("--stable_offset", type=int, default=3)
    parser.add_argument("--stable_count", type=int, default=27)
    parser.add_argument("--subset_margin", type=int, default=8)
    parser.add_argument("--subset_count", type=int, default=31)
    parser.add_argument("--steps_per_frame", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher_device", default=None)
    parser.add_argument("--raw_device", default=None)
    parser.add_argument("--token_device", default=None)
    parser.add_argument("--skip_raw", action="store_true")
    parser.add_argument("--skip_teacher", action="store_true")
    parser.add_argument("--skip_tokens", action="store_true")
    parser.add_argument("--no_pool_large_tokens", action="store_true")
    parser.add_argument("--cleanup_after_tokens", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Abort on the first failed case.")
    parser.add_argument("--retry_failed", action="store_true", help="Retry cases previously marked as failed.")
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if count <= 0:
        raise ValueError(f"Could not infer frame count: {path}")
    return count


# **========== 原始代码 ==========**
# def selected_detections(args: argparse.Namespace) -> list[dict]:
#     manifest = json.loads(args.source_manifest.read_text())
#     rows = []
#     for item in manifest.get("videos", []):
#         info = item.get("info", {})
#         fps = float(info.get("fps", 30.0) or 30.0)
#         for det in item.get("detections", []):
#             output_path = Path(det["output_path"])
#             if output_path.parent.resolve() != args.clip_root.resolve():
#                 continue
#             if det.get("status") != "written" or not output_path.is_file():
#                 continue
#             if args.min_detection_score >= 0 and float(det.get("score", 0.0)) < float(args.min_detection_score):
#                 continue
#             rows.append({"info": info, "detection": det, "fps": fps, "clip_path": output_path})
#     rows = sorted(rows, key=lambda row: row["clip_path"].name)
#     start = max(0, int(args.start_index))
#     end = start + int(args.num_clips)
#     return rows[start:end]


# **========== 新代码 ==========**
def selected_refined_clips(args: argparse.Namespace) -> list[dict]:
    manifest = json.loads(args.refined_manifest.read_text())
    rows = []
    for record in manifest.get("accepted", []):
        output_path = Path(record["output_path"])
        if not output_path.is_file():
            continue
        boundary = int(record.get("refined_boundary_local_frame", record.get("pre_count", 10)))
        info = {
            "refined_manifest": str(args.refined_manifest),
            "source_clip": record.get("source_clip"),
            "preview_path": record.get("preview_path"),
        }
        rows.append(
            {
                "info": info,
                "detection": record,
                "fps": float(record.get("fps", 30.0) or 30.0),
                "clip_path": output_path,
                "boundary": boundary,
            }
        )
    rows = sorted(rows, key=lambda row: row["clip_path"].name)
    start = max(0, int(args.start_index))
    end = start + int(args.num_clips)
    return rows[start:end]


def selected_detections(args: argparse.Namespace) -> list[dict]:
    if args.refined_manifest is not None:
        return selected_refined_clips(args)
    manifest = json.loads(args.source_manifest.read_text())
    rows = []
    for item in manifest.get("videos", []):
        info = item.get("info", {})
        fps = float(info.get("fps", 30.0) or 30.0)
        for det in item.get("detections", []):
            output_path = Path(det["output_path"])
            if output_path.parent.resolve() != args.clip_root.resolve():
                continue
            if det.get("status") != "written" or not output_path.is_file():
                continue
            if args.min_detection_score >= 0 and float(det.get("score", 0.0)) < float(args.min_detection_score):
                continue
            rows.append({"info": info, "detection": det, "fps": fps, "clip_path": output_path})
    rows = sorted(rows, key=lambda row: row["clip_path"].name)
    start = max(0, int(args.start_index))
    end = start + int(args.num_clips)
    return rows[start:end]
# **========== 结束 ==========**


# **========== 原始代码 ==========**
# def make_case(row: dict, output_root: Path, args: argparse.Namespace) -> tuple[SmokeCase, dict]:
#     clip_path = row["clip_path"].resolve()
#     det = row["detection"]
#     fps = float(row["fps"])
#     frame_count = video_frame_count(clip_path)
#     boundary = int(round((float(det["boundary_time"]) - float(det["clip_start"])) * fps))
#     boundary = max(1, min(frame_count - 1, boundary))
#     stable_start = min(frame_count - 1, boundary + int(args.stable_offset))
#     stable_end = min(frame_count - 1, stable_start + int(args.stable_count) - 1)
#     subset_start = max(0, boundary - int(args.subset_margin))
#     subset_count = min(int(args.subset_count), frame_count - subset_start)
#     case_name = clip_path.stem
#     case_dir = output_root / case_name
#     case = SmokeCase(
#         name=case_name,
#         source_video=clip_path,
#         raw_output_dir=case_dir / "human3r_raw",
#         boundary=boundary,
#         target_count=int(args.target_count),
#         stable_start=stable_start,
#         stable_end=stable_end,
#         subset_start=subset_start,
#         subset_count=subset_count,
#     )
#     metadata = {
#         "case": asdict(case),
#         "frame_count": int(frame_count),
#         "fps": fps,
#         "source_detection": det,
#         "source_info": row["info"],
#     }
#     return case, metadata


# **========== 新代码 ==========**
def make_case(row: dict, output_root: Path, args: argparse.Namespace) -> tuple[SmokeCase, dict]:
    clip_path = row["clip_path"].resolve()
    det = row["detection"]
    fps = float(row["fps"])
    frame_count = video_frame_count(clip_path)
    if "boundary" in row:
        boundary = int(row["boundary"])
    else:
        boundary = int(round((float(det["boundary_time"]) - float(det["clip_start"])) * fps))
    boundary = max(1, min(frame_count - 1, boundary))
    stable_start = min(frame_count - 1, boundary + int(args.stable_offset))
    stable_end = min(frame_count - 1, stable_start + int(args.stable_count) - 1)
    subset_start = max(0, boundary - int(args.subset_margin))
    subset_count = min(int(args.subset_count), frame_count - subset_start)
    case_name = clip_path.stem
    case_dir = output_root / case_name
    case = SmokeCase(
        name=case_name,
        source_video=clip_path,
        raw_output_dir=case_dir / "human3r_raw",
        boundary=boundary,
        target_count=int(args.target_count),
        stable_start=stable_start,
        stable_end=stable_end,
        subset_start=subset_start,
        subset_count=subset_count,
    )
    metadata = {
        "case": asdict(case),
        "frame_count": int(frame_count),
        "fps": fps,
        "source_detection": det,
        "source_info": row["info"],
        "input_mode": "refined_manifest" if "boundary" in row else "source_manifest",
    }
    return case, metadata
# **========== 结束 ==========**


def run_command(cmd: list[str], repo_root: Path) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def has_saved_raw(raw_dir: Path) -> bool:
    return (raw_dir / "camera" / "000000.npz").is_file()


def run_raw(case: SmokeCase, args: argparse.Namespace, repo_root: Path) -> None:
    if has_saved_raw(case.raw_output_dir) and not args.overwrite:
        return
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_human3r_save_output.py"),
        "--model_path",
        str((repo_root / args.model_path).resolve() if not args.model_path.is_absolute() else args.model_path.resolve()),
        "--seq_path",
        str(case.source_video),
        "--output_dir",
        str(case.raw_output_dir),
        "--device",
        str(args.raw_device or args.device),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_command(cmd, repo_root)


def run_token_dump(case: SmokeCase, case_dir: Path, args: argparse.Namespace, repo_root: Path) -> None:
    output_npz = case_dir / "v7_tokens.npz"
    if output_npz.is_file() and not args.overwrite:
        return
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "archive_v7" / "dump_v7_implicit_tokens.py"),
        "--model_path",
        str((repo_root / args.model_path).resolve() if not args.model_path.is_absolute() else args.model_path.resolve()),
        "--seq_path",
        str(case.source_video),
        "--pseudo_labels",
        str(case_dir / "pseudo_gt_labels.npz"),
        "--output_npz",
        str(output_npz),
        "--device",
        str(args.token_device or args.device),
        "--overwrite",
    ]
    if not args.no_pool_large_tokens:
        cmd.extend(["--pool_scene_tokens", "--pool_memory_tokens"])
    run_command(cmd, repo_root)


def cleanup_case(case: SmokeCase, case_dir: Path) -> None:
    metrics_src = case_dir / "teacher_corrected" / "post_shot_local_gauge_teacher_metrics.json"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, case_dir / "teacher_metrics.json")
    for path in [
        case.raw_output_dir,
        case_dir / "teacher_corrected",
        case_dir / "teacher_subset",
        case_dir / "raw_subset",
    ]:
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    output_root = (repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = selected_detections(args)
    if not rows:
        raise ValueError("No shot2 clips selected")

    manifest_cases = []
    for row in rows:
        case, metadata = make_case(row, output_root, args)
        case_dir = output_root / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_config_path = case_dir / "case_config.json"
        if not args.dry_run and not args.overwrite and not args.retry_failed and case_config_path.is_file():
            try:
                previous_metadata = json.loads(case_config_path.read_text())
            except json.JSONDecodeError:
                previous_metadata = None
            if previous_metadata is not None and previous_metadata.get("status") == "failed":
                manifest_cases.append(previous_metadata)
                continue
        if args.dry_run:
            metadata["status"] = "dry_run"
        with open(case_config_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        manifest_cases.append(metadata)
        if args.dry_run:
            continue

        # **========== 原始代码 ==========**
        # if not args.skip_raw:
        #     run_raw(case, args, repo_root)
        # if not args.skip_teacher:
        #     teacher_args = SimpleNamespace(
        #         device=args.teacher_device or args.device,
        #         steps_per_frame=int(args.steps_per_frame),
        #         overwrite=bool(args.overwrite),
        #     )
        #     run_teacher(case, case_dir, teacher_args, repo_root)
        #     metadata["pseudo_summary"] = build_pseudo_labels(case, case_dir)
        # if not args.skip_tokens:
        #     run_token_dump(case, case_dir, args, repo_root)
        # if args.cleanup_after_tokens and (case_dir / "v7_tokens.npz").is_file():
        #     cleanup_case(case, case_dir)

        # **========== 新代码 ==========**
        metadata["status"] = "running"
        metadata["stage"] = "raw"
        try:
            if not args.skip_raw:
                run_raw(case, args, repo_root)
            metadata["stage"] = "teacher"
            if not args.skip_teacher:
                teacher_args = SimpleNamespace(
                    device=args.teacher_device or args.device,
                    steps_per_frame=int(args.steps_per_frame),
                    overwrite=bool(args.overwrite),
                )
                run_teacher(case, case_dir, teacher_args, repo_root)
                metadata["pseudo_summary"] = build_pseudo_labels(case, case_dir)
            metadata["stage"] = "tokens"
            if not args.skip_tokens:
                run_token_dump(case, case_dir, args, repo_root)
            metadata["stage"] = "cleanup"
            if args.cleanup_after_tokens and (case_dir / "v7_tokens.npz").is_file():
                cleanup_case(case, case_dir)
            metadata["status"] = "ok"
            metadata.pop("stage", None)
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["failed_stage"] = metadata.pop("stage", "unknown")
            metadata["error_type"] = type(exc).__name__
            metadata["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                metadata["returncode"] = int(exc.returncode)
                metadata["cmd"] = [str(part) for part in exc.cmd]
            print(
                json.dumps(
                    {
                        "case": case.name,
                        "status": "failed",
                        "failed_stage": metadata["failed_stage"],
                        "error_type": metadata["error_type"],
                        "error": metadata["error"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.strict:
                raise
        # **========== 结束 ==========**
        with open(case_config_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=str)
            f.write("\n")

    manifest = {
        "source_manifest": str(args.source_manifest),
        "refined_manifest": str(args.refined_manifest) if args.refined_manifest is not None else None,
        "input_mode": "refined_manifest" if args.refined_manifest is not None else "source_manifest",
        "clip_root": str(args.clip_root),
        "output_root": str(output_root),
        "num_cases": len(manifest_cases),
        "min_detection_score": float(args.min_detection_score),
        "pool_large_tokens": not args.no_pool_large_tokens,
        "cleanup_after_tokens": bool(args.cleanup_after_tokens),
        "cases": manifest_cases,
    }
    with open(output_root / "stage_a_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
