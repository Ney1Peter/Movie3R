#!/usr/bin/env python3
"""Watch a long-running V7 Stage-A generation job and validate outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--status_json", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--cleanup_failed_tmp", action="store_true")
    parser.add_argument("--min_free_gb", type=float, default=20.0)
    return parser.parse_args()


def process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free) / 1024.0 / 1024.0 / 1024.0


def cleanup_failed_tmp(root: Path, failures: list[dict]) -> list[str]:
    removed = []
    tmp_root = root / "_tmp_saved_outputs"
    if not tmp_root.is_dir():
        return removed
    for failure in failures:
        name = failure.get("name")
        if not name:
            continue
        path = tmp_root / str(name)
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(name))
    return removed


def collect_status(root: Path, pid: int) -> dict:
    input_manifest = load_json(root / "all_refined_accepted_manifest.json")
    stage_manifest = load_json(root / "stage_a_manifest_floor_locked_human.json")
    single_manifest = load_json(root / "usable_cases_floor_locked_human_single_human.json")
    cases = stage_manifest.get("cases", []) if stage_manifest else []
    failures = stage_manifest.get("failures", []) if stage_manifest else []
    case_root = root / "cases"
    tmp_root = root / "_tmp_saved_outputs"
    tokens = list(case_root.glob("*/v7_tokens.npz")) if case_root.is_dir() else []
    labels = list(case_root.glob("*/pseudo_gt_labels.npz")) if case_root.is_dir() else []
    metrics = list(case_root.glob("*/teacher_metrics.json")) if case_root.is_dir() else []
    tmp_dirs = [p.name for p in tmp_root.iterdir() if p.is_dir()] if tmp_root.is_dir() else []
    total = int(input_manifest.get("num_cases", 0) or 0)
    ok = int(stage_manifest.get("num_cases", len(cases)) or 0)
    failed = int(stage_manifest.get("num_failures", len(failures)) or 0)
    processed = ok + failed
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "pid": int(pid),
        "pid_running": process_running(pid),
        "total_cases": total,
        "ok_cases": ok,
        "failed_cases": failed,
        "processed_cases": processed,
        "remaining_cases": max(0, total - processed) if total else None,
        "progress": float(processed / total) if total else None,
        "single_human_cases": single_manifest.get("num_cases") if single_manifest else None,
        "dropped_multi_human_cases": single_manifest.get("num_dropped_multi_human_cases") if single_manifest else None,
        "token_files": len(tokens),
        "label_files": len(labels),
        "metric_files": len(metrics),
        "last_ok_case": cases[-1].get("name") if cases else None,
        "last_failure": failures[-1] if failures else None,
        "tmp_dirs": tmp_dirs,
        "tmp_dir_count": len(tmp_dirs),
        "free_gb": disk_free_gb(root),
    }


def validate_outputs(root: Path) -> dict:
    manifest = load_json(root / "stage_a_manifest_floor_locked_human.json")
    cases = manifest.get("cases", [])
    failures = manifest.get("failures", [])
    missing = []
    bad_tokens = []
    for case in cases:
        name = case.get("name")
        labels_path = Path(case.get("labels_npz", ""))
        tokens_path = Path(case.get("tokens_npz", ""))
        metrics_path = Path(case.get("teacher_metrics", ""))
        for kind, path in [("labels", labels_path), ("tokens", tokens_path), ("metrics", metrics_path)]:
            if not path.is_file():
                missing.append({"case": name, "kind": kind, "path": str(path)})
        if tokens_path.is_file():
            try:
                tokens = np.load(tokens_path)
                required = {"target_mask", "target_delta_t", "target_delta_rotvec", "target_alpha", "human_token_mask"}
                absent = sorted(required - set(tokens.files))
                target_frames = case.get("target_frames", [10, 11, 12])
                target_count = len(target_frames)
                target_sum = int(tokens["target_mask"].sum()) if "target_mask" in tokens.files else -1
                if absent or target_sum != target_count:
                    bad_tokens.append({"case": name, "missing_keys": absent, "target_mask_sum": target_sum, "expected": target_count})
            except Exception as exc:  # pragma: no cover - runtime guard
                bad_tokens.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_cases": len(cases),
        "num_failures": len(failures),
        "missing_files": missing,
        "bad_tokens": bad_tokens,
        "ok": not missing and not bad_tokens,
    }
    (root / "final_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def write_line(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    status_json = args.status_json or (args.root / "watch_status.json")
    log_path = args.log or (args.root / "logs" / "watch_status.log")
    done_once = False
    while True:
        status = collect_status(args.root, int(args.pid))
        removed = cleanup_failed_tmp(args.root, load_json(args.root / "stage_a_manifest_floor_locked_human.json").get("failures", [])) if args.cleanup_failed_tmp else []
        if removed:
            status["removed_failed_tmp_dirs"] = removed
        if status["free_gb"] < float(args.min_free_gb):
            status["warning"] = f"free disk below {args.min_free_gb:.1f} GB"
        status_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_line(log_path, status)

        total = status.get("total_cases") or 0
        processed = status.get("processed_cases") or 0
        done = total > 0 and processed >= total
        if done and not status.get("pid_running", False):
            validation = validate_outputs(args.root)
            write_line(log_path, {"final_validation": validation})
            break
        if done and done_once:
            validation = validate_outputs(args.root)
            write_line(log_path, {"final_validation": validation, "note": "processed_all_cases_while_pid_still_reported_running"})
            break
        done_once = bool(done)
        time.sleep(max(10, int(args.interval)))


if __name__ == "__main__":
    main()
