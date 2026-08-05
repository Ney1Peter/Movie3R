#!/usr/bin/env python3
"""Run the frozen Movie3R-v15 runtime over a JSONL case manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CASE = Path(__file__).resolve().with_name("run_case.py")
DEFAULT_OUTPUT = REPO_ROOT / "output/v15/cases"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    p.add_argument("--start-line", type=int, default=1)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--disable-adaptive-joint", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def safe_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT not in resolved.parents:
        raise ValueError(f"Batch output must be inside the repository: {resolved}")
    if resolved in {Path("/"), Path("/data"), Path("/data/wangzheng")}:
        raise ValueError(f"Refusing broad output path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for physical_line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError(f"Manifest line {physical_line} is not a JSON object")
        rows.append((physical_line, value))
    return rows


def main() -> None:
    args = parse_args()
    if args.start_line < 1:
        raise ValueError("--start-line is 1-based")
    if args.max_cases < 0:
        raise ValueError("--max-cases cannot be negative")
    output_root = safe_output(args.output_root)
    rows = [(line, row) for line, row in read_rows(args.manifest) if line >= args.start_line]
    if args.max_cases:
        rows = rows[: args.max_cases]
    if not rows:
        raise ValueError("No manifest cases selected")

    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(REPO_ROOT / "output/v15/tmp")})
    summary: dict[str, Any] = {
        "release": "Movie3R-v15-final",
        "manifest": str(args.manifest.resolve()),
        "output_root": str(output_root),
        "device": "cpu",
        "dry_run": bool(args.dry_run),
        "cases": [],
    }
    for physical_line, record in rows:
        started = time.perf_counter()
        command = [sys.executable, str(RUN_CASE), "--record-json", json.dumps(record, ensure_ascii=False), "--output-root", str(output_root)]
        if args.overwrite:
            command.append("--overwrite")
        if args.disable_adaptive_joint:
            command.append("--disable-adaptive-joint")
        if args.checkpoint is not None:
            command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        item: dict[str, Any] = {"manifest_line": physical_line, "record": record, "command": command}
        if args.dry_run:
            item.update({"status": "dry_run", "elapsed_seconds": 0.0})
            summary["cases"].append(item)
            print(json.dumps(item, ensure_ascii=False))
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
        item.update({
            "status": "ok" if completed.returncode == 0 else "error",
            "returncode": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        })
        summary["cases"].append(item)
        print(json.dumps({"line": physical_line, "status": item["status"], "case_id": record.get("case_id")}, ensure_ascii=False))
        if completed.returncode != 0 and not args.continue_on_error:
            summary["stopped_after_error"] = physical_line
            break

    summary["counts"] = {
        "selected": len(rows),
        "ok": sum(item.get("status") == "ok" for item in summary["cases"]),
        "error": sum(item.get("status") == "error" for item in summary["cases"]),
        "dry_run": sum(item.get("status") == "dry_run" for item in summary["cases"]),
    }
    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "counts": summary["counts"]}, ensure_ascii=False, indent=2))
    if summary["counts"]["error"] and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
