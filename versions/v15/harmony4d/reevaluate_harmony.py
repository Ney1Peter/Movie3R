#!/usr/bin/env python3
"""Re-run the GT-only evaluator over completed immutable Harmony caches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = Path(__file__).resolve().with_name("evaluate_harmony.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtimes = sorted(args.predictions.glob("*.runtime.json"))
    if not runtimes:
        raise ValueError(f"No runtime reports under {args.predictions}")
    args.metrics.mkdir(parents=True, exist_ok=True)
    rows = []
    environment = {**os.environ, "TMPDIR": "/data/wangzheng/iJCV-CODE/data/Harmony4D_work/tmp"}
    for runtime in runtimes:
        case_id = runtime.name.removesuffix(".runtime.json")
        cache = runtime.with_name(case_id + ".npz")
        output = args.metrics / (case_id + ".json")
        if not cache.is_file():
            rows.append({"case_id": case_id, "status": "missing_cache"})
            if not args.continue_on_error:
                break
            continue
        completed = subprocess.run([
            sys.executable, str(EVALUATOR),
            "--cache", str(cache.resolve()),
            "--runtime-report", str(runtime.resolve()),
            "--extracted-root", str(args.extracted_root.resolve()),
            "--output", str(output.resolve()),
        ], cwd=REPO_ROOT, env=environment, text=True, capture_output=True)
        log = args.metrics / "logs" / (case_id + ".reevaluation.json")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({
            "command_evaluator": str(EVALUATOR),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append({
            "case_id": case_id,
            "status": "ok" if completed.returncode == 0 else "evaluation_error",
        })
        print(json.dumps(rows[-1]), flush=True)
        if completed.returncode and not args.continue_on_error:
            break
    summary = {
        "prediction_directory": str(args.predictions.resolve()),
        "evaluator": str(EVALUATOR),
        "selected": len(runtimes),
        "ok": sum(row["status"] == "ok" for row in rows),
        "errors": sum(row["status"] != "ok" for row in rows),
        "rows": rows,
    }
    (args.metrics / "reevaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
