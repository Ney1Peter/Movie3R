#!/usr/bin/env python3
"""Resumable, disk-bounded evaluation of the frozen v16 method on H4D test.

The seven test archives are staged one at a time.  Each archive is verified by
the v15 staging tool, evaluated from immutable cached predictions, and removed
immediately after a complete result has been validated.  The outer ZIP and all
prediction caches are read-only inputs.  No test metric is used for selection
or parameter changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGER = REPO_ROOT / "versions/v15/harmony4d/stage_archive.py"
PROBE = Path(__file__).resolve().with_name("probe_causal_stabilization.py")
REQUIRED_METRICS = {
    "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm",
    "Accel_mm_frame2", "RTE_H3R_percent", "ROE_joint_proxy_deg",
    "Jitter_H3R", "Foot_sliding_cm", "ATE_Sim3_m", "IDs", "IDF1", "Coverage",
}
REFERENCES = (
    "m0_strict_human3r",
    "m15_safe_boundary_permutation_causal_gru",
)

# Small archives first: useful results arrive early and peak disk use stays low.
SEQUENCES: tuple[dict[str, Any], ...] = (
    {
        "slug": "test_01_hugging", "entry": "test/01_hugging.zip",
        "prediction_roots": ("test_01_hugging",), "expected_cases": 4,
    },
    {
        "slug": "test_15_mma4", "entry": "test/15_mma4.zip",
        "prediction_roots": ("test_15_mma4",), "expected_cases": 4,
    },
    {
        "slug": "test_05_sword2", "entry": "test/05_sword2.zip",
        "prediction_roots": ("test_05_sword2",), "expected_cases": 4,
    },
    {
        "slug": "test_08_ballroom2", "entry": "test/08_ballroom2.zip",
        "prediction_roots": ("test_08_ballroom2",), "expected_cases": 4,
    },
    {
        "slug": "test_16_mma5", "entry": "test/16_mma5.zip",
        "prediction_roots": ("test_16_mma5",), "expected_cases": 4,
    },
    {
        "slug": "test_03_grappling2", "entry": "test/03_grappling2.zip",
        "prediction_roots": ("test_03_grappling2",), "expected_cases": 4,
    },
    {
        "slug": "test_06_sword3", "entry": "test/06_sword3.zip",
        "prediction_roots": ("test_06_sword3_a", "test_06_sword3_b"),
        "expected_cases": 4,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outer", type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Harmony4D.zip"),
    )
    parser.add_argument(
        "--work-root", type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v16_test"),
    )
    parser.add_argument(
        "--prediction-root", type=Path,
        default=REPO_ROOT / "output/v15_harmony4d/predictions",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "output/v16_harmony4d/test_batch",
    )
    parser.add_argument(
        "--candidate-json", type=Path,
        default=REPO_ROOT / "versions/v16/harmony4d/frozen_harmony_candidate.json",
    )
    parser.add_argument("--only", nargs="*", help="Optional sequence slugs.")
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def disk_state(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gib": usage.total / 1024**3,
        "used_gib": usage.used / 1024**3,
        "free_gib": usage.free / 1024**3,
    }


def valid_result(path: Path, expected_cases: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("case_count") != expected_cases or payload.get("errors"):
        return False
    rows = [row for row in payload.get("rows", []) if row.get("status") == "complete"]
    references = [
        row for row in payload.get("reference_rows", [])
        if row.get("status") == "complete"
    ]
    if len(rows) != 2 * expected_cases or len(references) != len(REFERENCES) * expected_cases:
        return False
    return all(REQUIRED_METRICS <= set(row.get("metrics", {})) for row in rows + references)


def run(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "TMPDIR": "/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v16_test/tmp",
            },
        )
    return completed.returncode, time.time() - started


def remove_staging(path: Path, work_root: Path) -> None:
    resolved = path.resolve()
    staging_root = (work_root.resolve() / "staging")
    if staging_root not in resolved.parents or resolved == staging_root:
        raise ValueError(f"Refusing unsafe staging removal: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def main() -> None:
    args = parse_args()
    selected = [row for row in SEQUENCES if not args.only or row["slug"] in set(args.only)]
    unknown = set(args.only or ()) - {row["slug"] for row in SEQUENCES}
    if unknown:
        raise ValueError(f"Unknown sequence slugs: {sorted(unknown)}")
    for required in (args.outer, args.candidate_json, args.prediction_root):
        if not required.exists():
            raise FileNotFoundError(required)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "batch_state.json"
    state: dict[str, Any] = {
        "schema_version": "Movie3R-v16-Harmony4D-frozen-test-batch-v1",
        "started_at_epoch": time.time(),
        "outer_preserved": str(args.outer.resolve()),
        "candidate_json": str(args.candidate_json.resolve()),
        "test_used_for_parameter_selection": False,
        "clip_length": 150,
        "sequences": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["started_at_epoch"] = previous.get("started_at_epoch", state["started_at_epoch"])
        state["sequences"] = previous.get("sequences", {})

    failures = []
    for index, spec in enumerate(selected, start=1):
        slug = str(spec["slug"])
        result_path = args.output_root / "per_sequence" / f"{slug}.json"
        stage_path = args.work_root / "staging" / slug
        print(
            f"[{index}/{len(selected)}] {slug}: free={disk_state(args.work_root)['free_gib']:.1f} GiB",
            flush=True,
        )
        if valid_result(result_path, int(spec["expected_cases"])):
            state["sequences"][slug] = {
                "status": "complete_cached", "result": str(result_path.resolve()),
                "disk_after": disk_state(args.work_root),
            }
            atomic_json(state_path, state)
            print(f"[{slug}] validated cached result; skipping", flush=True)
            continue

        staging_outputs = args.output_root / "staging" / slug
        stage_command = [
            sys.executable, str(STAGER),
            "--outer", str(args.outer.resolve()),
            "--entry", str(spec["entry"]),
            "--work-root", str(args.work_root.resolve()),
            "--audit-output", str((staging_outputs / "audit.json").resolve()),
            "--index-output", str((staging_outputs / "index.json").resolve()),
            "--manifest-output", str((staging_outputs / "generated_manifest.jsonl").resolve()),
            "--ledger-output", str((staging_outputs / "stage_ledger.json").resolve()),
            "--reserve-gib", str(float(args.reserve_gib)),
        ]
        returncode, stage_seconds = run(
            stage_command, args.output_root / "logs" / f"{slug}.stage.log"
        )
        if returncode:
            error = f"stage_exit_{returncode}"
            failures.append({"sequence": slug, "error": error})
            state["sequences"][slug] = {
                "status": "error", "error": error, "staging_retained": str(stage_path),
            }
            atomic_json(state_path, state)
            if not args.continue_on_error:
                break
            continue

        prediction_roots = [
            (args.prediction_root / name).resolve() for name in spec["prediction_roots"]
        ]
        missing = [str(path) for path in prediction_roots if not path.is_dir()]
        if missing:
            raise FileNotFoundError(missing)
        probe_command = [
            sys.executable, str(PROBE),
            "--prediction-roots", *map(str, prediction_roots),
            "--extracted-root", str(stage_path.resolve()),
            "--output", str(result_path.resolve()),
            "--candidate-json", str(args.candidate_json.resolve()),
            "--reference-methods", *REFERENCES,
        ]
        returncode, probe_seconds = run(
            probe_command, args.output_root / "logs" / f"{slug}.probe.log"
        )
        complete = returncode == 0 and valid_result(result_path, int(spec["expected_cases"]))
        if not complete:
            error = f"probe_exit_{returncode}_or_incomplete_result"
            failures.append({"sequence": slug, "error": error})
            state["sequences"][slug] = {
                "status": "error", "error": error,
                "stage_seconds": stage_seconds, "probe_seconds": probe_seconds,
                "staging_retained": str(stage_path.resolve()),
            }
            atomic_json(state_path, state)
            if not args.continue_on_error:
                break
            continue

        if not args.keep_staging:
            remove_staging(stage_path, args.work_root)
        state["sequences"][slug] = {
            "status": "complete", "result": str(result_path.resolve()),
            "stage_seconds": stage_seconds, "probe_seconds": probe_seconds,
            "staging_removed": not args.keep_staging,
            "disk_after": disk_state(args.work_root),
        }
        atomic_json(state_path, state)
        print(
            f"[{slug}] complete in {stage_seconds + probe_seconds:.1f}s; "
            f"free={state['sequences'][slug]['disk_after']['free_gib']:.1f} GiB",
            flush=True,
        )

    state["finished_at_epoch"] = time.time()
    state["failures"] = failures
    state["complete_sequences"] = sum(
        row.get("status") in {"complete", "complete_cached"}
        for row in state["sequences"].values()
    )
    state["requested_sequences"] = len(selected)
    atomic_json(state_path, state)
    print(json.dumps({
        "state": str(state_path.resolve()),
        "complete_sequences": state["complete_sequences"],
        "requested_sequences": len(selected),
        "failures": failures,
        "disk": disk_state(args.work_root),
    }, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
