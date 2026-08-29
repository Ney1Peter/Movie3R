#!/usr/bin/env python3
"""Disk-bounded formal-90 EgoHumans boundary-association audit.

The script first seals exact runtime/cache bindings without opening GT.  It
then stages one capture at a time from the immutable outer ZIP, evaluates only
the formal cases for that capture, removes the expanded capture, and finally
aggregates all case records.  Existing complete per-capture reports are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication.bridge3r_iclr2027 import evaluate_harmony4d_boundary_association as common  # noqa: E402
from versions.v19.egohumans.stage_capture import safe_stage_path, slug  # noqa: E402


STAGER = REPO_ROOT / "versions/v19/egohumans/stage_capture.py"
EVALUATOR = REPO_ROOT / "publication/bridge3r_iclr2027/evaluate_egohumans_boundary_association.py"
SCHEMA = "Bridge3R-EgoHumans-formal90-boundary-association-run-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--capture-report-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=120.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(common.jsonable(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_metrics(capture_report_root: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    reports = sorted(capture_report_root.glob("*/candidate_frozen_final_candidate_*.json"))
    if not reports:
        raise FileNotFoundError(f"No frozen candidate reports under {capture_report_root}")
    for report in reports:
        payload = read_json(report)
        for row in payload.get("rows", []):
            if str(row.get("candidate")) != "v19_egohumans_frozen":
                continue
            case_id = str(row["case_id"])
            if case_id in values:
                raise ValueError(f"Duplicate final candidate row: {case_id}")
            values[case_id] = float(row["metrics"]["IDF1"])
    return values


def seal_bindings(
    rows: list[dict[str, Any]],
    prediction_root: Path,
    capture_report_root: Path,
    output: Path,
) -> list[dict[str, Any]]:
    final_idf1 = candidate_metrics(capture_report_root)
    bindings = []
    for row in rows:
        case_id = str(row["case_id"])
        capture_token = slug(str(row["archive_entry"]))
        runtime = prediction_root / capture_token / f"{case_id}.runtime.json"
        cache = prediction_root / capture_token / f"{case_id}.npz"
        if not runtime.is_file() or not cache.is_file():
            raise FileNotFoundError(f"Missing frozen prediction for {case_id}")
        if case_id not in final_idf1:
            raise KeyError(f"Missing final IDF1 for {case_id}")
        runtime_payload = read_json(runtime)
        pairs = common.parse_runtime_pairs(runtime_payload)
        bindings.append(
            {
                "case_id": case_id,
                "archive_entry": str(row["archive_entry"]),
                "angle_stratum": str(row["angle_stratum"]),
                "camera_rotation_span_deg_evaluator_only": row.get(
                    "camera_rotation_span_deg_evaluator_only"
                ),
                "runtime": str(runtime.resolve()),
                "runtime_sha256": sha256(runtime),
                "cache": str(cache.resolve()),
                "cache_sha256": sha256(cache),
                "final_boundary_pairs": [list(value) for value in pairs],
                "final_idf1": final_idf1[case_id],
            }
        )
    bindings.sort(key=lambda value: value["case_id"])
    text = "".join(json.dumps(value, sort_keys=True) + "\n" for value in bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, output)
    return bindings


def safe_remove(path: Path, parent: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    allowed = parent.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Refusing cleanup outside {allowed}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "TMPDIR": str(log.parent)},
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log}")


def aggregate_case_rows(rows: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "all": common.aggregate(rows, samples, 20260829),
        "by_angle_stratum": {},
    }
    for stratum in ("small", "medium", "large", "extreme"):
        subset = [row for row in rows if row["angle_stratum"] == stratum]
        if subset:
            summary["by_angle_stratum"][stratum] = common.aggregate(
                subset, samples, 20260829 + len(summary["by_angle_stratum"]) + 1
            )
    high_angle = [
        row
        for row in rows
        if row.get("camera_rotation_span_deg_evaluator_only") is not None
        and float(row["camera_rotation_span_deg_evaluator_only"]) >= 150.0
    ]
    summary["ge150deg"] = common.aggregate(high_angle, samples, 20260839) if high_angle else None
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    def line(label: str, value: dict[str, Any]) -> str:
        corr = value["first_post_cut_correspondence"]
        macro = corr["case_macro_accuracy"]
        return (
            f"| {label} | {value['case_count']} | "
            f"{100.0 * corr['pair_micro_accuracy']:.2f} | "
            f"{100.0 * macro['mean']:.2f} "
            f"[{100.0 * macro['ci95_low']:.2f}, {100.0 * macro['ci95_high']:.2f}] | "
            f"{100.0 * corr['correct_continuation_coverage']:.2f} | "
            f"{100.0 * corr['runtime_abstention_rate']:.2f} |\n"
        )

    summary = payload["summary"]
    text = [
        "# EgoHumans formal-90 direct boundary-association audit\n\n",
        "Runtime pairs are prediction-only; GT is opened only by this evaluator.\n\n",
        "| Subset | Cases | Pair micro (%) | Case macro (95% CI, %) | Correct continuation coverage (%) | Abstention (%) |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        line("All", summary["all"]),
    ]
    for name, value in summary["by_angle_stratum"].items():
        text.append(line(name.title(), value))
    if summary.get("ge150deg") is not None:
        text.append(line(r"$\geq150^\circ$", summary["ge150deg"]))
    path.write_text("".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outer = args.outer.resolve()
    formal_manifest = args.formal_manifest.resolve()
    prediction_root = args.prediction_root.resolve()
    capture_report_root = args.capture_report_root.resolve()
    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()
    for path in (outer, formal_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (prediction_root, capture_report_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    formal_rows = load_rows(formal_manifest)
    binding_path = output_root / "formal90_runtime_binding.jsonl"
    bindings = seal_bindings(formal_rows, prediction_root, capture_report_root, binding_path)
    bindings_by_entry: dict[str, list[dict[str, Any]]] = {}
    for binding in bindings:
        bindings_by_entry.setdefault(binding["archive_entry"], []).append(binding)

    state_path = output_root / "run_state.json"
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "running",
        "outer": str(outer),
        "formal_manifest": str(formal_manifest),
        "formal_manifest_sha256": sha256(formal_manifest),
        "runtime_binding": str(binding_path),
        "runtime_binding_sha256": sha256(binding_path),
        "case_count": len(bindings),
        "capture_count": len(bindings_by_entry),
        "captures": {},
        "started_at": time.time(),
    }
    if state_path.is_file():
        previous = read_json(state_path)
        if previous.get("runtime_binding_sha256") == state["runtime_binding_sha256"]:
            state["started_at"] = previous.get("started_at", state["started_at"])
            state["captures"] = previous.get("captures", {})
    atomic_json(state_path, state)

    for index, entry in enumerate(sorted(bindings_by_entry), start=1):
        token = slug(entry)
        capture_output = output_root / "captures" / f"{token}.json"
        expected_ids = sorted(value["case_id"] for value in bindings_by_entry[entry])
        if capture_output.is_file():
            prior = read_json(capture_output)
            if sorted(value["case_id"] for value in prior.get("cases", [])) == expected_ids:
                state["captures"][entry] = {"status": "complete_reused", "output": str(capture_output)}
                atomic_json(state_path, state)
                continue

        print(f">> [{index}/{len(bindings_by_entry)}] {entry}", flush=True)
        audit = output_root / "staging_audits" / token / "audit.json"
        ledger = output_root / "staging_audits" / token / "stage_ledger.json"
        archive_path, stage_root, _ = safe_stage_path(work_root, entry)
        stage_command = [
            sys.executable,
            str(STAGER),
            "--outer", str(outer),
            "--entry", entry,
            "--work-root", str(work_root),
            "--audit-output", str(audit),
            "--ledger-output", str(ledger),
            "--reserve-gib", str(args.reserve_gib),
        ]
        run(stage_command, output_root / "logs" / f"{token}.stage.log")
        subset_manifest = output_root / "bindings" / f"{token}.jsonl"
        subset_manifest.parent.mkdir(parents=True, exist_ok=True)
        subset_manifest.write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in bindings_by_entry[entry]),
            encoding="utf-8",
        )
        evaluate_command = [
            sys.executable,
            str(EVALUATOR),
            "--runtime-manifest", str(subset_manifest),
            "--extracted-root", str(stage_root),
            "--output", str(capture_output),
            "--bootstrap-samples", str(args.bootstrap_samples),
        ]
        try:
            run(evaluate_command, output_root / "logs" / f"{token}.evaluate.log")
        finally:
            safe_remove(stage_root, work_root / "staging")
            safe_remove(archive_path, work_root / "archives")
        state["captures"][entry] = {"status": "complete", "output": str(capture_output)}
        atomic_json(state_path, state)

    case_rows = []
    for entry in sorted(bindings_by_entry):
        report = read_json(output_root / "captures" / f"{slug(entry)}.json")
        case_rows.extend(report["cases"])
    case_rows.sort(key=lambda value: value["case_id"])
    if len(case_rows) != len(bindings):
        raise ValueError(f"Aggregated {len(case_rows)} cases, expected {len(bindings)}")
    payload = {
        "schema_version": "Bridge3R-EgoHumans-formal90-boundary-association-final-v1",
        "protocol": "Bridge3R-EgoHumans-formal90-v1",
        "case_count": len(case_rows),
        "capture_count": len(bindings_by_entry),
        "formal_manifest": str(formal_manifest),
        "formal_manifest_sha256": sha256(formal_manifest),
        "runtime_binding": str(binding_path),
        "runtime_binding_sha256": sha256(binding_path),
        "runtime_gt_access": False,
        "future_post_frames_used_by_association": 0,
        "summary": aggregate_case_rows(case_rows, int(args.bootstrap_samples)),
        "cases": case_rows,
    }
    final_path = output_root / "final.json"
    atomic_json(final_path, payload)
    write_markdown(output_root / "FINAL_REPORT.md", payload)
    state.update(status="complete", completed_at=time.time(), final=str(final_path))
    atomic_json(state_path, state)
    print(json.dumps({"status": "complete", "final": str(final_path), "cases": len(case_rows)}, indent=2))


if __name__ == "__main__":
    main()
