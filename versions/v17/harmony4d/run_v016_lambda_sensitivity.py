#!/usr/bin/env python3
"""Run the preregistered, train-only BRIDGE3R lambda sensitivity study.

This runner has deliberately no switch for a test archive, candidate grid, or
case-selection rule.  It stages three previously unused Harmony4D *train*
archives, freezes their deterministic CS150 manifests, writes a provenance lock
before inference, reuses immutable base-model caches across all lambda values,
and reports every finite-grid result.  The study is sensitivity evidence only:
it cannot alter the fixed publication configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGER = REPO_ROOT / "versions/v15/harmony4d/stage_archive.py"
BUILDER = REPO_ROOT / "versions/v15/harmony4d/build_manifest.py"
RUN_CASE = REPO_ROOT / "versions/v15/harmony4d/run_harmony_case.py"
PROBE = REPO_ROOT / "versions/v17/harmony4d/probe_parallel.py"
OUTER = REPO_ROOT.parent / "data/Harmony4D.zip"
WORK = REPO_ROOT.parent / "data/Harmony4D_work_v016_lambda_sensitivity"
OUTPUT = REPO_ROOT / "output/v17_harmony4d/v016_lambda_sensitivity"
PREREGISTRATION = REPO_ROOT / "versions/v17/HARMONY4D_V016_LAMBDA_SENSITIVITY_PREREGISTRATION_20260826.md"
CANDIDATES = REPO_ROOT / "versions/v17/harmony4d/bridge3r_v016_lambda_sensitivity_candidates.json"
ENTRIES = (
    "train/02_grappling.zip",
    "train/07_ballroom.zip",
    "train/12_mma.zip",
)
EXPECTED_CANDIDATES = (
    "v16_0_m15_geometry",
    "bridge3r_lambda_025",
    "bridge3r_lambda_050",
    "bridge3r_lambda_075",
    "bridge3r_lambda_100",
)
PARENT = EXPECTED_CANDIDATES[0]
METRICS = (
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "MPVPE_mm",
    "Accel_mm_frame2",
    "ATE_Sim3_m",
    "Seam_root_m",
    "IDF1",
    "Coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--reserve-gib", type=float, default=100.0)
    parser.add_argument("--probe-workers", type=int, default=3)
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Freeze only train manifests; do not initialize a model or evaluator.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "TMPDIR": str(WORK / "tmp")},
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log}")


def label(entry: str) -> str:
    return PurePosixPath(entry).stem


def validate_static_inputs() -> None:
    for path in (OUTER, STAGER, BUILDER, RUN_CASE, PROBE, PREREGISTRATION, CANDIDATES):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate = read_json(CANDIDATES)
    observed = tuple(str(row.get("name", "")) for row in candidate.get("candidates", []))
    if observed != EXPECTED_CANDIDATES:
        raise ValueError(f"Unexpected sensitivity candidate grid: {observed}")
    publication = candidate.get("publication_configuration_unchanged", {})
    expected_publication = {
        "name": "bridge3r_unified_half_translation",
        "camera_alpha": 1.0,
        "boundary_kind": "translation",
        "boundary_blend": 0.5,
        "reliability_gate": False,
        "root_filter": False,
    }
    if publication != expected_publication:
        raise ValueError("Publication configuration assertion differs from the frozen contract")


def stage_entry(entry: str, reserve_gib: float) -> dict[str, Any]:
    current = label(entry)
    meta = OUTPUT / "staging" / current
    ledger_path = meta / "stage_ledger.json"
    dev_manifest = meta / "development_manifest.jsonl"
    dev_spec = dev_manifest.with_suffix(".spec.json")
    stage_dir = WORK / "staging" / "_".join(PurePosixPath(entry).with_suffix("").parts)
    if not ledger_path.is_file():
        command(
            [
                sys.executable,
                str(STAGER),
                "--outer", str(OUTER),
                "--entry", entry,
                "--work-root", str(WORK),
                "--audit-output", str(meta / "audit.json"),
                "--index-output", str(meta / "index.json"),
                "--manifest-output", str(meta / "stager_manifest.jsonl"),
                "--ledger-output", str(ledger_path),
                "--reserve-gib", str(reserve_gib),
            ],
            OUTPUT / "logs" / f"{current}.stage.log",
        )
    if not (dev_manifest.is_file() and dev_spec.is_file()):
        ledger = read_json(ledger_path)
        selected = Path(str(ledger.get("selected_audit", "")))
        if not selected.is_file():
            raise FileNotFoundError(f"The stager did not retain its selected audit: {selected}")
        command(
            [
                sys.executable,
                str(BUILDER),
                "--audits", str(selected),
                "--split", "dev",
                "--output", str(dev_manifest),
                "--pre-count", "75",
                "--post-count", "75",
            ],
            OUTPUT / "logs" / f"{current}.manifest.log",
        )
    ledger, spec = read_json(ledger_path), read_json(dev_spec)
    if ledger.get("entry") != entry or ledger.get("status") != "staged_audited_manifest_frozen":
        raise ValueError(f"Invalid staging ledger for {entry}")
    if spec.get("split") != "dev" or int(spec.get("case_count", 0)) != 4:
        raise ValueError(f"Expected four dev cases for {entry}, got {spec}")
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"Staged RGB/GT root is absent: {stage_dir}")
    observed = sha256(dev_manifest)
    if observed != spec.get("manifest_sha256"):
        raise ValueError(f"Development manifest hash mismatch for {entry}")
    return {
        "entry": entry,
        "label": current,
        "staging": str(stage_dir.resolve()),
        "ledger": str(ledger_path.resolve()),
        "selected_audit": str(ledger["selected_audit"]),
        "manifest": str(dev_manifest.resolve()),
        "manifest_sha256": observed,
        "case_count": int(spec["case_count"]),
    }


def provenance_lock(stages: list[dict[str, Any]]) -> dict[str, Any]:
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256(path)
        for path in (PREREGISTRATION, CANDIDATES, RUN_CASE, PROBE)
    }
    return {
        "schema_version": "BRIDGE3R-Harmony4D-v016-lambda-sensitivity-lock-v1",
        "created_at_epoch": time.time(),
        "scope": "train_only_sensitivity; publication configuration unchanged",
        "entries": stages,
        "candidate_order": list(EXPECTED_CANDIDATES),
        "source_hashes": source_hashes,
        "test_archives_allowed": False,
        "selection_depends_on_model_result": False,
    }


def manifest_rows(stage: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(stage["manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(stage["case_count"]):
        raise ValueError(f"Manifest count changed after lock: {stage['manifest']}")
    if any(row.get("split") != "dev" for row in rows):
        raise ValueError("A non-development row reached the inference scheduler")
    return rows


def valid_cache(path: Path, record: dict[str, Any]) -> bool:
    runtime = path.with_name(path.stem + ".runtime.json")
    if not path.is_file() or not runtime.is_file():
        return False
    try:
        return read_json(runtime).get("record", {}).get("case_id") == record["case_id"]
    except (OSError, json.JSONDecodeError):
        return False


def run_inference(stages: list[dict[str, Any]], devices: list[str]) -> dict[str, Any]:
    jobs: deque[tuple[dict[str, Any], int, dict[str, Any], Path]] = deque()
    state: dict[str, Any] = {"status": "inference_started", "cases": {}}
    for stage in stages:
        prediction_root = OUTPUT / "predictions" / str(stage["label"])
        prediction_root.mkdir(parents=True, exist_ok=True)
        for line, record in enumerate(manifest_rows(stage), start=1):
            cache = prediction_root / f"{record['case_id']}.npz"
            if valid_cache(cache, record):
                state["cases"][record["case_id"]] = {"status": "cached", "cache": str(cache)}
            else:
                jobs.append((stage, line, record, cache))
    active: dict[str, tuple[subprocess.Popen[Any], Any, str, float, Path]] = {}
    failures: list[dict[str, Any]] = []
    while jobs or active:
        for device in [value for value in devices if value not in active]:
            if not jobs:
                break
            stage, line, record, cache = jobs.popleft()
            log = cache.parent / "logs" / f"{record['case_id']}.inference.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            cmd = [
                sys.executable, str(RUN_CASE), "--manifest", str(stage["manifest"]),
                "--line", str(line), "--extracted-root", str(stage["staging"]),
                "--output", str(cache), "--device", device,
            ]
            handle.write("COMMAND " + json.dumps(cmd, ensure_ascii=False) + "\n")
            handle.flush()
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "TMPDIR": str(WORK / "tmp")},
            )
            active[device] = (process, handle, str(record["case_id"]), time.monotonic(), cache)
            state["cases"][record["case_id"]] = {"status": "running", "device": device, "pid": process.pid}
        write_json(OUTPUT / "state.json", state)
        if not active:
            continue
        time.sleep(0.5)
        for device, (process, handle, case_id, started, cache) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            row = {
                "status": "complete" if returncode == 0 else "error",
                "device": device,
                "returncode": returncode,
                "seconds": time.monotonic() - started,
                "cache": str(cache),
            }
            state["cases"][case_id] = row
            if returncode:
                failures.append({"case_id": case_id, **row})
            del active[device]
    state["status"] = "inference_complete" if not failures else "inference_error"
    state["failures"] = failures
    write_json(OUTPUT / "state.json", state)
    if failures:
        raise RuntimeError(f"Inference failures: {failures}")
    return state


def run_probes(stages: list[dict[str, Any]], workers: int) -> list[Path]:
    reports = []
    for stage in stages:
        current = str(stage["label"])
        report = OUTPUT / "per_sequence" / f"{current}.json"
        if report.is_file():
            payload = read_json(report)
            if tuple(payload.get("aggregate", {}).get("summary", {}).keys()) == EXPECTED_CANDIDATES:
                reports.append(report)
                continue
        command(
            [
                sys.executable, str(PROBE),
                "--prediction-roots", str(OUTPUT / "predictions" / current),
                "--extracted-root", str(stage["staging"]),
                "--output", str(report),
                "--candidate-json", str(CANDIDATES),
                "--source-method", "m3_b0_only",
                "--workers", str(max(1, min(int(workers), int(stage["case_count"])))),
            ],
            OUTPUT / "logs" / f"{current}.probe.log",
        )
        reports.append(report)
    return reports


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def tex_number(value: Any, digits: int) -> str:
    number = finite(value)
    return "--" if number is None else f"{number:.{digits}f}"


def write_paper_table(summary: dict[str, Any]) -> None:
    """Materialize a supplementary-only table; it never ranks/selects lambda."""

    names = (
        (PARENT, "--"),
        ("bridge3r_lambda_025", "0.25"),
        ("bridge3r_lambda_050", "0.50 (fixed)"),
        ("bridge3r_lambda_075", "0.75"),
        ("bridge3r_lambda_100", "1.00"),
    )
    lines = [
        "% Auto-generated by run_v016_lambda_sensitivity.py from the locked train-only protocol.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Train-only Harmony4D blend sensitivity on the frozen 12-case development manifest. \\method{} remains fixed at $\\lambda=0.50$ regardless of these results; this table is descriptive and does not select a configuration. All rows reuse identical base predictions and differ only in the causal shared-translation blend.}",
        "  \\label{tab:harmony-lambda-sensitivity}",
        "  \\small",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrrrrrr}",
        "    \\toprule",
        r"    Shared-translation blend $\lambda$ & W-MPJPE $\downarrow$ & WA-MPJPE $\downarrow$ & MPJPE $\downarrow$ & MPVPE $\downarrow$ & ATE-Sim3 $\downarrow$ & IDF1 $\uparrow$ & Coverage $\uparrow$ \\",
        "    \\midrule",
    ]
    for candidate, display in names:
        metrics = summary["candidates"][candidate]["metrics"]
        display_tex = "\\textbf{" + display + "}" if candidate == "bridge3r_lambda_050" else display
        values = (
            tex_number(metrics["W-MPJPE_mm"]["mean"], 1),
            tex_number(metrics["WA-MPJPE_mm"]["mean"], 1),
            tex_number(metrics["MPJPE_mm"]["mean"], 1),
            tex_number(metrics["MPVPE_mm"]["mean"], 1),
            tex_number(metrics["ATE_Sim3_m"]["mean"], 3),
            tex_number(metrics["IDF1"]["mean"], 3),
            tex_number(metrics["Coverage"]["mean"], 3),
        )
        lines.append("    " + display_tex + " & " + " & ".join(values) + r" \\")
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table*}",
        "",
    ])
    destination = OUTPUT / "paper" / "lambda_sensitivity.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def summarise(reports: list[Path], lock: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for report in reports:
        payload = read_json(report)
        if payload.get("errors"):
            raise RuntimeError(f"Probe errors in {report}: {payload['errors']}")
        rows.extend(row for row in payload.get("rows", []) if row.get("status") == "complete")
        unavailable.extend(payload.get("skipped_cases", []))
    result: dict[str, Any] = {
        "schema_version": "BRIDGE3R-Harmony4D-v016-lambda-sensitivity-summary-v1",
        "lock": str((OUTPUT / "pre_inference_lock.json").resolve()),
        "lock_sha256": sha256(OUTPUT / "pre_inference_lock.json"),
        "source_reports": [str(path.resolve()) for path in reports],
        "case_count_complete": len({str(row["case_id"]) for row in rows}),
        "evaluator_unavailable_count": len(unavailable),
        "evaluator_unavailable_cases": unavailable,
        "candidates": {},
        "reporting_scope": "train-only sensitivity; does not select or alter the publication configuration",
    }
    by_candidate = {name: [row for row in rows if row.get("candidate") == name] for name in EXPECTED_CANDIDATES}
    if not by_candidate[PARENT]:
        raise ValueError("Parent has no completed development row")
    parent_means: dict[str, float | None] = {}
    for metric in METRICS:
        values = [finite(row.get("metrics", {}).get(metric)) for row in by_candidate[PARENT]]
        values = [value for value in values if value is not None]
        parent_means[metric] = sum(values) / len(values) if values else None
    for candidate, candidate_rows in by_candidate.items():
        if not candidate_rows:
            raise ValueError(f"No completed row for candidate {candidate}")
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = [finite(row.get("metrics", {}).get(metric)) for row in candidate_rows]
            values = [value for value in values if value is not None]
            value = sum(values) / len(values) if values else None
            parent = parent_means[metric]
            metrics[metric] = {
                "count": len(values),
                "mean": value,
                "median": median(values) if values else None,
                "delta_to_parent": None if value is None or parent is None else value - parent,
                "ratio_to_parent": None if value is None or parent is None or abs(parent) < 1e-12 else value / parent,
            }
        mpjpe_ratio = metrics["MPJPE_mm"]["ratio_to_parent"]
        mpvpe_ratio = metrics["MPVPE_mm"]["ratio_to_parent"]
        coverage_delta = metrics["Coverage"]["delta_to_parent"]
        result["candidates"][candidate] = {
            "completed_case_count": len({str(row["case_id"]) for row in candidate_rows}),
            "metrics": metrics,
            "safe_alternative": bool(
                (mpjpe_ratio is None or mpjpe_ratio <= 1.02)
                and (mpvpe_ratio is None or mpvpe_ratio <= 1.02)
                and (coverage_delta is None or coverage_delta >= -1e-12)
            ),
        }
    write_json(OUTPUT / "paper" / "summary.json", result)
    write_paper_table(result)
    return result


def main() -> None:
    args = parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not 1 <= len(devices) <= 3 or len(set(devices)) != len(devices):
        raise ValueError("Use one to three distinct GPU devices")
    validate_static_inputs()
    stages = [stage_entry(entry, args.reserve_gib) for entry in ENTRIES]
    stage_state = {
        "schema_version": "BRIDGE3R-Harmony4D-v016-lambda-sensitivity-stage-v1",
        "stages": stages,
        "candidate_sha256": sha256(CANDIDATES),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "test_archives_allowed": False,
    }
    write_json(OUTPUT / "stage_state.json", stage_state)
    if args.stage_only:
        print(json.dumps(stage_state, indent=2, ensure_ascii=False))
        return
    lock = provenance_lock(stages)
    lock_path = OUTPUT / "pre_inference_lock.json"
    if lock_path.is_file():
        previous = read_json(lock_path)
        if previous.get("candidate_order") != lock["candidate_order"] or previous.get("entries") != lock["entries"] or previous.get("source_hashes") != lock["source_hashes"]:
            raise ValueError("Existing inference lock differs; refusing to mix artifacts")
    else:
        write_json(lock_path, lock)
    run_inference(stages, devices)
    reports = run_probes(stages, args.probe_workers)
    summary = summarise(reports, lock)
    print(json.dumps({
        "summary": str((OUTPUT / "paper" / "summary.json").resolve()),
        "complete_cases": summary["case_count_complete"],
        "evaluator_unavailable": summary["evaluator_unavailable_count"],
        "scope": summary["reporting_scope"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
