#!/usr/bin/env python3
"""Bind every Harmony4D final-table case to its exact frozen runtime cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FINAL_CANDIDATE = "bridge3r_unified_half_translation"
SCHEMA = "Bridge3R-Harmony4D-boundary-association-result-binding-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_final_rows(audit_root: Path) -> tuple[dict[str, tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    rows: dict[str, tuple[dict[str, Any], Path]] = {}
    reports = []
    for report_path in sorted(audit_root.resolve().glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append({"path": str(report_path), "sha256": sha256(report_path)})
        for row in report.get("rows", []):
            if row.get("candidate") != FINAL_CANDIDATE or row.get("status") != "complete":
                continue
            case_id = str(row["case_id"])
            if case_id in rows:
                raise ValueError(f"Duplicate final audit case: {case_id}")
            pairs = row.get("diagnostics", {}).get("boundary", {}).get("matched_pairs")
            if not isinstance(pairs, list):
                raise ValueError(f"{case_id}: final audit lacks diagnostics.boundary.matched_pairs")
            rows[case_id] = (row, report_path)
    if not rows:
        raise ValueError(f"No complete {FINAL_CANDIDATE} rows under {audit_root}")
    return rows, reports


def runtime_index(roots: list[Path]) -> dict[str, tuple[Path, dict[str, Any], Path]]:
    output: dict[str, tuple[Path, dict[str, Any], Path]] = {}
    for root in roots:
        resolved = root.resolve()
        for path in sorted(resolved.rglob("*.runtime.json")):
            runtime = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(runtime.get("record", {}).get("case_id", ""))
            if not case_id:
                raise ValueError(f"{path}: missing record.case_id")
            if case_id in output:
                raise ValueError(f"Duplicate runtime case across roots: {case_id}")
            output[case_id] = (path, runtime, resolved)
    if not output:
        raise ValueError("No runtime reports under --prediction-roots")
    return output


def main() -> None:
    args = parse_args()
    final, reports = read_final_rows(args.audit_root)
    runtimes = runtime_index(list(args.prediction_roots))
    missing = sorted(set(final).difference(runtimes))
    if missing:
        raise ValueError(f"Missing frozen runtime for {len(missing)} final cases: {missing[:10]}")
    records = []
    for case_id in sorted(final):
        row, report_path = final[case_id]
        runtime_path, runtime, root = runtimes[case_id]
        cache_path = runtime_path.with_name(runtime_path.name.replace(".runtime.json", ".npz"))
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        expected_pairs = [list(map(int, pair)) for pair in row["diagnostics"]["boundary"]["matched_pairs"]]
        observed_pairs = [
            list(map(int, pair))
            for pair in runtime.get("geometry", {}).get("association", {}).get("pairs", [])
        ]
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"{case_id}: final-audit boundary pairs disagree with frozen runtime "
                f"{observed_pairs} != {expected_pairs}"
            )
        records.append({
            "case_id": case_id,
            "sequence": str(row["sequence"]),
            "runtime": str(runtime_path.resolve()),
            "cache": str(cache_path.resolve()),
            "source_prediction_root": str(root),
            "source_method": "m3_b0_only",
            "final_audit_report": str(report_path.resolve()),
            "final_boundary_pairs": expected_pairs,
            "final_idf1": float(row["metrics"]["IDF1"]),
            "final_identity_policy": str(row["diagnostics"]["identity"]["policy"]),
        })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(content, encoding="utf-8")
    partial.replace(output)
    specification = {
        "schema_version": SCHEMA,
        "candidate": FINAL_CANDIDATE,
        "case_count": len(records),
        "manifest": str(output),
        "manifest_sha256": sha256(output),
        "audit_reports": reports,
        "prediction_roots": [str(path.resolve()) for path in args.prediction_roots],
        "binding": (
            "Every manifest row validates exact equality between the final unified-audit "
            "boundary matched_pairs and frozen RGB runtime geometry.association.pairs."
        ),
    }
    output.with_suffix(".spec.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(specification, indent=2))


if __name__ == "__main__":
    main()
