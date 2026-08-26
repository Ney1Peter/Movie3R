#!/usr/bin/env python3
"""Stage and evaluate the frozen Harmony4D boundary-association manifest.

This CPU-only runner never calls a reconstruction model.  It stages one nested
Harmony4D test archive at a time, evaluates the exact cases bound to the final
unified audit, and merges only after all 88 bound cases are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication.bridge3r_iclr2027.evaluate_harmony4d_boundary_association import (  # noqa: E402
    SCHEMA,
    aggregate,
    jsonable,
)


PYTHON = REPO_ROOT / ".venv/bin/python"
STAGER = REPO_ROOT / "versions/v15/harmony4d/stage_archive.py"
EVALUATOR = REPO_ROOT / "publication/bridge3r_iclr2027/evaluate_harmony4d_boundary_association.py"
ENTRIES = {
    "01_hugging": "test/01_hugging.zip",
    "03_grappling2": "test/03_grappling2.zip",
    "05_sword2": "test/05_sword2.zip",
    "06_sword3": "test/06_sword3.zip",
    "08_ballroom2": "test/08_ballroom2.zip",
    "15_mma4": "test/15_mma4.zip",
    "16_mma5": "test/16_mma5.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*", choices=tuple(ENTRIES), default=list(ENTRIES))
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")


def valid_sequence_report(path: Path, sequence: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("schema_version") == SCHEMA
        and payload.get("runtime_manifest") is not None
        and payload.get("cases")
        and {str(row["sequence"]) for row in payload["cases"]} == {sequence}
    )


def stage_and_evaluate(args: argparse.Namespace, sequence: str) -> Path:
    work, output = args.work_root.resolve(), args.output_root.resolve()
    entry = ENTRIES[sequence]
    metadata = work / "metadata" / sequence
    stage = work / "staging" / entry.replace("/", "_").removesuffix(".zip")
    report = output / "per_sequence" / f"{sequence}.json"
    if not valid_sequence_report(report, sequence):
        stage_command = [
            str(PYTHON), str(STAGER), "--outer", str(args.outer.resolve()), "--entry", entry,
            "--work-root", str(work), "--audit-output", str(metadata / "audit.json"),
            "--index-output", str(metadata / "index.json"),
            "--manifest-output", str(metadata / "manifest.jsonl"),
            "--ledger-output", str(metadata / "ledger.json"), "--reserve-gib", str(args.reserve_gib),
        ]
        run(stage_command, output / "logs" / f"{sequence}.stage.log")
        if not stage.is_dir():
            raise FileNotFoundError(stage)
        evaluate_command = [
            str(PYTHON), str(EVALUATOR), "--runtime-manifest", str(args.runtime_manifest.resolve()),
            "--include-sequence", sequence, "--extracted-root", str(stage), "--output", str(report),
            "--bootstrap-samples", str(args.bootstrap_samples), "--seed", str(args.seed),
        ]
        run(evaluate_command, output / "logs" / f"{sequence}.evaluate.log")
    return report


def merge(args: argparse.Namespace, reports: list[Path]) -> Path:
    manifest_rows = [
        json.loads(line) for line in args.runtime_manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    expected = {str(row["case_id"]) for row in manifest_rows}
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    rows = sorted([row for payload in payloads for row in payload["cases"]], key=lambda row: row["case_id"])
    observed = {str(row["case_id"]) for row in rows}
    if observed != expected or len(rows) != len(observed):
        raise ValueError(f"Merge coverage mismatch: observed={len(observed)}, expected={len(expected)}")
    output = args.output_root.resolve() / "final_v1.json"
    final = {
        "schema_version": SCHEMA,
        "status": "complete",
        "runtime_manifest": str(args.runtime_manifest.resolve()),
        "runtime_manifest_sha256": sha256(args.runtime_manifest),
        "case_count": len(rows),
        "sequences": sorted({str(row["sequence"]) for row in rows}),
        "bootstrap": {"samples": int(args.bootstrap_samples), "seed": int(args.seed), "unit": "case"},
        "summary": aggregate(rows, int(args.bootstrap_samples), int(args.seed)),
        "per_sequence_reports": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in reports
        ],
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(final), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    output.with_suffix(".cases.jsonl").write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return output


def main() -> None:
    args = parse_args()
    if not args.outer.is_file() or not args.runtime_manifest.is_file():
        raise FileNotFoundError("--outer or --runtime-manifest")
    reports = [stage_and_evaluate(args, sequence) for sequence in args.sequences]
    if set(args.sequences) == set(ENTRIES):
        final = merge(args, reports)
        print(json.dumps({"status": "complete", "final": str(final), "sequences": list(args.sequences)}, indent=2))
    else:
        print(json.dumps({"status": "partial", "reports": [str(path) for path in reports]}, indent=2))


if __name__ == "__main__":
    main()
