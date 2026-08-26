#!/usr/bin/env python3
"""Verify detector/cache boundary equivalence for frozen EgoHumans Test cases.

The audit is read-only: it hashes no predictions and never invokes a model.
It verifies whether every cache materialised at the evaluator boundary is
numerically reusable under the causal detector's first-positive trigger.  A
mismatch is fail-closed: the RGB backbone must be rerun at that proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "output/v19_egohumans/test/predictions"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "evidence/egohumans_detector_equivalence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(runtime_root: Path) -> dict[str, Any]:
    reports = sorted(runtime_root.rglob("*.runtime.json"))
    if not reports:
        raise FileNotFoundError(f"no runtime reports under {runtime_root}")
    rows = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("record", {})
        detector = payload.get("runtime", {}).get("causal_gru_detector", {})
        expected = record.get("boundary_index")
        proposed = detector.get("first_positive_index")
        if expected is None or proposed is None:
            raise ValueError(f"incomplete detector record: {path}")
        rows.append(
            {
                "case_id": str(record.get("case_id", "")),
                "expected_boundary": int(expected),
                "causal_first_positive": int(proposed),
                "match": int(expected) == int(proposed),
                "runtime_report": str(path.relative_to(REPO_ROOT)),
                "runtime_report_sha256": sha256(path),
            }
        )
    if len({row["case_id"] for row in rows}) != len(rows) or any(not row["case_id"] for row in rows):
        raise ValueError("runtime reports do not provide one unique case ID each")
    mismatches = [row for row in rows if not row["match"]]
    return {
        "schema_version": "Bridge3R-EgoHumans-detector-equivalence-audit-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "runtime_report_count": len(rows),
        "matched_first_positive_count": len(rows) - len(mismatches),
        "mismatch_count": len(mismatches),
        "conclusion": (
            "detector-driven boundary is numerically identical to the evaluated boundary "
            "for every frozen runtime report"
            if not mismatches
            else "detector/evaluator boundary mismatch exists; oracle-equivalence cannot be claimed"
        ),
        "rows": rows,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = audit(args.runtime_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key not in {"rows", "mismatches"}}, indent=2))
    if payload["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
