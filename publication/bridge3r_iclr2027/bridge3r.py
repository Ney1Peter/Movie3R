#!/usr/bin/env python3
"""Single publication entry point for the locked Bridge3R boundary core.

This entry point accepts only the algorithm in ``PAPER_METHOD_LOCK.json``.
Dataset adapters are responsible for RGB staging, causal cut detection, and
Human3R clean-reset/B0 inference; once they emit the standard array payload,
this script applies precisely the same post-cut transaction across datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # package import for tests and library callers
    from .audit_egohumans_detector_equivalence import audit as audit_egohumans
    from .runtime_contract import apply_locked_transaction, load_method_lock
except ImportError:  # direct ``python publication/.../bridge3r.py`` execution
    from audit_egohumans_detector_equivalence import audit as audit_egohumans
    from runtime_contract import apply_locked_transaction, load_method_lock


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ARRAYS = (
    "cameras_c2w",
    "joints_world",
    "vertices_world",
    "valid",
    "native_ids",
    "persistent_ids",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bindings() -> dict[str, Any]:
    lock = load_method_lock()
    checked = []
    for binding in lock["frozen_result_bindings"]:
        path = REPO_ROOT / binding["result_summary"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != binding["result_summary_sha256"]:
            raise ValueError(f"result source hash mismatch for {binding['dataset']}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if binding["publication_method_id"] not in summary.get("methods", {}):
            raise ValueError(f"publication method absent from {binding['dataset']} summary")
        checked.append(
            {
                "dataset": binding["dataset"],
                "method": binding["publication_method_id"],
                "summary": str(path.relative_to(REPO_ROOT)),
                "sha256": observed,
            }
        )
    return {
        "schema_version": "Bridge3R-publication-binding-validation-v1",
        "status": "PASS",
        "method": lock["method"]["name"],
        "fixed_parameters": lock["method"]["fixed_parameters"],
        "bindings": checked,
    }


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in REQUIRED_ARRAYS if key not in payload]
        if missing:
            raise ValueError(f"missing standard Bridge3R arrays: {missing}")
        arrays = {key: np.asarray(payload[key]).copy() for key in REQUIRED_ARRAYS}
    frame_count = len(arrays["valid"])
    if not frame_count or any(len(arrays[key]) != frame_count for key in REQUIRED_ARRAYS):
        raise ValueError("standard arrays have inconsistent frame dimensions")
    return arrays


def apply_npz(args: argparse.Namespace) -> dict[str, Any]:
    arrays = load_arrays(args.input.resolve())
    pairs = json.loads(args.pairs_json)
    if not isinstance(pairs, list):
        raise ValueError("--pairs-json must encode a JSON list")
    output, diagnostics = apply_locked_transaction(
        arrays,
        boundary=None if args.no_cut else int(args.boundary),
        pairs=[tuple(pair) for pair in pairs],
        cut_detected=not args.no_cut,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    diagnostic_path = args.output.with_suffix(args.output.suffix + ".runtime.json")
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema_version": "Bridge3R-publication-array-runtime-v1",
                "input": str(args.input),
                "output": str(args.output),
                "boundary": None if args.no_cut else int(args.boundary),
                "pairs": pairs,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "output": str(args.output),
        "diagnostics": str(diagnostic_path),
        "cut_applied": not args.no_cut,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-bindings", help="hash-check all frozen paper result sources")
    detector = subparsers.add_parser("audit-egohumans-detector", help="audit frozen detector/evaluator boundary equivalence")
    detector.add_argument("--runtime-root", type=Path, default=REPO_ROOT / "output/v19_egohumans/test/predictions")
    detector.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evidence/egohumans_detector_equivalence.json")
    equivalence = subparsers.add_parser("audit-egohumans-entry", help="compare the publication core against the frozen EgoHumans final adapter")
    equivalence.add_argument("--runtime-root", type=Path, default=REPO_ROOT / "output/v19_egohumans/test/predictions")
    equivalence.add_argument("--candidate", type=Path, default=REPO_ROOT / "versions/v19/egohumans/frozen_final_candidate.json")
    equivalence.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evidence/egohumans_publication_entry_equivalence.json")
    apply = subparsers.add_parser("apply-arrays", help="apply the locked post-cut transaction to standard clean-reset/B0 arrays")
    apply.add_argument("--input", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--boundary", type=int, default=None)
    apply.add_argument("--pairs-json", default="[]")
    apply.add_argument("--no-cut", action="store_true")
    args = parser.parse_args()

    if args.command == "validate-bindings":
        result: dict[str, Any] = validate_bindings()
    elif args.command == "audit-egohumans-detector":
        result = audit_egohumans(args.runtime_root.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["mismatch_count"]:
            raise SystemExit("detector equivalence audit failed")
        result = {
            key: value for key, value in result.items()
            if key not in {"rows", "mismatches"}
        }
    elif args.command == "audit-egohumans-entry":
        try:
            from .audit_egohumans_publication_equivalence import audit as audit_entry
        except ImportError:
            from audit_egohumans_publication_equivalence import audit as audit_entry
        result = audit_entry(args.runtime_root.resolve(), args.candidate.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["mismatch_count"]:
            raise SystemExit("publication-entry equivalence audit failed")
        result = {
            key: value for key, value in result.items()
            if key not in {"rows", "mismatches"}
        }
    else:
        if not args.no_cut and args.boundary is None:
            raise SystemExit("--boundary is required unless --no-cut is supplied")
        result = apply_npz(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
