#!/usr/bin/env python3
"""Compare the legacy EgoHumans final adapter with the publication core.

Both sides receive the same immutable `m3_b0_only` cache and the same
prediction-only association pairs.  This audit never stages RGB, opens GT, or
calls an evaluator.  It establishes exact array equivalence for all retained
EgoHumans Test cache records before the paper may describe the new core as a
replacement for that adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from publication.bridge3r_iclr2027.runtime_contract import (
    apply_locked_transaction,
    load_method_lock,
)
from versions.v16.harmony4d.causal_stabilization import Candidate, apply_candidate


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "output/v19_egohumans/test/predictions"
DEFAULT_CANDIDATE = REPO_ROOT / "versions/v19/egohumans/frozen_final_candidate.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "evidence/egohumans_publication_entry_equivalence.json"
ARRAY_KEYS = (
    "cameras_c2w",
    "joints_world",
    "vertices_world",
    "valid",
    "native_ids",
    "persistent_ids",
)


def arrays_numerically_identical(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare result arrays exactly while treating paired NaNs as missingness.

    The packed prediction caches use NaN to pad an absent person slot.  NumPy's
    default ``array_equal`` deliberately regards ``NaN != NaN``; that would
    falsely classify two byte-identical post-processing paths as different
    whenever both preserve the same padding.  The publication criterion is
    therefore exact finite values, exact dtype/shape, and an identical NaN
    mask--not a lossy tolerance-based comparison.
    """

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_geometry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = str(payload.get("final_method_name", ""))
    rows = [row for row in payload.get("candidates", []) if row.get("name") == selected]
    if len(rows) != 1:
        raise ValueError("could not resolve one frozen EgoHumans final candidate")
    geometry = dict(rows[0].get("geometry") or {})
    expected = load_method_lock()["method"]["fixed_parameters"]
    actual = {
        "camera_alpha": geometry.get("camera_alpha"),
        "boundary_kind": geometry.get("boundary_kind"),
        "boundary_blend": geometry.get("boundary_blend"),
        "reliability_gate": geometry.get("gate_max_boundary_residual_m") is not None,
        "root_filter": geometry.get("root_alpha") is not None,
    }
    if actual != expected:
        raise ValueError(f"legacy EgoHumans final geometry differs from publication lock: {actual}")
    return geometry


def source_arrays(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        output = {}
        for key in ARRAY_KEYS:
            source_key = "m3_b0_only__" + key
            if source_key not in cache:
                raise ValueError(f"{cache_path}: missing {source_key}")
            output[key] = np.asarray(cache[source_key]).copy()
    return output


def audit(runtime_root: Path, candidate_path: Path) -> dict[str, Any]:
    geometry = final_geometry(candidate_path)
    reports = sorted(runtime_root.rglob("*.runtime.json"))
    if not reports:
        raise FileNotFoundError(runtime_root)
    rows = []
    for runtime_path in reports:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        record = runtime.get("record", {})
        pairs = [tuple(map(int, pair)) for pair in runtime.get("geometry", {}).get("association", {}).get("pairs", [])]
        cache_path = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
        source = source_arrays(cache_path)
        legacy, _ = apply_candidate(
            source,
            int(record["boundary_index"]),
            pairs,
            Candidate(**geometry),
        )
        publication, debug = apply_locked_transaction(
            source,
            boundary=int(record["boundary_index"]),
            pairs=pairs,
            cut_detected=True,
        )
        mismatched = [
            key
            for key in ARRAY_KEYS
            if not arrays_numerically_identical(legacy[key], publication[key])
        ]
        rows.append(
            {
                "case_id": str(record.get("case_id", "")),
                "cache": str(cache_path.relative_to(REPO_ROOT)),
                "cache_sha256": sha256(cache_path),
                "boundary": int(record["boundary_index"]),
                "pair_count": len(pairs),
                "array_equivalent": not mismatched,
                "mismatched_arrays": mismatched,
                "publication_gate_enabled": bool(debug["reliability_gate"]["enabled"]),
                "publication_root_filter": debug["root_filter"] is not None,
            }
        )
    if len({row["case_id"] for row in rows}) != len(rows) or any(not row["case_id"] for row in rows):
        raise ValueError("one unique case ID per runtime report is required")
    failures = [row for row in rows if not row["array_equivalent"]]
    return {
        "schema_version": "Bridge3R-EgoHumans-publication-entry-equivalence-audit-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "candidate": str(candidate_path.relative_to(REPO_ROOT)),
        "candidate_sha256": sha256(candidate_path),
        "array_comparison": (
            "exact dtype/shape and finite values, with identical NaN padding masks "
            "treated as equivalent missingness"
        ),
        "runtime_report_count": len(rows),
        "array_equivalent_count": len(rows) - len(failures),
        "mismatch_count": len(failures),
        "conclusion": (
            "the publication transaction is array-identical to the frozen EgoHumans final adapter on every retained Test cache"
            if not failures
            else "publication-entry equivalence failed; affected arrays are listed"
        ),
        "rows": rows,
        "mismatches": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = audit(args.runtime_root.resolve(), args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"rows", "mismatches"}}, indent=2))
    if payload["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
