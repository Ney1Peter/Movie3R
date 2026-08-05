#!/usr/bin/env python3
"""Build a new pair-disjoint train ledger for explicit boundary residuals.

The original cross96 pairs trained B0 itself and proved unrealistically easy
for an auxiliary residual head.  This builder therefore selects new training
events whose unordered camera pairs are disjoint from every pair already used
by B0 training, frozen evaluations, VSP development, and VSP confirmation.
It is a data ledger operation only: no model or output is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_manifests import (  # noqa: E402
    FROZEN_180,
    FROZEN_FILES,
    FROZEN_ROOT,
    SOURCE_MANIFESTS,
    canonical_pair,
    ordered_candidates,
    read_jsonl,
    to_cut_event,
)


TRAIN96_ROOT = Path(__file__).resolve().parent / "manifests/train96ps"
VSP_ROOT = REPO_ROOT / "config/manifests/v14_vsp_pair_disjoint_20260802"
DEFAULT_OUTPUT = REPO_ROOT / "config/manifests/v14_explicit_boundary_residual_pair_disjoint_20260803"
SOURCES = tuple(SOURCE_MANIFESTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-source", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def pairs_from_cut_events(path: Path) -> set[tuple[str, str]]:
    output: set[tuple[str, str]] = set()
    for record in read_jsonl(path):
        # cross96 cut events use ``seqs`` while the later VSP ledger retains
        # its source-manifest ``seqA``/``seqB`` schema.
        if "seqs" in record:
            first, second = record["seqs"][-2], record["seqs"][-1]
        else:
            first, second = record["seqA"], record["seqB"]
        output.add(canonical_pair(first, second))
    return output


def excluded_pairs(source: str) -> set[tuple[str, str]]:
    excluded = pairs_from_cut_events(FROZEN_ROOT / FROZEN_FILES[source])
    excluded |= {
        canonical_pair(record["seqA"], record["seqB"])
        for record in read_jsonl(FROZEN_180)
        if str(record["source"]) == source
    }
    excluded |= pairs_from_cut_events(TRAIN96_ROOT / f"{source}.jsonl")
    excluded |= pairs_from_cut_events(VSP_ROOT / "dev" / f"{source}.jsonl")
    excluded |= pairs_from_cut_events(VSP_ROOT / "confirm" / f"{source}.jsonl")
    return excluded


def decorate(record: dict[str, Any], source: str, index: int) -> dict[str, Any]:
    item = to_cut_event(record, source)
    item["source_manifest_index"] = int(index)
    item["pattern_id"] = (
        f"v14_ebr_train_{source}_{item['angle_bucket']}_{item['group']}_"
        f"{item['frames'][-1]}_{item['seqs'][-2].split('/')[-1]}_{item['seqs'][-1].split('/')[-1]}"
    )
    return item


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.per_source <= 0:
        raise ValueError("--per-source must be positive")
    metadata: dict[str, Any] = {
        "purpose": "new explicit-SE(3) residual training pairs, disjoint from B0 training/dev/confirm",
        "seed": args.seed, "per_source": args.per_source,
        "exclusion": ["cross96 train96 pairs", "frozen10 pairs", "frozen180 pairs", "VSP dev pairs", "VSP confirm pairs"],
        "sources": {},
    }
    all_records: list[dict[str, Any]] = []
    for source in SOURCES:
        excluded = excluded_pairs(source)
        candidates = ordered_candidates(source, args.seed, excluded)
        if len(candidates) < args.per_source:
            raise RuntimeError(f"{source}: only {len(candidates)} eligible events; require {args.per_source}")
        records = [decorate(record, source, int(record["source_manifest_index"])) for record in candidates[:args.per_source]]
        pairs = [canonical_pair(record["seqs"][-2], record["seqs"][-1]) for record in records]
        if set(pairs) & excluded:
            raise RuntimeError(f"{source}: pair-disjointness audit failed")
        if len({record["pattern_id"] for record in records}) != len(records):
            raise RuntimeError(f"{source}: duplicate pattern ID")
        write_jsonl(args.output_dir / "train" / f"{source}.jsonl", records)
        all_records.extend(records)
        metadata["sources"][source] = {
            "records": len(records), "unique_pairs": len(set(pairs)), "excluded_pair_count": len(excluded),
            "angle_buckets": dict(sorted(Counter(record["angle_bucket"] for record in records).items())),
            "pair_overlap_with_exclusion": 0,
        }
    write_jsonl(args.output_dir / "train_all.jsonl", all_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
