#!/usr/bin/env python3
"""Build pair-disjoint development/confirmation manifests for VSP research.

The cross96 checkpoint already trained on 96 events/source and the original
frozen10/frozen180 were used for prior checkpoint evaluation.  This builder
selects two new event pools from the same AABB source manifests, with camera
pairs disjoint from *all* of those pools and from each other.  The first pool
is for selecting a GT-free VSP gate; the second is untouched confirmation.

It writes only repository-local manifest files.  No model, GPU, or data cache
is touched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
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
    read_jsonl,
)


DEFAULT_OUTPUT = REPO_ROOT / "config/manifests/v14_vsp_pair_disjoint_20260802"
TRAIN96_ROOT = Path(__file__).resolve().parent / "manifests/train96ps"
SOURCES = tuple(SOURCE_MANIFESTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-source", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def rank(record: dict[str, Any], source: str, seed: int) -> str:
    value = "|".join((
        str(seed), source, str(record.get("group", "")), str(record["seqA"]),
        str(record["seqB"]), str(record["start_frame"]),
    ))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def excluded_pairs(source: str) -> set[tuple[str, str]]:
    output = {
        canonical_pair(record["seqs"][-2], record["seqs"][-1])
        for record in read_jsonl(FROZEN_ROOT / FROZEN_FILES[source])
    }
    output |= {
        canonical_pair(record["seqA"], record["seqB"])
        for record in read_jsonl(FROZEN_180)
        if str(record["source"]) == source
    }
    output |= {
        canonical_pair(record["seqs"][-2], record["seqs"][-1])
        for record in read_jsonl(TRAIN96_ROOT / f"{source}.jsonl")
    }
    return output


def decorate(record: dict[str, Any], source: str, index: int, split: str) -> dict[str, Any]:
    item = dict(record)
    item["source"] = source
    item["source_manifest_index"] = int(index)
    item["pattern_id"] = (
        f"v14_vsp_{split}_{source}_{item.get('angle_bucket', 'unknown')}_"
        f"{item.get('group', 'none')}_{item['start_frame']}_"
        f"{str(item['seqA']).split('/')[-1]}_{str(item['seqB']).split('/')[-1]}"
    )
    return item


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.per_source <= 0:
        raise ValueError("--per-source must be positive")
    output_meta: dict[str, Any] = {
        "purpose": "VSP development/confirmation with camera-pair disjointness",
        "seed": args.seed,
        "per_source": args.per_source,
        "exclusion": ["cross96 train pairs", "frozen10 pairs", "frozen180 pairs", "dev/confirm mutual pairs"],
        "sources": {},
    }
    all_dev, all_confirm = [], []
    for source in SOURCES:
        excluded = excluded_pairs(source)
        raw = read_jsonl(SOURCE_MANIFESTS[source])
        candidates: list[tuple[int, dict[str, Any]]] = []
        seen_events = set()
        for index, record in enumerate(raw):
            pair = canonical_pair(record["seqA"], record["seqB"])
            event = (pair, int(record["start_frame"]))
            if pair in excluded or event in seen_events or int(record["start_frame"]) < 1:
                continue
            seen_events.add(event)
            candidates.append((index, record))
        candidates.sort(key=lambda pair: rank(pair[1], source, args.seed))
        # Pair diversity first: a confirmation split must not reuse a pair from
        # dev even at another timestamp.
        unique_pairs, repeated_pairs, seen_pairs = [], [], set()
        for candidate in candidates:
            pair = canonical_pair(candidate[1]["seqA"], candidate[1]["seqB"])
            (repeated_pairs if pair in seen_pairs else unique_pairs).append(candidate)
            seen_pairs.add(pair)
        candidates = unique_pairs + repeated_pairs
        dev, confirm, dev_pairs, confirm_pairs = [], [], set(), set()
        for index, record in candidates:
            pair = canonical_pair(record["seqA"], record["seqB"])
            if len(dev) < args.per_source and pair not in dev_pairs:
                dev.append(decorate(record, source, index, "dev"))
                dev_pairs.add(pair)
            elif len(confirm) < args.per_source and pair not in dev_pairs and pair not in confirm_pairs:
                confirm.append(decorate(record, source, index, "confirm"))
                confirm_pairs.add(pair)
            if len(dev) == args.per_source and len(confirm) == args.per_source:
                break
        if len(dev) != args.per_source or len(confirm) != args.per_source:
            raise RuntimeError(f"{source}: insufficient pair-disjoint candidates ({len(dev)}, {len(confirm)})")
        if (dev_pairs | confirm_pairs) & excluded or dev_pairs & confirm_pairs:
            raise RuntimeError(f"{source}: pair-disjointness assertion failed")
        write_jsonl(args.output_dir / "dev" / f"{source}.jsonl", dev)
        write_jsonl(args.output_dir / "confirm" / f"{source}.jsonl", confirm)
        all_dev.extend(dev)
        all_confirm.extend(confirm)
        output_meta["sources"][source] = {
            "dev_records": len(dev), "confirm_records": len(confirm),
            "dev_pairs": len(dev_pairs), "confirm_pairs": len(confirm_pairs),
            "excluded_pair_count": len(excluded),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "dev_all.jsonl", all_dev)
    write_jsonl(args.output_dir / "confirm_all.jsonl", all_confirm)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(output_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
