#!/usr/bin/env python3
"""Merge V12 cache shards and create source/group-disjoint LOSO splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "output" / "v12_gated_first_write" / "teacher_cache_loso_mvhuman200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--test_source", default="mvhuman200")
    return parser.parse_args()


def validation_group(source: str, group: str) -> bool:
    digest = hashlib.sha1(f"{source}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 == 0


def loaded_row(row: dict) -> dict:
    pair = torch.load(row["path"], map_location="cpu", weights_only=False)
    metadata = pair["metadata"]
    labels = pair["labels"]
    return {"path": row["path"], **metadata, **labels}


def main() -> None:
    args = parse_args()
    indices = sorted(args.cache_dir.glob("index_real_*_of_*.json"))
    indices += sorted(args.cache_dir.glob("index_pseudo_*_of_*.json"))
    if not indices:
        raise FileNotFoundError(args.cache_dir)
    rows = []
    for path in indices:
        rows.extend(loaded_row(row) for row in json.loads(path.read_text(encoding="utf-8")))
    names = [row["case_name"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V12 cache cases")
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        if row["kind"] == "real" and row["source"] == args.test_source:
            split = "test"
        elif row["kind"] == "pseudo":
            split = "validation" if row["group"] == "clip03" else "train"
        else:
            split = "validation" if validation_group(row["source"], row["group"]) else "train"
        row["split"] = split
        splits[split].append(row)
    summary = {
        "test_source": args.test_source,
        "case_count": len(rows),
        "splits": {},
    }
    for split, items in splits.items():
        gates = np.asarray([float(row["gate_target"]) for row in items], dtype=np.float64)
        summary["splits"][split] = {
            "count": len(items),
            "real": sum(row["kind"] == "real" for row in items),
            "pseudo": sum(row["kind"] == "pseudo" for row in items),
            "sources": {source: sum(row["source"] == source for row in items) for source in sorted(set(row["source"] for row in items))},
            "mean_gate_target": float(gates.mean()) if len(gates) else None,
            "positive_gate_rate": float(np.mean(gates > 0.05)) if len(gates) else None,
        }
        (args.cache_dir / f"index_{split}.json").write_text(
            json.dumps(items, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
            encoding="utf-8",
        )
    (args.cache_dir / "index_all.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.cache_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
