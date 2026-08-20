#!/usr/bin/env python3
"""Index, CRC-deduplicate, and preregister EgoHuman.zip capture splits."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v19.egohumans.dataset import PROTOCOL_NAME, PROTOCOL_SEED, atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-crc",
        action="store_true",
        help="Read all 221 GiB now; otherwise each entry is CRC-checked when staged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with zipfile.ZipFile(args.outer) as archive:
        infos = [info for info in archive.infolist() if info.filename.endswith(".tar.gz")]
        bad = archive.testzip() if args.test_crc else None
    if args.test_crc and bad is not None:
        raise ValueError(f"Outer ZIP CRC failure: {bad}")
    unique = []
    duplicates = []
    seen: dict[tuple[int, int], str] = {}
    for info in sorted(infos, key=lambda value: value.filename):
        key = (int(info.file_size), int(info.CRC))
        if key in seen:
            duplicates.append(
                {
                    "entry": info.filename,
                    "duplicate_of": seen[key],
                    "file_size": int(info.file_size),
                    "crc32": f"{info.CRC:08x}",
                }
            )
            continue
        seen[key] = info.filename
        unique.append(info)
    by_action: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in unique:
        action = info.filename.split("/", 1)[0]
        by_action[action].append(info)
    entries = []
    for action, values in sorted(by_action.items()):
        ranked = sorted(values, key=lambda value: (value.file_size, value.filename))
        if len(ranked) < 3:
            raise ValueError(f"Action {action} has only {len(ranked)} unique captures")
        for rank, info in enumerate(ranked):
            split = "development" if rank == 0 else ("holdout" if rank == 1 else "test")
            entries.append(
                {
                    "entry": info.filename,
                    "action": action,
                    "split": split,
                    "size_rank_within_action": rank + 1,
                    "file_size": int(info.file_size),
                    "compressed_size_in_outer": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                }
            )
    payload = {
        "schema_version": "Movie3R-v19-EgoHumans-outer-index-v1",
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "outer": str(args.outer.resolve()),
        "outer_size_bytes": args.outer.stat().st_size,
        "zip_crc_test": "passed" if args.test_crc else "deferred_per_entry_during_staging",
        "raw_tar_entries": len(infos),
        "unique_tar_entries": len(unique),
        "duplicate_entries": duplicates,
        "split_rule": (
            "within each action, unique archives sorted by inner tar byte size then name; "
            "rank1 development, rank2 holdout, remaining test"
        ),
        "split_counts": {
            split: sum(row["split"] == split for row in entries)
            for split in ("development", "holdout", "test")
        },
        "action_counts": {
            action: sum(row["action"] == action for row in entries)
            for action in sorted(by_action)
        },
        "entries": sorted(entries, key=lambda row: (row["split"], row["action"], row["entry"])),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
