#!/usr/bin/env python3
"""Evaluator-only creation of compact EgoHumans GT caches.

This program is intentionally separate from the RGB-only inference workers.
For frozen Test it must be invoked only after prediction/transform sealing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.registration_baselines.prepare_inputs import value_sha256  # noqa: E402
from experiments.registration_baselines.run_egohumans_inference import (  # noqa: E402
    safe_cleanup,
    stage_capture,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v19.egohumans.evaluate_egohumans import load_gt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    runtime = read_rows(args.runtime_manifest.resolve())
    evaluator = read_rows(args.evaluator_manifest.resolve())
    runtime_by_case = {str(row["case_id"]): row for row in runtime}
    evaluator_by_case = {str(row["case_id"]): row for row in evaluator}
    if set(runtime_by_case) != set(evaluator_by_case):
        raise ValueError("runtime/evaluator case mismatch")
    for case_id, row in evaluator_by_case.items():
        expected = row.get("runtime_row_sha256")
        if expected is not None and str(expected) != value_sha256(runtime_by_case[case_id]):
            raise ValueError(f"runtime byte-contract mismatch: {case_id}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runtime:
        grouped[str(row["archive_entry"])].append(row)
    entries = sorted(grouped)
    selected = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard_index]
    topology = CommonTopology.load()
    cache_root = args.output_root / "work/gt_cache/egohumans" / args.split
    worker_root = args.output_root / "work/egohumans_evaluator_stage" / args.split / f"shard{args.shard_index:02d}"
    completed, reused, failures = [], [], []
    for entry_index, entry in enumerate(selected, start=1):
        pending = []
        for record in grouped[entry]:
            path = cache_root / f"{record['case_id']}.gt.npz"
            metadata = cache_root / f"{record['case_id']}.gt.json"
            if path.is_file() and metadata.is_file():
                reused.append(record["case_id"])
            else:
                pending.append(record)
        if not pending:
            print(f"[{entry_index}/{len(selected)}] reusable GT capture {entry}", flush=True)
            continue
        stage = archive = None
        try:
            stage, archive, archive_meta = stage_capture(args.outer.resolve(), entry, worker_root)
            for record in pending:
                case_id = str(record["case_id"])
                try:
                    # Formal Test deliberately omits camera names from the
                    # runtime manifest.  They are evaluator-only fields and
                    # may be merged only here, after the prediction seal has
                    # been written.  Development/Holdout rows happen to carry
                    # the same fields, but using the evaluator row uniformly
                    # keeps the isolation contract explicit.
                    evaluation_record = dict(record)
                    evaluation_record.update({
                        "pre_camera": evaluator_by_case[case_id]["pre_camera"],
                        "post_camera": evaluator_by_case[case_id]["post_camera"],
                    })
                    gt, identities = load_gt(evaluation_record, stage, topology)
                    path = cache_root / f"{case_id}.gt.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(path.suffix + ".partial")
                    with partial.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            **gt,
                            identities=np.asarray(identities, dtype="U64"),
                        )
                    os.replace(partial, path)
                    atomic_json(cache_root / f"{case_id}.gt.json", {
                        "schema_version": "Shot3R-registration-EgoHumans-GT-cache-v1",
                        "dataset": "egohumans", "split": args.split,
                        "case_id": case_id, "archive": archive_meta,
                        "frame_count": int(len(gt["frames"])),
                        "identities": identities,
                        "visible_gt_person_frames": int(gt["visible"].sum()),
                        "evaluator_only": True,
                    })
                    completed.append(case_id)
                    print(f"[{entry_index}/{len(selected)}] cached GT {case_id}", flush=True)
                except Exception as exc:
                    failure = {
                        "case_id": case_id, "entry": entry,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"GT CASE FAILED {failure}", flush=True)
        except Exception as exc:
            for record in pending:
                failures.append({"case_id": record["case_id"], "entry": entry, "error": f"{type(exc).__name__}: {exc}"})
            print(f"GT CAPTURE FAILED {entry}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if stage is not None:
                safe_cleanup(stage, worker_root / "staging")
            if archive is not None and archive.exists():
                archive.unlink()
    ledger = args.output_root / "logs" / f"gt_egohumans_{args.split}_shard{args.shard_index:02d}.json"
    atomic_json(ledger, {
        "schema_version": "Shot3R-registration-EgoHumans-GT-ledger-v1",
        "split": args.split, "shard_index": args.shard_index,
        "num_shards": args.num_shards, "completed": completed,
        "reused": reused, "failures": failures,
        "evaluator_manifest": str(args.evaluator_manifest.resolve()),
    })
    if failures:
        raise SystemExit(f"{len(failures)} GT cases failed; see {ledger}")


if __name__ == "__main__":
    main()
