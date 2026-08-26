#!/usr/bin/env python3
"""Parallel, resumable materialisation driver for frozen AIST++ source clips.

``derive_sequences.py`` is deliberately single-source atomic: its three
protocol videos, labels and case artifact either all appear, or the source is
left for inspection.  This driver invokes that safe unit on disjoint frozen
sources in a bounded worker pool.  It never alters source selection and ends
with one no-force finalisation pass that writes the immutable role manifests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .protocol import DEFAULT_BUNDLE_ROOT, DEFAULT_DERIVED_ROOT, atomic_json, load_frozen_sources, verify_input_manifest_freeze
except ImportError:
    from protocol import DEFAULT_BUNDLE_ROOT, DEFAULT_DERIVED_ROOT, atomic_json, load_frozen_sources, verify_input_manifest_freeze  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-parallel-derivation-v1"
DERIVER = Path(__file__).with_name("derive_sequences.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--roles", default="test")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--encoder", choices=("x264", "nvenc"), default="x264")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def run_source(args: argparse.Namespace, source_id: str) -> dict[str, Any]:
    command = [
        sys.executable, str(DERIVER), "--bundle-root", str(args.bundle_root.resolve()),
        "--derived-root", str(args.derived_root.resolve()), "--roles", args.roles,
        "--encoder", args.encoder, "--gpu", str(args.gpu), "--source-id", source_id,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "source_id": source_id, "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr,
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    roles = tuple(token.strip() for token in args.roles.split(",") if token.strip())
    if not roles or set(roles) - {"pilot", "test"}:
        raise SystemExit("--roles must be a nonempty subset of pilot,test")
    input_hashes = verify_input_manifest_freeze(args.bundle_root.resolve())
    sources = load_frozen_sources(args.bundle_root.resolve(), roles)
    worker_rows: list[dict[str, Any]] = []
    if not args.finalize_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_source, args, str(source["source_id"])): source for source in sources}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                worker_rows.append(row)
                print(f"derivation workers: {index}/{len(sources)} {row['source_id']} rc={row['returncode']}", flush=True)
    errors = [row for row in worker_rows if row["returncode"]]
    if errors:
        report = {
            "schema_version": SCHEMA, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_manifest_sha256": input_hashes, "roles": list(roles), "workers": args.workers,
            "status": "worker_errors", "worker_rows": worker_rows,
        }
        output = args.derived_root.resolve() / "derivation_runs" / f"parallel_derive_{'_'.join(sorted(roles))}.json"
        atomic_json(output, report)
        raise SystemExit(f"{len(errors)} source derivations failed; details: {output}")
    # This invocation is intentionally no-force and has no source filter: it
    # verifies every source artifact once more and is the only action that
    # writes complete runtime/evaluator manifests for the role.
    finalize = [
        sys.executable, str(DERIVER), "--bundle-root", str(args.bundle_root.resolve()),
        "--derived-root", str(args.derived_root.resolve()), "--roles", args.roles,
        "--encoder", args.encoder, "--gpu", str(args.gpu),
    ]
    completed = subprocess.run(finalize, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report = {
        "schema_version": SCHEMA, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes, "roles": list(roles), "workers": args.workers,
        "status": "ok" if completed.returncode == 0 else "finalization_error",
        "worker_rows": worker_rows,
        "finalization": {"command": finalize, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
    }
    output = args.derived_root.resolve() / "derivation_runs" / f"parallel_derive_{'_'.join(sorted(roles))}.json"
    atomic_json(output, report)
    if completed.returncode:
        raise SystemExit(f"Finalization failed; details: {output}")
    print(json.dumps({"status": "ok", "report": str(output), "source_count": len(sources)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
