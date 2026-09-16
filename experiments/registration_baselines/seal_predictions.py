#!/usr/bin/env python3
"""Seal a complete frozen-Test prediction set before any GT evaluator runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHODS = ("r0", "r1", "r2", "r3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("egobody", "egohumans"), required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    manifest = args.runtime_manifest.resolve()
    config = args.config.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(args.expected_cases) or any(str(row.get("split")) != "test" for row in rows):
        raise ValueError("runtime manifest is not the expected complete Test split")
    files: list[Path] = [manifest, config]
    cases = []
    for row in rows:
        case_id = str(row["case_id"])
        required = [
            root / "predictions" / args.dataset / "test" / f"{case_id}.r0_scene.npz",
            root / "logs/inference" / args.dataset / "test" / f"{case_id}.runtime.json",
            root / "predictions" / args.dataset / "test" / f"{case_id}.npz",
            root / "predictions" / args.dataset / "test" / f"{case_id}.runtime.json",
            *[
                root / "transforms" / args.dataset / "test" / case_id / f"{method}.json"
                for method in METHODS
            ],
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete Test prediction {case_id}: {missing}")
        runtime = json.loads(required[3].read_text(encoding="utf-8"))
        if runtime.get("case_id") != case_id or runtime.get("record") != row:
            raise ValueError(f"prediction runtime does not bind exact row: {case_id}")
        statuses = {}
        for method, path in zip(METHODS, required[4:]):
            transform = json.loads(path.read_text(encoding="utf-8"))
            if transform.get("case_id") != case_id or transform.get("method_id") != method:
                raise ValueError(f"transform identity mismatch: {path}")
            if transform.get("gt_used_for_registration") is not False:
                raise ValueError(f"GT access declaration is not false: {path}")
            statuses[method] = transform.get("status")
        files.extend(required)
        cases.append({"case_id": case_id, "statuses": statuses})
    unique = sorted(set(path.resolve() for path in files), key=lambda path: str(path))
    entries = []
    for index, path in enumerate(unique, start=1):
        value = sha256(path)
        entries.append({
            "path": str(path), "relative_path": str(path.relative_to(root)) if root in path.parents else str(path),
            "size_bytes": path.stat().st_size, "sha256": value,
        })
        print(f"[{index}/{len(unique)}] sealed {path.name}", flush=True)
    payload: dict[str, Any] = {
        "schema_version": "Shot3R-traditional-registration-Test-prediction-seal-v1",
        "dataset": args.dataset, "split": "test", "case_count": len(cases),
        "expected_case_count": int(args.expected_cases), "cases": cases,
        "runtime_manifest": str(manifest), "runtime_manifest_sha256": sha256(manifest),
        "frozen_config": str(config), "frozen_config_sha256": sha256(config),
        "sealed_before_evaluator": True, "evaluator_manifest_opened": False,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(), "entries": entries,
    }
    json_path = root / "provenance" / f"{args.dataset}_test_prediction_seal.json"
    lines_path = root / "provenance" / f"{args.dataset}_test_prediction_seal.sha256"
    atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    atomic_text(lines_path, "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in entries))
    print(json.dumps({"seal": str(json_path), "cases": len(cases), "files": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
