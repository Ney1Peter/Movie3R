#!/usr/bin/env python3
"""Write a reproducible, pre-inference lock for one AIST++ CS150 run.

The lock intentionally records only provenance and immutable runtime/evaluator
manifest identities.  It does not read labels or report any score, and it
does not alter an already written lock.  Formal inference is started only
after this file has been successfully created and reviewed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from .protocol import atomic_json, canonical_json_digest, sha256_file
except ImportError:
    from protocol import atomic_json, canonical_json_digest, sha256_file  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402


SCHEMA = "Bridge3R-AIST-SinglePerson-CS150-pre-inference-lock-v1"
EXPECTED_RUNTIME_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("pilot", "test"), required=True)
    parser.add_argument("--devices", required=True, help="One to three physical CUDA indices, e.g. 1,2,3.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is not a nonempty object JSONL manifest")
    return rows


def git_state() -> dict[str, Any]:
    def query(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, check=True, capture_output=True).stdout.strip()
    return {"commit": query("rev-parse", "HEAD"), "status_porcelain": query("status", "--porcelain")}


def main() -> None:
    args = parse_args()
    devices = tuple(int(value.strip()) for value in args.devices.split(",") if value.strip())
    if not devices or len(devices) > 3 or len(set(devices)) != len(devices) or any(value < 0 for value in devices):
        raise ValueError("--devices must be one to three distinct nonnegative CUDA indices")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing lock: {args.output}")

    runtime, evaluator = read_jsonl(args.runtime_manifest.resolve()), read_jsonl(args.evaluator_manifest.resolve())
    if any(set(row) != EXPECTED_RUNTIME_KEYS for row in runtime):
        raise ValueError("Runtime manifest schema drifted")
    runtime_ids, evaluator_ids = [str(row["case_id"]) for row in runtime], [str(row.get("case_id")) for row in evaluator]
    if len(runtime_ids) != len(set(runtime_ids)) or set(runtime_ids) != set(evaluator_ids):
        raise ValueError("Runtime/evaluator case identities differ")
    if any(row["dataset"] != "AIST++" or row["protocol"] != "CS150" or row["role"] != args.role or int(row["fps"]) != 30 or int(row["num_frames"]) != 150 for row in runtime):
        raise ValueError("Runtime protocol contract drifted")
    if any(row.get("protocol") != "CS150" or row.get("role") != args.role for row in evaluator):
        raise ValueError("Evaluator protocol contract drifted")

    current, original = (path.resolve() for path in frozen.default_checkpoints())
    detector = Path(frozen.DETECTOR_PATH).resolve()
    paths = {
        "runner": Path(__file__).with_name("run_aist_case.py").resolve(),
        "protocol_runner": Path(__file__).with_name("run_protocol.py").resolve(),
        "evaluator": Path(__file__).with_name("evaluate_aist.py").resolve(),
        "aggregator": Path(__file__).with_name("aggregate_aist.py").resolve(),
        "current_checkpoint": current,
        "original_checkpoint": original,
        "causal_detector": detector,
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(missing)
    git = git_state()
    if git["status_porcelain"]:
        raise RuntimeError("Movie3R Git tree is not clean; commit or isolate changes before formal inference")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "role": args.role,
        "formal_case_count": len(runtime_ids),
        "runtime_manifest": str(args.runtime_manifest.resolve()),
        "runtime_manifest_sha256": sha256_file(args.runtime_manifest.resolve()),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()),
        "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest.resolve()),
        "devices": list(devices),
        "git": git,
        "artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "bridge3r_operating_point": {
            "name": "v19_ungated_translation_b050",
            "camera_alpha": 1.0,
            "boundary_kind": "translation",
            "boundary_blend": 0.5,
            "trigger": "CausalGRUShotDetector first positive only",
            "detector_miss_policy": "exact unmodified current parent",
        },
        "runtime_evaluator_separation": "GPU runner receives runtime rows only; labels, camera IDs, transition angles and cut indices are evaluator-only.",
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output), "role": args.role, "cases": len(runtime_ids), "content_sha256": payload["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
