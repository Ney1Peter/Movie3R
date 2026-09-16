#!/usr/bin/env python3
"""Prepare frozen, GT-isolated manifests and a protocol/provenance lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[3]
MOVIE3R = WORKSPACE / "Movie3R"
DEFAULT_OUTPUT = MOVIE3R / "output/shot3r_registration_baselines_v1_20260915"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(value)
    os.replace(partial, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_bytes(path, ("".join(canonical(row) + "\n" for row in rows)).encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_egobody(output: Path) -> dict[str, Any]:
    source = output / "work/manifests/egobody"
    target = output / "work/frozen_manifests/egobody"
    target.mkdir(parents=True, exist_ok=True)
    details = {}
    for split in ("development", "holdout", "test"):
        runtime = read_jsonl(source / f"egobody_cs150_{split}.runtime.jsonl")
        evaluator = read_jsonl(source / f"egobody_cs150_{split}.evaluator.jsonl")
        if split == "holdout":
            historical = {
                path.name.removesuffix(".evaluation.json")
                for path in (MOVIE3R / "output/v20_egobody/formal/holdout/evaluations").glob("*.evaluation.json")
            }
            runtime = [row for row in runtime if str(row["case_id"]) in historical]
            evaluator = [row for row in evaluator if str(row["case_id"]) in historical]
        if split == "test":
            # Use the byte-frozen pair named in the execution protocol.
            runtime_source = WORKSPACE / "data/OnlineHMR_work_v1/manifests/egobody_cs150_test.runtime.jsonl"
            evaluator_source = WORKSPACE / "data/OnlineHMR_work_v1/work_egobody/frozen_manifests/egobody_cs150_test.evaluator.jsonl"
            runtime = read_jsonl(runtime_source)
            evaluator = read_jsonl(evaluator_source)
        runtime.sort(key=lambda row: str(row["case_id"]))
        evaluator.sort(key=lambda row: str(row["case_id"]))
        if [row["case_id"] for row in runtime] != [row["case_id"] for row in evaluator]:
            raise ValueError(f"EgoBody {split} runtime/evaluator mismatch")
        runtime_path = target / f"egobody_cs150_{split}.runtime.jsonl"
        evaluator_path = target / f"egobody_cs150_{split}.evaluator.jsonl"
        write_jsonl(runtime_path, runtime)
        write_jsonl(evaluator_path, evaluator)
        details[split] = {
            "cases": len(runtime),
            "recordings": len({row["recording"] for row in runtime}),
            "runtime": str(runtime_path.resolve()),
            "runtime_sha256": file_sha256(runtime_path),
            "evaluator": str(evaluator_path.resolve()),
            "evaluator_sha256": file_sha256(evaluator_path),
        }
    return details


def runtime_egohumans(row: dict[str, Any]) -> dict[str, Any]:
    forbidden = {
        key for key in row
        if str(key).endswith("_evaluator_only") or str(key).startswith("gt_")
    }
    runtime = {key: value for key, value in row.items() if key not in forbidden}
    runtime.pop("angle_stratum", None)
    runtime["runtime_gt_access"] = False
    runtime["selection_depends_on_model_result"] = False
    capture = str(runtime["capture_relative"])
    members = [
        f"{capture}/exo/{runtime['pre_camera']}/images/{int(frame):05d}.jpg"
        for frame in runtime["pre_frame_numbers"]
    ] + [
        f"{capture}/exo/{runtime['post_camera']}/images/{int(frame):05d}.jpg"
        for frame in runtime["post_frame_numbers"]
    ]
    runtime["image_members"] = members
    runtime["image_member_layout"] = "<capture_relative>/exo/<rgb_stream>/images/<frame:05d>.jpg"
    runtime["schema_version"] = "Shot3R-registration-EgoHumans-runtime-row-v1"
    return runtime


def prepare_egohumans(output: Path) -> dict[str, Any]:
    target = output / "work/frozen_manifests/egohumans"
    target.mkdir(parents=True, exist_ok=True)
    details = {}
    for split in ("development", "holdout"):
        source_root = MOVIE3R / f"output/v19_egohumans/{split}/captures"
        full_rows = []
        for manifest in sorted(source_root.glob("*/manifest.jsonl")):
            full_rows.extend(read_jsonl(manifest))
        full_rows.sort(key=lambda row: str(row["case_id"]))
        runtime = [runtime_egohumans(row) for row in full_rows]
        evaluator = []
        for source_row, runtime_row in zip(full_rows, runtime):
            row = dict(source_row)
            row["runtime_row_sha256"] = value_sha256(runtime_row)
            row["schema_version"] = "Shot3R-registration-EgoHumans-evaluator-row-v1"
            evaluator.append(row)
        runtime_path = target / f"egohumans_{split}.runtime.jsonl"
        evaluator_path = target / f"egohumans_{split}.evaluator.jsonl"
        write_jsonl(runtime_path, runtime)
        write_jsonl(evaluator_path, evaluator)
        details[split] = {
            "cases": len(runtime),
            "captures": len({row["archive_entry"] for row in runtime}),
            "runtime": str(runtime_path.resolve()),
            "runtime_sha256": file_sha256(runtime_path),
            "evaluator": str(evaluator_path.resolve()),
            "evaluator_sha256": file_sha256(evaluator_path),
        }
    test_runtime = WORKSPACE / "data/OnlineHMR_work_v1/manifests/egohumans_formal90.runtime.jsonl"
    test_evaluator = WORKSPACE / "data/OnlineHMR_work_v1/manifests/egohumans_formal90.evaluator.jsonl"
    runtime_rows, evaluator_rows = read_jsonl(test_runtime), read_jsonl(test_evaluator)
    if [row["case_id"] for row in runtime_rows] != [row["case_id"] for row in evaluator_rows]:
        raise ValueError("EgoHumans formal90 runtime/evaluator order mismatch")
    runtime_path = target / "egohumans_test.runtime.jsonl"
    evaluator_path = target / "egohumans_test.evaluator.jsonl"
    write_jsonl(runtime_path, runtime_rows)
    write_jsonl(evaluator_path, evaluator_rows)
    details["test"] = {
        "cases": len(runtime_rows),
        "captures": len({row["archive_entry"] for row in runtime_rows}),
        "runtime": str(runtime_path.resolve()),
        "runtime_sha256": file_sha256(runtime_path),
        "evaluator": str(evaluator_path.resolve()),
        "evaluator_sha256": file_sha256(evaluator_path),
    }
    return details


def qualitative_lock(output: Path, datasets: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for dataset, prefix in (("egobody", "egobody_cs150"), ("egohumans", "egohumans")):
        path = output / f"work/frozen_manifests/{dataset}/{prefix}_test.evaluator.jsonl"
        rows = read_jsonl(path)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            # EgoBody stores the evaluator-only field with an explicit suffix,
            # whereas the frozen EgoHumans formal-90 manifest uses the already
            # isolated ``angle_stratum`` spelling.  Accept both without opening
            # any prediction or metric file; qualitative cases remain the first
            # two case IDs in each pre-existing stratum.
            stratum = str(
                row.get(
                    "angle_stratum_evaluator_only",
                    row.get("angle_stratum", "unknown"),
                )
            )
            groups.setdefault(stratum, []).append(row)
        picks = []
        for stratum in sorted(groups):
            for row in sorted(groups[stratum], key=lambda item: str(item["case_id"]))[:2]:
                picks.append({"case_id": row["case_id"], "angle_stratum": stratum})
        selected[dataset] = picks
    payload = {
        "schema_version": "Shot3R-registration-qualitative-preselection-v1",
        "locked_before_registration_test_metrics": True,
        "selection_rule": "lexicographically first two fixed Test cases in each pre-existing angle stratum; independent of method result",
        "datasets": selected,
    }
    path = output / "config/qualitative_cases_pre_test.json"
    atomic_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    egobody = prepare_egobody(output)
    egohumans = prepare_egohumans(output)
    datasets = {"egobody": egobody, "egohumans": egohumans}
    qualitative = qualitative_lock(output, datasets)
    checkpoint = MOVIE3R / "src/human3r_896L.pth"
    checkpoint_hash = file_sha256(checkpoint)
    expected_checkpoint = "1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377"
    if checkpoint_hash != expected_checkpoint:
        raise ValueError(f"checkpoint SHA-256 mismatch: {checkpoint_hash}")
    expected_test = {
        "egobody": {
            "runtime": "8a5861bd3e4ee55dd1639c86526d21c96a73bb44fe07ff9848ef7b6b7645b02b",
            "evaluator": "87144f01a8dc7b0630b1e7e9613a8b11904847a97c438e9ac9693b3453ac534f",
        },
        "egohumans": {
            "runtime": "dccbad5a4b4c771bc5637fca0f2b44bc7ce2ac07fc84e2d827d33f3a1f67823b",
            "evaluator": "a65a8e5b9955483fd60e1308080fcb68488239cbf4d54a328514ca1c37156fdf",
        },
    }
    for dataset, expected in expected_test.items():
        for kind in ("runtime", "evaluator"):
            observed = datasets[dataset]["test"][f"{kind}_sha256"]
            if observed != expected[kind]:
                raise ValueError(f"{dataset} Test {kind} SHA mismatch: {observed}")
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=WORKSPACE, text=True,
        capture_output=True, check=True,
    ).stdout
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    atomic_bytes(output / "provenance/git_status.txt", git_status.encode("utf-8"))
    lock = {
        "schema_version": "Shot3R-traditional-registration-protocol-lock-v1",
        "protocol_document": str((WORKSPACE / "Shot3R_传统配准基线实验执行协议_20260915.md").resolve()),
        "checkpoint": {"path": str(checkpoint.resolve()), "size_bytes": checkpoint.stat().st_size, "sha256": checkpoint_hash},
        "datasets": datasets,
        "raw_archives": {
            name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "integrity_policy": "size/ZIP CRC during selected extraction; no repeated whole-archive hash"}
            for name, path in {
                "egobody": WORKSPACE / "data/EgoBody.zip",
                "egohumans": WORKSPACE / "data/EgoHuman.zip",
            }.items()
        },
        "split_policy": {
            "egobody": "official train->Development, historical 48-case val Holdout gate, official Test-129",
            "egohumans": "historical Development-28/Holdout-24, frozen formal Test-90",
        },
        "test_gt_isolation": True,
        "test_parameter_tuning": False,
        "fixed_failure_fallback": "R0 identity transform",
        "gpu_limit": 3,
        "allowed_devices": ["cuda:5", "cuda:6", "cuda:7"],
        "random_seed": 20260915,
        "git_commit_at_lock": git_commit,
        "qualitative_preselection": qualitative,
    }
    atomic_json(output / "protocol_lock.json", lock)
    input_rows = [f"{checkpoint_hash}  {checkpoint.resolve()}"]
    for dataset in datasets.values():
        for split in dataset.values():
            input_rows.extend((
                f"{split['runtime_sha256']}  {split['runtime']}",
                f"{split['evaluator_sha256']}  {split['evaluator']}",
            ))
    atomic_bytes(output / "provenance/input_sha256.txt", ("\n".join(input_rows) + "\n").encode("utf-8"))
    atomic_bytes(output / "environment.md", (
        "# Environment lock\n\n"
        f"- Python: `{sys.version.split()[0]}`\n"
        f"- Platform: `{platform.platform()}`\n"
        "- Inference environment: `Movie3R/.venv`\n"
        "- Open3D: `0.19.0`, isolated under the result directory `work/vendor`\n"
        "- Allowed GPUs: `cuda:5,cuda:6,cuda:7` (maximum three devices)\n"
        "- Seed: `20260915`\n"
    ).encode("utf-8"))
    print(json.dumps({"protocol_lock": str((output / 'protocol_lock.json').resolve()), "datasets": datasets}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
