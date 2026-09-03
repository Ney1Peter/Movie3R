#!/usr/bin/env python3
"""Build exact final-case and viewpoint-stratified OnlineHMR manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
MOVIE3R = SCRIPT.parents[3]
WORKSPACE = MOVIE3R.parent
DEFAULT_OUTPUT = WORKSPACE / "data/OnlineHMR_work_v1/manifests"
KNOWN_EGOBODY_RUNTIME_SHA256 = (
    "8a5861bd3e4ee55dd1639c86526d21c96a73bb44fe07ff9848ef7b6b7645b02b"
)
SEED = 20260903


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for row in rows
    ).encode("utf-8")


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace differing manifest: {path}")
        return
    path.write_bytes(payload)


def paired_subset(
    runtime_path: Path,
    evaluator_path: Path,
    ordered_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = {str(row["case_id"]): row for row in read_jsonl(runtime_path)}
    evaluator = {str(row["case_id"]): row for row in read_jsonl(evaluator_path)}
    if len(runtime) != len(read_jsonl(runtime_path)):
        raise ValueError(f"duplicate runtime case ID in {runtime_path}")
    if len(evaluator) != len(read_jsonl(evaluator_path)):
        raise ValueError(f"duplicate evaluator case ID in {evaluator_path}")
    missing = [case_id for case_id in ordered_ids if case_id not in runtime or case_id not in evaluator]
    if missing:
        raise ValueError(f"missing {len(missing)} paired cases; examples={missing[:3]}")
    return [runtime[value] for value in ordered_ids], [evaluator[value] for value in ordered_ids]


def label(case_id: str) -> str:
    for value in ("small", "medium", "large", "extreme"):
        if f"_{value}_" in case_id:
            return value
    raise ValueError(f"case ID has no viewpoint stratum: {case_id}")


def choose(rows: list[dict[str, Any]], dataset: str, stratum: str, count: int = 1) -> list[str]:
    candidates = [row for row in rows if label(str(row["case_id"])) == stratum]
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{SEED}:{dataset}:{stratum}:{row['case_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    used_units: set[str] = set()
    for row in ranked:
        unit = str(row.get("recording", row.get("capture", row["case_id"])))
        if unit in used_units:
            continue
        selected.append(row)
        used_units.add(unit)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"cannot select {count} independent {dataset}/{stratum} cases")
    return [str(row["case_id"]) for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()

    egohuman_ids_path = (
        WORKSPACE / "ICLR-paper/bridge3r_iclr2027/private_audit/"
        "egohumans_cs100_formal_case_ids.txt"
    )
    egohuman_ids = [
        line.strip() for line in egohuman_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(egohuman_ids) != 90 or len(set(egohuman_ids)) != 90:
        raise ValueError("EgoHumans final protocol must contain 90 unique case IDs")
    eh_root = WORKSPACE / "data/EgoHuman_work_v19/external_predictions/trace_egohumans_v2/manifests"
    eh_runtime_source = eh_root / "egohumans_test.runtime.jsonl"
    eh_evaluator_source = eh_root / "egohumans_test.evaluator.jsonl"
    eh_runtime, eh_evaluator = paired_subset(
        eh_runtime_source, eh_evaluator_source, egohuman_ids
    )

    h4d_csv = (
        MOVIE3R / "output/v17_harmony4d/unified_half_translation_audit/"
        "paper/case_metrics.csv"
    )
    with h4d_csv.open(newline="", encoding="utf-8") as handle:
        h4d_metric_rows = list(csv.DictReader(handle))
    h4d_ids = sorted(
        {str(row["case_id"]) for row in h4d_metric_rows if row["method"] == "m0_strict_human3r"}
    )
    if len(h4d_ids) != 88:
        raise ValueError(f"Harmony4D final protocol must contain 88 cases, found {len(h4d_ids)}")
    h4d_root = (
        WORKSPACE / "data/Harmony4D_work_v17_full_test/external_predictions/trace/manifests"
    )
    h4d_runtime_source = h4d_root / "harmony4d_test.runtime.jsonl"
    h4d_evaluator_source = h4d_root / "harmony4d_test.evaluator.jsonl"
    h4d_runtime, h4d_evaluator = paired_subset(
        h4d_runtime_source, h4d_evaluator_source, h4d_ids
    )

    egobody_evaluations = MOVIE3R / "output/v20_egobody/formal/test/evaluations"
    egobody_runtime = []
    for path in sorted(egobody_evaluations.glob("*.evaluation.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        egobody_runtime.append(dict(payload["record_runtime_fields"]))
    egobody_runtime.sort(key=lambda row: str(row["case_id"]))
    if len(egobody_runtime) != 129:
        raise ValueError(f"EgoBody final protocol must contain 129 cases, found {len(egobody_runtime)}")

    outputs = {
        "egobody": {
            "runtime": ("egobody_cs150_test.runtime.jsonl", egobody_runtime),
            "evaluator": None,
            "expected_cases": 129,
            "source": str(egobody_evaluations),
        },
        "egohumans": {
            "runtime": ("egohumans_formal90.runtime.jsonl", eh_runtime),
            "evaluator": ("egohumans_formal90.evaluator.jsonl", eh_evaluator),
            "expected_cases": 90,
            "source": [str(eh_runtime_source), str(eh_evaluator_source), str(egohuman_ids_path)],
        },
        "harmony4d": {
            "runtime": ("harmony4d_formal88.runtime.jsonl", h4d_runtime),
            "evaluator": ("harmony4d_formal88.evaluator.jsonl", h4d_evaluator),
            "expected_cases": 88,
            "source": [str(h4d_runtime_source), str(h4d_evaluator_source), str(h4d_csv)],
        },
    }

    spec: dict[str, Any] = {
        "schema_version": "Bridge3R-OnlineHMR-three-dataset-protocol-v1",
        "selection_seed": SEED,
        "selection_depends_on_model_result": False,
        "runtime_gt_access": False,
        "datasets": {},
    }
    materialized: dict[str, list[dict[str, Any]]] = {}
    for dataset, entry in outputs.items():
        runtime_name, runtime_rows = entry["runtime"]
        runtime_path = output / runtime_name
        write_once(runtime_path, canonical_jsonl(runtime_rows))
        if len(runtime_rows) != int(entry["expected_cases"]):
            raise AssertionError(dataset)
        materialized[dataset] = runtime_rows
        dataset_spec: dict[str, Any] = {
            "case_count": len(runtime_rows),
            "runtime_manifest": str(runtime_path),
            "runtime_manifest_sha256": sha256(runtime_path),
            "source": entry["source"],
        }
        evaluator_entry = entry["evaluator"]
        if evaluator_entry is not None:
            evaluator_name, evaluator_rows = evaluator_entry
            evaluator_path = output / evaluator_name
            write_once(evaluator_path, canonical_jsonl(evaluator_rows))
            dataset_spec.update(
                evaluator_manifest=str(evaluator_path),
                evaluator_manifest_sha256=sha256(evaluator_path),
            )
        spec["datasets"][dataset] = dataset_spec

    if spec["datasets"]["egobody"]["runtime_manifest_sha256"] != KNOWN_EGOBODY_RUNTIME_SHA256:
        raise ValueError("reconstructed EgoBody runtime manifest differs from frozen historical hash")

    pilot_ids = {
        "egobody": (
            choose(materialized["egobody"], "egobody", "small")
            + choose(materialized["egobody"], "egobody", "medium")
            + choose(materialized["egobody"], "egobody", "extreme", count=2)
        ),
        "egohumans": sum(
            (choose(materialized["egohumans"], "egohumans", value) for value in ("small", "medium", "large", "extreme")),
            [],
        ),
        "harmony4d": sum(
            (choose(materialized["harmony4d"], "harmony4d", value) for value in ("small", "medium", "large", "extreme")),
            [],
        ),
    }
    pilot = {
        "schema_version": "Bridge3R-OnlineHMR-pilot-selection-v1",
        "selection_seed": SEED,
        "selection_rule": "minimum SHA-256(seed:dataset:stratum:case_id), with independent units for duplicate-stratum picks",
        "selection_depends_on_model_result": False,
        "cases": [
            {
                "dataset": dataset,
                "case_id": case_id,
                "angle_stratum": label(case_id),
            }
            for dataset in ("egobody", "egohumans", "harmony4d")
            for case_id in pilot_ids[dataset]
        ],
    }
    pilot_path = output / "onlinehmr_pilot12.json"
    write_once(
        pilot_path,
        (json.dumps(pilot, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    spec["pilot"] = {
        "manifest": str(pilot_path),
        "manifest_sha256": sha256(pilot_path),
        "case_count": len(pilot["cases"]),
    }
    spec_path = output / "protocol.json"
    write_once(
        spec_path,
        (json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    print(json.dumps({"protocol": str(spec_path), **spec}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
