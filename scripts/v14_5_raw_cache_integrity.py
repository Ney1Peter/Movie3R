#!/usr/bin/env python3
"""Prepare and compare the V14.5 raw-RGB cache-integrity rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V14_4 = (
    ROOT
    / "output/v14_4_unified_similarity_reanchoring/full180_final/"
    "v14_4_unified_similarity_reanchoring.json"
)
DEFAULT_OLD_V10 = (
    ROOT
    / "output/v10_candidate_selection/oracle_gt_4source/"
    "oracle_candidate_selection_metrics.json"
)
DEFAULT_RERUN = ROOT / "output/v14_5_final_audit/raw_rgb_rerun"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "compare"), required=True)
    parser.add_argument("--v14_4_report", type=Path, default=DEFAULT_V14_4)
    parser.add_argument("--old_v10_report", type=Path, default=DEFAULT_OLD_V10)
    parser.add_argument("--rerun_root", type=Path, default=DEFAULT_RERUN)
    parser.add_argument(
        "--rerun_report",
        type=Path,
        default=None,
        help="Optional full-context V10 report; selection still comes from rerun_root.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def select_cases(rows: list[dict]) -> list[dict]:
    selected: dict[str, tuple[dict, set[str]]] = {}

    def add(row: dict, tag: str) -> None:
        name = str(row["case_name"])
        if name not in selected:
            selected[name] = (row, set())
        selected[name][1].add(tag)

    for source in sorted({row["source"] for row in rows}):
        group = [row for row in rows if row["source"] == source]
        add(
            min(
                group,
                key=lambda row: row["methods"][
                    "v11_4_uniform_similarity_conditional_vggt"
                ]["camera"]["translation_m"],
            ),
            "source_best_v11_4_camera",
        )
        add(
            max(
                group,
                key=lambda row: row["methods"][
                    "v11_4_uniform_similarity_conditional_vggt"
                ]["camera"]["translation_m"],
            ),
            "source_worst_v11_4_camera",
        )

    triggered = [row for row in rows if row["conditional_vggt_triggered"]]
    if triggered:
        add(
            max(
                triggered,
                key=lambda row: row["methods"][
                    "v11_4_uniform_similarity_conditional_vggt"
                ]["camera"]["rotation_deg"],
            ),
            "conditional_vggt_triggered",
        )
    add(
        max(
            rows,
            key=lambda row: abs(
                row["scales"]["v11_4_post"] / row["scales"]["common_pre"] - 1.0
            ),
        ),
        "largest_shot_scale_correction",
    )

    # Preserve two cases per source while forcing the global diagnostic cases in.
    by_source: dict[str, list[tuple[dict, set[str]]]] = defaultdict(list)
    for row, tags in selected.values():
        by_source[row["source"]].append((row, tags))
    output = []
    for source in sorted({row["source"] for row in rows}):
        group = by_source[source]
        if len(group) > 2:
            group.sort(
                key=lambda value: (
                    not bool(
                        value[1]
                        & {"conditional_vggt_triggered", "largest_shot_scale_correction"}
                    ),
                    -len(value[1]),
                    value[0]["case_name"],
                )
            )
            group = group[:2]
        while len(group) < 2:
            pool = [
                row
                for row in rows
                if row["source"] == source
                and all(row["case_name"] != value[0]["case_name"] for value in group)
            ]
            row = min(pool, key=lambda value: value["case_name"])
            group.append((row, {"deterministic_fill"}))
        output.extend(group)
    return [
        {
            "case_name": row["case_name"],
            "source": row["source"],
            "record": row["record"],
            "selection_tags": sorted(tags),
            "conditional_vggt_triggered": bool(row["conditional_vggt_triggered"]),
            "pre_scale": float(row["scales"]["common_pre"]),
            "post_scale": float(row["scales"]["v11_4_post"]),
            "v11_4_camera_m": float(
                row["methods"]["v11_4_uniform_similarity_conditional_vggt"]["camera"][
                    "translation_m"
                ]
            ),
        }
        for row, tags in sorted(output, key=lambda value: value[0]["case_name"])
    ]


def prepare(args: argparse.Namespace) -> None:
    payload = json.loads(args.v14_4_report.read_text(encoding="utf-8"))
    selected = select_cases(payload["cases"])
    records_dir = args.rerun_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {}
    for source in sorted({row["source"] for row in selected}):
        path = records_dir / f"{source}.jsonl"
        write_jsonl(path, [row["record"] for row in selected if row["source"] == source])
        source_paths[source] = str(path)
    (records_dir / "manifest_map.json").write_text(
        json.dumps({"source_manifests": source_paths}, indent=2) + "\n",
        encoding="utf-8",
    )
    (records_dir / "selection.json").write_text(
        json.dumps(
            {
                "case_count": len(selected),
                "selection_frozen_before_rerun": True,
                "by_source": dict(Counter(row["source"] for row in selected)),
                "cases": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((records_dir / "selection.json").read_text(encoding="utf-8"))


def load_numeric(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npy":
        return {"array": np.asarray(np.load(path))}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {
                key: np.asarray(payload[key])
                for key in payload.files
                if np.asarray(payload[key]).dtype.kind in "biufc"
            }
    return {}


def compare_trees(old: Path, new: Path) -> dict:
    suffixes = {".npy", ".npz"}
    old_files = {
        path.relative_to(old): path
        for path in old.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }
    new_files = {
        path.relative_to(new): path
        for path in new.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }
    common = sorted(set(old_files) & set(new_files))
    differences = {}
    global_max = 0.0
    shape_mismatch = []
    for relative in common:
        first = load_numeric(old_files[relative])
        second = load_numeric(new_files[relative])
        for key in sorted(set(first) & set(second)):
            if first[key].shape != second[key].shape:
                shape_mismatch.append(
                    {
                        "path": str(relative),
                        "key": key,
                        "old": list(first[key].shape),
                        "new": list(second[key].shape),
                    }
                )
                continue
            finite = np.isfinite(first[key]) & np.isfinite(second[key])
            maximum = (
                float(np.max(np.abs(first[key][finite] - second[key][finite])))
                if np.any(finite)
                else 0.0
            )
            differences[f"{relative}:{key}"] = maximum
            global_max = max(global_max, maximum)
    return {
        "old_numeric_file_count": len(old_files),
        "new_numeric_file_count": len(new_files),
        "common_numeric_file_count": len(common),
        "missing_in_rerun": sorted(str(path) for path in set(old_files) - set(new_files)),
        "extra_in_rerun": sorted(str(path) for path in set(new_files) - set(old_files)),
        "shape_mismatch": shape_mismatch,
        "maximum_absolute_difference": global_max,
        "per_array_maximum": differences,
    }


def compare(args: argparse.Namespace) -> None:
    selection = json.loads(
        (args.rerun_root / "records/selection.json").read_text(encoding="utf-8")
    )
    old_payload = json.loads(args.old_v10_report.read_text(encoding="utf-8"))
    old = {row["case_name"]: row for row in old_payload["cases"]}
    rerun = {}
    if args.rerun_report is not None:
        payload = json.loads(args.rerun_report.read_text(encoding="utf-8"))
        rerun.update({row["case_name"]: row for row in payload["cases"]})
    else:
        for source in selection["by_source"]:
            path = args.rerun_root / "v10" / source / "oracle_candidate_selection_metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            rerun.update({row["case_name"]: row for row in payload["cases"]})
    cases = []
    for selected in selection["cases"]:
        name = selected["case_name"]
        first = old[name]
        second = rerun[name]
        tree = compare_trees(
            Path(first["paths"]["human3r_local_reset"]),
            Path(second["paths"]["human3r_local_reset"]),
        )
        fixed_old = first["fixed_explicit"]
        fixed_new = second["fixed_explicit"]
        tree["fixed_transform_maximum"] = float(
            np.max(
                np.abs(
                    np.asarray(fixed_old["transform"], dtype=np.float64)
                    - np.asarray(fixed_new["transform"], dtype=np.float64)
                )
            )
        )
        tree["fixed_translation_metric_difference_m"] = abs(
            float(fixed_old["metrics"]["boundary_translation_m"])
            - float(fixed_new["metrics"]["boundary_translation_m"])
        )
        tree["fixed_rotation_metric_difference_deg"] = abs(
            float(fixed_old["metrics"]["boundary_rotation_deg"])
            - float(fixed_new["metrics"]["boundary_rotation_deg"])
        )
        cases.append({**selected, "comparison": tree})

    report = {
        "experiment": "V14.5 raw-RGB cache integrity",
        "case_count": len(cases),
        "raw_inference_rerun": True,
        "full_original_manifest_context": args.rerun_report is not None,
        "candidate_rng_order_sensitivity_controlled": args.rerun_report is not None,
        "old_report_sha256": sha256(args.old_v10_report),
        "maximum": {
            key: max(case["comparison"][key] for case in cases)
            for key in (
                "maximum_absolute_difference",
                "fixed_transform_maximum",
                "fixed_translation_metric_difference_m",
                "fixed_rotation_metric_difference_deg",
            )
        },
        "cases": cases,
    }
    output = args.output or (args.rerun_root / "raw_cache_integrity.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"output": str(output), **report["maximum"]}, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
