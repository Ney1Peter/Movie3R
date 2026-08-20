#!/usr/bin/env python3
"""Aggregate EgoHumans reports with action/capture/camera-pair hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS: tuple[tuple[str, str, bool], ...] = (
    ("W-MPJPE_mm", "W-MPJPE", False),
    ("WA-MPJPE_mm", "WA-MPJPE", False),
    ("MPJPE_mm", "MPJPE", False),
    ("PA-MPJPE_mm", "PA-MPJPE", False),
    ("MPVPE_mm", "MPVPE", False),
    ("Accel_mm_frame2", "Accel", False),
    ("RTE_H3R_percent", "RTE-H3R", False),
    ("ROE_joint_proxy_deg", "ROE-joint proxy", False),
    ("Jitter_H3R", "Jitter", False),
    ("Foot_sliding_cm", "Foot sliding", False),
    ("ATE_Sim3_m", "ATE-Sim3", False),
    ("ATE_SE3_m", "ATE-SE3", False),
    ("Boundary_camera_t_m", "Boundary camera-t", False),
    ("Boundary_camera_R_deg", "Boundary camera-R", False),
    ("Boundary_root_m", "Boundary root", False),
    ("Post_root_m", "Post root", False),
    ("Seam_camera_t_m", "Seam camera-t", False),
    ("Seam_camera_R_deg", "Seam camera-R", False),
    ("Seam_root_m", "Seam root", False),
    ("Seam_CHRGE_m", "Seam CHRGE", False),
    ("CHRGE_m", "CHRGE", False),
    ("Pair_vector_m", "Pair vector", False),
    ("IDs", "IDs", False),
    ("IDF1", "IDF1", True),
    ("Coverage", "Coverage", True),
    ("Detection_precision", "Detection precision", True),
)
METRIC_KEYS = tuple(row[0] for row in METRICS)
CORE = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "Seam_root_m")
LITERATURE_MULTI_THUMBS_EGOHUMANS = {
    "W-MPJPE_mm": 279.0,
    "WA-MPJPE_mm": 166.0,
    "MPJPE_mm": 228.3,
    "MPVPE_mm": 262.2,
    "Accel_mm_frame2": 27.3,
    "ATE_Sim3_m": 0.7,
    "IDs": 0.97,
}
DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "m15_safe_boundary_permutation_causal_gru": "Movie3R-v15",
    "v16_0_m15_geometry": "Movie3R-v17 parent",
    "v17_harmony_multicue_safe": "Movie3R-v17 MultiCue-Safe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--primary")
    parser.add_argument("--parent", default="v16_0_m15_geometry")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def report_paths(inputs: list[Path]) -> list[Path]:
    files = set()
    for path in inputs:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(path.resolve().rglob("*.json"))
        else:
            raise FileNotFoundError(path)
    output = []
    for path in sorted(files):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1":
            output.append(path)
    if not output:
        raise ValueError("No EgoHumans candidate probe reports found")
    return output


def compatible_rows(first: dict[str, Any], second: dict[str, Any]) -> bool:
    for key in METRIC_KEYS:
        a, b = finite(first.get("metrics", {}).get(key)), finite(second.get("metrics", {}).get(key))
        if a is None or b is None:
            if a != b:
                return False
        elif not np.isclose(a, b, rtol=1e-6, atol=1e-5):
            return False
    return True


def collect(files: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    skipped: dict[str, dict[str, Any]] = {}
    sources = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors"):
            raise ValueError(f"Incomplete report {path}: {payload['errors']}")
        sources.append(str(path))
        for row in payload.get("skipped_cases", []):
            skipped[str(row["case_id"])] = row
        for kind, rows in (("method", payload.get("reference_rows", [])), ("candidate", payload.get("rows", []))):
            for row in rows:
                if row.get("status") != "complete":
                    continue
                method = str(row[kind])
                key = (str(row["case_id"]), method)
                value = {**row, "method_key": method}
                if key in unique and not compatible_rows(unique[key], value):
                    raise ValueError(f"Conflicting duplicate row: {key}")
                unique[key] = value
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, method), row in unique.items():
        grouped[method].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda value: str(value["case_id"]))
    return grouped, list(skipped.values()), sources


def mean(values: list[float | None]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return float(array.mean()) if len(array) else None


def macro(rows: list[dict[str, Any]], metric: str, unit: str) -> float | None:
    if unit == "case":
        return mean([finite(row["metrics"].get(metric)) for row in rows])
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row["metrics"].get(metric))
        if value is None:
            continue
        if unit == "capture":
            key = (str(row["sequence"]), str(row.get("capture")))
        elif unit == "action":
            key = (str(row["sequence"]),)
        else:
            raise ValueError(unit)
        groups[key].append(value)
    return mean([float(np.mean(values)) for values in groups.values()])


def distribution(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "std")}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "std": float(array.std()),
    }


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], metric: str, samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    hierarchy: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    actions = sorted({str(row["sequence"]) for row in rows})
    for action in actions:
        captures = sorted({str(row.get("capture")) for row in rows if str(row["sequence"]) == action})
        for capture in captures:
            values = [
                finite(row["metrics"].get(metric))
                for row in rows
                if str(row["sequence"]) == action and str(row.get("capture")) == capture
            ]
            array = np.asarray([value for value in values if value is not None], dtype=np.float64)
            if len(array):
                hierarchy[action][capture] = array
    actions = sorted(action for action, captures in hierarchy.items() if captures)
    if not actions:
        return {"low": None, "high": None, "samples": samples, "unit": "action_capture_case"}
    # Draw the same action -> capture -> camera-pair hierarchy in batches.
    # The former sample-wise Python loop was exact but made a 10k bootstrap
    # dominate the complete experiment runtime.  Each repeated action/capture
    # draw below still receives an independent lower-level resample.
    action_indices = rng.integers(0, len(actions), size=(samples, len(actions)))
    action_draws = np.empty((samples, len(actions)), dtype=np.float64)
    for draw_position in range(len(actions)):
        for action_index, action in enumerate(actions):
            sample_rows = np.flatnonzero(action_indices[:, draw_position] == action_index)
            if not len(sample_rows):
                continue
            captures = sorted(hierarchy[action])
            capture_indices = rng.integers(
                0, len(captures), size=(len(sample_rows), len(captures))
            )
            capture_draws = np.empty_like(capture_indices, dtype=np.float64)
            for capture_position in range(len(captures)):
                for capture_index, capture in enumerate(captures):
                    selected = np.flatnonzero(
                        capture_indices[:, capture_position] == capture_index
                    )
                    if not len(selected):
                        continue
                    values = hierarchy[action][capture]
                    case_indices = rng.integers(
                        0, len(values), size=(len(selected), len(values))
                    )
                    capture_draws[selected, capture_position] = values[case_indices].mean(axis=1)
            action_draws[sample_rows, draw_position] = capture_draws.mean(axis=1)
    estimates = action_draws.mean(axis=1)
    return {
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
        "samples": samples,
        "unit": "action_then_capture_then_camera_pair",
    }


def candidate_config(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        value = row.get("diagnostics", {}).get("candidate")
        if isinstance(value, dict):
            return value
    return None


def promotion(
    rows: list[dict[str, Any]], parent_rows: list[dict[str, Any]], summary: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    ratios = {}
    for metric in METRIC_KEYS:
        first, second = parent["case_macro"].get(metric), summary["case_macro"].get(metric)
        ratios[metric] = None if first is None or second is None or abs(first) < 1e-12 else float(second / first)
    core = [ratios[key] for key in CORE if ratios[key] is not None and ratios[key] > 0]
    geometric = float(np.exp(np.mean(np.log(core)))) if len(core) == len(CORE) else None
    parent_by_case = {str(row["case_id"]): row for row in parent_rows}
    w_harms = []
    for row in rows:
        other = parent_by_case.get(str(row["case_id"]))
        if other is None:
            continue
        a, b = finite(row["metrics"].get("W-MPJPE_mm")), finite(other["metrics"].get("W-MPJPE_mm"))
        if a is not None and b is not None and b > 1e-9:
            w_harms.append(a / b)
    checks = {
        "all_required_metrics_defined": all(
            finite(row["metrics"].get(key)) is not None
            for row in rows
            for key in ("W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "Seam_root_m", "IDF1", "Coverage")
            if (
                str(row["case_id"]) in parent_by_case
                and finite(parent_by_case[str(row["case_id"])]["metrics"].get(key)) is not None
            )
        ),
        "core_three_of_five_improve": sum(ratios[key] is not None and ratios[key] < 1.0 for key in CORE) >= 3,
        "core_geomean_improvement_ge_3pct": geometric is not None and geometric <= 0.97,
        "mpjpe_noninferior_2pct": ratios["MPJPE_mm"] is not None and ratios["MPJPE_mm"] <= 1.02,
        "mpvpe_noninferior_2pct": ratios["MPVPE_mm"] is not None and ratios["MPVPE_mm"] <= 1.02,
        "coverage_drop_le_1pp": (
            summary["case_macro"]["Coverage"] is not None
            and parent["case_macro"]["Coverage"] is not None
            and summary["case_macro"]["Coverage"] >= parent["case_macro"]["Coverage"] - 0.01
        ),
        "idf1_drop_le_0p01": (
            summary["case_macro"]["IDF1"] is not None
            and parent["case_macro"]["IDF1"] is not None
            and summary["case_macro"]["IDF1"] >= parent["case_macro"]["IDF1"] - 0.01
        ),
        "worst_case_w_harm_le_20pct": not w_harms or max(w_harms) <= 1.20,
    }
    return {
        "passes_development_gate": bool(all(checks.values())),
        "checks": checks,
        "ratios_to_parent": ratios,
        "core_geometric_mean_ratio": geometric,
        "core_relative_improvement": None if geometric is None else 1.0 - geometric,
        "worst_case_w_ratio": max(w_harms, default=None),
        "w_nonworse_rate": float(np.mean(np.asarray(w_harms) <= 1.0)) if w_harms else None,
    }


def latex_table(methods: dict[str, Any], order: list[str]) -> str:
    columns = ("W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE_Sim3_m", "IDs")
    labels = ("W", "WA", "MPJPE", "MPVPE", "Accel", "ATE", "IDs")
    end = r" \\"
    lines = [
        "% Auto-generated for Movie3R-EgoHumans-CS100-v1; Multi-THuMBS is protocol-different context.",
        "\\begin{tabular}{l" + "r" * len(columns) + "}",
        "\\toprule",
        "Method & " + " & ".join(labels) + end,
        "\\midrule",
    ]
    for method in order:
        if method not in methods:
            continue
        values = []
        for key in columns:
            value = methods[method]["case_macro"].get(key)
            if value is None:
                values.append("--")
            elif key == "ATE_Sim3_m":
                values.append(f"{value:.3f}")
            elif key == "IDs":
                values.append(f"{value:.2f}")
            else:
                values.append(f"{value:.1f}")
        lines.append(f"{DISPLAY.get(method, method)} & " + " & ".join(values) + end)
    ref = LITERATURE_MULTI_THUMBS_EGOHUMANS
    lines.extend(
        (
            "\\midrule",
            "Multi-THuMBS$^{\\dagger}$ & "
            + " & ".join(
                f"{ref[key]:.3f}" if key == "ATE_Sim3_m" else f"{ref[key]:.1f}"
                for key in columns
            )
            + end,
            "\\bottomrule",
            "\\end{tabular}",
            "% $^{\\dagger}$ Literature reference only; official manifest/evaluator is unavailable.",
            "",
        )
    )
    return "\n".join(lines)


def markdown(result: dict[str, Any], order: list[str]) -> str:
    columns = ("W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE_Sim3_m", "ATE_SE3_m", "IDF1", "IDs")
    lines = [
        f"# Movie3R EgoHumans {result['split']} summary",
        "",
        f"- Protocol: Movie3R-EgoHumans-CS100-v1 (100 frames, 50 pre + 50 post, 20 FPS)",
        f"- Cases: {result['case_count']}; captures: {result['capture_count']}; actions: {result['action_count']}",
        f"- Method-independent evaluator-unavailable cases: {result['evaluator_unavailable_count']}",
        "- Primary aggregation: case macro; uncertainty: action → capture → camera-pair bootstrap.",
        "",
        "| Method | W | WA | MPJPE | MPVPE | Accel | ATE-Sim3 | ATE-SE3 | IDF1 | IDs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in order:
        if method not in result["methods"]:
            continue
        values = []
        for key in columns:
            value = result["methods"][method]["case_macro"].get(key)
            if value is None:
                values.append("--")
            elif key in {"ATE_Sim3_m", "ATE_SE3_m", "IDF1"}:
                values.append(f"{value:.4f}")
            else:
                values.append(f"{value:.1f}")
        lines.append(f"| {DISPLAY.get(method, method)} | " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "Multi-THuMBS EgoHumans values are retained as a literature target only. Its exact capture/camera/cut/evaluator protocol is not public, so this is not a same-protocol leaderboard claim.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    files = report_paths(args.inputs)
    grouped, skipped, sources = collect(files)
    case_sets = {method: {str(row["case_id"]) for row in rows} for method, rows in grouped.items()}
    all_cases = set().union(*case_sets.values()) if case_sets else set()
    complete_methods = {
        method: rows for method, rows in grouped.items() if case_sets[method] == all_cases
    }
    if args.parent not in complete_methods:
        raise ValueError(f"Parent {args.parent} does not cover all {len(all_cases)} cases")
    rng = np.random.default_rng(int(args.seed))
    methods = {}
    for method, rows in sorted(complete_methods.items()):
        methods[method] = {
            "display_name": DISPLAY.get(method, method),
            "candidate": candidate_config(rows),
            "case_count": len(rows),
            "case_macro": {key: macro(rows, key, "case") for key in METRIC_KEYS},
            "capture_macro": {key: macro(rows, key, "capture") for key in METRIC_KEYS},
            "action_macro": {key: macro(rows, key, "action") for key in METRIC_KEYS},
            "case_distribution": {
                key: distribution([finite(row["metrics"].get(key)) for row in rows])
                for key in METRIC_KEYS
            },
            "finite_case_count": {
                key: sum(finite(row["metrics"].get(key)) is not None for row in rows)
                for key in METRIC_KEYS
            },
            "hierarchical_bootstrap_ci95": {
                key: hierarchical_bootstrap(rows, key, int(args.bootstrap), rng)
                for key in METRIC_KEYS
            },
        }
    parent_rows = complete_methods[args.parent]
    parent_summary = methods[args.parent]
    ranking = []
    for method, rows in complete_methods.items():
        if method == args.parent or method.startswith("m"):
            continue
        decision = promotion(rows, parent_rows, methods[method], parent_summary)
        methods[method]["development_promotion"] = decision
        ranking.append(method)
    ranking.sort(
        key=lambda method: (
            not methods[method]["development_promotion"]["passes_development_gate"],
            methods[method]["development_promotion"]["core_geometric_mean_ratio"]
            if methods[method]["development_promotion"]["core_geometric_mean_ratio"] is not None
            else float("inf"),
        )
    )
    primary = args.primary or (ranking[0] if ranking else args.parent)
    if primary not in methods:
        raise ValueError(f"Primary {primary} is unavailable")
    primary_rows = complete_methods[primary]
    accepted = sum(
        bool(row.get("diagnostics", {}).get("reliability_gate", {}).get("accepted"))
        for row in primary_rows
    )
    captures = {(str(row["sequence"]), str(row.get("capture"))) for row in primary_rows}
    actions = {str(row["sequence"]) for row in primary_rows}
    result = {
        "schema_version": "Movie3R-v19-EgoHumans-CS100-summary-v1",
        "protocol": {
            "name": "Movie3R-EgoHumans-CS100-v1",
            "frames": 100,
            "pre_frames": 50,
            "post_frames": 50,
            "fps": 20,
            "split": args.split,
            "parameter_selection_allowed": args.split == "development",
            "primary_aggregation": "case_macro",
            "uncertainty": "action_then_capture_then_camera_pair_bootstrap",
        },
        "split": args.split,
        "sources": sources,
        "primary": primary,
        "parent": args.parent,
        "case_count": len(all_cases),
        "capture_count": len(captures),
        "action_count": len(actions),
        "actions": sorted(actions),
        "evaluator_unavailable_count": len(skipped),
        "skipped_cases": sorted(skipped, key=lambda row: str(row["case_id"])),
        "methods": methods,
        "candidate_ranking": ranking,
        "passing_development_candidates": [
            method for method in ranking
            if methods[method]["development_promotion"]["passes_development_gate"]
        ],
        "primary_gate": {"accepted": accepted, "fallback": len(primary_rows) - accepted},
        "multi_thumbs_literature_reference": {
            "dataset": "EgoHumans",
            "values": LITERATURE_MULTI_THUMBS_EGOHUMANS,
            "contract": "literature target only; exact official capture/camera/cut/evaluator unavailable",
        },
        "metric_contract": {
            "Multi-THuMBS_named": "public-description reproduction, not official evaluator",
            "ATE": "both Sim3 and SE3 are reported; literature paper only labels ATE",
            "IDs": "total persistent-ID switches per case before macro averaging",
            "GT_runtime_isolation": True,
        },
    }
    order = [
        "m0_strict_human3r",
        "m15_safe_boundary_permutation_causal_gru",
        args.parent,
        "v17_harmony_multicue_safe",
        primary,
    ]
    order = list(dict.fromkeys(order))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "summary.json", result)
    fields = ["case_id", "action", "capture", "angle_stratum", "person_count", "method", *METRIC_KEYS]
    with (args.output / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in order + [value for value in sorted(methods) if value not in order]:
            for row in complete_methods.get(method, []):
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "action": row["sequence"],
                        "capture": row.get("capture"),
                        "angle_stratum": row.get("angle_stratum"),
                        "person_count": row.get("person_count"),
                        "method": method,
                        **{key: row["metrics"].get(key) for key in METRIC_KEYS},
                    }
                )
    (args.output / "main_table.tex").write_text(latex_table(methods, order), encoding="utf-8")
    (args.output / "SUMMARY.md").write_text(markdown(result, order), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "split": args.split,
                "cases": len(all_cases),
                "captures": len(captures),
                "primary": primary,
                "top10": ranking[:10],
                "passing": result["passing_development_candidates"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
