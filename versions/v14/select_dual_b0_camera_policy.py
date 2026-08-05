#!/usr/bin/env python3
"""Dev-only selection and confirmation of a dual-B0 camera safety selector.

This optional cost ablation chooses between the cross96 B0 and an old B0 that
has already been converted to the cross96 state/gauge.  A shallow decision
tree sees only causal proposal features.  GT camera errors are used here only
to select/freeze the tree on pair-disjoint dev data, never at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV = REPO_ROOT / "output/v14_cut_first_cross_source/dual_b0_camera_features_dev96/report.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/dual_b0_camera_safety_selector"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
    parser.add_argument("--dev-report", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--confirm-report", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def feature_map(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for branch in ("current", "old"):
        payload = row[branch]["features"]
        prefix = "cross96" if branch == "current" else "old"
        result[f"{prefix}_boundary_translation_norm_m"] = finite(payload["boundary_translation_norm_m"])
        result[f"{prefix}_boundary_rotation_deg"] = finite(payload["boundary_rotation_deg"])
        for key, value in payload["geometry_self_consistency"].items():
            result[f"{prefix}_geometry_{key}"] = finite(value)
        for key, value in payload["token"].items():
            result[f"{prefix}_{key}"] = finite(value)
    for key, value in row["proposal_disagreement"].items():
        result[f"proposal_disagreement_{key}"] = finite(value)
    return result


def matrix(report: dict[str, Any], names: list[str] | None = None, medians: np.ndarray | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = [feature_map(row) for row in report["cases"]]
    names = sorted(maps[0]) if names is None else names
    values = np.asarray([[mapping[name] for name in names] for mapping in maps], dtype=np.float64)
    if medians is None:
        medians = np.nanmedian(values, axis=0)
        medians[~np.isfinite(medians)] = 0.0
    values = np.where(np.isfinite(values), values, medians[None])
    return values, names, medians


def metrics(rows: list[dict[str, Any]], choose_old: np.ndarray) -> dict[str, Any]:
    current = np.asarray([row["current"]["metrics_in_current_gauge"]["composite"] for row in rows], dtype=np.float64)
    old = np.asarray([row["old"]["metrics_adapted_to_current_gauge"]["composite"] for row in rows], dtype=np.float64)
    selected = np.where(choose_old, old, current)
    current_cat = np.asarray([row["current"]["metrics_in_current_gauge"]["catastrophic"] for row in rows], dtype=bool)
    old_cat = np.asarray([row["old"]["metrics_adapted_to_current_gauge"]["catastrophic"] for row in rows], dtype=bool)
    selected_cat = np.where(choose_old, old_cat, current_cat)
    by_source = {}
    for source in SOURCES:
        indices = np.asarray([row["source"] == source for row in rows])
        if not indices.any():
            continue
        by_source[source] = {
            "count": int(indices.sum()), "switch_count": int(choose_old[indices].sum()),
            "current_composite": float(current[indices].mean()),
            "selected_composite": float(selected[indices].mean()),
            "relative_gain": float(1 - selected[indices].mean() / current[indices].mean()),
            "current_catastrophic": int(current_cat[indices].sum()),
            "selected_catastrophic": int(selected_cat[indices].sum()),
        }
    return {
        "count": len(rows), "switch_count": int(choose_old.sum()), "coverage": float(choose_old.mean()),
        "current_composite": {"mean": float(current.mean()), "p95": float(np.quantile(current, .95))},
        "selected_composite": {"mean": float(selected.mean()), "p95": float(np.quantile(selected, .95))},
        "relative_gain": float(1 - selected.mean() / current.mean()),
        "current_catastrophic": int(current_cat.sum()), "selected_catastrophic": int(selected_cat.sum()),
        "catastrophic_relative_reduction": float(1 - selected_cat.sum() / max(current_cat.sum(), 1)),
        "by_source": by_source,
    }


def qualifies(summary: dict[str, Any]) -> bool:
    source_gains = [row["relative_gain"] for row in summary["by_source"].values()]
    source_cat_reduction = [
        row["selected_catastrophic"] < row["current_catastrophic"]
        for row in summary["by_source"].values()
    ]
    return bool(
        summary["switch_count"] >= 6
        and summary["relative_gain"] >= .01
        and summary["catastrophic_relative_reduction"] >= .10
        and summary["selected_composite"]["p95"] <= summary["current_composite"]["p95"] + 1e-12
        and sum(gain >= -1e-9 for gain in source_gains) >= 3
        and sum(source_cat_reduction) >= 2
    )


def tree_json(tree: DecisionTreeClassifier, names: list[str], node: int = 0) -> dict[str, Any]:
    children_left, children_right = tree.tree_.children_left, tree.tree_.children_right
    if children_left[node] == children_right[node]:
        counts = tree.tree_.value[node][0]
        return {"leaf_probability_old": float(counts[1] / max(counts.sum(), 1.0)), "samples": int(tree.tree_.n_node_samples[node])}
    feature = int(tree.tree_.feature[node])
    return {
        "feature": names[feature], "threshold": float(tree.tree_.threshold[node]),
        "left": tree_json(tree, names, int(children_left[node])),
        "right": tree_json(tree, names, int(children_right[node])),
    }


def tree_probability(node: dict[str, Any], values: dict[str, float]) -> float:
    if "leaf_probability_old" in node:
        return float(node["leaf_probability_old"])
    return tree_probability(node["left"] if values[node["feature"]] <= node["threshold"] else node["right"], values)


def rank(row: dict[str, Any]) -> tuple[float, float, float, int, int, int]:
    params = row["parameters"]
    # Prefer risk reduction, then mean gain; conservative trees are tie breakers.
    return (
        -row["oof"]["catastrophic_relative_reduction"], -row["oof"]["relative_gain"],
        row["oof"]["coverage"], int(params["max_depth"]), -int(params["min_samples_leaf"]),
        0 if params["class_weight"] is None else 1,
    )


def dev(args: argparse.Namespace) -> None:
    report = json.loads(args.dev_report.read_text(encoding="utf-8"))
    if report["failures"]:
        raise RuntimeError("Dev report has forward failures")
    rows = report["cases"]
    X, names, medians = matrix(report)
    current_error = np.asarray([row["current"]["metrics_in_current_gauge"]["composite"] for row in rows])
    old_error = np.asarray([row["old"]["metrics_adapted_to_current_gauge"]["composite"] for row in rows])
    target = (old_error < current_error).astype(int)
    groups = np.asarray([row["source"] for row in rows])
    records = []
    for max_depth in (1, 2, 3):
        for min_leaf in (2, 4, 8, 12):
            for class_weight in (None, "balanced"):
                for threshold in (.35, .45, .55, .65):
                    probability = np.zeros(len(rows), dtype=np.float64)
                    for train, test in LeaveOneGroupOut().split(X, target, groups):
                        model = DecisionTreeClassifier(
                            max_depth=max_depth, min_samples_leaf=min_leaf,
                            class_weight=class_weight, random_state=20260803,
                        ).fit(X[train], target[train])
                        probabilities = model.predict_proba(X[test])
                        if 1 in model.classes_:
                            probability[test] = probabilities[:, int(np.where(model.classes_ == 1)[0][0])]
                        else:
                            probability[test] = 0.0
                    decision = probability >= threshold
                    result = {
                        "parameters": {
                            "max_depth": max_depth, "min_samples_leaf": min_leaf,
                            "class_weight": class_weight, "old_probability_threshold": threshold,
                        },
                        "oof": metrics(rows, decision),
                    }
                    result["qualified"] = qualifies(result["oof"])
                    records.append(result)
    qualified = [row for row in records if row["qualified"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan = {
        "experiment": "dual B0 camera selector leave-one-source-out development",
        "input": str(args.dev_report), "checkpoint": report["checkpoints"],
        "features": names, "candidate_count": len(records), "qualified_count": len(qualified),
        "candidates": sorted(records, key=rank),
    }
    (args.output_dir / "DEV_GROUP_CV.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    if not qualified:
        print("NO_GO_DUAL_B0_CAMERA_SELECTOR")
        return
    winner = min(qualified, key=rank)
    parameters = winner["parameters"]
    final = DecisionTreeClassifier(
        max_depth=parameters["max_depth"], min_samples_leaf=parameters["min_samples_leaf"],
        class_weight=parameters["class_weight"], random_state=20260803,
    ).fit(X, target)
    policy = {
        "freeze_id": "DUAL_B0_CAMERA_SELECTOR_V1_20260803",
        "status": "frozen_after_pair_disjoint_dev_before_confirmation",
        "method": "cross96/old B0 shallow safety selector (cost ablation)",
        "runtime": {
            "base": "cross96 B0", "alternative": "old B0 converted to cross96 raw/pre gauge",
            "features": names, "feature_medians": medians.tolist(),
            "tree": tree_json(final, names),
            "old_probability_threshold": parameters["old_probability_threshold"],
            "fallback": "exact cross96 B0", "future_frames": 0, "camera_only": True,
        },
        "development_oof_result": winner["oof"], "selection_parameters": parameters,
        "confirmation_status": "not_run",
    }
    (args.output_dir / "FROZEN_POLICY_BEFORE_CONFIRM.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": parameters, "oof": winner["oof"]}, indent=2))


def confirm(args: argparse.Namespace) -> None:
    if args.confirm_report is None or args.policy is None:
        raise ValueError("confirm requires --confirm-report and --policy")
    report, policy = json.loads(args.confirm_report.read_text(encoding="utf-8")), json.loads(args.policy.read_text(encoding="utf-8"))
    if report["failures"]:
        raise RuntimeError("Confirmation report has forward failures")
    runtime = policy["runtime"]
    names, medians = runtime["features"], np.asarray(runtime["feature_medians"], dtype=np.float64)
    _, _, _ = matrix(report, names, medians)  # validate all expected features exist
    probabilities = []
    for row in report["cases"]:
        values = feature_map(row)
        values = {name: (values[name] if np.isfinite(values[name]) else float(medians[index])) for index, name in enumerate(names)}
        probabilities.append(tree_probability(runtime["tree"], values))
    decision = np.asarray(probabilities) >= float(runtime["old_probability_threshold"])
    result = {
        "experiment": "dual B0 camera selector confirmation", "input": str(args.confirm_report),
        "policy": str(args.policy), "summary": metrics(report["cases"], decision),
        "old_probability": {"mean": float(np.mean(probabilities)), "p90": float(np.quantile(probabilities, .90))},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CONFIRMATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


def main() -> None:
    args = parse_args()
    dev(args) if args.phase == "dev" else confirm(args)


if __name__ == "__main__":
    main()
