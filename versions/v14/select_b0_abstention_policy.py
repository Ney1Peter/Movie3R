#!/usr/bin/env python3
"""Select a causal cross96-B0 commit/abstain gate with source-group CV.

Rejected cases do *not* silently replace B0 with raw reset and claim a better
camera.  They explicitly abstain from global-gauge commitment and retain a
clean local reset trajectory.  Consequently the output is a risk--coverage
curve, not an artificial full-coverage camera mean.
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
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/b0_abstention_gate"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
    parser.add_argument("--dev-report", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--confirm-report", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def features(row: dict[str, Any]) -> dict[str, float]:
    payload = row["current"]["features"]
    values = {
        "boundary_translation_norm_m": payload["boundary_translation_norm_m"],
        "boundary_rotation_deg": payload["boundary_rotation_deg"],
        "proposal_disagreement_translation_m": row["proposal_disagreement"]["translation_m"],
        "proposal_disagreement_rotation_deg": row["proposal_disagreement"]["rotation_deg"],
    }
    # The disagreement fields are intentionally excluded below: the gate must
    # remain a single-cross96 runtime and cannot depend on the old proposal.
    for key, value in payload["geometry_self_consistency"].items():
        values[f"geometry_{key}"] = value
    for key, value in payload["token"].items():
        values[key] = value
    return {key: (float(value) if np.isfinite(float(value)) else float("nan")) for key, value in values.items()}


def design(report: dict[str, Any], names: list[str] | None = None, medians: np.ndarray | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = [features(row) for row in report["cases"]]
    names = sorted(key for key in maps[0] if not key.startswith("proposal_disagreement")) if names is None else names
    matrix = np.asarray([[mapping[key] for key in names] for mapping in maps], dtype=np.float64)
    if medians is None:
        medians = np.nanmedian(matrix, axis=0); medians[~np.isfinite(medians)] = 0.0
    return np.where(np.isfinite(matrix), matrix, medians[None]), names, medians


def summary(rows: list[dict[str, Any]], commit: np.ndarray) -> dict[str, Any]:
    metrics = [row["current"]["metrics_in_current_gauge"] for row in rows]
    catastrophic = np.asarray([row["catastrophic"] for row in metrics], dtype=bool)
    composite = np.asarray([row["composite"] for row in metrics], dtype=np.float64)
    output = {
        "case_count": len(rows), "commit_count": int(commit.sum()), "coverage": float(commit.mean()),
        "all_case_catastrophic_rate": float(catastrophic.mean()),
        "committed_catastrophic_count": int(catastrophic[commit].sum()),
        "committed_catastrophic_rate": float(catastrophic[commit].mean()) if commit.any() else float("nan"),
        "committed_composite": {
            "mean": float(composite[commit].mean()) if commit.any() else float("nan"),
            "p95": float(np.quantile(composite[commit], .95)) if commit.any() else float("nan"),
        },
        "by_source": {},
    }
    for source in SOURCES:
        indices = np.asarray([row["source"] == source for row in rows])
        chosen = commit & indices
        if not indices.any():
            continue
        output["by_source"][source] = {
            "case_count": int(indices.sum()), "commit_count": int(chosen.sum()),
            "coverage": float(chosen.sum() / indices.sum()),
            "all_catastrophic_rate": float(catastrophic[indices].mean()),
            "committed_catastrophic_rate": float(catastrophic[chosen].mean()) if chosen.any() else float("nan"),
        }
    return output


def qualifies(value: dict[str, Any]) -> bool:
    rows = value["by_source"].values()
    finite_sources = [row for row in rows if row["commit_count"] >= 4]
    return bool(
        value["coverage"] >= .40
        and value["committed_catastrophic_rate"] <= value["all_case_catastrophic_rate"] * .70
        and len(finite_sources) >= 3
        and sum(
            row["committed_catastrophic_rate"] <= row["all_catastrophic_rate"] + 1e-12
            for row in finite_sources
        ) >= 3
    )


def to_json(tree: DecisionTreeClassifier, names: list[str], node: int = 0) -> dict[str, Any]:
    left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
    if left == right:
        values = tree.tree_.value[node][0]
        safe = int(np.where(tree.classes_ == 1)[0][0]) if 1 in tree.classes_ else None
        return {"safe_probability": float(values[safe] / max(values.sum(), 1.0)) if safe is not None else 0.0}
    return {"feature": names[int(tree.tree_.feature[node])], "threshold": float(tree.tree_.threshold[node]), "left": to_json(tree, names, int(left)), "right": to_json(tree, names, int(right))}


def probability(node: dict[str, Any], value: dict[str, float]) -> float:
    if "safe_probability" in node:
        return float(node["safe_probability"])
    return probability(node["left"] if value[node["feature"]] <= node["threshold"] else node["right"], value)


def rank(row: dict[str, Any]) -> tuple[float, float, int, int, int]:
    params, value = row["parameters"], row["oof"]
    return (value["committed_catastrophic_rate"], -value["coverage"], int(params["max_depth"]), -int(params["min_samples_leaf"]), -int(params["safe_probability_threshold"] * 100))


def dev(args: argparse.Namespace) -> None:
    report = json.loads(args.dev_report.read_text(encoding="utf-8"))
    if report["failures"]:
        raise RuntimeError("Dev report has forward failures")
    rows = report["cases"]
    X, names, medians = design(report)
    label = np.asarray([not row["current"]["metrics_in_current_gauge"]["catastrophic"] for row in rows], dtype=int)
    groups = np.asarray([row["source"] for row in rows])
    candidates = []
    for depth in (1, 2, 3):
        for leaf in (4, 8, 12, 16):
            for threshold in (.35, .45, .55, .65, .75):
                scores = np.zeros(len(rows), dtype=np.float64)
                for train, test in LeaveOneGroupOut().split(X, label, groups):
                    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=20260803).fit(X[train], label[train])
                    predicted = tree.predict_proba(X[test])
                    scores[test] = predicted[:, int(np.where(tree.classes_ == 1)[0][0])] if 1 in tree.classes_ else 0.0
                item = {"parameters": {"max_depth": depth, "min_samples_leaf": leaf, "safe_probability_threshold": threshold}, "oof": summary(rows, scores >= threshold)}
                item["qualified"] = qualifies(item["oof"]); candidates.append(item)
    qualified = [row for row in candidates if row["qualified"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan = {"experiment": "cross96 B0 abstention gate leave-one-source-out development", "input": str(args.dev_report), "features": names, "candidate_count": len(candidates), "qualified_count": len(qualified), "candidates": sorted(candidates, key=rank)}
    (args.output_dir / "DEV_GROUP_CV.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    if not qualified:
        print("NO_GO_B0_ABSTENTION_GATE"); return
    winner = min(qualified, key=rank)
    p = winner["parameters"]
    tree = DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], class_weight="balanced", random_state=20260803).fit(X, label)
    policy = {"freeze_id": "B0_ABSTENTION_GATE_V1_20260803", "status": "frozen_after_dev_before_confirmation", "runtime": {"features": names, "feature_medians": medians.tolist(), "tree": to_json(tree, names), "safe_probability_threshold": p["safe_probability_threshold"], "commit": "cross96 B0", "reject": "do not commit a global Boundary; keep clean raw-reset shot-local trajectory", "future_frames": 0}, "development_oof": winner["oof"]}
    (args.output_dir / "FROZEN_POLICY_BEFORE_CONFIRM.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"winner": p, "oof": winner["oof"]}, indent=2))


def confirm(args: argparse.Namespace) -> None:
    if args.confirm_report is None or args.policy is None:
        raise ValueError("confirm requires --confirm-report and --policy")
    report, policy = json.loads(args.confirm_report.read_text(encoding="utf-8")), json.loads(args.policy.read_text(encoding="utf-8"))
    runtime = policy["runtime"]; names, medians = runtime["features"], np.asarray(runtime["feature_medians"])
    _, _, _ = design(report, names, medians)
    scores = []
    for row in report["cases"]:
        item = features(row); item = {name: item[name] if np.isfinite(item[name]) else float(medians[index]) for index, name in enumerate(names)}
        scores.append(probability(runtime["tree"], item))
    result = {"experiment": "cross96 B0 abstention confirmation", "input": str(args.confirm_report), "policy": str(args.policy), "summary": summary(report["cases"], np.asarray(scores) >= runtime["safe_probability_threshold"]), "safe_probability": {"mean": float(np.mean(scores)), "p90": float(np.quantile(scores, .90))}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CONFIRMATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


def main() -> None:
    args = parse_args(); dev(args) if args.phase == "dev" else confirm(args)


if __name__ == "__main__":
    main()
