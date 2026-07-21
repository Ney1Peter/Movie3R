#!/usr/bin/env python3
"""Evaluate V14 depth-free selective World-Memory relocalization with LOSO gates."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v14_selective_world_memory" / "candidate_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v14_selective_world_memory" / "evaluation"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
ANCHOR_STRATEGIES = (
    "confidence",
    "spatial",
    "temporal",
    "confidence_spatial",
    "temporal_spatial",
    "temporal_spatial_static",
)

GEOMETRY_FEATURES = (
    "fit_failed",
    "correspondence_count",
    "mutual_match_count",
    "fit_residual_mean_m",
    "fit_residual_median_m",
    "fit_residual_p90_m",
    "inlier_ratio_0_10m",
    "inlier_ratio_0_20m",
    "robust_inlier_ratio",
    "image_coverage_8x8",
    "query_confidence_mean",
    "query_static_rate",
    "anchor_confidence_mean",
    "anchor_observation_count_mean",
    "anchor_xyz_variance_mean",
    "anchor_descriptor_variance_mean",
    "anchor_normal_stability_mean",
    "anchor_static_rate_mean",
    "anchor_edge_score_mean",
    "source_condition_number",
    "source_planarity_ratio",
    "source_linearity_ratio",
    "source_volume_proxy",
    "source_extent_norm",
    "target_condition_number",
    "target_planarity_ratio",
    "target_linearity_ratio",
    "target_volume_proxy",
    "target_extent_norm",
    "top1_top3_translation_consistency_m",
    "top1_top3_rotation_consistency_deg",
    "top3_top5_translation_consistency_m",
    "top3_top5_rotation_consistency_deg",
    "one_three_translation_consistency_m",
    "one_three_rotation_consistency_deg",
    "strategy_translation_consistency_m",
    "strategy_rotation_consistency_deg",
    "refinement_translation_m",
    "refinement_rotation_deg",
    "icp_iteration_count",
    "icp_last_delta_translation_m",
    "icp_last_delta_rotation_deg",
)
TOKEN_FEATURES = (
    "global_top1_score",
    "global_top1_margin",
    "mean_cosine",
    "mean_margin",
    "texture_score",
)
TORSO_FEATURES = ("human_torso_jump_deg",)
HUMAN_FEATURES = ("human_root_jump_m", "precut_human_speed_m")
GRAVITY_FEATURES = ("normal_conflict_deg",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def load_cases(input_dir: Path) -> list[dict]:
    shards = sorted(input_dir.glob("v14_candidates_shard_*_of_*.json"))
    if not shards:
        raise FileNotFoundError(input_dir)
    cases = []
    for shard in shards:
        cases.extend(json.loads(shard.read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V14 cases")
    return cases


def source(case: dict) -> str:
    return str(case["record"].get("source", "unknown"))


def candidate_key(descriptor: str, strategy: str, frame_count: int) -> str:
    return f"{descriptor}__{strategy}__{frame_count}f"


def configs(cases: list[dict]) -> list[tuple[str, str]]:
    names = set.intersection(*(set(case["variants"]) for case in cases))
    output = set()
    for name in names:
        descriptor, strategy, frame = name.split("__")
        if frame == "3f":
            output.add((descriptor, strategy))
    return sorted(output)


def finite(value, default=float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def rotation_error_deg_local(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def candidate_error(row: dict | None) -> tuple[float, float]:
    if row is None or row.get("fit_failed", False):
        return float("inf"), float("inf")
    return finite(row.get("camera_translation_error_m"), float("inf")), finite(
        row.get("camera_rotation_error_deg"), float("inf")
    )


def joint_cost(row: dict | None) -> float:
    translation, rotation = candidate_error(row)
    if not math.isfinite(translation) or not math.isfinite(rotation):
        return 100.0
    return translation / 0.25 + rotation / 5.0


def is_catastrophic(row: dict | None) -> bool:
    translation, rotation = candidate_error(row)
    return not math.isfinite(translation) or not math.isfinite(rotation) or translation > 1.0 or rotation > 30.0


def helpful(candidate: dict | None, fixed: dict) -> bool:
    return not is_catastrophic(candidate) and joint_cost(candidate) + 0.10 < joint_cost(fixed)


def safe_selection_cost(row: dict | None) -> float:
    return joint_cost(row) + (20.0 if is_catastrophic(row) else 0.0)


def oracle_gain(case: dict, descriptor: str, strategy: str) -> float:
    fixed = case["baselines"]["fixed_explicit"]
    rows = [
        fixed,
        case["variants"].get(candidate_key(descriptor, strategy, 1)),
        case["variants"].get(candidate_key(descriptor, strategy, 3)),
    ]
    return safe_selection_cost(fixed) - min(safe_selection_cost(row) for row in rows)


def select_config(cases: list[dict], available: list[tuple[str, str]]) -> tuple[str, str, dict]:
    scored = []
    for descriptor, strategy in available:
        gains = np.asarray([oracle_gain(case, descriptor, strategy) for case in cases], dtype=np.float64)
        three = [case["variants"].get(candidate_key(descriptor, strategy, 3)) for case in cases]
        helpful_rate = float(np.mean([helpful(row, case["baselines"]["fixed_explicit"]) for row, case in zip(three, cases)]))
        catastrophic = float(np.mean([is_catastrophic(row) for row in three]))
        score = float(gains.mean() + 0.25 * np.percentile(gains, 75) + 0.5 * helpful_rate - 0.05 * catastrophic)
        scored.append((score, descriptor, strategy, helpful_rate, catastrophic, float(gains.mean())))
    scored.sort(reverse=True)
    best = scored[0]
    return best[1], best[2], {
        "score": best[0],
        "helpful_rate": best[3],
        "catastrophic_rate": best[4],
        "oracle_gain_mean": best[5],
        "ranking": [
            {
                "descriptor": row[1],
                "strategy": row[2],
                "score": row[0],
                "helpful_rate": row[3],
                "catastrophic_rate": row[4],
                "oracle_gain_mean": row[5],
            }
            for row in scored
        ],
    }


def nested(row: dict, group: str, key: str, default=float("nan")) -> float:
    value = row.get(group, {})
    return finite(value.get(key), default) if isinstance(value, dict) else default


def feature_value(case: dict, row: dict | None, name: str, frame_count: int) -> float:
    if row is None:
        return 1.0 if name == "fit_failed" else float("nan")
    if name == "fit_failed":
        return float(bool(row.get("fit_failed", True)))
    if name == "texture_score":
        return finite(case.get("texture_score"))
    if name.startswith("source_"):
        key = name[len("source_") :]
        if key == "extent_norm":
            extent = np.asarray(row.get("source_geometry", {}).get("extent_xyz_m", [np.nan] * 3), dtype=float)
            return float(np.linalg.norm(extent))
        return nested(row, "source_geometry", key)
    if name.startswith("target_"):
        key = name[len("target_") :]
        if key == "extent_norm":
            extent = np.asarray(row.get("target_geometry", {}).get("extent_xyz_m", [np.nan] * 3), dtype=float)
            return float(np.linalg.norm(extent))
        return nested(row, "target_geometry", key)
    if name.startswith("one_three_") and frame_count == 1:
        return float("nan")
    if name.startswith("strategy_") and row.get("transform") is not None:
        current = np.asarray(row["transform"], dtype=np.float32)
        values = []
        for anchor_strategy in ANCHOR_STRATEGIES:
            other = case["variants"].get(
                candidate_key(str(row.get("descriptor")), anchor_strategy, frame_count)
            )
            if other is None or other.get("transform") is None:
                continue
            transform = np.asarray(other["transform"], dtype=np.float32)
            if name == "strategy_translation_consistency_m":
                values.append(float(np.linalg.norm(current[:3, 3] - transform[:3, 3])))
            else:
                values.append(rotation_error_deg_local(current[:3, :3], transform[:3, :3]))
        return float(np.median(values)) if values else float("nan")
    if name == "icp_iteration_count":
        return float(len(row.get("icp_iterations", [])))
    if name == "icp_last_delta_translation_m":
        iterations = row.get("icp_iterations", [])
        return finite(iterations[-1].get("delta_translation_m")) if iterations else float("nan")
    if name == "icp_last_delta_rotation_deg":
        iterations = row.get("icp_iterations", [])
        return finite(iterations[-1].get("delta_rotation_deg")) if iterations else float("nan")
    return finite(row.get(name))


def feature_matrix(
    cases: list[dict],
    descriptor: str,
    strategy: str,
    frame_count: int,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    rows = []
    for case in cases:
        candidate = case["variants"].get(candidate_key(descriptor, strategy, frame_count))
        rows.append([feature_value(case, candidate, name, frame_count) for name in feature_names])
    return np.asarray(rows, dtype=np.float32)


@dataclass
class Transformer:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, value: np.ndarray) -> "Transformer":
        finite_value = np.where(np.isfinite(value), value, np.nan)
        median = np.nanmedian(finite_value, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(value), value, median)
        signed_log = np.sign(filled) * np.log1p(np.abs(filled))
        mean = signed_log.mean(axis=0)
        std = signed_log.std(axis=0)
        std = np.where(std > 1e-5, std, 1.0)
        return cls(median=median.astype(np.float32), mean=mean.astype(np.float32), std=std.astype(np.float32))

    def apply(self, value: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(value), value, self.median)
        signed_log = np.sign(filled) * np.log1p(np.abs(filled))
        return ((signed_log - self.mean) / self.std).astype(np.float32)


@dataclass
class LinearModel:
    weight: np.ndarray
    bias: float
    constant: float | None = None

    def predict(self, value: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(value), self.constant, dtype=np.float32)
        logit = value @ self.weight + self.bias
        return (1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))).astype(np.float32)


def fit_linear(value: np.ndarray, target: np.ndarray, device: torch.device, steps: int, seed: int) -> LinearModel:
    target = target.astype(np.float32)
    if len(np.unique(target)) < 2:
        return LinearModel(np.zeros(value.shape[1], np.float32), 0.0, float(target.mean()))
    torch.manual_seed(seed)
    x = torch.as_tensor(value, dtype=torch.float32, device=device)
    y = torch.as_tensor(target[:, None], dtype=torch.float32, device=device)
    layer = torch.nn.Linear(value.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.03, weight_decay=0.02)
    positive = max(float(target.sum()), 1.0)
    negative = max(float(len(target) - target.sum()), 1.0)
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = layer(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    return LinearModel(
        layer.weight.detach().cpu().numpy()[0].astype(np.float32),
        float(layer.bias.detach().cpu().item()),
    )


def rank_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    positive, negative = int(target.sum()), int((~target).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def hand_score(case: dict, row: dict | None, frame_count: int) -> float:
    if row is None or row.get("fit_failed", True):
        return -20.0
    inlier = finite(row.get("inlier_ratio_0_20m"), 0.0)
    robust = finite(row.get("robust_inlier_ratio"), 0.0)
    coverage = finite(row.get("image_coverage_8x8"), 0.0)
    residual = finite(row.get("fit_residual_median_m"), 10.0)
    condition = nested(row, "target_geometry", "condition_number", 1e10)
    planarity = nested(row, "target_geometry", "planarity_ratio", 0.0)
    observation = finite(row.get("anchor_observation_count_mean"), 1.0)
    xyz_variance = finite(row.get("anchor_xyz_variance_mean"), 1.0)
    consistency_t = finite(row.get("top1_top3_translation_consistency_m"), 5.0)
    consistency_r = finite(row.get("top1_top3_rotation_consistency_deg"), 180.0)
    refinement_t = finite(row.get("refinement_translation_m"), 1.0)
    refinement_r = finite(row.get("refinement_rotation_deg"), 30.0)
    score = 3.0 * inlier + robust + 2.0 * coverage + 0.25 * min(observation, 4.0)
    score += 0.5 * planarity
    score -= 1.5 * math.log1p(max(residual, 0.0) * 10.0)
    score -= 0.25 * math.log1p(max(condition, 1.0))
    score -= math.log1p(max(xyz_variance, 0.0) * 20.0)
    score -= 0.5 * math.log1p(max(consistency_t, 0.0) * 5.0)
    score -= 0.25 * math.log1p(max(consistency_r, 0.0) / 5.0)
    score -= 0.75 * math.log1p(max(refinement_t, 0.0) * 5.0)
    score -= 0.50 * math.log1p(max(refinement_r, 0.0) / 5.0)
    if frame_count == 3:
        score -= 0.5 * math.log1p(max(finite(row.get("one_three_translation_consistency_m"), 5.0), 0.0))
        score -= 0.25 * math.log1p(max(finite(row.get("one_three_rotation_consistency_deg"), 180.0), 0.0) / 5.0)
    return float(score)


def threshold_grid(*arrays: np.ndarray) -> np.ndarray:
    values = np.concatenate([np.asarray(array, dtype=np.float64) for array in arrays])
    values = values[np.isfinite(values)]
    if not len(values):
        return np.asarray([0.5])
    quantiles = np.quantile(values, np.linspace(0.05, 0.95, 11))
    return np.unique(np.concatenate([quantiles, [values.min() - 1e-5, values.max() + 1e-5]]))


def execute_strategy(
    cases: list[dict],
    descriptor: str,
    strategy: str,
    accept_score: np.ndarray,
    wait_score: np.ndarray,
    three_score: np.ndarray,
    thresholds: tuple[float, float, float],
) -> list[dict]:
    accept_threshold, wait_threshold, three_threshold = thresholds
    outcomes = []
    for index, case in enumerate(cases):
        fixed = case["baselines"]["fixed_explicit"]
        one = case["variants"].get(candidate_key(descriptor, strategy, 1))
        three = case["variants"].get(candidate_key(descriptor, strategy, 3))
        if accept_score[index] >= accept_threshold and accept_score[index] >= wait_score[index]:
            action, selected, lookahead = "accept_1f", one, 1
        elif wait_score[index] >= wait_threshold:
            if three_score[index] >= three_threshold:
                action, selected, lookahead = "wait_accept_3f", three, 3
            else:
                action, selected, lookahead = "wait_fallback", fixed, 3
        else:
            action, selected, lookahead = "fallback_1f", fixed, 1
        if selected is None or selected.get("fit_failed", False):
            action, selected = "invalid_fallback", fixed
        accepted = action in ("accept_1f", "wait_accept_3f")
        false_accept = accepted and (is_catastrophic(selected) or joint_cost(selected) >= joint_cost(fixed))
        outcomes.append(
            {
                "case_name": case["case_name"],
                "source": source(case),
                "action": action,
                "lookahead": lookahead,
                "accepted": accepted,
                "false_accept": bool(false_accept),
                "selected": selected,
                "fixed": fixed,
                "one": one,
                "three": three,
                "accept_score": float(accept_score[index]),
                "wait_score": float(wait_score[index]),
                "three_score": float(three_score[index]),
            }
        )
    return outcomes


def outcome_objective(outcomes: list[dict]) -> float:
    final_cost = np.mean([joint_cost(row["selected"]) for row in outcomes])
    false_accept = np.mean([row["false_accept"] for row in outcomes])
    wait_rate = np.mean([row["lookahead"] == 3 for row in outcomes])
    catastrophic = np.mean([is_catastrophic(row["selected"]) for row in outcomes])
    return float(final_cost + 6.0 * false_accept + 0.15 * wait_rate + catastrophic)


def tune_thresholds(
    cases: list[dict],
    descriptor: str,
    strategy: str,
    accept_score: np.ndarray,
    wait_score: np.ndarray,
    three_score: np.ndarray,
) -> tuple[tuple[float, float, float], dict]:
    grids = [threshold_grid(accept_score), threshold_grid(wait_score), threshold_grid(three_score)]
    best = None
    fixed_cost = float(np.mean([joint_cost(case["baselines"]["fixed_explicit"]) for case in cases]))
    for accept_threshold in grids[0]:
        for wait_threshold in grids[1]:
            for three_threshold in grids[2]:
                thresholds = (float(accept_threshold), float(wait_threshold), float(three_threshold))
                outcomes = execute_strategy(
                    cases, descriptor, strategy, accept_score, wait_score, three_score, thresholds
                )
                objective = outcome_objective(outcomes)
                summary = summarize_outcomes(outcomes)
                false_accept = summary["false_accept_rate_all"]
                false_accept_accepted = summary["false_accept_rate_accepted"]
                if false_accept > 0.05:
                    objective += 15.0 * (false_accept - 0.05)
                if false_accept_accepted > 0.10:
                    objective += 30.0 * (false_accept_accepted - 0.10)
                if summary["joint_cost_mean"] > fixed_cost:
                    objective += 8.0 * (summary["joint_cost_mean"] - fixed_cost)
                current = (objective, thresholds, outcomes)
                if best is None or current[0] < best[0]:
                    best = current
    assert best is not None
    return best[1], summarize_outcomes(best[2])


def stats(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "max")}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def summarize_rows(rows: list[dict | None]) -> dict:
    valid = [candidate_error(row) for row in rows]
    valid_finite = [(t, r) for t, r in valid if math.isfinite(t) and math.isfinite(r)]
    total = len(rows)
    return {
        "case_count": total,
        "fit_failure_rate": float(1.0 - len(valid_finite) / max(total, 1)),
        "translation_m": stats(t for t, _ in valid_finite),
        "rotation_deg": stats(r for _, r in valid_finite),
        "rates": {
            "strict_success": float(sum(t < 0.10 and r < 2.0 for t, r in valid_finite) / max(total, 1)),
            "success": float(sum(t < 0.25 and r < 5.0 for t, r in valid_finite) / max(total, 1)),
            "catastrophic": float(sum(is_catastrophic(row) for row in rows) / max(total, 1)),
        },
        "joint_cost_mean": float(np.mean([joint_cost(row) for row in rows])),
    }


def summarize_outcomes(outcomes: list[dict]) -> dict:
    selected = [row["selected"] for row in outcomes]
    accepted = [row for row in outcomes if row["accepted"]]
    fallback = [row for row in outcomes if not row["accepted"]]
    return {
        **summarize_rows(selected),
        "coverage": float(len(accepted) / max(len(outcomes), 1)),
        "false_accept_rate_all": float(np.mean([row["false_accept"] for row in outcomes])),
        "false_accept_rate_accepted": float(np.mean([row["false_accept"] for row in accepted])) if accepted else 0.0,
        "wait_rate": float(np.mean([row["lookahead"] == 3 for row in outcomes])),
        "actions": {
            action: int(sum(row["action"] == action for row in outcomes))
            for action in sorted({row["action"] for row in outcomes})
        },
        "accepted_improvement_joint": float(
            np.mean([joint_cost(row["fixed"]) - joint_cost(row["selected"]) for row in accepted])
        )
        if accepted
        else 0.0,
        "fallback_identity_error": float(
            np.mean([abs(joint_cost(row["selected"]) - joint_cost(row["fixed"])) < 1e-8 for row in fallback])
        )
        if fallback
        else 1.0,
        "mean_lookahead_frames": float(np.mean([row["lookahead"] for row in outcomes])),
    }


def direct_method(cases: list[dict], descriptor: str, strategy: str, method: str) -> list[dict | None]:
    if method == "fixed_explicit":
        return [case["baselines"]["fixed_explicit"] for case in cases]
    if method == "boundary_oracle":
        return [case["baselines"]["boundary_oracle"] for case in cases]
    frame_count = 1 if method == "world_memory_1f" else 3
    return [case["variants"].get(candidate_key(descriptor, strategy, frame_count)) for case in cases]


def oracle_rows(cases: list[dict], descriptor: str, strategy: str) -> tuple[list[dict], list[str]]:
    selected, actions = [], []
    for case in cases:
        options = {
            "fixed": case["baselines"]["fixed_explicit"],
            "world_1f": case["variants"].get(candidate_key(descriptor, strategy, 1)),
            "world_3f": case["variants"].get(candidate_key(descriptor, strategy, 3)),
        }
        name = min(options, key=lambda key: (is_catastrophic(options[key]), joint_cost(options[key])))
        selected.append(options[name])
        actions.append(name)
    return selected, actions


def oracle_all_rows(cases: list[dict]) -> tuple[list[dict], list[str]]:
    selected, actions = [], []
    for case in cases:
        options = {"fixed": case["baselines"]["fixed_explicit"], **case["variants"]}
        name = min(options, key=lambda key: (is_catastrophic(options[key]), joint_cost(options[key])))
        selected.append(options[name])
        actions.append(name)
    return selected, actions


def risk_coverage(
    cases: list[dict],
    descriptor: str,
    strategy: str,
    scores: np.ndarray,
) -> list[dict]:
    candidates = [case["variants"].get(candidate_key(descriptor, strategy, 3)) for case in cases]
    order = np.argsort(-scores)
    output = []
    previous_risk = -float("inf")
    monotonic_steps = []
    for coverage in np.linspace(0.10, 1.0, 10):
        count = max(1, int(math.ceil(len(cases) * coverage)))
        ids = order[:count]
        selected = [candidates[index] for index in ids]
        fixed = [cases[index]["baselines"]["fixed_explicit"] for index in ids]
        risk = float(np.mean([joint_cost(row) for row in selected]))
        catastrophic = float(np.mean([is_catastrophic(row) for row in selected]))
        false_accept = float(
            np.mean([is_catastrophic(candidate) or joint_cost(candidate) >= joint_cost(base) for candidate, base in zip(selected, fixed)])
        )
        improvement = float(np.mean([joint_cost(base) - joint_cost(candidate) for candidate, base in zip(selected, fixed)]))
        monotonic_steps.append(risk >= previous_risk - 1e-8)
        previous_risk = risk
        output.append(
            {
                "coverage": float(coverage),
                "accepted_count": count,
                "risk_joint_cost": risk,
                "catastrophic_rate": catastrophic,
                "false_accept_rate": false_accept,
                "improvement_vs_fixed_joint": improvement,
            }
        )
    for row in output:
        row["risk_curve_monotonic_fraction"] = float(np.mean(monotonic_steps[1:]))
    return output


def train_scores(
    train_cases: list[dict],
    test_cases: list[dict],
    descriptor: str,
    strategy: str,
    feature_names: tuple[str, ...],
    device: torch.device,
    steps: int,
    seed: int,
) -> dict:
    train_one = feature_matrix(train_cases, descriptor, strategy, 1, feature_names)
    train_three = feature_matrix(train_cases, descriptor, strategy, 3, feature_names)
    test_one = feature_matrix(test_cases, descriptor, strategy, 1, feature_names)
    test_three = feature_matrix(test_cases, descriptor, strategy, 3, feature_names)
    transformer_one, transformer_three = Transformer.fit(train_one), Transformer.fit(train_three)
    train_one_x, test_one_x = transformer_one.apply(train_one), transformer_one.apply(test_one)
    train_three_x, test_three_x = transformer_three.apply(train_three), transformer_three.apply(test_three)
    train_fixed = [case["baselines"]["fixed_explicit"] for case in train_cases]
    train_one_rows = [case["variants"].get(candidate_key(descriptor, strategy, 1)) for case in train_cases]
    train_three_rows = [case["variants"].get(candidate_key(descriptor, strategy, 3)) for case in train_cases]
    test_fixed = [case["baselines"]["fixed_explicit"] for case in test_cases]
    test_one_rows = [case["variants"].get(candidate_key(descriptor, strategy, 1)) for case in test_cases]
    test_three_rows = [case["variants"].get(candidate_key(descriptor, strategy, 3)) for case in test_cases]
    train_accept = np.asarray([helpful(row, base) for row, base in zip(train_one_rows, train_fixed)], dtype=np.float32)
    train_three_help = np.asarray([helpful(row, base) for row, base in zip(train_three_rows, train_fixed)], dtype=np.float32)
    train_wait = train_three_help * (1.0 - train_accept)
    test_accept = np.asarray([helpful(row, base) for row, base in zip(test_one_rows, test_fixed)], dtype=np.float32)
    test_three_help = np.asarray([helpful(row, base) for row, base in zip(test_three_rows, test_fixed)], dtype=np.float32)
    test_wait = test_three_help * (1.0 - test_accept)
    accept_model = fit_linear(train_one_x, train_accept, device, steps, seed)
    wait_model = fit_linear(train_one_x, train_wait, device, steps, seed + 1)
    three_model = fit_linear(train_three_x, train_three_help, device, steps, seed + 2)
    train_scores = {
        "accept": accept_model.predict(train_one_x),
        "wait": wait_model.predict(train_one_x),
        "three": three_model.predict(train_three_x),
    }
    test_scores = {
        "accept": accept_model.predict(test_one_x),
        "wait": wait_model.predict(test_one_x),
        "three": three_model.predict(test_three_x),
    }
    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_labels": {"accept": train_accept, "wait": train_wait, "three": train_three_help},
        "test_labels": {"accept": test_accept, "wait": test_wait, "three": test_three_help},
        "auroc": {
            "accept": rank_auc(test_accept, test_scores["accept"]),
            "wait": rank_auc(test_wait, test_scores["wait"]),
            "three": rank_auc(test_three_help, test_scores["three"]),
        },
        "weights": {
            "accept": dict(zip(feature_names, accept_model.weight.astype(float).tolist())),
            "wait": dict(zip(feature_names, wait_model.weight.astype(float).tolist())),
            "three": dict(zip(feature_names, three_model.weight.astype(float).tolist())),
        },
    }


def train_tertiles(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return 0.0, 0.0
    return tuple(float(value) for value in np.percentile(array, [33.333, 66.667]))


def tertile_label(value: float, thresholds: tuple[float, float], names=("low", "medium", "high")) -> str:
    if not math.isfinite(value):
        return "unknown"
    return names[0] if value <= thresholds[0] else names[1] if value <= thresholds[1] else names[2]


def evaluation_groups(
    train_cases: list[dict],
    test_cases: list[dict],
    descriptor: str,
    strategy: str,
    fixed_rows: list[dict],
    one_rows: list[dict | None],
    three_rows: list[dict | None],
    oracle: list[dict],
    models: dict,
    complete_name: str,
) -> dict:
    def row_value(case: dict, name: str) -> float:
        row = case["variants"].get(candidate_key(descriptor, strategy, 3))
        if name == "texture":
            return finite(case.get("texture_score"))
        if name == "overlap":
            return finite(row.get("global_top1_score")) if row is not None else float("nan")
        if name == "geometry":
            return nested(row or {}, "target_geometry", "planarity_ratio")
        raise KeyError(name)

    thresholds = {
        name: train_tertiles([row_value(case, name) for case in train_cases])
        for name in ("texture", "overlap", "geometry")
    }
    labels = {
        "texture": [tertile_label(row_value(case, "texture"), thresholds["texture"]) for case in test_cases],
        "overlap": [tertile_label(row_value(case, "overlap"), thresholds["overlap"]) for case in test_cases],
        "geometry": [
            tertile_label(
                row_value(case, "geometry"), thresholds["geometry"], ("planar", "medium", "nondegenerate")
            )
            for case in test_cases
        ],
        "angle_bucket": [str(case["record"].get("angle_bucket", "unknown")) for case in test_cases],
    }
    method_rows = {
        "fixed_explicit": fixed_rows,
        "world_memory_1f": one_rows,
        "world_memory_3f": three_rows,
        "oracle_select": oracle,
    }
    for model_name in ("hand_rules", "geometry", "token", "geometry_token", complete_name):
        actions = models[model_name]["case_actions"]
        method_rows[f"strategy_{model_name}"] = [
            {
                "fit_failed": not math.isfinite(row["selected_translation_m"]),
                "camera_translation_error_m": row["selected_translation_m"],
                "camera_rotation_error_deg": row["selected_rotation_deg"],
            }
            for row in actions
        ]
    output = {}
    for group_name, group_labels in labels.items():
        output[group_name] = {}
        for label in sorted(set(group_labels)):
            ids = [index for index, value in enumerate(group_labels) if value == label]
            output[group_name][label] = {
                method: summarize_rows([rows[index] for index in ids]) for method, rows in method_rows.items()
            }
    return output


def evaluate_fold(
    train_cases: list[dict],
    test_cases: list[dict],
    held_out: str,
    available: list[tuple[str, str]],
    device: torch.device,
    steps: int,
    seed: int,
) -> dict:
    descriptor, strategy, config_meta = select_config(train_cases, available)
    fixed_rows = direct_method(test_cases, descriptor, strategy, "fixed_explicit")
    one_rows = direct_method(test_cases, descriptor, strategy, "world_memory_1f")
    three_rows = direct_method(test_cases, descriptor, strategy, "world_memory_3f")
    oracle, oracle_actions = oracle_rows(test_cases, descriptor, strategy)
    oracle_all, oracle_all_actions = oracle_all_rows(test_cases)
    feature_sets = {
        "geometry": GEOMETRY_FEATURES,
        "token": TOKEN_FEATURES,
        "geometry_token": GEOMETRY_FEATURES + TOKEN_FEATURES,
        "geometry_torso": GEOMETRY_FEATURES + TORSO_FEATURES,
        "geometry_human": GEOMETRY_FEATURES + HUMAN_FEATURES,
        "geometry_gravity": GEOMETRY_FEATURES + GRAVITY_FEATURES,
        "geometry_human_gravity": GEOMETRY_FEATURES + HUMAN_FEATURES + TORSO_FEATURES + GRAVITY_FEATURES,
    }
    models = {}
    for index, (name, feature_names) in enumerate(feature_sets.items()):
        fitted = train_scores(
            train_cases,
            test_cases,
            descriptor,
            strategy,
            feature_names,
            device,
            steps,
            seed + index * 100,
        )
        thresholds, train_summary = tune_thresholds(
            train_cases,
            descriptor,
            strategy,
            fitted["train_scores"]["accept"],
            fitted["train_scores"]["wait"],
            fitted["train_scores"]["three"],
        )
        outcomes = execute_strategy(
            test_cases,
            descriptor,
            strategy,
            fitted["test_scores"]["accept"],
            fitted["test_scores"]["wait"],
            fitted["test_scores"]["three"],
            thresholds,
        )
        models[name] = {
            "feature_names": feature_names,
            "thresholds": thresholds,
            "train_summary": train_summary,
            "test_summary": summarize_outcomes(outcomes),
            "auroc": fitted["auroc"],
            "weights": fitted["weights"],
            "risk_coverage": risk_coverage(
                test_cases, descriptor, strategy, fitted["test_scores"]["three"]
            ),
            "case_actions": [
                {
                    **{key: row[key] for key in ("case_name", "action", "lookahead", "accepted", "false_accept")},
                    "selected_translation_m": candidate_error(row["selected"])[0],
                    "selected_rotation_deg": candidate_error(row["selected"])[1],
                    "fixed_translation_m": candidate_error(row["fixed"])[0],
                    "fixed_rotation_deg": candidate_error(row["fixed"])[1],
                    "world3_translation_m": candidate_error(row["three"])[0],
                    "world3_rotation_deg": candidate_error(row["three"])[1],
                }
                for row in outcomes
            ],
        }
    train_hand_one = np.asarray(
        [hand_score(case, case["variants"].get(candidate_key(descriptor, strategy, 1)), 1) for case in train_cases]
    )
    train_hand_three = np.asarray(
        [hand_score(case, case["variants"].get(candidate_key(descriptor, strategy, 3)), 3) for case in train_cases]
    )
    test_hand_one = np.asarray(
        [hand_score(case, case["variants"].get(candidate_key(descriptor, strategy, 1)), 1) for case in test_cases]
    )
    test_hand_three = np.asarray(
        [hand_score(case, case["variants"].get(candidate_key(descriptor, strategy, 3)), 3) for case in test_cases]
    )
    train_fixed = [case["baselines"]["fixed_explicit"] for case in train_cases]
    train_one = [case["variants"].get(candidate_key(descriptor, strategy, 1)) for case in train_cases]
    train_three = [case["variants"].get(candidate_key(descriptor, strategy, 3)) for case in train_cases]
    hand_accept_label = np.asarray([helpful(row, base) for row, base in zip(train_one, train_fixed)])
    hand_three_label = np.asarray([helpful(row, base) for row, base in zip(train_three, train_fixed)])
    # Moderate one-frame geometry is used as the depth-free indication that waiting may help.
    train_hand_wait = -np.abs(train_hand_one - np.median(train_hand_one[hand_three_label])) if hand_three_label.any() else train_hand_one
    test_hand_wait = -np.abs(test_hand_one - np.median(train_hand_one[hand_three_label])) if hand_three_label.any() else test_hand_one
    hand_thresholds, hand_train_summary = tune_thresholds(
        train_cases,
        descriptor,
        strategy,
        train_hand_one,
        train_hand_wait,
        train_hand_three,
    )
    hand_outcomes = execute_strategy(
        test_cases,
        descriptor,
        strategy,
        test_hand_one,
        test_hand_wait,
        test_hand_three,
        hand_thresholds,
    )
    models["hand_rules"] = {
        "thresholds": hand_thresholds,
        "train_summary": hand_train_summary,
        "test_summary": summarize_outcomes(hand_outcomes),
        "auroc": {
            "accept": rank_auc(
                np.asarray(
                    [
                        helpful(case["variants"].get(candidate_key(descriptor, strategy, 1)), case["baselines"]["fixed_explicit"])
                        for case in test_cases
                    ]
                ),
                test_hand_one,
            ),
            "three": rank_auc(
                np.asarray(
                    [
                        helpful(case["variants"].get(candidate_key(descriptor, strategy, 3)), case["baselines"]["fixed_explicit"])
                        for case in test_cases
                    ]
                ),
                test_hand_three,
            ),
        },
        "risk_coverage": risk_coverage(test_cases, descriptor, strategy, test_hand_three),
        "case_actions": [
            {
                **{key: row[key] for key in ("case_name", "action", "lookahead", "accepted", "false_accept")},
                "selected_translation_m": candidate_error(row["selected"])[0],
                "selected_rotation_deg": candidate_error(row["selected"])[1],
                "fixed_translation_m": candidate_error(row["fixed"])[0],
                "fixed_rotation_deg": candidate_error(row["fixed"])[1],
                "world3_translation_m": candidate_error(row["three"])[0],
                "world3_rotation_deg": candidate_error(row["three"])[1],
            }
            for row in hand_outcomes
        ],
    }
    complete_candidates = ("geometry", "geometry_token", "geometry_human_gravity", "hand_rules")
    complete_name = min(
        complete_candidates,
        key=lambda name: models[name]["train_summary"]["joint_cost_mean"]
        + 6.0 * models[name]["train_summary"]["false_accept_rate_all"],
    )
    anchor_comparison = {}
    for anchor_strategy in ANCHOR_STRATEGIES:
        candidates_for_strategy = [item for item in available if item[1] == anchor_strategy]
        icp_descriptors = sorted(
            descriptor_name
            for descriptor_name, _ in candidates_for_strategy
            if descriptor_name.startswith("explicit_icp_")
        )
        if icp_descriptors:
            local_descriptor = icp_descriptors[0]
        else:
            local_descriptor, _, _ = select_config(train_cases, candidates_for_strategy)
        rows = [
            case["variants"].get(candidate_key(local_descriptor, anchor_strategy, 3)) for case in test_cases
        ]
        anchor_comparison[anchor_strategy] = {
            "descriptor": local_descriptor,
            "metrics": summarize_rows(rows),
            "helpful_rate": float(
                np.mean(
                    [
                        helpful(row, case["baselines"]["fixed_explicit"])
                        for row, case in zip(rows, test_cases)
                    ]
                )
            ),
        }
    groups = evaluation_groups(
        train_cases,
        test_cases,
        descriptor,
        strategy,
        fixed_rows,
        one_rows,
        three_rows,
        oracle,
        models,
        complete_name,
    )
    return {
        "held_out_source": held_out,
        "train_case_count": len(train_cases),
        "test_case_count": len(test_cases),
        "selected_config": {"descriptor": descriptor, "anchor_strategy": strategy, **config_meta},
        "baselines": {
            "fixed_explicit": summarize_rows(fixed_rows),
            "world_memory_1f": summarize_rows(one_rows),
            "world_memory_3f": summarize_rows(three_rows),
            "oracle_select": {
                **summarize_rows(oracle),
                "actions": {name: int(oracle_actions.count(name)) for name in sorted(set(oracle_actions))},
            },
            "oracle_all_candidates": {
                **summarize_rows(oracle_all),
                "actions": {
                    name: int(oracle_all_actions.count(name)) for name in sorted(set(oracle_all_actions))
                },
            },
            "boundary_oracle": summarize_rows(direct_method(test_cases, descriptor, strategy, "boundary_oracle")),
        },
        "models": models,
        "complete_strategy_model": complete_name,
        "complete_strategy": models[complete_name],
        "anchor_comparison": anchor_comparison,
        "groups": groups,
    }


def aggregate_folds(folds: dict) -> dict:
    methods = [
        "fixed_explicit",
        "world_memory_1f",
        "world_memory_3f",
        "oracle_select",
        "oracle_all_candidates",
        "boundary_oracle",
    ]
    output = {"baselines": {}, "models": {}, "anchor_comparison": {}}

    def available_average(values: list[float | None], weights: list[int]) -> float | None:
        pairs = [
            (float(value), int(weight))
            for value, weight in zip(values, weights)
            if value is not None and math.isfinite(float(value)) and weight > 0
        ]
        if not pairs:
            return None
        return float(np.average([value for value, _ in pairs], weights=[weight for _, weight in pairs]))

    for method in methods:
        output["baselines"][method] = {
            metric: available_average(
                [fold["baselines"][method][metric]["mean"] for fold in folds.values()],
                [fold["baselines"][method][metric]["count"] for fold in folds.values()],
            )
            for metric in ("translation_m", "rotation_deg")
        }
        output["baselines"][method]["catastrophic_rate"] = float(np.average(
            [fold["baselines"][method]["rates"]["catastrophic"] for fold in folds.values()],
            weights=[fold["test_case_count"] for fold in folds.values()],
        ))
        output["baselines"][method]["success_rate"] = float(np.average(
            [fold["baselines"][method]["rates"]["success"] for fold in folds.values()],
            weights=[fold["test_case_count"] for fold in folds.values()],
        ))
    model_names = sorted(set.intersection(*(set(fold["models"]) for fold in folds.values())))
    for name in model_names:
        output["models"][name] = {
            "translation_m": float(np.average(
                [fold["models"][name]["test_summary"]["translation_m"]["mean"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
            "rotation_deg": float(np.average(
                [fold["models"][name]["test_summary"]["rotation_deg"]["mean"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
            "catastrophic_rate": float(np.average(
                [fold["models"][name]["test_summary"]["rates"]["catastrophic"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
            "coverage": float(np.average(
                [fold["models"][name]["test_summary"]["coverage"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
            "false_accept_rate": float(np.average(
                [fold["models"][name]["test_summary"]["false_accept_rate_all"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
            "three_auroc": float(np.nanmean([fold["models"][name]["auroc"].get("three", np.nan) for fold in folds.values()])),
        }
    for strategy in ANCHOR_STRATEGIES:
        output["anchor_comparison"][strategy] = {
            "translation_m": available_average(
                [fold["anchor_comparison"][strategy]["metrics"]["translation_m"]["mean"] for fold in folds.values()],
                [fold["anchor_comparison"][strategy]["metrics"]["translation_m"]["count"] for fold in folds.values()],
            ),
            "rotation_deg": available_average(
                [fold["anchor_comparison"][strategy]["metrics"]["rotation_deg"]["mean"] for fold in folds.values()],
                [fold["anchor_comparison"][strategy]["metrics"]["rotation_deg"]["count"] for fold in folds.values()],
            ),
            "catastrophic_rate": float(np.average(
                [fold["anchor_comparison"][strategy]["metrics"]["rates"]["catastrophic"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
            "helpful_rate": float(np.average(
                [fold["anchor_comparison"][strategy]["helpful_rate"] for fold in folds.values()],
                weights=[fold["test_case_count"] for fold in folds.values()],
            )),
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    def fmt(value, digits: int) -> str:
        return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"

    lines = [
        "# V14 Depth-Free Selective World-Memory Relocalization",
        "",
        f"Cases: {report['case_count']}",
        "",
        "All candidate matching is depth-free and uses no GT correspondence. GT camera is used only for labels and evaluation.",
        "",
        "## LOSO Results",
        "",
        "| Held-out | Config | Fixed T/R | WM3 T/R | Oracle T/R | Geometry T/R | Token T/R | Complete |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for source_name, fold in report["folds"].items():
        config = fold["selected_config"]
        fixed = fold["baselines"]["fixed_explicit"]
        wm = fold["baselines"]["world_memory_3f"]
        oracle = fold["baselines"]["oracle_select"]
        geometry = fold["models"]["geometry"]["test_summary"]
        token = fold["models"]["token"]["test_summary"]
        complete = fold["complete_strategy"]["test_summary"]
        lines.append(
            f"| {source_name} | {config['descriptor']} + {config['anchor_strategy']} | "
            f"{fmt(fixed['translation_m']['mean'], 3)}/{fmt(fixed['rotation_deg']['mean'], 2)} | "
            f"{fmt(wm['translation_m']['mean'], 3)}/{fmt(wm['rotation_deg']['mean'], 2)} | "
            f"{fmt(oracle['translation_m']['mean'], 3)}/{fmt(oracle['rotation_deg']['mean'], 2)} | "
            f"{fmt(geometry['translation_m']['mean'], 3)}/{fmt(geometry['rotation_deg']['mean'], 2)} | "
            f"{fmt(token['translation_m']['mean'], 3)}/{fmt(token['rotation_deg']['mean'], 2)} | "
            f"{fold['complete_strategy_model']} ({complete['coverage'] * 100:.1f}% accept) |"
        )
    lines.extend(["", "## Aggregate", ""])
    lines.append("| Method | T mean | R mean | Catastrophic | Success/Coverage | False accept |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, row in report["aggregate"]["baselines"].items():
        lines.append(
            f"| {name} | {fmt(row['translation_m'], 4)} | {fmt(row['rotation_deg'], 3)} | "
            f"{100 * row['catastrophic_rate']:.1f}% | {100 * row['success_rate']:.1f}% success | - |"
        )
    for name, row in report["aggregate"]["models"].items():
        lines.append(
            f"| {name} | {row['translation_m']:.4f} | {row['rotation_deg']:.3f} | "
            f"{100 * row['catastrophic_rate']:.1f}% | {100 * row['coverage']:.1f}% coverage | "
            f"{100 * row['false_accept_rate']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V14 calibrator training requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.input_dir)
    available = configs(cases)
    device = torch.device(args.device)
    folds = {}
    for index, held_out in enumerate(SOURCES):
        train_cases = [case for case in cases if source(case) != held_out]
        test_cases = [case for case in cases if source(case) == held_out]
        if not test_cases:
            raise RuntimeError(f"No held-out cases for {held_out}")
        folds[held_out] = evaluate_fold(
            train_cases,
            test_cases,
            held_out,
            available,
            device,
            args.steps,
            args.seed + index * 10000,
        )
        print(
            f">> {held_out}: config={folds[held_out]['selected_config']['descriptor']}+"
            f"{folds[held_out]['selected_config']['anchor_strategy']} "
            f"complete={folds[held_out]['complete_strategy_model']}",
            flush=True,
        )
    report = {
        "experiment": "V14 Depth-Free Selective World-Memory Relocalization",
        "case_count": len(cases),
        "sources": {name: int(sum(source(case) == name for case in cases)) for name in SOURCES},
        "protocol": {
            "leave_one_source_out": True,
            "gt_depth_used": False,
            "gt_correspondence_used": False,
            "gt_camera_use": "training labels and final evaluation only",
            "human3r_frozen": True,
            "shot_level_transform": True,
        },
        "folds": folds,
    }
    report["aggregate"] = aggregate_folds(folds)
    (args.output_dir / "v14_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "v14_summary.md", report)
    print(f">> wrote {args.output_dir / 'v14_eval.json'}", flush=True)


if __name__ == "__main__":
    main()
