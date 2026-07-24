#!/usr/bin/env python3
"""V13 Phase 2: training-free multi-human Boundary fusion analysis.

This evaluator reuses the strict GT-ID candidates and raw caches produced by
Phase 1. It does not rerun Human3R and changes only the fusion of per-human
rotation and translation candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13.gt_id_consensus import (  # noqa: E402
    evaluate_solution,
    finite_distribution,
    jsonable,
    make_transform,
    method_solutions,
    rotation_error_deg,
    so3_mean,
    translation_candidates,
)


DEFAULT_PHASE1 = (
    ROOT
    / "output/v20_phase1_gt_id_multihuman_consensus"
    / "v20_phase1_gtid_v2_offsets_0_1_2_4_8.json"
)
DEFAULT_CACHE = ROOT / "output/v20_phase1_gt_id_multihuman_consensus/case_cache"
CAMERA_METRICS = (
    "camera_translation_error_m",
    "camera_rotation_error_deg",
    "camera_composite",
)
FULL_METRICS = CAMERA_METRICS + (
    "human_root_error_m",
    "human_joint_error_m",
    "human_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1_report", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "output/v13/phase2_fusion"
    )
    parser.add_argument(
        "--dev_timestamps", type=int, nargs="+", default=(500, 700, 900)
    )
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def as_candidates(payload: dict) -> dict[str, dict]:
    array_fields = (
        "rotation",
        "translation",
        "anchor",
        "post_root",
        "post_torso",
        "target_torso",
        "root_velocity_m_per_frame",
    )
    output = {}
    for identity, source in payload.items():
        row = dict(source)
        for field in array_fields:
            if field in row:
                row[field] = np.asarray(row[field], dtype=np.float64)
        output[identity] = row
    return output


def strict_cache_from_report(cache: dict, assignment: list[dict]) -> dict:
    """Apply the saved v2 GT-ID assignment without re-reading GT meshes."""
    if len(cache["humans"]) != len(assignment):
        raise ValueError("Phase 1 cache/report frame count mismatch")
    humans = []
    for old_humans, assignment_row in zip(cache["humans"], assignment):
        detections = {
            int(row["detection_index"]): dict(row) for row in old_humans.values()
        }
        assigned = {}
        for item in assignment_row.get("assignments", []):
            detection = int(item["detection_index"])
            if detection not in detections:
                raise KeyError(f"Missing detection {detection} in Phase 1 cache")
            identity = str(item["identity"])
            row = dict(detections[detection])
            row["identity"] = identity
            assigned[identity] = row
        humans.append(assigned)
    output = dict(cache)
    output["humans"] = humans
    output["assignment"] = assignment
    return output


def camera_evaluation(cache: dict, solution: dict) -> dict:
    boundary = make_transform(solution["rotation"], solution["translation"])
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    target = gauge @ gt_post
    final = boundary @ post_pose
    translation = float(np.linalg.norm(final[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(final, target)
    return {
        "camera_translation_error_m": translation,
        "camera_rotation_error_deg": rotation,
        "camera_composite": translation + 0.02 * rotation,
        "catastrophic": bool(translation > 2.0 or rotation > 45.0),
    }


def normalize_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 1e-6)
    return values / values.sum()


def rotation_distances(rotations: list[np.ndarray], center: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            math.degrees(
                np.linalg.norm(Rotation.from_matrix(center.T @ rotation).as_rotvec())
            )
            for rotation in rotations
        ],
        dtype=np.float64,
    )


def candidate_features(
    candidates: dict[str, dict], identities: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    rotations = [np.asarray(candidates[i]["rotation"]) for i in identities]
    translations = np.stack([candidates[i]["translation"] for i in identities])
    rotation_center = so3_mean(rotations)
    translation_center = translations.mean(axis=0)
    shared_translations = translation_candidates(candidates, identities, rotation_center)
    shared_center = shared_translations.mean(axis=0)
    rotation_deviation = rotation_distances(rotations, rotation_center)
    translation_deviation = np.linalg.norm(translations - translation_center, axis=1)
    shared_translation_deviation = np.linalg.norm(
        shared_translations - shared_center, axis=1
    )
    output = {}
    for index, identity in enumerate(identities):
        candidate = candidates[identity]
        layout = []
        for other in identities:
            if identity == other:
                continue
            layout.append(
                np.linalg.norm(
                    (candidate["anchor"] - candidates[other]["anchor"])
                    - rotation_center
                    @ (candidate["post_root"] - candidates[other]["post_root"])
                )
            )
        output[identity] = {
            "quality": float(candidate["quality"]),
            "score": float(max(candidate["post_score"], 1e-6)),
            "completeness": float(max(candidate["post_completeness"], 1e-6)),
            "motion_dispersion_m": float(candidate["motion_dispersion_m"]),
            "angular_speed_deg_per_frame": float(
                candidate.get("torso_motion", {}).get(
                    "angular_speed_deg_per_frame", 0.0
                )
            ),
            "rotation_deviation_deg": float(rotation_deviation[index]),
            "translation_deviation_m": float(translation_deviation[index]),
            "shared_translation_deviation_m": float(
                shared_translation_deviation[index]
            ),
            "layout_residual_m": float(np.mean(layout)) if layout else 0.0,
        }
    return output


def soft_weight_rules(
    features: dict[str, dict[str, float]], identities: tuple[str, ...]
) -> dict[str, np.ndarray]:
    def values(name: str) -> np.ndarray:
        return np.asarray([features[i][name] for i in identities], dtype=np.float64)

    quality = np.maximum(values("quality"), 1e-8)
    score = np.maximum(values("score"), 1e-8)
    completeness = np.maximum(values("completeness"), 1e-4)
    motion = values("motion_dispersion_m")
    angular = values("angular_speed_deg_per_frame")
    rotation_dev = values("rotation_deviation_deg")
    translation_dev = values("translation_deviation_m")
    shared_translation_dev = values("shared_translation_deviation_m")
    layout = values("layout_residual_m")
    rules = {
        "quality_p025": quality**0.25,
        "quality_p050": quality**0.50,
        "quality_p100": quality,
        "visibility_soft": score**0.25 * completeness**0.50,
    }
    for tau in (0.02, 0.05, 0.10):
        rules[f"motion_tau{int(1000 * tau):03d}"] = np.exp(-motion / tau)
    for tau in (5.0, 10.0, 20.0):
        rules[f"rotation_consensus_tau{int(tau):02d}"] = np.exp(
            -rotation_dev / tau
        )
    for tau in (0.10, 0.25, 0.50):
        rules[f"translation_consensus_tau{int(100 * tau):02d}"] = np.exp(
            -translation_dev / tau
        )
        rules[f"shared_translation_consensus_tau{int(100 * tau):02d}"] = np.exp(
            -shared_translation_dev / tau
        )
        rules[f"layout_tau{int(100 * tau):02d}"] = np.exp(-layout / tau)
    rules["motion_angular_soft"] = np.exp(-motion / 0.05 - angular / 2.0)
    rules["rt_consensus_soft"] = np.exp(
        -rotation_dev / 10.0 - translation_dev / 0.25
    )
    rules["quality_rt_soft"] = quality**0.25 * np.exp(
        -rotation_dev / 10.0 - translation_dev / 0.25
    )
    rules["visibility_motion_soft"] = (
        score**0.25
        * completeness**0.50
        * np.exp(-motion / 0.05 - angular / 2.0)
    )
    return {name: normalize_weights(weight) for name, weight in rules.items()}


def weighted_solution(
    candidates: dict[str, dict],
    identities: tuple[str, ...],
    weights: np.ndarray,
    translation_mode: str,
) -> dict:
    rotation = so3_mean([candidates[i]["rotation"] for i in identities], weights)
    if translation_mode == "raw":
        translations = np.stack([candidates[i]["translation"] for i in identities])
    elif translation_mode == "shared":
        translations = translation_candidates(candidates, identities, rotation)
    else:
        raise ValueError(translation_mode)
    translation = np.average(translations, axis=0, weights=weights)
    return {
        "rotation": rotation,
        "translation": translation,
        "identities": identities,
        "weights": {i: float(w) for i, w in zip(identities, weights)},
    }


def phase2_solutions(
    candidates: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, dict[str, float]]]:
    identities = tuple(sorted(candidates))
    base = method_solutions(candidates)
    output = {
        name: base[name]
        for name in (
            "single_first",
            "single_largest",
            "single_highest_confidence",
            "naive_mean",
            "shared_rotation_mean",
            "confidence_weighted",
        )
        if name in base
    }
    single_names = [f"single_{identity}" for identity in identities]
    for name in single_names:
        output[name] = base[name]
    if len(identities) < 2:
        return output, candidate_features(candidates, identities)

    confidence_identity = max(
        identities, key=lambda identity: candidates[identity]["quality"]
    )
    confidence_rotation = candidates[confidence_identity]["rotation"]
    mean_rotation = so3_mean([candidates[i]["rotation"] for i in identities])
    output["translation_only_consensus"] = {
        "rotation": confidence_rotation,
        "translation": np.mean(
            translation_candidates(candidates, identities, confidence_rotation), axis=0
        ),
        "identities": identities,
    }
    output["rotation_only_consensus"] = {
        "rotation": mean_rotation,
        "translation": (
            candidates[confidence_identity]["anchor"]
            - mean_rotation @ candidates[confidence_identity]["post_root"]
        ),
        "identities": identities,
    }
    features = candidate_features(candidates, identities)
    for rule_name, weights in soft_weight_rules(features, identities).items():
        for translation_mode in ("raw", "shared"):
            name = f"soft_{rule_name}_{translation_mode}_t"
            output[name] = weighted_solution(
                candidates, identities, weights, translation_mode
            )
    return output, features


def compact_full_evaluation(value: dict) -> dict:
    keep = FULL_METRICS + ("catastrophic", "per_person")
    return {key: value[key] for key in keep}


def camera_summary(rows: list[dict]) -> dict:
    return {
        "valid_cases": len(rows),
        **{
            metric: finite_distribution([float(row[metric]) for row in rows])
            for metric in CAMERA_METRICS
        },
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows]))
        if rows
        else float("nan"),
    }


def full_summary(rows: list[dict]) -> dict:
    return {
        "valid_cases": len(rows),
        **{
            metric: finite_distribution([float(row[metric]) for row in rows])
            for metric in FULL_METRICS
        },
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows]))
        if rows
        else float("nan"),
    }


def paired(rows: list[dict], first: str, second: str) -> dict:
    common = [row for row in rows if first in row["camera"] and second in row["camera"]]
    output = {"valid_cases": len(common)}
    for metric in CAMERA_METRICS:
        a = np.asarray([row["camera"][first][metric] for row in common])
        b = np.asarray([row["camera"][second][metric] for row in common])
        delta = b - a
        try:
            p = float(wilcoxon(b, a).pvalue) if len(a) > 1 else float("nan")
        except ValueError:
            p = 1.0
        output[metric] = {
            "first_mean": float(a.mean()) if len(a) else float("nan"),
            "second_mean": float(b.mean()) if len(b) else float("nan"),
            "second_minus_first_mean": float(delta.mean()) if len(delta) else float("nan"),
            "second_improvement_rate": float(np.mean(delta < 0.0))
            if len(delta)
            else float("nan"),
            "second_harmful_rate": float(np.mean(delta > 0.0))
            if len(delta)
            else float("nan"),
            "wilcoxon_p": p,
        }
    return output


def safe_spearman(x: list[float], y: list[float]) -> dict:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.ptp(first[valid]) == 0 or np.ptp(second[valid]) == 0:
        return {"n": int(valid.sum()), "rho": float("nan"), "p": float("nan")}
    result = spearmanr(first[valid], second[valid])
    return {"n": int(valid.sum()), "rho": float(result.statistic), "p": float(result.pvalue)}


def split_rows(rows: list[dict], dev_timestamps: set[int]) -> tuple[list[dict], list[dict]]:
    dev = [row for row in rows if int(row["case"]["timestamp"]) in dev_timestamps]
    test = [row for row in rows if int(row["case"]["timestamp"]) not in dev_timestamps]
    return dev, test


def select_rule(dev_rows: list[dict], rule_names: list[str]) -> tuple[str, dict]:
    candidates = []
    for name in rule_names:
        common = [
            row
            for row in dev_rows
            if "naive_mean" in row["camera"] and name in row["camera"]
        ]
        naive = camera_summary([row["camera"]["naive_mean"] for row in common])
        summary = camera_summary([row["camera"][name] for row in common])
        safe = (
            summary["camera_composite"]["p90"]
            <= naive["camera_composite"]["p90"] + 1e-12
            and summary["catastrophic_rate"] <= naive["catastrophic_rate"] + 1e-12
        )
        candidates.append(
            (not safe, summary["camera_composite"]["mean"], name, summary, naive)
        )
    _, _, winner, summary, naive = min(candidates)
    return winner, {
        "selection": "minimum dev mean composite subject to no worse dev P90/catastrophic; fallback to minimum mean",
        "winner": winner,
        "winner_dev_summary": summary,
        "naive_dev_summary": naive,
    }


def markdown_report(report: dict) -> str:
    summaries = report["main_summaries"]["all"]
    winner = report["rule_selection"]["winner"]
    names = [
        "single_highest_confidence",
        "oracle_best_single",
        "naive_mean",
        "translation_only_consensus",
        "rotation_only_consensus",
        "shared_rotation_mean",
        "confidence_weighted",
        winner,
    ]
    lines = [
        "# V13 Phase 2: Multi-Human Boundary Fusion Optimization",
        "",
        "Historical request name: V20 Phase 2.",
        "",
        "## Protocol",
        "",
        f"- Cases: {report['case_count']} strict GT-ID Phase 1 caches.",
        f"- Development timestamps: {report['split']['dev_timestamps']}.",
        f"- Held-out timestamps: {report['split']['test_timestamps']}.",
        "- Frozen Human3R + Hard Reset + Fixed Explicit + V16 + s=1.",
        "- DA3, VGGT, V11.4 scale and token Re-ID remain disabled.",
        "",
        "## All-Case Main Results",
        "",
        "| Method | N | T mean | R mean | Composite mean | Composite P90 | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in dict.fromkeys(names):
        if name not in summaries:
            continue
        row = summaries[name]
        lines.append(
            f"| {name} | {row['valid_cases']} | "
            f"{row['camera_translation_error_m']['mean']:.3f} | "
            f"{row['camera_rotation_error_deg']['mean']:.2f} | "
            f"{row['camera_composite']['mean']:.3f} | "
            f"{row['camera_composite']['p90']:.3f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% |"
        )
    heldout = report["main_summaries"]["test"]
    lines.extend(
        [
            "",
            "## Rule Selection",
            "",
            f"Development-selected soft rule: `{winner}`.",
            "",
            "| Held-out method | Composite mean | Composite P90 | Catastrophic |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("naive_mean", winner):
        row = heldout[name]
        lines.append(
            f"| {name} | {row['camera_composite']['mean']:.3f} | "
            f"{row['camera_composite']['p90']:.3f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% |"
        )
    decomposition = report["paired"]["decomposition"]
    lines.extend(
        [
            "",
            "## Decomposition",
            "",
            "Improvement rates below are paired against highest-confidence single:",
            "",
            f"- translation-only: {100.0 * decomposition['translation_only_consensus']['camera_composite']['second_improvement_rate']:.1f}%.",
            f"- rotation-only: {100.0 * decomposition['rotation_only_consensus']['camera_composite']['second_improvement_rate']:.1f}%.",
            f"- joint naive mean: {100.0 * decomposition['naive_mean']['camera_composite']['second_improvement_rate']:.1f}%.",
            "",
            "## Interpretation",
            "",
            report["decision"]["summary"],
            "",
            "Full rule grid, feature correlations, leave-one-out diagnostics and per-case results are in the JSON file.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase1 = json.loads(args.phase1_report.read_text(encoding="utf-8"))
    phase1_cases = phase1["cases"]
    if args.max_cases > 0:
        phase1_cases = phase1_cases[: args.max_cases]

    rows = []
    candidate_diagnostics = []
    leave_one_out = []
    number_rows: dict[int, list[dict]] = defaultdict(list)
    for index, source in enumerate(phase1_cases):
        case = source["case"]
        cache_path = args.cache_dir / f"{case['key']}.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache = strict_cache_from_report(cache, source["assignment"])
        candidates = as_candidates(source["candidates"])
        solutions, features = phase2_solutions(candidates)
        camera = {
            name: camera_evaluation(cache, solution)
            for name, solution in solutions.items()
        }
        single_names = [name for name in camera if name.startswith("single_person")]
        best_single = min(single_names, key=lambda name: camera[name]["camera_composite"])
        solutions["oracle_best_single"] = solutions[best_single]
        camera["oracle_best_single"] = camera[best_single]

        full_names = (
            "single_first",
            "single_largest",
            "single_highest_confidence",
            "oracle_best_single",
            "naive_mean",
            "shared_rotation_mean",
            "confidence_weighted",
            "translation_only_consensus",
            "rotation_only_consensus",
        )
        full = {
            name: compact_full_evaluation(evaluate_solution(cache, solutions[name]))
            for name in full_names
            if name in solutions
        }
        identities = tuple(sorted(candidates))
        for identity in identities:
            candidate_diagnostics.append(
                {
                    "case": case["key"],
                    "timestamp": int(case["timestamp"]),
                    "identity": identity,
                    "single_composite": camera[f"single_{identity}"]["camera_composite"],
                    "is_best_single": bool(best_single == f"single_{identity}"),
                    **features[identity],
                }
            )
        if len(identities) >= 2:
            for size in range(1, len(identities) + 1):
                for subset in combinations(identities, size):
                    uniform = np.full(len(subset), 1.0 / len(subset))
                    solution = weighted_solution(candidates, subset, uniform, "raw")
                    number_rows[size].append(camera_evaluation(cache, solution))
        if len(identities) == 3:
            all_error = camera["naive_mean"]["camera_composite"]
            for removed in identities:
                kept = tuple(identity for identity in identities if identity != removed)
                solution = weighted_solution(
                    candidates, kept, np.full(2, 0.5), "raw"
                )
                error = camera_evaluation(cache, solution)["camera_composite"]
                leave_one_out.append(
                    {
                        "case": case["key"],
                        "identity": removed,
                        "removal_gain": all_error - error,
                        **features[removed],
                    }
                )
        rows.append(
            {
                "case": case,
                "camera": camera,
                "full": full,
                "features": features,
                "candidate_dispersion": source["candidate_dispersion"],
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(phase1_cases):
            print(f">> Phase 2 evaluated {index + 1}/{len(phase1_cases)}", flush=True)

    dev_timestamps = set(args.dev_timestamps)
    dev_rows, test_rows = split_rows(rows, dev_timestamps)
    if not dev_rows or not test_rows:
        raise ValueError("Development and held-out splits must both be non-empty")
    rule_names = sorted(
        {
            name
            for row in rows
            for name in row["camera"]
            if name.startswith("soft_")
        }
    )
    winner, selection = select_rule(dev_rows, rule_names)

    # The selected soft rule receives the same full human evaluation as baselines.
    source_by_key = {row["case"]["key"]: row for row in phase1_cases}
    for index, row in enumerate(rows):
        source = source_by_key[row["case"]["key"]]
        cache = torch.load(
            args.cache_dir / f"{row['case']['key']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        cache = strict_cache_from_report(cache, source["assignment"])
        candidates = as_candidates(source["candidates"])
        solutions, _ = phase2_solutions(candidates)
        if winner in solutions:
            row["full"][winner] = compact_full_evaluation(
                evaluate_solution(cache, solutions[winner])
            )
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f">> Full selected-rule evaluation {index + 1}/{len(rows)}", flush=True)

    main_names = (
        "single_first",
        "single_largest",
        "single_highest_confidence",
        "oracle_best_single",
        "naive_mean",
        "shared_rotation_mean",
        "confidence_weighted",
        "translation_only_consensus",
        "rotation_only_consensus",
        winner,
    )

    def summaries(subset: list[dict]) -> dict:
        output = {}
        for name in main_names:
            values = [row["full"][name] for row in subset if name in row["full"]]
            output[name] = full_summary(values)
        return output

    rule_summaries = {
        name: {
            "all": camera_summary(
                [row["camera"][name] for row in rows if name in row["camera"]]
            ),
            "dev": camera_summary(
                [row["camera"][name] for row in dev_rows if name in row["camera"]]
            ),
            "test": camera_summary(
                [row["camera"][name] for row in test_rows if name in row["camera"]]
            ),
        }
        for name in rule_names
    }
    feature_names = [
        "quality",
        "score",
        "completeness",
        "motion_dispersion_m",
        "angular_speed_deg_per_frame",
        "rotation_deviation_deg",
        "translation_deviation_m",
        "shared_translation_deviation_m",
        "layout_residual_m",
    ]
    feature_diagnostics = {}
    for name in feature_names:
        values = [row[name] for row in candidate_diagnostics]
        errors = [row["single_composite"] for row in candidate_diagnostics]
        selectors = defaultdict(list)
        for row in candidate_diagnostics:
            selectors[row["case"]].append(row)
        if name in {"quality", "score", "completeness"}:
            selected = [max(group, key=lambda item: item[name]) for group in selectors.values()]
        else:
            selected = [min(group, key=lambda item: item[name]) for group in selectors.values()]
        loo = [row for row in leave_one_out]
        feature_diagnostics[name] = {
            "spearman_with_single_composite": safe_spearman(values, errors),
            "best_single_selection_rate": float(
                np.mean([row["is_best_single"] for row in selected])
            ),
            "spearman_with_leave_one_out_removal_gain": safe_spearman(
                [row[name] for row in loo], [row["removal_gain"] for row in loo]
            ),
        }

    dispersion = {}
    multi_rows = [
        row
        for row in rows
        if "single_highest_confidence" in row["camera"]
        and "naive_mean" in row["camera"]
    ]
    gain = [
        row["camera"]["single_highest_confidence"]["camera_composite"]
        - row["camera"]["naive_mean"]["camera_composite"]
        for row in multi_rows
    ]
    for name in (
        "translation_pairwise_mean_m",
        "translation_pairwise_max_m",
        "rotation_pairwise_mean_deg",
        "rotation_pairwise_max_deg",
    ):
        dispersion[name] = safe_spearman(
            [row["candidate_dispersion"][name] for row in multi_rows], gain
        )

    paired_decomposition = {
        name: paired(rows, "single_highest_confidence", name)
        for name in (
            "translation_only_consensus",
            "rotation_only_consensus",
            "naive_mean",
        )
    }
    selected_vs_naive = {
        "all": paired(rows, "naive_mean", winner),
        "dev": paired(dev_rows, "naive_mean", winner),
        "test": paired(test_rows, "naive_mean", winner),
    }
    all_main = summaries(rows)
    test_main = summaries(test_rows)
    selected_test_delta = selected_vs_naive["test"]["camera_composite"]
    success = bool(
        selected_test_delta["second_minus_first_mean"] < 0.0
        and test_main[winner]["camera_composite"]["p90"]
        <= test_main["naive_mean"]["camera_composite"]["p90"]
        and test_main[winner]["catastrophic_rate"]
        <= test_main["naive_mean"]["catastrophic_rate"]
    )
    decision_summary = (
        "The development-selected soft uncertainty rule improves naive mean on the "
        "timestamp-held-out split without worsening P90/catastrophic rate. It is a "
        "promising fusion candidate, but cross-sequence validation is still required."
        if success
        else "No soft uncertainty rule has yet shown a held-out improvement over naive "
        "mean while preserving both P90 and catastrophic rate. Keep naive mean as the "
        "V13 default and treat the tested uncertainty cues as diagnostic only."
    )
    report = {
        "experiment": "V13 Phase 2 Multi-Human Boundary Fusion Optimization",
        "legacy_name": "V20 Phase 2",
        "protocol": {
            "phase1_report": str(args.phase1_report.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "human3r_rerun": False,
            "strict_gt_id_from_phase1_v2": True,
            "changed_component": "multi-human fusion strategy only",
            "fixed_components": "Human3R + hard reset + Fixed Explicit + V16 20deg + s=1",
            "disabled": ["DA3", "VGGT", "V11.4 scale", "token Re-ID", "scene refinement"],
        },
        "case_count": len(rows),
        "split": {
            "dev_timestamps": sorted(dev_timestamps),
            "test_timestamps": sorted(
                {int(row["case"]["timestamp"]) for row in test_rows}
            ),
            "dev_cases": len(dev_rows),
            "test_cases": len(test_rows),
        },
        "rule_selection": selection,
        "main_summaries": {
            "all": all_main,
            "dev": summaries(dev_rows),
            "test": test_main,
        },
        "soft_rule_summaries": rule_summaries,
        "paired": {
            "decomposition": paired_decomposition,
            "selected_soft_vs_naive": selected_vs_naive,
        },
        "number_ablation": {
            str(size): camera_summary(values)
            for size, values in sorted(number_rows.items())
        },
        "feature_diagnostics": feature_diagnostics,
        "dispersion_vs_naive_gain": dispersion,
        "leave_one_out": {
            "rows": leave_one_out,
            "positive_removal_gain_rate": float(
                np.mean([row["removal_gain"] > 0.0 for row in leave_one_out])
            )
            if leave_one_out
            else float("nan"),
        },
        "decision": {
            "heldout_soft_rule_success": success,
            "retain_default": winner if success else "naive_mean",
            "summary": decision_summary,
        },
        "cases": rows,
    }
    json_path = args.output_dir / "v13_phase2_fusion.json"
    markdown_path = args.output_dir / "v13_phase2_fusion.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f">> JSON: {json_path}", flush=True)
    print(f">> Report: {markdown_path}", flush=True)
    print(f">> Selected rule: {winner}", flush=True)
    print(f">> Decision: {report['decision']['retain_default']}", flush=True)


if __name__ == "__main__":
    main()
