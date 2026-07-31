#!/usr/bin/env python3
"""Robust residual-probe analysis using camera-pair CV and learning curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_probe import (
    REPO_ROOT,
    feature_groups,
    ridge_fit_predict,
    stack_features,
    stack_targets,
    summarize_predictions,
    target_vector,
)


DEFAULT_CACHE = REPO_ROOT / "output/v9_multilayer_information_probe/scale96/descriptor_cache.pt"
DEFAULT_OUTPUT = REPO_ROOT / "output/v9_multilayer_information_probe/scale96/robust_analysis.json"
GROUP_NAMES = (
    "raw_pose_only",
    "cut3r_l05",
    "cut3r_l23",
    "dino_l05",
    "dino_l23",
    "decoder_l02",
    "decoder_l05",
    "decoder_l08",
    "decoder_l11",
    "decoder_multi",
    "all_multi",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SIZES_PER_SOURCE = (4, 8, 16, 24, 48, 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    return parser.parse_args()


def names_for_group(rows: list[dict], group_name: str) -> tuple[str, ...]:
    if group_name == "raw_pose_only":
        return ()
    return feature_groups(rows)[group_name]


def pair_fold(row: dict, folds: int) -> int:
    pre, post = map(str, row["seqs"][-2:])
    unordered = "|".join(sorted((pre, post)))
    digest = hashlib.sha1(f"{row['source']}|{unordered}".encode()).hexdigest()
    return int(digest[:8], 16) % folds


def mean_composite(rows: list[dict], prediction: np.ndarray) -> float:
    return float(summarize_predictions(rows, prediction, "residual")["composite"]["mean"])


def cross_validate_alpha(
    rows: list[dict], names: tuple[str, ...], folds: int
) -> tuple[float, dict[str, float]]:
    fold_ids = np.asarray([pair_fold(row, folds) for row in rows])
    all_x = stack_features(rows, names, include_raw_pose=True)
    all_y = stack_targets(rows, "residual")
    scores = {}
    for alpha in ALPHAS:
        values = []
        for fold in range(folds):
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            if not train_mask.any() or not val_mask.any():
                continue
            prediction = ridge_fit_predict(
                all_x[train_mask], all_y[train_mask], all_x[val_mask], alpha
            )
            val_rows = [row for row, keep in zip(rows, val_mask) if keep]
            values.append(mean_composite(val_rows, prediction))
        scores[str(alpha)] = float(np.mean(values))
    best = min(ALPHAS, key=lambda alpha: scores[str(alpha)])
    return best, scores


def fit_eval(
    train_rows: list[dict],
    eval_rows: list[dict],
    names: tuple[str, ...],
    alpha: float,
) -> dict:
    train_x = stack_features(train_rows, names, include_raw_pose=True)
    eval_x = stack_features(eval_rows, names, include_raw_pose=True)
    prediction = ridge_fit_predict(
        train_x, stack_targets(train_rows, "residual"), eval_x, alpha
    )
    return summarize_predictions(eval_rows, prediction, "residual")


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for source in sorted(grouped):
        source_rows = grouped[source]
        indices = rng.choice(len(source_rows), size=min(size, len(source_rows)), replace=False)
        selected.extend(source_rows[index] for index in indices)
    return selected


def paired_bootstrap(reference: list[float], candidate: list[float], seed: int = 20260731) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    gains = reference - candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gains), size=(20000, len(gains)))
    samples = gains[indices].mean(axis=1)
    return {
        "mean_gain": float(gains.mean()),
        "median_gain": float(np.median(gains)),
        "win_fraction": float((gains > 0).mean()),
        "bootstrap_ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Robust V9 Multi-Layer Residual Analysis",
        "",
        "Camera-pair grouped CV selects ridge regularization without reading the frozen ten-cut evaluation.",
        "",
        "## Frozen Evaluation",
        "",
        "| Feature | CV alpha | Translation | Rotation | Composite | P90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ranking = sorted(report["groups"].items(), key=lambda item: item[1]["frozen_eval"]["composite"]["mean"])
    for name, values in ranking:
        result = values["frozen_eval"]
        lines.append(
            f"| {name} | {values['selected_alpha']:.1f} | "
            f"{result['translation_m']['mean']:.4f} | {result['rotation_deg']['mean']:.2f} | "
            f"{result['composite']['mean']:.4f} | {result['composite']['p90']:.4f} |"
        )
    lines.extend(["", "## Learning Curve", "", "Mean composite over repeated stratified subsets.", ""])
    lines.append("| Cases/source | " + " | ".join(GROUP_NAMES) + " |")
    lines.append("|---:|" + "---:|" * len(GROUP_NAMES))
    for size in map(str, SIZES_PER_SOURCE):
        cells = [size]
        for group in GROUP_NAMES:
            value = report["groups"][group]["learning_curve"][size]
            cells.append(f"{value['mean']:.3f} +/- {value['std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    comparison = report["decoder_l08_vs_raw"]
    lines.extend(
        [
            "",
            "## Paired Comparison",
            "",
            f"`decoder_l08` minus raw-pose-only gain: `{comparison['mean_gain']:.4f}` composite; "
            f"95% bootstrap CI `{comparison['bootstrap_ci95'][0]:.4f}` to "
            f"`{comparison['bootstrap_ci95'][1]:.4f}`; per-cut win fraction "
            f"`{comparison['win_fraction']:.2f}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval10"]
    report = {
        "cache": str(args.cache),
        "train_cases": len(train_rows),
        "eval_cases": len(eval_rows),
        "folds": args.folds,
        "seeds": args.seeds,
        "groups": {},
    }
    for group_name in GROUP_NAMES:
        names = names_for_group(rows, group_name)
        alpha, cv_scores = cross_validate_alpha(train_rows, names, args.folds)
        frozen_eval = fit_eval(train_rows, eval_rows, names, alpha)
        learning_curve = {}
        for size in SIZES_PER_SOURCE:
            values = []
            for seed_offset in range(args.seeds):
                subset = stratified_sample(train_rows, size, 20260731 + seed_offset)
                values.append(fit_eval(subset, eval_rows, names, alpha)["composite"]["mean"])
            learning_curve[str(size)] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        report["groups"][group_name] = {
            "features": (["raw_relative_pose"] + list(names)),
            "selected_alpha": alpha,
            "cv_scores": cv_scores,
            "frozen_eval": frozen_eval,
            "learning_curve": learning_curve,
        }
        print(
            f"{group_name:18s} alpha={alpha:6.1f} "
            f"frozen={frozen_eval['composite']['mean']:.4f}",
            flush=True,
        )

    raw_cases = report["groups"]["raw_pose_only"]["frozen_eval"]["per_case"]
    decoder_cases = report["groups"]["decoder_l08"]["frozen_eval"]["per_case"]
    report["decoder_l08_vs_raw"] = paired_bootstrap(
        [case["composite"] for case in raw_cases],
        [case["composite"] for case in decoder_cases],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
