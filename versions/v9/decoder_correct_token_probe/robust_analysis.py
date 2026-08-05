#!/usr/bin/env python3
"""Robust validation for selected decoder/correction-token residual readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_token_depth_probe import (
    REPO_ROOT,
    mlp_fit_predict,
    ridge_fit_predict,
    stack_features,
    stack_targets,
    summarize_baseline,
    summarize_residual,
)


DEFAULT_CACHE = REPO_ROOT / "output/v9_decoder_correct_token_probe/full24/token_cache.pt"
DEFAULT_OUTPUT = REPO_ROOT / "output/v9_decoder_correct_token_probe/full24/robust_analysis.json"
CANDIDATES = {
    "raw_pose_only": (),
    "final_corr_mean": ("decoder_l11_corr_mean",),
    "l08_pose": ("decoder_l08_pose",),
    "l11_pose": ("decoder_l11_pose",),
    "l08_sem_momentum": ("decoder_l08_semantic", "decoder_l08_momentum"),
    "l08_sem_align": ("decoder_l08_semantic", "decoder_l08_alignment"),
    "l11_alignment": ("decoder_l11_alignment",),
    "l11_momentum": ("decoder_l11_momentum",),
    "l11_image_mean": ("decoder_l11_image_mean",),
}
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mlp-steps", type=int, default=400)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def pair_fold(row: dict, folds: int) -> int:
    pre, post = map(str, row["seqs"][-2:])
    pair = "|".join(sorted((pre, post)))
    digest = hashlib.sha1(f"{row['source']}|{pair}".encode()).hexdigest()
    return int(digest[:8], 16) % folds


def mean_composite(rows: list[dict], predictions: np.ndarray) -> float:
    return float(summarize_residual(rows, predictions)["composite"]["mean"])


def source_summary(summary: dict) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in summary["per_case"]:
        grouped[case["source"]].append(case)
    result = {}
    for source, cases in sorted(grouped.items()):
        result[source] = {
            key: float(np.mean([case[key] for case in cases]))
            for key in ("translation_m", "rotation_deg", "composite")
        }
    return result


def fit_mlp(
    train_rows: list[dict],
    eval_rows: list[dict],
    names: tuple[str, ...],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict]:
    _, prediction = mlp_fit_predict(
        stack_features(train_rows, names),
        stack_targets(train_rows),
        stack_features(eval_rows, names),
        args.mlp_steps,
        seed,
        args.device,
    )
    return prediction, summarize_residual(eval_rows, prediction)


def grouped_mlp_cv(
    rows: list[dict], names: tuple[str, ...], args: argparse.Namespace, seed_base: int
) -> dict:
    fold_ids = np.asarray([pair_fold(row, args.folds) for row in rows])
    per_seed = []
    per_source: dict[str, list[float]] = defaultdict(list)
    for seed_offset in range(args.seeds):
        fold_scores = []
        source_cases: dict[str, list[float]] = defaultdict(list)
        for fold in range(args.folds):
            train_rows = [row for row, value in zip(rows, fold_ids) if value != fold]
            eval_rows = [row for row, value in zip(rows, fold_ids) if value == fold]
            if not train_rows or not eval_rows:
                continue
            _, summary = fit_mlp(
                train_rows, eval_rows, names, args, seed_base + 100 * seed_offset + fold
            )
            fold_scores.append(summary["composite"]["mean"])
            for case in summary["per_case"]:
                source_cases[case["source"]].append(case["composite"])
        per_seed.append(float(np.mean(fold_scores)))
        for source, values in source_cases.items():
            per_source[source].append(float(np.mean(values)))
    return {
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
        "seeds": per_seed,
        "per_source": {
            source: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for source, values in sorted(per_source.items())
        },
    }


def leave_source_out(
    rows: list[dict], names: tuple[str, ...], args: argparse.Namespace, seed_base: int
) -> dict:
    sources = sorted({row["source"] for row in rows})
    result = {}
    for source_index, source in enumerate(sources):
        train_rows = [row for row in rows if row["source"] != source]
        eval_rows = [row for row in rows if row["source"] == source]
        values = []
        for seed_offset in range(args.seeds):
            _, summary = fit_mlp(
                train_rows,
                eval_rows,
                names,
                args,
                seed_base + 100 * seed_offset + source_index,
            )
            values.append(summary["composite"]["mean"])
        result[source] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    result["macro_mean"] = float(np.mean([result[source]["mean"] for source in sources]))
    return result


def ridge_cv(rows: list[dict], names: tuple[str, ...], folds: int) -> dict:
    fold_ids = np.asarray([pair_fold(row, folds) for row in rows])
    x = stack_features(rows, names)
    y = stack_targets(rows)
    scores = {}
    for alpha in ALPHAS:
        values = []
        for fold in range(folds):
            train_mask = fold_ids != fold
            eval_mask = fold_ids == fold
            if not train_mask.any() or not eval_mask.any():
                continue
            prediction = ridge_fit_predict(x[train_mask], y[train_mask], x[eval_mask], alpha)
            eval_rows = [row for row, keep in zip(rows, eval_mask) if keep]
            values.append(mean_composite(eval_rows, prediction))
        scores[str(alpha)] = float(np.mean(values))
    best = min(ALPHAS, key=lambda value: scores[str(value)])
    return {"selected_alpha": best, "scores": scores}


def frozen_eval(
    train_rows: list[dict],
    eval_rows: list[dict],
    names: tuple[str, ...],
    args: argparse.Namespace,
    seed_base: int,
) -> dict:
    predictions = []
    seed_scores = []
    for seed_offset in range(args.seeds):
        prediction, summary = fit_mlp(
            train_rows, eval_rows, names, args, seed_base + seed_offset
        )
        predictions.append(prediction)
        seed_scores.append(summary["composite"]["mean"])
    ensemble = np.mean(predictions, axis=0)
    summary = summarize_residual(eval_rows, ensemble)
    return {
        "seed_mean": float(np.mean(seed_scores)),
        "seed_std": float(np.std(seed_scores)),
        "ensemble": summary,
        "per_source": source_summary(summary),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Robust Decoder / Correct-Token Analysis",
        "",
        "All model selection uses training-only camera-pair grouped CV. The frozen ten cuts are read once for reporting.",
        "",
        "## Baselines",
        "",
        "| Method | Composite | P90 |",
        "|---|---:|---:|",
    ]
    for name in ("raw", "formal_v9"):
        values = report["baselines"][name]
        lines.append(
            f"| {name} | {values['composite']['mean']:.4f} | {values['composite']['p90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Validation",
            "",
            "| Candidate | Pair-CV | Leave-source-out | Frozen ensemble | Frozen P90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    ranking = sorted(
        report["candidates"].items(),
        key=lambda item: item[1]["frozen_eval"]["ensemble"]["composite"]["mean"],
    )
    for name, values in ranking:
        frozen = values["frozen_eval"]["ensemble"]["composite"]
        lines.append(
            f"| {name} | {values['pair_grouped_mlp_cv']['mean']:.4f} | "
            f"{values['leave_source_out']['macro_mean']:.4f} | {frozen['mean']:.4f} | "
            f"{frozen['p90']:.4f} |"
        )
    lines.extend(["", "## Frozen Per-Source Composite", ""])
    sources = sorted(next(iter(report["candidates"].values()))["frozen_eval"]["per_source"])
    lines.append("| Candidate | " + " | ".join(sources) + " |")
    lines.append("|---|" + "---:|" * len(sources))
    for name, values in ranking:
        cells = [name] + [
            f"{values['frozen_eval']['per_source'][source]['composite']:.4f}"
            for source in sources
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval10"]
    raw = summarize_baseline(eval_rows, "raw_relative")
    formal = summarize_baseline(eval_rows, "full_relative")
    report: dict[str, Any] = {
        "protocol": {
            "cache": str(args.cache),
            "train_cases": len(train_rows),
            "eval_cases": len(eval_rows),
            "folds": args.folds,
            "seeds": args.seeds,
            "mlp_steps": args.mlp_steps,
        },
        "baselines": {
            "raw": {**raw, "per_source": source_summary(raw)},
            "formal_v9": {**formal, "per_source": source_summary(formal)},
        },
        "candidates": {},
    }
    for index, (name, features) in enumerate(CANDIDATES.items()):
        seed_base = 20260801 + 10000 * index
        values = {
            "features": ["raw_relative_pose", *features],
            "ridge_cv": ridge_cv(train_rows, features, args.folds),
            "pair_grouped_mlp_cv": grouped_mlp_cv(
                train_rows, features, args, seed_base
            ),
            "leave_source_out": leave_source_out(
                train_rows, features, args, seed_base + 5000
            ),
            "frozen_eval": frozen_eval(
                train_rows, eval_rows, features, args, seed_base + 9000
            ),
        }
        report["candidates"][name] = values
        print(
            f"{name:22s} pair_cv={values['pair_grouped_mlp_cv']['mean']:.4f} "
            f"source_cv={values['leave_source_out']['macro_mean']:.4f} "
            f"frozen={values['frozen_eval']['ensemble']['composite']['mean']:.4f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(report), indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
