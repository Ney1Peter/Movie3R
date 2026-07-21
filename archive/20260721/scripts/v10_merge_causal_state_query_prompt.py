#!/usr/bin/env python3
"""Merge parallel causal state-query rollout reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_causal_state_query_prompt_validation import generic_aggregate, write_summary_csv  # noqa: E402


def plot_validation(output_dir: Path, training: dict, validation: dict) -> None:
    latent_names = ["raw", "early"]
    train_values = [training[name]["train"]["recovery"] for name in latent_names]
    val_values = [training[name]["validation"]["recovery"] for name in latent_names]
    x = np.arange(len(latent_names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, train_values, width=0.36, label="train")
    ax.bar(x + 0.18, val_values, width=0.36, label="unseen clip")
    ax.set_xticks(x, latent_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Latent recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latent_train_validation_recovery.png", dpi=180)
    plt.close(fig)

    names = [
        "B_reset_baseline",
        "C_boundary_output_oracle",
        "D_raw_state_query_prompt",
        "E_early_state_query_prompt",
        "F_early_prompt_plus_output_oracle",
        "G_first_write_state_oracle",
    ]
    labels = ["Reset", "Output oracle", "Raw prompt", "Early prompt", "Prompt+output", "Write oracle"]
    metrics = [
        ("camera_rotation_deg", "Camera rotation (deg)"),
        ("camera_translation_m", "Camera translation (m)"),
        ("pointmap_world_mean_m", "World pointmap (m)"),
        ("human_world_root_m", "Human root (m)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [validation[name]["mean_error"][metric] for name in names]
        ax.bar(np.arange(len(names)), values)
        ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_mean_errors.png", dpi=180)
    plt.close(fig)

    curve_names = [
        "C_boundary_output_oracle",
        "D_raw_state_query_prompt",
        "E_early_state_query_prompt",
        "F_early_prompt_plus_output_oracle",
        "G_first_write_state_oracle",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in (
        (axes[0], "camera_rotation_deg", "Camera rotation recovery"),
        (axes[1], "pointmap_world_mean_m", "World pointmap recovery"),
        (axes[2], "human_world_root_m", "Human root recovery"),
    ):
        for name in curve_names:
            offsets = sorted(validation[name]["offset_recovery"], key=int)
            values = [validation[name]["offset_recovery"][offset][metric] for offset in offsets]
            ax.plot([int(offset) for offset in offsets], values, marker="o", label=name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Post-cut offset")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_offset_recovery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    cases = []
    for path in args.reports:
        cases.extend(json.loads(path.read_text(encoding="utf-8"))["cases"])
    train_cases = [case for case in cases if case["split"] == "train"]
    val_cases = [case for case in cases if case["split"] == "validation"]
    rollout = {
        "overall": generic_aggregate(cases),
        "train": generic_aggregate(train_cases),
        "validation": generic_aggregate(val_cases),
        "cases": cases,
    }
    training = {}
    for name, checkpoint in (
        ("raw", args.output_dir / "checkpoints" / "raw_first_write.pth"),
        ("early", args.output_dir / "checkpoints" / "early_query_first_write.pth"),
    ):
        training[name] = torch.load(checkpoint, map_location="cpu", weights_only=False)["report"]
    report = {
        "experiment": "Causal State-query Shot Prompt Validation",
        "constraints": {
            "human3r_frozen": True,
            "old_state_read_only": True,
            "fresh_state_only_committed": True,
            "parallel_gpu_evaluation": True,
        },
        "training": training,
        "rollout": rollout,
    }
    output = args.output_dir / "causal_state_query_prompt_metrics.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_summary_csv(args.output_dir / "rollout_summary.csv", rollout["overall"])
    plot_validation(args.output_dir, training, rollout["validation"])
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
