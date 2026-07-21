#!/usr/bin/env python3
"""Training-free cross-view retrieval probe on cached Human3R token summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "output" / "v10_latent_token_probe" / "token_cache" / "cache_index.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_latent_token_probe" / "cross_view_retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_index", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cache(index_path: Path) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
    metadata = read_jsonl(index_path)
    arrays = []
    for row in metadata:
        with np.load(row["cache_path"]) as data:
            arrays.append({key: data[key].astype(np.float32) for key in data.files})
    return metadata, arrays


def feature_table(arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    sample = arrays[0]
    table = {}
    for layer in range(sample["encoder_layer_pool"].shape[1]):
        table[f"encoder_image_l{layer:02d}"] = np.stack([row["encoder_layer_pool"][:, layer] for row in arrays])
    for layer in range(sample["decoder_image_layer_pool"].shape[1]):
        table[f"decoder_image_l{layer:02d}"] = np.stack([row["decoder_image_layer_pool"][:, layer] for row in arrays])
    for layer in range(sample["decoder_state_layer_pool"].shape[1]):
        table[f"decoder_state_l{layer:02d}"] = np.stack([row["decoder_state_layer_pool"][:, layer] for row in arrays])
    for name, key in (
        ("camera_initial", "camera_initial"),
        ("camera_refined", "camera_refined"),
        ("human_prompt", "human_prompt_pool"),
        ("human_refined", "human_refined_pool"),
        ("persistent_state", "persistent_state_pool"),
        ("new_state", "new_state_pool"),
        ("dino_global", "dino_pool"),
    ):
        table[name] = np.stack([row[key] for row in arrays])
    return table


def normalize(features: np.ndarray) -> np.ndarray:
    features = features - features.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-8)


def retrieval_metrics(
    features: np.ndarray,
    metadata: list[dict],
    indices: np.ndarray,
) -> dict:
    boundary = int(metadata[0]["boundary"])
    query = normalize(features[indices, boundary])
    key = normalize(features[indices, boundary - 1])
    similarity = query @ key.T
    order = np.argsort(-similarity, axis=1)
    exact_ranks = []
    group_ranks = []
    for local_idx, global_idx in enumerate(indices):
        exact_rank = int(np.where(order[local_idx] == local_idx)[0][0]) + 1
        exact_ranks.append(exact_rank)
        group = (metadata[global_idx]["record"]["source"], metadata[global_idx]["record"].get("group"))
        group_rank = len(indices) + 1
        for rank, candidate_local in enumerate(order[local_idx], start=1):
            candidate_global = int(indices[candidate_local])
            if candidate_global == global_idx:
                continue
            candidate_group = (
                metadata[candidate_global]["record"]["source"],
                metadata[candidate_global]["record"].get("group"),
            )
            if candidate_group == group:
                group_rank = rank
                break
        group_ranks.append(group_rank)
    exact_ranks = np.asarray(exact_ranks)
    group_ranks = np.asarray(group_ranks)
    return {
        "count": len(indices),
        "exact_case_recall_at_1": float(np.mean(exact_ranks <= 1)),
        "exact_case_recall_at_5": float(np.mean(exact_ranks <= 5)),
        "exact_case_median_rank": float(np.median(exact_ranks)),
        "same_group_excluding_self_recall_at_1": float(np.mean(group_ranks <= 1)),
        "same_group_excluding_self_recall_at_5": float(np.mean(group_ranks <= 5)),
        "same_group_excluding_self_median_rank": float(np.median(group_ranks)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, arrays = load_cache(args.cache_index)
    features = feature_table(arrays)
    groupings: dict[str, np.ndarray] = {"overall": np.arange(len(metadata), dtype=np.int64)}
    for source in sorted({row["record"]["source"] for row in metadata}):
        groupings[f"source:{source}"] = np.asarray(
            [idx for idx, row in enumerate(metadata) if row["record"]["source"] == source],
            dtype=np.int64,
        )
    for angle in sorted({row["record"].get("angle_bucket", "unknown") for row in metadata}):
        groupings[f"angle:{angle}"] = np.asarray(
            [idx for idx, row in enumerate(metadata) if row["record"].get("angle_bucket", "unknown") == angle],
            dtype=np.int64,
        )
    rows = []
    for feature_name, values in features.items():
        for group_name, indices in groupings.items():
            if len(indices) < 2:
                continue
            rows.append(
                {
                    "feature": feature_name,
                    "group": group_name,
                    **retrieval_metrics(values, metadata, indices),
                }
            )
    with (args.output_dir / "cross_view_retrieval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "cross_view_retrieval.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    overall = [row for row in rows if row["group"] == "overall"]
    overall.sort(key=lambda row: row["exact_case_recall_at_1"], reverse=True)
    selected = overall[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(selected))
    ax.barh(y - 0.18, [row["exact_case_recall_at_1"] for row in selected], height=0.35, label="Exact case R@1")
    ax.barh(
        y + 0.18,
        [row["same_group_excluding_self_recall_at_5"] for row in selected],
        height=0.35,
        label="Same group excl. self R@5",
    )
    ax.set_yticks(y, [row["feature"] for row in selected], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_title("Training-free Cross-view Global Retrieval")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "cross_view_retrieval_top15.png", dpi=180)
    plt.close(fig)
    summary = {
        "case_count": len(metadata),
        "best_exact_case": selected[0] if selected else None,
        "best_same_group": max(overall, key=lambda row: row["same_group_excluding_self_recall_at_5"]),
        "note": "This is global token retrieval. It does not claim local physical 3D point correspondence.",
    }
    (args.output_dir / "cross_view_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
