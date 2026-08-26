#!/usr/bin/env python3
"""Render a real-output qualitative comparison from retained EgoHumans caches.

The figure uses no RGB synthesis, GT annotation, or evaluator values.  It
renders camera centres and predicted pelvis trajectories from one
lexicographically first retained frozen Test case; the selection is therefore
independent of an outcome-quality search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from publication.bridge3r_iclr2027.runtime_contract import apply_locked_transaction


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "output/v19_egohumans/test/predictions"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures/egohumans_cache_qualitative.pdf"
DEFAULT_PROVENANCE = Path(__file__).resolve().parent / "evidence/egohumans_cache_qualitative.json"


def source_arrays(path: Path) -> dict[str, np.ndarray]:
    keys = ("cameras_c2w", "joints_world", "vertices_world", "valid", "native_ids", "persistent_ids")
    with np.load(path, allow_pickle=False) as cache:
        return {key: np.asarray(cache[f"m3_b0_only__{key}"]).copy() for key in keys}


def pelvis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(joints)[..., [1, 2], :].mean(axis=-2)


def final_arrays(runtime_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    cache_path = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
    arrays = source_arrays(cache_path)
    pairs = [tuple(map(int, pair)) for pair in runtime["geometry"]["association"]["pairs"]]
    output, _ = apply_locked_transaction(
        arrays,
        boundary=int(runtime["record"]["boundary_index"]),
        pairs=pairs,
        cut_detected=True,
    )
    return output, runtime


def strict_arrays(cache_path: Path) -> dict[str, np.ndarray]:
    keys = ("cameras_c2w", "joints_world", "vertices_world", "valid", "native_ids", "persistent_ids")
    with np.load(cache_path, allow_pickle=False) as cache:
        return {key: np.asarray(cache[f"m0_strict_human3r__{key}"]).copy() for key in keys}


def finite_extent(arrays_by_method: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for arrays in arrays_by_method:
        camera = np.asarray(arrays["cameras_c2w"], dtype=np.float64)[:, :3, 3]
        root = pelvis(np.asarray(arrays["joints_world"], dtype=np.float64))
        points.extend((camera, root.reshape(-1, 3)))
    merged = np.concatenate(points, axis=0)
    merged = merged[np.isfinite(merged).all(axis=1)]
    lower, upper = np.percentile(merged, [1, 99], axis=0)
    span = max(float(np.max(upper - lower)), 1e-3)
    centre = 0.5 * (lower + upper)
    return centre, np.full(3, span * 0.58)


def draw(ax: Any, arrays: dict[str, np.ndarray], boundary: int, title: str, centre: np.ndarray, half: np.ndarray) -> None:
    camera = np.asarray(arrays["cameras_c2w"], dtype=np.float64)[:, :3, 3]
    roots = pelvis(np.asarray(arrays["joints_world"], dtype=np.float64))
    valid = np.asarray(arrays["valid"], dtype=bool)
    identities = np.asarray(arrays["persistent_ids"], dtype=np.int64)
    ax.plot(*camera[:boundary].T, color="#4c566a", lw=1.6, label="camera (pre)")
    ax.plot(*camera[boundary - 1:].T, color="#d08770", lw=1.6, ls="--", label="camera (post)")
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    observed = sorted({int(x) for x in identities[valid] if x >= 0})
    for colour, identity in zip(colours, observed):
        mask = valid & (identities == identity)
        values = roots.copy()
        values[~mask] = np.nan
        ax.plot(*values[:boundary].T, color=colour, lw=2.0, label=f"person {identity}")
        ax.plot(*values[boundary - 1:].T, color=colour, lw=2.0, ls="--")
    ax.scatter(*camera[[boundary - 1, boundary]].T, c=["#4c566a", "#d08770"], s=24, depthshade=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*(centre[0] + np.array([-half[0], half[0]])))
    ax.set_ylim(*(centre[1] + np.array([-half[1], half[1]])))
    ax.set_zlim(*(centre[2] + np.array([-half[2], half[2]])))
    ax.set_xlabel("x (m)", labelpad=-7)
    ax.set_ylabel("y (m)", labelpad=-7)
    ax.set_zlabel("z (m)", labelpad=-7)
    ax.view_init(elev=19, azim=-63)
    ax.grid(False)
    ax.set_box_aspect((1, 1, 0.75))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()
    runtime_path = args.runtime or sorted(DEFAULT_ROOT.rglob("*.runtime.json"))[0]
    bridge, runtime = final_arrays(runtime_path)
    cache_path = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
    strict = strict_arrays(cache_path)
    boundary = int(runtime["record"]["boundary_index"])
    centre, half = finite_extent([strict, bridge])
    figure = plt.figure(figsize=(8.2, 3.9), constrained_layout=True)
    left = figure.add_subplot(1, 2, 1, projection="3d")
    right = figure.add_subplot(1, 2, 2, projection="3d")
    draw(left, strict, boundary, "Strict Human3R", centre, half)
    draw(right, bridge, boundary, "Bridge3R (locked transaction)", centre, half)
    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    payload = {
        "schema_version": "Bridge3R-real-output-qualitative-v1",
        "selection_rule": "lexicographically first retained EgoHumans Test runtime report; no prediction-quality selection",
        "runtime_report": str(runtime_path.relative_to(REPO_ROOT)),
        "cache": str(cache_path.relative_to(REPO_ROOT)),
        "case_id": runtime["record"]["case_id"],
        "boundary": boundary,
        "rendered_quantities": "retained model-output camera centres and predicted pelvis trajectories",
        "not_rendered": "RGB, ground truth, evaluator values, or synthesized imagery",
        "methods": ["m0_strict_human3r", "Bridge3R locked publication transaction"],
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
