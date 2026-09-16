#!/usr/bin/env python3
"""Generate result-independent qualitative diagnostics for the locked cases.

The case list is fixed in ``config/qualitative_cases_pre_test.json``.  This
script never ranks or filters cases by a method metric.  Each panel expresses
camera centers and human pelvis locations in the last pre-cut camera frame,
which removes the arbitrary global gauge while preserving the predicted
cross-shot displacement and rotation.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


METHODS = ("r0", "r2", "r3", "shot3r")
LABELS = {
    "r0": "Per-shot reset",
    "r2": "Human-joint SE(3)",
    "r3": "Scene FPFH + ICP",
    "shot3r": "Shot3R",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def save_image_bytes(value: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(value)) as image:
        image.convert("RGB").save(path, quality=95)


def stage_egobody_inputs(
    rows: dict[str, dict[str, Any]], selected: list[dict[str, Any]],
    staged_root: Path, destination: Path,
) -> None:
    for item in selected:
        case_id = str(item["case_id"])
        row = rows[case_id]
        boundary = int(row["boundary_index"])
        for label, index in (("pre", boundary - 1), ("post", boundary)):
            source = staged_root / str(row["image_members"][index])
            target = destination / f"{case_id}_{label}.jpg"
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    image.convert("RGB").save(target, quality=95)


def stage_egohumans_inputs(
    rows: dict[str, dict[str, Any]], selected: list[dict[str, Any]],
    outer_zip: Path, destination: Path,
) -> None:
    # Stream each nested tar.gz exactly once and write only the two requested
    # JPEGs per case; no full capture is materialized.
    by_archive: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for item in selected:
        case_id = str(item["case_id"])
        row = rows[case_id]
        boundary = int(row["boundary_index"])
        for label, index in (("pre", boundary - 1), ("post", boundary)):
            target = destination / f"{case_id}_{label}.jpg"
            if not target.is_file():
                member = str(row["image_members"][index]).lstrip("./")
                by_archive[str(row["archive_entry"])][member].append(target)
    if not by_archive:
        return
    with zipfile.ZipFile(outer_zip) as outer:
        for archive_name in sorted(by_archive):
            pending = dict(by_archive[archive_name])
            print(f"streaming {archive_name} for {len(pending)} qualitative frames", flush=True)
            with outer.open(archive_name) as inner:
                with tarfile.open(fileobj=inner, mode="r|gz") as archive:
                    for member in archive:
                        normalized = member.name.lstrip("./")
                        matched = next(
                            (name for name in pending if normalized.endswith(name)),
                            None,
                        )
                        if matched is None or not member.isfile():
                            continue
                        handle = archive.extractfile(member)
                        if handle is None:
                            continue
                        value = handle.read()
                        for target in pending.pop(matched):
                            save_image_bytes(value, target)
                        if not pending:
                            break
            if pending:
                raise FileNotFoundError(
                    f"{archive_name} misses qualitative members: {sorted(pending)}"
                )


def local_points(points: np.ndarray, reference_c2w: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return (value - reference_c2w[:3, 3]) @ reference_c2w[:3, :3]


def local_cameras(cameras: np.ndarray, reference_c2w: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(reference_c2w, dtype=np.float64))
    return np.einsum("ij,tjk->tik", inverse, np.asarray(cameras, dtype=np.float64))


def method_arrays(cache: Any, prefix: str) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(cache[f"{prefix}__{key}"])
        for key in ("cameras_c2w", "joints_world", "valid")
    }


def frozen_shot3r(workspace: Path, dataset: str, case_id: str) -> tuple[Path, str]:
    if dataset == "egobody":
        path = (
            workspace
            / "Movie3R/output/shot3r_v056_alignment_continued_egobody129/formal/predictions"
            / f"{case_id}.npz"
        )
        return path, "m3_alignment_continued"
    candidates = list(
        (workspace / "Movie3R/output/v19_egohumans/test/predictions").glob(
            f"*/{case_id}.npz"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one frozen EgoHumans prediction for {case_id}")
    return candidates[0], "m15_safe_boundary_permutation_causal_gru"


def camera_and_pelvis(arrays: dict[str, np.ndarray], boundary: int) -> dict[str, np.ndarray]:
    cameras = np.asarray(arrays["cameras_c2w"], dtype=np.float64)
    reference = cameras[boundary - 1]
    cameras_local = local_cameras(cameras, reference)
    pelvis = np.asarray(arrays["joints_world"], dtype=np.float64)[:, :, [1, 2]].mean(axis=2)
    pelvis_local = local_points(pelvis.reshape(-1, 3), reference).reshape(pelvis.shape)
    return {
        "camera": cameras_local[:, :3, 3],
        "forward": cameras_local[:, :3, 2],
        "pelvis": pelvis_local,
        "valid": np.asarray(arrays["valid"]).astype(bool),
    }


def gt_arrays(path: Path, boundary: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        arrays = {
            "cameras_c2w": np.asarray(cache["cameras_c2w"]),
            "joints_world": np.asarray(cache["joints_world"]),
            "valid": np.asarray(cache["visible"]),
        }
    return camera_and_pelvis(arrays, boundary)


def draw_layout(
    axis: plt.Axes, value: dict[str, np.ndarray], gt: dict[str, np.ndarray],
    boundary: int, title: str,
) -> None:
    pre = np.arange(max(0, boundary - 5), boundary)
    post = np.arange(boundary, min(len(value["camera"]), boundary + 5))
    axis.plot(value["camera"][pre, 0], value["camera"][pre, 2], "o-", color="#3B82F6", lw=1.8, ms=3.5, label="pre")
    axis.plot(value["camera"][post, 0], value["camera"][post, 2], "o-", color="#F97316", lw=1.8, ms=3.5, label="post")
    axis.plot(gt["camera"][[boundary - 1, boundary], 0], gt["camera"][[boundary - 1, boundary], 2], "x--", color="#111827", lw=1.2, ms=5.5, label="GT boundary")
    for indices, color in ((pre, "#2563EB"), (post, "#EA580C")):
        for frame in indices:
            valid = value["valid"][frame]
            points = value["pelvis"][frame, valid]
            if len(points):
                axis.scatter(points[:, 0], points[:, 2], s=13, color=color, alpha=0.70, edgecolors="none")
    for frame in (boundary - 1, boundary):
        valid = gt["valid"][frame]
        points = gt["pelvis"][frame, valid]
        if len(points):
            axis.scatter(points[:, 0], points[:, 2], s=27, marker="x", color="#111827", linewidths=1.2)
    axis.axhline(0.0, color="#CBD5E1", lw=0.5)
    axis.axvline(0.0, color="#CBD5E1", lw=0.5)
    axis.set_title(title, fontsize=9, pad=4)
    axis.set_xlabel("camera-local x (m)", fontsize=7)
    axis.set_ylabel("camera-local z (m)", fontsize=7)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.15)
    axis.set_aspect("equal", adjustable="datalim")


def render_case(
    workspace: Path, output_root: Path, dataset: str,
    row: dict[str, Any], stratum: str, destination: Path,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    boundary = int(row["boundary_index"])
    prediction = output_root / "predictions" / dataset / "test" / f"{case_id}.npz"
    gt_path = output_root / "work/gt_cache" / dataset / "test" / f"{case_id}.gt.npz"
    shot_path, shot_prefix = frozen_shot3r(workspace, dataset, case_id)
    values: dict[str, dict[str, np.ndarray]] = {}
    with np.load(prediction, allow_pickle=False) as cache:
        for method in ("r0", "r2", "r3"):
            values[method] = camera_and_pelvis(method_arrays(cache, method), boundary)
    with np.load(shot_path, allow_pickle=False) as cache:
        values["shot3r"] = camera_and_pelvis(method_arrays(cache, shot_prefix), boundary)
    gt = gt_arrays(gt_path, boundary)

    inputs = destination / "inputs"
    pre_path = inputs / f"{case_id}_pre.jpg"
    post_path = inputs / f"{case_id}_post.jpg"
    fig = plt.figure(figsize=(12.2, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.35))
    for column, (path, title) in enumerate(((pre_path, "Last frame before cut"), (post_path, "First frame after cut"))):
        axis = fig.add_subplot(grid[0, column * 2:(column + 1) * 2])
        axis.imshow(Image.open(path))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    for column, method in enumerate(METHODS):
        axis = fig.add_subplot(grid[1, column])
        title = LABELS[method]
        if method in {"r2", "r3"}:
            transform = json.loads(
                (output_root / "transforms" / dataset / "test" / case_id / f"{method}.json").read_text(encoding="utf-8")
            )
            title += f"\nstatus={transform['status']}"
        draw_layout(axis, values[method], gt, boundary, title)
    fig.suptitle(f"{dataset} · {stratum} · {case_id}", fontsize=10)
    pdf = destination / f"{dataset}_{stratum}_{case_id}.pdf"
    png = destination / f"{dataset}_{stratum}_{case_id}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "dataset": dataset,
        "stratum": stratum,
        "case_id": case_id,
        "figure_pdf": str(pdf.relative_to(output_root)),
        "figure_png": str(png.relative_to(output_root)),
        "input_pre": str(pre_path.relative_to(output_root)),
        "input_post": str(post_path.relative_to(output_root)),
        "shot3r_prediction": str(shot_path.resolve()),
        "shot3r_array_prefix": shot_prefix,
    }


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    root = args.output_root.resolve()
    lock = json.loads((root / "config/qualitative_cases_pre_test.json").read_text(encoding="utf-8"))
    manifests = {
        "egobody": read_rows(root / "work/frozen_manifests/egobody/egobody_cs150_test.runtime.jsonl"),
        "egohumans": read_rows(root / "work/frozen_manifests/egohumans/egohumans_test.runtime.jsonl"),
    }
    destination = root / "figures/qualitative"
    inputs = destination / "inputs"
    stage_egobody_inputs(
        manifests["egobody"], lock["datasets"]["egobody"],
        root / "work/staging/egobody_rgb", inputs,
    )
    stage_egohumans_inputs(
        manifests["egohumans"], lock["datasets"]["egohumans"],
        workspace / "data/EgoHuman.zip", inputs,
    )
    outputs = []
    for dataset in ("egobody", "egohumans"):
        for item in lock["datasets"][dataset]:
            case_id = str(item["case_id"])
            outputs.append(
                render_case(
                    workspace, root, dataset, manifests[dataset][case_id],
                    str(item["angle_stratum"]), destination,
                )
            )
            print(f"rendered {case_id}", flush=True)
    lines = [
        "# Locked qualitative registration diagnostics",
        "",
        "Cases were fixed independently of registration results: the first two case IDs in each pre-existing viewpoint stratum. Coordinates in every method panel are expressed in that method's last pre-cut camera frame. Black crosses/dashes denote GT boundary locations and are used only for post-seal visualization, never for registration.",
        "",
        "The EgoBody Shot3R visualization uses the archived alignment-continued prediction cache for the same frozen test case; the quantitative table remains bound to the v20 source-of-truth CSV. EgoHumans uses the archived `m15_safe_boundary_permutation_causal_gru` array underlying the frozen v19 alias.",
        "",
        "| Dataset | Stratum | Case | PDF | PNG |",
        "|---|---|---|---|---|",
    ]
    for item in outputs:
        lines.append(
            f"| {item['dataset']} | {item['stratum']} | `{item['case_id']}` | "
            f"[{Path(item['figure_pdf']).name}]({item['figure_pdf'].replace('figures/qualitative/', '')}) | "
            f"[{Path(item['figure_png']).name}]({item['figure_png'].replace('figures/qualitative/', '')}) |"
        )
    lines.extend(("", "## Machine-readable provenance", "", "```json", json.dumps(outputs, indent=2, ensure_ascii=False), "```", ""))
    atomic_text(destination / "QUALITATIVE_INDEX.md", "\n".join(lines))
    print(json.dumps({"cases": len(outputs), "index": str(destination / 'QUALITATIVE_INDEX.md')}, indent=2))


if __name__ == "__main__":
    main()
