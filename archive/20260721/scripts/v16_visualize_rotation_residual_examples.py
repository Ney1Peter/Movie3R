#!/usr/bin/env python3
"""Render qualitative V16 boundary-alignment comparisons from cached outputs.

The script is evaluation-only: it reads the frozen Human3R local-reset cache and
the V16 candidate cache, then compares Fixed Explicit, the bounded torso-motion
rotation residual, and the Boundary Oracle in a shared prediction gauge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    background_cloud,
    history_background_cloud,
)
from v10_oracle_candidate_selection_probe import predicted_poses  # noqa: E402


DEFAULT_V10_REPORT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_CANDIDATE_DIR = (
    REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "visual_examples"
)


EXAMPLES = (
    {
        "case_name": "avatarrex_120_150_lbn2_1842_22010710_22070932",
        "slug": "01_avatarrex_clear_gain",
        "caption": "AvatarReX: clear rotation correction",
        "mode": "fixed",
    },
    {
        "case_name": "thuman_090_120_thuman02_2772_cam08_cam16",
        "slug": "02_thuman_clear_gain",
        "caption": "THuman: near-oracle rotation after V16",
        "mode": "fixed",
    },
    {
        "case_name": "mvhuman100_090_120_100003_338_CC32871A035_CC32871A008",
        "slug": "03_mvhuman100_large_angle_gain",
        "caption": "MVHuman100: large-angle tail reduction",
        "mode": "fixed",
    },
    {
        "case_name": "mvhuman200_060_090_200003_426_22327109_22327073",
        "slug": "04_mvhuman200_large_angle_gain",
        "caption": "MVHuman200: large-angle correction",
        "mode": "fixed",
    },
    {
        "case_name": "avatarrex_150_180_lbn1_1632_22010716_22139907",
        "slug": "05_avatarrex_failure",
        "caption": "Failure case: V16 changes an already good rotation",
        "mode": "fixed",
    },
    {
        "case_name": "thuman_120_150_thuman00_2442_cam04_cam19",
        "slug": "06_v15_v16_modular_gain",
        "caption": "Modular test: V15 coarse pose followed by V16",
        "mode": "v15",
    },
)


METHOD_COLORS = {
    "Fixed Explicit": "#d97706",
    "V16 Torso Motion": "#059669",
    "V15 Coarse": "#dc6b35",
    "V15 + V16": "#0d9488",
    "Boundary Oracle": "#2563eb",
}
HISTORY_COLOR = "#8b95a5"
TARGET_COLOR = "#c026d3"
OLD_CAMERA_COLOR = "#111827"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10_REPORT)
    parser.add_argument("--candidate_dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cloud_points_per_frame", type=int, default=2500)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def load_maps(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    v10 = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v10_cases = {str(case["case_name"]): case for case in v10["cases"]}
    candidates: dict[str, dict] = {}
    for path in sorted(args.candidate_dir.glob("v16_candidates_shard_*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        for case in shard["cases"]:
            name = str(case["case_name"])
            if name in candidates:
                raise ValueError(f"Duplicate V16 candidate case: {name}")
            candidates[name] = case
    return v10_cases, candidates


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nj->ni", transform[:3, :3], points) + transform[:3, 3]


def candidate_methods(case: dict, mode: str) -> list[tuple[str, dict]]:
    if mode == "fixed":
        return [
            ("Fixed Explicit", case["baselines"]["fixed_explicit"]),
            ("V16 Torso Motion", case["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]),
            ("Boundary Oracle", case["baselines"]["boundary_oracle"]),
        ]
    if mode == "v15":
        return [
            ("V15 Coarse", case["baselines"]["v15_coarse"]),
            ("V15 + V16", case["v15_candidates"]["v15_torso_motion_1f_resolve_t"]),
            ("Boundary Oracle", case["baselines"]["boundary_oracle"]),
        ]
    raise ValueError(f"Unknown example mode: {mode}")


def load_rgb(local_dir: Path, frame: int) -> np.ndarray:
    return np.asarray(Image.open(local_dir / "color" / f"{frame:06d}.png").convert("RGB"))


def projected_bounds(
    clouds: list[np.ndarray],
    camera_poses: list[np.ndarray],
    axes: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    sampled = []
    for cloud in clouds:
        finite = cloud[np.isfinite(cloud).all(axis=1)]
        if len(finite) > 4000:
            finite = finite[:: max(len(finite) // 4000, 1)]
        sampled.append(finite[:, axes])
    values = np.concatenate(sampled, axis=0)
    low = np.quantile(values, 0.01, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    for pose in camera_poses:
        point = pose[:3, 3][list(axes)]
        low = np.minimum(low, point)
        high = np.maximum(high, point)
    center = (low + high) * 0.5
    span = max(float(np.max(high - low)), 0.5) * 1.18
    return (
        (float(center[0] - span * 0.5), float(center[0] + span * 0.5)),
        (float(center[1] - span * 0.5), float(center[1] + span * 0.5)),
    )


def draw_camera(
    ax: plt.Axes,
    pose: np.ndarray,
    axes: tuple[int, int],
    color: str,
    marker: str,
    label: str | None,
    arrow_length: float,
    alpha: float = 1.0,
    dashed: bool = False,
) -> None:
    center = pose[:3, 3]
    forward = pose[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    direction = forward[list(axes)]
    norm = float(np.linalg.norm(direction))
    if norm > 1e-8:
        direction = direction / norm * arrow_length
        ax.plot(
            [center[axes[0]], center[axes[0]] + direction[0]],
            [center[axes[1]], center[axes[1]] + direction[1]],
            color=color,
            linewidth=1.7,
            alpha=alpha,
            linestyle="--" if dashed else "-",
            zorder=6,
        )
    ax.scatter(
        [center[axes[0]]],
        [center[axes[1]]],
        c=[color],
        marker=marker,
        s=58 if marker == "*" else 30,
        edgecolors="white",
        linewidths=0.6,
        alpha=alpha,
        label=label,
        zorder=7,
    )


def plot_projection(
    ax: plt.Axes,
    history: np.ndarray,
    fresh_world: np.ndarray,
    old_pose: np.ndarray,
    estimated_pose: np.ndarray,
    target_pose: np.ndarray,
    method_name: str,
    axes: tuple[int, int],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    axis_labels: tuple[str, str],
    show_title: bool,
) -> None:
    color = METHOD_COLORS[method_name]
    ax.scatter(
        history[:, axes[0]],
        history[:, axes[1]],
        s=1.2,
        c=HISTORY_COLOR,
        alpha=0.30,
        rasterized=True,
    )
    ax.scatter(
        fresh_world[:, axes[0]],
        fresh_world[:, axes[1]],
        s=1.5,
        c=color,
        alpha=0.44,
        rasterized=True,
    )
    span = bounds[0][1] - bounds[0][0]
    draw_camera(ax, old_pose, axes, OLD_CAMERA_COLOR, "s", None, span * 0.075, alpha=0.9)
    draw_camera(ax, target_pose, axes, TARGET_COLOR, "*", None, span * 0.085, dashed=True)
    draw_camera(ax, estimated_pose, axes, color, "o", None, span * 0.085)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.grid(True, linewidth=0.5, alpha=0.22)
    if show_title:
        ax.set_title(method_name, color=color, fontsize=11, fontweight="bold", pad=8)


def metrics_table(ax: plt.Axes, methods: list[tuple[str, dict]], mode: str) -> None:
    ax.axis("off")
    rows = []
    for name, result in methods:
        residual = result.get("bounded_residual_deg")
        rows.append(
            [
                name,
                f"{float(result['camera_translation_error_m']):.2f}",
                f"{float(result['camera_rotation_error_deg']):.1f}",
                f"{float(result['yaw_error_deg']):.1f}",
                "-" if residual is None else f"{float(residual):.1f}",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["Candidate", "T err (m)", "R err (deg)", "Yaw (deg)", "Residual (deg)"],
        colWidths=[0.33, 0.16, 0.18, 0.16, 0.18],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d8df")
        if row == 0:
            cell.set_facecolor("#eef1f5")
            cell.set_text_props(weight="bold", color="#1f2937")
        else:
            cell.set_facecolor("white")
            if col == 0:
                method = rows[row - 1][0]
                cell.set_text_props(weight="bold", color=METHOD_COLORS[method])
    coarse = methods[0][1]
    refined = methods[1][1]
    rotation_gain = float(coarse["camera_rotation_error_deg"]) - float(
        refined["camera_rotation_error_deg"]
    )
    translation_gain = float(coarse["camera_translation_error_m"]) - float(
        refined["camera_translation_error_m"]
    )
    prefix = "V15 -> V15+V16" if mode == "v15" else "Fixed -> V16"
    ax.text(
        0.5,
        0.07,
        f"{prefix}\nrotation gain {rotation_gain:+.1f} deg  |  "
        f"translation gain {translation_gain:+.2f} m",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.4,
        color="#111827",
        fontweight="bold",
    )


def render_example(
    spec: dict,
    v10_case: dict,
    candidate_case: dict,
    args: argparse.Namespace,
) -> Path:
    local_dir = Path(v10_case["paths"]["human3r_local_reset"])
    methods = candidate_methods(candidate_case, str(spec["mode"]))
    poses = predicted_poses(local_dir)
    history, _ = history_background_cloud(local_dir, [0, 1], args.cloud_points_per_frame)
    fresh, _ = background_cloud(local_dir, 2, args.cloud_points_per_frame, seed=20260719)
    if len(history) == 0 or len(fresh) == 0:
        raise RuntimeError(f"No usable point cloud for {spec['case_name']}")

    transforms = [np.asarray(result["transform"], dtype=np.float32) for _, result in methods]
    fresh_world = [transform_points(transform, fresh) for transform in transforms]
    estimated_poses = [transform @ poses[2] for transform in transforms]
    oracle_transform = transforms[-1]
    target_pose = oracle_transform @ poses[2]
    old_pose = poses[1]
    all_poses = [old_pose, target_pose, *estimated_poses]
    top_bounds = projected_bounds([history, *fresh_world], all_poses, (0, 2))
    side_bounds = projected_bounds([history, *fresh_world], all_poses, (2, 1))

    fig = plt.figure(figsize=(16, 11), facecolor="#f7f8fa")
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(0.90, 1.18, 1.18),
        hspace=0.34,
        wspace=0.22,
        left=0.045,
        right=0.98,
        top=0.92,
        bottom=0.08,
    )
    pre_ax = fig.add_subplot(grid[0, 0])
    post_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[0, 2])
    pre_ax.imshow(load_rgb(local_dir, 1))
    post_ax.imshow(load_rgb(local_dir, 2))
    for ax, title in ((pre_ax, "Last pre-cut frame"), (post_ax, "First post-cut frame")):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    metrics_table(table_ax, methods, str(spec["mode"]))
    fig.text(
        0.5,
        0.022,
        "Point clouds: pre-cut history = gray; transformed fresh frame = candidate color.   "
        "Cameras: old = black square; GT target = magenta star; estimate = colored circle.",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#4b5563",
    )

    for column, ((method_name, _), transformed, estimated_pose) in enumerate(
        zip(methods, fresh_world, estimated_poses)
    ):
        plot_projection(
            fig.add_subplot(grid[1, column]),
            history,
            transformed,
            old_pose,
            estimated_pose,
            target_pose,
            method_name,
            (0, 2),
            top_bounds,
            ("world X (m)", "world Z (m)"),
            show_title=True,
        )
        plot_projection(
            fig.add_subplot(grid[2, column]),
            history,
            transformed,
            old_pose,
            estimated_pose,
            target_pose,
            method_name,
            (2, 1),
            side_bounds,
            ("world Z (m)", "world Y (m)"),
            show_title=False,
        )

    record = candidate_case["record"]
    fig.suptitle(
        f"{spec['caption']}\n{spec['case_name']}  |  source={record['source']}  |  "
        f"view angle={float(record.get('view_angle_deg', float('nan'))):.1f} deg",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        y=0.975,
    )
    output_path = args.output_dir / f"{spec['slug']}.png"
    fig.savefig(output_path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def build_contact_sheet(paths: list[Path], output_path: Path) -> None:
    thumb_width = 920
    gap = 22
    header = 52
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        height = int(round(image.height * thumb_width / image.width))
        thumbs.append((path, image.resize((thumb_width, height), Image.Resampling.LANCZOS)))
    thumb_height = max(image.height for _, image in thumbs)
    rows = (len(thumbs) + 1) // 2
    sheet = Image.new(
        "RGB",
        (thumb_width * 2 + gap * 3, rows * (thumb_height + header) + gap * (rows + 1)),
        "#f3f4f6",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(thumbs):
        row, col = divmod(index, 2)
        x = gap + col * (thumb_width + gap)
        y = gap + row * (thumb_height + header + gap)
        draw.text((x, y + 12), path.stem.replace("_", " "), fill="#111827")
        sheet.paste(image, (x, y + header))
    sheet.save(output_path, quality=92)


def write_index(paths: list[Path], args: argparse.Namespace) -> None:
    lines = [
        "# V16 qualitative examples",
        "",
        "Each figure compares the same cached Human3R fresh point cloud under three shot-level transforms.",
        "GT is used only by the Boundary Oracle panel and evaluation metrics.",
        "",
        "- `01`-`04`: representative improvements across all four sources.",
        "- `05`: a deliberate failure example showing false correction risk.",
        "- `06`: V15 coarse pose with and without the same V16 residual.",
        "- `contact_sheet.png`: compact overview.",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v10_cases, candidate_cases = load_maps(args)
    rendered = []
    for spec in EXAMPLES:
        name = str(spec["case_name"])
        if name not in v10_cases:
            raise KeyError(f"Case missing from V10 report: {name}")
        if name not in candidate_cases:
            raise KeyError(f"Case missing from V16 cache: {name}")
        path = render_example(spec, v10_cases[name], candidate_cases[name], args)
        rendered.append(path)
        print(path)
    contact_sheet = args.output_dir / "contact_sheet.png"
    build_contact_sheet(rendered, contact_sheet)
    write_index([*rendered, contact_sheet], args)
    print(contact_sheet)


if __name__ == "__main__":
    main()
