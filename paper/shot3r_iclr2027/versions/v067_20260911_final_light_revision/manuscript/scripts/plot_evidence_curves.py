#!/usr/bin/env python3
"""Generate the three evidence visualizations used in the v063 manuscript.

All curves are presentation-only transformations of retained aggregate records:
the fixed EgoHumans viewpoint bootstrap, the same-checkpoint state-isolation
audit, and the seven-method Harmony4D full-system timing study.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

TEAL = "#008F88"
BLUE_GRAY = "#526777"
CORAL = "#C45E68"
AMBER = "#D99A3D"
LIGHT_GRAY = "#D8DEE3"
DARK = "#26323B"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _finish_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=LIGHT_GRAY, linewidth=0.55, alpha=0.85)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#AAB4BC")
    ax.tick_params(width=0.6, length=2.7, colors="#4D5962")


def _save(fig: plt.Figure, stem: str, title: str) -> None:
    metadata = {
        "Title": title,
        "Author": "Anonymous ICLR submission",
        "Subject": "Shot3R quantitative evidence",
    }
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=metadata,
    )
    fig.savefig(
        FIGURES / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Title": title, "Creator": "Anonymous ICLR submission"},
    )
    plt.close(fig)


def plot_viewpoint_response() -> None:
    source = ROOT / "artifacts" / "egohumans_formal90" / "angle_strata_bootstrap.json"
    payload = json.loads(source.read_text())
    strata = ["small", "medium", "large", "extreme"]
    angles = np.array([payload["strata"][name]["mean_angle_deg"] for name in strata])

    method_names = list(payload["strata"][strata[0]]["methods"])
    human_key = next(name for name in method_names if "Human3R" in name)
    shot_key = next(name for name in method_names if "Human3R" not in name)

    specs = [
        ("W-MPJPE_mm", "W-MPJPE (mm)", "lower is better", (300, 1950)),
        ("IDF1", "IDF1", "higher is better", (0.28, 0.75)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.28))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.78, wspace=0.28)

    for ax, (metric, ylabel, direction, ylim) in zip(axes, specs):
        for key, label, color, marker, linestyle in (
            (human_key, "Human3R", BLUE_GRAY, "o", "--"),
            (shot_key, "Shot3R", TEAL, "s", "-"),
        ):
            records = [payload["strata"][name]["methods"][key][metric] for name in strata]
            means = np.array([record["mean"] for record in records])
            lows = np.array([record["ci95"][0] for record in records])
            highs = np.array([record["ci95"][1] for record in records])
            ax.fill_between(angles, lows, highs, color=color, alpha=0.10, linewidth=0)
            ax.plot(
                angles,
                means,
                color=color,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white" if key == human_key else color,
                markeredgewidth=0.9,
                linestyle=linestyle,
                label=label,
                zorder=3,
            )

        ax.set_xlim(20, 188)
        ax.set_ylim(*ylim)
        ax.set_xticks(angles, ["31°\nSmall", "84°\nMedium", "130°\nLarge", "177°\nExtreme"])
        ax.set_xlabel("Viewpoint change")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ·  {direction}", color=DARK, fontweight="bold")
        _finish_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.3,
        fontsize=7.2,
    )
    _save(fig, "egohumans_angle_strata", "Shot3R viewpoint-response curves")


def plot_state_drift() -> None:
    source = ROOT / "artifacts" / "egohumans_state_isolation" / "offset_curve.csv"
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    offsets = np.array([0] + [int(row["offset"]) for row in rows])
    series = {
        "continued_translation": np.array(
            [0.0] + [float(row["continued_translation_capture_macro"]) for row in rows]
        ),
        "reset_translation": np.array(
            [0.0] + [float(row["reset_translation_capture_macro"]) for row in rows]
        ),
        "continued_rotation": np.array(
            [0.0] + [float(row["continued_rotation_capture_macro"]) for row in rows]
        ),
        "reset_rotation": np.array(
            [0.0] + [float(row["reset_rotation_capture_macro"]) for row in rows]
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.18))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.78, wspace=0.27)
    panels = [
        ("translation", "Translation drift (m)", (0.0, 1.9)),
        ("rotation", "Rotation drift (deg)", (0.0, 18.3)),
    ]
    for ax, (suffix, ylabel, ylim) in zip(axes, panels):
        ax.plot(
            offsets,
            series[f"continued_{suffix}"],
            color=CORAL,
            linestyle="--",
            label="Carry over old state",
        )
        ax.plot(
            offsets,
            series[f"reset_{suffix}"],
            color=TEAL,
            label="Reinitialize state",
        )
        for x_mark in (16, 49):
            idx = int(np.where(offsets == x_mark)[0][0])
            ax.scatter(
                [x_mark, x_mark],
                [series[f"continued_{suffix}"][idx], series[f"reset_{suffix}"][idx]],
                c=[CORAL, TEAL],
                s=15,
                zorder=4,
                edgecolor="white",
                linewidth=0.55,
            )
        ax.set_xlim(0, 49)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 8, 16, 24, 32, 40, 49])
        ax.set_xlabel("Frames after transition")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, color=DARK, fontweight="bold")
        _finish_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.5,
        fontsize=7.2,
    )
    _save(fig, "state_carryover_drift", "Recurrent-state carry-over drift")


def plot_runtime_comparison() -> None:
    source = ROOT / "artifacts" / "runtime_h4d150" / "runtime_summary.csv"
    with source.open(newline="") as handle:
        records = {row["method"]: row for row in csv.DictReader(handle)}

    order = ["prompthmr", "tram", "josh", "onlinehmr", "trace", "human3r", "shot3r"]
    category = {
        "prompthmr": "Offline",
        "tram": "Offline",
        "josh": "Offline",
        "onlinehmr": "Semi-online",
        "trace": "Online",
        "human3r": "Online",
        "shot3r": "Online",
    }
    colors = {
        "Offline": "#8B969E",
        "Semi-online": AMBER,
        "Online": "#507FA3",
    }

    y = np.arange(len(order))[::-1]
    means = np.array([float(records[key]["mean_seconds_per_150_frames"]) for key in order])
    stds = np.array([float(records[key]["population_stdev_seconds"]) for key in order])
    fps = np.array([float(records[key]["micro_average_fps"]) for key in order])
    point_colors = [TEAL if key == "shot3r" else colors[category[key]] for key in order]

    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.77)
    xmin = 28.0
    for yi, mean, std, color in zip(y, means, stds, point_colors):
        ax.hlines(yi, xmin, mean, color=color, alpha=0.32, linewidth=2.0)
        ax.errorbar(
            mean,
            yi,
            xerr=std,
            fmt="o",
            markersize=5.2,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=3,
        )

    labels = [records[key]["display"] for key in order]
    ax.set_yticks(y, labels)
    for tick, key in zip(ax.get_yticklabels(), order):
        if key == "shot3r":
            tick.set_fontweight("bold")
            tick.set_color(TEAL)
    ax.set_xscale("log")
    ax.set_xlim(xmin, 1500)
    ax.set_xticks([30, 50, 100, 200, 500, 1000], ["30", "50", "100", "200", "500", "1000"])
    ax.set_xlabel("Wall-clock seconds per 150 frames (log scale, lower is better)")
    ax.set_title("Full-system runtime on one NVIDIA L20", color=DARK, fontweight="bold")
    _finish_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=5)

    for yi, mean, rate in zip(y, means, fps):
        ax.annotate(
            f"{mean:.1f}s  ·  {rate:.3f} FPS",
            xy=(mean, yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.6,
            color=DARK,
        )

    fig.legend(
        handles=[
            Patch(facecolor=colors["Offline"], label="Offline"),
            Patch(facecolor=colors["Semi-online"], label="Semi-online"),
            Patch(facecolor=colors["Online"], label="Online"),
            Patch(facecolor=TEAL, label="Shot3R"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.985),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.35,
        fontsize=6.9,
    )
    _save(fig, "runtime_comparison", "Seven-method full-system runtime")


if __name__ == "__main__":
    plot_viewpoint_response()
    plot_state_drift()
    plot_runtime_comparison()
