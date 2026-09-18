#!/usr/bin/env python3
"""Create the vector Figure 3 used by the Shot3R manuscript.

The plotted values are deterministic transformations of the fixed EgoHumans
stratum means in artifacts/egohumans_formal90/angle_strata.tex.  The header
values reproduce the dataset-balanced summary formerly shown in the separate
main-paper viewpoint table.  This script changes presentation only.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "egohumans_angle_strata"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.6,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# Fixed EgoHumans stratum means (Human3R, Shot3R).
human_w = np.array([767.7408046, 824.6593686, 1221.1347642, 1228.4077169])
shot_w = np.array([850.8621079, 801.1848583, 930.6648081, 886.0507997])
human_wa = np.array([331.5, 345.9, 388.6, 400.0])
shot_wa = np.array([357.1, 341.5, 331.1, 315.9])
human_idf1 = np.array([0.5585245570, 0.4136165220, 0.5014201970, 0.4042443394])
shot_idf1 = np.array([0.5929725463, 0.5723598652, 0.5579542676, 0.5585322233])

series = [
    100.0 * (human_w - shot_w) / human_w,
    100.0 * (human_wa - shot_wa) / human_wa,
    shot_idf1 - human_idf1,
]

titles = [
    "W-MPJPE reduction (%)",
    "WA-MPJPE reduction (%)",
    "IDF1 gain",
]
summaries = [
    "3-dataset mean   All  9.5%   |   Large-view  19.1%",
    "3-dataset mean   All  8.8%   |   Large-view  18.7%",
    "3-dataset mean   All  +0.099   |   Large-view  +0.195",
]
ylims = [(-15.0, 34.0), (-12.0, 27.0), (-0.025, 0.19)]
yticks = [
    [-10, 0, 10, 20, 30],
    [-10, 0, 10, 20],
    [0.00, 0.05, 0.10, 0.15],
]
labels = ["Small\n31°", "Medium\n84°", "Large\n130°", "Extreme\n177°"]

positive = "#00918C"  # Shot3R teal used throughout the paper.
negative = "#C85A63"  # Muted coral for a measured regression.
grid = "#D8DDE2"
text = "#27313A"

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.40), constrained_layout=False)
fig.subplots_adjust(left=0.060, right=0.995, bottom=0.205, top=0.765, wspace=0.31)

x = np.arange(4)
for panel, (ax, values, title, summary, ylim, ticks) in enumerate(
    zip(axes, series, titles, summaries, ylims, yticks)
):
    colors = [positive if value >= 0 else negative for value in values]
    bars = ax.bar(x, values, width=0.62, color=colors, edgecolor="white", linewidth=0.6, zorder=3)

    ax.axhline(0, color="#6C757D", linewidth=0.8, zorder=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=grid, linewidth=0.55)
    ax.xaxis.grid(False)
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.5, width=0.6, colors="#4E5963")
    ax.set_title(title, color=text, fontweight="bold", pad=7)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#AEB6BE")

    for rect, value in zip(bars, values):
        if panel < 2:
            label = f"{value:+.1f}%"
            offset = 1.15
        else:
            label = f"{value:+.3f}"
            offset = 0.006
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=6.7,
            fontweight="bold",
            color=text,
            clip_on=False,
        )

    ax.text(
        0.5,
        1.24,
        summary,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        color="#40505E",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#F2F5F6", edgecolor="#D5DCE1", linewidth=0.55),
    )

fig.legend(
    handles=[
        Patch(facecolor=positive, edgecolor="none", label="Shot3R improvement"),
        Patch(facecolor=negative, edgecolor="none", label="Regression"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=2,
    frameon=False,
    handlelength=1.05,
    handleheight=0.75,
    columnspacing=1.8,
    fontsize=6.8,
)

for suffix in (".pdf", ".svg"):
    fig.savefig(OUT.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
