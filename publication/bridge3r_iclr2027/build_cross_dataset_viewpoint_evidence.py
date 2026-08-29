#!/usr/bin/env python3
"""Build the BRIDGE3R all-view and extreme-viewpoint paper evidence.

The program consumes only the retained case/recording-level evaluation CSVs.
It does not run a model and never edits an existing result.  Paper-facing
tables, a machine-readable ledger, paired extreme-stratum uncertainty, and an
editable vector summary are written to a new manuscript artifact directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT = Path(__file__).resolve()
MOVIE3R = SCRIPT.parents[2]
WORKSPACE = SCRIPT.parents[3]
DEFAULT_OUTPUT = (
    WORKSPACE
    / "ICLR-paper/bridge3r_iclr2027/versions/"
    "v025_20260830_requirement_consolidation/manuscript/artifacts/"
    "cross_dataset_viewpoint"
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    case_csv: Path
    all_csv: Path
    method_column: str
    strict_key: str
    bridge_key: str
    cluster_column: str
    angle_column: str
    camera_metric: str
    camera_label: str
    all_unit: str
    extreme_unit: str


SPECS = (
    DatasetSpec(
        key="egobody",
        display="EgoBody",
        case_csv=MOVIE3R / "output/v20_egobody/formal/test/aggregate/case_metrics.csv",
        all_csv=MOVIE3R / "output/v20_egobody/formal/test/aggregate/recording_metrics.csv",
        method_column="name",
        strict_key="m0_strict_human3r",
        bridge_key="v19_ungated_translation_b050",
        cluster_column="recording",
        angle_column="angle_stratum",
        camera_metric="ATE_Sim3_m",
        camera_label="ATE-Sim3",
        all_unit="43 recordings",
        extreme_unit="43 farthest-view recordings",
    ),
    DatasetSpec(
        key="egohumans",
        display="EgoHumans",
        case_csv=MOVIE3R / "output/v19_egohumans/test/summary/case_metrics.csv",
        all_csv=MOVIE3R / "output/v19_egohumans/test/summary/case_metrics.csv",
        method_column="method",
        strict_key="m0_strict_human3r",
        bridge_key="v19_egohumans_frozen",
        cluster_column="capture",
        angle_column="angle_stratum",
        camera_metric="ATE_SE3_m",
        camera_label="ATE-SE3",
        all_unit="90 clips / 27 captures",
        extreme_unit="22 captures at >=150 deg",
    ),
    DatasetSpec(
        key="harmony4d",
        display="Harmony4D",
        case_csv=(
            MOVIE3R
            / "output/v17_harmony4d/unified_half_translation_audit/paper/case_metrics.csv"
        ),
        all_csv=(
            MOVIE3R
            / "output/v17_harmony4d/unified_half_translation_audit/paper/case_metrics.csv"
        ),
        method_column="method",
        strict_key="m0_strict_human3r",
        bridge_key="bridge3r_unified_half_translation",
        cluster_column="capture",
        angle_column="angle_stratum",
        camera_metric="ATE_Sim3_m",
        camera_label="ATE-Sim3",
        all_unit="88 cases / 25 captures",
        extreme_unit="25 extreme-view captures",
    ),
)


METRICS = ("W-MPJPE_mm", "WA-MPJPE_mm", "IDF1", "Coverage")
ERROR_METRICS = {"W-MPJPE_mm", "WA-MPJPE_mm", "ATE_Sim3_m", "ATE_SE3_m"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(rows: list[dict[str, str]], metrics: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"row_count": len(rows)}
    for metric in metrics:
        values = finite(number(row, metric) for row in rows)
        output[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "support": int(len(values)),
        }
    return output


def paired_rows(
    rows: list[dict[str, str]], spec: DatasetSpec
) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = row[spec.method_column]
        if method not in (spec.strict_key, spec.bridge_key):
            continue
        case_id = row.get("case_id")
        if not case_id:
            raise ValueError(f"{spec.display} case CSV has no case_id")
        if method in by_case[case_id]:
            raise ValueError(f"duplicate {spec.display} case/method: {case_id} {method}")
        by_case[case_id][method] = row
    missing = [case for case, pair in by_case.items() if len(pair) != 2]
    if missing:
        raise ValueError(f"{spec.display} has unpaired cases: {missing[:5]}")
    return [
        (by_case[case][spec.strict_key], by_case[case][spec.bridge_key])
        for case in sorted(by_case)
    ]


def improvement(metric: str, strict: float, bridge: float) -> float:
    return strict - bridge if metric in ERROR_METRICS else bridge - strict


def bootstrap_extreme(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    spec: DatasetSpec,
    metric: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    usable = [
        (strict, bridge)
        for strict, bridge in pairs
        if math.isfinite(number(strict, metric)) and math.isfinite(number(bridge, metric))
    ]
    clusters: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for strict, bridge in usable:
        cluster = strict.get(spec.cluster_column)
        if cluster != bridge.get(spec.cluster_column) or not cluster:
            raise ValueError(f"invalid {spec.display} paired cluster")
        clusters[cluster].append((strict, bridge))
    keys = sorted(clusters)
    if not keys:
        return {"support": 0, "cluster_count": 0, "mean_gain": None, "ci95": [None, None]}

    observed = np.asarray(
        [improvement(metric, number(a, metric), number(b, metric)) for a, b in usable],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_clusters = rng.integers(0, len(keys), size=len(keys))
        values: list[float] = []
        for cluster_index in selected_clusters:
            cluster_pairs = clusters[keys[int(cluster_index)]]
            selected_cases = rng.integers(0, len(cluster_pairs), size=len(cluster_pairs))
            for case_index in selected_cases:
                strict, bridge = cluster_pairs[int(case_index)]
                values.append(improvement(metric, number(strict, metric), number(bridge, metric)))
        samples[draw] = float(np.mean(values))

    tolerance = 1e-9
    wins = int(np.count_nonzero(observed > tolerance))
    losses = int(np.count_nonzero(observed < -tolerance))
    ties = int(len(observed) - wins - losses)
    return {
        "support": int(len(usable)),
        "cluster_count": int(len(keys)),
        "mean_gain": float(observed.mean()),
        "ci95": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
        "win_tie_loss": [wins, ties, losses],
        "positive_is_better": True,
    }


def filter_method(
    rows: list[dict[str, str]], method_column: str, method: str
) -> list[dict[str, str]]:
    return [row for row in rows if row.get(method_column) == method]


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, Any] = {
        "schema_version": "Bridge3R-cross-dataset-viewpoint-evidence-v1",
        "bootstrap": {"draws": int(args.bootstrap_draws), "seed": int(args.seed)},
        "datasets": {},
    }
    source_paths: set[Path] = set()
    for dataset_index, spec in enumerate(SPECS):
        all_rows = read_csv(spec.all_csv)
        case_rows = read_csv(spec.case_csv)
        source_paths.update((spec.all_csv, spec.case_csv))
        metrics = (*METRICS, spec.camera_metric)
        all_strict = filter_method(all_rows, spec.method_column, spec.strict_key)
        all_bridge = filter_method(all_rows, spec.method_column, spec.bridge_key)
        extreme_rows = [row for row in case_rows if row.get(spec.angle_column) == "extreme"]
        extreme_strict = filter_method(extreme_rows, spec.method_column, spec.strict_key)
        extreme_bridge = filter_method(extreme_rows, spec.method_column, spec.bridge_key)
        extreme_pairs = paired_rows(extreme_rows, spec)
        if len(extreme_strict) != len(extreme_bridge):
            raise ValueError(f"{spec.display} extreme denominator differs by method")
        ledger["datasets"][spec.key] = {
            "display": spec.display,
            "camera_metric": spec.camera_label,
            "camera_column": spec.camera_metric,
            "all_unit": spec.all_unit,
            "extreme_unit": spec.extreme_unit,
            "all": {
                "strict": summarize(all_strict, metrics),
                "bridge3r": summarize(all_bridge, metrics),
            },
            "extreme": {
                "strict": summarize(extreme_strict, metrics),
                "bridge3r": summarize(extreme_bridge, metrics),
                "paired": {
                    metric: bootstrap_extreme(
                        extreme_pairs,
                        spec,
                        metric,
                        int(args.bootstrap_draws),
                        int(args.seed) + dataset_index * 100 + metric_index,
                    )
                    for metric_index, metric in enumerate(metrics)
                },
            },
        }

    macro: dict[str, dict[str, float]] = {}
    for stratum in ("all", "extreme"):
        w_reductions, wa_reductions, id_gains, coverage_changes = [], [], [], []
        for spec in SPECS:
            row = ledger["datasets"][spec.key][stratum]
            strict, bridge = row["strict"], row["bridge3r"]
            sw = strict["W-MPJPE_mm"]["mean"]
            bw = bridge["W-MPJPE_mm"]["mean"]
            swa = strict["WA-MPJPE_mm"]["mean"]
            bwa = bridge["WA-MPJPE_mm"]["mean"]
            w_reductions.append(100.0 * (sw - bw) / sw)
            wa_reductions.append(100.0 * (swa - bwa) / swa)
            id_gains.append(bridge["IDF1"]["mean"] - strict["IDF1"]["mean"])
            coverage_changes.append(
                bridge["Coverage"]["mean"] - strict["Coverage"]["mean"]
            )
        macro[stratum] = {
            "W_relative_reduction_percent": float(np.mean(w_reductions)),
            "WA_relative_reduction_percent": float(np.mean(wa_reductions)),
            "IDF1_absolute_gain": float(np.mean(id_gains)),
            "Coverage_absolute_change": float(np.mean(coverage_changes)),
        }
    ledger["dataset_equal_macro"] = macro
    ledger["sources"] = [
        {
            "path": str(path.relative_to(WORKSPACE)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(source_paths)
    ]
    return ledger


def metric_value(row: dict[str, Any], metric: str) -> float:
    value = row[metric]["mean"]
    if value is None:
        return float("nan")
    return float(value)


def write_csvs(output: Path, ledger: dict[str, Any]) -> None:
    rows = []
    for spec in SPECS:
        dataset = ledger["datasets"][spec.key]
        for stratum in ("all", "extreme"):
            for method in ("strict", "bridge3r"):
                values = dataset[stratum][method]
                rows.append(
                    {
                        "dataset": spec.display,
                        "stratum": stratum,
                        "method": "Strict Human3R" if method == "strict" else "BRIDGE3R",
                        "unit": dataset[f"{stratum}_unit"],
                        "W-MPJPE_mm": metric_value(values, "W-MPJPE_mm"),
                        "W_support": values["W-MPJPE_mm"]["support"],
                        "WA-MPJPE_mm": metric_value(values, "WA-MPJPE_mm"),
                        "WA_support": values["WA-MPJPE_mm"]["support"],
                        "camera_metric": dataset["camera_metric"],
                        "camera_value_m": metric_value(values, dataset["camera_column"]),
                        "camera_support": values[dataset["camera_column"]]["support"],
                        "IDF1": metric_value(values, "IDF1"),
                        "Coverage": metric_value(values, "Coverage"),
                    }
                )
    with (output / "viewpoint_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    paired_rows_out = []
    for spec in SPECS:
        paired = ledger["datasets"][spec.key]["extreme"]["paired"]
        for metric, values in paired.items():
            paired_rows_out.append(
                {
                    "dataset": spec.display,
                    "metric": metric,
                    "support": values["support"],
                    "cluster_count": values["cluster_count"],
                    "mean_gain": values["mean_gain"],
                    "ci95_low": values["ci95"][0],
                    "ci95_high": values["ci95"][1],
                    "wins": values["win_tie_loss"][0],
                    "ties": values["win_tie_loss"][1],
                    "losses": values["win_tie_loss"][2],
                }
            )
    with (output / "extreme_paired_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows_out[0]))
        writer.writeheader()
        writer.writerows(paired_rows_out)


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace(">=", r"$\geq$")


def write_tex(output: Path, ledger: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Method & W$\downarrow$ & WA$\downarrow$ & Camera$\downarrow$ & IDF1$\uparrow$ & Cov.$\uparrow$ \\",
        r"\midrule",
    ]
    for index, spec in enumerate(SPECS):
        dataset = ledger["datasets"][spec.key]
        strict = dataset["extreme"]["strict"]
        bridge = dataset["extreme"]["bridge3r"]
        camera = dataset["camera_column"]
        w_red = 100.0 * (
            metric_value(strict, "W-MPJPE_mm") - metric_value(bridge, "W-MPJPE_mm")
        ) / metric_value(strict, "W-MPJPE_mm")
        wa_red = 100.0 * (
            metric_value(strict, "WA-MPJPE_mm") - metric_value(bridge, "WA-MPJPE_mm")
        ) / metric_value(strict, "WA-MPJPE_mm")
        lines.extend(
            [
                (
                    f"{tex_escape(spec.display)} & Strict \\humanthree{{}} & "
                    f"{metric_value(strict, 'W-MPJPE_mm'):.1f} & "
                    f"{metric_value(strict, 'WA-MPJPE_mm'):.1f} & "
                    f"{metric_value(strict, camera):.3f} & "
                    f"{metric_value(strict, 'IDF1'):.3f} & "
                    f"{metric_value(strict, 'Coverage'):.3f} \\\\"
                ),
                (
                    f" & \\method{{}} & {metric_value(bridge, 'W-MPJPE_mm'):.1f} "
                    f"({abs(w_red):.1f}\\%$\\{('downarrow' if w_red >= 0 else 'uparrow')}$) & "
                    f"{metric_value(bridge, 'WA-MPJPE_mm'):.1f} "
                    f"({abs(wa_red):.1f}\\%$\\{('downarrow' if wa_red >= 0 else 'uparrow')}$) & "
                    f"{metric_value(bridge, camera):.3f} & "
                    f"{metric_value(bridge, 'IDF1'):.3f} & "
                    f"{metric_value(bridge, 'Coverage'):.3f} \\\\"
                ),
            ]
        )
        if index != len(SPECS) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "extreme_main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    macro = ledger["dataset_equal_macro"]
    macro_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Viewpoint subset & W reduction & WA reduction & IDF1 gain & Coverage change \\",
        r"\midrule",
        (
            f"All views & {macro['all']['W_relative_reduction_percent']:.1f}\\% & "
            f"{macro['all']['WA_relative_reduction_percent']:.1f}\\% & "
            f"{macro['all']['IDF1_absolute_gain']:+.3f} & "
            f"{macro['all']['Coverage_absolute_change']:+.3f} \\\\"
        ),
        (
            f"Extreme/farthest & {macro['extreme']['W_relative_reduction_percent']:.1f}\\% & "
            f"{macro['extreme']['WA_relative_reduction_percent']:.1f}\\% & "
            f"{macro['extreme']['IDF1_absolute_gain']:+.3f} & "
            f"{macro['extreme']['Coverage_absolute_change']:+.3f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (output / "dataset_equal_macro_table.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8"
    )


def write_figure(output: Path, ledger: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt

    macro = ledger["dataset_equal_macro"]
    panels = (
        ("W-MPJPE", "W_relative_reduction_percent", "Relative reduction (%)"),
        ("WA-MPJPE", "WA_relative_reduction_percent", "Relative reduction (%)"),
        ("IDF1", "IDF1_absolute_gain", "Absolute gain"),
    )
    colors = ("#8DA0CB", "#D95F5F")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
    for axis, (title, key, ylabel) in zip(axes, panels):
        values = [macro["all"][key], macro["extreme"][key]]
        bars = axis.bar([0, 1], values, color=colors, width=0.62)
        axis.set_xticks([0, 1], ["All views", "Extreme /\nfarthest"])
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        upper = max(values) * 1.22 if max(values) > 0 else 1.0
        axis.set_ylim(0.0, upper)
        for bar, value in zip(bars, values):
            suffix = "%" if "percent" in key else ""
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.025,
                f"{value:.1f}{suffix}" if suffix else f"{value:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    fig.tight_layout(w_pad=1.0)
    fig.savefig(output / "all_vs_extreme_macro.svg", bbox_inches="tight")
    fig.savefig(output / "all_vs_extreme_macro.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(output: Path, ledger: dict[str, Any]) -> None:
    macro = ledger["dataset_equal_macro"]
    source_lines = "\n".join(
        f"- `{row['path']}` — SHA-256 `{row['sha256']}`" for row in ledger["sources"]
    )
    text = f"""# Cross-dataset viewpoint evidence

Generated by `Movie3R/publication/bridge3r_iclr2027/{SCRIPT.name}` from retained
case/recording-level result files. No model inference or manual paper-number
entry is performed by the generator.

Dataset-equal macro:

- all-view W/WA reduction: {macro['all']['W_relative_reduction_percent']:.3f}% / {macro['all']['WA_relative_reduction_percent']:.3f}%
- extreme/farthest W/WA reduction: {macro['extreme']['W_relative_reduction_percent']:.3f}% / {macro['extreme']['WA_relative_reduction_percent']:.3f}%
- IDF1 gain: {macro['all']['IDF1_absolute_gain']:+.6f} -> {macro['extreme']['IDF1_absolute_gain']:+.6f}
- Coverage change: {macro['all']['Coverage_absolute_change']:+.6f} -> {macro['extreme']['Coverage_absolute_change']:+.6f}

The macro weights the three datasets equally. Camera ATE is not averaged
because EgoHumans uses ATE-SE3 while EgoBody and Harmony4D use ATE-Sim3.

Sources:

{source_lines}
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ledger = build(args)
    (args.output / "viewpoint_evidence.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csvs(args.output, ledger)
    write_tex(args.output, ledger)
    write_figure(args.output, ledger)
    write_readme(args.output, ledger)


if __name__ == "__main__":
    main()
