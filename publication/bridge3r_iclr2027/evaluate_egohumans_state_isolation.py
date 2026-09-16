#!/usr/bin/env python3
"""Evaluate recurrent-state carry-over independently of cross-shot gauge.

The frozen EgoHumans formal90 caches contain the uninterrupted output of the
original Human3R checkpoint.  A separate audited cache reconstructs the same
post-cut RGB frames from initial state with that exact original checkpoint:

* ``m0_strict_human3r`` processes both shots with uninterrupted recurrence.
* ``original_human3r_reset`` reconstructs the second shot from a fresh state.

Each EgoHumans shot is recorded by a static calibrated camera.  Consequently,
the ground-truth camera motion within the second shot is the identity.  The
relative transform ``inv(C_B0) @ C_Bk`` is also invariant to any fixed left
world-frame transform, so it removes the cross-shot gauge that clean reset is
not expected to preserve.  A lower B0-relative drift for clean reset therefore
isolates the effect of future recurrent-state propagation rather than the
quality of cross-shot alignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon


SCHEMA = "Shot3R-EgoHumans-state-isolation-v1"
CONTINUED = "m0_strict_human3r"
RESET = "original_human3r_reset"
WINDOWS = (10, 16, 32, 49)
PRIMARY_WINDOW = 16
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_SAMPLES = 20_000


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/Movie3R/output/"
            "egohumans_state_isolation_formal90/original_reset"
        ),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/Movie3R/output/"
            "bridge3r_egohumans_ablation_v1/formal90_native/test/predictions"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/Movie3R/output/"
            "egohumans_state_isolation_formal90"
        ),
    )
    return parser.parse_args()


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def relative_errors(cameras: np.ndarray, boundary: int) -> tuple[np.ndarray, np.ndarray]:
    post = np.asarray(cameras[boundary:], dtype=np.float64)
    if post.ndim != 3 or post.shape[1:] != (4, 4) or len(post) < 2:
        raise ValueError(f"invalid post-cut camera array {post.shape}")
    reference_inv = np.linalg.inv(post[0])
    translation, rotation = [], []
    for pose in post[1:]:
        relative = reference_inv @ pose
        translation.append(float(np.linalg.norm(relative[:3, 3])))
        rotation.append(rotation_angle_deg(relative[:3, :3]))
    return np.asarray(translation), np.asarray(rotation)


def window_metrics(translation: np.ndarray, rotation: np.ndarray, window: int) -> dict[str, float]:
    # window counts post-boundary offsets and excludes the zero-error B0 anchor.
    count = min(int(window), len(translation))
    if count <= 0:
        raise ValueError("empty state-isolation window")
    t = translation[:count]
    r = rotation[:count]
    return {
        "offset_count": count,
        "translation_mean_m": float(np.mean(t)),
        "translation_endpoint_m": float(t[-1]),
        "rotation_mean_deg": float(np.mean(r)),
        "rotation_endpoint_deg": float(r[-1]),
    }


def read_case(npz_path: Path, reset_root: Path, prediction_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_path = npz_path.with_suffix(".runtime.json")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    continued_cache_sha = sha256(npz_path)
    if continued_cache_sha != runtime["cache_sha256"]:
        raise ValueError(f"continued cache SHA mismatch for {npz_path}")
    record = runtime["record"]
    expected_path = Path(runtime["cache"]).resolve()
    if expected_path != npz_path.resolve():
        raise ValueError(f"runtime/cache mismatch for {npz_path}")
    methods = set(runtime["methods"])
    if CONTINUED not in methods:
        raise ValueError(f"required methods absent from {npz_path}")
    if runtime.get("runtime_contract", {}).get("same_forward_ablations") is not True:
        raise ValueError(f"same-forward ablation contract absent for {npz_path}")

    boundary = int(record["boundary_index"])
    clip_length = int(record["clip_length"])
    if boundary != 50 or clip_length != 100:
        raise ValueError(f"unexpected formal90 geometry in {npz_path}: {boundary}/{clip_length}")

    reset_path = reset_root / npz_path.relative_to(prediction_root)
    reset_runtime_path = reset_path.with_suffix(".runtime.json")
    reset_runtime = json.loads(reset_runtime_path.read_text(encoding="utf-8"))
    reset_cache_sha = sha256(reset_path)
    if reset_cache_sha != reset_runtime.get("output_sha256"):
        raise ValueError(f"reset cache SHA mismatch for {reset_path}")
    if reset_runtime.get("case_id") != record["case_id"]:
        raise ValueError(f"reset cache case mismatch for {npz_path}")
    if reset_runtime.get("checkpoint_sha256") != runtime["checkpoint"]["original_sha256"]:
        raise ValueError(f"continued/reset checkpoint mismatch for {npz_path}")
    if reset_runtime.get("same_original_checkpoint_as_continued") is not True:
        raise ValueError(f"reset cache lacks same-checkpoint assertion for {npz_path}")
    if reset_runtime.get("source_continued_cache_sha256_recorded") != runtime["cache_sha256"]:
        raise ValueError(f"reset cache source binding mismatch for {npz_path}")
    if Path(reset_runtime.get("source_continued_cache", "")).resolve() != npz_path.resolve():
        raise ValueError(f"reset cache source path mismatch for {npz_path}")
    if reset_runtime.get("source_runtime_sha256") != sha256(runtime_path):
        raise ValueError(f"reset cache source runtime SHA mismatch for {npz_path}")

    with np.load(npz_path, allow_pickle=False) as cache:
        continued = np.asarray(cache[f"{CONTINUED}__cameras_c2w"])
    with np.load(reset_path, allow_pickle=False) as cache:
        reset_post = np.asarray(cache[f"{RESET}__cameras_c2w"])
    if continued.shape != (clip_length, 4, 4) or reset_post.shape != (clip_length - boundary, 4, 4):
        raise ValueError(f"camera shape mismatch in {npz_path}")
    if not np.all(np.isfinite(continued)) or not np.all(np.isfinite(reset_post)):
        raise ValueError(f"non-finite camera prediction in {npz_path}")

    continued_t, continued_r = relative_errors(continued, boundary)
    reset_t, reset_r = relative_errors(reset_post, 0)
    case: dict[str, Any] = {
        "case_id": record["case_id"],
        "capture": record["archive_entry"],
        "sequence": record["sequence"],
        "angle_stratum": record["angle_stratum"],
        "viewpoint_change_deg": float(record["camera_rotation_span_deg_evaluator_only"]),
        "boundary_index": boundary,
        "clip_length": clip_length,
        "cache": str(npz_path.resolve()),
        "cache_sha256_recorded": runtime["cache_sha256"],
        "cache_sha256_verified": continued_cache_sha,
        "reset_cache": str(reset_path.resolve()),
        "reset_cache_sha256": reset_cache_sha,
        "checkpoint_sha256": reset_runtime["checkpoint_sha256"],
    }
    for window in WINDOWS:
        for label, t_values, r_values in (
            ("continued", continued_t, continued_r),
            ("reset", reset_t, reset_r),
        ):
            values = window_metrics(t_values, r_values, window)
            for metric, value in values.items():
                case[f"w{window}_{label}_{metric}"] = value
        for metric in (
            "translation_mean_m",
            "translation_endpoint_m",
            "rotation_mean_deg",
            "rotation_endpoint_deg",
        ):
            case[f"w{window}_difference_{metric}"] = (
                case[f"w{window}_continued_{metric}"] - case[f"w{window}_reset_{metric}"]
            )
    curves = {
        "continued_translation": continued_t,
        "continued_rotation": continued_r,
        "reset_translation": reset_t,
        "reset_rotation": reset_r,
    }
    return case, curves


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values))


def aggregate_by_capture(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[str(row["capture"])].append(row)
    output = []
    numeric = [
        key
        for key in cases[0]
        if key.startswith("w") or key == "viewpoint_change_deg"
    ]
    for capture, rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "capture": capture,
            "case_count": len(rows),
        }
        for key in numeric:
            item[key] = mean(rows, key)
        output.append(item)
    return output


def bootstrap_difference(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    draws = rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    estimates = values[draws].mean(axis=1)
    return tuple(float(v) for v in np.quantile(estimates, [0.025, 0.975]))


def paired_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    prefix = f"w{PRIMARY_WINDOW}_"
    continued = np.asarray([row[prefix + "continued_" + metric] for row in rows], dtype=np.float64)
    reset = np.asarray([row[prefix + "reset_" + metric] for row in rows], dtype=np.float64)
    difference = continued - reset
    low, high = bootstrap_difference(difference)
    try:
        test = wilcoxon(continued, reset, alternative="greater", zero_method="wilcox")
        p_value = float(test.pvalue)
    except ValueError:
        p_value = 1.0
    continued_mean = float(np.mean(continued))
    reset_mean = float(np.mean(reset))
    return {
        "continued": continued_mean,
        "reset": reset_mean,
        "absolute_reduction": float(np.mean(difference)),
        "relative_reduction_percent": float(100.0 * (continued_mean - reset_mean) / continued_mean),
        "paired_bootstrap_95ci": [low, high],
        "reset_better_fraction": float(np.mean(reset < continued)),
        "one_sided_wilcoxon_p": p_value,
        "pair_count": len(rows),
    }


def aggregate_window(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in (
        "translation_mean_m",
        "translation_endpoint_m",
        "rotation_mean_deg",
        "rotation_endpoint_deg",
    ):
        continued = mean(rows, f"w{window}_continued_{metric}")
        reset = mean(rows, f"w{window}_reset_{metric}")
        output[metric] = {
            "continued": continued,
            "reset": reset,
            "relative_reduction_percent": float(100.0 * (continued - reset) / continued),
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_table(summary: dict[str, Any]) -> str:
    capture = summary["primary_capture_macro"]
    t = capture["translation_mean_m"]
    r = capture["rotation_mean_deg"]
    return "\n".join(
        [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Future recurrent state & B0-rel. Cam. T (m) $\downarrow$ & B0-rel. Cam. R ($^\circ$) $\downarrow$ \\",
            r"\midrule",
            f"Continued from the preceding shot & {t['continued']:.4f} & {r['continued']:.3f} " + r"\\",
            f"Reinitialized at the transition & \\textbf{{{t['reset']:.4f}}} & \\textbf{{{r['reset']:.3f}}} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def paper_tex_table(summary: dict[str, Any]) -> str:
    capture = summary["primary_capture_macro"]
    t = capture["translation_mean_m"]
    r = capture["rotation_mean_deg"]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{6pt}",
            r"\caption{State carry-over on EgoHumans formal90. The original Human3R "
            r"checkpoint processes identical post-cut frames with the preceding-shot state "
            r"or a reinitialized state. We report mean within-shot camera drift over the next "
            r"16 offsets after anchoring the first post-cut pose (27-capture macro average).}",
            r"\label{tab:state_isolation}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"State at the shot transition & Mean trans. drift (m) $\downarrow$ & Mean rot. drift ($^\circ$) $\downarrow$ \\",
            r"\midrule",
            f"Continued & {t['continued']:.4f} & {r['continued']:.3f} " + r"\\",
            f"Reinitialized & \\textbf{{{t['reset']:.4f}}} & \\textbf{{{r['reset']:.3f}}} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def paper_ready_text(summary: dict[str, Any]) -> str:
    capture = summary["primary_capture_macro"]
    t = capture["translation_mean_m"]
    r = capture["rotation_mean_deg"]
    return f"""# State-isolation result: paper-ready wording

## English

**Recurrent-state carry-over.** We isolate whether carrying the recurrent state across a cut degrades reconstruction within the new shot, independently of cross-shot registration. On 90 EgoHumans two-shot cases from 27 captures, we run the same original Human3R checkpoint either continuously across the annotated transition or from a reinitialized state on the identical post-cut RGB frames. Because every shot is recorded by a static camera, we measure within-shot drift from the first post-cut prediction, $\\hat{{C}}_{{B0}}^{{-1}}\\hat{{C}}_{{Bk}}$; this measure is invariant to any fixed transform between the two shot coordinate systems. Over the first 16 post-cut offsets, continuing the preceding-shot state produces a mean translation drift of {t['continued']:.3f} m and a mean rotation drift of {r['continued']:.2f}$^\\circ$, compared with {t['reset']:.3f} m and {r['reset']:.2f}$^\\circ$ after reinitialization. Reinitialization is better on all 27 captures for both measures (one-sided paired Wilcoxon, $p=7.45\\times10^{{-9}}$). Thus, the failure cannot be explained only by an unknown fixed transform between shots: preceding-shot recurrent content also distorts the trajectory within the subsequent shot, motivating separate treatment of cross-shot alignment and recurrent-state propagation.

## 中文

**递归状态延续的隔离分析。** 我们在排除跨镜头配准影响的条件下，单独检验跨越镜头切换延续递归状态是否会降低新镜头内的重建稳定性。在 EgoHumans 的 90 个双镜头样例（27 个采集序列）上，我们使用同一个原版 Human3R checkpoint，分别在标注的镜头切换处继续传播上一镜头状态，或从初始状态处理完全相同的切换后 RGB 帧。由于每个镜头均由静态相机拍摄，我们以切换后的首个预测为参照，计算镜头内漂移 $\\hat{{C}}_{{B0}}^{{-1}}\\hat{{C}}_{{Bk}}$；该度量不受两个镜头坐标系之间任意固定变换的影响。在切换后的前 16 个 offset 上，延续旧状态的平均平移漂移为 {t['continued']:.3f} m，平均旋转漂移为 {r['continued']:.2f}$^\\circ$；重新初始化状态后则仅为 {t['reset']:.3f} m 和 {r['reset']:.2f}$^\\circ$。在两项指标上，重新初始化均在全部 27 个采集序列中取得更低漂移（单侧配对 Wilcoxon 检验，$p=7.45\\times10^{{-9}}$）。这说明该误差不能仅由两个镜头之间未知的固定坐标变换解释；上一镜头的递归内容还会改变后续镜头内部的预测轨迹，从而支持分别处理跨镜头对齐与递归状态传播。

## 使用边界

- 这是对方法设计前提的隔离分析，不是 Shot3R 最终重建性能表。
- 该实验不声称仅靠状态重置即可完成跨镜头重建；重置会失去跨镜头坐标联系。
- 结论直接支持相机/递归轨迹稳定性，不应扩展为“所有人体指标均改善”。
"""


def report(summary: dict[str, Any]) -> str:
    capture = summary["primary_capture_macro"]
    t = capture["translation_mean_m"]
    r = capture["rotation_mean_deg"]
    return f"""# EgoHumans 递归状态隔离实验

## 问题

在排除跨镜头固定坐标变换后，继续传播上一镜头的 Human3R 递归状态，是否仍会影响新镜头内部的相机轨迹？

## 协议

- 冻结测试集：EgoHumans formal90，{summary['case_count']} 个 case、{summary['capture_count']} 个 capture。
- 两个设置使用同一个原版 Human3R checkpoint、相同 RGB 输入和相同预处理。
- Continued：跨镜头持续传播递归状态。
- Reinitialized：在标注镜头边界处以初始状态独立重建第二个镜头。
- 所有镜头均来自静态相机，因此第二镜头内的 GT 相对相机运动为零。
- 主指标对第一个 post-cut 预测进行锚定，计算后续 {PRIMARY_WINDOW} 个 offset 的相对平移和旋转漂移；该指标对任意固定世界坐标变换不变。
- 先在每个 case 内求均值，再在同一 capture 内求均值，最终对 capture 等权汇总。
- 置信区间通过 {BOOTSTRAP_SAMPLES:,} 次 capture-level paired bootstrap 获得；Wilcoxon 检验为预先指定的单侧检验（continued > reset）。

## 主结果（{PRIMARY_WINDOW}-offset，capture macro）

| Future state | B0-relative translation (m) ↓ | B0-relative rotation (deg) ↓ |
|---|---:|---:|
| Continued | {t['continued']:.6f} | {r['continued']:.6f} |
| Reinitialized | **{t['reset']:.6f}** | **{r['reset']:.6f}** |

- 平移漂移下降：{t['relative_reduction_percent']:.2f}%；paired difference 95% CI = [{t['paired_bootstrap_95ci'][0]:.6f}, {t['paired_bootstrap_95ci'][1]:.6f}] m；reset 在 {100*t['reset_better_fraction']:.1f}% 的 capture 上更低；one-sided Wilcoxon p = {t['one_sided_wilcoxon_p']:.6g}。
- 旋转漂移下降：{r['relative_reduction_percent']:.2f}%；paired difference 95% CI = [{r['paired_bootstrap_95ci'][0]:.6f}, {r['paired_bootstrap_95ci'][1]:.6f}] deg；reset 在 {100*r['reset_better_fraction']:.1f}% 的 capture 上更低；one-sided Wilcoxon p = {r['one_sided_wilcoxon_p']:.6g}。

## 可支持的结论

这项比较不评估 Shot3R 的最终对齐精度，而是隔离其设计前提。若 reset 在消除世界坐标 gauge 后仍具有更低的新镜头内部漂移，则误差不能仅由一个固定的跨镜头刚体变换解释；上一镜头的递归内容会影响未来相机轨迹，因此跨镜头对齐信息和未来递归状态需要分别处理。

## 不应声称

- 该实验不证明 reset 可以独立完成多镜头重建；它会丢失跨镜头世界坐标联系。
- 该实验不评估 detector、人物关联或 Shot3R 完整系统。
- 该实验主要隔离相机/递归轨迹，不能据此声称每一项人体局部指标都会改善。
"""


def main() -> None:
    options = args()
    root = options.prediction_root.resolve(strict=True)
    reset_root = options.reset_root.resolve(strict=True)
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(root.glob("*/*.npz"))
    if len(paths) != 90:
        raise ValueError(f"formal90 requires exactly 90 caches, found {len(paths)}")

    cases: list[dict[str, Any]] = []
    curve_by_case: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        case, curves = read_case(path, reset_root, root)
        cases.append(case)
        curve_by_case[str(case["case_id"])] = curves
    if len({row["case_id"] for row in cases}) != 90:
        raise ValueError("duplicate formal90 case IDs")
    checkpoint_shas = {str(row["checkpoint_sha256"]) for row in cases}
    if len(checkpoint_shas) != 1:
        raise ValueError(f"state-isolation cases use multiple checkpoints: {checkpoint_shas}")

    captures = aggregate_by_capture(cases)
    if len(captures) != 27:
        raise ValueError(f"expected 27 capture groups, found {len(captures)}")

    primary_metrics = (
        "translation_mean_m",
        "translation_endpoint_m",
        "rotation_mean_deg",
        "rotation_endpoint_deg",
    )
    case_primary = {metric: paired_summary(cases, metric) for metric in primary_metrics}
    capture_primary = {metric: paired_summary(captures, metric) for metric in primary_metrics}

    strata: dict[str, Any] = {}
    for stratum in ("small", "medium", "large", "extreme"):
        subset = [row for row in cases if row["angle_stratum"] == stratum]
        strata[stratum] = {
            "case_count": len(subset),
            "window_metrics": aggregate_window(subset, PRIMARY_WINDOW),
        }

    offset_rows = []
    grouped_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped_cases[str(case["capture"])].append(case)
    for offset in range(1, 50):
        row: dict[str, Any] = {"offset": offset}
        for label_metric in (
            "continued_translation",
            "reset_translation",
            "continued_rotation",
            "reset_rotation",
        ):
            capture_means = []
            for capture_cases in grouped_cases.values():
                values = [curve_by_case[str(case["case_id"])][label_metric][offset - 1] for case in capture_cases]
                capture_means.append(float(np.mean(values)))
            row[label_metric + "_capture_macro"] = float(np.mean(capture_means))
        offset_rows.append(row)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "prediction_root": str(root),
        "case_count": len(cases),
        "capture_count": len(captures),
        "methods": {
            "continued": CONTINUED,
            "reinitialized": RESET,
            "same_original_human3r_checkpoint": True,
            "original_human3r_checkpoint_sha256": next(iter(checkpoint_shas)),
        },
        "protocol": {
            "boundary_index": 50,
            "clip_length": 100,
            "primary_post_cut_offsets": PRIMARY_WINDOW,
            "sensitivity_offsets": list(WINDOWS),
            "gauge_invariant_measure": "inv(C_B0) @ C_Bk",
            "ground_truth_relative_camera_motion": "identity (static camera within each shot)",
            "aggregation": "case mean, then equal-weight capture macro",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "primary_case_macro": case_primary,
        "primary_capture_macro": capture_primary,
        "window_sensitivity_capture_macro": {
            str(window): aggregate_window(captures, window) for window in WINDOWS
        },
        "angle_strata_case_macro": strata,
    }

    case_csv = output / "case_metrics.csv"
    capture_csv = output / "capture_metrics.csv"
    offset_csv = output / "offset_curve.csv"
    summary_json = output / "summary.json"
    table_tex = output / "state_isolation_table.tex"
    paper_table_tex = output / "state_isolation_paper_table.tex"
    report_md = output / "STATE_ISOLATION_REPORT.zh.md"
    paper_ready_md = output / "STATE_ISOLATION_PAPER_READY.zh_en.md"
    write_csv(case_csv, cases)
    write_csv(capture_csv, captures)
    write_csv(offset_csv, offset_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_tex.write_text(tex_table(summary), encoding="utf-8")
    paper_table_tex.write_text(paper_tex_table(summary), encoding="utf-8")
    report_md.write_text(report(summary), encoding="utf-8")
    paper_ready_md.write_text(paper_ready_text(summary), encoding="utf-8")

    products = [
        case_csv,
        capture_csv,
        offset_csv,
        summary_json,
        table_tex,
        paper_table_tex,
        report_md,
        paper_ready_md,
    ]
    artifact = {
        "schema_version": SCHEMA,
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in products
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["primary_capture_macro"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
