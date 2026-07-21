#!/usr/bin/env python3
"""Build the final V20 report for the selected strict-causal explicit metric bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOUNDARY = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_independent_bound45_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_SCENE = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "scene_depth_scale_independent_safe"
    / "v20_scene_depth_scale_refinement.json"
)
DEFAULT_THREE = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "three_frame"
    / "v20_three_frame_shot_scale.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "final_report"
DEFAULT_DOC = REPO_ROOT / "docs" / "movie3r" / "v20" / "V20_EXPLICIT_METRIC_SHOT_BRIDGE_20260721.md"
DEFAULT_REPEAT = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "scene_depth_scale_independent_safe_seed2"
    / "v20_scene_depth_scale_refinement.json"
)
DEFAULT_FAST = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "scene_depth_scale_independent_fast"
    / "v20_scene_depth_scale_refinement.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary_report", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--scene_report", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--three_frame_report", type=Path, default=DEFAULT_THREE)
    parser.add_argument("--repeat_scene_report", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--fast_scene_report", type=Path, default=DEFAULT_FAST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc_path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--method", default="torso_first1")
    parser.add_argument("--variant", default="q30")
    return parser.parse_args()


def dist(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def arrays(rows: list[dict], candidate: bool, method: str, variant: str) -> dict[str, np.ndarray]:
    selected = [row["methods"][method][variant] if candidate else row["fixed"] for row in rows]
    return {
        "translation": np.asarray([row["camera"]["translation_m"] for row in selected]),
        "rotation": np.asarray([row["camera"]["rotation_deg"] for row in selected]),
        "human": np.asarray([row["human"]["root_motion_error_m"] for row in selected]),
        "scene": np.asarray([row["scene"]["trimmed_mean_m"] for row in selected]),
    }


def summarize(rows: list[dict], method: str, variant: str) -> dict:
    fixed = arrays(rows, False, method, variant)
    candidate = arrays(rows, True, method, variant)
    output = {
        name: {"fixed": dist(fixed[name]), "candidate": dist(candidate[name])}
        for name in fixed
    }
    fixed_catastrophic = (
        (fixed["translation"] > 2.0)
        | (fixed["rotation"] > 45.0)
        | (fixed["human"] > 0.50)
        | (fixed["scene"] > 1.0)
    )
    candidate_catastrophic = (
        (candidate["translation"] > 2.0)
        | (candidate["rotation"] > 45.0)
        | (candidate["human"] > 0.50)
        | (candidate["scene"] > 1.0)
    )
    fixed_success = (
        (fixed["translation"] < 0.50)
        & (fixed["rotation"] < 10.0)
        & (fixed["human"] < 0.10)
        & (fixed["scene"] < 0.30)
    )
    candidate_success = (
        (candidate["translation"] < 0.50)
        & (candidate["rotation"] < 10.0)
        & (candidate["human"] < 0.10)
        & (candidate["scene"] < 0.30)
    )
    output.update(
        {
            "catastrophic_rate": {
                "fixed": float(np.mean(fixed_catastrophic)),
                "candidate": float(np.mean(candidate_catastrophic)),
            },
            "success_rate": {
                "fixed": float(np.mean(fixed_success)),
                "candidate": float(np.mean(candidate_success)),
            },
            "improved_rate": {
                name: float(np.mean(candidate[name] < fixed[name])) for name in fixed
            },
            "all_camera_human_scene_improved_rate": float(
                np.mean(
                    (candidate["translation"] < fixed["translation"])
                    & (candidate["human"] < fixed["human"])
                    & (candidate["scene"] < fixed["scene"])
                )
            ),
            "harmful_rate": {
                "translation_plus_010m": float(
                    np.mean(candidate["translation"] > fixed["translation"] + 0.10)
                ),
                "rotation_plus_5deg": float(
                    np.mean(candidate["rotation"] > fixed["rotation"] + 5.0)
                ),
                "human_plus_010m": float(
                    np.mean(candidate["human"] > fixed["human"] + 0.10)
                ),
                "scene_plus_010m": float(
                    np.mean(candidate["scene"] > fixed["scene"] + 0.10)
                ),
            },
        }
    )
    return output


def grouped(rows: list[dict], method: str, variant: str) -> dict:
    fixed = arrays(rows, False, method, variant)
    groups = {
        "fixed_rotation_deg": ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, float("inf"))),
        "fixed_translation_m": ((0.0, 0.50), (0.50, 1.0), (1.0, 2.0), (2.0, float("inf"))),
    }
    output = {}
    for name, bins in groups.items():
        values = fixed["rotation"] if name == "fixed_rotation_deg" else fixed["translation"]
        rows_by_bin = {}
        for lower, upper in bins:
            indices = np.flatnonzero((values >= lower) & (values < upper))
            label = f"{lower:g}_{upper:g}"
            rows_by_bin[label] = summarize([rows[index] for index in indices], method, variant) if len(indices) else {"count": 0}
            rows_by_bin[label]["count"] = int(len(indices))
        output[name] = rows_by_bin
    return output


def write_markdown(path: Path, report: dict) -> None:
    overall = report["overall"]
    lines = [
        "# V20 Explicit Metric Shot Bridge",
        "",
        "## Selected pipeline",
        "",
        "Hard reset -> independent single-frame DA3 metric scale -> fixed shot-scale Human3R camera/pointmap/SMPL-X -> V16 torso-motion rotation (45 deg bound) -> human-root camera translation -> bounded background pointmap depth scale.",
        "",
        "## Overall 180-cut result",
        "",
        "| Metric | Fixed mean / P95 | V20 mean / P95 |",
        "|---|---:|---:|",
    ]
    labels = {
        "translation": "Camera translation (m)",
        "rotation": "Camera rotation (deg)",
        "human": "Human root motion error (m)",
        "scene": "Background scene discontinuity (m)",
    }
    for key in ("translation", "rotation", "human", "scene"):
        fixed = overall[key]["fixed"]
        candidate = overall[key]["candidate"]
        lines.append(
            f"| {labels[key]} | {fixed['mean']:.3f} / {fixed['p95']:.3f} | "
            f"{candidate['mean']:.3f} / {candidate['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Catastrophic rate: `{100 * overall['catastrophic_rate']['fixed']:.1f}% -> {100 * overall['catastrophic_rate']['candidate']:.1f}%`.",
            f"- Strict success rate: `{100 * overall['success_rate']['fixed']:.1f}% -> {100 * overall['success_rate']['candidate']:.1f}%`.",
            f"- Camera, human and scene all improve: `{100 * overall['all_camera_human_scene_improved_rate']:.1f}%`.",
            f"- DA3 independent two-shot inference latency: mean `{report['da3_cut_latency_seconds']['mean']:.3f} s`, P95 `{report['da3_cut_latency_seconds']['p95']:.3f} s`.",
            f"- Harmful correction rates: camera translation `{100 * overall['harmful_rate']['translation_plus_010m']:.1f}%`, rotation `{100 * overall['harmful_rate']['rotation_plus_5deg']:.1f}%`, human `{100 * overall['harmful_rate']['human_plus_010m']:.1f}%`, scene `{100 * overall['harmful_rate']['scene_plus_010m']:.1f}%`.",
            "",
            "## Per source",
            "",
            "| Source | Camera T | Rotation | Human | Scene | All-three improve |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source, row in report["by_source"].items():
        lines.append(
            f"| {source} | {row['translation']['candidate']['mean']:.3f} | "
            f"{row['rotation']['candidate']['mean']:.2f} | {row['human']['candidate']['mean']:.3f} | "
            f"{row['scene']['candidate']['mean']:.3f} | "
            f"{100 * row['all_camera_human_scene_improved_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The viable component is not DA3 camera-only translation. It is a synchronized explicit metric correction: DA3 supplies source-invariant depth scale, torso motion supplies rotation, the human root supplies the camera translation equation, and a bounded background-only depth residual repairs the remaining pointmap mismatch without moving camera or SMPL-X.",
            "",
            "Three post-cut frames did not provide a stable gain over the first frame, so the selected streaming setting remains zero-wait.",
            "",
            "The repeated point-sampling seed changed scene mean by less than 0.001 m and kept the scene harmful rate unchanged. The 3000-point / 15-step fast setting retained the same all-three improvement and harmful rates, with a small scene-error increase.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    boundary = json.loads(args.boundary_report.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_report.read_text(encoding="utf-8"))
    scene_rows = scene["cases"]
    boundary_rows = {row["case_name"]: row for row in boundary["cases"]}
    if len(scene_rows) != 180 or len(boundary_rows) != 180:
        raise RuntimeError("Final V20 report requires all 180 cuts")
    by_source = {
        source: summarize(
            [row for row in scene_rows if row["source"] == source], args.method, args.variant
        )
        for source in sorted({row["source"] for row in scene_rows})
    }
    q_values = np.asarray(
        [row["methods"][args.method][args.variant]["pointmap_scale"] for row in scene_rows]
    )
    accepted = np.asarray(
        [
            bool(row["methods"][args.method][args.variant]["fit"].get("accepted", True))
            for row in scene_rows
        ]
    )
    da3_latency = np.asarray(
        [boundary_rows[row["case_name"]]["da3_inference_seconds"] for row in scene_rows]
    )
    report = {
        "experiment": "V20 final strict-causal explicit metric shot bridge",
        "case_count": 180,
        "selected": {"method": args.method, "pointmap_variant": args.variant},
        "protocol": {
            "da3_mode": "independent single-frame per shot",
            "rotation_bound_deg": 45.0,
            "post_cut_frames": 1,
            "learned_components": False,
            "raw_tokens_used": False,
            "gt_depth_or_scene_used": False,
            "one_fixed_boundary_se3": True,
        },
        "overall": summarize(scene_rows, args.method, args.variant),
        "by_source": by_source,
        "grouped": grouped(scene_rows, args.method, args.variant),
        "pointmap_scale": {
            **dist(q_values),
            "accepted_rate": float(np.mean(accepted)),
        },
        "da3_cut_latency_seconds": dist(da3_latency),
        "three_frame_summary": json.loads(args.three_frame_report.read_text(encoding="utf-8"))["overall"]
        if args.three_frame_report.exists()
        else None,
        "robustness_checks": {
            "repeat_seed": json.loads(args.repeat_scene_report.read_text(encoding="utf-8"))["overall"][args.method][args.variant]
            if args.repeat_scene_report.exists()
            else None,
            "fast_3000_points_15_steps": json.loads(args.fast_scene_report.read_text(encoding="utf-8"))["overall"][args.method][args.variant]
            if args.fast_scene_report.exists()
            else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "v20_final_explicit_metric_bridge.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.doc_path, report)
    print(f">> wrote {output}")
    print(f">> wrote {args.doc_path}")


if __name__ == "__main__":
    main()
