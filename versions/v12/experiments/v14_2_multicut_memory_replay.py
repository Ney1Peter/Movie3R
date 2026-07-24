#!/usr/bin/env python3
"""V14.2 causal 1/2/4/8-cut replay over repeated identities.

The V18 cache contains independent cross-camera boundaries rather than one
composable camera trajectory.  This script therefore replays canonical-body
memory causally for each repeated identity and reports per-boundary alignment,
memory drift, and the sum of boundary errors.  True recurrent-state multi-cut
and no-cut invariance remain covered by the V14.1 rollout report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from versions.v12.experiments.v14_2_canonical_human_memory_probe import (  # noqa: E402
    BodyReference,
    body_pair_for_variant,
    build_layer,
    choose_wrong_cases,
    ema,
    evaluate_boundary,
    geometry_continuity,
    load_cases,
    make_reference,
    physical_scale,
    scene_transform,
    solve_candidate,
)


STRATEGIES = (
    "no_memory",
    "every_frame_update",
    "running_median",
    "v14_1_alpha025",
    "first_high_quality_freeze",
    "best_quality_replacement",
    "top3_consensus",
    "wrong_video_memory",
)
PREFIXES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output/v18_human_metric_translation/stream_cache",
    )
    parser.add_argument(
        "--keypoint_dir",
        type=Path,
        default=REPO_ROOT / "output/v18_human_metric_translation/keypoint_cache",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=REPO_ROOT / "output/v18_human_metric_translation/v16_bound20_scene",
    )
    parser.add_argument(
        "--single_cut_report",
        type=Path,
        default=REPO_ROOT
        / "output/v14_2_canonical_human_memory/single_cut/v14_2_canonical_human_memory_probe.json",
    )
    parser.add_argument(
        "--v14_1_multicut_report",
        type=Path,
        default=REPO_ROOT
        / "output/v14_1_shot_aware_state_routing/multicut_true_reset/v14_1_multicut_rollout.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output/v14_2_canonical_human_memory/multicut_replay",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--memory_update_alpha", type=float, default=0.20)
    parser.add_argument("--continuity_alpha", type=float, default=0.25)
    return parser.parse_args()


def load_data(case: dict) -> dict:
    with np.load(case["cache_path"]) as stream, np.load(case["keypoint_path"]) as keypoint:
        data = {name: np.asarray(stream[name]) for name in stream.files}
        data.update({name: np.asarray(keypoint[name]) for name in keypoint.files})
    data["old_bodies"] = data["old_joints_camera"] - data["old_joints_camera"][:, :1]
    data["new_body"] = data["new_joints_camera"] - data["new_joints_camera"][:1]
    data["old_raw_roots"] = data["old_joints_camera"][:, 0]
    data["new_raw_root"] = data["new_joints_camera"][0]
    data["new_gt_camera_root"] = data["new_gt_joints_camera"][0]
    data["old_gt_world_root"] = data["old_gt_joints_target_world"][-1, 0]
    data["new_gt_world_root"] = data["new_gt_joints_target_world"][0]
    return data


def post_quality(data: dict, reference: BodyReference, reprojection_px: float, threshold: float) -> float:
    confidence = data["new_confidence"][[0, 1, 2, 12, 16, 17]]
    visibility = float(np.mean(confidence >= threshold))
    torso_confidence = float(np.mean(confidence))
    box_height = max(float(data["new_box"][3] - data["new_box"][1]), 0.0)
    body_fraction = min(box_height / max(float(data["new_image"].shape[0]), 1.0), 1.0)
    reprojection_quality = 1.0 / (1.0 + (float(reprojection_px) / 10.0) ** 2)
    shape_distance = float(np.linalg.norm(data["new_shape"] - reference.beta[:10]))
    shape_quality = float(np.exp(-shape_distance / 1.0))
    return float(
        0.20 * visibility
        + 0.20 * torso_confidence
        + 0.15 * np.clip(float(data["new_score"]), 0.0, 1.0)
        + 0.15 * body_fraction
        + 0.20 * reprojection_quality
        + 0.10 * shape_quality
    )


def initialize_state(
    case: dict,
    data: dict,
    wrong_case: dict,
) -> dict:
    scales = np.asarray([physical_scale(body) for body in data["old_bodies"]], dtype=np.float32)
    # Single-cut output already contains the frozen-detector quality scores used by V14.2.
    single = case["single_cut"]
    quality = np.asarray(single["memory"]["quality_scores"], dtype=np.float32)
    best = int(np.argmax(quality))
    top3 = np.argsort(quality)[-3:]
    with np.load(wrong_case["cache_path"]) as wrong:
        wrong_shapes = np.asarray(wrong["old_shape"], dtype=np.float32)
        wrong_joints = np.asarray(wrong["old_joints_camera"], dtype=np.float32)
    wrong_bodies = wrong_joints - wrong_joints[:, :1]
    wrong_scales = np.asarray([physical_scale(body) for body in wrong_bodies], dtype=np.float32)
    bank = [
        (np.asarray(beta, dtype=np.float32), float(scale), float(score))
        for beta, scale, score in zip(data["old_shape"], scales, quality)
    ]
    return {
        "initial_beta": data["old_shape"][-1].astype(np.float32),
        "initial_scale": float(scales[-1]),
        "every_frame_update": make_reference(data["old_shape"][-1], scales[-1]),
        "running_median": make_reference(np.median(data["old_shape"], axis=0), np.median(scales)),
        "v14_1_alpha025": make_reference(
            ema(data["old_shape"], 0.20), float(ema(scales[:, None], 0.20)[0])
        ),
        "first_high_quality_freeze": make_reference(data["old_shape"][best], scales[best]),
        "best_quality_replacement": make_reference(data["old_shape"][best], scales[best]),
        "best_quality_score": float(quality[best]),
        "top3_consensus": make_reference(
            np.median(data["old_shape"][top3], axis=0), float(np.median(scales[top3]))
        ),
        "wrong_video_memory": make_reference(
            ema(wrong_shapes, 0.20), float(ema(wrong_scales[:, None], 0.20)[0])
        ),
        "bank": bank,
    }


def reference_for(strategy: str, state: dict) -> BodyReference | None:
    return None if strategy == "no_memory" else state[strategy]


def update_state(state: dict, data: dict, quality: float, alpha: float) -> None:
    beta = data["new_shape"].astype(np.float32)
    scale = physical_scale(data["new_body"])
    state["every_frame_update"] = make_reference(beta, scale)

    state["bank"].append((beta, scale, float(quality)))
    shapes = np.stack([row[0] for row in state["bank"]])
    scales = np.asarray([row[1] for row in state["bank"]], dtype=np.float32)
    state["running_median"] = make_reference(np.median(shapes, axis=0), float(np.median(scales)))

    old = state["v14_1_alpha025"]
    state["v14_1_alpha025"] = make_reference(
        old.beta + float(alpha) * (beta - old.beta),
        old.physical_scale + float(alpha) * (scale - old.physical_scale),
    )
    if quality > state["best_quality_score"]:
        state["best_quality_score"] = float(quality)
        state["best_quality_replacement"] = make_reference(beta, scale)

    top3 = sorted(state["bank"], key=lambda row: row[2])[-3:]
    state["top3_consensus"] = make_reference(
        np.median(np.stack([row[0] for row in top3]), axis=0),
        float(np.median([row[1] for row in top3])),
    )


def evaluate_strategy(
    strategy: str,
    reference: BodyReference | None,
    data: dict,
    camera_rotation: np.ndarray,
    layer10,
    layer11,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    old_scale = physical_scale(data["old_bodies"][-1])
    new_scale = physical_scale(data["new_body"])
    if reference is None:
        old_body, new_body = data["old_bodies"][-1], data["new_body"]
    else:
        variant = "canonical_alpha025" if strategy == "v14_1_alpha025" else "canonical_beta_scale"
        old_body, new_body = body_pair_for_variant(
            variant,
            reference,
            data["old_rotvec"][-1],
            data["new_rotvec"],
            data["old_bodies"][-1],
            data["new_body"],
            data["old_shape"][-1],
            data["new_shape"],
            old_scale,
            new_scale,
            layer10,
            layer11,
            device,
            0.25,
        )
    boundary, diagnostics = solve_candidate(
        old_body, new_body, data, camera_rotation, float(args.keypoint_threshold)
    )
    metrics = evaluate_boundary(boundary, data["new_pose"], data["target_pose"], data["gt_boundary"])
    metrics.update(geometry_continuity(boundary, data))
    if reference is None:
        output_beta = data["new_shape"]
        output_scale = new_scale
    else:
        output_beta = data["new_shape"] + float(args.continuity_alpha) * (
            reference.beta[:10] - data["new_shape"]
        )
        output_scale = new_scale + float(args.continuity_alpha) * (
            reference.physical_scale - new_scale
        )
    metrics.update(
        output_beta=output_beta.astype(float).tolist(),
        output_scale=float(output_scale),
        gt_beta_error_l2=float(np.linalg.norm(output_beta - data["new_gt_shape"][:10])),
        gt_scale_error_abs=float(
            abs(output_scale - physical_scale(data["new_gt_joints_camera"] - data["new_gt_joints_camera"][:1]))
        ),
    )
    return metrics, diagnostics


def replay_group(
    cases: list[dict],
    wrong_case: dict,
    layer10,
    layer11,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    selected = sorted(cases, key=lambda row: int(row["record"]["start_frame"]))[:8]
    first_data = load_data(selected[0])
    state = initialize_state(selected[0], first_data, wrong_case)
    results = {name: [] for name in STRATEGIES}
    for case in selected:
        data = load_data(case)
        camera_rotation = (scene_transform(case) @ data["new_pose"])[:3, :3]
        diagnostics_by_strategy = {}
        for strategy in STRATEGIES:
            row, diagnostics = evaluate_strategy(
                strategy,
                reference_for(strategy, state),
                data,
                camera_rotation,
                layer10,
                layer11,
                device,
                args,
            )
            row["shape_drift_from_first_l2"] = float(
                np.linalg.norm(np.asarray(row["output_beta"]) - state["initial_beta"])
            )
            row["scale_drift_from_first_abs"] = float(abs(row["output_scale"] - state["initial_scale"]))
            results[strategy].append(row)
            diagnostics_by_strategy[strategy] = diagnostics
        quality = post_quality(
            data,
            state["best_quality_replacement"],
            diagnostics_by_strategy["best_quality_replacement"]["new_reprojection"]["reprojection_error_px"],
            float(args.keypoint_threshold),
        )
        update_state(state, data, quality, float(args.memory_update_alpha))
    return {
        "source": selected[0]["source"],
        "group": selected[0]["record"]["group"],
        "cut_count": len(selected),
        "case_names": [row["case_name"] for row in selected],
        "results": results,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def aggregate(groups: list[dict]) -> dict:
    output = {}
    for prefix in PREFIXES:
        eligible = [group for group in groups if group["cut_count"] >= prefix]
        output[str(prefix)] = {}
        for strategy in STRATEGIES:
            sequences = [group["results"][strategy][:prefix] for group in eligible]
            endpoint = [rows[-1] for rows in sequences]
            flat = [row for rows in sequences for row in rows]
            output[str(prefix)][strategy] = {
                "identity_count": len(eligible),
                "translation_m": distribution([row["camera_translation_error_m"] for row in flat]),
                "viewing_direction_m": distribution([row["viewing_direction_error_m"] for row in flat]),
                "root_jump_residual_m": distribution([row["visible_root_jump_residual_m"] for row in flat]),
                "translation_catastrophic_rate": float(
                    np.mean([row["camera_translation_error_m"] > 1.0 for row in flat])
                ),
                "cumulative_translation_error_sum_m": distribution(
                    [sum(row["camera_translation_error_m"] for row in rows) for rows in sequences]
                ),
                "endpoint_shape_drift_l2": distribution(
                    [row["shape_drift_from_first_l2"] for row in endpoint]
                ),
                "endpoint_scale_drift_abs": distribution(
                    [row["scale_drift_from_first_abs"] for row in endpoint]
                ),
                "endpoint_gt_beta_error_l2": distribution([row["gt_beta_error_l2"] for row in endpoint]),
                "endpoint_gt_scale_error_abs": distribution([row["gt_scale_error_abs"] for row in endpoint]),
            }
    return output


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V14.2 Multi-Cut Canonical Memory Replay",
        "",
        "This is a causal repeated-identity replay over independent V18 boundaries, not a composable global camera trajectory.",
        "",
    ]
    for prefix in PREFIXES:
        lines.extend(
            [
                f"## {prefix} Cut(s)",
                "",
                "| Strategy | T mean | View | T-cat | Cumulative T | Shape drift | Scale drift | Root residual |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy, row in report["summary"][str(prefix)].items():
            lines.append(
                "| {} | {:.3f} | {:.3f} | {:.1f}% | {:.3f} | {:.3f} | {:.5f} | {:.3f} |".format(
                    strategy,
                    row["translation_m"]["mean"],
                    row["viewing_direction_m"]["mean"],
                    100.0 * row["translation_catastrophic_rate"],
                    row["cumulative_translation_error_sum_m"]["mean"],
                    row["endpoint_shape_drift_l2"]["mean"],
                    row["endpoint_scale_drift_abs"]["mean"],
                    row["root_jump_residual_m"]["mean"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.2 multi-cut replay requires CUDA for SMPL-X body construction")
    all_cases = load_cases(args)
    single_payload = json.loads(args.single_cut_report.read_text(encoding="utf-8"))
    single = {str(row["case_name"]): row for row in single_payload["cases"]}
    for case in all_cases:
        case["single_cut"] = single[str(case["case_name"])]
    by_identity: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in all_cases:
        by_identity[(str(case["source"]), str(case["record"]["group"]))].append(case)
    eligible = {key: rows for key, rows in by_identity.items() if len(rows) >= 8}
    wrong_map = choose_wrong_cases(all_cases)
    device = torch.device(args.device)
    layer10 = build_layer(device, 10)
    layer11 = build_layer(device, 11)
    groups = []
    for index, (identity, cases) in enumerate(sorted(eligible.items())):
        wrong = wrong_map[str(cases[0]["case_name"])]
        groups.append(replay_group(cases, wrong, layer10, layer11, device, args))
        print(f">> V14.2 replay {index + 1}/{len(eligible)} {identity}", flush=True)

    v14_1 = json.loads(args.v14_1_multicut_report.read_text(encoding="utf-8"))
    report = {
        "experiment": "V14.2 causal repeated-identity multi-cut memory replay",
        "protocol": {
            "trajectory_limitation": (
                "V18 boundaries are independent cached cuts. Per-boundary errors and their sum are valid; "
                "the sum is not a physical globally composed camera trajectory."
            ),
            "memory_is_causal": True,
            "gt_used_for_memory_update": False,
            "current_pose_and_2d_used_for_every_alignment": True,
            "max_humans": 1,
        },
        "identity_count": len(groups),
        "summary": aggregate(groups),
        "v14_1_true_recurrent_multicut": v14_1["summary"],
        "no_cut_check": v14_1["no_cut_check"],
        "groups": groups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "v14_2_multicut_memory_replay.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v14_2_multicut_memory_replay.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
