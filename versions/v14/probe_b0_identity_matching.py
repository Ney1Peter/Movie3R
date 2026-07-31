#!/usr/bin/env python3
"""Measure whether learned V14/V9-parity B0 improves cross-cut identity matching."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    boundary_error,
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_CACHE = REPO_ROOT / "output/v20_phase1_gt_id_multihuman_consensus/case_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/b0_identity_matching"
MATCHERS = ("root", "torso", "root_torso", "root_torso_joints")
BOUNDARIES = ("direct", "camera_continuity", "learned_b0", "gt_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument(
        "--sequence", choices=tuple(geometry.SEQUENCE_IDENTITIES), default="three"
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--offsets", type=int, nargs="+", default=(0,))
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--evaluation_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_scale(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 1.0
    return max(float(np.median(finite)), 1e-6)


def angular_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return geometry.rotation_distance_deg(first, second)


def identity_cost_components(
    pre_humans: dict[str, dict],
    post_detections: list[tuple[str, dict]],
    boundary: np.ndarray,
    identities: tuple[str, ...],
) -> dict[str, np.ndarray]:
    row_count = len(identities)
    column_count = len(post_detections)
    root = np.zeros((row_count, column_count), dtype=np.float64)
    torso = np.zeros_like(root)
    joints = np.zeros_like(root)
    rotation = np.asarray(boundary, dtype=np.float64)[:3, :3]
    for row, identity in enumerate(identities):
        reference = pre_humans[identity]
        reference_joints = np.asarray(reference["joints"], dtype=np.float64)
        reference_centered = reference_joints - np.asarray(reference["root"])[None]
        for column, (_, detection) in enumerate(post_detections):
            mapped_root = geometry.transform_points(
                boundary, np.asarray(detection["root"])[None]
            )[0]
            root[row, column] = np.linalg.norm(mapped_root - reference["root"])
            torso[row, column] = angular_distance_deg(
                np.asarray(reference["torso"]), rotation @ np.asarray(detection["torso"])
            )
            post_joints = np.asarray(detection["joints"], dtype=np.float64)
            post_centered = (post_joints - np.asarray(detection["root"])[None]) @ rotation.T
            joint_count = min(len(reference_centered), len(post_centered))
            joints[row, column] = float(
                np.linalg.norm(
                    reference_centered[:joint_count] - post_centered[:joint_count], axis=1
                ).mean()
            )
    return {"root": root, "torso": torso, "joints": joints}


def matching_costs(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized = {
        name: value / safe_scale(value) for name, value in components.items()
    }
    return {
        "root": normalized["root"],
        "torso": normalized["torso"],
        "root_torso": normalized["root"] + normalized["torso"],
        "root_torso_joints": (
            normalized["root"] + normalized["torso"] + normalized["joints"]
        ),
    }


def evaluate_assignment(
    cost: np.ndarray,
    identities: tuple[str, ...],
    post_detections: list[tuple[str, dict]],
) -> dict:
    rows, columns = linear_sum_assignment(cost)
    predicted = {
        identities[int(row)]: post_detections[int(column)][0]
        for row, column in zip(rows, columns)
    }
    correct_count = sum(identity == predicted.get(identity) for identity in identities)
    true_columns = {
        identity: column
        for column, (identity, _) in enumerate(post_detections)
    }
    gt_permutation = tuple(true_columns[identity] for identity in identities)
    scored = sorted(
        (
            float(sum(cost[row, column] for row, column in enumerate(candidate))),
            tuple(int(value) for value in candidate),
        )
        for candidate in permutations(range(len(identities)))
    )
    gt_cost = float(
        sum(cost[row, column] for row, column in enumerate(gt_permutation))
    )
    gt_rank = 1 + sum(value[0] < gt_cost - 1e-10 for value in scored)
    best_cost = scored[0][0]
    second_cost = scored[1][0] if len(scored) > 1 else float("inf")
    return {
        "predicted_identity_by_pre_identity": predicted,
        "correct_count": int(correct_count),
        "person_count": len(identities),
        "assignment_accuracy": float(correct_count / len(identities)),
        "all_correct": bool(correct_count == len(identities)),
        "gt_permutation": gt_permutation,
        "gt_permutation_rank": int(gt_rank),
        "gt_assignment_cost": gt_cost,
        "best_assignment_cost": best_cost,
        "best_vs_second_margin": float(second_cost - best_cost),
        "gt_vs_best_gap": float(gt_cost - best_cost),
    }


def evaluate_matching(
    cache: dict,
    boundary: np.ndarray,
) -> dict:
    pre_set = set(cache["humans"][-2])
    post_set = set(cache["humans"][-1])
    if pre_set != post_set:
        raise ValueError(
            "Controlled identity probe requires the same visible GT identity set "
            f"on both sides of the cut, found pre={pre_set}, post={post_set}"
        )
    identities = tuple(identity for identity in geometry.IDENTITIES if identity in pre_set)
    if len(identities) < 2:
        raise ValueError(f"Need at least two shared identities, found {identities}")
    pre = cache["humans"][-2]
    post = sorted(
        ((identity, dict(human)) for identity, human in cache["humans"][-1].items()),
        key=lambda item: int(item[1]["detection_index"]),
    )
    components = identity_cost_components(pre, post, boundary, identities)
    costs = matching_costs(components)
    return {
        "identities": identities,
        "post_detection_gt_order": tuple(identity for identity, _ in post),
        "components": components,
        "matchers": {
            name: {**evaluate_assignment(value, identities, post), "cost": value}
            for name, value in costs.items()
        },
    }


def camera_span_deg(cache: dict) -> float:
    return angular_distance_deg(
        np.asarray(cache["gt"]["pre_c2w"])[:3, :3],
        np.asarray(cache["gt"]["post_c2w"])[:3, :3],
    )


def span_bucket(angle: float) -> str:
    if angle < 60.0:
        return "lt60"
    if angle < 120.0:
        return "60to120"
    return "ge120"


def strict_cache(args: argparse.Namespace, path: Path) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return geometry.reassign_cache_gt_identities(
        SimpleNamespace(
            data_root=args.data_root,
            size=int(args.size),
            sequence=str(args.sequence),
        ),
        cache,
    )


def prepare_case_paths(args: argparse.Namespace, cache: dict) -> tuple[list[Path], Path]:
    case = cache["case"]
    pre_frames = [int(value) for value in case["pre_frames"]][
        -int(args.history_frames) :
    ]
    source = int(case["source_camera"])
    target = int(case["target_camera"])
    pre = [geometry.extract_video_frame(args, source, frame) for frame in pre_frames]
    post = geometry.extract_video_frame(args, target, int(case["post_frame"]))
    return pre, post


def infer_learned_b0(model, args: argparse.Namespace, cache: dict) -> tuple[np.ndarray, dict]:
    from dust3r.v14_outputs import boundary_from_camera_predictions

    pre, post = prepare_case_paths(args, cache)
    shadow_views = set_event_indices(
        geometry.prepare_full_square_input(model, pre + [post], args), {len(pre)}
    )
    raw_views = set_event_indices(
        geometry.prepare_full_square_input(model, [post], args), set()
    )
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow_predictions, _ = model.forward_recurrent_lighter(
            shadow_views, str(args.device), ret_state=False, use_ttt3r=False
        )
        raw_predictions, _ = model.forward_recurrent_lighter(
            raw_views, str(args.device), ret_state=False, use_ttt3r=False
        )
    boundary = boundary_from_camera_predictions(
        shadow_predictions[-1], raw_predictions[0]
    )[0].detach().cpu().numpy().astype(np.float64)
    raw_camera = camera_matrix(raw_predictions[0]).astype(np.float64)
    cache_raw = np.asarray(cache["poses"][-1], dtype=np.float64)
    diagnostics = {
        "runtime_seconds": time.perf_counter() - started,
        "raw_current_vs_phase2_cache": boundary_error(raw_camera, cache_raw),
    }
    del shadow_predictions, raw_predictions, shadow_views, raw_views
    return boundary, diagnostics


def case_boundaries(cache: dict, learned_b0: np.ndarray) -> dict[str, np.ndarray]:
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_camera = np.asarray(cache["poses"][-1], dtype=np.float64)
    camera_continuity = pre_camera @ np.linalg.inv(post_camera)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_post = gauge @ gt_post
    gt_camera = target_post @ np.linalg.inv(post_camera)
    return {
        "direct": np.eye(4, dtype=np.float64),
        "camera_continuity": camera_continuity,
        "learned_b0": learned_b0,
        "gt_camera": gt_camera,
    }


def case_paths(args: argparse.Namespace) -> list[Path]:
    if args.case:
        paths = [args.cache_dir / f"{name}.pt" for name in args.case]
    else:
        paths = sorted(
            path
            for offset in sorted(set(int(value) for value in args.offsets))
            for path in args.cache_dir.glob(f"{args.sequence}_*_k{offset}.pt")
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths[: int(args.max_cases)] if int(args.max_cases) > 0 else paths


def controlled_case_paths(
    args: argparse.Namespace, paths: list[Path]
) -> tuple[list[Path], list[str]]:
    eligible = []
    excluded = []
    for path in paths:
        cache = strict_cache(args, path)
        pre_set = set(cache["humans"][-2])
        post_set = set(cache["humans"][-1])
        if pre_set == post_set and len(pre_set) >= 2:
            eligible.append(path)
        else:
            excluded.append(path.stem)
    return eligible, excluded


def process_case(model, args: argparse.Namespace, path: Path) -> dict:
    cache = strict_cache(args, path)
    learned_b0, inference = infer_learned_b0(model, args, cache)
    boundaries = case_boundaries(cache, learned_b0)
    rows = {
        name: evaluate_matching(cache, boundary)
        for name, boundary in boundaries.items()
    }
    gt_camera = boundaries["gt_camera"]
    output = {
        "case": cache["case"],
        "camera_span_deg": camera_span_deg(cache),
        "span_bucket": span_bucket(camera_span_deg(cache)),
        "inference": inference,
        "learned_b0_camera_error": boundary_error(learned_b0, gt_camera),
        "boundaries": boundaries,
        "matching": rows,
    }
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def summarize_rows(cases: list[dict], boundary: str, matcher: str) -> dict:
    rows = [case["matching"][boundary]["matchers"][matcher] for case in cases]
    return {
        "case_count": len(rows),
        "assignment_accuracy": float(
            sum(row["correct_count"] for row in rows)
            / max(sum(row["person_count"] for row in rows), 1)
        ),
        "all_correct_rate": float(np.mean([row["all_correct"] for row in rows])),
        "gt_top1_rate": float(
            np.mean([row["gt_permutation_rank"] == 1 for row in rows])
        ),
        "gt_rank_mean": float(np.mean([row["gt_permutation_rank"] for row in rows])),
        "best_vs_second_margin_median": float(
            np.median([row["best_vs_second_margin"] for row in rows])
        ),
        "gt_vs_best_gap_median": float(
            np.median([row["gt_vs_best_gap"] for row in rows])
        ),
    }


def aggregate(cases: list[dict]) -> dict:
    groups = {"all": cases}
    for offset in sorted({int(case["case"]["offset"]) for case in cases}):
        groups[f"k{offset}"] = [
            case for case in cases if int(case["case"]["offset"]) == offset
        ]
    for bucket in ("lt60", "60to120", "ge120"):
        groups[bucket] = [case for case in cases if case["span_bucket"] == bucket]
    output = {}
    for group, values in groups.items():
        if not values:
            continue
        output[group] = {
            boundary: {
                matcher: summarize_rows(values, boundary, matcher)
                for matcher in MATCHERS
            }
            for boundary in BOUNDARIES
        }
    return output


def percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_report(report: dict) -> str:
    protocol = report["protocol"]
    lines = [
        "# V14 Learned B0 Identity Matching Probe",
        "",
        f"Cases: `{len(report['cases'])}/{protocol['candidate_case_count']}` MultiHuman "
        f"`{protocol['sequence']}` cuts with the same >=2 visible identities on both sides.",
        "GT identity is used only to score anonymous Hungarian assignments.",
        "Cuts with entry/exit or detector-set changes are excluded to avoid adding a "
        "dustbin policy as a second experimental variable.",
        "",
    ]
    for group, values in report["summary"].items():
        lines.extend(
            [
                f"## {group}",
                "",
                "| Matcher | Direct accuracy | B0 accuracy | Direct all-correct | B0 all-correct | GT-camera all-correct |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for matcher in MATCHERS:
            direct = values["direct"][matcher]
            learned = values["learned_b0"][matcher]
            oracle = values["gt_camera"][matcher]
            lines.append(
                f"| {matcher} | {percentage(direct['assignment_accuracy'])} | "
                f"{percentage(learned['assignment_accuracy'])} | "
                f"{percentage(direct['all_correct_rate'])} | "
                f"{percentage(learned['all_correct_rate'])} | "
                f"{percentage(oracle['all_correct_rate'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "The matcher and all normalizations are identical before and after B0. "
            "Only the shared post-cut coordinate transform changes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[args.sequence]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.output_dir / "cases"
    if args.overwrite and case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    candidate_paths = case_paths(args)
    paths, excluded_cases = controlled_case_paths(args, candidate_paths)
    print(
        f">> controlled matching cases: {len(paths)}/{len(candidate_paths)} "
        f"(excluded variable-visibility cuts: {len(excluded_cases)})",
        flush=True,
    )

    model = None
    if not args.evaluation_only:
        if not args.model_path.is_file():
            raise FileNotFoundError(args.model_path)
        from dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
        flags = configure_model(model)
    else:
        flags = None

    cases = []
    for index, path in enumerate(paths, start=1):
        destination = case_dir / f"{path.stem}.json"
        if destination.is_file() and not args.overwrite:
            row = json.loads(destination.read_text(encoding="utf-8"))
        elif model is None:
            raise FileNotFoundError(f"Missing cached result: {destination}")
        else:
            row = process_case(model, args, path)
            destination.write_text(
                json.dumps(
                    geometry.jsonable(row),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=True,
                )
                + "\n",
                encoding="utf-8",
            )
        cases.append(row)
        learned = row["matching"]["learned_b0"]["matchers"]["root_torso"]
        direct = row["matching"]["direct"]["matchers"]["root_torso"]
        print(
            f"[{index}/{len(paths)}] {path.stem}: span={row['camera_span_deg']:.1f} "
            f"direct={direct['correct_count']}/{direct['person_count']} "
            f"B0={learned['correct_count']}/{learned['person_count']}",
            flush=True,
        )

    report = {
        "experiment": "V14/V9-parity learned coarse B0 before identity matching",
        "model_path": str(args.model_path.resolve()),
        "model_flags": flags,
        "protocol": {
            "sequence": args.sequence,
            "offsets": sorted(set(int(value) for value in args.offsets)),
            "history_frames": int(args.history_frames),
            "synchronized_cuts": all(int(value) == 0 for value in args.offsets),
            "gt_identity_in_matcher": False,
            "gt_identity_in_evaluator": True,
            "same_matcher_before_after_b0": True,
            "candidate_case_count": len(candidate_paths),
            "eligible_case_count": len(paths),
            "requires_same_visible_identity_set": True,
            "excluded_variable_visibility_cases": excluded_cases,
        },
        "summary": aggregate(cases),
        "cases": cases,
    }
    report_path = args.output_dir / "v14_b0_identity_matching.json"
    report_path.write_text(
        json.dumps(
            geometry.jsonable(report),
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = markdown_report(report)
    markdown_path = args.output_dir / "v14_b0_identity_matching.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(f">> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
