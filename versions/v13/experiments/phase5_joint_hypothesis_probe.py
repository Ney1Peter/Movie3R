#!/usr/bin/env python3
"""V13 Phase 5 Stage 2/3 bounded WHO-WHERE hypothesis probe.

The probe reads the frozen Stage-0 identity state trajectory.  It evaluates
each Top-K identity hypothesis with exactly one frozen Phase-2 Boundary solve,
but it does not yet let an unvalidated joint scorer update persistent state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v13.causal_who_where import (  # noqa: E402
    camera_metrics,
    fallback_solution,
    hypothesis_geometry,
    hypothesis_state_cache,
    transform_shot,
    update_geometry_history,
)
from versions.v13.experiments import phase3_cross_shot_identity as phase3  # noqa: E402
from versions.v13.experiments import phase5_causal_identity_state as phase5  # noqa: E402
from versions.v13.native_token_probe import jsonable  # noqa: E402
from versions.v13.shot_persistent_identity import (  # noqa: E402
    ShotPersistentIdentityState,
    assignment_result,
    enumerate_topk_hypotheses,
)


JOINT_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class FrozenIdentityRule:
    prototype: str
    track_normalized: bool
    dispersion_floor: float
    acceptance_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("three", "dance", "box"), default="three")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument(
        "--input_root", type=Path, default=ROOT / "output/v13/phase5_identity"
    )
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "output/v13/phase5_joint"
    )
    parser.add_argument("--state_ttl", type=int, default=60)
    parser.add_argument("--history_size", type=int, default=5)
    parser.add_argument("--max_tracks", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--max_streams", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> argparse.Namespace:
    args.sequence = str(args.sequence)
    args.input_dir = args.input_root / args.sequence
    args.output_dir = args.output_root / args.sequence
    args.shot_cache_dir = args.input_dir / "shot_cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[args.sequence]
    return args


def frozen_rule(args: argparse.Namespace) -> FrozenIdentityRule:
    report = json.loads(
        (args.input_dir / "v13_phase5_stage0_state.json").read_text(
            encoding="utf-8"
        )
    )
    if report["decision"] != "continue_to_joint_who_where":
        raise RuntimeError("Stage 0/1 did not authorize joint WHO-WHERE")
    method = report["selected_method"]
    return FrozenIdentityRule(
        prototype=str(method["prototype"]),
        track_normalized=bool(method["track_normalized"]),
        dispersion_floor=float(method["dispersion_floor"]),
        acceptance_margin=float(report["selected_acceptance_margin"]),
    )


def load_stream(args: argparse.Namespace, spec: phase5.StreamSpec) -> list[dict]:
    return [
        torch.load(
            args.shot_cache_dir
            / f"{phase5.shot_key(args.sequence, camera, start, spec.shot_length)}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for camera, start in zip(spec.cameras, spec.shot_starts)
    ]


def identity_is_complete(metrics: dict) -> bool:
    return bool(
        metrics["matchable"] > 0
        and metrics["false_positive"] == 0
        and metrics["true_positive"] == metrics["matchable"]
    )


def normalize01(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min()) if len(values) else 0.0
    return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)


def process_shot_state(
    state: ShotPersistentIdentityState,
    shot: dict,
    first_external: np.ndarray,
    native_to_external: dict[int, int],
    votes: dict[int, list[int]],
    global_time: int,
) -> tuple[dict, dict[int, int], int]:
    rows = []
    for frame_index, source in enumerate(shot["frames"]):
        row = dict(source)
        feature = row["feature"]
        if frame_index == 0:
            external = np.asarray(first_external, dtype=np.int64)
        else:
            external = phase5.no_cut_external_ids(
                state, row, native_to_external
            )
            valid = np.asarray(
                feature.get(
                    "appearance_valid",
                    np.ones(int(feature["count"]), dtype=bool),
                ),
                dtype=bool,
            )
            state.observe(feature, external, global_time, valid=valid)
        native_to_external = {
            int(native): int(track)
            for native, track in zip(phase5.native_ids(row), external)
            if int(native) >= 0
        }
        phase5.update_votes(votes, external, row["labels_evaluator_only"])
        row["external_ids"] = external
        rows.append(row)
        global_time += 1
    output = dict(shot)
    output["frames"] = rows
    return output, native_to_external, global_time


def geometry_row(
    args: argparse.Namespace,
    previous_shot: dict,
    geometry_history: dict[int, list[dict]],
    post_frame: dict,
    hypothesis: dict,
    identity_metrics: dict,
) -> dict:
    cache = hypothesis_state_cache(
        geometry_history,
        previous_shot,
        post_frame,
        hypothesis["result"],
        tuple(geometry.IDENTITIES),
    )
    diagnostic = hypothesis_geometry(cache)
    pre_frame = int(previous_shot["frames"][-1]["dataset_frame"])
    post_dataset_frame = int(post_frame["dataset_frame"])
    gt_pre = np.linalg.inv(
        geometry.gt_w2c(args, int(previous_shot["camera"]), pre_frame)
    )
    gt_post = np.linalg.inv(
        geometry.gt_w2c(args, int(post_frame["camera"]), post_dataset_frame)
    )
    camera = camera_metrics(cache, diagnostic["solution"], gt_pre, gt_post)
    return {
        "signature": hypothesis["signature"],
        "matched_count": int(hypothesis["matched_count"]),
        "identity_cost": float(hypothesis["identity_cost"]),
        "identity": identity_metrics,
        "identity_complete": identity_is_complete(identity_metrics),
        "geometry": diagnostic,
        "camera": camera,
    }


def probe_stream(
    args: argparse.Namespace,
    spec: phase5.StreamSpec,
    shots: list[dict],
    rule: FrozenIdentityRule,
) -> dict:
    state = ShotPersistentIdentityState(
        ttl=int(args.state_ttl),
        history_size=int(args.history_size),
        max_tracks=int(args.max_tracks),
    )
    votes: dict[int, list[int]] = defaultdict(list)
    native_to_external: dict[int, int] = {}
    global_time = 0

    first = shots[0]
    external = state.bootstrap(first["frames"][0]["feature"], global_time)
    first_with_ids, native_to_external, global_time = process_shot_state(
        state,
        first,
        external,
        native_to_external,
        votes,
        global_time,
    )
    previous_aligned = transform_shot(first_with_ids, np.eye(4))
    geometry_history: dict[int, list[dict]] = {}
    update_geometry_history(
        geometry_history, previous_aligned, history_size=int(args.history_size)
    )
    cuts = []

    for shot_index, raw_shot in enumerate(shots[1:], start=1):
        post_frame = raw_shot["frames"][0]
        bank_labels = phase5.majority(votes)
        payload, cost = state.cost_matrix(
            post_frame["feature"],
            phase5.FEATURE,
            rule.prototype,
            global_time,
            rule.track_normalized,
            rule.dispersion_floor,
        )
        hypotheses = enumerate_topk_hypotheses(
            payload, cost, top_k=int(args.top_k)
        )
        metrics = [
            phase5.evaluate_assignment(
                hypothesis["result"],
                bank_labels,
                post_frame["labels_evaluator_only"],
            )
            for hypothesis in hypotheses
        ]
        rows = [
            geometry_row(
                args,
                previous_aligned,
                geometry_history,
                post_frame,
                hypothesis,
                identity,
            )
            for hypothesis, identity in zip(hypotheses, metrics)
        ]
        identity_rank = normalize01([row["identity_cost"] for row in rows])
        geometry_rank = normalize01(
            [row["geometry"]["geometry_score"] for row in rows]
        )
        score_vectors = {
            "identity_top1": np.asarray(
                [row["identity_cost"] for row in rows], dtype=np.float64
            ),
            "geometry_only": geometry_rank,
        }
        for weight in JOINT_WEIGHTS:
            score_vectors[f"joint_w{weight:g}"] = (
                identity_rank + weight * geometry_rank
            )
        selections = {
            name: int(np.argmin(values)) for name, values in score_vectors.items()
        }
        score_margins = {}
        for name, values in score_vectors.items():
            ordered = np.sort(values)
            score_margins[name] = (
                float(ordered[1] - ordered[0]) if len(ordered) > 1 else float("inf")
            )

        margin = (
            float(hypotheses[1]["identity_cost"] - hypotheses[0]["identity_cost"])
            if len(hypotheses) > 1
            else float("inf")
        )
        accepted_result = (
            hypotheses[0]["result"]
            if margin >= rule.acceptance_margin
            else assignment_result(payload, cost, [])
        )
        selected_cache = hypothesis_state_cache(
            geometry_history,
            previous_aligned,
            post_frame,
            accepted_result,
            tuple(geometry.IDENTITIES),
        )
        if accepted_result["accepted_pairs"]:
            selected_solution = hypothesis_geometry(selected_cache)["solution"]
        else:
            selected_solution = fallback_solution(selected_cache)
        boundary = geometry.make_transform(
            selected_solution["rotation"], selected_solution["translation"]
        )
        external = state.commit(
            post_frame["feature"], accepted_result, global_time
        )
        current_with_ids, native_to_external, global_time = process_shot_state(
            state,
            raw_shot,
            external,
            native_to_external,
            votes,
            global_time,
        )
        previous_aligned = transform_shot(current_with_ids, boundary)
        update_geometry_history(
            geometry_history,
            previous_aligned,
            history_size=int(args.history_size),
        )
        cuts.append(
            {
                "stream": spec.name,
                "cut_index": shot_index,
                "source_camera": int(spec.cameras[shot_index - 1]),
                "target_camera": int(spec.cameras[shot_index]),
                "post_frame": int(post_frame["dataset_frame"]),
                "hypothesis_count": len(rows),
                "topk_gt_assignment": any(
                    row["identity_complete"] for row in rows
                ),
                "identity_margin": margin,
                "stage0_identity_accepted": bool(
                    margin >= rule.acceptance_margin
                ),
                "hypotheses": rows,
                "selections": selections,
                "score_margins": score_margins,
            }
        )
    return {"stream": spec.name, "cuts": cuts}


def identity_summary(rows: list[dict]) -> dict:
    return phase3.aggregate_identity(rows)


def camera_summary(rows: list[dict]) -> dict:
    fields = (
        "camera_translation_error_m",
        "camera_rotation_error_deg",
        "camera_composite",
    )
    return {
        "case_count": len(rows),
        **{
            name: geometry.finite_distribution([row[name] for row in rows])
            for name in fields
        },
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows])),
    }


def risk_summary(cuts: list[dict], method: str) -> tuple[dict, list[dict]]:
    margins = np.asarray(
        [float(cut["score_margins"][method]) for cut in cuts], dtype=np.float64
    )
    finite = margins[np.isfinite(margins)]
    thresholds = [0.0, float("inf")]
    if len(finite):
        thresholds.extend(
            np.quantile(finite, np.linspace(0.0, 1.0, min(len(finite), 40)))
        )
    rows = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        active = [
            cut
            for cut in cuts
            if float(cut["score_margins"][method]) >= threshold
        ]
        selected = [
            cut["hypotheses"][cut["selections"][method]] for cut in active
        ]
        accepted = sum(row["identity"]["accepted"] for row in selected)
        wrong = sum(row["identity"]["false_positive"] for row in selected)
        multi = [row for row in selected if row["matched_count"] >= 2]
        rows.append(
            {
                "threshold": threshold,
                "accepted": int(accepted),
                "wrong": int(wrong),
                "accepted_precision": (
                    (accepted - wrong) / accepted if accepted else float("nan")
                ),
                "multi_activation_coverage": len(multi) / max(len(cuts), 1),
                "cut_all_matches_correct": (
                    float(np.mean([row["identity_complete"] for row in multi]))
                    if multi
                    else float("nan")
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["wrong"] == 0,
            row["multi_activation_coverage"],
            row["accepted"],
        ),
        reverse=True,
    )
    return rows[0], rows


def analyze(streams: list[dict], rule: FrozenIdentityRule, args: argparse.Namespace) -> dict:
    cuts = [cut for stream in streams for cut in stream["cuts"]]
    names = list(cuts[0]["selections"])
    methods = {}
    for name in names:
        selected = [cut["hypotheses"][cut["selections"][name]] for cut in cuts]
        methods[name] = {
            "identity": identity_summary([row["identity"] for row in selected]),
            "cut_all_matches_correct": float(
                np.mean([row["identity_complete"] for row in selected])
            ),
            "camera": camera_summary([row["camera"] for row in selected]),
        }
        methods[name]["risk_selected"], methods[name]["risk_frontier"] = (
            risk_summary(cuts, name)
        )
    oracle_rows = []
    for cut in cuts:
        correct = [row for row in cut["hypotheses"] if row["identity_complete"]]
        oracle_rows.append(
            min(correct, key=lambda row: row["identity_cost"])
            if correct
            else cut["hypotheses"][0]
        )
    methods["oracle_identity_in_topk"] = {
        "identity": identity_summary([row["identity"] for row in oracle_rows]),
        "cut_all_matches_correct": float(
            np.mean([row["identity_complete"] for row in oracle_rows])
        ),
        "camera": camera_summary([row["camera"] for row in oracle_rows]),
    }
    automatic_names = [name for name in names if name.startswith("joint_")]
    selected_joint = max(
        automatic_names,
        key=lambda name: (
            methods[name]["risk_selected"]["wrong"] == 0,
            methods[name]["risk_selected"]["multi_activation_coverage"],
            methods[name]["risk_selected"]["accepted"],
            methods[name]["cut_all_matches_correct"],
        ),
    )
    report = {
        "experiment": "V13 Phase 5 bounded WHO-WHERE hypothesis probe",
        "sequence": args.sequence,
        "role": "development" if args.sequence == "three" else "frozen_evaluation",
        "stream_count": len(streams),
        "cut_count": len(cuts),
        "frozen_identity_rule": rule.__dict__,
        "top_k": int(args.top_k),
        "topk_gt_assignment_recall": float(
            np.mean([cut["topk_gt_assignment"] for cut in cuts])
        ),
        "methods": methods,
        "selected_joint_development_only": selected_joint,
        "protocol": {
            "one_frozen_boundary_per_hypothesis": True,
            "geometry": "Phase 2 Fixed Explicit + V16 20deg + mean_raw_t",
            "joint_scorer_updates_state": False,
            "state_trajectory": "frozen Stage-0 precision-first identity trajectory",
            "gt_used_for_candidate_or_score": False,
            "gt_used_for_evaluation_and_development_weight_selection": True,
            "future_frames": False,
        },
        "streams": streams,
    }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# V13 Phase 5: Bounded WHO-WHERE Hypothesis Probe",
        "",
        f"- streams / cuts: `{report['stream_count']} / {report['cut_count']}`",
        f"- Top-{report['top_k']} GT assignment recall: "
        f"`{report['topk_gt_assignment_recall']:.4f}`",
        f"- selected development scorer: "
        f"`{report['selected_joint_development_only']}`",
        "",
        "| Method | All correct | IDF1 | Wrong | Safe multi | Composite | P90 | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["methods"].items():
        camera = row["camera"]
        lines.append(
            f"| {name} | {row['cut_all_matches_correct']:.3f} | "
            f"{row['identity']['idf1']:.3f} | "
            f"{row['identity']['false_positive']} | "
            f"{row.get('risk_selected', {}).get('multi_activation_coverage', float('nan')):.3f} | "
            f"{camera['camera_composite']['mean']:.3f} | "
            f"{camera['camera_composite']['p90']:.3f} | "
            f"{camera['catastrophic_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The joint scorer is probe-only and did not update persistent state.",
            "GT identity/camera were attached only for evaluation and development scorer selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = configure(parse_args())
    rule = frozen_rule(args)
    specs = phase5.stream_specs(args.sequence)
    if int(args.max_streams) > 0:
        specs = specs[: int(args.max_streams)]
    streams = []
    for index, spec in enumerate(specs):
        print(f">> WHO-WHERE stream {index + 1}/{len(specs)} {spec.name}", flush=True)
        streams.append(probe_stream(args, spec, load_stream(args, spec), rule))
    report = analyze(streams, rule, args)
    output = args.output_dir / "v13_phase5_joint_probe.json"
    output.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "v13_phase5_joint_probe.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(f">> joint probe report: {output}")


if __name__ == "__main__":
    main()
