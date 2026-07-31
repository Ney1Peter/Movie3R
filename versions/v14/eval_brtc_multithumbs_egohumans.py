#!/usr/bin/env python3
"""CPU-only same-forward EgoHumans evaluation of raw, B0, and B0+BRTC-LC.

The existing V13 compact streams are complete, but were produced by the
original ``human3r_896L.pth`` checkpoint.  The cached B0 boundaries were
produced by the current V14/V9-parity checkpoint.  Mixing those artifacts
would not be a same-forward comparison.  This evaluator therefore rebuilds a
small, reusable CPU geometry cache from the current checkpoint, then replays
the already-frozen B0 boundaries and calls the deployable
``versions.v14.b0_person_triangulation.refine_matched_people`` runtime at each
boundary.  No DA3 and no GPU are used.

The three methods share detections, pose/shape, and cameras before alignment:

* ``raw_reset``: every five-frame shot stays in its local reset gauge;
* ``b0``: cached B0 transforms are composed causally over the three shots;
* ``b0_brtc_lc``: B0 plus a boundary person translation propagated over the
  corresponding post shot.  Cameras remain bit-exact B0.

GT labels are evaluator-only.  Boundary association is anonymous frozen-B0
root+torso+joints Hungarian matching and never reads a GT identity.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v13.egobody_probe import (  # noqa: E402
    IDENTITIES,
    assign_identities,
    load_colmap,
)
from versions.v13.gt_id_consensus import (  # noqa: E402
    layer_humans,
    prepare_full_square_input,
)
from versions.v14.b0_person_triangulation import (  # noqa: E402
    DEFAULT_CONFIG as BRTC_CONFIG,
    refine_matched_people,
)
from versions.v14.b0_brtc_huber_irls import (  # noqa: E402
    ReliabilityHuberConfig,
    refine_matched_people as refine_matched_people_huber,
)
from versions.v14.b0_person_triangulation_damped import (  # noqa: E402
    DEFAULT_DAMPED_CONFIG,
    refine_matched_people_damped,
)
from versions.v14.eval_multithumbs_protocol import (  # noqa: E402
    SparseVertexMap,
    acceleration_errors,
    acceleration_second_difference_errors,
    apply_similarity,
    fit_similarity,
    initial_frame_aligned_errors,
    mean_point_error,
    pelvis_center,
    pose_encoding_to_camera_matrix,
    trajectory_aligned_errors,
)
from versions.v14.probe_b0_identity_matching import (  # noqa: E402
    identity_cost_components,
    matching_costs,
)
from versions.v14.run_v14_2_multihuman_sequence import run_rollout  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth"
)
ACTIVE_CHECKPOINT_SHA256 = (
    "8379243216775adbc886d00e6f93b6492f7d8f1bd67adb4e8ad6fbdd84e47123"
)
BRTC_V1_RUNTIME_SHA256 = (
    "98b839f4ae2ff130b0c6ecbc4e0e634ba626d2433f148bee3e55ac169aab3327"
)
BIT_EXACT_BUILD_ALIAS = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble")
DEFAULT_B0 = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_da3_egohumans"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans"
)
DEFAULT_DOC = (
    REPO_ROOT / "versions/v14/docs/V14_BRTC_MULTITHUMBS_EGOHUMANS_20260801.md"
)
HUBER_POLICY_PATH = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/b0_brtc_huber_irls/"
    "FROZEN_HUBER_IRLS_POLICY_BEFORE_CONFIRM.json"
)
SMPLX2SMPL = REPO_ROOT / "src/models/smplx/smplx2smpl.pkl"
SMPL_NEUTRAL = REPO_ROOT / "src/models/smpl/SMPL_NEUTRAL.pkl"
CORE_METHODS = ("raw_reset", "b0", "b0_brtc_lc")
REFINEMENT_VARIANTS = (
    "b0_brtc_completeness_weighted",
    "b0_brtc_damped_0p8",
    "b0_brtc_huber_irls_frozen",
)
METHODS = CORE_METHODS + REFINEMENT_VARIANTS
CHAINS = (
    (
        ("cam01", tuple(range(296, 301))),
        ("cam06", tuple(range(300, 305))),
        ("cam07", tuple(range(304, 309))),
    ),
    (
        ("cam02", tuple(range(176, 181))),
        ("cam05", tuple(range(180, 185))),
        ("cam08", tuple(range(184, 189))),
    ),
    (
        ("cam03", tuple(range(416, 421))),
        ("cam04", tuple(range(420, 425))),
        ("cam01", tuple(range(424, 429))),
    ),
)
PAPER_REFERENCE = {
    "scope": "Multi-THuMBS EgoHumans paper table; reference only",
    "w_mpjpe_mm": 279.0,
    "wa_mpjpe_mm": 166.0,
    "mpjpe_mm": 228.3,
    "mpvpe_mm": 262.2,
    "accel_unit_unspecified": 27.3,
    "ate_alignment_or_unit_not_confirmed": 0.7,
    "ids_aggregation_not_confirmed": 0.97,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--b0_dir", type=Path, default=DEFAULT_B0)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--geometry_cache", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--assignment_threshold_px", type=float, default=24.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--build_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_person(transform: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(person)
    for key in ("root", "joints", "vertices"):
        output[key] = transform_points(transform, person[key])
    for key in ("torso", "root_rotation"):
        if key in person:
            output[key] = np.asarray(transform, dtype=np.float64)[:3, :3] @ np.asarray(
                person[key], dtype=np.float64
            )
    return output


def shift_person(person: dict[str, Any], shift: np.ndarray) -> dict[str, Any]:
    output = copy.deepcopy(person)
    shift = np.asarray(shift, dtype=np.float64)
    for key in ("root", "joints", "vertices"):
        output[key] = np.asarray(person[key], dtype=np.float64) + shift
    return output


def load_smpl_resources() -> tuple[SparseVertexMap, np.ndarray]:
    with SMPLX2SMPL.open("rb") as handle:
        dense = np.asarray(pickle.load(handle, encoding="latin1")["matrix"])
    rows, columns = np.nonzero(dense)
    counts = np.bincount(rows, minlength=dense.shape[0])
    width = int(counts.max())
    indices = np.full((dense.shape[0], width), -1, dtype=np.int64)
    weights = np.zeros((dense.shape[0], width), dtype=np.float32)
    offsets = np.zeros(dense.shape[0], dtype=np.int64)
    for row, column in zip(rows.tolist(), columns.tolist()):
        slot = int(offsets[row])
        indices[row, slot] = column
        weights[row, slot] = float(dense[row, column])
        offsets[row] += 1
    del dense, rows, columns
    with SMPL_NEUTRAL.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float64)[:24]
    return SparseVertexMap(indices=indices, weights=weights), regressor


def image_paths(data_root: Path, camera: str, frames: tuple[int, ...]) -> list[Path]:
    output = [data_root / f"exo/{camera}/images/{frame:05d}.jpg" for frame in frames]
    missing = [path for path in output if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return output


def case_path(b0_dir: Path, chain: int, cut: int) -> Path:
    pre = CHAINS[chain][cut][0]
    post = CHAINS[chain][cut + 1][0]
    return b0_dir / "cases" / f"chain{chain}_cut{cut}_{pre}_{post}.json"


def compact_person(
    human: dict[str, Any], prediction: dict, label: int
) -> dict[str, Any]:
    detection = int(human["detection_index"])
    ids = prediction.get("smpl_id")
    native_id = detection if ids is None else int(ids[0, detection].detach().cpu().item())
    return {
        "detection_index": detection,
        "native_track_id": native_id,
        "gt_label_evaluator_only": int(label),
        "root": np.asarray(human["root"], dtype=np.float32),
        "joints": np.asarray(human["joints"], dtype=np.float32),
        "vertices": np.asarray(human["vertices"], dtype=np.float32),
        "torso": np.asarray(human["torso"], dtype=np.float32),
        "root_rotation": np.asarray(human["root_rotation"], dtype=np.float32),
    }


def build_geometry_cache(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    if str(args.device) != "cpu":
        raise ValueError("This evaluator is frozen to --device cpu")
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    started = time.perf_counter()
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to("cpu")
    flags = configure_model(model)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to("cpu").eval()
    chains = []
    try:
        for chain_index, chain_spec in enumerate(CHAINS):
            segments = []
            for segment_index, (camera, frames) in enumerate(chain_spec):
                views = set_event_indices(
                    prepare_full_square_input(
                        model,
                        image_paths(args.data_root, camera, frames),
                        SimpleNamespace(size=int(args.size)),
                    ),
                    set(),
                )
                predictions, returned, debug, seconds = run_rollout(
                    model,
                    views,
                    "cpu",
                    f"egohumans_chain{chain_index}_segment{segment_index}_cpu",
                )
                labels, assignments = assign_identities(
                    args.data_root,
                    [camera] * len(frames),
                    list(frames),
                    returned,
                    debug,
                    int(args.size),
                    float(args.assignment_threshold_px),
                )
                frame_rows = []
                for frame, prediction, view, token, frame_labels in zip(
                    frames, predictions, returned, debug, labels
                ):
                    humans = layer_humans(prediction, view, token, layer)
                    if len(humans) != len(frame_labels):
                        raise ValueError(
                            f"Human/label count mismatch chain={chain_index} "
                            f"segment={segment_index} frame={frame}"
                        )
                    frame_rows.append(
                        {
                            "camera_name": camera,
                            "dataset_frame": int(frame),
                            "camera_c2w": camera_matrix(prediction).astype(np.float64),
                            "people": [
                                compact_person(human, prediction, int(frame_labels[index]))
                                for index, human in enumerate(humans)
                            ],
                        }
                    )
                segments.append(
                    {
                        "camera_name": camera,
                        "dataset_frames": tuple(int(value) for value in frames),
                        "inference_seconds": float(seconds),
                        "assignment_evaluator_only": assignments,
                        "frames": frame_rows,
                    }
                )
                print(
                    f">> cached chain={chain_index} segment={segment_index} "
                    f"people={sum(len(row['people']) for row in frame_rows)}",
                    flush=True,
                )
                del predictions, returned, debug, views
                gc.collect()
            boundaries = []
            reports = []
            for cut in (0, 1):
                source = case_path(args.b0_dir, chain_index, cut)
                report = json.loads(source.read_text(encoding="utf-8"))
                boundaries.append(np.asarray(report["boundaries"]["b0"], dtype=np.float64))
                reports.append(str(source))
            chains.append(
                {
                    "chain_index": chain_index,
                    "segments": segments,
                    "b0_boundaries": boundaries,
                    "b0_case_reports": reports,
                }
            )
    finally:
        del layer, model
        gc.collect()
    output = {
        "protocol": {
            "checkpoint": str(args.model_path),
            "device": "cpu",
            "same_forward_for_all_methods": True,
            "shot_construction": "three independent five-frame hard-reset shots per chain",
            "gt_use": "2D identity assignment and 3D metric evaluation only",
            "b0_source": "frozen cached V14 boundaries from b0_da3_egohumans",
            "model_flags": flags,
        },
        "chains": chains,
        "wall_seconds": time.perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    print(f">> wrote CPU geometry cache {path}", flush=True)
    return output


def anonymous_assignment(
    pre_by_track: dict[int, dict[str, Any]], post_people: list[dict[str, Any]]
) -> dict[str, Any]:
    tracks = tuple(sorted(pre_by_track))
    detections = [(str(index), person) for index, person in enumerate(post_people)]
    if not tracks or not detections:
        return {"track_to_post_index": {}, "matched_count": 0}
    components = identity_cost_components(
        {str(track): pre_by_track[track] for track in tracks},
        detections,
        np.eye(4, dtype=np.float64),
        tuple(str(track) for track in tracks),
    )
    cost = matching_costs(components)["root_torso_joints"]
    rows, columns = linear_sum_assignment(cost)
    assignment = {
        int(tracks[int(row)]): int(columns[index])
        for index, row in enumerate(rows)
    }
    correct = sum(
        int(pre_by_track[track].get("gt_label_evaluator_only", -1))
        == int(post_people[post].get("gt_label_evaluator_only", -2))
        for track, post in assignment.items()
    )
    return {
        "track_to_post_index": assignment,
        "matched_count": len(assignment),
        "correct_count_evaluator_only": int(correct),
        "accuracy_evaluator_only": float(correct / max(len(assignment), 1)),
        "cost": cost,
        "root_cost_m": components["root"],
        "torso_cost_deg": components["torso"],
        "joint_cost_m": components["joints"],
    }


def assign_segment_tracks(
    frames: list[dict[str, Any]], native_to_global: dict[int, int], next_track: int
) -> int:
    for frame in frames:
        for person in frame["people"]:
            native = int(person["native_track_id"])
            if native not in native_to_global:
                native_to_global[native] = next_track
                next_track += 1
            person["global_track_id"] = int(native_to_global[native])
    return next_track


def method_chains(cache: dict[str, Any]) -> tuple[dict[str, list[dict]], list[dict]]:
    all_methods = {name: [] for name in METHODS}
    boundary_debug = []
    for source_chain in cache["chains"]:
        raw_segments = copy.deepcopy(source_chain["segments"])
        raw_method_segments = []
        for segment_index, segment in enumerate(raw_segments):
            frames = copy.deepcopy(segment["frames"])
            for frame in frames:
                frame["method_camera_c2w"] = np.asarray(frame["camera_c2w"], dtype=np.float64)
                for person in frame["people"]:
                    person["global_track_id"] = (
                        segment_index * 1000 + int(person["native_track_id"])
                    )
            raw_method_segments.append(frames)

        b0_segments, associations = [], []
        cumulative = np.eye(4, dtype=np.float64)
        next_track = 0
        previous_last_by_track: dict[int, dict[str, Any]] = {}
        for segment_index, source_segment in enumerate(source_chain["segments"]):
            if segment_index:
                cumulative = cumulative @ np.asarray(
                    source_chain["b0_boundaries"][segment_index - 1], dtype=np.float64
                )
            frames = []
            for source_frame in source_segment["frames"]:
                frame = copy.deepcopy(source_frame)
                frame["method_camera_c2w"] = cumulative @ np.asarray(
                    source_frame["camera_c2w"], dtype=np.float64
                )
                frame["people"] = [
                    transform_person(cumulative, person) for person in source_frame["people"]
                ]
                frames.append(frame)
            if segment_index == 0:
                native_to_global: dict[int, int] = {}
                next_track = assign_segment_tracks(frames, native_to_global, next_track)
                association = None
            else:
                association = anonymous_assignment(previous_last_by_track, frames[0]["people"])
                native_to_global = {}
                for track, post_index in association["track_to_post_index"].items():
                    native = int(frames[0]["people"][post_index]["native_track_id"])
                    native_to_global[native] = int(track)
                next_track = assign_segment_tracks(frames, native_to_global, next_track)
            previous_last_by_track = {
                int(person["global_track_id"]): person for person in frames[-1]["people"]
            }
            b0_segments.append(frames)
            associations.append(association)

        brtc_segments = [copy.deepcopy(b0_segments[0])]
        for segment_index in (1, 2):
            post_frames = copy.deepcopy(b0_segments[segment_index])
            pre_frame = brtc_segments[-1][-1]
            pre_by_track = {
                int(person["global_track_id"]): person for person in pre_frame["people"]
            }
            b0_pre_by_track = {
                int(person["global_track_id"]): person
                for person in b0_segments[segment_index - 1][-1]["people"]
            }
            association = associations[segment_index]
            track_post_pairs = sorted(association["track_to_post_index"].items())
            pre_people = [pre_by_track[int(track)] for track, _ in track_post_pairs]
            post_people = post_frames[0]["people"]
            matches = [(index, int(post)) for index, (_, post) in enumerate(track_post_pairs)]
            corrected_first, debug = refine_matched_people(
                np.asarray(pre_frame["method_camera_c2w"], dtype=np.float64),
                np.asarray(post_frames[0]["method_camera_c2w"], dtype=np.float64),
                pre_people,
                post_people,
                matches,
                BRTC_CONFIG,
            )
            shift_by_native = {}
            for _, post_index in matches:
                native = int(post_people[post_index]["native_track_id"])
                shift_by_native[native] = (
                    np.asarray(corrected_first[post_index]["root"], dtype=np.float64)
                    - np.asarray(post_people[post_index]["root"], dtype=np.float64)
                )
            for frame in post_frames:
                frame["people"] = [
                    shift_person(person, shift_by_native.get(int(person["native_track_id"]), np.zeros(3)))
                    for person in frame["people"]
                ]
            boundary_debug.append(
                {
                    "chain_index": int(source_chain["chain_index"]),
                    "cut_index": segment_index - 1,
                    "association": association,
                    "brtc_runtime_module": refine_matched_people.__module__,
                    "brtc": debug,
                    "shift_by_native_track": shift_by_native,
                    "pre_inherited_brtc_shift_by_global_track": {
                        int(track): (
                            np.asarray(person["root"], dtype=np.float64)
                            - np.asarray(b0_pre_by_track[track]["root"], dtype=np.float64)
                        )
                        for track, person in pre_by_track.items()
                        if track in b0_pre_by_track
                    },
                }
            )
            brtc_segments.append(post_frames)

        for method, segments in (
            ("raw_reset", raw_method_segments),
            ("b0", b0_segments),
            ("b0_brtc_lc", brtc_segments),
        ):
            all_methods[method].append(
                {
                    "chain_index": int(source_chain["chain_index"]),
                    "segments": segments,
                    "frames": [frame for segment in segments for frame in segment],
                }
            )
    return all_methods, boundary_debug


RefinementCallback = Callable[
    [
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[tuple[int, int]],
    ],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


def frozen_huber_config(path: Path = HUBER_POLICY_PATH) -> ReliabilityHuberConfig:
    """Load the policy frozen on the independent ``three offset0`` dev split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload["policy"])
    values["joint_ids"] = tuple(int(value) for value in values["joint_ids"])
    values["residual_lambda_grid"] = tuple(
        float(value) for value in values["residual_lambda_grid"]
    )
    if values.get("huber_delta_m") == "inf":
        values["huber_delta_m"] = float("inf")
    return ReliabilityHuberConfig(**values)


def refinement_variant_registry() -> dict[str, dict[str, Any]]:
    """Return frozen GT-free callbacks evaluated on exactly the same B0 geometry.

    The completeness runtime is imported lazily because it is an independent
    deployable module.  None of the policies below were tuned on the three
    EgoHumans confirmation chains used by this evaluator.
    """
    from versions.v14.b0_person_triangulation_completeness_weighted import (
        refine_matched_people_completeness_weighted,
    )

    huber_config = frozen_huber_config()

    def current(
        pre_camera: np.ndarray,
        post_camera: np.ndarray,
        pre_people: list[dict[str, Any]],
        post_people: list[dict[str, Any]],
        matches: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return refine_matched_people(
            pre_camera, post_camera, pre_people, post_people, matches, BRTC_CONFIG
        )

    def completeness(
        pre_camera: np.ndarray,
        post_camera: np.ndarray,
        pre_people: list[dict[str, Any]],
        post_people: list[dict[str, Any]],
        matches: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return refine_matched_people_completeness_weighted(
            pre_camera, post_camera, pre_people, post_people, matches, BRTC_CONFIG
        )

    def damped(
        pre_camera: np.ndarray,
        post_camera: np.ndarray,
        pre_people: list[dict[str, Any]],
        post_people: list[dict[str, Any]],
        matches: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return refine_matched_people_damped(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            BRTC_CONFIG,
            DEFAULT_DAMPED_CONFIG,
        )

    def huber(
        pre_camera: np.ndarray,
        post_camera: np.ndarray,
        pre_people: list[dict[str, Any]],
        post_people: list[dict[str, Any]],
        matches: list[tuple[int, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return refine_matched_people_huber(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            huber_config,
        )

    return {
        "b0_brtc_lc_replay": {
            "callback": current,
            "runtime_module": refine_matched_people.__module__,
            "policy": "frozen BRTC-LC v1",
            "runtime_sha256": BRTC_V1_RUNTIME_SHA256,
        },
        "b0_brtc_completeness_weighted": {
            "callback": completeness,
            "runtime_module": refine_matched_people_completeness_weighted.__module__,
            "policy": (
                "v1 final accepted shifts multiplied by matched_count / "
                "max(pre_count, post_count, 1)"
            ),
        },
        "b0_brtc_damped_0p8": {
            "callback": damped,
            "runtime_module": refine_matched_people_damped.__module__,
            "policy": "frozen observable-independent action_scale=0.8",
        },
        "b0_brtc_huber_irls_frozen": {
            "callback": huber,
            "runtime_module": refine_matched_people_huber.__module__,
            "policy": "frozen reliability-weighted Huber-IRLS",
            "policy_source": str(HUBER_POLICY_PATH),
            "config": dict(vars(huber_config)),
        },
    }


def replay_refinement_variant(
    b0_chains: list[dict[str, Any]],
    frozen_boundary_rows: list[dict[str, Any]],
    name: str,
    callback: RefinementCallback,
    runtime_module: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Causally replay one person refinement over the two cuts of each chain."""
    boundary_by_key = {
        (int(row["chain_index"]), int(row["cut_index"])): row
        for row in frozen_boundary_rows
    }
    output_chains, runtime_rows = [], []
    for b0_chain in b0_chains:
        chain_index = int(b0_chain["chain_index"])
        b0_segments = b0_chain["segments"]
        candidate_segments = [copy.deepcopy(b0_segments[0])]
        for segment_index in (1, 2):
            post_frames = copy.deepcopy(b0_segments[segment_index])
            pre_frame = candidate_segments[-1][-1]
            pre_by_track = {
                int(person["global_track_id"]): person
                for person in pre_frame["people"]
            }
            b0_pre_by_track = {
                int(person["global_track_id"]): person
                for person in b0_segments[segment_index - 1][-1]["people"]
            }
            frozen = boundary_by_key[(chain_index, segment_index - 1)]
            association = frozen["association"]
            track_post_pairs = sorted(association["track_to_post_index"].items())
            # Supply the full observable pre population.  Frozen BRTC only
            # reads matched indices, while completeness weighting also needs
            # the true entry/exit denominator.
            pre_people = list(pre_frame["people"])
            pre_index_by_track = {
                int(person["global_track_id"]): index
                for index, person in enumerate(pre_people)
            }
            post_people = post_frames[0]["people"]
            matches = [
                (pre_index_by_track[int(track)], int(post_index))
                for track, post_index in track_post_pairs
            ]
            corrected_first, debug = callback(
                np.asarray(pre_frame["method_camera_c2w"], dtype=np.float64),
                np.asarray(post_frames[0]["method_camera_c2w"], dtype=np.float64),
                pre_people,
                post_people,
                matches,
            )
            if len(corrected_first) != len(post_people):
                raise ValueError(f"{name} changed the post person count")
            if debug.get("camera_update") != "none":
                raise ValueError(f"{name} attempted a camera update")
            if int(debug.get("matched_count", -1)) != len(matches):
                raise ValueError(f"{name} returned an inconsistent matched_count")

            matched_post = {post_index for _, post_index in matches}
            shift_by_native: dict[int, np.ndarray] = {}
            unmatched_max_abs_change = 0.0
            for post_index, (before, after) in enumerate(
                zip(post_people, corrected_first)
            ):
                shift = (
                    np.asarray(after["root"], dtype=np.float64)
                    - np.asarray(before["root"], dtype=np.float64)
                )
                native = int(before["native_track_id"])
                shift_by_native[native] = shift
                if post_index not in matched_post:
                    unmatched_max_abs_change = max(
                        unmatched_max_abs_change, float(np.max(np.abs(shift)))
                    )
            if unmatched_max_abs_change > 1e-12:
                raise ValueError(f"{name} modified an unmatched person")

            zero = np.zeros(3, dtype=np.float64)
            for frame in post_frames:
                frame["people"] = [
                    shift_person(
                        person,
                        shift_by_native.get(int(person["native_track_id"]), zero),
                    )
                    for person in frame["people"]
                ]
            runtime_rows.append(
                {
                    "variant": name,
                    "runtime_module": runtime_module,
                    "chain_index": chain_index,
                    "cut_index": segment_index - 1,
                    "association": association,
                    "refinement": debug,
                    "shift_by_native_track": shift_by_native,
                    "unmatched_max_abs_change": unmatched_max_abs_change,
                    "pre_inherited_variant_shift_by_global_track": {
                        int(track): (
                            np.asarray(person["root"], dtype=np.float64)
                            - np.asarray(b0_pre_by_track[track]["root"], dtype=np.float64)
                        )
                        for track, person in pre_by_track.items()
                        if track in b0_pre_by_track
                    },
                }
            )
            candidate_segments.append(post_frames)
        output_chains.append(
            {
                "chain_index": chain_index,
                "segments": candidate_segments,
                "frames": [
                    frame for segment in candidate_segments for frame in segment
                ],
            }
        )
    return output_chains, runtime_rows


def geometry_parity_audit(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> dict[str, Any]:
    deltas = {"camera": 0.0, "root": 0.0, "joints": 0.0, "vertices": 0.0}
    if len(expected) != len(actual):
        raise ValueError("Parity chain count mismatch")
    for expected_chain, actual_chain in zip(expected, actual):
        if len(expected_chain["frames"]) != len(actual_chain["frames"]):
            raise ValueError("Parity frame count mismatch")
        for expected_frame, actual_frame in zip(
            expected_chain["frames"], actual_chain["frames"]
        ):
            deltas["camera"] = max(
                deltas["camera"],
                float(
                    np.max(
                        np.abs(
                            np.asarray(expected_frame["method_camera_c2w"])
                            - np.asarray(actual_frame["method_camera_c2w"])
                        )
                    )
                ),
            )
            expected_people = {
                int(person["native_track_id"]): person
                for person in expected_frame["people"]
            }
            actual_people = {
                int(person["native_track_id"]): person
                for person in actual_frame["people"]
            }
            if set(expected_people) != set(actual_people):
                raise ValueError("Parity native person IDs differ")
            for native in expected_people:
                for key in ("root", "joints", "vertices"):
                    deltas[key] = max(
                        deltas[key],
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(expected_people[native][key])
                                    - np.asarray(actual_people[native][key])
                                )
                            )
                        ),
                    )
    return {
        "max_abs_delta": deltas,
        "bit_parity": bool(max(deltas.values()) == 0.0),
        "tolerance_parity_1e_12": bool(max(deltas.values()) <= 1e-12),
    }


def camera_exactness_audit(
    b0_chains: list[dict[str, Any]], candidate_chains: list[dict[str, Any]]
) -> dict[str, Any]:
    deltas = []
    for b0_chain, candidate_chain in zip(b0_chains, candidate_chains):
        for b0_frame, candidate_frame in zip(
            b0_chain["frames"], candidate_chain["frames"]
        ):
            deltas.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(b0_frame["method_camera_c2w"])
                            - np.asarray(candidate_frame["method_camera_c2w"])
                        )
                    )
                )
            )
    maximum = float(max(deltas, default=0.0))
    return {"max_abs_change": maximum, "bit_exact": bool(maximum == 0.0)}


def refinement_runtime_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = sum(int(row["refinement"]["matched_count"]) for row in rows)
    accepted = sum(int(row["refinement"]["accepted_count"]) for row in rows)
    unmatched_change = max(
        (float(row["unmatched_max_abs_change"]) for row in rows), default=0.0
    )
    scales = [
        float(
            row["refinement"].get(
                "match_scale", row["refinement"].get("completeness")
            )
        )
        for row in rows
        if "match_scale" in row["refinement"]
        or "completeness" in row["refinement"]
    ]
    return {
        "boundary_count": len(rows),
        "matched_count": matched,
        "accepted_count": accepted,
        "acceptance": float(accepted / max(matched, 1)),
        "unmatched_max_abs_change": unmatched_change,
        "match_scales": scales,
        "partial_match_boundary_count": int(sum(value < 1.0 for value in scales)),
    }


def gt_frame(data_root: Path, frame: int) -> dict[str, dict]:
    value = np.load(
        data_root / f"processed_data/smpl/{frame:05d}.npy", allow_pickle=True
    ).item()
    return {str(key): row for key, row in value.items()}


def pa_error(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_point_error(target, apply_similarity(prediction, fit_similarity(target, prediction))))


def identity_switch_count(values: list[int | None]) -> int:
    previous = None
    switches = 0
    for value in values:
        if value is None:
            continue
        if previous is not None and int(value) != int(previous):
            switches += 1
        previous = int(value)
    return switches


def finite_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def evaluate_chain(
    chain: dict[str, Any],
    data_root: Path,
    exo: dict,
    vertex_map: SparseVertexMap,
    joint_regressor: np.ndarray,
    fps: float,
) -> tuple[dict, dict[str, np.ndarray], dict[tuple[int, str], float]]:
    frames = chain["frames"]
    first = frames[0]
    first_target_camera = np.asarray(exo[first["camera_name"]]["c2w_aria01"], dtype=np.float64)
    gauge = np.asarray(first["method_camera_c2w"], dtype=np.float64) @ np.linalg.inv(
        first_target_camera
    )
    tracks = {
        identity: {
            "frame_indices": [],
            "pred_world_joints": [],
            "target_world_joints": [],
            "pred_camera_centered_joints": [],
            "target_camera_centered_joints": [],
        }
        for identity in IDENTITIES
    }
    track_ids = {identity: [None] * len(frames) for identity in IDENTITIES}
    arrays: dict[str, list[float]] = {
        "fixed_root_m": [],
        "fixed_joint_m": [],
        "fixed_vertex_m": [],
        "pelvis_mpjpe_m": [],
        "pelvis_mpvpe_m": [],
        "pa_mpjpe_m": [],
        "pa_mpvpe_m": [],
        "layout_distance_m": [],
        "layout_vector_m": [],
    }
    pred_cameras, target_cameras = [], []
    root_errors: dict[tuple[int, str], float] = {}
    matched = 0
    for frame_index, frame in enumerate(frames):
        pred_camera = np.asarray(frame["method_camera_c2w"], dtype=np.float64)
        target_camera = gauge @ np.asarray(
            exo[frame["camera_name"]]["c2w_aria01"], dtype=np.float64
        )
        pred_cameras.append(pred_camera)
        target_cameras.append(target_camera)
        target_bodies = gt_frame(data_root, int(frame["dataset_frame"]))
        predicted_roots, target_roots = {}, {}
        for person in frame["people"]:
            label = int(person["gt_label_evaluator_only"])
            if not (0 <= label < len(IDENTITIES)):
                continue
            identity = IDENTITIES[label]
            if identity not in target_bodies:
                continue
            pred_vertices_world = vertex_map.apply(
                np.asarray(person["vertices"], dtype=np.float32)[None]
            )[0].astype(np.float64)
            pred_joints_world = joint_regressor @ pred_vertices_world
            target_vertices_world = transform_points(
                gauge, np.asarray(target_bodies[identity]["vertices"], dtype=np.float64)
            )
            target_joints_world = joint_regressor @ target_vertices_world
            pred_vertices_camera = transform_points(
                np.linalg.inv(pred_camera), pred_vertices_world
            )
            pred_joints_camera = joint_regressor @ pred_vertices_camera
            target_vertices_camera = transform_points(
                np.linalg.inv(target_camera), target_vertices_world
            )
            target_joints_camera = joint_regressor @ target_vertices_camera
            pred_center_joints, pred_center_vertices = pelvis_center(
                pred_joints_camera[None], pred_vertices_camera[None]
            )
            target_center_joints, target_center_vertices = pelvis_center(
                target_joints_camera[None], target_vertices_camera[None]
            )
            pred_center_joints = pred_center_joints[0]
            pred_center_vertices = pred_center_vertices[0]
            target_center_joints = target_center_joints[0]
            target_center_vertices = target_center_vertices[0]
            root_error = float(np.linalg.norm(pred_joints_world[0] - target_joints_world[0]))
            arrays["fixed_root_m"].append(root_error)
            arrays["fixed_joint_m"].append(
                float(np.linalg.norm(pred_joints_world - target_joints_world, axis=1).mean())
            )
            arrays["fixed_vertex_m"].append(
                float(np.linalg.norm(pred_vertices_world - target_vertices_world, axis=1).mean())
            )
            arrays["pelvis_mpjpe_m"].append(
                float(mean_point_error(target_center_joints, pred_center_joints))
            )
            arrays["pelvis_mpvpe_m"].append(
                float(mean_point_error(target_center_vertices, pred_center_vertices))
            )
            arrays["pa_mpjpe_m"].append(pa_error(target_center_joints, pred_center_joints))
            arrays["pa_mpvpe_m"].append(pa_error(target_center_vertices, pred_center_vertices))
            tracks[identity]["frame_indices"].append(frame_index)
            tracks[identity]["pred_world_joints"].append(pred_joints_world)
            tracks[identity]["target_world_joints"].append(target_joints_world)
            tracks[identity]["pred_camera_centered_joints"].append(pred_center_joints)
            tracks[identity]["target_camera_centered_joints"].append(target_center_joints)
            track_ids[identity][frame_index] = int(person["global_track_id"])
            root_errors[(frame_index, identity)] = root_error
            predicted_roots[identity] = pred_joints_world[0]
            target_roots[identity] = target_joints_world[0]
            matched += 1
        for first_identity, second_identity in combinations(sorted(predicted_roots), 2):
            pred_vector = predicted_roots[first_identity] - predicted_roots[second_identity]
            target_vector = target_roots[first_identity] - target_roots[second_identity]
            arrays["layout_distance_m"].append(
                abs(float(np.linalg.norm(pred_vector) - np.linalg.norm(target_vector)))
            )
            arrays["layout_vector_m"].append(float(np.linalg.norm(pred_vector - target_vector)))

    trajectory_arrays: dict[str, list[np.ndarray]] = {
        "w_mpjpe_m": [],
        "wa_mpjpe_m": [],
        "accel_delta2_m": [],
        "accel_physical_m_s2": [],
        "world_joint_accel_delta2_m": [],
        "world_root_accel_delta2_m": [],
    }
    per_identity = {}
    for identity, values in tracks.items():
        indices = np.asarray(values["frame_indices"], dtype=np.int64)
        if len(indices) < 2:
            continue
        pred_world = np.stack(values["pred_world_joints"])
        target_world = np.stack(values["target_world_joints"])
        pred_center = np.stack(values["pred_camera_centered_joints"])
        target_center = np.stack(values["target_camera_centered_joints"])
        row_arrays = {
            "w_mpjpe_m": initial_frame_aligned_errors(target_world, pred_world),
            "wa_mpjpe_m": trajectory_aligned_errors(target_world, pred_world),
            "accel_delta2_m": acceleration_second_difference_errors(
                target_center, pred_center, indices
            ),
            "accel_physical_m_s2": acceleration_errors(
                target_center, pred_center, fps, indices
            ),
            "world_joint_accel_delta2_m": acceleration_second_difference_errors(
                target_world, pred_world, indices
            ),
            "world_root_accel_delta2_m": acceleration_second_difference_errors(
                target_world[:, :1], pred_world[:, :1], indices
            ),
        }
        for key, value in row_arrays.items():
            trajectory_arrays[key].append(value)
        per_identity[identity] = {
            "observed_frames": int(len(indices)),
            "identity_switches": identity_switch_count(track_ids[identity]),
            "w_mpjpe_mm": float(row_arrays["w_mpjpe_m"].mean() * 1000.0),
            "wa_mpjpe_mm": float(row_arrays["wa_mpjpe_m"].mean() * 1000.0),
        }
    flat_trajectory = {
        key: np.concatenate(value) if value else np.empty(0, dtype=np.float64)
        for key, value in trajectory_arrays.items()
    }
    pred_centers = np.stack(pred_cameras)[:, :3, 3]
    target_centers = np.stack(target_cameras)[:, :3, 3]
    camera_fit = fit_similarity(target_centers, pred_centers)
    ate = np.linalg.norm(target_centers - apply_similarity(pred_centers, camera_fit), axis=1)
    ids_by_identity = {
        identity: identity_switch_count(values) for identity, values in track_ids.items()
    }
    result = {
        "chain_index": int(chain["chain_index"]),
        "frame_count": len(frames),
        "matched_person_frames": matched,
        "possible_person_frames": len(frames) * len(IDENTITIES),
        "coverage": float(matched / (len(frames) * len(IDENTITIES))),
        "metrics": {
            "w_mpjpe_mm": finite_mean(flat_trajectory["w_mpjpe_m"]) * 1000.0,
            "wa_mpjpe_mm": finite_mean(flat_trajectory["wa_mpjpe_m"]) * 1000.0,
            "pelvis_mpjpe_mm": finite_mean(arrays["pelvis_mpjpe_m"]) * 1000.0,
            "pelvis_mpvpe_mm": finite_mean(arrays["pelvis_mpvpe_m"]) * 1000.0,
            "procrustes_pa_mpjpe_mm": finite_mean(arrays["pa_mpjpe_m"]) * 1000.0,
            "procrustes_pa_mpvpe_mm": finite_mean(arrays["pa_mpvpe_m"]) * 1000.0,
            "accel_delta2_mm_per_frame2": finite_mean(flat_trajectory["accel_delta2_m"]) * 1000.0,
            "accel_physical_m_per_s2": finite_mean(flat_trajectory["accel_physical_m_s2"]),
            "world_joint_accel_delta2_mm_per_frame2": finite_mean(
                flat_trajectory["world_joint_accel_delta2_m"]
            ) * 1000.0,
            "world_root_accel_delta2_mm_per_frame2": finite_mean(
                flat_trajectory["world_root_accel_delta2_m"]
            ) * 1000.0,
            "ate_m_sim3": float(np.sqrt(np.mean(np.square(ate)))),
            "identity_switches": int(sum(ids_by_identity.values())),
            "fixed_world_root_mm": finite_mean(arrays["fixed_root_m"]) * 1000.0,
            "fixed_world_joint_mm": finite_mean(arrays["fixed_joint_m"]) * 1000.0,
            "fixed_world_vertex_mm": finite_mean(arrays["fixed_vertex_m"]) * 1000.0,
            "pairwise_root_distance_mm": finite_mean(arrays["layout_distance_m"]) * 1000.0,
            "pairwise_root_vector_mm": finite_mean(arrays["layout_vector_m"]) * 1000.0,
        },
        "identity_switches_by_gt_identity": ids_by_identity,
        "per_identity": per_identity,
    }
    exported = {
        **flat_trajectory,
        **{key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
        "ate_m": ate,
    }
    return result, exported, root_errors


def aggregate_method(
    results: list[dict], arrays: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    def joined(key: str) -> np.ndarray:
        values = [row[key] for row in arrays if len(row[key])]
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)

    possible = sum(row["possible_person_frames"] for row in results)
    matched = sum(row["matched_person_frames"] for row in results)
    return {
        "chain_count": len(results),
        "frame_count": sum(row["frame_count"] for row in results),
        "matched_person_frames": matched,
        "possible_person_frames": possible,
        "coverage": float(matched / possible),
        "metrics": {
            "w_mpjpe_mm": finite_mean(joined("w_mpjpe_m")) * 1000.0,
            "wa_mpjpe_mm": finite_mean(joined("wa_mpjpe_m")) * 1000.0,
            "pelvis_mpjpe_mm": finite_mean(joined("pelvis_mpjpe_m")) * 1000.0,
            "pelvis_mpvpe_mm": finite_mean(joined("pelvis_mpvpe_m")) * 1000.0,
            "procrustes_pa_mpjpe_mm": finite_mean(joined("pa_mpjpe_m")) * 1000.0,
            "procrustes_pa_mpvpe_mm": finite_mean(joined("pa_mpvpe_m")) * 1000.0,
            "accel_delta2_mm_per_frame2": finite_mean(joined("accel_delta2_m")) * 1000.0,
            "accel_physical_m_per_s2": finite_mean(joined("accel_physical_m_s2")),
            "world_joint_accel_delta2_mm_per_frame2": finite_mean(
                joined("world_joint_accel_delta2_m")
            ) * 1000.0,
            "world_root_accel_delta2_mm_per_frame2": finite_mean(
                joined("world_root_accel_delta2_m")
            ) * 1000.0,
            "ate_m_sim3": float(np.sqrt(np.mean(np.square(joined("ate_m"))))),
            "identity_switches_mean_per_stream": float(
                np.mean([row["metrics"]["identity_switches"] for row in results])
            ),
            "identity_switches_total": int(
                sum(row["metrics"]["identity_switches"] for row in results)
            ),
            "fixed_world_root_mm": finite_mean(joined("fixed_root_m")) * 1000.0,
            "fixed_world_joint_mm": finite_mean(joined("fixed_joint_m")) * 1000.0,
            "fixed_world_vertex_mm": finite_mean(joined("fixed_vertex_m")) * 1000.0,
            "pairwise_root_distance_mm": finite_mean(joined("layout_distance_m")) * 1000.0,
            "pairwise_root_vector_mm": finite_mean(joined("layout_vector_m")) * 1000.0,
        },
        "per_chain": results,
    }


def b0_replay_audit(
    cache: dict[str, Any], b0_methods: list[dict], data_root: Path, exo: dict,
    vertex_map: SparseVertexMap, joint_regressor: np.ndarray,
) -> dict[str, Any]:
    del b0_methods
    deltas = {"root_error_m": [], "joint_error_m": [], "vertex_error_m": []}
    labels_equal = []
    aligned_camera_max_abs_delta = []
    for chain in cache["chains"]:
        for cut in (0, 1):
            source = json.loads(Path(chain["b0_case_reports"][cut]).read_text(encoding="utf-8"))
            pre_segment = chain["segments"][cut]
            post_segment = chain["segments"][cut + 1]
            boundary = np.asarray(chain["b0_boundaries"][cut], dtype=np.float64)
            pre_camera = np.asarray(pre_segment["frames"][-1]["camera_c2w"], dtype=np.float64)
            gt_pre_camera = np.asarray(exo[pre_segment["camera_name"]]["c2w_aria01"], dtype=np.float64)
            gauge = pre_camera @ np.linalg.inv(gt_pre_camera)
            for frame_index, (cached_frame, source_frame) in enumerate(
                zip(post_segment["frames"], source["frames"])
            ):
                replay_camera = boundary @ np.asarray(
                    cached_frame["camera_c2w"], dtype=np.float64
                )
                aligned_camera_max_abs_delta.append(
                    float(
                        np.max(
                            np.abs(
                                replay_camera
                                - np.asarray(
                                    source_frame["methods"]["b0"]["estimated_camera"],
                                    dtype=np.float64,
                                )
                            )
                        )
                    )
                )
                labels_equal.append(
                    [int(person["gt_label_evaluator_only"]) for person in cached_frame["people"]]
                    == [int(value) for value in source_frame["labels_by_detection"]]
                )
                targets = gt_frame(data_root, int(cached_frame["dataset_frame"]))
                for person in cached_frame["people"]:
                    label = int(person["gt_label_evaluator_only"])
                    if not (0 <= label < len(IDENTITIES)):
                        continue
                    identity = IDENTITIES[label]
                    expected = source_frame["methods"]["b0"]["per_person"].get(identity)
                    if expected is None:
                        continue
                    pred_vertices = vertex_map.apply(
                        np.asarray(
                            transform_points(boundary, person["vertices"]), dtype=np.float32
                        )[None]
                    )[0].astype(np.float64)
                    pred_joints = joint_regressor @ pred_vertices
                    target_vertices = transform_points(
                        gauge, np.asarray(targets[identity]["vertices"], dtype=np.float64)
                    )
                    target_joints = joint_regressor @ target_vertices
                    observed = {
                        "root_error_m": float(np.linalg.norm(pred_joints[0] - target_joints[0])),
                        "joint_error_m": float(
                            np.linalg.norm(pred_joints - target_joints, axis=1).mean()
                        ),
                        "vertex_error_m": float(
                            np.linalg.norm(pred_vertices - target_vertices, axis=1).mean()
                        ),
                    }
                    for key in deltas:
                        deltas[key].append(abs(observed[key] - float(expected[key])))
    maximum = {key: float(max(value, default=float("nan"))) for key, value in deltas.items()}
    return {
        "purpose": "verify CPU-regenerated current-checkpoint geometry replays cached GPU B0 scores",
        "labels_exact": bool(all(labels_equal)),
        "compared_person_frames": len(deltas["root_error_m"]),
        "b0_aligned_camera_max_abs_delta": float(
            max(aligned_camera_max_abs_delta, default=float("nan"))
        ),
        "max_abs_metric_delta_m": maximum,
        "same_forward_replay_verified_1cm": bool(
            all(value <= 0.01 for value in maximum.values()) and all(labels_equal)
        ),
    }


def v13_raw_parity_audit(
    cache: dict[str, Any], current_raw: dict[str, Any]
) -> dict[str, Any]:
    paths = (
        REPO_ROOT / "output/v13/egobody/v13_egobody_compact_tokens.pt",
        REPO_ROOT
        / "output/v13/egobody_cam02_cam05_cam08/v13_egobody_compact_tokens.pt",
        REPO_ROOT
        / "output/v13/egobody_cam03_cam04_cam01/v13_egobody_compact_tokens.pt",
    )
    camera_deltas = []
    frame_counts = []
    for source_chain, path in zip(cache["chains"], paths):
        old = torch.load(path, map_location="cpu", weights_only=False)
        old_predictions = old["predictions"]
        current_frames = [
            frame
            for segment in source_chain["segments"]
            for frame in segment["frames"]
        ]
        if len(old_predictions) != len(current_frames):
            raise ValueError(f"V13/current frame count mismatch at {path}")
        frame_counts.append(len(current_frames))
        for prediction, frame in zip(old_predictions, current_frames):
            camera_deltas.append(
                float(
                    np.max(
                        np.abs(
                            pose_encoding_to_camera_matrix(prediction["camera_pose"][0])
                            - np.asarray(frame["camera_c2w"], dtype=np.float64)
                        )
                    )
                )
            )
    old_report_path = (
        REPO_ROOT
        / "output/v14/fine_alignment_research/multithumbs_protocol/"
        "human3r_raw_egohumans_provisional.json"
    )
    old_metric = json.loads(old_report_path.read_text(encoding="utf-8"))["aggregate"][
        "metrics"
    ]
    current = current_raw["metrics"]
    metric_pairs = {
        "w_mpjpe_mm": (current["w_mpjpe_mm"], old_metric["w_mpjpe_mm"]),
        "wa_mpjpe_mm": (current["wa_mpjpe_mm"], old_metric["wa_mpjpe_mm"]),
        "pelvis_mpjpe_mm": (current["pelvis_mpjpe_mm"], old_metric["mpjpe_mm"]),
        "pelvis_mpvpe_mm": (current["pelvis_mpvpe_mm"], old_metric["mpvpe_mm"]),
        "accel_delta2_mm_per_frame2": (
            current["accel_delta2_mm_per_frame2"],
            old_metric["accel_second_difference_mm_per_frame2"],
        ),
        "accel_physical_m_per_s2": (
            current["accel_physical_m_per_s2"],
            old_metric["accel_physical_m_per_s2"],
        ),
        "ate_m_sim3": (current["ate_m_sim3"], old_metric["ate_m_sim3_translation_rmse"]),
    }
    absolute_deltas = {
        key: abs(float(first) - float(second))
        for key, (first, second) in metric_pairs.items()
    }
    return {
        "v13_compact_frame_counts": frame_counts,
        "raw_camera_c2w_max_abs_delta": float(max(camera_deltas, default=float("nan"))),
        "aggregate_metric_abs_delta": absolute_deltas,
        "metric_parity_verified": bool(
            absolute_deltas["w_mpjpe_mm"] < 0.1
            and absolute_deltas["wa_mpjpe_mm"] < 0.1
            and absolute_deltas["pelvis_mpjpe_mm"] < 0.1
            and absolute_deltas["pelvis_mpvpe_mm"] < 0.1
            and absolute_deltas["accel_delta2_mm_per_frame2"] < 0.1
            and absolute_deltas["accel_physical_m_per_s2"] < 0.01
            and absolute_deltas["ate_m_sim3"] < 1e-3
        ),
        "ids_excluded_from_parity": (
            "this evaluator prefixes raw IDs by shot to make hard resets explicit; "
            "the older evaluator compared reused local integer IDs directly"
        ),
    }


def harm_audit(
    b0_roots: list[dict[tuple[int, str], float]],
    brtc_roots: list[dict[tuple[int, str], float]],
) -> dict[str, Any]:
    all_delta, boundary_delta = [], []
    for b0, brtc in zip(b0_roots, brtc_roots):
        shared = sorted(set(b0) & set(brtc))
        for key in shared:
            delta = float(brtc[key] - b0[key])
            if key[0] >= 5:
                all_delta.append(delta)
            if key[0] in (5, 10):
                boundary_delta.append(delta)

    def summarize(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(len(array)),
            "mean_delta_mm": float(array.mean() * 1000.0),
            "improve_rate": float(np.mean(array < 0.0)),
            "harm_over_5cm_rate": float(np.mean(array > 0.05)),
        }

    return {
        "all_person_frames_in_corrected_post_shots": summarize(all_delta),
        "first_post_boundary_person_frames": summarize(boundary_delta),
    }


def promotion_audit(
    method_reports: dict[str, dict[str, Any]],
    harms: dict[str, dict[str, Any]],
    camera_exactness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare frozen candidates against current BRTC without GT-side tuning."""
    reference_name = "b0_brtc_lc"
    reference = method_reports[reference_name]
    reference_metrics = reference["metrics"]
    reference_harm = harms[reference_name][
        "all_person_frames_in_corrected_post_shots"
    ]["harm_over_5cm_rate"]
    required_metric_keys = (
        "w_mpjpe_mm",
        "wa_mpjpe_mm",
        "fixed_world_root_mm",
        "world_root_accel_delta2_mm_per_frame2",
    )
    candidates = {}
    for name in REFINEMENT_VARIANTS:
        candidate = method_reports[name]
        metrics = candidate["metrics"]
        harm = harms[name]["all_person_frames_in_corrected_post_shots"][
            "harm_over_5cm_rate"
        ]
        deltas = {
            key: float(metrics[key] - reference_metrics[key])
            for key in required_metric_keys
        }
        deltas["harm_over_5cm_rate"] = float(harm - reference_harm)
        required_better = {
            key: bool(value < -1e-12) for key, value in deltas.items()
        }
        dominates = bool(all(required_better.values()))
        coverage_preserved = bool(candidate["coverage"] >= reference["coverage"] - 1e-12)
        camera_exact = bool(camera_exactness[name]["bit_exact"])
        harm_safe = bool(harm <= 0.10)
        paper_gaps = {
            "w_mpjpe_mm": float(metrics["w_mpjpe_mm"] - PAPER_REFERENCE["w_mpjpe_mm"]),
            "wa_mpjpe_mm": float(
                metrics["wa_mpjpe_mm"] - PAPER_REFERENCE["wa_mpjpe_mm"]
            ),
        }
        candidates[name] = {
            "delta_vs_current_brtc": deltas,
            "strictly_better_by_required_item": required_better,
            "strictly_dominates_current_brtc_required_items": dominates,
            "coverage_preserved": coverage_preserved,
            "camera_bit_exact": camera_exact,
            "harm_under_10pct_safety_line": harm_safe,
            "paper_reference_gap_mm_provisional": paper_gaps,
            "below_paper_w_wa_reference_provisional": bool(
                paper_gaps["w_mpjpe_mm"] < 0.0
                and paper_gaps["wa_mpjpe_mm"] < 0.0
            ),
            "eligible_for_final_mainline": bool(
                dominates and coverage_preserved and camera_exact and harm_safe
            ),
        }
    dominant = [
        name
        for name, value in candidates.items()
        if value["strictly_dominates_current_brtc_required_items"]
        and value["coverage_preserved"]
        and value["camera_bit_exact"]
    ]
    eligible = [
        name
        for name, value in candidates.items()
        if value["eligible_for_final_mainline"]
    ]
    return {
        "reference_method": reference_name,
        "required_items": [
            *required_metric_keys,
            "harm_over_5cm_rate",
            "coverage_preserved",
            "camera_bit_exact",
        ],
        "candidates": candidates,
        "dominant_candidate_order": dominant,
        "final_mainline_eligible": eligible,
        "decision": (
            "PROMOTE_" + eligible[0]
            if eligible
            else (
                "KEEP_AS_BEST_SAFER_CANDIDATE_BUT_NOT_FINAL_"
                + dominant[0]
                if dominant
                else "KEEP_CURRENT_BRTC_NO_CANDIDATE_DOMINATES"
            )
        ),
    }


def boundary_causal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    association_correct = sum(
        int(row["association"].get("correct_count_evaluator_only", 0)) for row in rows
    )
    association_total = sum(
        int(row["association"].get("matched_count", 0)) for row in rows
    )
    brtc_accepted = sum(int(row["brtc"]["accepted_count"]) for row in rows)
    brtc_matched = sum(int(row["brtc"]["matched_count"]) for row in rows)
    second_cut_inherited = []
    for row in rows:
        if int(row["cut_index"]) != 1:
            continue
        norms = [
            float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
            for value in row["pre_inherited_brtc_shift_by_global_track"].values()
        ]
        second_cut_inherited.extend(norms)
    return {
        "b0_composition": "G0=I; G1=B01; G2=B01@B12",
        "brtc_causal_rule": (
            "cut0 shift is propagated through segment1; cut1 consumes that corrected "
            "segment1 last frame and propagates a new shift through segment2"
        ),
        "anonymous_association_correct_evaluator_only": association_correct,
        "anonymous_association_total": association_total,
        "anonymous_association_accuracy_evaluator_only": float(
            association_correct / max(association_total, 1)
        ),
        "brtc_accepted_person_boundaries": brtc_accepted,
        "brtc_matched_person_boundaries": brtc_matched,
        "brtc_acceptance": float(brtc_accepted / max(brtc_matched, 1)),
        "second_cut_pre_inherited_shift_person_count": len(second_cut_inherited),
        "second_cut_pre_inherited_shift_nonzero_count": int(
            sum(value > 1e-10 for value in second_cut_inherited)
        ),
        "second_cut_pre_inherited_shift_norm_m": second_cut_inherited,
        "causal_inheritance_observed": bool(any(value > 1e-10 for value in second_cut_inherited)),
    }


def cache_audit() -> dict[str, Any]:
    paths = [
        REPO_ROOT / "output/v13/egobody/v13_egobody_compact_tokens.pt",
        REPO_ROOT / "output/v13/egobody_cam02_cam05_cam08/v13_egobody_compact_tokens.pt",
        REPO_ROOT / "output/v13/egobody_cam03_cam04_cam01/v13_egobody_compact_tokens.pt",
    ]
    return {
        "v13_compact_cache_count": sum(path.is_file() for path in paths),
        "v13_fields": ["predictions", "token_debug", "labels"],
        "v13_prediction_fields": [
            "camera_pose", "smpl_shape", "smpl_transl", "smpl_rotmat", "smpl_id"
        ],
        "v13_reuse_decision": (
            "not assumed interchangeable before parity audit: the archive was written by "
            "the original human3r_896L run while frozen B0 records a V14/V9 checkpoint"
        ),
        "b0_case_cache_fields": (
            "boundaries plus scalar frame errors; no saved current-checkpoint joints/vertices"
        ),
        "recovered_by_this_evaluator": (
            "current-checkpoint CPU raw cameras/SMPL-X geometry for all 45 observations"
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    rows = report["methods"]
    promotion = report["promotion_audit"]
    dominant = promotion["dominant_candidate_order"]
    best_name = dominant[0] if dominant else "b0_brtc_lc"
    best_metric = rows[best_name]["metrics"]
    best_harm = report["b0_to_refinement_harm"][best_name]
    current_metric = rows["b0_brtc_lc"]["metrics"]
    current_harm = report["b0_to_refinement_harm"]["b0_brtc_lc"]
    lines = [
        "# Frozen B0+BRTC variants：EgoHumans 同 forward Multi-THuMBS provisional 评测",
        "",
        "> 2026-08-01；全程 CPU，未使用 DA3/GPU。三路共享当前 V14/V9 checkpoint 的",
        "> 人体、相机与检测，只改变跨 shot 对齐。不是 Multi-THuMBS 官方 split/协议。",
        "> Active B0：`checkpoints/v14_brtc_lc_v1_b0/checkpoint-best.pth`，SHA256",
        f"> `{report['execution']['checkpoint_sha256']}`。cache/B0 case 中旧 `/dev/shm` 路径",
        "> 已由 `cmp -s` 验证为 bit-exact alias。",
        "",
        "## 1. 方法",
        "",
        "- `raw_reset`：三个五帧 shot 各自留在本地 gauge；",
        "- `b0`：将缓存的两个 frozen B0 边界按时间顺序累乘；",
        "- `b0_brtc_lc`：在每个边界用匿名 root+torso+joints Hungarian 匹配，调用",
        "  `versions/v14/b0_person_triangulation.py`，把修正平移传播到对应 post shot；",
        "- `b0_brtc_completeness_weighted`：先完整运行 frozen BRTC v1，再把 accepted final",
        "  shift 乘 `matched / max(pre人数, post人数)`；完整匹配时与 v1 bit-exact；",
        "- `b0_brtc_damped_0p8`：使用独立冻结的常数 `0.8` 缩放 individual proposal，再做",
        "  原有 group/layout consensus；",
        "- `b0_brtc_huber_irls_frozen`：使用在独立 `three offset0` 冻结的可靠性加权",
        "  Huber-IRLS ray center，不在本 EgoHumans confirmation 上调参；",
        "- 因果组合严格为 `G0=I, G1=B01, G2=B01@B12`；第二个 cut 的 pre 人体包含",
        "  第一个 cut 已传播的 person shift，再估计第三个 shot 的新 shift；",
        "- GT identity 只用于 evaluator；BRTC 匹配和修正不读 GT。",
        "",
        "## 2. 主要指标",
        "",
        "| Method | W | WA | pelvis MPJPE | pelvis MPVPE | PA-MPJPE | PA-MPVPE | Accel Δ² | Accel physical | ATE | IDs/stream | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = rows[method]
        value = row["metrics"]
        lines.append(
            f"| {method} | {value['w_mpjpe_mm']:.1f} | {value['wa_mpjpe_mm']:.1f} | "
            f"{value['pelvis_mpjpe_mm']:.1f} | {value['pelvis_mpvpe_mm']:.1f} | "
            f"{value['procrustes_pa_mpjpe_mm']:.1f} | {value['procrustes_pa_mpvpe_mm']:.1f} | "
            f"{value['accel_delta2_mm_per_frame2']:.2f} | "
            f"{value['accel_physical_m_per_s2']:.2f} | {value['ate_m_sim3']:.3f} | "
            f"{value['identity_switches_mean_per_stream']:.2f} | {row['coverage']:.1%} |"
        )
    lines.extend(
        [
            "",
            "这里 `pelvis MPJPE/MPVPE` 对应本地 Human3R/GVHMR 惯例；`PA-*` 另做每帧",
            "Sim(3) Procrustes。论文主文未确认其精确 pelvis/PA 口径。Accel 同时给离散",
            "二阶差分和按 30 fps 换算的物理单位；论文 `27.3` 的公式/单位仍未知。",
            "",
            "## 3. 对齐敏感的 fixed-world / layout proxy",
            "",
            "| Method | Root | World joint | World vertex | Pair distance | Pair vector | World root Accel Δ² |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        value = rows[method]["metrics"]
        lines.append(
            f"| {method} | {value['fixed_world_root_mm']:.1f} | "
            f"{value['fixed_world_joint_mm']:.1f} | {value['fixed_world_vertex_mm']:.1f} | "
            f"{value['pairwise_root_distance_mm']:.1f} | {value['pairwise_root_vector_mm']:.1f} | "
            f"{value['world_root_accel_delta2_mm_per_frame2']:.2f} |"
        )
    b0_metric = rows["b0"]["metrics"]
    lines.extend(
        [
            "",
            "## 4. 当前答案",
            "",
            f"BRTC-LC 在同 forward 连续链上确实有效：相对 B0，W-MPJPE "
            f"`{b0_metric['w_mpjpe_mm']:.1f}→{current_metric['w_mpjpe_mm']:.1f} mm`，"
            f"WA-MPJPE `{b0_metric['wa_mpjpe_mm']:.1f}→{current_metric['wa_mpjpe_mm']:.1f} mm`，"
            f"fixed-world root `{b0_metric['fixed_world_root_mm']:.1f}→"
            f"{current_metric['fixed_world_root_mm']:.1f} mm`，pair vector "
            f"`{b0_metric['pairwise_root_vector_mm']:.1f}→"
            f"{current_metric['pairwise_root_vector_mm']:.1f} mm`。",
            "",
            f"本轮最明确的新规律是 `{best_name}`。它相对 current BRTC 同时改善：",
            "",
            f"- W：`{current_metric['w_mpjpe_mm']:.3f} → {best_metric['w_mpjpe_mm']:.3f} mm`；",
            f"- WA：`{current_metric['wa_mpjpe_mm']:.3f} → {best_metric['wa_mpjpe_mm']:.3f} mm`；",
            f"- fixed-world root：`{current_metric['fixed_world_root_mm']:.3f} → "
            f"{best_metric['fixed_world_root_mm']:.3f} mm`；",
            f"- world-root Accel：`{current_metric['world_root_accel_delta2_mm_per_frame2']:.3f} → "
            f"{best_metric['world_root_accel_delta2_mm_per_frame2']:.3f} mm/frame²`；",
            f"- corrected-post >5 cm harm："
            f"`{current_harm['all_person_frames_in_corrected_post_shots']['harm_over_5cm_rate']:.1%} → "
            f"{best_harm['all_person_frames_in_corrected_post_shots']['harm_over_5cm_rate']:.1%}`。",
            "",
            "这是本 EgoHumans 小样本中唯一在 W、WA、fixed-world root、world-root Accel 和 harm 五项上",
            "同时严格优于 current BRTC 的候选。规律也可解释：5/6 个完整匹配 boundary 保持",
            "v1 原样，唯一 `1→3` 的不完整集合自动缩到 `1/3`，直接减少缺人场景中过激的",
            "多人 group action。它是无 GT、无新阈值、无未来帧的可部署安全变体。",
            "",
            "但它随后在独立 MultiHuman variable-visibility 22-cut 确认集上没有通过严格",
            "non-regression：相对 v1 虽将 harm `4.5%→2.3%`，并改善两种 layout，却使",
            "root/joint/vertex 分别退化约 `9.8/10.0/8.1 mm`。所以它不能冻结为新主线，只能作为",
            "“人数变化时需要更保守，但线性 completeness 阻尼过强”的探索证据。此外 EgoHumans 的",
            f"{best_harm['all_person_frames_in_corrected_post_shots']['harm_over_5cm_rate']:.1%} "
            "harm 仍高于 10% 安全线，first-post harm",
            f"仍为 `{best_harm['first_post_boundary_person_frames']['harm_over_5cm_rate']:.1%}`；"
            f"本地 W/WA 距论文参考值还差 `+{best_metric['w_mpjpe_mm'] - PAPER_REFERENCE['w_mpjpe_mm']:.1f}`/"
            f"`+{best_metric['wa_mpjpe_mm'] - PAPER_REFERENCE['wa_mpjpe_mm']:.1f} mm`。",
            "",
        ]
    )
    replay = report["same_forward_replay_audit"]
    v13_parity = report["v13_raw_parity_audit"]
    causal = report["boundary_causal_audit"]
    parity = report["current_brtc_generic_replay_parity"]
    lines.extend(
        [
            "",
            "## 5. 安全性与真实性审计",
            "",
            "| Method | Accept | Mean root Δ | Improve | Harm >5cm | First-post harm | Camera max Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("b0_brtc_lc", *REFINEMENT_VARIANTS):
        runtime = report["variant_runtime_audit"][method]
        harm = report["b0_to_refinement_harm"][method]
        all_post = harm["all_person_frames_in_corrected_post_shots"]
        first_post = harm["first_post_boundary_person_frames"]
        camera = report["camera_exactness"][method]["max_abs_change"]
        lines.append(
            f"| {method} | {runtime['accepted_count']}/{runtime['matched_count']} "
            f"({runtime['acceptance']:.1%}) | {all_post['mean_delta_mm']:.1f} mm | "
            f"{all_post['improve_rate']:.1%} | {all_post['harm_over_5cm_rate']:.1%} | "
            f"{first_post['harm_over_5cm_rate']:.1%} | {camera:.1e} |"
        )
    lines.extend(
        [
            "",
            f"- generic callback harness 重放 current BRTC v1：bit parity="
            f"`{parity['bit_parity']}`，geometry max Δ="
            f"`{max(parity['max_abs_delta'].values()):.3e}`；",
            f"- frozen v1 runtime SHA256 仍为 `{BRTC_V1_RUNTIME_SHA256}`；",
            f"- CPU current-checkpoint 几何回放 cached GPU B0：`{replay['same_forward_replay_verified_1cm']}`，"
            f"labels exact=`{replay['labels_exact']}`，最大 root/joint/vertex 差 "
            f"`{max(replay['max_abs_metric_delta_m'].values()) * 1000.0:.2f} mm`。",
            f"- B0 aligned camera 回放最大矩阵差："
            f"`{replay['b0_aligned_camera_max_abs_delta']:.3e}`；",
            f"- V13 hard-reset raw camera 最大矩阵差："
            f"`{v13_parity['raw_camera_c2w_max_abs_delta']:.3e}`，指标 parity="
            f"`{v13_parity['metric_parity_verified']}`；",
            f"- 匿名边界关联 evaluator-only 正确率："
            f"`{causal['anonymous_association_correct_evaluator_only']}/"
            f"{causal['anonymous_association_total']}` "
            f"(`{causal['anonymous_association_accuracy_evaluator_only']:.1%}`)；",
            f"- BRTC gate 接受：`{causal['brtc_accepted_person_boundaries']}/"
            f"{causal['brtc_matched_person_boundaries']}` "
            f"(`{causal['brtc_acceptance']:.1%}`)；",
            f"- 第二个 cut 的 pre 是否继承第一个 cut 的 person shift："
            f"`{causal['causal_inheritance_observed']}`，非零继承 "
            f"`{causal['second_cut_pre_inherited_shift_nonzero_count']}/"
            f"{causal['second_cut_pre_inherited_shift_person_count']}` 人。",
            "- 所有 refinement 的 camera 均与 B0 bit-exact；unmatched person 最大改动均为 `0`。",
            "",
            "## 6. 与论文的关系",
            "",
            "Multi-THuMBS EgoHumans 参考线为 W/WA/MPJPE/MPVPE = "
            "`279.0/166.0/228.3/262.2 mm`，Accel/ATE/IDs = `27.3/0.7/0.97`。",
            "",
            "| Method | Local W | Gap to 279 | Local WA | Gap to 166 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("b0", "b0_brtc_lc", *REFINEMENT_VARIANTS):
        metric = rows[method]["metrics"]
        lines.append(
            f"| {method} | {metric['w_mpjpe_mm']:.1f} | "
            f"{metric['w_mpjpe_mm'] - PAPER_REFERENCE['w_mpjpe_mm']:+.1f} | "
            f"{metric['wa_mpjpe_mm']:.1f} | "
            f"{metric['wa_mpjpe_mm'] - PAPER_REFERENCE['wa_mpjpe_mm']:+.1f} |"
        )
    lines.extend(
        [
            "",
            "所以按本地 provisional 公式，当前最好候选仍没有达到论文 W/WA 参考线。",
            "本地 pelvis MPJPE/MPVPE 和 ATE 虽数值更小，也不能宣称胜出：论文未发布",
            "supplementary/evaluator/split，且本地 pose 只统计成功匹配帧，ATE 采用短链 Sim(3)。",
            "当前只可证明同 forward 内部增益；正式胜负必须等官方 manifest/公式后重跑。",
            "",
            "## 7. 缓存结论",
            "",
            "V13 compact cache 字段完整；由于 archive/checkpoint 标签不同，本轮先不假设可混用。",
            "实际 parity 审计证明它与 CPU current-checkpoint hard-reset raw 的 camera/aggregate",
            "指标一致。原 B0 JSON 只保存边界和标量误差，缺 joints/vertices；因此本轮仍用",
            "current checkpoint CPU 重建 45 帧紧凑几何，确保 B0/BRTC 的来源无歧义。",
            "",
            "## 8. 产物",
            "",
            "```text",
            "versions/v14/b0_person_triangulation_completeness_weighted.py",
            "versions/v14/tests/test_b0_person_triangulation_completeness_weighted.py",
            "versions/v14/eval_brtc_multithumbs_egohumans.py",
            "versions/v14/docs/V14_BRTC_MULTITHUMBS_EGOHUMANS_20260801.md",
            "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    rng = np.random.default_rng(20260801)
    points = rng.normal(size=(24, 3))
    transform = np.eye(4)
    angle = 0.3
    transform[:3, :3] = np.asarray(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    transform[:3, 3] = [0.2, -0.1, 0.4]
    assert pa_error(transform_points(transform, points), points) < 1e-10
    assert refine_matched_people.__module__ == "versions.v14.b0_person_triangulation"
    frames = [
        {"people": [{"native_track_id": 3}, {"native_track_id": 8}]},
        {"people": [{"native_track_id": 8}, {"native_track_id": 3}]},
    ]
    next_track = assign_segment_tracks(frames, {3: 10}, 11)
    assert next_track == 12
    assert [person["global_track_id"] for person in frames[1]["people"]] == [11, 10]


def checkpoint_provenance(cache: dict[str, Any], requested: Path) -> dict[str, Any]:
    built_from = Path(str(cache["protocol"].get("checkpoint", "")))
    requested = requested.resolve()
    active = DEFAULT_CHECKPOINT.resolve()
    if requested == active and built_from == BIT_EXACT_BUILD_ALIAS:
        return {
            "active_frozen_path": str(active),
            "sha256": ACTIVE_CHECKPOINT_SHA256,
            "geometry_cache_built_from": str(built_from),
            "bit_exact_alias": True,
            "verification": "cmp -s returned 0 on 2026-08-01",
        }
    if built_from.resolve() != requested:
        raise ValueError(
            f"Geometry cache checkpoint {built_from} does not match {requested}"
        )
    return {
        "active_frozen_path": str(requested),
        "sha256": ACTIVE_CHECKPOINT_SHA256 if requested == active else None,
        "geometry_cache_built_from": str(built_from),
        "bit_exact_alias": bool(requested == active),
        "verification": "cache path exactly matches requested checkpoint",
    }


def main() -> None:
    args = parse_args()
    run_self_test()
    if args.self_test:
        print(">> self-test passed")
        return
    if str(args.device) != "cpu":
        raise ValueError("Only --device cpu is permitted for this evaluator")
    for path in (args.output_dir, args.doc.parent):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All outputs must stay inside the Movie3R /data workspace")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.geometry_cache or args.output_dir / "current_v14_cpu_geometry.pt"
    if not str(geometry_path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Geometry cache must stay inside the Movie3R /data workspace")
    if args.overwrite_cache or (args.build_cache and not geometry_path.is_file()):
        cache = build_geometry_cache(args, geometry_path)
    elif geometry_path.is_file():
        cache = torch.load(geometry_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(
            f"Missing {geometry_path}; run once with --build_cache --device cpu"
        )
    provenance = checkpoint_provenance(cache, args.model_path)

    methods, boundary_debug = method_chains(cache)
    registry = refinement_variant_registry()
    replayed_current, replayed_current_debug = replay_refinement_variant(
        methods["b0"],
        boundary_debug,
        "b0_brtc_lc_replay",
        registry["b0_brtc_lc_replay"]["callback"],
        registry["b0_brtc_lc_replay"]["runtime_module"],
    )
    current_replay_parity = geometry_parity_audit(
        methods["b0_brtc_lc"], replayed_current
    )
    if not current_replay_parity["tolerance_parity_1e_12"]:
        raise RuntimeError("Generic variant harness does not replay current BRTC v1")
    variant_debug: dict[str, list[dict[str, Any]]] = {
        "b0_brtc_lc": replayed_current_debug
    }
    for method in REFINEMENT_VARIANTS:
        spec = registry[method]
        methods[method], variant_debug[method] = replay_refinement_variant(
            methods["b0"],
            boundary_debug,
            method,
            spec["callback"],
            spec["runtime_module"],
        )
    _, exo = load_colmap(args.data_root)
    vertex_map, joint_regressor = load_smpl_resources()
    method_reports, method_arrays, method_roots = {}, {}, {}
    for method in METHODS:
        per_chain, arrays, roots = [], [], []
        for chain in methods[method]:
            result, raw_arrays, root_errors = evaluate_chain(
                chain, args.data_root, exo, vertex_map, joint_regressor, float(args.fps)
            )
            per_chain.append(result)
            arrays.append(raw_arrays)
            roots.append(root_errors)
        method_reports[method] = aggregate_method(per_chain, arrays)
        method_arrays[method] = arrays
        method_roots[method] = roots
    del method_arrays
    camera_exactness = {
        method: camera_exactness_audit(methods["b0"], methods[method])
        for method in ("b0_brtc_lc", *REFINEMENT_VARIANTS)
    }
    harms = {
        method: harm_audit(method_roots["b0"], method_roots[method])
        for method in ("b0_brtc_lc", *REFINEMENT_VARIANTS)
    }
    runtime_audits = {
        method: refinement_runtime_audit(rows)
        for method, rows in variant_debug.items()
    }
    promotion = promotion_audit(method_reports, harms, camera_exactness)
    replay = b0_replay_audit(
        cache, methods["b0"], args.data_root, exo, vertex_map, joint_regressor
    )
    v13_parity = v13_raw_parity_audit(cache, method_reports["raw_reset"])
    causal = boundary_causal_audit(boundary_debug)
    report = {
        "title": "Same-forward frozen B0+BRTC variants EgoHumans provisional evaluation",
        "execution": {
            "device": "cpu",
            "gpu_used": False,
            "da3_used": False,
            "geometry_cache": geometry_path,
            "checkpoint": provenance["active_frozen_path"],
            "checkpoint_sha256": provenance["sha256"],
            "checkpoint_provenance": provenance,
        },
        "protocol": {
            "scope": "three self-built 15-frame chains from EgoHumans 001_legoassemble",
            "same_forward_all_methods": True,
            "same_timestamp_repeated_at_two_boundaries": True,
            "w_wa": "local GVHMR-style Sim(3), initial two observed frames / complete track",
            "pose": "both pelvis-centered and per-frame Procrustes PA on SMPL 24/6890",
            "accel": "pelvis-centered camera joints; delta2 and fps^2 units both reported",
            "ate": "per-chain Sim(3)-aligned camera-center RMSE",
            "ids": "native reset IDs for raw; anonymous B0 Hungarian-linked global IDs for B0/BRTC",
            "gt_use": "2D association and evaluation only",
            "official_multi_thumbs_protocol": False,
        },
        "cache_audit": cache_audit(),
        "b0_boundary_provenance": {
            "case_report_checkpoint": str(BIT_EXACT_BUILD_ALIAS),
            "active_frozen_checkpoint": provenance["active_frozen_path"],
            "bit_exact_alias": provenance["bit_exact_alias"],
            "sha256": provenance["sha256"],
        },
        "same_forward_replay_audit": replay,
        "v13_raw_parity_audit": v13_parity,
        "current_brtc_generic_replay_parity": current_replay_parity,
        "boundary_causal_audit": causal,
        "paper_reference_only": PAPER_REFERENCE,
        "methods": method_reports,
        "variant_registry": {
            name: {key: value for key, value in spec.items() if key != "callback"}
            for name, spec in registry.items()
        },
        "variant_runtime_audit": runtime_audits,
        "b0_to_refinement_harm": harms,
        "b0_to_brtc_harm": harms["b0_brtc_lc"],
        "camera_exactness": camera_exactness,
        "brtc_camera_max_abs_change": camera_exactness["b0_brtc_lc"][
            "max_abs_change"
        ],
        "promotion_audit": promotion,
        "boundary_runtime_debug": boundary_debug,
        "variant_boundary_runtime_debug": variant_debug,
        "limitations": [
            "This is not the unpublished Multi-THuMBS split or official evaluator.",
            "The two cut sides repeat the same dataset timestamp; Accel treats stream indices as adjacent.",
            "GT association affects metric coverage but not B0/BRTC proposals.",
            "B0 boundaries were frozen earlier; no EgoHumans metric was used to tune BRTC config.",
            "Completeness weighting is a parameter-free count ratio; damped and Huber policies were frozen outside these EgoHumans chains.",
        ],
    }
    text = markdown(report)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    args.doc.write_text(text, encoding="utf-8")
    print(text)
    print(f">> wrote {json_path}")


if __name__ == "__main__":
    main()
