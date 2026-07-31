#!/usr/bin/env python3
"""Controlled token probe for camera-isolated post-cut person residuals.

This probe is deliberately archive-only with respect to Human3R: it reads the
saved camera/SMPL/token outputs and never instantiates or forwards Human3R.  GT
SMPL and camera metadata are reloaded from the manifest records only to build
evaluation labels.

Protocol
--------
* offset0 archive: development/training only;
* offset50 archive: locked held-out evaluation only;
* camera is analytically isolated before defining the person target;
* the person update is one rigid translation along its current camera ray;
* no source/capture/person IDs are model inputs.

The primary anchor is SMPL-X joint 0 (pelvis), matching the current V14 root
evaluator.  A pelvis+hips+shoulders centroid is also recorded as a semantic
sensitivity check, but is not silently substituted for the primary label.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from smplx import create as create_smplx
from smplx.joint_names import JOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.datasets.avatarrex import (  # noqa: E402
    _avatarrex_scene_path,
    _load_avatarrex_raw_calibration,
    _raw_calibration_c2w,
)
from dust3r.smpl_model import SMPLX_DIR  # noqa: E402


DEFAULT_TRAIN_ARCHIVE = (
    REPO_ROOT
    / "output/archive/20260721/v10_geometry_anchor_weight_probe"
    / "medium_s50_human_token_nogate_20260709"
)
DEFAULT_HELDOUT_ARCHIVE = (
    REPO_ROOT
    / "output/archive/20260721/v10_geometry_anchor_weight_probe"
    / "medium_s50_human_token_nogate_heldout_offset50_20260709"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/controlled_token_person_residual"
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data")
DEFAULT_RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
    "lbn2": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn2",
    "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
}

PELVIS = 0
TORSO5 = (0, 1, 2, 16, 17)
BODY25_TO_SMPLX = {
    8: JOINT_NAMES.index("pelvis"),
    12: JOINT_NAMES.index("left_hip"),
    9: JOINT_NAMES.index("right_hip"),
    5: JOINT_NAMES.index("left_shoulder"),
    2: JOINT_NAMES.index("right_shoulder"),
}


@dataclass
class FramePayload:
    pose: np.ndarray
    pred_pose_params: np.ndarray
    pred_shape: np.ndarray
    pred_transl: np.ndarray
    pred_expression: np.ndarray
    gt_c2w: np.ndarray
    gt_annot: dict[str, Any]


@dataclass
class SplitPayload:
    name: str
    records: list[dict[str, Any]]
    features: dict[str, np.ndarray]
    labels: np.ndarray
    torso_labels: np.ndarray
    pred_roots: np.ndarray
    gt_roots: np.ndarray
    pred_torso: np.ndarray
    gt_torso: np.ndarray
    rays: np.ndarray
    root_errors: np.ndarray
    tangential_errors: np.ndarray
    rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_archive", type=Path, default=DEFAULT_TRAIN_ARCHIVE)
    parser.add_argument("--heldout_archive", type=Path, default=DEFAULT_HELDOUT_ARCHIVE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smpl_chunk", type=int, default=32)
    parser.add_argument("--label_only", action="store_true")
    parser.add_argument(
        "--development_only",
        action="store_true",
        help="Run offset0 CV diagnostics only; never load or predict offset50.",
    )
    parser.add_argument("--overwrite_labels", action="store_true")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--pca_dims", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--ridge_alphas", type=float, nargs="+", default=(0.1, 1.0, 10.0, 100.0, 1000.0))
    parser.add_argument("--include_mlp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def sample_dirs(archive: Path, count: int) -> list[Path]:
    by_index: dict[int, Path] = {}
    for path in (archive / "samples").iterdir():
        if not path.is_dir():
            continue
        try:
            index = int(path.name.split("_", 1)[0])
        except ValueError:
            continue
        if index in by_index:
            raise RuntimeError(f"Duplicate sample index {index} in {archive}")
        by_index[index] = path
    missing = [index for index in range(count) if index not in by_index]
    if missing:
        raise FileNotFoundError(f"Missing sample directories in {archive}: {missing[:10]}")
    return [by_index[index] for index in range(count)]


def source_split(source: str) -> str:
    return "Training/mvhuman" if source.startswith("mvhuman") else "Training"


def load_camera_pose(data_root: Path, split: str, seq: str, frame: int) -> np.ndarray:
    path = data_root / split / Path(seq) / "cam" / f"{frame:08d}.npz"
    with np.load(path) as data:
        return data["pose"].astype(np.float64)


def load_gt_annot(data_root: Path, split: str, seq: str, frame: int) -> dict[str, Any]:
    path = data_root / split / Path(seq) / "smpl" / f"{frame:08d}.pkl"
    with path.open("rb") as handle:
        humans = pickle.load(handle)
    if len(humans) != 1:
        raise ValueError(f"Controlled max_humans=1 label requires one GT person, got {len(humans)}: {path}")
    return dict(humans[0])


def gt_camera_pose(
    source: str,
    seq: str,
    frame: int,
    split: str,
    data_root: Path,
    raw_calibration: Any,
) -> np.ndarray:
    raw = None
    if source == "avatarrex":
        raw = _raw_calibration_c2w(raw_calibration, seq)
    if raw is not None:
        return np.asarray(raw, dtype=np.float64)
    return load_camera_pose(data_root, split, seq, frame)


def load_predicted_frame(local_dir: Path, frame: int) -> tuple[np.ndarray, ...]:
    with np.load(local_dir / "camera" / f"{frame:06d}.npz") as camera:
        pose = camera["pose"].astype(np.float64)
    with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
        if smpl["shape"].shape[0] != 1:
            raise ValueError(f"Expected exactly one predicted person in {local_dir}, frame={frame}")
        expression = smpl["expression"]
        if expression is None or len(expression) == 0:
            expression = np.zeros((1, 10), dtype=np.float32)
        return (
            pose,
            smpl["rotvec"][0].astype(np.float32),
            smpl["shape"][0].astype(np.float32),
            smpl["transl"][0].astype(np.float32),
            expression[0].astype(np.float32),
        )


def token_pair(frames: np.ndarray, boundary: int = 2) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    hist = frames[:boundary].mean(axis=0)
    cur = frames[boundary]
    return np.concatenate((hist, cur, cur - hist, np.abs(cur - hist))).astype(np.float32)


def load_archive_metadata(
    name: str,
    archive: Path,
    data_root: Path,
    raw_calibration: Any,
) -> tuple[list[dict[str, Any]], list[list[FramePayload]], dict[str, np.ndarray]]:
    run_args = json.loads((archive / "run_args.json").read_text(encoding="utf-8"))
    if int(run_args.get("boundary", -1)) != 2:
        raise ValueError(f"Expected boundary=2 in {archive}, got {run_args.get('boundary')}")
    expected_offset = 0 if name == "offset0" else 50
    if int(run_args.get("source_offset", -1)) != expected_offset:
        raise ValueError(
            f"Split contract violated for {name}: source_offset={run_args.get('source_offset')}"
        )
    records = json.loads((archive / "selected_records.json").read_text(encoding="utf-8"))
    directories = sample_dirs(archive, len(records))
    frames_by_sample: list[list[FramePayload]] = []
    feature_lists: dict[str, list[np.ndarray]] = defaultdict(list)

    for index, (record, sample_dir) in enumerate(zip(records, directories)):
        source = str(record["source"])
        split = source_split(source)
        seq_a = str(record["seqA"])
        seq_b = str(record["seqB"])
        start = int(record["start_frame"])
        local_dir = sample_dir / "original_human3r_local_reset"
        token_path = sample_dir / "token_features.npz"
        with np.load(token_path) as token:
            human_in = token["human_token_in"].astype(np.float32)
            human_out = token["human_token_out"].astype(np.float32)
        feature_lists["human_out_pair"].append(token_pair(human_out))
        feature_lists["human_in_pair"].append(token_pair(human_in))
        feature_lists["human_inout_delta_pair"].append(token_pair(human_out - human_in))

        payloads: list[FramePayload] = []
        for relative in range(4):
            frame = start + relative
            camera_seq = seq_a if relative < 2 else seq_b
            pred = load_predicted_frame(local_dir, relative)
            # AABB uses the motion/SMPL stream from seqA for all four frames;
            # seqB changes only the camera after the boundary.
            annot = load_gt_annot(data_root, split, seq_a, frame)
            c2w = gt_camera_pose(source, camera_seq, frame, split, data_root, raw_calibration)
            payloads.append(
                FramePayload(
                    pose=pred[0],
                    pred_pose_params=pred[1],
                    pred_shape=pred[2],
                    pred_transl=pred[3],
                    pred_expression=pred[4],
                    gt_c2w=c2w,
                    gt_annot=annot,
                )
            )
        frames_by_sample.append(payloads)

        if (index + 1) % 50 == 0:
            print(f"[{name}] metadata {index + 1}/{len(records)}", flush=True)

    return records, frames_by_sample, {
        key: np.stack(value).astype(np.float32) for key, value in feature_lists.items()
    }


def batch_slices(length: int, chunk: int):
    for start in range(0, length, chunk):
        yield slice(start, min(length, start + chunk))


def make_model(num_betas: int, device: torch.device):
    return create_smplx(
        SMPLX_DIR,
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
    ).to(device).eval()


def predicted_joints(
    payloads: list[FramePayload],
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    model = make_model(10, device)
    pose = np.stack([p.pred_pose_params for p in payloads]).astype(np.float32)
    shape = np.stack([p.pred_shape for p in payloads]).astype(np.float32)
    transl = np.stack([p.pred_transl for p in payloads]).astype(np.float32)
    expression = np.stack([p.pred_expression for p in payloads]).astype(np.float32)
    outputs = []
    head_index = JOINT_NAMES.index("head")
    for part in batch_slices(len(payloads), chunk):
        pose_t = torch.from_numpy(pose[part]).to(device)
        shape_t = torch.from_numpy(shape[part]).to(device)
        transl_t = torch.from_numpy(transl[part]).to(device)
        expr_t = torch.from_numpy(expression[part]).to(device)
        count = pose_t.shape[0]
        with torch.no_grad():
            out = model(
                betas=shape_t,
                global_orient=model.global_orient.repeat(count, 1),
                body_pose=pose_t[:, 1:22].flatten(1),
                left_hand_pose=pose_t[:, 22:37].flatten(1),
                right_hand_pose=pose_t[:, 37:52].flatten(1),
                jaw_pose=pose_t[:, 52:53].flatten(1),
                leye_pose=model.leye_pose.repeat(count, 1),
                reye_pose=model.reye_pose.repeat(count, 1),
                expression=expr_t,
            )
            joints = out.joints
            pelvis = joints[:, [0]]
            rotation = torch.from_numpy(pose[part, 0]).to(device)
            import roma

            rotation = roma.rotvec_to_rotmat(rotation)
            joints = (rotation[:, None] @ (joints - pelvis)[..., None]).squeeze(-1)
            joints = joints - joints[:, [head_index]] + transl_t[:, None]
        outputs.append(joints.detach().cpu().numpy().astype(np.float64))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0)


def gt_local_joints(
    payloads: list[FramePayload],
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """Return the five evaluation joints in GT camera coordinates."""
    result = np.zeros((len(payloads), len(TORSO5), 3), dtype=np.float64)
    needs_model = []
    for index, payload in enumerate(payloads):
        annot = payload.gt_annot
        has_precomputed = float(annot.get("smplx_has_precomputed_keypoints", 0.0)) > 0.5
        if has_precomputed and "smplx_body25_world" in annot:
            world = np.zeros((len(TORSO5), 3), dtype=np.float64)
            body25 = np.asarray(annot["smplx_body25_world"], dtype=np.float64)
            inverse_map = {smplx: body25_id for body25_id, smplx in BODY25_TO_SMPLX.items()}
            for output_index, smplx_index in enumerate(TORSO5):
                world[output_index] = body25[inverse_map[smplx_index]]
            w2c = np.linalg.inv(payload.gt_c2w)
            result[index] = world @ w2c[:3, :3].T + w2c[:3, 3]
        else:
            needs_model.append(index)

    if needs_model:
        model = make_model(11, device)
        for indices_part in batch_slices(len(needs_model), chunk):
            indices = needs_model[indices_part]
            annots = [payloads[index].gt_annot for index in indices]
            def stack(key: str) -> torch.Tensor:
                return torch.from_numpy(np.stack([np.asarray(a[key], dtype=np.float32) for a in annots])).to(device)

            root = stack("smplx_root_pose").reshape(-1, 3)
            body = stack("smplx_body_pose").reshape(-1, 21 * 3)
            left = stack("smplx_left_hand_pose").reshape(-1, 15 * 3)
            right = stack("smplx_right_hand_pose").reshape(-1, 15 * 3)
            jaw = stack("smplx_jaw_pose").reshape(-1, 3)
            leye = stack("smplx_leye_pose").reshape(-1, 3)
            reye = stack("smplx_reye_pose").reshape(-1, 3)
            shape = stack("smplx_shape").reshape(-1, 11)
            transl = stack("smplx_transl").reshape(-1, 3)
            count = len(indices)
            with torch.no_grad():
                out = model(
                    global_orient=root,
                    body_pose=body,
                    left_hand_pose=left,
                    right_hand_pose=right,
                    jaw_pose=jaw,
                    leye_pose=leye,
                    reye_pose=reye,
                    betas=shape,
                    transl=transl,
                    expression=model.expression.repeat(count, 1),
                )
                world = out.joints[:, list(TORSO5)].detach().cpu().numpy().astype(np.float64)
            for local_index, payload_index in enumerate(indices):
                scale = float(annots[local_index].get("smplx_world_scale", 1.0))
                world_one = world[local_index] * scale
                w2c = np.linalg.inv(payloads[payload_index].gt_c2w)
                result[payload_index] = world_one @ w2c[:3, :3].T + w2c[:3, 3]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_split_payload(
    name: str,
    archive: Path,
    data_root: Path,
    raw_calibration: Any,
    device: torch.device,
    chunk: int,
) -> SplitPayload:
    records, frames_by_sample, features = load_archive_metadata(
        name, archive, data_root, raw_calibration
    )
    flat = [frame for sample in frames_by_sample for frame in sample]
    pred_all = predicted_joints(flat, device, chunk)
    gt_all = gt_local_joints(flat, device, chunk)
    pred_all = pred_all.reshape(len(records), 4, pred_all.shape[1], 3)
    gt_all = gt_all.reshape(len(records), 4, len(TORSO5), 3)

    pred_roots = pred_all[:, 2, PELVIS]
    gt_roots = gt_all[:, 2, 0]
    pred_torso = pred_all[:, 2, list(TORSO5)].mean(axis=1)
    gt_torso = gt_all[:, 2].mean(axis=1)
    ray_norm = np.linalg.norm(pred_roots, axis=1, keepdims=True)
    if np.any(ray_norm < 1e-6):
        raise ValueError(f"Degenerate predicted root ray in {name}")
    rays = pred_roots / ray_norm
    residual = gt_roots - pred_roots
    labels = np.einsum("ni,ni->n", residual, rays)
    tangential = np.linalg.norm(residual - labels[:, None] * rays, axis=1)
    root_errors = np.linalg.norm(residual, axis=1)
    torso_rays = pred_torso / np.linalg.norm(pred_torso, axis=1, keepdims=True)
    torso_labels = np.einsum("ni,ni->n", gt_torso - pred_torso, torso_rays)

    rows = []
    max_isolation_delta = 0.0
    for index, record in enumerate(records):
        target_c2w = frames_by_sample[index][2].gt_c2w
        target_R = target_c2w[:3, :3]
        target_t = target_c2w[:3, 3]
        pred_world = target_R @ pred_roots[index] + target_t
        gt_world = target_R @ gt_roots[index] + target_t
        ray_world = target_R @ rays[index]
        world_label = float(np.dot(gt_world - pred_world, ray_world))
        isolation_delta = abs(world_label - float(labels[index]))
        max_isolation_delta = max(max_isolation_delta, isolation_delta)
        rows.append(
            {
                "split": name,
                "index": index,
                "source": record["source"],
                "group": record.get("group", ""),
                "pattern_id": record.get("pattern_id", ""),
                "source_local_index": int(record.get("source_local_index", -1)),
                "root_ray_label_m": float(labels[index]),
                "torso5_ray_label_m": float(torso_labels[index]),
                "root_error_exact_camera_m": float(root_errors[index]),
                "root_tangential_error_m": float(tangential[index]),
                "camera_isolation_abs_delta_m": float(isolation_delta),
                "pred_root_local": pred_roots[index].tolist(),
                "gt_root_local": gt_roots[index].tolist(),
                "root_ray_local": rays[index].tolist(),
            }
        )
    # GT camera matrices are stored as float32 and can be a few ppm away from
    # perfectly orthonormal.  The local/world constructions must agree to
    # numerical camera-file precision, not float64 machine epsilon.
    if max_isolation_delta > 1e-5:
        raise AssertionError(f"Camera-isolation invariance failed in {name}: {max_isolation_delta}")

    return SplitPayload(
        name=name,
        records=records,
        features=features,
        labels=labels,
        torso_labels=torso_labels,
        pred_roots=pred_roots,
        gt_roots=gt_roots,
        pred_torso=pred_torso,
        gt_torso=gt_torso,
        rays=rays,
        root_errors=root_errors,
        tangential_errors=tangential,
        rows=rows,
    )


def split_audit(payload: SplitPayload) -> dict[str, Any]:
    sources = Counter(str(record["source"]) for record in payload.records)
    groups = Counter(f"{record['source']}::{record.get('group', '')}" for record in payload.records)
    source_array = np.asarray([str(record["source"]) for record in payload.records])
    per_source = {}
    for source in sorted(sources):
        indices = np.flatnonzero(source_array == source)
        per_source[source] = {
            "count": int(len(indices)),
            "root_ray_label_m": distribution(payload.labels[indices]),
            "root_ray_positive_rate": float(np.mean(payload.labels[indices] > 0.0)),
            "exact_camera_root_error_m": distribution(payload.root_errors[indices]),
            "tangential_error_m": distribution(payload.tangential_errors[indices]),
        }
    return {
        "count": len(payload.records),
        "source_counts": dict(sorted(sources.items())),
        "group_counts": dict(sorted(groups.items())),
        "root_ray_label_m": distribution(payload.labels),
        "root_ray_positive_rate": float(np.mean(payload.labels > 0.0)),
        "exact_camera_root_error_m": distribution(payload.root_errors),
        "irreducible_tangential_error_m": distribution(payload.tangential_errors),
        "torso5_ray_label_m": distribution(payload.torso_labels),
        "pelvis_vs_torso5_label_pearson": pearson(payload.labels, payload.torso_labels),
        "pelvis_vs_torso5_sign_agreement": float(
            np.mean(np.sign(payload.labels) == np.sign(payload.torso_labels))
        ),
        "camera_isolation_max_abs_delta_m": float(
            max(row["camera_isolation_abs_delta_m"] for row in payload.rows)
        ),
        "feature_shapes": {key: list(value.shape) for key, value in payload.features.items()},
        "per_source": per_source,
    }


def save_label_cache(output_dir: Path, train: SplitPayload, heldout: SplitPayload) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = train.rows + heldout.rows
    json_path = output_dir / "controlled_person_labels.json"
    json_path.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "controlled_person_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        scalar_keys = [key for key, value in all_rows[0].items() if not isinstance(value, list)]
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row[key] for key in scalar_keys})
    audit = {
        "label_contract": {
            "boundary": 2,
            "primary_anchor": "SMPL-X joint 0 pelvis",
            "ray": "normalize(predicted pelvis in predicted camera coordinates)",
            "target": "dot(GT pelvis local - predicted pelvis local, ray)",
            "equivalent_world_construction": (
                "apply exact target-camera gauge to the predicted local pelvis, then project "
                "the target-minus-predicted anchor residual onto that transformed ray"
            ),
            "person_update": "rigid translation of this person only along its current root ray",
            "camera_update": "none",
            "human3r_execution": "none; archive read only",
            "torso5_role": "semantic sensitivity audit only",
        },
        "offset0": split_audit(train),
        "offset50": split_audit(heldout),
    }
    (output_dir / "label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def groups_for(payload: SplitPayload) -> np.ndarray:
    return np.asarray(
        [f"{r['source']}::{r.get('group', '')}" for r in payload.records], dtype=object
    )


def corrected_root_metrics(payload: SplitPayload, prediction: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    corrected = payload.pred_roots + prediction[:, None] * payload.rays
    error = np.linalg.norm(payload.gt_roots - corrected, axis=1)
    delta = error - payload.root_errors
    return {
        "root_error_m": distribution(error),
        "mean_delta_m": float(delta.mean()),
        "relative_gain": float((payload.root_errors.mean() - error.mean()) / payload.root_errors.mean()),
        "improve_rate": float(np.mean(delta < -1e-12)),
        "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        "correlation": pearson(prediction, payload.labels),
        "sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(payload.labels))),
        "prediction_m": distribution(prediction),
    }


def fit_projection(x_train: np.ndarray, x_eval: np.ndarray, dim: int, seed: int):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)
    actual_dim = min(int(dim), len(x_train) - 1, x_train.shape[1])
    pca = PCA(n_components=actual_dim, svd_solver="randomized", random_state=seed)
    return scaler, pca, pca.fit_transform(x_train_s), pca.transform(x_eval_s)


def cross_validate(train: SplitPayload, args: argparse.Namespace):
    groups = groups_for(train)
    splitter = GroupKFold(n_splits=min(int(args.cv_splits), len(np.unique(groups))))
    folds = list(splitter.split(np.arange(len(groups)), groups=groups))
    configs: list[dict[str, Any]] = []
    oof_by_key: dict[str, np.ndarray] = {}

    for feature_name, x in train.features.items():
        max_dim = max(int(dim) for dim in args.pca_dims)
        max_fold_cache = []
        for fold, (train_idx, val_idx) in enumerate(folds):
            _, _, z_train, z_val = fit_projection(
                x[train_idx], x[val_idx], max_dim, int(args.seed) + fold
            )
            max_fold_cache.append((train_idx, val_idx, z_train, z_val))
        for dim in args.pca_dims:
            fold_cache = [
                (train_idx, val_idx, z_train[:, : int(dim)], z_val[:, : int(dim)])
                for train_idx, val_idx, z_train, z_val in max_fold_cache
            ]
            for alpha in args.ridge_alphas:
                prediction = np.zeros(len(x), dtype=np.float64)
                for train_idx, val_idx, z_train, z_val in fold_cache:
                    model = Ridge(alpha=float(alpha))
                    model.fit(z_train, train.labels[train_idx])
                    prediction[val_idx] = model.predict(z_val)
                key = f"ridge::{feature_name}::pca{int(dim)}::alpha{float(alpha):g}"
                metrics = corrected_root_metrics(train, prediction)
                configs.append(
                    {
                        "key": key,
                        "model_type": "ridge",
                        "feature_name": feature_name,
                        "pca_dim": int(dim),
                        "alpha": float(alpha),
                        "metrics": metrics,
                    }
                )
                oof_by_key[key] = prediction

            if args.include_mlp and int(dim) in (32, 64):
                for width, alpha in ((16, 0.01), (32, 0.1)):
                    prediction = np.zeros(len(x), dtype=np.float64)
                    for fold, (train_idx, val_idx, z_train, z_val) in enumerate(fold_cache):
                        model = MLPRegressor(
                            hidden_layer_sizes=(width,),
                            alpha=alpha,
                            learning_rate_init=1e-3,
                            early_stopping=True,
                            validation_fraction=0.15,
                            max_iter=1000,
                            random_state=int(args.seed) + fold,
                        )
                        model.fit(z_train, train.labels[train_idx])
                        prediction[val_idx] = model.predict(z_val)
                    key = f"mlp::{feature_name}::pca{int(dim)}::w{width}::alpha{alpha:g}"
                    metrics = corrected_root_metrics(train, prediction)
                    configs.append(
                        {
                            "key": key,
                            "model_type": "mlp",
                            "feature_name": feature_name,
                            "pca_dim": int(dim),
                            "width": width,
                            "alpha": alpha,
                            "metrics": metrics,
                        }
                    )
                    oof_by_key[key] = prediction

    constant_oof = np.zeros(len(train.labels), dtype=np.float64)
    for train_idx, val_idx in folds:
        constant_oof[val_idx] = float(np.mean(train.labels[train_idx]))
    constant_metrics = corrected_root_metrics(train, constant_oof)
    noop_metrics = corrected_root_metrics(train, np.zeros(len(train.labels)))
    best = min(configs, key=lambda row: row["metrics"]["root_error_m"]["mean"])
    return configs, best, oof_by_key[best["key"]], constant_oof, constant_metrics, noop_metrics


def select_action_policy(payload: SplitPayload, raw_prediction: np.ndarray) -> dict[str, Any]:
    candidates = []
    for threshold in (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30):
        for cap in (0.05, 0.10, 0.20, 0.30, 0.50):
            accepted = np.abs(raw_prediction) >= threshold
            action = np.where(accepted, np.clip(raw_prediction, -cap, cap), 0.0)
            metrics = corrected_root_metrics(payload, action)
            candidates.append(
                {
                    "threshold_m": threshold,
                    "cap_m": cap,
                    "coverage": float(np.mean(accepted)),
                    "metrics": metrics,
                    "action": action,
                }
            )
    safe = [
        row
        for row in candidates
        if row["coverage"] >= 0.20 and row["metrics"]["harm_over_5cm_rate"] <= 0.10
    ]
    pool = safe if safe else [row for row in candidates if row["coverage"] >= 0.20]
    selected = min(pool, key=lambda row: row["metrics"]["root_error_m"]["mean"])
    return selected


def fit_selected_model(train: SplitPayload, heldout: SplitPayload, config: dict, seed: int):
    feature_name = config["feature_name"]
    scaler, pca, z_train, z_heldout = fit_projection(
        train.features[feature_name], heldout.features[feature_name], config["pca_dim"], seed
    )
    if config["model_type"] == "ridge":
        model = Ridge(alpha=config["alpha"])
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(config["width"],),
            alpha=config["alpha"],
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=1000,
            random_state=seed,
        )
    model.fit(z_train, train.labels)
    return {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "feature_name": feature_name,
        "config": config,
    }, model.predict(z_train), model.predict(z_heldout)


def per_source_metrics(payload: SplitPayload, action: np.ndarray) -> dict[str, Any]:
    output = {}
    sources = np.asarray([record["source"] for record in payload.records])
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        subset = SplitPayload(
            name=payload.name,
            records=[payload.records[i] for i in indices],
            features={},
            labels=payload.labels[indices],
            torso_labels=payload.torso_labels[indices],
            pred_roots=payload.pred_roots[indices],
            gt_roots=payload.gt_roots[indices],
            pred_torso=payload.pred_torso[indices],
            gt_torso=payload.gt_torso[indices],
            rays=payload.rays[indices],
            root_errors=payload.root_errors[indices],
            tangential_errors=payload.tangential_errors[indices],
            rows=[],
        )
        output[source] = corrected_root_metrics(subset, action[indices])
    return output


def write_report(output_dir: Path, audit: dict, results: dict) -> None:
    lines = [
        "# Controlled Human-Token Person Residual Probe",
        "",
        "Human3R was not run. Offset0 was used for development; offset50 was used once as held-out.",
        "Camera is analytically isolated and remains unchanged. The only allowed update is a capped rigid",
        "person translation along the current pelvis ray.",
        "",
        "## Label audit",
        "",
        "| Split | N | Exact-camera root | Tangential floor | Label mean | Label P95 | Pelvis/torso corr | Isolation max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("offset0", "offset50"):
        row = audit[split]
        lines.append(
            f"| {split} | {row['count']} | {row['exact_camera_root_error_m']['mean']:.4f} | "
            f"{row['irreducible_tangential_error_m']['mean']:.4f} | "
            f"{row['root_ray_label_m']['mean']:+.4f} | {row['root_ray_label_m']['p95']:.4f} | "
            f"{row['pelvis_vs_torso5_label_pearson']:.3f} | "
            f"{row['camera_isolation_max_abs_delta_m']:.2e} |"
        )
    heldout = results["heldout"]
    lines += [
        "",
        "The label marginals are strongly source-dependent. This is an absolute camera-local Human3R",
        "depth/root diagnostic, not a source-neutral proof that the model learned a universal cut residual.",
        "Offset50 marginals were inspected before model evaluation for this semantic audit, so the held-out",
        "prediction is policy-locked but not a pristine first reveal. No offset100 token archive exists.",
        "",
        "## Locked result",
        "",
        f"Selected by offset0 group CV: `{results['selected_config']['key']}`.",
        f"Policy: abs(raw) >= {results['policy']['threshold_m']:.3f} m, cap ±{results['policy']['cap_m']:.3f} m.",
        "",
        "| Method | Root mean | Relative gain | Improve | Harm >5cm | Corr | Sign | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("noop", "train_mean", "raw_model", "locked_policy"):
        row = heldout[name]
        coverage = results["heldout_coverage"] if name == "locked_policy" else 1.0
        lines.append(
            f"| {name} | {row['root_error_m']['mean']:.4f} | {row['relative_gain']:+.1%} | "
            f"{row['improve_rate']:.1%} | {row['harm_over_5cm_rate']:.1%} | "
            f"{row['correlation']:.3f} | {row['sign_accuracy']:.1%} | {coverage:.1%} |"
        )
    lines += ["", "## Held-out per source", "", "| Source | Root mean | Gain | Harm >5cm |", "|---|---:|---:|---:|"]
    for source, row in results["heldout_per_source"].items():
        lines.append(
            f"| {source} | {row['root_error_m']['mean']:.4f} | {row['relative_gain']:+.1%} | "
            f"{row['harm_over_5cm_rate']:.1%} |"
        )
    lines += [
        "",
        "The raw token Ridge is also reported diagnostically per source in `results.json`. The deployed",
        "locked policy remains the pre-held-out threshold/cap above.",
    ]
    if results.get("postfreeze_mlp_diagnostic"):
        mlp = results["postfreeze_mlp_diagnostic"]
        lines += [
            "",
            "## Post-freeze offset0-only MLP diagnostic",
            "",
            f"Best small MLP: `{mlp['key']}`, root mean {mlp['metrics']['root_error_m']['mean']:.4f} m, "
            f"relative gain {mlp['metrics']['relative_gain']:+.1%}. It did not replace the frozen Ridge.",
        ]
    lines += [
        "",
        "## Decision",
        "",
        results["decision"],
        "",
        "The torso-5 label is an audit only. The trained target and all reported correction metrics use",
        "joint-0 pelvis, so the experiment does not change root semantics after seeing the result.",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay inside the Movie3R workspace under /data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    raw_calibration = _load_avatarrex_raw_calibration(DEFAULT_RAW_ROOTS)

    print("Loading offset0 labels without Human3R...", flush=True)
    train = build_split_payload(
        "offset0", args.train_archive, args.data_root, raw_calibration, device, args.smpl_chunk
    )
    if args.development_only:
        configs, best, best_oof, constant_oof, constant_cv, noop_cv = cross_validate(train, args)
        policy = select_action_policy(train, best_oof)
        diagnostic = {
            "status": "OFFSET0_POST_FREEZE_DIAGNOSTIC_ONLY",
            "heldout_loaded": False,
            "policy_change_authorized": False,
            "include_mlp": bool(args.include_mlp),
            "best": best,
            "best_policy": {key: value for key, value in policy.items() if key != "action"},
            "constant": constant_cv,
            "noop": noop_cv,
            "configs": configs,
        }
        (args.output_dir / "offset0_development_diagnostic.json").write_text(
            json.dumps(json_ready(diagnostic), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(json_ready({key: diagnostic[key] for key in diagnostic if key != "configs"}), indent=2), flush=True)
        return
    if args.label_only:
        print("Loading offset50 labels for semantic audit only...", flush=True)
        heldout = build_split_payload(
            "offset50", args.heldout_archive, args.data_root, raw_calibration, device, args.smpl_chunk
        )
        audit = save_label_cache(args.output_dir, train, heldout)
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
        return

    configs, best, best_oof, constant_oof, constant_cv, noop_cv = cross_validate(train, args)
    policy = select_action_policy(train, best_oof)
    frozen = {
        "status": "FROZEN_BEFORE_OFFSET50_PREDICTION",
        "development_split": "offset0 only",
        "grouping": "source::group GroupKFold",
        "selected_config": best,
        "selected_policy": {key: value for key, value in policy.items() if key != "action"},
        "cv_noop": noop_cv,
        "cv_train_fold_mean": constant_cv,
        "cv_selected_raw": corrected_root_metrics(train, best_oof),
        "train_label_mean_m": float(np.mean(train.labels)),
        "heldout_disclosure": (
            "offset50 label marginals were previously inspected for label-semantic/domain-bias "
            "audit, but were not used in any model/config/policy selection; this weakens a "
            "pristine one-shot claim"
        ),
        "fully_pristine_confirm_split": "none in the archived token set; no offset100 token archive found",
    }
    frozen_path = args.output_dir / "FROZEN_POLICY_BEFORE_HELDOUT.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        same_lock = (
            existing.get("selected_config", {}).get("key") == best.get("key")
            and existing.get("selected_policy", {}).get("threshold_m") == policy.get("threshold_m")
            and existing.get("selected_policy", {}).get("cap_m") == policy.get("cap_m")
        )
        if not same_lock:
            raise RuntimeError(
                f"Refusing to overwrite a different frozen held-out policy: {frozen_path}"
            )
    else:
        frozen_path.write_text(
            json.dumps(json_ready(frozen), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"Frozen offset0 policy written before held-out prediction: {frozen_path}", flush=True)

    print("Loading offset50 labels after policy freeze...", flush=True)
    heldout = build_split_payload(
        "offset50", args.heldout_archive, args.data_root, raw_calibration, device, args.smpl_chunk
    )
    audit = save_label_cache(args.output_dir, train, heldout)
    artifact, train_raw, heldout_raw = fit_selected_model(train, heldout, best, int(args.seed))
    heldout_mean = np.full(len(heldout.labels), float(np.mean(train.labels)))
    accepted = np.abs(heldout_raw) >= float(policy["threshold_m"])
    heldout_action = np.where(
        accepted,
        np.clip(heldout_raw, -float(policy["cap_m"]), float(policy["cap_m"])),
        0.0,
    )
    heldout_metrics = {
        "noop": corrected_root_metrics(heldout, np.zeros(len(heldout.labels))),
        "train_mean": corrected_root_metrics(heldout, heldout_mean),
        "raw_model": corrected_root_metrics(heldout, heldout_raw),
        "locked_policy": corrected_root_metrics(heldout, heldout_action),
    }
    locked = heldout_metrics["locked_policy"]
    feasible = (
        locked["relative_gain"] >= 0.03
        and locked["harm_over_5cm_rate"] <= 0.10
        and locked["correlation"] >= 0.20
        and np.mean(accepted) >= 0.20
    )
    decision = (
        "PASS: controlled Human3R token evidence clears the provisional held-out person-residual gate."
        if feasible
        else "NO-GO: this controlled token head does not clear the provisional held-out person-residual gate; do not deploy it as the fine-alignment main line."
    )
    results = {
        "protocol": {
            "train": "offset0 only",
            "model_selection": "offset0 source::group GroupKFold",
            "heldout": "offset50 exactly once after freezing model and action policy",
            "features": "token values only; no source/capture/person IDs",
        },
        "selected_config": best,
        "cv_noop": noop_cv,
        "cv_train_fold_mean": constant_cv,
        "cv_selected_raw": corrected_root_metrics(train, best_oof),
        "policy": {key: value for key, value in policy.items() if key != "action"},
        "heldout": heldout_metrics,
        "heldout_coverage": float(np.mean(accepted)),
        "heldout_per_source": per_source_metrics(heldout, heldout_action),
        "heldout_per_source_raw_model": per_source_metrics(heldout, heldout_raw),
        "decision": decision,
        "provisional_gate": {
            "relative_gain_min": 0.03,
            "harm_over_5cm_max": 0.10,
            "correlation_min": 0.20,
            "coverage_min": 0.20,
        },
        "heldout_disclosure": frozen["heldout_disclosure"],
        "fully_pristine_confirm_split": frozen["fully_pristine_confirm_split"],
    }
    diagnostic_path = args.output_dir / "offset0_development_diagnostic.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        mlp_rows = [row for row in diagnostic.get("configs", []) if row.get("model_type") == "mlp"]
        if mlp_rows:
            results["postfreeze_mlp_diagnostic"] = min(
                mlp_rows, key=lambda row: row["metrics"]["root_error_m"]["mean"]
            )
    (args.output_dir / "cv_all_configs.json").write_text(
        json.dumps(json_ready(configs), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(json_ready(results), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact["label_contract"] = audit["label_contract"]
    artifact["policy"] = {key: value for key, value in policy.items() if key != "action"}
    joblib.dump(artifact, args.output_dir / "controlled_token_person_residual.joblib")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        train_label=train.labels,
        train_oof_raw=best_oof,
        train_fit_raw=train_raw,
        heldout_label=heldout.labels,
        heldout_raw=heldout_raw,
        heldout_action=heldout_action,
        heldout_accepted=accepted,
    )
    write_report(args.output_dir, audit, results)
    print(json.dumps(json_ready(results), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
