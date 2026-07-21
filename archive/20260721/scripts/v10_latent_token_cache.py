#!/usr/bin/env python3
"""Cache compact layer-wise Human3R token summaries for latent probes.

Human3R is frozen.  The cache keeps pooled features for all four AABB frames
and a small fixed patch sample around the boundary.  It deliberately does not
save complete state/image tensors for every layer and case.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from dust3r.utils.image import pad_image  # noqa: E402
from v10_oracle_candidate_selection_probe import case_name  # noqa: E402
from v10_token_alignment_4source_probe import load_aabb_views_for_record  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from v9_online_stream_human3r_segment_align import strict_original_model  # noqa: E402


DEFAULT_RECORDS = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "selected_records.jsonl"
)
DEFAULT_CANDIDATE_REPORT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_latent_token_probe" / "token_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--candidate_report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:6" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sources", nargs="*", default=("avatarrex", "thuman", "mvhuman100", "mvhuman200"))
    parser.add_argument("--cases_per_source", type=int, default=0)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--patch_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_records(args: argparse.Namespace) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(args.records):
        source = str(record.get("source", ""))
        if source in args.sources:
            grouped[source].append(record)
    selected = []
    for source in args.sources:
        rows = grouped[source]
        if int(args.cases_per_source) > 0:
            rows = rows[: int(args.cases_per_source)]
        selected.extend(rows)
    return selected


def build_model(args: argparse.Namespace) -> ARCroco3DStereo:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    strict_original_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pool_token(token: torch.Tensor | None) -> np.ndarray:
    if token is None:
        return np.empty((0,), dtype=np.float16)
    value = token.detach().float()
    pooled = torch.cat([value.mean(dim=1), value.std(dim=1, unbiased=False)], dim=-1)
    return pooled[0].cpu().numpy().astype(np.float16)


def flat_token(token: torch.Tensor | None) -> np.ndarray:
    if token is None:
        return np.empty((0,), dtype=np.float16)
    return token.detach().float().reshape(token.shape[0], -1)[0].cpu().numpy().astype(np.float16)


def block_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(type(output))


def fixed_patch_ids(count: int, samples: int, seed: int) -> np.ndarray:
    take = min(int(samples), int(count))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=take, replace=False)).astype(np.int64)


class LayerwiseCollector(AbstractContextManager):
    def __init__(self, model: ARCroco3DStereo, boundary: int, patch_samples: int, seed: int):
        self.model = model
        self.boundary = int(boundary)
        self.patch_frames = {self.boundary - 1, self.boundary}
        self.patch_samples = int(patch_samples)
        self.seed = int(seed)
        self.encode_index = -1
        self.rollout_index = -1
        self.active_rollout = -1
        self.handles = []
        self.original_encode = None
        self.original_rollout = None
        self.frame_data: dict[int, dict] = defaultdict(
            lambda: {
                "encoder_pool": [],
                "decoder_image_pool": [],
                "decoder_state_pool": [],
                "encoder_patch": [],
                "decoder_patch": [],
                "cross_attention": [],
            }
        )

    def _encoder_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            token = block_tensor(output)
            frame = self.encode_index
            self.frame_data[frame]["encoder_pool"].append(pool_token(token))
            if frame in self.patch_frames:
                ids = self.frame_data[frame].get("patch_ids")
                if ids is None:
                    ids = fixed_patch_ids(token.shape[1], self.patch_samples, self.seed + frame)
                    self.frame_data[frame]["patch_ids"] = ids
                self.frame_data[frame]["encoder_patch"].append(
                    token[0, torch.as_tensor(ids, device=token.device)].detach().cpu().to(torch.float16).numpy()
                )
            return output

        return hook

    def _decoder_image_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            token = block_tensor(output)
            frame = self.active_rollout
            n_scene = int(self.frame_data[frame]["n_scene"])
            scene = token[:, 1 : 1 + n_scene]
            self.frame_data[frame]["decoder_image_pool"].append(pool_token(scene))
            if frame in self.patch_frames:
                ids = self.frame_data[frame]["patch_ids"]
                self.frame_data[frame]["decoder_patch"].append(
                    scene[0, torch.as_tensor(ids, device=scene.device)].detach().cpu().to(torch.float16).numpy()
                )
            return output

        return hook

    def _decoder_state_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            token = block_tensor(output)
            self.frame_data[self.active_rollout]["decoder_state_pool"].append(pool_token(token))
            return output

        return hook

    def _backbone_hook(self, _module, _inputs, output):
        token = block_tensor(output)
        frame = self.encode_index
        self.frame_data[frame]["dino_pool"] = pool_token(token)
        if frame in self.patch_frames:
            side = int(round(math.sqrt(token.shape[1])))
            positions = self.frame_data[frame].get("positions")
            ids = self.frame_data[frame].get("patch_ids")
            if positions is not None and ids is not None and side * side == token.shape[1]:
                sampled_pos = positions[ids]
                y_extent = max(float(positions[:, 0].max() + 1), 1.0)
                x_extent = max(float(positions[:, 1].max() + 1), 1.0)
                yy = np.clip(((sampled_pos[:, 0] + 0.5) / y_extent * side).astype(np.int64), 0, side - 1)
                xx = np.clip(((sampled_pos[:, 1] + 0.5) / x_extent * side).astype(np.int64), 0, side - 1)
                dino_ids = yy * side + xx
                self.frame_data[frame]["dino_patch"] = (
                    token[0, torch.as_tensor(dino_ids, device=token.device)].detach().cpu().to(torch.float16).numpy()
                )
        return output

    def __enter__(self):
        self.original_encode = self.model._encode_image
        self.original_rollout = self.model._recurrent_rollout

        def encode_wrapper(_model, *args, **kwargs):
            self.encode_index += 1
            result = self.original_encode(*args, **kwargs)
            frame = self.encode_index
            self.frame_data[frame]["positions"] = result[1][0].detach().cpu().numpy().astype(np.float32)
            self.frame_data[frame]["encoder_final_pool"] = pool_token(result[0][-1])
            return result

        def rollout_wrapper(_model, *args, **kwargs):
            self.rollout_index += 1
            self.active_rollout = self.rollout_index
            frame = self.active_rollout
            self.frame_data[frame]["n_scene"] = int(args[2].shape[1])
            self.frame_data[frame]["n_human"] = int(args[6].shape[1]) if args[6] is not None else 0
            self.frame_data[frame]["camera_initial"] = flat_token(args[4])
            self.frame_data[frame]["human_prompt"] = pool_token(args[6])
            self.frame_data[frame]["persistent_state"] = pool_token(args[0])
            result = self.original_rollout(*args, **kwargs)
            self.frame_data[frame]["new_state"] = pool_token(result[0])
            final = result[1][-1]
            self.frame_data[frame]["camera_refined"] = flat_token(final[:, :1])
            n_human = self.frame_data[frame]["n_human"]
            self.frame_data[frame]["human_refined"] = pool_token(final[:, -n_human:]) if n_human else np.empty((0,), np.float16)
            attention_rows = []
            for attention in result[2]:
                if attention is None:
                    attention_rows.append(np.full(4, np.nan, dtype=np.float16))
                    continue
                value = attention.detach().float()
                attention_rows.append(
                    np.asarray(
                        [
                            float(value.mean()),
                            float(value.std(unbiased=False)),
                            float(value.max()),
                            float(value.min()),
                        ],
                        dtype=np.float16,
                    )
                )
            self.frame_data[frame]["cross_attention"] = attention_rows
            self.active_rollout = -1
            return result

        self.model._encode_image = MethodType(encode_wrapper, self.model)
        self.model._recurrent_rollout = MethodType(rollout_wrapper, self.model)
        for idx, block in enumerate(self.model.enc_blocks):
            self.handles.append(block.register_forward_hook(self._encoder_hook(idx)))
        for idx, block in enumerate(self.model.dec_blocks):
            self.handles.append(block.register_forward_hook(self._decoder_image_hook(idx)))
        for idx, block in enumerate(self.model.dec_blocks_state):
            self.handles.append(block.register_forward_hook(self._decoder_state_hook(idx)))
        self.handles.append(self.model.backbone.register_forward_hook(self._backbone_hook))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.model._encode_image = self.original_encode
        self.model._recurrent_rollout = self.original_rollout
        return False


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def predicted_pose(prediction: dict) -> np.ndarray:
    return tensor_numpy(pose_encoding_to_camera(prediction["camera_pose"].detach().float()))[0]


def pose_targets(poses: np.ndarray) -> dict[str, np.ndarray]:
    first_inv = np.linalg.inv(poses[0])
    relative = np.stack([first_inv @ pose for pose in poses]).astype(np.float32)
    absolute_euler = Rotation.from_matrix(poses[:, :3, :3]).as_euler("zyx", degrees=True).astype(np.float32)
    relative_euler = Rotation.from_matrix(relative[:, :3, :3]).as_euler("zyx", degrees=True).astype(np.float32)
    return {
        "camera_gt_translation": poses[:, :3, 3].astype(np.float32),
        "camera_gt_euler_zyx_deg": absolute_euler,
        "camera_gt_relative_translation": relative[:, :3, 3].astype(np.float32),
        "camera_gt_relative_euler_zyx_deg": relative_euler,
        "camera_gt_distance_from_first": np.linalg.norm(relative[:, :3, 3], axis=1).astype(np.float32),
    }


def human_targets(views: list[dict]) -> dict[str, np.ndarray]:
    roots = []
    headings = []
    valid = []
    for view in views:
        mask = tensor_numpy(view["smpl_mask"])[0].astype(bool)
        body = tensor_numpy(view.get("smplx_body25_world", torch.empty(0)))
        pelvis = tensor_numpy(view.get("smplx_pelvis_world", torch.empty(0)))
        if not mask.any() or body.size == 0 or pelvis.size == 0:
            roots.append(np.full(3, np.nan, dtype=np.float32))
            headings.append(np.full(3, np.nan, dtype=np.float32))
            valid.append(False)
            continue
        body = body[0, 0]
        root = pelvis[0, 0].reshape(-1)[:3]
        left_shoulder, right_shoulder = body[5], body[2]
        left_hip, right_hip = body[12], body[9]
        up = 0.5 * (left_shoulder + right_shoulder) - 0.5 * (left_hip + right_hip)
        right = right_shoulder - left_shoulder
        forward = np.cross(right, up)
        norm = float(np.linalg.norm(forward))
        heading = forward / norm if np.isfinite(norm) and norm > 1e-6 else np.full(3, np.nan)
        roots.append(root.astype(np.float32))
        headings.append(np.asarray(heading, dtype=np.float32))
        valid.append(bool(np.isfinite(root).all() and np.isfinite(heading).all()))
    roots = np.stack(roots)
    headings = np.stack(headings)
    velocity = np.zeros_like(roots)
    angular_velocity = np.zeros(len(roots), dtype=np.float32)
    for idx in range(1, len(roots)):
        velocity[idx] = roots[idx] - roots[idx - 1]
        if np.isfinite(headings[idx]).all() and np.isfinite(headings[idx - 1]).all():
            angular_velocity[idx] = math.degrees(
                math.acos(float(np.clip(np.dot(headings[idx], headings[idx - 1]), -1.0, 1.0)))
            )
        else:
            angular_velocity[idx] = np.nan
    return {
        "human_gt_world_root": roots.astype(np.float32),
        "human_gt_torso_heading": headings.astype(np.float32),
        "human_gt_root_velocity": velocity.astype(np.float32),
        "human_gt_angular_velocity_deg": angular_velocity,
        "human_gt_valid": np.asarray(valid, dtype=np.bool_),
    }


def boundary_targets(gt_poses: np.ndarray, boundary: int) -> dict[str, np.ndarray]:
    transform = np.linalg.inv(gt_poses[boundary - 1]) @ gt_poses[boundary]
    euler = Rotation.from_matrix(transform[:3, :3]).as_euler("zyx", degrees=True).astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    norm = float(np.linalg.norm(translation))
    direction = translation / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)
    return {
        "boundary_transform_prev_to_current": transform.astype(np.float32),
        "boundary_euler_zyx_deg": euler,
        "boundary_translation": translation,
        "boundary_translation_direction": direction.astype(np.float32),
        "boundary_translation_norm": np.asarray(norm, dtype=np.float32),
    }


def sample_patch_targets(
    prediction: dict,
    pose: np.ndarray,
    positions: np.ndarray,
    ids: np.ndarray,
    gt_pose: np.ndarray,
) -> dict[str, np.ndarray]:
    points = tensor_numpy(prediction["pts3d_in_self_view"])[0]
    confidence = tensor_numpy(prediction["conf_self"])[0]
    if confidence.ndim == 3:
        confidence = confidence[..., 0]
    mask_value = prediction.get("msk")
    if mask_value is None:
        human_mask = np.zeros(points.shape[:2], dtype=np.float32)
    else:
        human_mask = tensor_numpy(mask_value)[0]
        human_mask = np.squeeze(human_mask)
    sampled_pos = positions[ids]
    patch_size = 16.0
    yy = np.clip(np.round((sampled_pos[:, 0] + 0.5) * patch_size).astype(np.int64), 0, points.shape[0] - 1)
    xx = np.clip(np.round((sampled_pos[:, 1] + 0.5) * patch_size).astype(np.int64), 0, points.shape[1] - 1)
    camera_points = points[yy, xx].astype(np.float32)
    world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    normals = []
    for y, x in zip(yy, xx):
        y0, y1 = max(0, y - 1), min(points.shape[0] - 1, y + 1)
        x0, x1 = max(0, x - 1), min(points.shape[1] - 1, x + 1)
        dx = points[y, x1] - points[y, x0]
        dy = points[y1, x] - points[y0, x]
        normal = np.cross(dx, dy)
        normal_norm = float(np.linalg.norm(normal))
        normals.append(normal / normal_norm if np.isfinite(normal_norm) and normal_norm > 1e-8 else np.full(3, np.nan))
    normals = np.asarray(normals, dtype=np.float32)
    up_camera = gt_pose[:3, :3].T @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    gravity_alignment = np.abs(normals @ up_camera).astype(np.float32)
    scene_class = np.full(len(ids), 2, dtype=np.int64)
    scene_class[gravity_alignment > 0.75] = 0
    scene_class[gravity_alignment < 0.25] = 1
    conf_values = confidence[yy, xx].astype(np.float32)
    finite = np.isfinite(camera_points).all(axis=1) & np.isfinite(conf_values)
    conf_floor = float(np.quantile(conf_values[finite], 0.5)) if finite.any() else float("inf")
    static_background = human_mask[yy, xx] < 0.1
    suitable = finite & static_background & (conf_values >= conf_floor)
    return {
        "patch_position_yx": sampled_pos.astype(np.float32),
        "patch_camera_point_pred": camera_points,
        "patch_world_point_pred": world_points.astype(np.float32),
        "patch_depth_pred": camera_points[:, 2].astype(np.float32),
        "patch_normal_pred": normals,
        "patch_scene_class_pseudo": scene_class,
        "patch_static_background": static_background.astype(np.bool_),
        "patch_confidence_pred": conf_values,
        "patch_alignment_suitable_pseudo": suitable.astype(np.bool_),
    }


def stack_frame_value(frame_data: dict[int, dict], key: str, frame_count: int) -> np.ndarray:
    return np.stack([np.asarray(frame_data[idx][key]) for idx in range(frame_count)])


def candidate_failure_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(case["case_name"]): {
            "explicit_translation_error_m": case["fixed_explicit"]["metrics"]["mean_translation_m"],
            "explicit_rotation_error_deg": case["fixed_explicit"]["metrics"]["mean_rotation_deg"],
            "explicit_failure_relaxed": not case["fixed_explicit"]["metrics"]["success_relaxed"],
            "explicit_catastrophic": case["fixed_explicit"]["metrics"]["catastrophic"],
        }
        for case in report["cases"]
    }


def cache_case(
    model: ARCroco3DStereo,
    record: dict,
    args: argparse.Namespace,
    device: torch.device,
    index: int,
    failure_map: dict[str, dict],
) -> dict:
    name = case_name(record)
    case_dir = args.output_dir / "cases" / name
    cache_path = case_dir / "tokens_and_targets.npz"
    meta_path = case_dir / "metadata.json"
    if cache_path.is_file() and meta_path.is_file() and not args.overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    views = load_aabb_views_for_record(record, args, device)
    for view in views:
        view["img_mhmr"] = pad_image(view["img"], int(model.mhmr_img_res))
    start = time.perf_counter()
    with LayerwiseCollector(model, args.boundary, args.patch_samples, args.seed + index * 100) as collector:
        with torch.no_grad():
            predictions, _ = model.forward_recurrent_lighter(
                views,
                str(device),
                ret_state=False,
                use_ttt3r=False,
                return_token_debug=False,
            )
    elapsed = time.perf_counter() - start
    frame_count = len(views)
    gt_poses = np.stack([tensor_numpy(gt_pose_from_view(view)) for view in views]).astype(np.float32)
    pred_poses = np.stack([predicted_pose(prediction) for prediction in predictions]).astype(np.float32)
    arrays = {
        "encoder_layer_pool": stack_frame_value(collector.frame_data, "encoder_pool", frame_count),
        "encoder_final_pool": stack_frame_value(collector.frame_data, "encoder_final_pool", frame_count),
        "decoder_image_layer_pool": stack_frame_value(collector.frame_data, "decoder_image_pool", frame_count),
        "decoder_state_layer_pool": stack_frame_value(collector.frame_data, "decoder_state_pool", frame_count),
        "camera_initial": stack_frame_value(collector.frame_data, "camera_initial", frame_count),
        "camera_refined": stack_frame_value(collector.frame_data, "camera_refined", frame_count),
        "human_prompt_pool": stack_frame_value(collector.frame_data, "human_prompt", frame_count),
        "human_refined_pool": stack_frame_value(collector.frame_data, "human_refined", frame_count),
        "persistent_state_pool": stack_frame_value(collector.frame_data, "persistent_state", frame_count),
        "new_state_pool": stack_frame_value(collector.frame_data, "new_state", frame_count),
        "dino_pool": stack_frame_value(collector.frame_data, "dino_pool", frame_count),
        "state_image_cross_attention": stack_frame_value(collector.frame_data, "cross_attention", frame_count),
        "camera_pred_pose": pred_poses,
        "camera_gt_pose": gt_poses,
    }
    arrays.update(pose_targets(gt_poses))
    arrays.update(human_targets(views))
    arrays.update(boundary_targets(gt_poses, int(args.boundary)))

    patch_frames = [int(args.boundary) - 1, int(args.boundary)]
    arrays["patch_frame_indices"] = np.asarray(patch_frames, dtype=np.int64)
    arrays["patch_ids"] = np.stack([collector.frame_data[idx]["patch_ids"] for idx in patch_frames])
    arrays["encoder_layer_patch"] = np.stack(
        [np.stack(collector.frame_data[idx]["encoder_patch"]) for idx in patch_frames]
    )
    arrays["decoder_layer_patch"] = np.stack(
        [np.stack(collector.frame_data[idx]["decoder_patch"]) for idx in patch_frames]
    )
    arrays["dino_patch"] = np.stack([collector.frame_data[idx]["dino_patch"] for idx in patch_frames])
    patch_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    for idx in patch_frames:
        targets = sample_patch_targets(
            predictions[idx],
            pred_poses[idx],
            collector.frame_data[idx]["positions"],
            collector.frame_data[idx]["patch_ids"],
            gt_poses[idx],
        )
        for key, value in targets.items():
            patch_targets[key].append(value)
    arrays.update({key: np.stack(value) for key, value in patch_targets.items()})
    np.savez_compressed(cache_path, **arrays)

    explicit = failure_map.get(name, {})
    metadata = {
        "case_name": name,
        "record": record,
        "cache_path": str(cache_path),
        "frame_count": frame_count,
        "boundary": int(args.boundary),
        "encoder_depth": len(model.enc_blocks),
        "decoder_depth": len(model.dec_blocks),
        "state_tokens": int(model.state_size),
        "patch_samples": int(args.patch_samples),
        "elapsed_seconds": elapsed,
        "geometry_supervision": {
            "camera": "raw/official dataset calibration",
            "human": "dataset world SMPL-X/body25 annotations",
            "patch_scene": "Human3R predicted pointmap pseudo-target; not physical GT",
            "physical_patch_correspondence_available": False,
            "reason": "The current 180-case sources do not expose a uniformly verified static-scene GT depth/mesh cache.",
        },
        **explicit,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del views, predictions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(args)
    if not records:
        raise RuntimeError("No records selected")
    failure_map = candidate_failure_map(args.candidate_report)
    device = torch.device(args.device)
    model = build_model(args)
    metadata = []
    failures = []
    for index, record in enumerate(records):
        print(f">> [{index + 1}/{len(records)}] {record['source']} {record['pattern_id']}", flush=True)
        try:
            metadata.append(cache_case(model, record, args, device, index, failure_map))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            failures.append({"record": record, "error": str(exc)})
            print(f"!! skip {record.get('pattern_id')}: {exc}", flush=True)
    index_path = args.output_dir / "cache_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
        encoding="utf-8",
    )
    summary = {
        "requested": len(records),
        "cached": len(metadata),
        "failed": len(failures),
        "failures": failures,
        "total_seconds": float(sum(row["elapsed_seconds"] for row in metadata)),
        "physical_patch_correspondence_available": False,
    }
    (args.output_dir / "cache_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f">> cached {len(metadata)}/{len(records)} cases at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
