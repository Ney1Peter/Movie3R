#!/usr/bin/env python3
"""Causal activation-patching probe for frozen Human3R latent tokens.

The experiment compares a continuous teacher branch with a fresh-state student
branch on the same boundary RGB frame.  It patches latent activations only; no
camera, pointmap, or SMPL-X output is copied between branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from dust3r.datasets.avatarrex import AvatarReX_Video  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.image import pad_image  # noqa: E402
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
from v10_token_alignment_4source_probe import raw_roots_for_record, source_split_and_scope  # noqa: E402
from v9_online_stream_human3r_segment_align import strict_original_model  # noqa: E402


DEFAULT_RECORDS = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "selected_records.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_latent_token_probe" / "activation_patching"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:6" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sources", nargs="*", default=("avatarrex", "thuman", "mvhuman100", "mvhuman200"))
    parser.add_argument("--cases_per_source", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--point_sample", type=int, default=20000)
    parser.add_argument("--decoder_layers", type=int, nargs="*", default=None)
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


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def select_records(args: argparse.Namespace) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(args.records):
        source = str(record.get("source", ""))
        if source in args.sources:
            by_source[source].append(record)
    selected = []
    for source in args.sources:
        selected.extend(by_source[source][: int(args.cases_per_source)])
    if not selected:
        raise RuntimeError(f"No records selected from {args.records}")
    return selected


def configure_views(views: list[dict]) -> list[dict]:
    for view in views:
        view["img_mask"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["ray_mask"] = torch.zeros_like(view["ray_mask"], dtype=torch.bool)
        view["update"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_state"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_mem"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_v8_history"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
    return views


def load_video_views(
    record: dict,
    args: argparse.Namespace,
    device: torch.device,
    mhmr_img_res: int,
) -> list[dict]:
    split, _ = source_split_and_scope(record)
    dataset = AvatarReX_Video(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=int(args.num_views),
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=[(str(record["seqA"]), int(record["start_frame"]))],
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(record),
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    views = configure_views(next(iter(loader)))
    for view in views:
        view["img_mhmr"] = pad_image(view["img"], int(mhmr_img_res))
    return todevice(views, device)


def build_model(args: argparse.Namespace) -> ARCroco3DStereo:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    strict_original_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cpu_token(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float16).clone()


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported block output type: {type(output)}")


def replace_first_tensor(output, value: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return value
    items = list(output)
    items[0] = value
    return tuple(items) if isinstance(output, tuple) else items


@dataclass(frozen=True)
class PatchSpec:
    name: str
    components: tuple[str, ...]
    source_mode: str = "teacher"


class LatentController(AbstractContextManager):
    """Experiment-local hooks for capture or replacement of one frame."""

    def __init__(
        self,
        model: ARCroco3DStereo,
        target_frame: int,
        capture: bool,
        patch: PatchSpec | None = None,
        source: dict | None = None,
        seed: int = 0,
    ):
        self.model = model
        self.target_frame = int(target_frame)
        self.capture_enabled = bool(capture)
        self.patch = patch
        self.source = source or {}
        self.seed = int(seed)
        self.captured: dict = {"encoder_layers": [], "decoder_image_layers": [], "decoder_state_layers": []}
        self.encode_index = -1
        self.rollout_index = -1
        self.active_rollout = -1
        self.n_scene = 0
        self.n_human = 0
        self.handles = []
        self.original_encode = None
        self.original_rollout = None
        self.original_inquire = None
        self.original_update_mem = None
        self.read_pass = False
        self.read_camera_layers: list[torch.Tensor] = []
        self.read_patch_layer: int | None = None
        self.read_patch_value: torch.Tensor | None = None
        self.skipped_replacements: list[str] = []

    @property
    def components(self) -> set[str]:
        return set() if self.patch is None else set(self.patch.components)

    def _source_value(self, key: str, reference: torch.Tensor) -> torch.Tensor | None:
        value = self.source.get(key)
        if value is None:
            self.skipped_replacements.append(f"missing:{key}")
            return None
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.shape != reference.shape:
            self.skipped_replacements.append(f"shape:{key}:{tuple(value.shape)}!={tuple(reference.shape)}")
            return None
        mode = "teacher" if self.patch is None else self.patch.source_mode
        if mode == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + sum(ord(ch) for ch in key))
            noise = torch.randn(value.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(value.device)
            noise = noise * value.float().std(unbiased=False).clamp_min(1e-5) + value.float().mean()
            value = noise.to(device=reference.device, dtype=reference.dtype)
        elif mode == "shuffle" and value.ndim == 3 and value.shape[1] > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + sum(ord(ch) for ch in key))
            order = torch.randperm(value.shape[1], generator=generator).to(value.device)
            value = value[:, order]
        return value

    def _patch_full(self, key: str, reference: torch.Tensor) -> torch.Tensor:
        value = self._source_value(key, reference)
        return reference if value is None else value

    def _encoder_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            token = first_tensor(output)
            if self.encode_index == self.target_frame and self.capture_enabled:
                self.captured["encoder_layers"].append(cpu_token(token))
            component = f"encoder_l{layer_idx}"
            if self.encode_index == self.target_frame and component in self.components:
                patched = self._patch_full(component, token)
                return replace_first_tensor(output, patched)
            return output

        return hook

    def _decoder_image_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            token = first_tensor(output)
            if self.read_pass:
                self.read_camera_layers.append(token[:, :1].detach().clone())
                return output
            if self.active_rollout == self.target_frame and self.capture_enabled:
                self.captured["decoder_image_layers"].append(cpu_token(token))
            if self.active_rollout != self.target_frame:
                return output
            patched = token
            camera_component = f"camera_l{layer_idx}"
            if camera_component in self.components:
                value = self._source_value(camera_component, patched[:, :1])
                if value is not None:
                    patched = patched.clone()
                    patched[:, :1] = value
            if self.read_patch_layer == layer_idx and self.read_patch_value is not None:
                patched = patched.clone()
                patched[:, :1] = self.read_patch_value.to(device=patched.device, dtype=patched.dtype)
            component = f"decoder_image_l{layer_idx}"
            if component in self.components:
                scene = patched[:, 1 : 1 + self.n_scene]
                value = self._source_value(component, scene)
                if value is not None:
                    patched = patched.clone()
                    patched[:, 1 : 1 + self.n_scene] = value
            if layer_idx == len(self.model.dec_blocks) - 1:
                if "camera_refined" in self.components:
                    value = self._source_value("camera_refined", patched[:, :1])
                    if value is not None:
                        patched = patched.clone()
                        patched[:, :1] = value
                if "human_refined" in self.components and self.n_human > 0:
                    value = self._source_value("human_refined", patched[:, -self.n_human :])
                    if value is not None:
                        patched = patched.clone()
                        patched[:, -self.n_human :] = value
                if "decoder_final_full" in self.components:
                    patched = self._patch_full("decoder_final_full", patched)
            return replace_first_tensor(output, patched)

        return hook

    def _decoder_state_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            token = first_tensor(output)
            if self.active_rollout == self.target_frame and self.capture_enabled:
                self.captured["decoder_state_layers"].append(cpu_token(token))
            component = f"decoder_state_l{layer_idx}"
            if self.active_rollout == self.target_frame and component in self.components:
                patched = self._patch_full(component, token)
                return replace_first_tensor(output, patched)
            return output

        return hook

    def __enter__(self):
        self.original_encode = self.model._encode_image
        self.original_rollout = self.model._recurrent_rollout
        self.original_inquire = self.model.pose_retriever.inquire
        self.original_update_mem = self.model.pose_retriever.update_mem

        def encode_wrapper(_model, *args, **kwargs):
            self.encode_index += 1
            result = self.original_encode(*args, **kwargs)
            final = result[0][-1]
            if self.encode_index == self.target_frame and self.capture_enabled:
                self.captured["encoder_final"] = cpu_token(final)
            if self.encode_index == self.target_frame and "encoder_final" in self.components:
                final = self._patch_full("encoder_final", final)
                result = ([final], result[1], result[2])
            return result

        def rollout_wrapper(_model, *args, **kwargs):
            self.rollout_index += 1
            self.active_rollout = self.rollout_index
            args = list(args)
            self.n_scene = int(args[2].shape[1])
            self.n_human = int(args[6].shape[1]) if args[6] is not None else 0
            if self.active_rollout == self.target_frame:
                if self.capture_enabled:
                    self.captured["persistent_state"] = cpu_token(args[0])
                    self.captured["camera_initial"] = cpu_token(args[4])
                    self.captured["human_prompt"] = cpu_token(args[6]) if args[6] is not None else None
                for component, arg_idx in (("persistent_state", 0), ("camera_initial", 4), ("human_prompt", 6)):
                    if component in self.components and args[arg_idx] is not None:
                        args[arg_idx] = self._patch_full(component, args[arg_idx])
                if "post_update_state_input" in self.components:
                    value = self._source_value("new_state", args[0])
                    if value is not None:
                        args[0] = value
                if "read_old_pose_memory" in self.components:
                    source_mem = self.source.get("pose_memory_before")
                    if source_mem is not None:
                        reference_mem = source_mem.to(device=args[0].device, dtype=args[0].dtype)
                        old_mem = self._source_value("pose_memory_before", reference_mem)
                        image_query = self.model._get_img_level_feat(args[2])
                        args[4] = self.original_inquire(image_query, old_mem)
                read_components = [
                    component
                    for component in self.components
                    if component.startswith("read_old_write_fresh_l")
                ]
                if read_components:
                    self.read_patch_layer = int(read_components[0].rsplit("l", 1)[1])
                    old_state = self._source_value("persistent_state", args[0])
                    if old_state is not None:
                        read_args = list(args)
                        read_args[0] = old_state
                        self.read_camera_layers = []
                        self.read_pass = True
                        self.original_rollout(*read_args, **kwargs)
                        self.read_pass = False
                        if self.read_patch_layer < len(self.read_camera_layers):
                            self.read_patch_value = self.read_camera_layers[self.read_patch_layer]
            result = self.original_rollout(*args, **kwargs)
            if self.active_rollout == self.target_frame and "first_write_state" in self.components:
                value = self._source_value("new_state", result[0])
                if value is not None:
                    result = (value, result[1], result[2])
            if self.active_rollout == self.target_frame and self.capture_enabled:
                self.captured["new_state"] = cpu_token(result[0])
                # Decoder hooks observe the pre-norm activation.  Patch values
                # must use the same convention because they are inserted before
                # model.dec_norm on the student branch.
                final = self.captured["decoder_image_layers"][-1]
                self.captured["decoder_final_full"] = final
                self.captured["camera_refined"] = final[:, :1].clone()
                self.captured["human_refined"] = (
                    final[:, -self.n_human :].clone() if self.n_human > 0 else None
                )
                self.captured["n_scene"] = self.n_scene
                self.captured["n_human"] = self.n_human
            self.active_rollout = -1
            self.read_patch_layer = None
            self.read_patch_value = None
            return result

        def inquire_wrapper(_retriever, query, mem):
            output = self.original_inquire(query, mem)
            frame = self.encode_index
            if frame == self.target_frame and self.capture_enabled:
                self.captured["pose_query_image"] = cpu_token(query)
                self.captured["pose_memory_before"] = cpu_token(mem)
                self.captured["pose_query_output"] = cpu_token(output)
            return output

        def update_mem_wrapper(_retriever, mem, feat_k, feat_v):
            output = self.original_update_mem(mem, feat_k, feat_v)
            frame = self.encode_index
            if frame == self.target_frame and self.capture_enabled:
                self.captured["pose_memory_write_input"] = cpu_token(mem)
                self.captured["pose_memory_after"] = cpu_token(output)
            return output

        self.model._encode_image = MethodType(encode_wrapper, self.model)
        self.model._recurrent_rollout = MethodType(rollout_wrapper, self.model)
        self.model.pose_retriever.inquire = MethodType(inquire_wrapper, self.model.pose_retriever)
        self.model.pose_retriever.update_mem = MethodType(update_mem_wrapper, self.model.pose_retriever)
        for idx, block in enumerate(self.model.enc_blocks):
            self.handles.append(block.register_forward_hook(self._encoder_hook(idx)))
        for idx, block in enumerate(self.model.dec_blocks):
            self.handles.append(block.register_forward_hook(self._decoder_image_hook(idx)))
        for idx, block in enumerate(self.model.dec_blocks_state):
            self.handles.append(block.register_forward_hook(self._decoder_state_hook(idx)))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.model._encode_image = self.original_encode
        self.model._recurrent_rollout = self.original_rollout
        self.model.pose_retriever.inquire = self.original_inquire
        self.model.pose_retriever.update_mem = self.original_update_mem
        return False


def source_dict(latents: dict) -> dict[str, torch.Tensor]:
    result = {}
    for key in (
        "encoder_final",
        "persistent_state",
        "camera_initial",
        "human_prompt",
        "camera_refined",
        "human_refined",
        "decoder_final_full",
        "new_state",
        "pose_query_image",
        "pose_memory_before",
        "pose_query_output",
        "pose_memory_write_input",
        "pose_memory_after",
    ):
        if isinstance(latents.get(key), torch.Tensor):
            result[key] = latents[key]
    for idx, value in enumerate(latents.get("encoder_layers", [])):
        result[f"encoder_l{idx}"] = value
    n_scene = int(latents.get("n_scene", 0))
    for idx, value in enumerate(latents.get("decoder_image_layers", [])):
        result[f"decoder_image_l{idx}"] = value[:, 1 : 1 + n_scene]
        result[f"camera_l{idx}"] = value[:, :1]
    for idx, value in enumerate(latents.get("decoder_state_layers", [])):
        result[f"decoder_state_l{idx}"] = value
    return result


def run_branch(
    model: ARCroco3DStereo,
    views: list[dict],
    device: torch.device,
    target_frame: int,
    capture: bool = False,
    patch: PatchSpec | None = None,
    source: dict | None = None,
    seed: int = 0,
) -> tuple[list[dict], dict, float, list[str]]:
    start = time.perf_counter()
    with LatentController(model, target_frame, capture, patch, source, seed) as controller:
        with torch.no_grad():
            predictions, _views = model.forward_recurrent_lighter(
                views,
                str(device),
                ret_state=False,
                use_ttt3r=False,
                return_token_debug=False,
            )
    elapsed = time.perf_counter() - start
    return predictions, controller.captured, elapsed, controller.skipped_replacements


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def camera_matrix(prediction: dict) -> np.ndarray:
    pose = pose_encoding_to_camera(prediction["camera_pose"].detach().float())
    return as_numpy(pose)[0]


def pointmap(prediction: dict, world: bool) -> np.ndarray:
    points = as_numpy(prediction["pts3d_in_self_view"])[0].reshape(-1, 3)
    if world:
        pose = camera_matrix(prediction)
        points = points @ pose[:3, :3].T + pose[:3, 3]
    return points.astype(np.float32)


def first_human(prediction: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    transl = prediction.get("smpl_transl")
    rotmat = prediction.get("smpl_rotmat")
    if transl is None or rotmat is None or transl.shape[1] == 0:
        return None
    pose = camera_matrix(prediction)
    root_cam = as_numpy(transl)[0, 0]
    rotations = as_numpy(rotmat)[0, 0]
    root_world = pose[:3, :3] @ root_cam + pose[:3, 3]
    root_orientation = pose[:3, :3] @ rotations[0]
    return root_world.astype(np.float32), root_orientation.astype(np.float32), rotations[1:].astype(np.float32)


def rotation_batch_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    count = min(len(a), len(b))
    if count == 0:
        return float("nan")
    relative = np.einsum("nij,njk->nik", a[:count], np.swapaxes(b[:count], -1, -2))
    trace = np.trace(relative, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(angle).mean())


def frame_errors(prediction: dict, teacher: dict, args: argparse.Namespace, seed: int) -> dict:
    pose = camera_matrix(prediction)
    target_pose = camera_matrix(teacher)
    cam_points = pointmap(prediction, False)
    target_cam_points = pointmap(teacher, False)
    world_points = pointmap(prediction, True)
    target_world_points = pointmap(teacher, True)
    point_count = min(len(cam_points), len(target_cam_points), len(world_points), len(target_world_points))
    valid = (
        np.isfinite(cam_points[:point_count]).all(axis=1)
        & np.isfinite(target_cam_points[:point_count]).all(axis=1)
        & np.isfinite(world_points[:point_count]).all(axis=1)
        & np.isfinite(target_world_points[:point_count]).all(axis=1)
    )
    valid_ids = np.flatnonzero(valid)
    if len(valid_ids) > int(args.point_sample):
        rng = np.random.default_rng(seed)
        valid_ids = rng.choice(valid_ids, size=int(args.point_sample), replace=False)
    if len(valid_ids) == 0:
        camera_point_error = float("nan")
        world_point_error = float("nan")
    else:
        camera_point_error = float(
            np.linalg.norm(cam_points[valid_ids] - target_cam_points[valid_ids], axis=1).mean()
        )
        world_point_error = float(
            np.linalg.norm(world_points[valid_ids] - target_world_points[valid_ids], axis=1).mean()
        )
    result = {
        "camera_translation_m": float(np.linalg.norm(pose[:3, 3] - target_pose[:3, 3])),
        "camera_rotation_deg": rotation_error_deg(pose, target_pose),
        "pointmap_camera_mean_m": camera_point_error,
        "pointmap_world_mean_m": world_point_error,
    }
    human = first_human(prediction)
    target_human = first_human(teacher)
    if human is not None and target_human is not None:
        result.update(
            {
                "human_world_root_m": float(np.linalg.norm(human[0] - target_human[0])),
                "human_global_orientation_deg": rotation_error_deg(human[1], target_human[1]),
                "human_local_pose_deg": rotation_batch_error_deg(human[2], target_human[2]),
            }
        )
    else:
        result.update(
            {
                "human_world_root_m": float("nan"),
                "human_global_orientation_deg": float("nan"),
                "human_local_pose_deg": float("nan"),
            }
        )
    return result


def evaluate_branch(predictions: list[dict], teacher: list[dict], args: argparse.Namespace, seed: int) -> dict:
    per_frame = [
        {"post_index": idx, **frame_errors(pred, target, args, seed + idx)}
        for idx, (pred, target) in enumerate(zip(predictions, teacher))
    ]
    keys = [key for key in per_frame[0] if key != "post_index"]
    mean = {}
    for key in keys:
        values = np.asarray([row[key] for row in per_frame], dtype=np.float64)
        mean[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return {"mean": mean, "boundary": per_frame[0], "per_frame": per_frame}


def recovery_ratio(reset_value: float, patched_value: float, teacher_value: float = 0.0) -> float:
    denominator = reset_value - teacher_value
    if not np.isfinite(denominator) or abs(denominator) < 1e-8 or not np.isfinite(patched_value):
        return float("nan")
    return float((reset_value - patched_value) / denominator)


def add_recovery(variants: dict[str, dict]) -> None:
    reset = variants["reset_raw"]["metrics"]["mean"]
    for name, row in variants.items():
        mean = row["metrics"]["mean"]
        row["recovery"] = {
            key: recovery_ratio(float(reset[key]), float(mean[key]))
            for key in reset
        }


def patch_specs(decoder_depth: int, decoder_layers: list[int] | None) -> list[PatchSpec]:
    layers = list(range(decoder_depth)) if decoder_layers is None else sorted(set(decoder_layers))
    specs = [
        PatchSpec("encoder_final", ("encoder_final",)),
        PatchSpec("camera_initial", ("camera_initial",)),
        PatchSpec("camera_refined", ("camera_refined",)),
        PatchSpec("human_prompt", ("human_prompt",)),
        PatchSpec("human_refined", ("human_refined",)),
        PatchSpec("persistent_state", ("persistent_state",)),
    ]
    specs.extend(PatchSpec(f"decoder_image_l{idx:02d}", (f"decoder_image_l{idx}",)) for idx in layers)
    state_layers = sorted(set([0, decoder_depth // 2, decoder_depth - 1]))
    specs.extend(PatchSpec(f"decoder_state_l{idx:02d}", (f"decoder_state_l{idx}",)) for idx in state_layers)
    final_scene = f"decoder_image_l{decoder_depth - 1}"
    specs.extend(
        [
            PatchSpec("camera_plus_state", ("camera_initial", "persistent_state")),
            PatchSpec("image_plus_state", (final_scene, "persistent_state")),
            PatchSpec("camera_plus_image", ("camera_initial", final_scene)),
            PatchSpec(
                "camera_image_human",
                ("camera_initial", final_scene, "human_refined"),
            ),
            PatchSpec(
                "all_key_tokens",
                ("persistent_state", "camera_initial", "human_prompt", "decoder_final_full"),
            ),
            PatchSpec("control_random_final_scene", (final_scene,), "random"),
            PatchSpec("control_shuffle_final_scene", (final_scene,), "shuffle"),
        ]
    )
    return specs


def find_donor(record: dict, records: list[dict]) -> dict | None:
    for candidate in records:
        if candidate is record:
            continue
        if candidate.get("source") == record.get("source") and candidate.get("seqA") != record.get("seqA"):
            return candidate
    return None


def run_case(
    model: ARCroco3DStereo,
    record: dict,
    all_records: list[dict],
    args: argparse.Namespace,
    device: torch.device,
    case_index: int,
    views_override: list[dict] | None = None,
) -> dict:
    case_name = safe_name(str(record.get("pattern_id", f"case_{case_index:03d}")))
    case_dir = args.output_dir / "cases" / case_name
    report_path = case_dir / "activation_metrics.json"
    if report_path.is_file() and not args.overwrite:
        return json.loads(report_path.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)

    views = (
        load_video_views(record, args, device, model.mhmr_img_res)
        if views_override is None
        else views_override
    )
    boundary = int(args.boundary)
    if boundary <= 0 or boundary >= len(views):
        raise ValueError(f"Invalid boundary={boundary} for {len(views)} views")
    teacher_predictions, teacher_latents, teacher_seconds, _ = run_branch(
        model, views, device, boundary, capture=True, seed=args.seed + case_index
    )
    teacher_post = teacher_predictions[boundary:]
    student_views = views[boundary:]
    reset_predictions, reset_latents, reset_seconds, _ = run_branch(
        model, student_views, device, 0, capture=True, seed=args.seed + case_index
    )

    variants = {
        "teacher_continuous": {
            "metrics": evaluate_branch(teacher_post, teacher_post, args, args.seed + case_index),
            "seconds": teacher_seconds,
            "patch": [],
            "skipped_replacements": [],
        },
        "reset_raw": {
            "metrics": evaluate_branch(reset_predictions, teacher_post, args, args.seed + case_index),
            "seconds": reset_seconds,
            "patch": [],
            "skipped_replacements": [],
        },
    }
    teacher_source = source_dict(teacher_latents)
    for spec_index, spec in enumerate(patch_specs(len(model.dec_blocks), args.decoder_layers)):
        predictions, _captured, seconds, skipped = run_branch(
            model,
            student_views,
            device,
            0,
            capture=False,
            patch=spec,
            source=teacher_source,
            seed=args.seed + case_index * 1000 + spec_index,
        )
        variants[spec.name] = {
            "metrics": evaluate_branch(predictions, teacher_post, args, args.seed + case_index),
            "seconds": seconds,
            "patch": list(spec.components),
            "source_mode": spec.source_mode,
            "skipped_replacements": skipped,
        }

    self_spec = PatchSpec(
        "control_self_noop",
        ("persistent_state", "camera_initial", "human_prompt", "decoder_final_full"),
        "teacher",
    )
    self_predictions, _captured, seconds, skipped = run_branch(
        model,
        student_views,
        device,
        0,
        patch=self_spec,
        source=source_dict(reset_latents),
        seed=args.seed + case_index,
    )
    variants[self_spec.name] = {
        "metrics": evaluate_branch(self_predictions, teacher_post, args, args.seed + case_index),
        "seconds": seconds,
        "patch": list(self_spec.components),
        "source_mode": "self",
        "skipped_replacements": skipped,
    }

    donor_record = None if views_override is not None else find_donor(record, all_records)
    donor_debug = None
    if donor_record is not None:
        try:
            donor_views = load_video_views(donor_record, args, device, model.mhmr_img_res)
            _donor_predictions, donor_latents, donor_seconds, _ = run_branch(
                model, donor_views, device, boundary, capture=True, seed=args.seed + case_index + 50000
            )
            donor_spec = PatchSpec(
                "control_other_video",
                ("persistent_state", "camera_initial", "human_prompt", "decoder_final_full"),
            )
            donor_predictions, _captured, seconds, skipped = run_branch(
                model,
                student_views,
                device,
                0,
                patch=donor_spec,
                source=source_dict(donor_latents),
                seed=args.seed + case_index + 60000,
            )
            variants[donor_spec.name] = {
                "metrics": evaluate_branch(donor_predictions, teacher_post, args, args.seed + case_index),
                "seconds": seconds,
                "patch": list(donor_spec.components),
                "source_mode": "other_video",
                "skipped_replacements": skipped,
            }
            donor_debug = {
                "record": donor_record,
                "capture_seconds": donor_seconds,
            }
            del donor_views
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            donor_debug = {"record": donor_record, "error": str(exc)}

    add_recovery(variants)
    report = {
        "case_name": case_name,
        "record": record,
        "teacher_frames": len(views),
        "post_frames": len(student_views),
        "boundary": boundary,
        "model_depth": {
            "encoder": len(model.enc_blocks),
            "decoder": len(model.dec_blocks),
            "state_tokens": int(model.state_size),
        },
        "latent_shapes": {
            key: list(value.shape)
            for key, value in teacher_source.items()
        },
        "donor": donor_debug,
        "variants": variants,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    del views, student_views
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def aggregate(cases: list[dict]) -> dict:
    names = sorted(set.intersection(*(set(case["variants"]) for case in cases)))
    result = {}
    for name in names:
        metric_keys = cases[0]["variants"][name]["metrics"]["mean"].keys()
        recovery_keys = cases[0]["variants"][name]["recovery"].keys()
        result[name] = {
            "count": len(cases),
            "mean_error": {
                key: finite_mean([case["variants"][name]["metrics"]["mean"][key] for case in cases])
                for key in metric_keys
            },
            "mean_recovery": {
                key: finite_mean([case["variants"][name]["recovery"][key] for case in cases])
                for key in recovery_keys
            },
            "seconds": finite_mean([case["variants"][name]["seconds"] for case in cases]),
        }
    return result


def write_csv(path: Path, cases: list[dict]) -> None:
    rows = []
    for case in cases:
        for name, variant in case["variants"].items():
            row = {
                "case_name": case["case_name"],
                "source": case["record"].get("source"),
                "variant": name,
                "seconds": variant["seconds"],
            }
            row.update({f"error_{key}": value for key, value in variant["metrics"]["mean"].items()})
            row.update({f"recovery_{key}": value for key, value in variant["recovery"].items()})
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_recovery(path: Path, overall: dict) -> None:
    preferred = [
        "reset_raw",
        "camera_initial",
        "camera_refined",
        "human_prompt",
        "human_refined",
        "persistent_state",
        "camera_plus_state",
        "image_plus_state",
        "camera_image_human",
        "all_key_tokens",
        "control_random_final_scene",
        "control_shuffle_final_scene",
        "control_other_video",
    ]
    names = [name for name in preferred if name in overall]
    metrics = [
        "camera_translation_m",
        "camera_rotation_deg",
        "pointmap_world_mean_m",
        "human_world_root_m",
        "human_global_orientation_deg",
    ]
    matrix = np.asarray([[overall[name]["mean_recovery"].get(key, np.nan) for key in metrics] for name in names])
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(names))))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)), [key.replace("_", "\n") for key in metrics], fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for y in range(len(names)):
        for x in range(len(metrics)):
            value = matrix[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Activation Patching Recovery Ratio")
    fig.colorbar(image, ax=ax, label="Recovery ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_layer_curves(path: Path, overall: dict) -> None:
    names = sorted(
        (name for name in overall if name.startswith("decoder_image_l")),
        key=lambda value: int(value.rsplit("l", 1)[1]),
    )
    if not names:
        return
    layers = [int(name.rsplit("l", 1)[1]) for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in (
        (axes[0], "camera_rotation_deg", "Camera rotation"),
        (axes[1], "pointmap_world_mean_m", "World pointmap"),
        (axes[2], "human_world_root_m", "Human world root"),
    ):
        values = [overall[name]["mean_recovery"].get(key, np.nan) for name in names]
        ax.plot(layers, values, marker="o")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Decoder layer")
        ax.set_ylabel("Recovery ratio")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def architecture_judgement(overall: dict) -> list[str]:
    def score(name: str, key: str) -> float:
        return float(overall.get(name, {}).get("mean_recovery", {}).get(key, float("nan")))

    conclusions = []
    camera_candidates = {
        "camera_initial": score("camera_initial", "camera_rotation_deg"),
        "camera_refined": score("camera_refined", "camera_rotation_deg"),
        "persistent_state": score("persistent_state", "camera_rotation_deg"),
    }
    valid_camera = {key: value for key, value in camera_candidates.items() if np.isfinite(value)}
    if valid_camera:
        best = max(valid_camera, key=valid_camera.get)
        conclusions.append(f"相机旋转因果恢复最强的单项是 {best}，Recovery={valid_camera[best]:.3f}。")
    state_keys = ("camera_rotation_deg", "pointmap_world_mean_m", "human_world_root_m")
    state_scores = [score("persistent_state", key) for key in state_keys]
    if all(np.isfinite(value) and value > 0.5 for value in state_scores):
        conclusions.append("persistent state 同时恢复 camera、scene 和 human，优先设计只读 state-query Shot Prompt。")
    elif any(np.isfinite(value) and value > 0.3 for value in state_scores):
        conclusions.append("persistent state 有部分 world-context 因果作用，但需要与 camera/image token 联合读取。")
    else:
        conclusions.append("persistent state 的跨分支因果恢复较弱，不应直接把 raw state 当作完整世界坐标记忆。")
    human_id = score("human_refined", "human_global_orientation_deg")
    if np.isfinite(human_id) and human_id > 0.3:
        conclusions.append("refined human token 对人体朝向有因果作用，可用于身份、torso heading 和运动约束。")
    decoder_names = [name for name in overall if name.startswith("decoder_image_l")]
    if decoder_names:
        best_scene = max(decoder_names, key=lambda name: score(name, "pointmap_world_mean_m"))
        conclusions.append(
            f"world pointmap 最适合插入 scene residual 的层为 {best_scene}，Recovery={score(best_scene, 'pointmap_world_mean_m'):.3f}。"
        )
    all_score = finite_mean(
        [score("all_key_tokens", key) for key in ("camera_rotation_deg", "pointmap_world_mean_m", "human_world_root_m")]
    )
    if np.isfinite(all_score) and all_score < 0.15:
        conclusions.append("关键 token 联合替换仍几乎不能恢复连续分支，现有 latent 不适合承担 gauge recovery。")
    return conclusions


def write_markdown(path: Path, report: dict) -> None:
    overall = report["overall"]
    keys = [
        "reset_raw",
        "camera_initial",
        "camera_refined",
        "human_prompt",
        "human_refined",
        "persistent_state",
        "camera_plus_state",
        "image_plus_state",
        "camera_image_human",
        "all_key_tokens",
        "control_self_noop",
        "control_random_final_scene",
        "control_shuffle_final_scene",
        "control_other_video",
    ]
    lines = [
        "# V10 Human3R Latent Activation Patching Probe",
        "",
        "Human3R 主体冻结。Teacher 连续运行，Student 在同一边界 RGB 前 fresh reset；只替换 latent activation，不复制最终输出。",
        "",
        "| Variant | Cam T err | Cam R err | World PM err | Root err | Cam R recovery | Scene recovery | Root recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in keys:
        if name not in overall:
            continue
        error = overall[name]["mean_error"]
        recovery = overall[name]["mean_recovery"]
        lines.append(
            f"| {name} | {error['camera_translation_m']:.4f} | {error['camera_rotation_deg']:.2f} | "
            f"{error['pointmap_world_mean_m']:.4f} | {error['human_world_root_m']:.4f} | "
            f"{recovery['camera_rotation_deg']:.3f} | {recovery['pointmap_world_mean_m']:.3f} | "
            f"{recovery['human_world_root_m']:.3f} |"
        )
    lines.extend(["", "## 自动判定", ""])
    lines.extend(f"- {item}" for item in report["architecture_judgement"])
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- Recovery Ratio=0 表示相对 reset 无改善，1 表示恢复到 continuous teacher，负值表示被破坏。",
            "- Encoder token 对同一 RGB 理论上应基本一致；其 self-patch 主要用于验证插桩正确性。",
            "- 该实验测量相对于连续 Human3R 分支的因果恢复，不把 teacher 输出当作绝对 GT。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(args)
    all_records = read_jsonl(args.records)
    device = torch.device(args.device)
    model = build_model(args)
    cases = []
    failures = []
    for index, record in enumerate(records):
        print(f">> [{index + 1}/{len(records)}] {record['source']} {record['pattern_id']}", flush=True)
        try:
            cases.append(run_case(model, record, all_records, args, device, index))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            failures.append({"record": record, "error": str(exc)})
            print(f"!! skip {record.get('pattern_id')}: {exc}", flush=True)
    if not cases:
        raise RuntimeError(f"All activation-patching cases failed: {failures}")
    overall = aggregate(cases)
    report = {
        "experiment": "Human3R latent activation patching",
        "args": vars(args),
        "case_count": len(cases),
        "failures": failures,
        "overall": overall,
        "architecture_judgement": architecture_judgement(overall),
        "cases": cases,
    }
    serializable = json.loads(json.dumps(report, default=str, allow_nan=True))
    (args.output_dir / "activation_patching_metrics.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "activation_patching_cases.csv", cases)
    plot_recovery(args.output_dir / "activation_recovery_matrix.png", overall)
    plot_layer_curves(args.output_dir / "decoder_layer_recovery_curves.png", overall)
    write_markdown(args.output_dir / "activation_patching_metrics.md", serializable)
    print(f">> wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
