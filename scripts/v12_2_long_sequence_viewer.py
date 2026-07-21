#!/usr/bin/env python3
"""V12.2 interactive 10+10 frame viewer for retained boundary methods."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import boundary_viewer_base as base  # noqa: E402
import v11_1_boundary_method_comparison_viewer as comparison  # noqa: E402


EXAMPLES = {
    "AvatarReX 148 deg cut": "avatarrex_120_150_lbn2_1651_22010714_22070932",
    "THuman clean alignment": "thuman_060_090_thuman02_2770_cam12_cam07",
    "MVHuman100 101 deg rescue": "mvhuman100_090_120_100003_338_CC32871A035_CC32871A008",
    "MVHuman200 122 deg extreme": "mvhuman200_120_150_200002_410_22327109_22236235",
}

METHODS = {
    "Fixed Explicit": None,
    "Torso Only": None,
    "Conditional Wide Rotation": None,
    "Contact-Preserving Alignment": None,
    "Uniform Similarity - Torso": None,
    "Uniform Similarity - Conditional Wide": None,
}

COLORS = {
    **comparison.COLORS,
    "Uniform Similarity - Torso": (234, 88, 12),
    "Uniform Similarity - Conditional Wide": (5, 150, 105),
}

VIEW_FRAMES = {
    "Boundary pair": (9, 10),
    "All 20 frames": tuple(range(20)),
    "Every second frame": tuple(range(0, 20, 2)),
    "Pre-cut 10 frames": tuple(range(10)),
    "Post-cut 10 frames": tuple(range(10, 20)),
}
VIEW_FRAMES.update({f"Single frame {index:02d}": (index,) for index in range(20)})


def parse_args() -> argparse.Namespace:
    args = comparison.parse_args()
    args.boundary = 10
    args.point_stride = 10
    args.point_size = 0.007
    args.long_cache = ROOT / "output/v52_long_sequence_visualization/cache"
    args.uniform_similarity_report = (
        ROOT
        / "output/v53_uniform_similarity_integrity"
        / "v53_uniform_similarity_integrity_probe.json"
    )
    return args


def uniform_method(case: dict, variant: str, branch: str) -> dict:
    value = case["variants"][variant]
    scales = value["scales"]
    return {
        "transform": value["transform"],
        "camera_translation_error_m": float(value["camera"]["translation_m"]),
        "camera_rotation_error_deg": float(value["camera"]["rotation_deg"]),
        "root_scales": {
            "old": float(scales["old"]),
            "new": float(scales["new"]),
        },
        "scene_scales": {
            "old": float(scales["old"]),
            "new": float(scales["new"]),
        },
        "human_motion_error_m": float(value["human"]["root_motion_error_m"]),
        "scene_error_m": float(value["scene"]["trimmed_mean_m"]),
        "contact_distortion_before_m": 0.0,
        "contact_distortion_after_m": 0.0,
        "contact_correction_m": 0.0,
        "human_reprojection_shift_px": 0.0,
        "rigid_local_geometry": False,
        "uniform_similarity": True,
        "preserve_contact": False,
        "branch": branch,
    }


def load_frame(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path / "camera" / f"{index:06d}.npz") as camera:
        pose = camera["pose"].astype(np.float32)
        intrinsics = camera["intrinsics"].astype(np.float32)
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as smpl_file:
        smpl = {key: smpl_file[key] for key in smpl_file.files}
    depth = np.load(path / "depth" / f"{index:06d}.npy").astype(np.float32)
    confidence = np.load(path / "conf" / f"{index:06d}.npy").astype(np.float32)
    image = np.asarray(Image.open(path / "color" / f"{index:06d}.png").convert("RGB"))
    return pose, intrinsics, smpl, depth, confidence, image


def load_long_case(
    name: str,
    candidate: dict,
    cache_root: Path,
    layer,
    device: torch.device,
    args: argparse.Namespace,
) -> base.CachedCase:
    manifest_path = cache_root / name / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = int(manifest["frames_per_shot"])
    if count != int(args.boundary):
        raise ValueError(f"Viewer boundary {args.boundary} does not match cache count {count}")
    side_specs = (
        (Path(manifest["pre_dir"]), np.asarray(manifest["pre_to_v10_gauge"], dtype=np.float32)),
        (Path(manifest["post_dir"]), np.asarray(manifest["post_to_v10_gauge"], dtype=np.float32)),
    )

    points_world: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    images: list[np.ndarray] = []
    rotvecs, shapes, translations, expressions = [], [], [], []
    masks = []
    kernel = np.ones((args.mask_dilate, args.mask_dilate), dtype=np.uint8)

    for path, gauge in side_specs:
        for index in range(count):
            raw_pose, K, smpl, depth, confidence, image = load_frame(path, index)
            if len(smpl["shape"]) == 0:
                raise ValueError(f"No predicted SMPL-X person in {name}: {path.name} frame {index}")
            pose = (gauge @ raw_pose).astype(np.float32)
            mask = np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8)
            if int(args.mask_dilate) > 1:
                mask = cv2.dilate(mask, kernel, iterations=1)
            world = base.reconstruct_world_pointmap(pose, K, depth)
            valid = (
                np.isfinite(world).all(axis=-1)
                & np.isfinite(depth)
                & np.isfinite(confidence)
                & (depth > 0.05)
                & (depth < 50.0)
                & (confidence > float(args.confidence_threshold))
                & (mask == 0)
            )
            ids = np.flatnonzero(valid.reshape(-1))[:: max(int(args.point_stride), 1)]
            points_world.append(world.reshape(-1, 3)[ids].astype(np.float32))
            point_colors.append(image.reshape(-1, 3)[ids].astype(np.uint8))
            poses.append(pose)
            intrinsics.append(K)
            images.append(image)
            rotvecs.append(np.asarray(smpl["rotvec"][0], dtype=np.float32))
            shapes.append(np.asarray(smpl["shape"][0], dtype=np.float32))
            translations.append(np.asarray(smpl["transl"][0], dtype=np.float32))
            expression = smpl["expression"]
            expressions.append(
                np.zeros(10, dtype=np.float32)
                if expression is None or len(expression) == 0
                else np.asarray(expression[0], dtype=np.float32)
            )
            masks.append(mask)

    pose_array = np.stack(poses).astype(np.float32)
    intrinsic_array = np.stack(intrinsics).astype(np.float32)
    with torch.no_grad():
        body = layer(
            torch.from_numpy(np.stack(rotvecs)).to(device),
            torch.from_numpy(np.stack(shapes)).to(device),
            torch.from_numpy(np.stack(translations)).to(device),
            None,
            None,
            K=torch.from_numpy(intrinsic_array).to(device),
            expression=torch.from_numpy(np.stack(expressions)).to(device),
        )
    vertices_camera = body["smpl_v3d"].detach().float().cpu().numpy().astype(np.float32)
    joints_camera = body["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32)
    vertices_world = (
        np.einsum("nij,nvj->nvi", pose_array[:, :3, :3], vertices_camera)
        + pose_array[:, None, :3, 3]
    ).astype(np.float32)
    case = base.CachedCase(
        case_name=name,
        candidate=candidate,
        points_world=points_world,
        point_colors=point_colors,
        camera_poses=pose_array,
        intrinsics=intrinsic_array,
        images=images,
        smpl_vertices_world=[vertices_world[index] for index in range(2 * count)],
    )
    names = layer.joint_names
    foot_indices = [
        names.index("left_big_toe"),
        names.index("left_small_toe"),
        names.index("left_heel"),
        names.index("right_big_toe"),
        names.index("right_small_toe"),
        names.index("right_heel"),
    ]
    case.smpl_root_camera = np.stack(translations).astype(np.float32)
    case.smpl_pelvis_camera = joints_camera[:, names.index("pelvis")]
    case.smpl_feet_camera = np.median(joints_camera[:, foot_indices], axis=1).astype(np.float32)
    case.long_manifest = manifest
    return case


class LongSequenceViewer(comparison.RetainedMethodsViewer):
    def __init__(self, *args, **kwargs) -> None:
        base.ComparisonViewer.__init__(self, *args, **kwargs)
        self.show_oracle_gui.value = False
        self.method_gui.value = "Uniform Similarity - Conditional Wide"
        self.view_gui.value = "All 20 frames"
        self.render_scene()

    def scaled_geometry(
        self,
        case: base.CachedCase,
        result: dict,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not bool(result.get("uniform_similarity", False)):
            return super().scaled_geometry(case, result, frame)
        side = "old" if frame < int(self.args.boundary) else "new"
        scale = float(result["scene_scales"][side])
        raw_pose = case.camera_poses[frame]
        pose = comparison.scale_pose(raw_pose, scale)
        local_points = comparison.camera_points(case.points_world[frame], raw_pose) * scale
        local_vertices = comparison.camera_points(case.smpl_vertices_world[frame], raw_pose) * scale
        points = (
            np.einsum("ij,nj->ni", raw_pose[:3, :3], local_points)
            + pose[:3, 3]
        ).astype(np.float32)
        vertices = (
            np.einsum("ij,nj->ni", raw_pose[:3, :3], local_vertices)
            + pose[:3, 3]
        ).astype(np.float32)
        root_camera = np.asarray(case.smpl_root_camera[frame], dtype=np.float32) * scale
        root_world = raw_pose[:3, :3] @ root_camera + pose[:3, 3]
        return points, vertices, pose, root_world.astype(np.float32)

    def update_metrics(self, case: base.CachedCase, result: dict, method: str) -> None:
        super().update_metrics(case, result, method)
        method_note = (
            "Contact-Preserving Alignment includes a per-frame foot/contact diagnostic correction."
            if method == "Contact-Preserving Alignment"
            else (
                "One shot scale preserves the complete camera, human, and scene similarity."
                if bool(result.get("uniform_similarity", False))
                else "The same boundary transform is fixed for all 10 post-cut frames."
            )
        )
        self.metrics_gui.content += (
            "\n- Long sequence: **10 pre-cut + 10 post-cut frames**"
            f"\n- Streaming behavior: **{method_note}**"
        )

    def render_scene(self) -> None:
        super().render_scene()
        case = self.selected_case()
        boundary = int(self.args.boundary)
        self.pre_image_gui.image = case.images[boundary - 1]
        self.post_image_gui.image = case.images[boundary]

    def run(self) -> None:
        print(f"V12.2 long-sequence 3D viewer: http://127.0.0.1:{self.args.port}", flush=True)
        while True:
            time.sleep(10.0)


def main() -> None:
    args = parse_args()
    args.boundary = 10
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    comparison.EXAMPLES = EXAMPLES
    comparison.METHODS = METHODS
    comparison.COLORS = COLORS
    base.EXAMPLES = EXAMPLES
    base.METHODS = METHODS
    base.VIEW_FRAMES = VIEW_FRAMES
    base.CAMERA_COLORS = {key: COLORS[key] for key in METHODS}
    base.method_result = comparison.method_result

    _, candidates = comparison.load_maps(args)
    uniform_payload = json.loads(args.uniform_similarity_report.read_text(encoding="utf-8"))
    uniform_cases = {row["case_name"]: row for row in uniform_payload["cases"]}
    for name in EXAMPLES.values():
        candidates[name]["viewer_methods"]["Uniform Similarity - Torso"] = uniform_method(
            uniform_cases[name],
            "torso_uniform_scene",
            "Torso rotation + one integrity-preserving scene scale",
        )
        candidates[name]["viewer_methods"]["Uniform Similarity - Conditional Wide"] = uniform_method(
            uniform_cases[name],
            "v47_uniform_scene",
            "conditional wide rotation + one integrity-preserving scene scale",
        )
    device = torch.device(args.device)
    layer = base.build_smpl_layer(device)
    cache = {}
    for label, name in EXAMPLES.items():
        print(f"Preparing 20-frame interactive scene: {label}", flush=True)
        cache[label] = load_long_case(
            name,
            candidates[name],
            args.long_cache,
            layer,
            device,
            args,
        )
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    del layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    viewer = LongSequenceViewer(cache, faces, args)
    viewer.run()


if __name__ == "__main__":
    main()
