#!/usr/bin/env python3
"""Internal Human3R-style 3D viewer for cached boundary examples.

This uses the same viser stack and SMPL-X reconstruction convention as
``demo.py``. Human3R inference is not rerun: RGB pointmaps, camera poses, SMPL-X
parameters and boundary transforms are loaded from completed experiment caches.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import viser
import viser.transforms as tf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402


DEFAULT_V10_REPORT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_CANDIDATE_DIR = (
    REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
)


EXAMPLES = {
    "AvatarReX clear gain": "avatarrex_120_150_lbn2_1842_22010710_22070932",
    "THuman clear gain": "thuman_090_120_thuman02_2772_cam08_cam16",
    "MVHuman100 large angle": "mvhuman100_090_120_100003_338_CC32871A035_CC32871A008",
    "MVHuman200 large angle": "mvhuman200_060_090_200003_426_22327109_22327073",
    "AvatarReX failure case": "avatarrex_150_180_lbn1_1632_22010716_22139907",
    "Wide + torso modular gain": "thuman_120_150_thuman00_2442_cam04_cam19",
}

METHODS = {
    "Hard Reset": ("baselines", "hard_reset"),
    "Fixed Explicit": ("baselines", "fixed_explicit"),
    "Torso Motion": ("fixed_candidates", "fixed_torso_motion_1f_resolve_t"),
    "Wide Coarse": ("baselines", "v15_coarse"),
    "Wide + Torso": ("v15_candidates", "v15_torso_motion_1f_resolve_t"),
    "Boundary Oracle": ("baselines", "boundary_oracle"),
}

VIEW_FRAMES = {
    "Boundary pair": (1, 2),
    "All four frames": (0, 1, 2, 3),
    "Pre-cut only": (0, 1),
    "Post-cut only": (2, 3),
}

CAMERA_COLORS = {
    "Hard Reset": (217, 70, 70),
    "Fixed Explicit": (220, 124, 25),
    "Torso Motion": (18, 160, 111),
    "Wide Coarse": (214, 91, 54),
    "Wide + Torso": (13, 148, 136),
    "Boundary Oracle": (37, 99, 235),
}
PRE_CAMERA_COLOR = (31, 41, 55)
ORACLE_CAMERA_COLOR = (192, 38, 211)
PRE_HUMAN_COLOR = (68, 142, 247)
POST_HUMAN_COLOR = (245, 125, 54)


@dataclass
class CachedCase:
    case_name: str
    candidate: dict
    points_world: list[np.ndarray]
    point_colors: list[np.ndarray]
    camera_poses: np.ndarray
    intrinsics: np.ndarray
    images: list[np.ndarray]
    smpl_vertices_world: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10_REPORT)
    parser.add_argument("--candidate_dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--point_stride", type=int, default=6)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--point_size", type=float, default=0.008)
    return parser.parse_args()


def load_case_maps(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    report = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v10_cases = {str(case["case_name"]): case for case in report["cases"]}
    candidates: dict[str, dict] = {}
    for path in sorted(args.candidate_dir.glob("v16_candidates_shard_*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        for case in shard["cases"]:
            candidates[str(case["case_name"])] = case
    return v10_cases, candidates


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nj->ni", transform[:3, :3], points) + transform[:3, 3]


def reconstruct_world_pointmap(
    pose: np.ndarray,
    intrinsics: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    points_camera = np.stack(
        [
            (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
            (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
            depth,
        ],
        axis=-1,
    )
    return (
        np.einsum("ij,hwj->hwi", pose[:3, :3], points_camera)
        + pose[:3, 3]
    ).astype(np.float32)


def build_smpl_layer(device: torch.device) -> SMPL_Layer:
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    layer.eval()
    return layer


def load_cached_case(
    case_name: str,
    v10_case: dict,
    candidate: dict,
    layer: SMPL_Layer,
    device: torch.device,
    args: argparse.Namespace,
) -> CachedCase:
    local_dir = Path(v10_case["paths"]["human3r_local_reset"])
    points_world: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    poses, intrinsics, images = [], [], []
    rotvecs, shapes, translations, expressions = [], [], [], []

    kernel = np.ones((args.mask_dilate, args.mask_dilate), dtype=np.uint8)
    for frame in range(4):
        with np.load(local_dir / "camera" / f"{frame:06d}.npz") as camera:
            pose = camera["pose"].astype(np.float32)
            K = camera["intrinsics"].astype(np.float32)
        with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
            if len(smpl["shape"]) == 0:
                raise ValueError(f"No predicted SMPL-X person in {case_name}, frame {frame}")
            mask = np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8)
            if args.mask_dilate > 1:
                mask = cv2.dilate(mask, kernel, iterations=1)
            rotvecs.append(smpl["rotvec"][0].astype(np.float32))
            shapes.append(smpl["shape"][0].astype(np.float32))
            translations.append(smpl["transl"][0].astype(np.float32))
            expression = smpl["expression"]
            expressions.append(
                np.zeros(10, dtype=np.float32)
                if expression is None or len(expression) == 0
                else expression[0].astype(np.float32)
            )

        depth = np.load(local_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
        confidence = np.load(local_dir / "conf" / f"{frame:06d}.npy").astype(np.float32)
        image = np.asarray(
            Image.open(local_dir / "color" / f"{frame:06d}.png").convert("RGB")
        )
        world = reconstruct_world_pointmap(pose, K, depth)
        valid = (
            np.isfinite(world).all(axis=-1)
            & np.isfinite(depth)
            & np.isfinite(confidence)
            & (depth > 0.05)
            & (depth < 50.0)
            & (confidence > float(args.confidence_threshold))
            & (mask == 0)
        )
        indices = np.flatnonzero(valid.reshape(-1))[:: max(int(args.point_stride), 1)]
        points_world.append(world.reshape(-1, 3)[indices].astype(np.float32))
        point_colors.append(image.reshape(-1, 3)[indices].astype(np.uint8))
        poses.append(pose)
        intrinsics.append(K)
        images.append(image)

    with torch.no_grad():
        smpl_output = layer(
            torch.from_numpy(np.stack(rotvecs)).to(device),
            torch.from_numpy(np.stack(shapes)).to(device),
            torch.from_numpy(np.stack(translations)).to(device),
            None,
            None,
            K=torch.from_numpy(np.stack(intrinsics)).to(device),
            expression=torch.from_numpy(np.stack(expressions)).to(device),
        )
    vertices_camera = smpl_output["smpl_v3d"].detach().float().cpu().numpy().astype(np.float32)
    poses_np = np.stack(poses).astype(np.float32)
    vertices_world = (
        np.einsum("nij,nvj->nvi", poses_np[:, :3, :3], vertices_camera)
        + poses_np[:, None, :3, 3]
    ).astype(np.float32)
    return CachedCase(
        case_name=case_name,
        candidate=candidate,
        points_world=points_world,
        point_colors=point_colors,
        camera_poses=poses_np,
        intrinsics=np.stack(intrinsics).astype(np.float32),
        images=images,
        smpl_vertices_world=[vertices_world[index] for index in range(4)],
    )


def method_result(case: CachedCase, method: str) -> dict:
    group, key = METHODS[method]
    return case.candidate[group][key]


class ComparisonViewer:
    def __init__(
        self,
        cache: dict[str, CachedCase],
        smpl_faces: np.ndarray,
        args: argparse.Namespace,
    ) -> None:
        self.cache = cache
        self.smpl_faces = np.asarray(smpl_faces, dtype=np.int32)
        self.args = args
        self.lock = threading.Lock()
        self.handles: dict[str, list] = {
            "pointcloud": [],
            "smpl": [],
            "camera": [],
            "oracle_camera": [],
            "trajectory": [],
            "label": [],
        }
        self.scene_center = np.zeros(3, dtype=np.float32)
        self.scene_radius = 3.0

        first_label = next(iter(EXAMPLES))
        first_case = self.cache[first_label]
        self.server = viser.ViserServer(
            host="0.0.0.0",
            port=int(args.port),
            label="Boundary Alignment 3D Comparison",
        )
        self.server.scene.set_up_direction("-y")
        self.server.scene.add_frame("/world", show_axes=True, axes_length=0.35, axes_radius=0.012)

        with self.server.gui.add_folder("Comparison"):
            self.example_gui = self.server.gui.add_dropdown(
                "Example", tuple(EXAMPLES.keys()), initial_value=first_label
            )
            self.method_gui = self.server.gui.add_dropdown(
                "Boundary candidate",
                tuple(METHODS.keys()),
                initial_value="Fixed Explicit",
            )
            self.view_gui = self.server.gui.add_dropdown(
                "Frames", tuple(VIEW_FRAMES.keys()), initial_value="Boundary pair"
            )
            self.reset_gui = self.server.gui.add_button("Reset view")

        with self.server.gui.add_folder("Geometry"):
            self.show_points_gui = self.server.gui.add_checkbox("Show RGB pointmap", True)
            self.show_smpl_gui = self.server.gui.add_checkbox("Show SMPL-X", True)
            self.show_cameras_gui = self.server.gui.add_checkbox("Show cameras", True)
            self.show_oracle_gui = self.server.gui.add_checkbox("Show Oracle camera", True)
            self.point_size_gui = self.server.gui.add_slider(
                "Point size",
                min=0.001,
                max=0.03,
                step=0.001,
                initial_value=float(args.point_size),
            )

        self.metrics_gui = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Boundary RGB"):
            self.pre_image_gui = self.server.gui.add_image(
                first_case.images[1], label="Last pre-cut frame"
            )
            self.post_image_gui = self.server.gui.add_image(
                first_case.images[2], label="First post-cut frame"
            )

        @self.example_gui.on_update
        def _(_) -> None:
            self.render_scene()

        @self.method_gui.on_update
        def _(_) -> None:
            self.render_scene()

        @self.view_gui.on_update
        def _(_) -> None:
            self.render_scene()

        @self.show_points_gui.on_update
        def _(_) -> None:
            self.set_category_visibility("pointcloud", self.show_points_gui.value)

        @self.show_smpl_gui.on_update
        def _(_) -> None:
            self.set_category_visibility("smpl", self.show_smpl_gui.value)

        @self.show_cameras_gui.on_update
        def _(_) -> None:
            self.set_category_visibility("camera", self.show_cameras_gui.value)
            self.set_category_visibility("trajectory", self.show_cameras_gui.value)
            self.set_category_visibility("label", self.show_cameras_gui.value)

        @self.show_oracle_gui.on_update
        def _(_) -> None:
            self.set_category_visibility("oracle_camera", self.show_oracle_gui.value)

        @self.point_size_gui.on_update
        def _(_) -> None:
            for handle in self.handles["pointcloud"]:
                handle.point_size = float(self.point_size_gui.value)

        @self.reset_gui.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.reset_client_camera(event.client)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.reset_client_camera(client)

        self.render_scene()

    def selected_case(self) -> CachedCase:
        return self.cache[str(self.example_gui.value)]

    def clear_scene(self) -> None:
        for category in self.handles.values():
            for handle in category:
                try:
                    handle.remove()
                except (KeyError, AttributeError):
                    pass
            category.clear()

    def set_category_visibility(self, category: str, visible: bool) -> None:
        for handle in self.handles[category]:
            handle.visible = bool(visible)

    def frame_transform(self, transform: np.ndarray, frame: int) -> np.ndarray:
        if frame < int(self.args.boundary):
            return np.eye(4, dtype=np.float32)
        return transform

    @staticmethod
    def line_segments(points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.empty((0, 2, 3), dtype=np.float32)
        return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)

    def add_camera(
        self,
        name: str,
        pose: np.ndarray,
        K: np.ndarray,
        image: np.ndarray | None,
        color: tuple[int, int, int],
        category: str,
        scale: float,
    ) -> None:
        height = int(image.shape[0]) if image is not None else int(round(K[1, 2] * 2.0))
        width = int(image.shape[1]) if image is not None else int(round(K[0, 2] * 2.0))
        fov = 2.0 * np.arctan((height * 0.5) / max(float(K[1, 1]), 1e-6))
        handle = self.server.scene.add_camera_frustum(
            name,
            fov=float(fov),
            aspect=float(width / max(height, 1)),
            scale=scale,
            line_width=2.0,
            color=color,
            image=image,
            wxyz=tf.SO3.from_matrix(pose[:3, :3]).wxyz,
            position=pose[:3, 3],
        )
        self.handles[category].append(handle)

    def update_metrics(self, case: CachedCase, result: dict, method: str) -> None:
        fixed = method_result(case, "Fixed Explicit")
        rotation_gain = float(fixed["camera_rotation_error_deg"]) - float(
            result["camera_rotation_error_deg"]
        )
        translation_gain = float(fixed["camera_translation_error_m"]) - float(
            result["camera_translation_error_m"]
        )
        residual = result.get("bounded_residual_deg")
        residual_text = "n/a" if residual is None else f"{float(residual):.1f} deg"
        self.metrics_gui.content = (
            f"### {method}\n"
            f"- Translation error: **{float(result['camera_translation_error_m']):.3f} m**\n"
            f"- Rotation error: **{float(result['camera_rotation_error_deg']):.2f} deg**\n"
            f"- Yaw / pitch / roll: **{float(result['yaw_error_deg']):.2f} / "
            f"{float(result['pitch_error_deg']):.2f} / {float(result['roll_error_deg']):.2f} deg**\n"
            f"- Bounded residual: **{residual_text}**\n"
            f"- Gain over Fixed: **{rotation_gain:+.2f} deg R, {translation_gain:+.3f} m T**"
        )

    def render_scene(self) -> None:
        with self.lock:
            case = self.selected_case()
            method = str(self.method_gui.value)
            frames = VIEW_FRAMES[str(self.view_gui.value)]
            result = method_result(case, method)
            transform = np.asarray(result["transform"], dtype=np.float32)
            oracle = np.asarray(
                method_result(case, "Boundary Oracle")["transform"], dtype=np.float32
            )
            candidate_color = CAMERA_COLORS[method]
            transformed_clouds = []
            candidate_poses = []
            oracle_poses = []

            with self.server.atomic():
                self.clear_scene()
                for frame in frames:
                    frame_transform = self.frame_transform(transform, frame)
                    points = transform_points(frame_transform, case.points_world[frame])
                    vertices = transform_points(
                        frame_transform, case.smpl_vertices_world[frame]
                    )
                    pose = frame_transform @ case.camera_poses[frame]
                    oracle_pose = self.frame_transform(oracle, frame) @ case.camera_poses[frame]
                    transformed_clouds.append(points)
                    candidate_poses.append(pose)
                    oracle_poses.append(oracle_pose)

                    point_handle = self.server.scene.add_point_cloud(
                        f"/comparison/pointmap/frame_{frame}",
                        points=points,
                        colors=case.point_colors[frame],
                        point_size=float(self.point_size_gui.value),
                        point_shape="rounded",
                        precision="float32",
                        visible=bool(self.show_points_gui.value),
                    )
                    self.handles["pointcloud"].append(point_handle)

                    mesh_handle = self.server.scene.add_mesh_simple(
                        f"/comparison/smplx/frame_{frame}",
                        vertices=vertices,
                        faces=self.smpl_faces,
                        color=PRE_HUMAN_COLOR if frame < self.args.boundary else POST_HUMAN_COLOR,
                        flat_shading=False,
                        side="double",
                        opacity=0.92,
                        visible=bool(self.show_smpl_gui.value),
                    )
                    self.handles["smpl"].append(mesh_handle)

                    camera_color = PRE_CAMERA_COLOR if frame < self.args.boundary else candidate_color
                    self.add_camera(
                        f"/comparison/cameras/frame_{frame}",
                        pose,
                        case.intrinsics[frame],
                        case.images[frame],
                        camera_color,
                        "camera",
                        0.14,
                    )
                    label = self.server.scene.add_label(
                        f"/comparison/labels/frame_{frame}",
                        text=f"frame {frame} {'pre' if frame < self.args.boundary else 'post'}",
                        position=pose[:3, 3],
                        anchor="bottom-center",
                        visible=bool(self.show_cameras_gui.value),
                    )
                    self.handles["label"].append(label)

                    if frame >= self.args.boundary:
                        self.add_camera(
                            f"/comparison/oracle_cameras/frame_{frame}",
                            oracle_pose,
                            case.intrinsics[frame],
                            None,
                            ORACLE_CAMERA_COLOR,
                            "oracle_camera",
                            0.17,
                        )

                candidate_centers = np.stack([pose[:3, 3] for pose in candidate_poses])
                candidate_segments = self.line_segments(candidate_centers)
                if len(candidate_segments):
                    handle = self.server.scene.add_line_segments(
                        "/comparison/candidate_trajectory",
                        candidate_segments,
                        colors=candidate_color,
                        line_width=3.0,
                        visible=bool(self.show_cameras_gui.value),
                    )
                    self.handles["trajectory"].append(handle)

                oracle_centers = np.stack([pose[:3, 3] for pose in oracle_poses])
                oracle_segments = self.line_segments(oracle_centers)
                if len(oracle_segments):
                    handle = self.server.scene.add_line_segments(
                        "/comparison/oracle_trajectory",
                        oracle_segments,
                        colors=ORACLE_CAMERA_COLOR,
                        line_width=2.0,
                        visible=bool(self.show_oracle_gui.value),
                    )
                    self.handles["oracle_camera"].append(handle)

                self.set_category_visibility("camera", self.show_cameras_gui.value)
                self.set_category_visibility("oracle_camera", self.show_oracle_gui.value)
                all_points = np.concatenate(transformed_clouds, axis=0)
                finite = all_points[np.isfinite(all_points).all(axis=1)]
                low = np.quantile(finite, 0.02, axis=0)
                high = np.quantile(finite, 0.98, axis=0)
                self.scene_center = ((low + high) * 0.5).astype(np.float32)
                self.scene_radius = max(float(np.linalg.norm(high - low)), 1.5)

            self.pre_image_gui.image = case.images[1]
            self.post_image_gui.image = case.images[2]
            self.update_metrics(case, result, method)

    def reset_client_camera(self, client: viser.ClientHandle) -> None:
        center = self.scene_center.astype(np.float32)
        radius = float(self.scene_radius)
        client.camera.up_direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        client.camera.look_at = center
        client.camera.position = center + np.asarray(
            [radius * 0.72, -radius * 0.48, radius * 0.82], dtype=np.float32
        )

    def run(self) -> None:
        print(f"Boundary comparison viewer: http://127.0.0.1:{self.args.port}", flush=True)
        while True:
            time.sleep(10.0)


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SMPL-X cache preparation but is unavailable")
    device = torch.device(args.device)
    v10_cases, candidate_cases = load_case_maps(args)
    layer = build_smpl_layer(device)
    cache: dict[str, CachedCase] = {}
    for label, case_name in EXAMPLES.items():
        if case_name not in v10_cases or case_name not in candidate_cases:
            raise KeyError(f"Missing cached experiment case: {case_name}")
        print(f"Preparing interactive scene: {label}", flush=True)
        cache[label] = load_cached_case(
            case_name,
            v10_cases[case_name],
            candidate_cases[case_name],
            layer,
            device,
            args,
        )
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    del layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    viewer = ComparisonViewer(cache, faces, args)
    viewer.run()


if __name__ == "__main__":
    main()
