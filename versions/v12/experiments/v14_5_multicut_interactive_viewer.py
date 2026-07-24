#!/usr/bin/env python3
"""Demo-style 3D viewer for true recurrent 1/2/4/8-cut rollouts."""

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


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v12.experiments.v14_3_human_continuity_visualization import (  # noqa: E402
    body_batch,
    camera_points,
    load_frame,
    transform_points,
)
from boundary_shot_scale_support import scale_pose  # noqa: E402
from versions.v12.experiments.v14_3_interactive_continuity_viewer import (  # noqa: E402
    line_segments,
    skeleton_segments,
)


DEFAULT_REPORT = (
    ROOT
    / "output/v14_5_final_audit/true_recurrent_multicut"
    / "v14_5_true_recurrent_multicut.json"
)
DEFAULT_CASE_ROOT = ROOT / "output/v14_5_final_audit/true_recurrent_multicut/cases"
METHODS = (
    "hard_reset_fixed",
    "v11_1",
    "v11_4",
    "unified_da3",
)
METHOD_LABELS = (
    "Hard Reset + Fixed",
    "V11.1 Raw Scale",
    "V11.4 Uniform Similarity (historical VGGT cache)",
    "Unified DA3",
)
METHOD_COLORS = (
    (128, 135, 142),
    (219, 107, 52),
    (28, 146, 161),
    (43, 132, 83),
)
PREFIX_OPTIONS = (
    "1 cut (short)",
    "2 cuts (short)",
    "4 cuts",
    "8 cuts (stress test)",
)
PREFIX_COUNTS = dict(zip(PREFIX_OPTIONS, (1, 2, 4, 8)))
VIEW_FOCUS_OPTIONS = (
    "Current human",
    "Current shot",
    "Full rollout",
)
CASE_ROLES = {
    "avatarrex": "short-term gain, moderate 8-cut drift",
    "thuman": "camera gain, human-root drift after repeated cuts",
    "mvhuman100": "clear 8-cut failure case",
    "mvhuman200": "camera gain with scene trade-off",
}


@dataclass
class MultiCutSequence:
    name: str
    source: str
    cuts: list[int]
    images: list[np.ndarray]
    intrinsics: np.ndarray
    local_poses: np.ndarray
    points_local: list[np.ndarray]
    point_colors: list[np.ndarray]
    joints_camera: np.ndarray
    vertices_camera: np.ndarray
    report_case: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--case_root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--port", type=int, default=8107)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--point_stride", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--point_size", type=float, default=0.005)
    parser.add_argument("--camera_scale", type=float, default=0.14)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def load_multicut_sequence(
    case: dict,
    case_root: Path,
    layer: SMPL_Layer,
    device: torch.device,
    args: argparse.Namespace,
) -> MultiCutSequence:
    name = str(case["case_name"])
    local_dir = case_root / name / "human3r_true_reset"
    frame_count = int(case["methods"]["v11_4"]["prefixes"]["8"]["final_frame"]) + 1
    frames = [load_frame(local_dir, index) for index in range(frame_count)]
    intrinsics = np.stack([frame["K"] for frame in frames]).astype(np.float32)
    rotvecs = np.stack([frame["rotvec"] for frame in frames]).astype(np.float32)
    shapes = np.stack([frame["shape"] for frame in frames]).astype(np.float32)
    translations = np.stack([frame["transl"] for frame in frames]).astype(np.float32)
    expressions = np.stack([frame["expression"] for frame in frames]).astype(np.float32)
    joints, vertices = body_batch(
        layer,
        rotvecs,
        shapes,
        translations,
        expressions,
        intrinsics,
        device,
    )

    points_local: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    for frame in frames:
        depth = frame["depth"]
        confidence = frame["confidence"]
        mask = frame["mask"]
        if confidence.shape != depth.shape:
            confidence = cv2.resize(
                confidence,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if int(args.mask_dilate) > 1:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        points_camera = camera_points(depth, frame["K"])
        valid = (
            np.isfinite(points_camera).all(axis=-1)
            & np.isfinite(confidence)
            & (depth > 0.05)
            & (depth < 30.0)
            & (confidence > float(args.confidence_threshold))
            & (mask == 0)
        )
        ids = np.flatnonzero(valid.reshape(-1))[:: max(int(args.point_stride), 1)]
        points_local.append(
            transform_points(frame["pose"], points_camera.reshape(-1, 3)[ids])
        )
        point_colors.append(frame["image"].reshape(-1, 3)[ids].astype(np.uint8))

    return MultiCutSequence(
        name=name,
        source=str(case["source"]),
        cuts=[int(value) for value in case["cuts"]],
        images=[frame["image"] for frame in frames],
        intrinsics=intrinsics,
        local_poses=np.stack([frame["pose"] for frame in frames]).astype(np.float32),
        points_local=points_local,
        point_colors=point_colors,
        joints_camera=joints,
        vertices_camera=vertices,
        report_case=case,
    )


def current_state(sequence: MultiCutSequence, method_key: str, frame: int) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(sequence.report_case["shot_scales"]["0"]["scene_scale"])
    gauge = np.eye(4, dtype=np.float32)
    root_delta = np.zeros(3, dtype=np.float32)
    state_rows = sequence.report_case["methods"][method_key]["shot_state"]
    reached = [cut for cut in sequence.cuts if cut <= frame]
    if reached:
        state = state_rows[str(reached[-1])]
        scale = float(state["scale"])
        gauge = np.asarray(state["gauge"], dtype=np.float32)
        root_delta = np.asarray(state["root_delta"], dtype=np.float32)
    return scale, gauge, root_delta


def method_geometry(sequence: MultiCutSequence, method_key: str, frame: int) -> dict:
    scale, gauge, root_delta = current_state(sequence, method_key, frame)
    camera_pose = (gauge @ scale_pose(sequence.local_poses[frame], scale)).astype(np.float32)
    raw_root = sequence.joints_camera[frame, 0]
    root_camera = raw_root * scale + root_delta
    root_world = camera_pose[:3, :3] @ root_camera + camera_pose[:3, 3]
    joints_centered = (sequence.joints_camera[frame] - raw_root) * scale
    vertices_centered = (sequence.vertices_camera[frame] - raw_root) * scale
    joints_world = joints_centered @ camera_pose[:3, :3].T + root_world
    vertices_world = vertices_centered @ camera_pose[:3, :3].T + root_world
    points_world = transform_points(gauge, sequence.points_local[frame] * scale)
    return {
        "camera_pose": camera_pose,
        "root_world": root_world.astype(np.float32),
        "joints_world": joints_world.astype(np.float32),
        "vertices_world": vertices_world.astype(np.float32),
        "points_world": points_world.astype(np.float32),
        "scale": scale,
        "root_delta": root_delta,
    }


def case_label(case: dict) -> str:
    source = str(case["source"])
    role = CASE_ROLES.get(source, "representative rollout")
    return f"{source} | {role}"


def prefix_metrics(method: dict, count: int) -> dict:
    return method["prefixes"][str(count)]


class MultiCutViewer:
    def __init__(
        self,
        sequences: list[MultiCutSequence],
        faces: np.ndarray,
        args: argparse.Namespace,
    ) -> None:
        self.args = args
        self.faces = np.asarray(faces, dtype=np.int32)
        self.sequences = {sequence.name: sequence for sequence in sequences}
        self.case_options = {case_label(sequence.report_case): sequence.name for sequence in sequences}
        self.method_options = dict(zip(METHOD_LABELS, METHODS))
        self.geometry_cache = {
            sequence.name: {
                method: [
                    method_geometry(sequence, method, frame)
                    for frame in range(len(sequence.images))
                ]
                for method in METHODS
            }
            for sequence in sequences
        }
        self.lock = threading.RLock()
        self.static_handles: list = []
        self.frame_handles: list = []
        self.scene_center = np.zeros(3, dtype=np.float32)
        self.scene_radius = 3.0

        self.server = viser.ViserServer(
            host="0.0.0.0",
            port=int(args.port),
            label="V14.5 True Recurrent Multi-Cut",
        )
        self.server.scene.set_up_direction("-y")
        self.server.scene.add_frame(
            "/world", show_axes=True, axes_length=0.35, axes_radius=0.012
        )

        default_case = next(
            label for label, name in self.case_options.items() if self.sequences[name].source == "thuman"
        )
        with self.server.gui.add_folder("Rollout"):
            self.case_gui = self.server.gui.add_dropdown(
                "Sequence", tuple(self.case_options), initial_value=default_case
            )
            self.prefix_gui = self.server.gui.add_dropdown(
                "Visible prefix", PREFIX_OPTIONS, initial_value=PREFIX_OPTIONS[1]
            )
            self.frame_gui = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=17,
                step=1,
                initial_value=5,
                marks=((0, "start"), (2, "C1"), (4, "C2"), (8, "C4"), (16, "C8")),
            )
            self.previous_gui = self.server.gui.add_button("Previous frame")
            self.next_gui = self.server.gui.add_button("Next frame")
            self.play_gui = self.server.gui.add_checkbox("Play", False)
            self.fps_gui = self.server.gui.add_slider(
                "FPS", min=1.0, max=8.0, step=0.5, initial_value=float(args.fps)
            )

        with self.server.gui.add_folder("Method comparison"):
            self.method_gui = self.server.gui.add_dropdown(
                "Solid method", METHOD_LABELS, initial_value=METHOD_LABELS[2]
            )
            self.comparison_gui = self.server.gui.add_dropdown(
                "Wireframe method", METHOD_LABELS, initial_value=METHOD_LABELS[0]
            )
            self.show_comparison_gui = self.server.gui.add_checkbox(
                "Show wireframe comparison", True
            )
            self.show_previous_gui = self.server.gui.add_checkbox(
                "Show previous-frame body", True
            )
            self.show_human_gui = self.server.gui.add_checkbox("Show SMPL-X", True)
            self.show_joints_gui = self.server.gui.add_checkbox("Show joints", True)

        with self.server.gui.add_folder("World geometry"):
            self.show_scene_gui = self.server.gui.add_checkbox(
                "Show accumulated scene", True
            )
            self.show_comparison_scene_gui = self.server.gui.add_checkbox(
                "Show comparison scene", False
            )
            self.show_current_points_gui = self.server.gui.add_checkbox(
                "Show current pointmaps", False
            )
            self.show_camera_gui = self.server.gui.add_checkbox(
                "Show camera trajectories", True
            )
            self.show_root_gui = self.server.gui.add_checkbox(
                "Show human trajectories", True
            )
            self.camera_scale_gui = self.server.gui.add_slider(
                "Camera size",
                min=0.05,
                max=0.40,
                step=0.01,
                initial_value=float(args.camera_scale),
            )
            self.view_focus_gui = self.server.gui.add_button_group(
                "View focus", VIEW_FOCUS_OPTIONS
            )
            self.reset_gui = self.server.gui.add_button("Center 3D view")

        self.metrics_gui = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Current RGB"):
            self.image_gui = self.server.gui.add_image(
                self.sequence.images[int(self.frame_gui.value)], label="Current frame"
            )

        self._register_callbacks()
        self._rebuild_all()

    @property
    def sequence(self) -> MultiCutSequence:
        return self.sequences[self.case_options[str(self.case_gui.value)]]

    def method_key(self) -> str:
        return self.method_options[str(self.method_gui.value)]

    def comparison_key(self) -> str:
        return self.method_options[str(self.comparison_gui.value)]

    def prefix_count(self) -> int:
        return PREFIX_COUNTS[str(self.prefix_gui.value)]

    def final_frame(self) -> int:
        row = prefix_metrics(
            self.sequence.report_case["methods"][self.method_key()], self.prefix_count()
        )
        return int(row["final_frame"])

    def geometries(self, method: str) -> list[dict]:
        return self.geometry_cache[self.sequence.name][method]

    @staticmethod
    def _remove_handles(handles: list) -> None:
        for handle in handles:
            try:
                handle.remove()
            except (AttributeError, KeyError):
                pass
        handles.clear()

    def _rebuild_all(self, *, refocus: bool = False) -> None:
        with self.lock, self.server.atomic():
            self._remove_handles(self.frame_handles)
            self._remove_handles(self.static_handles)
            limit = self.final_frame() + 1
            method = self.method_key()
            comparison = self.comparison_key()
            selected = self.geometries(method)[:limit]
            compared = self.geometries(comparison)[:limit]

            selected_points = np.concatenate([row["points_world"] for row in selected])
            selected_colors = np.concatenate(self.sequence.point_colors[:limit])
            comparison_points = np.concatenate([row["points_world"] for row in compared])
            scene = self.server.scene.add_point_cloud(
                "/rollout/scene/selected",
                points=selected_points,
                colors=selected_colors,
                point_size=float(self.args.point_size),
                point_shape="rounded",
                precision="float32",
                visible=bool(self.show_scene_gui.value),
            )
            comparison_scene = self.server.scene.add_point_cloud(
                "/rollout/scene/comparison",
                points=comparison_points,
                colors=np.tile(np.asarray((210, 68, 88), dtype=np.uint8), (len(comparison_points), 1)),
                point_size=float(self.args.point_size) * 0.72,
                point_shape="circle",
                precision="float32",
                visible=bool(self.show_comparison_scene_gui.value),
            )
            self.static_handles.extend((scene, comparison_scene))

            selected_cameras = np.stack([row["camera_pose"][:3, 3] for row in selected])
            comparison_cameras = np.stack([row["camera_pose"][:3, 3] for row in compared])
            selected_roots = np.stack([row["root_world"] for row in selected])
            comparison_roots = np.stack([row["root_world"] for row in compared])
            selected_camera_line = self.server.scene.add_line_segments(
                "/rollout/trajectory/camera_selected",
                points=line_segments(selected_cameras),
                colors=METHOD_COLORS[METHODS.index(method)],
                line_width=4.0,
                visible=bool(self.show_camera_gui.value),
            )
            comparison_camera_line = self.server.scene.add_line_segments(
                "/rollout/trajectory/camera_comparison",
                points=line_segments(comparison_cameras),
                colors=METHOD_COLORS[METHODS.index(comparison)],
                line_width=2.0,
                visible=bool(self.show_camera_gui.value and self.show_comparison_gui.value),
            )
            selected_root_line = self.server.scene.add_line_segments(
                "/rollout/trajectory/root_selected",
                points=line_segments(selected_roots),
                colors=METHOD_COLORS[METHODS.index(method)],
                line_width=6.0,
                visible=bool(self.show_root_gui.value),
            )
            comparison_root_line = self.server.scene.add_line_segments(
                "/rollout/trajectory/root_comparison",
                points=line_segments(comparison_roots),
                colors=METHOD_COLORS[METHODS.index(comparison)],
                line_width=2.0,
                visible=bool(self.show_root_gui.value and self.show_comparison_gui.value),
            )
            self.static_handles.extend(
                (
                    selected_camera_line,
                    comparison_camera_line,
                    selected_root_line,
                    comparison_root_line,
                )
            )

            reached_cuts = [cut for cut in self.sequence.cuts if cut < limit]
            cut_points = np.stack([selected[cut]["camera_pose"][:3, 3] for cut in reached_cuts])
            cut_cloud = self.server.scene.add_point_cloud(
                "/rollout/cuts",
                points=cut_points,
                colors=(221, 53, 60),
                point_size=0.055,
                point_shape="circle",
                precision="float32",
                visible=True,
            )
            self.static_handles.append(cut_cloud)
            for index, (cut, position) in enumerate(zip(reached_cuts, cut_points), start=1):
                label = self.server.scene.add_label(
                    f"/rollout/cut_labels/{index}",
                    text=f"CUT {index} | frame {cut}",
                    position=position,
                    anchor="bottom-center",
                )
                self.static_handles.append(label)

            if int(self.frame_gui.value) > self.final_frame():
                self.frame_gui.value = self.final_frame()
            self._render_frame(int(self.frame_gui.value))

        if refocus:
            self._refocus_clients()

    @staticmethod
    def _robust_bounds(points: np.ndarray, quantile: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        finite = np.asarray(points, dtype=np.float32)
        finite = finite[np.isfinite(finite).all(axis=1)]
        if len(finite) == 0:
            return np.full(3, -1.0, dtype=np.float32), np.full(3, 1.0, dtype=np.float32)
        low = np.quantile(finite, quantile, axis=0)
        high = np.quantile(finite, 1.0 - quantile, axis=0)
        return low.astype(np.float32), high.astype(np.float32)

    def _view_target(self) -> tuple[np.ndarray, float]:
        frame = int(np.clip(self.frame_gui.value, 0, self.final_frame()))
        geometries = self.geometries(self.method_key())
        current = geometries[frame]
        focus = str(self.view_focus_gui.value)

        if focus == "Current human":
            low, high = self._robust_bounds(current["vertices_world"])
            center = (low + high) * 0.5
            body_diagonal = float(np.linalg.norm(high - low))
            return center.astype(np.float32), max(body_diagonal * 1.05, 1.65)

        if focus == "Current shot":
            shot_start = max((cut for cut in self.sequence.cuts if cut <= frame), default=0)
            shot_end = min(
                next((cut for cut in self.sequence.cuts if cut > frame), self.final_frame() + 1),
                self.final_frame() + 1,
            )
            rows = geometries[shot_start:shot_end]
            anchors = np.concatenate(
                (
                    np.stack([row["root_world"] for row in rows]),
                    np.stack([row["camera_pose"][:3, 3] for row in rows]),
                    current["vertices_world"],
                ),
                axis=0,
            )
            low, high = self._robust_bounds(anchors)
            center = (low + high) * 0.5
            return center.astype(np.float32), max(float(np.linalg.norm(high - low)) * 0.85, 2.0)

        rows = geometries[: self.final_frame() + 1]
        rollout = np.concatenate(
            (
                np.concatenate([row["points_world"] for row in rows]),
                np.stack([row["root_world"] for row in rows]),
                np.stack([row["camera_pose"][:3, 3] for row in rows]),
            ),
            axis=0,
        )
        low, high = self._robust_bounds(rollout, quantile=0.03)
        center = (low + high) * 0.5
        return center.astype(np.float32), max(float(np.linalg.norm(high - low)) * 0.72, 2.5)

    def _refocus_clients(self) -> None:
        for client in self.server.get_clients().values():
            self.reset_camera(client)

    def _render_frame(self, frame: int) -> None:
        self._remove_handles(self.frame_handles)
        frame = int(np.clip(frame, 0, self.final_frame()))
        method = self.method_key()
        comparison = self.comparison_key()
        selected = self.geometries(method)[frame]
        compared = self.geometries(comparison)[frame]
        selected_color = METHOD_COLORS[METHODS.index(method)]
        comparison_color = METHOD_COLORS[METHODS.index(comparison)]
        prefix = "/rollout/current"

        selected_points = self.server.scene.add_point_cloud(
            f"{prefix}/pointmap_selected",
            points=selected["points_world"],
            colors=self.sequence.point_colors[frame],
            point_size=float(self.args.point_size) * 1.4,
            point_shape="rounded",
            precision="float32",
            visible=bool(self.show_current_points_gui.value),
        )
        comparison_points = self.server.scene.add_point_cloud(
            f"{prefix}/pointmap_comparison",
            points=compared["points_world"],
            colors=comparison_color,
            point_size=float(self.args.point_size),
            point_shape="circle",
            precision="float32",
            visible=bool(
                self.show_current_points_gui.value and self.show_comparison_gui.value
            ),
        )
        self.frame_handles.extend((selected_points, comparison_points))

        image = self.sequence.images[frame]
        K = self.sequence.intrinsics[frame]
        height, width = image.shape[:2]
        fov = 2.0 * np.arctan((height * 0.5) / max(float(K[1, 1]), 1e-6))
        for label, geometry, color, line_width in (
            ("selected", selected, selected_color, 2.5),
            ("comparison", compared, comparison_color, 1.3),
        ):
            visible = bool(
                self.show_camera_gui.value
                and (label == "selected" or self.show_comparison_gui.value)
            )
            pose = geometry["camera_pose"]
            camera = self.server.scene.add_camera_frustum(
                f"{prefix}/camera_{label}",
                fov=float(fov),
                aspect=float(width / max(height, 1)),
                scale=float(self.camera_scale_gui.value),
                line_width=line_width,
                color=color,
                image=image if label == "selected" else None,
                jpeg_quality=80,
                wxyz=tf.SO3.from_matrix(pose[:3, :3]).wxyz,
                position=pose[:3, 3],
                visible=visible,
            )
            self.frame_handles.append(camera)

        selected_mesh = self.server.scene.add_mesh_simple(
            f"{prefix}/human_selected",
            vertices=selected["vertices_world"],
            faces=self.faces,
            color=selected_color,
            opacity=0.82,
            flat_shading=False,
            side="double",
            visible=bool(self.show_human_gui.value),
        )
        comparison_mesh = self.server.scene.add_mesh_simple(
            f"{prefix}/human_comparison",
            vertices=compared["vertices_world"],
            faces=self.faces,
            color=comparison_color,
            wireframe=True,
            opacity=0.72,
            flat_shading=False,
            side="double",
            visible=bool(self.show_human_gui.value and self.show_comparison_gui.value),
        )
        self.frame_handles.extend((selected_mesh, comparison_mesh))
        if frame > 0:
            previous = self.geometries(method)[frame - 1]
            previous_mesh = self.server.scene.add_mesh_simple(
                f"{prefix}/human_previous",
                vertices=previous["vertices_world"],
                faces=self.faces,
                color=(110, 116, 122),
                wireframe=True,
                opacity=0.25,
                flat_shading=False,
                side="double",
                visible=bool(self.show_human_gui.value and self.show_previous_gui.value),
            )
            self.frame_handles.append(previous_mesh)

        joints = selected["joints_world"][:22]
        joint_cloud = self.server.scene.add_point_cloud(
            f"{prefix}/joints",
            points=joints,
            colors=selected_color,
            point_size=0.028,
            point_shape="circle",
            precision="float32",
            visible=bool(self.show_joints_gui.value),
        )
        skeleton = self.server.scene.add_line_segments(
            f"{prefix}/skeleton",
            points=skeleton_segments(joints),
            colors=selected_color,
            line_width=3.0,
            visible=bool(self.show_joints_gui.value),
        )
        self.frame_handles.extend((joint_cloud, skeleton))
        self.image_gui.image = image
        self._update_metrics(frame, selected, compared)

    def _update_metrics(self, frame: int, selected: dict, compared: dict) -> None:
        count = self.prefix_count()
        method = self.method_key()
        comparison = self.comparison_key()
        selected_prefix = prefix_metrics(
            self.sequence.report_case["methods"][method], count
        )
        comparison_prefix = prefix_metrics(
            self.sequence.report_case["methods"][comparison], count
        )
        selected_frame = self.sequence.report_case["methods"][method]["per_frame"][frame]
        comparison_frame = self.sequence.report_case["methods"][comparison]["per_frame"][frame]
        root_delta = float(np.linalg.norm(selected["root_world"] - compared["root_world"]))
        mesh_delta = float(
            np.mean(
                np.linalg.norm(
                    selected["vertices_world"] - compared["vertices_world"], axis=1
                )
            )
        )
        current_cut = sum(cut <= frame for cut in self.sequence.cuts)

        rows = []
        for label, key in zip(METHOD_LABELS, METHODS):
            metric = prefix_metrics(self.sequence.report_case["methods"][key], count)
            rows.append(
                f"| {label} | {metric['camera_cumulative_drift_m']:.3f} | "
                f"{metric['camera_cumulative_rotation_deg']:.1f} | "
                f"{metric['human_root_cumulative_drift_m']:.3f} | "
                f"{metric['human_joint_cumulative_drift_m']:.3f} | "
                f"{metric['scene_discontinuity_m']['mean']:.3f} |"
            )
        table = "\n".join(rows)
        selected_name = METHOD_LABELS[METHODS.index(method)]
        comparison_name = METHOD_LABELS[METHODS.index(comparison)]
        horizon = "SHORT-SHOT RANGE" if count <= 2 else "ACCUMULATION STRESS TEST"
        self.metrics_gui.content = (
            f"### {self.sequence.source} · {self.sequence.name}\n"
            f"**{count} cut prefix · {horizon} · frame {frame}/"
            f"{self.final_frame()} · reached cuts {current_cut}**\n\n"
            f"- Solid: **{selected_name}**; wireframe: **{comparison_name}**\n"
            f"- Current camera error: **{selected_frame['camera_translation_m']:.3f} m / "
            f"{selected_frame['camera_rotation_deg']:.1f} deg** "
            f"(wireframe {comparison_frame['camera_translation_m']:.3f} m / "
            f"{comparison_frame['camera_rotation_deg']:.1f} deg)\n"
            f"- Current human root/joints error: **{selected_frame['human_root_m']:.3f} / "
            f"{selected_frame['human_joints_m']:.3f} m**\n"
            f"- Solid-vs-wire body difference: root **{root_delta:.3f} m**, "
            f"mesh mean **{mesh_delta:.3f} m**\n"
            f"- Current shot scale: **{selected['scale']:.4f}**; root correction "
            f"**{float(np.linalg.norm(selected['root_delta'])):.3f} m**\n\n"
            f"#### Prefix endpoint: {selected_name} vs {comparison_name}\n"
            f"- Camera translation: **{selected_prefix['camera_cumulative_drift_m']:.3f} vs "
            f"{comparison_prefix['camera_cumulative_drift_m']:.3f} m**\n"
            f"- Camera rotation: **{selected_prefix['camera_cumulative_rotation_deg']:.1f} vs "
            f"{comparison_prefix['camera_cumulative_rotation_deg']:.1f} deg**\n"
            f"- Human root: **{selected_prefix['human_root_cumulative_drift_m']:.3f} vs "
            f"{comparison_prefix['human_root_cumulative_drift_m']:.3f} m**\n"
            f"- Scene discontinuity: **{selected_prefix['scene_discontinuity_m']['mean']:.3f} vs "
            f"{comparison_prefix['scene_discontinuity_m']['mean']:.3f} m**\n\n"
            "#### All frozen methods at this prefix\n"
            "| Method | Camera m | Rotation deg | Root m | Joints m | Scene m |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            f"{table}\n\n"
            "**Scope:** V11.4 is evaluated as short-shot re-anchoring. The 4/8-cut "
            "views expose accumulated drift and are not evidence of unlimited-horizon stability."
        )

    def _set_frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.final_frame()))
        if int(self.frame_gui.value) != frame:
            self.frame_gui.value = frame
            return
        with self.lock, self.server.atomic():
            self._render_frame(frame)

    def _register_callbacks(self) -> None:
        @self.case_gui.on_update
        def _(_) -> None:
            self.frame_gui.value = min(int(self.frame_gui.value), self.final_frame())
            self._rebuild_all(refocus=True)

        @self.prefix_gui.on_update
        def _(_) -> None:
            self.frame_gui.value = self.final_frame()
            self._rebuild_all(refocus=True)

        @self.method_gui.on_update
        def _(_) -> None:
            self._rebuild_all(refocus=True)

        @self.comparison_gui.on_update
        def _(_) -> None:
            self._rebuild_all()

        @self.view_focus_gui.on_click
        def _(_) -> None:
            self._refocus_clients()

        @self.frame_gui.on_update
        def _(_) -> None:
            self._set_frame(int(self.frame_gui.value))

        @self.previous_gui.on_click
        def _(_) -> None:
            self.frame_gui.value = (int(self.frame_gui.value) - 1) % (self.final_frame() + 1)

        @self.next_gui.on_click
        def _(_) -> None:
            self.frame_gui.value = (int(self.frame_gui.value) + 1) % (self.final_frame() + 1)

        for handle in (
            self.show_comparison_gui,
            self.show_previous_gui,
            self.show_human_gui,
            self.show_joints_gui,
            self.show_scene_gui,
            self.show_comparison_scene_gui,
            self.show_current_points_gui,
            self.show_camera_gui,
            self.show_root_gui,
            self.camera_scale_gui,
        ):
            handle.on_update(lambda _: self._rebuild_all())

        @self.reset_gui.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.reset_camera(event.client)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.reset_camera(client)

    def reset_camera(self, client: viser.ClientHandle) -> None:
        center, radius = self._view_target()
        self.scene_center = center
        self.scene_radius = radius
        up = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        frame = int(np.clip(self.frame_gui.value, 0, self.final_frame()))
        current = self.geometries(self.method_key())[frame]
        camera_to_human = current["root_world"] - current["camera_pose"][:3, 3]
        norm = float(np.linalg.norm(camera_to_human))
        forward = (
            camera_to_human / norm
            if norm > 1e-5
            else np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        )
        right = np.cross(forward, up)
        right_norm = float(np.linalg.norm(right))
        right = (
            right / right_norm
            if right_norm > 1e-5
            else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        )
        client.camera.position = center - forward * radius + right * radius * 0.55 + up * radius * 0.18
        client.camera.look_at = center
        client.camera.up_direction = up
        client.camera.fov = float(np.deg2rad(50.0))

    def run(self) -> None:
        print(f">> V14.5 multi-cut viewer: http://127.0.0.1:{self.args.port}", flush=True)
        while True:
            if bool(self.play_gui.value):
                self.frame_gui.value = (int(self.frame_gui.value) + 1) % (
                    self.final_frame() + 1
                )
            time.sleep(1.0 / max(float(self.fps_gui.value), 0.5))


def build_sequences(args: argparse.Namespace) -> tuple[list[MultiCutSequence], np.ndarray]:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("SMPL-X preparation must run on the requested CUDA device")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    sequences = []
    for index, case in enumerate(report["cases"], start=1):
        print(
            f">> Preparing recurrent viewer sequence {index}/{len(report['cases'])} "
            f"on {device}: {case['case_name']}",
            flush=True,
        )
        sequences.append(load_multicut_sequence(case, args.case_root, layer, device, args))
    del layer
    torch.cuda.empty_cache()
    return sequences, faces


def validate(sequences: list[MultiCutSequence]) -> dict:
    output = {}
    for sequence in sequences:
        methods = {}
        for method in METHODS:
            geometries = [
                method_geometry(sequence, method, frame)
                for frame in range(len(sequence.images))
            ]
            vertices = np.concatenate([row["vertices_world"] for row in geometries])
            points = np.concatenate([row["points_world"] for row in geometries])
            cameras = np.stack([row["camera_pose"] for row in geometries])
            methods[method] = {
                "frames": len(geometries),
                "finite_vertices": bool(np.isfinite(vertices).all()),
                "finite_points": bool(np.isfinite(points).all()),
                "finite_cameras": bool(np.isfinite(cameras).all()),
                "vertex_extent_m": np.ptp(vertices, axis=0).astype(float).tolist(),
                "point_extent_m": np.ptp(points, axis=0).astype(float).tolist(),
            }
        output[sequence.name] = {
            "source": sequence.source,
            "cuts": sequence.cuts,
            "methods": methods,
        }
    return output


def main() -> None:
    args = parse_args()
    sequences, faces = build_sequences(args)
    validation = validate(sequences)
    if args.validate_only:
        print(json.dumps(validation, indent=2, ensure_ascii=False), flush=True)
        return
    print(
        ">> GPU body preparation complete; playback uses cached CPU geometry",
        flush=True,
    )
    MultiCutViewer(sequences, faces, args).run()


if __name__ == "__main__":
    main()
