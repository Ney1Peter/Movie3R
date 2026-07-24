#!/usr/bin/env python3
"""Interactive demo.py-style 3D viewer for V14.3 human continuity.

The viewer consumes completed V14.3 long-sequence caches.  SMPL-X bodies are
constructed once on CUDA during startup; no Human3R/DA3 inference or per-frame
Boundary solve is performed.  The selected Boundary transform is fixed for the
entire post-cut shot.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as tf


ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v12.experiments.v14_3_human_continuity_visualization import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_QUANTITATIVE,
    DEFAULT_V14_2,
    METHOD_LABELS,
    load_sequence,
    mean_projected_displacement,
    method_geometry,
)


DEFAULT_VISUALIZATION_MANIFEST = (
    ROOT
    / "output/v14_3_projection_consistent_reanchoring/visualization_da3"
    / "visualization_manifest.json"
)
DEFAULT_CASE = "thuman_150_180_thuman00_2180_cam02_cam16"
DISPLAY_METHODS = (METHOD_LABELS[2], METHOD_LABELS[3])

METHOD_COLORS = (
    (220, 84, 62),
    (26, 147, 166),
    (230, 151, 42),
    (38, 145, 91),
)
PAIR_METHOD = {0: 1, 1: 0, 2: 3, 3: 2}
SKELETON_EDGES = np.asarray(
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12),
        (9, 13),
        (9, 14),
        (12, 15),
        (13, 16),
        (14, 17),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
    ],
    dtype=np.int32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantitative", type=Path, default=DEFAULT_QUANTITATIVE)
    parser.add_argument("--v14_2_report", type=Path, default=DEFAULT_V14_2)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--visualization_manifest", type=Path, default=DEFAULT_VISUALIZATION_MANIFEST
    )
    parser.add_argument("--case", default=None)
    parser.add_argument("--alignment_method", default="v18_coupled_full")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--point_stride", type=int, default=24)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--point_size", type=float, default=0.009)
    parser.add_argument("--camera_scale", type=float, default=0.16)
    return parser.parse_args()


def select_case(args: argparse.Namespace, available: set[str]) -> str:
    if args.case is not None:
        if args.case not in available:
            raise KeyError(f"Long-sequence cache is missing for {args.case}")
        return str(args.case)
    if DEFAULT_CASE in available:
        return DEFAULT_CASE
    if args.visualization_manifest.is_file():
        manifest = json.loads(args.visualization_manifest.read_text(encoding="utf-8"))
        candidates = [row for row in manifest["cases"] if row["case_name"] in available]
        if candidates:
            best = max(
                candidates,
                key=lambda row: float(
                    row["visual_change"]["continuity_mesh_shift_mean_px"]
                ),
            )
            return str(best["case_name"])
    if not available:
        raise RuntimeError(f"No long-sequence cache found under {args.cache_dir}")
    return sorted(available)[0]


def line_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)


def skeleton_segments(joints: np.ndarray) -> np.ndarray:
    usable = SKELETON_EDGES[(SKELETON_EDGES < len(joints)).all(axis=1)]
    return np.stack((joints[usable[:, 0]], joints[usable[:, 1]]), axis=1).astype(
        np.float32
    )


def build_sequence(args: argparse.Namespace):
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("SMPL-X cache construction must run on the requested CUDA device")
    quantitative = json.loads(args.quantitative.read_text(encoding="utf-8"))
    v14_2 = json.loads(args.v14_2_report.read_text(encoding="utf-8"))
    quantitative_map = {row["case_name"]: row for row in quantitative["cases"]}
    v14_map = {row["case_name"]: row for row in v14_2["cases"]}
    available = {path.parent.name for path in args.cache_dir.glob("*/manifest.json")}
    name = select_case(args, available & quantitative_map.keys() & v14_map.keys())

    device = torch.device(args.device)
    print(f">> Building cached SMPL-X sequence on {device}: {name}", flush=True)
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    sequence = load_sequence(
        name,
        args.cache_dir,
        quantitative_map[name],
        v14_map[name],
        layer,
        device,
        args,
    )
    del layer
    torch.cuda.empty_cache()
    print(">> GPU preparation complete; interactive playback now uses cached CPU data", flush=True)
    return sequence, faces


class ContinuityViewer:
    def __init__(self, sequence, faces: np.ndarray, args: argparse.Namespace) -> None:
        self.sequence = sequence
        self.faces = np.asarray(faces, dtype=np.int32)
        self.args = args
        self.frame_count = len(sequence.images)
        self.lock = threading.RLock()
        self.current_frame = min(int(sequence.count), self.frame_count - 1)
        self.geometries = [
            [method_geometry(sequence, method, frame) for frame in range(self.frame_count)]
            for method in range(len(METHOD_LABELS))
        ]

        self.current_points: list = []
        self.cameras: list = []
        self.labels: list = []
        self.selected_meshes: list = []
        self.paired_meshes: list = []
        self.ghost_meshes: list = []
        self.joint_handles: list = []
        self.method_handles: list = []
        self.root_handles: list = []
        self.frame_handles: list = []
        self.scene_center = np.zeros(3, dtype=np.float32)
        self.scene_radius = 3.0

        self.server = viser.ViserServer(
            host="0.0.0.0",
            port=int(args.port),
            label="V14.3 Human Continuity 3D",
        )
        self.server.scene.set_up_direction("-y")
        self.server.scene.add_frame(
            "/world", show_axes=True, axes_length=0.35, axes_radius=0.012
        )

        with self.server.gui.add_folder("Frame playback"):
            self.frame_gui = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=self.frame_count - 1,
                step=1,
                initial_value=self.current_frame,
                marks=(
                    (0, "start"),
                    (sequence.count - 1, "pre"),
                    (sequence.count, "CUT"),
                    (self.frame_count - 1, "end"),
                ),
            )
            self.prev_gui = self.server.gui.add_button("Previous frame")
            self.next_gui = self.server.gui.add_button("Next frame")
            self.play_gui = self.server.gui.add_checkbox("Play", False)
            self.fps_gui = self.server.gui.add_slider(
                "FPS", min=1.0, max=10.0, step=0.5, initial_value=float(args.fps)
            )

        with self.server.gui.add_folder("Human comparison"):
            self.method_gui = self.server.gui.add_dropdown(
                "Current method",
                DISPLAY_METHODS,
                initial_value=METHOD_LABELS[3],
            )
            self.overlay_gui = self.server.gui.add_checkbox(
                "Overlay paired baseline", True
            )
            self.ghost_gui = self.server.gui.add_checkbox(
                "Show previous-frame body", True
            )
            self.show_human_gui = self.server.gui.add_checkbox("Show SMPL-X", True)
            self.show_joints_gui = self.server.gui.add_checkbox("Show joints", True)
            self.show_root_gui = self.server.gui.add_checkbox("Show root trajectory", True)

        with self.server.gui.add_folder("Scene and camera"):
            self.show_scene_gui = self.server.gui.add_checkbox(
                "Show accumulated RGB scene", True
            )
            self.show_current_points_gui = self.server.gui.add_checkbox(
                "Show current-frame pointmap", False
            )
            self.show_camera_gui = self.server.gui.add_checkbox("Show camera", True)
            self.camera_scale_gui = self.server.gui.add_slider(
                "Camera size",
                min=0.05,
                max=0.50,
                step=0.01,
                initial_value=float(args.camera_scale),
            )
            self.reset_gui = self.server.gui.add_button("Reset view")

        self.metrics_gui = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Current RGB"):
            self.image_gui = self.server.gui.add_image(
                sequence.images[self.current_frame], label="Current frame"
            )

        self._build_static_scene()
        self._rebuild_method_geometry()
        self._register_callbacks()

    def method_index(self) -> int:
        return METHOD_LABELS.index(str(self.method_gui.value))

    def _build_static_scene(self) -> None:
        base = self.geometries[0]
        accumulated_points = np.concatenate([item["points_world"] for item in base])
        accumulated_colors = np.concatenate(self.sequence.point_colors)
        self.accumulated_scene = self.server.scene.add_point_cloud(
            "/sequence/accumulated_scene",
            points=accumulated_points,
            colors=accumulated_colors,
            point_size=float(self.args.point_size),
            point_shape="rounded",
            precision="float32",
            visible=bool(self.show_scene_gui.value),
        )

        finite = accumulated_points[np.isfinite(accumulated_points).all(axis=1)]
        low = np.quantile(finite, 0.02, axis=0)
        high = np.quantile(finite, 0.98, axis=0)
        self.scene_center = ((low + high) * 0.5).astype(np.float32)
        self.scene_radius = max(float(np.linalg.norm(high - low)), 1.5)

        camera_centers = np.stack([item["camera_pose"][:3, 3] for item in base])
        self.camera_trajectory = self.server.scene.add_line_segments(
            "/sequence/camera_trajectory",
            points=line_segments(camera_centers),
            colors=(45, 70, 105),
            line_width=2.0,
            visible=bool(self.show_camera_gui.value),
        )

    def _remove_handles(self, handles: list) -> None:
        for handle in handles:
            try:
                handle.remove()
            except (AttributeError, KeyError):
                pass

    def _remove_frame_geometry(self) -> None:
        self._remove_handles(self.frame_handles)
        self.current_points.clear()
        self.cameras.clear()
        self.labels.clear()
        self.selected_meshes.clear()
        self.paired_meshes.clear()
        self.ghost_meshes.clear()
        self.joint_handles.clear()
        self.method_handles.clear()
        self.frame_handles.clear()

    def _remove_method_geometry(self) -> None:
        self._remove_frame_geometry()
        self._remove_handles(self.root_handles)
        self.root_handles.clear()

    def _rebuild_method_geometry(self) -> None:
        with self.lock, self.server.atomic():
            self._remove_method_geometry()
            method = self.method_index()
            paired = PAIR_METHOD[method]
            selected_color = METHOD_COLORS[method]
            paired_color = METHOD_COLORS[paired]

            selected_roots = np.stack(
                [item["root_world"] for item in self.geometries[method]]
            )
            paired_roots = np.stack(
                [item["root_world"] for item in self.geometries[paired]]
            )
            selected_root = self.server.scene.add_line_segments(
                "/sequence/human_root/selected",
                points=line_segments(selected_roots),
                colors=selected_color,
                line_width=5.0,
                visible=bool(self.show_root_gui.value),
            )
            self.root_handles.append(selected_root)
            if float(np.max(np.abs(selected_roots - paired_roots))) > 1e-7:
                paired_root = self.server.scene.add_line_segments(
                    "/sequence/human_root/paired",
                    points=line_segments(paired_roots),
                    colors=paired_color,
                    line_width=2.0,
                    visible=bool(self.show_root_gui.value and self.overlay_gui.value),
                )
                self.root_handles.append(paired_root)

            self._render_current_frame()

    def _render_current_frame(self) -> None:
        self._remove_frame_geometry()
        frame = self.current_frame
        method = self.method_index()
        paired = PAIR_METHOD[method]
        geometry = self.geometries[method][frame]
        paired_geometry = self.geometries[paired][frame]
        selected_color = METHOD_COLORS[method]
        paired_color = METHOD_COLORS[paired]
        prefix = "/sequence/current"

        point_handle = self.server.scene.add_point_cloud(
            f"{prefix}/pointmap",
            points=geometry["points_world"],
            colors=self.sequence.point_colors[frame],
            point_size=float(self.args.point_size) * 1.25,
            point_shape="rounded",
            precision="float32",
            visible=bool(self.show_current_points_gui.value),
        )
        self.current_points.append(point_handle)
        self.frame_handles.append(point_handle)

        pose = geometry["camera_pose"]
        image = self.sequence.images[frame]
        K = self.sequence.intrinsics[frame]
        height, width = image.shape[:2]
        fov = 2.0 * np.arctan((height * 0.5) / max(float(K[1, 1]), 1e-6))
        camera = self.server.scene.add_camera_frustum(
            f"{prefix}/camera",
            fov=float(fov),
            aspect=float(width / max(height, 1)),
            scale=float(self.camera_scale_gui.value),
            line_width=2.0,
            color=(33, 82, 126) if frame < self.sequence.count else (181, 91, 36),
            image=image,
            jpeg_quality=80,
            wxyz=tf.SO3.from_matrix(pose[:3, :3]).wxyz,
            position=pose[:3, 3],
            visible=bool(self.show_camera_gui.value),
        )
        label = self.server.scene.add_label(
            f"{prefix}/label",
            text=f"frame {frame} | {'PRE-CUT' if frame < self.sequence.count else 'POST-CUT'}",
            position=pose[:3, 3],
            anchor="bottom-center",
            visible=bool(self.show_camera_gui.value),
        )
        self.cameras.append(camera)
        self.labels.append(label)
        self.frame_handles.extend((camera, label))

        selected = self.server.scene.add_mesh_simple(
            f"{prefix}/human/selected",
            vertices=geometry["vertices_world"],
            faces=self.faces,
            color=selected_color,
            opacity=0.78,
            flat_shading=False,
            side="double",
            visible=bool(self.show_human_gui.value),
        )
        compared = self.server.scene.add_mesh_simple(
            f"{prefix}/human/paired",
            vertices=paired_geometry["vertices_world"],
            faces=self.faces,
            color=paired_color,
            wireframe=True,
            opacity=0.72,
            flat_shading=False,
            side="double",
            visible=bool(self.show_human_gui.value and self.overlay_gui.value),
        )
        self.selected_meshes.append(selected)
        self.paired_meshes.append(compared)
        self.method_handles.extend((selected, compared))
        self.frame_handles.extend((selected, compared))

        if frame > 0:
            ghost = self.server.scene.add_mesh_simple(
                f"{prefix}/human/previous_frame",
                vertices=self.geometries[method][frame - 1]["vertices_world"],
                faces=self.faces,
                color=(118, 124, 128),
                wireframe=True,
                opacity=0.25,
                flat_shading=False,
                side="double",
                visible=bool(self.show_human_gui.value and self.ghost_gui.value),
            )
            self.ghost_meshes.append(ghost)
            self.method_handles.append(ghost)
            self.frame_handles.append(ghost)

        joints = geometry["joints_world"][:22]
        joint_cloud = self.server.scene.add_point_cloud(
            f"{prefix}/human/joints",
            points=joints,
            colors=selected_color,
            point_size=0.028,
            point_shape="circle",
            precision="float32",
            visible=bool(self.show_joints_gui.value),
        )
        skeleton = self.server.scene.add_line_segments(
            f"{prefix}/human/skeleton",
            points=skeleton_segments(joints),
            colors=selected_color,
            line_width=3.0,
            visible=bool(self.show_joints_gui.value),
        )
        self.joint_handles.extend((joint_cloud, skeleton))
        self.method_handles.extend((joint_cloud, skeleton))
        self.frame_handles.extend((joint_cloud, skeleton))
        self.image_gui.image = self.sequence.images[frame]
        self._update_metrics(frame)

    def _update_metrics(self, frame: int) -> None:
        method = self.method_index()
        paired = PAIR_METHOD[method]
        selected = self.geometries[method][frame]
        comparison = self.geometries[paired][frame]
        continuity = method in (1, 3)
        selected_beta = (
            self.sequence.output_betas[frame]
            if continuity
            else self.sequence.raw_betas[frame]
        )
        selected_scale = float(
            self.sequence.output_scales[frame]
            if continuity
            else self.sequence.raw_scales[frame]
        )
        raw_scale = float(self.sequence.raw_scales[frame])
        memory_scale = float(self.sequence.output_scales[frame])
        raw_residual = float(self.sequence.raw_local_residual_deg[frame])
        memory_residual = float(self.sequence.output_local_residual_deg[frame])
        mesh_delta_m = float(
            np.linalg.norm(
                selected["vertices_world"] - comparison["vertices_world"], axis=1
            ).mean()
        )
        mesh_delta_px = mean_projected_displacement(
            selected["vertices_camera"][::40],
            comparison["vertices_camera"][::40],
            self.sequence.intrinsics[frame],
        )
        camera_delta = float(
            np.max(np.abs(selected["camera_pose"] - comparison["camera_pose"]))
        )
        root_delta = float(
            np.linalg.norm(selected["root_world"] - comparison["root_world"])
        )
        point_delta = float(
            np.max(np.abs(selected["points_world"] - comparison["points_world"]))
        )

        if frame > 0:
            current_centered = selected["vertices_camera"] - selected["joints_camera"][:1]
            previous = self.geometries[method][frame - 1]
            previous_centered = previous["vertices_camera"] - previous["joints_camera"][:1]
            frame_motion = float(
                np.linalg.norm(current_centered - previous_centered, axis=1).mean()
            )
        else:
            frame_motion = 0.0

        aggregate = self.sequence.v14_2_case["continuity"]
        hard = aggregate["hard_reset"]
        memory = aggregate["shape_scale_local_pose_memory"]
        quantitative = self.sequence.quantitative_case["methods"]
        fixed_metric = quantitative["fixed_explicit"]
        coupled_key = (
            "v18_coupled_full_continuity" if method == 3 else "v18_coupled_full"
        )
        coupled_metric = quantitative[coupled_key]
        phase = "CUT 前（memory 尚未触发）" if frame < self.sequence.count else "CUT 后"
        selected_hex = "#%02x%02x%02x" % METHOD_COLORS[method]
        paired_hex = "#%02x%02x%02x" % METHOD_COLORS[paired]
        self.metrics_gui.content = (
            f"### {self.sequence.name}\n"
            f"**Frame {frame}/{self.frame_count - 1} · {phase}**  "
            f"（cut: {self.sequence.count - 1} → {self.sequence.count}）\n\n"
            f"- 当前：<span style='color:{selected_hex}'>**{METHOD_LABELS[method]}**</span>\n"
            f"- 线框对照：<span style='color:{paired_hex}'>**{METHOD_LABELS[paired]}**</span>\n"
            f"- 当前帧两人体差异：**{mesh_delta_m * 100.0:.2f} cm / {mesh_delta_px:.2f} px**\n"
            f"- body scale（raw / memory）：**{raw_scale:.4f} / {memory_scale:.4f}**；"
            f"当前显示 **{selected_scale:.4f}**\n"
            f"- local-pose residual（raw / memory）：**{raw_residual:.2f}° / "
            f"{memory_residual:.2f}°**\n"
            f"- 当前人体相对上一帧的 root-centered mesh 变化：**{frame_motion * 100.0:.2f} cm**\n"
            f"- beta norm（当前显示）：**{float(np.linalg.norm(selected_beta)):.3f}**\n\n"
            "#### 当前统一的人—相机对齐\n"
            f"Camera translation **{fixed_metric['camera']['translation_m']:.3f} → "
            f"{coupled_metric['camera']['translation_m']:.3f} m**；human world root "
            f"**{fixed_metric['human']['world_root_error_m']:.3f} → "
            f"{coupled_metric['human']['world_root_error_m']:.3f} m**；world joints "
            f"**{fixed_metric['human']['world_joint_mean_error_m']:.3f} → "
            f"{coupled_metric['human']['world_joint_mean_error_m']:.3f} m**。  \n"
            f"Torso reprojection **{fixed_metric['projection']['torso_mean_px']:.1f} → "
            f"{coupled_metric['projection']['torso_mean_px']:.1f} px**；mesh bbox IoU "
            f"**{fixed_metric['projection']['mesh_bbox']['iou']:.3f} → "
            f"{coupled_metric['projection']['mesh_bbox']['iou']:.3f}**；foot-scene distance "
            f"**{fixed_metric['scene']['foot_nearest_mean_m']:.3f} → "
            f"{coupled_metric['scene']['foot_nearest_mean_m']:.3f} m**。\n\n"
            "#### 这个对比隔离了什么\n"
            f"Camera 最大差异 **{camera_delta:.1e}**；human root 差异 "
            f"**{root_delta:.1e} m**；scene pointmap 最大差异 **{point_delta:.1e} m**。  "
            "因此实心/线框人体的差异只来自 shape、scale 和 root-centered local pose，"
            "不是相机或 Boundary 被重新对齐。\n\n"
            "#### 该序列在 cut 边界的定量变化\n"
            f"Shape jump **{hard['shape_jump_l2']:.3f} → {memory['shape_jump_l2']:.3f}**；"
            f"scale jump **{hard['body_scale_jump_abs']:.4f} → "
            f"{memory['body_scale_jump_abs']:.4f}**；local pose **"
            f"{hard['local_pose_jump_residual_deg']:.2f}° → "
            f"{memory['local_pose_jump_residual_deg']:.2f}°**。"
        )

    def _show_frame(self, frame: int, force: bool = False) -> None:
        frame = int(np.clip(frame, 0, self.frame_count - 1))
        with self.lock, self.server.atomic():
            if not force and frame == self.current_frame:
                return
            self.current_frame = frame
            self._render_current_frame()

    def _set_human_visibility(self) -> None:
        show = bool(self.show_human_gui.value)
        for handle in self.selected_meshes:
            handle.visible = show
        for handle in self.paired_meshes:
            handle.visible = show and bool(self.overlay_gui.value)
        for handle in self.ghost_meshes:
            handle.visible = show and bool(self.ghost_gui.value)

    def _register_callbacks(self) -> None:
        @self.frame_gui.on_update
        def _(_) -> None:
            self._show_frame(int(self.frame_gui.value))

        @self.prev_gui.on_click
        def _(_) -> None:
            self.frame_gui.value = (int(self.frame_gui.value) - 1) % self.frame_count

        @self.next_gui.on_click
        def _(_) -> None:
            self.frame_gui.value = (int(self.frame_gui.value) + 1) % self.frame_count

        @self.play_gui.on_update
        def _(_) -> None:
            playing = bool(self.play_gui.value)
            self.frame_gui.disabled = playing
            self.prev_gui.disabled = playing
            self.next_gui.disabled = playing

        @self.method_gui.on_update
        def _(_) -> None:
            self._rebuild_method_geometry()

        @self.overlay_gui.on_update
        def _(_) -> None:
            self._set_human_visibility()
            for index, handle in enumerate(self.root_handles[1:]):
                handle.visible = bool(self.show_root_gui.value and self.overlay_gui.value)

        @self.ghost_gui.on_update
        def _(_) -> None:
            self._set_human_visibility()

        @self.show_human_gui.on_update
        def _(_) -> None:
            self._set_human_visibility()

        @self.show_joints_gui.on_update
        def _(_) -> None:
            for handle in self.joint_handles:
                handle.visible = bool(self.show_joints_gui.value)

        @self.show_root_gui.on_update
        def _(_) -> None:
            for index, handle in enumerate(self.root_handles):
                handle.visible = bool(
                    self.show_root_gui.value and (index == 0 or self.overlay_gui.value)
                )

        @self.show_scene_gui.on_update
        def _(_) -> None:
            self.accumulated_scene.visible = bool(self.show_scene_gui.value)

        @self.show_current_points_gui.on_update
        def _(_) -> None:
            for handle in self.current_points:
                handle.visible = bool(self.show_current_points_gui.value)

        @self.show_camera_gui.on_update
        def _(_) -> None:
            visible = bool(self.show_camera_gui.value)
            self.camera_trajectory.visible = visible
            for handle in self.cameras + self.labels:
                handle.visible = visible

        @self.camera_scale_gui.on_update
        def _(_) -> None:
            for handle in self.cameras:
                handle.scale = float(self.camera_scale_gui.value)

        @self.reset_gui.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self.reset_camera(event.client)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.reset_camera(client)

    def reset_camera(self, client: viser.ClientHandle) -> None:
        center = self.scene_center.astype(np.float32)
        radius = float(self.scene_radius)
        client.camera.up_direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        client.camera.look_at = center
        client.camera.position = center + np.asarray(
            [radius * 0.65, -radius * 0.45, radius * 0.80], dtype=np.float32
        )

    def run(self) -> None:
        print(f">> Interactive V14.3 viewer: http://127.0.0.1:{self.args.port}", flush=True)
        while True:
            if bool(self.play_gui.value):
                self.frame_gui.value = (int(self.frame_gui.value) + 1) % self.frame_count
            time.sleep(1.0 / max(float(self.fps_gui.value), 0.5))


def main() -> None:
    args = parse_args()
    sequence, faces = build_sequence(args)
    viewer = ContinuityViewer(sequence, faces, args)
    viewer.run()


if __name__ == "__main__":
    main()
