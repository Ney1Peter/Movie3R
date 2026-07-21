#!/usr/bin/env python3
"""V11.1 interactive comparison of retained boundary-alignment methods."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import boundary_viewer_base as base  # noqa: E402


EXAMPLES = {
    "AvatarReX metric-scale gain": "avatarrex_120_150_lbn2_1651_22010714_22070932",
    "THuman clean alignment": "thuman_060_090_thuman02_2770_cam12_cam07",
    "MVHuman100 large correction": "mvhuman100_120_150_100003_356_CC32871A059_CC32871A057",
    "MVHuman100 wide-rotation rescue": "mvhuman100_090_120_100003_338_CC32871A035_CC32871A008",
    "MVHuman200 strong wide-rotation rescue": "mvhuman200_060_090_200004_379_22327109_22327118",
    "MVHuman200 extreme cut": "mvhuman200_120_150_200002_410_22327109_22236235",
}

METHODS = {
    "Hard Reset": None,
    "Fixed Explicit": None,
    "Torso Only": None,
    "Wide Rotation Only (Unsafe)": None,
    "Conditional Wide Rotation": None,
    "DA3 Bounded Translation (Diagnostic)": None,
    "Global Metric Alignment": None,
    "Contact-Preserving Alignment": None,
}

COLORS = {
    "Hard Reset": (217, 70, 70),
    "Fixed Explicit": (220, 124, 25),
    "Torso Only": (14, 116, 144),
    "Wide Rotation Only (Unsafe)": (190, 24, 93),
    "Conditional Wide Rotation": (13, 148, 136),
    "DA3 Bounded Translation (Diagnostic)": (37, 99, 235),
    "Global Metric Alignment": (22, 163, 74),
    "Contact-Preserving Alignment": (124, 58, 237),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10_report", type=Path, default=base.DEFAULT_V10_REPORT)
    parser.add_argument("--candidate_dir", type=Path, default=base.DEFAULT_CANDIDATE_DIR)
    parser.add_argument(
        "--bridge_report",
        type=Path,
        default=ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json",
    )
    parser.add_argument(
        "--selected_export",
        type=Path,
        default=ROOT / "output/v45_final_autonomous_explicit_bridge/v45_selected_explicit_bridge_180.json",
    )
    parser.add_argument(
        "--contact_report",
        type=Path,
        default=(
            ROOT
            / "output/v46_contact_preserving_metric_bridge"
            / "v46_contact_preserving_metric_bridge_probe.json"
        ),
    )
    parser.add_argument(
        "--component_report",
        type=Path,
        default=(
            ROOT
            / "output/v48_component_necessity_ablation"
            / "v48_component_necessity_ablation.json"
        ),
    )
    parser.add_argument(
        "--da3_translation_report",
        type=Path,
        default=(
            ROOT
            / "output/v51_safe_da3_translation_bridge"
            / "v51_safe_da3_translation_bridge.json"
        ),
    )
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--point_stride", type=int, default=6)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--point_size", type=float, default=0.008)
    return parser.parse_args()


def raw_method(row: dict) -> dict:
    return {
        "transform": row["transform"],
        "camera_translation_error_m": float(row["camera_translation_error_m"]),
        "camera_rotation_error_deg": float(row["camera_rotation_error_deg"]),
        "root_scales": {"old": 1.0, "new": 1.0},
        "scene_scales": {"old": 1.0, "new": 1.0},
    }


def metric_method(case: dict, variant: str) -> dict:
    value = case["variants"][variant]
    return {
        "transform": value["transform"],
        "camera_translation_error_m": float(value["camera"]["translation_m"]),
        "camera_rotation_error_deg": float(value["camera"]["rotation_deg"]),
        "root_scales": {
            "old": float(case["root_scales"]["old"]),
            "new": float(case["root_scales"]["new"]),
        },
        "scene_scales": {
            "old": float(case["scene_scale_sets"]["absolute"]["old"]),
            "new": float(case["scene_scale_sets"]["absolute"]["new"]),
        },
        "human_motion_error_m": float(value["human"]["root_motion_error_m"]),
        "scene_error_m": float(value["scene"]["trimmed_mean_m"]),
        "branch": "torso/gravity" if variant == "v22" else str(case["v32_branch"]),
    }


def contact_method(case: dict, variant: str) -> dict:
    value = case[variant]
    metric = variant != "raw_scale_v32"
    contact = value.get("contact", {})
    before = contact.get("mean_absolute_contact_distortion_m")
    if before is None and "old" in contact and "new" in contact:
        before = 0.5 * (
            float(contact["old"]["absolute_contact_distortion_m"])
            + float(contact["new"]["absolute_contact_distortion_m"])
        )
    return {
        "transform": value["transform"],
        "camera_translation_error_m": float(value["camera"]["translation_m"]),
        "camera_rotation_error_deg": float(value["camera"]["rotation_deg"]),
        "root_scales": {
            "old": float(case["root_scales"]["old"]) if metric else 1.0,
            "new": float(case["root_scales"]["new"]) if metric else 1.0,
        },
        "scene_scales": {
            "old": float(case["scene_scales"]["old"]) if metric else 1.0,
            "new": float(case["scene_scales"]["new"]) if metric else 1.0,
        },
        "human_motion_error_m": float(value["human"]["root_motion_error_m"]),
        "scene_error_m": float(value["scene"]["trimmed_mean_m"]),
        "contact_distortion_before_m": float(before or 0.0),
        "contact_distortion_after_m": float(
            contact.get("post_correction_contact_proxy_m", before or 0.0)
        ),
        "contact_correction_m": float(contact.get("mean_correction_m", 0.0)),
        "human_reprojection_shift_px": float(
            value.get("integrity", {}).get("human_reprojection_shift_px", 0.0)
        ),
        "rigid_local_geometry": bool(
            value.get("integrity", {}).get("rigid_local_geometry", False)
        ),
        "preserve_contact": variant == "contact_v32",
        "branch": {
            "raw_scale_v32": "conditional wide rotation, original Human3R gauge",
            "current_v45": "conditional wide rotation + independent metric scaling",
            "contact_v32": "conditional wide rotation + metric scaling + contact correction",
        }[variant],
    }


def ablation_method(case: dict, variant: str, branch: str) -> dict:
    value = case["variants"][variant]
    integrity = value["integrity"]
    return {
        "transform": value["transform"],
        "camera_translation_error_m": float(value["camera"]["translation_m"]),
        "camera_rotation_error_deg": float(value["camera"]["rotation_deg"]),
        "root_scales": {"old": 1.0, "new": 1.0},
        "scene_scales": {"old": 1.0, "new": 1.0},
        "human_motion_error_m": float(value["human"]["root_motion_error_m"]),
        "scene_error_m": float(value["scene"]["trimmed_mean_m"]),
        "contact_distortion_before_m": float(
            integrity["foot_ground_distortion_m"]
        ),
        "contact_distortion_after_m": float(
            integrity["foot_ground_distortion_m"]
        ),
        "contact_correction_m": 0.0,
        "human_reprojection_shift_px": float(
            integrity["human_reprojection_shift_px"]
        ),
        "rigid_local_geometry": bool(integrity["rigid_local_geometry"]),
        "preserve_contact": False,
        "branch": branch,
    }


def selected_da3_method(case: dict) -> dict:
    value = case["result"]
    return {
        "transform": value["transform"],
        "camera_translation_error_m": float(value["camera"]["translation_m"]),
        "camera_rotation_error_deg": float(value["camera"]["rotation_deg"]),
        "root_scales": {"old": 1.0, "new": 1.0},
        "scene_scales": {"old": 1.0, "new": 1.0},
        "human_motion_error_m": float(value["human"]["root_motion_error_m"]),
        "scene_error_m": float(value["scene"]["trimmed_mean_m"]),
        "contact_distortion_before_m": 0.0,
        "contact_distortion_after_m": 0.0,
        "contact_correction_m": 0.0,
        "human_reprojection_shift_px": 0.0,
        "rigid_local_geometry": True,
        "preserve_contact": False,
        "branch": (
            "DA3 prior > 0.5 m, bounded 0.2 m translation"
            if bool(case["use_da3"])
            else "conditional-wide fallback: DA3 disagreement below 0.5 m"
        ),
    }


def load_maps(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    v10_payload = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v10 = {row["case_name"]: row for row in v10_payload["cases"]}
    v16 = {}
    for path in sorted(args.candidate_dir.glob("v16_candidates_shard_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            v16[row["case_name"]] = row
    bridge_payload = json.loads(args.bridge_report.read_text(encoding="utf-8"))
    bridge = {row["case_name"]: row for row in bridge_payload["cases"]}
    export_payload = json.loads(args.selected_export.read_text(encoding="utf-8"))
    exported = {row["case_name"]: row for row in export_payload["cases"]}
    v46_payload = json.loads(args.contact_report.read_text(encoding="utf-8"))
    v46 = {row["case_name"]: row for row in v46_payload["cases"]}
    v48_payload = json.loads(args.component_report.read_text(encoding="utf-8"))
    v48 = {row["case_name"]: row for row in v48_payload["cases"]}
    v51_payload = json.loads(args.da3_translation_report.read_text(encoding="utf-8"))
    v51 = {row["case_name"]: row for row in v51_payload["cases"]}

    methods = {}
    for name in EXAMPLES.values():
        if (
            name not in v10
            or name not in v16
            or name not in bridge
            or name not in exported
            or name not in v46
            or name not in v48
            or name not in v51
        ):
            raise KeyError(f"Missing cached retained-method viewer case: {name}")
        final = metric_method(bridge[name], "v32")
        if not np.allclose(
            np.asarray(final["transform"]),
            np.asarray(exported[name]["transform"]),
            atol=1e-5,
        ):
            raise RuntimeError(f"Selected export mismatch for {name}")
        methods[name] = {
            "viewer_methods": {
                "Hard Reset": raw_method(v16[name]["baselines"]["hard_reset"]),
                "Fixed Explicit": raw_method(v16[name]["baselines"]["fixed_explicit"]),
                "Torso Only": ablation_method(
                    v48[name], "torso_raw", "Fixed + torso motion, original scale"
                ),
                "Wide Rotation Only (Unsafe)": ablation_method(
                    v48[name], "vggt_raw", "Pure VGGT rotation, original scale"
                ),
                "Conditional Wide Rotation": contact_method(
                    v46[name], "raw_scale_v32"
                ),
                "DA3 Bounded Translation (Diagnostic)": selected_da3_method(v51[name]),
                "Global Metric Alignment": contact_method(v46[name], "current_v45"),
                "Contact-Preserving Alignment": contact_method(v46[name], "contact_v32"),
            }
        }
    return v10, methods


def method_result(case: base.CachedCase, method: str) -> dict:
    return case.candidate["viewer_methods"][method]


def scale_pose(pose: np.ndarray, scale: float) -> np.ndarray:
    output = np.asarray(pose, dtype=np.float32).copy()
    output[:3, 3] *= float(scale)
    return output


def camera_points(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,nj->ni",
        pose[:3, :3].T,
        points_world - pose[:3, 3],
    )


class RetainedMethodsViewer(base.ComparisonViewer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.show_oracle_gui.value = False
        self.method_gui.value = "Conditional Wide Rotation"
        self.render_scene()

    def shot_scales(self, result: dict, frame: int) -> tuple[float, float]:
        side = "old" if frame < int(self.args.boundary) else "new"
        return float(result["root_scales"][side]), float(result["scene_scales"][side])

    def scaled_geometry(
        self,
        case: base.CachedCase,
        result: dict,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        root_scale, scene_scale = self.shot_scales(result, frame)
        raw_pose = case.camera_poses[frame]
        pose = scale_pose(raw_pose, root_scale)

        local_points = camera_points(case.points_world[frame], raw_pose) * scene_scale
        points = (
            np.einsum("ij,nj->ni", raw_pose[:3, :3], local_points)
            + pose[:3, 3]
        ).astype(np.float32)

        local_vertices = camera_points(case.smpl_vertices_world[frame], raw_pose)
        root = np.asarray(case.smpl_root_camera[frame], dtype=np.float32)
        correction = np.zeros(3, dtype=np.float32)
        if bool(result.get("preserve_contact", False)):
            correction = self.contact_correction(case, frame, root_scale, scene_scale)
        corrected_root = root * root_scale + correction
        local_vertices = local_vertices - root + corrected_root
        vertices = (
            np.einsum("ij,nj->ni", raw_pose[:3, :3], local_vertices)
            + pose[:3, 3]
        ).astype(np.float32)
        root_world = raw_pose[:3, :3] @ corrected_root + pose[:3, 3]
        return points, vertices, pose, root_world.astype(np.float32)

    @staticmethod
    def contact_correction(
        case: base.CachedCase,
        frame: int,
        root_scale: float,
        scene_scale: float,
    ) -> np.ndarray:
        root = np.asarray(case.smpl_root_camera[frame], dtype=np.float32)
        pelvis = np.asarray(case.smpl_pelvis_camera[frame], dtype=np.float32)
        foot = np.asarray(case.smpl_feet_camera[frame], dtype=np.float32)
        offset = foot - root
        scaled_human_foot = root * float(root_scale) + offset
        scaled_scene_contact = foot * float(scene_scale)
        down = foot - pelvis
        down /= max(float(np.linalg.norm(down)), 1e-8)
        signed_error = float(np.dot(scaled_human_foot - scaled_scene_contact, down))
        return (-signed_error * down).astype(np.float32)

    def update_metrics(self, case: base.CachedCase, result: dict, method: str) -> None:
        fixed = method_result(case, "Fixed Explicit")
        pre = self.scaled_geometry(case, result, int(self.args.boundary) - 1)[3]
        post = self.scaled_geometry(case, result, int(self.args.boundary))[3]
        transform = np.asarray(result["transform"], dtype=np.float32)
        post = base.transform_points(transform, post[None])[0]
        root_jump = float(np.linalg.norm(post - pre))
        branch = result.get("branch", "raw boundary transform")
        human = result.get("human_motion_error_m")
        scene = result.get("scene_error_m")
        contact_before = result.get("contact_distortion_before_m")
        contact_after = result.get("contact_distortion_after_m")
        correction = result.get("contact_correction_m")
        reprojection = result.get("human_reprojection_shift_px")
        extra = ""
        if human is not None:
            extra += f"\n- Human motion error: **{float(human):.3f} m**"
        if scene is not None:
            extra += f"\n- Scene continuity: **{float(scene):.3f} m**"
        if contact_before is not None:
            extra += f"\n- Foot/ground distortion before: **{float(contact_before):.3f} m**"
        if contact_after is not None:
            extra += f"\n- Foot/ground distortion after: **{float(contact_after):.3f} m**"
        if correction is not None and float(correction) > 0.0:
            extra += f"\n- Root contact correction: **{float(correction):.3f} m**"
        if reprojection is not None:
            extra += f"\n- Human reprojection shift: **{float(reprojection):.1f} px**"
        extra += (
            "\n- Local geometry: **rigidly preserved**"
            if bool(result.get("rigid_local_geometry", False))
            else "\n- Local geometry: **modified diagnostic**"
        )
        scales = result["root_scales"], result["scene_scales"]
        extra += (
            f"\n- Root scale old/new: **{scales[0]['old']:.3f} / {scales[0]['new']:.3f}**"
            f"\n- Scene scale old/new: **{scales[1]['old']:.3f} / {scales[1]['new']:.3f}**"
        )
        self.metrics_gui.content = (
            f"### {method}\n"
            f"- Camera translation error: **{float(result['camera_translation_error_m']):.3f} m**\n"
            f"- Camera rotation error: **{float(result['camera_rotation_error_deg']):.2f} deg**\n"
            f"- Visible boundary root jump: **{root_jump:.3f} m**\n"
            f"- Gain over Fixed: **{float(fixed['camera_translation_error_m']) - float(result['camera_translation_error_m']):+.3f} m T, "
            f"{float(fixed['camera_rotation_error_deg']) - float(result['camera_rotation_error_deg']):+.2f} deg R**\n"
            f"- Rotation branch: **{branch}**"
            f"{extra}"
        )

    def render_scene(self) -> None:
        with self.lock:
            case = self.selected_case()
            method = str(self.method_gui.value)
            frames = base.VIEW_FRAMES[str(self.view_gui.value)]
            result = method_result(case, method)
            transform = np.asarray(result["transform"], dtype=np.float32)
            color = COLORS[method]
            clouds = []
            poses = []

            with self.server.atomic():
                self.clear_scene()
                for frame in frames:
                    points, vertices, pose, _ = self.scaled_geometry(case, result, frame)
                    if frame >= int(self.args.boundary):
                        points = base.transform_points(transform, points)
                        vertices = base.transform_points(transform, vertices)
                        pose = transform @ pose
                    clouds.append(points)
                    poses.append(pose)
                    self.handles["pointcloud"].append(
                        self.server.scene.add_point_cloud(
                            f"/comparison/pointmap/frame_{frame}",
                            points=points,
                            colors=case.point_colors[frame],
                            point_size=float(self.point_size_gui.value),
                            point_shape="rounded",
                            precision="float32",
                            visible=bool(self.show_points_gui.value),
                        )
                    )
                    self.handles["smpl"].append(
                        self.server.scene.add_mesh_simple(
                            f"/comparison/smplx/frame_{frame}",
                            vertices=vertices,
                            faces=self.smpl_faces,
                            color=base.PRE_HUMAN_COLOR if frame < self.args.boundary else base.POST_HUMAN_COLOR,
                            flat_shading=False,
                            side="double",
                            opacity=0.92,
                            visible=bool(self.show_smpl_gui.value),
                        )
                    )
                    camera_color = base.PRE_CAMERA_COLOR if frame < self.args.boundary else color
                    self.add_camera(
                        f"/comparison/cameras/frame_{frame}",
                        pose,
                        case.intrinsics[frame],
                        case.images[frame],
                        camera_color,
                        "camera",
                        0.14,
                    )
                    self.handles["label"].append(
                        self.server.scene.add_label(
                            f"/comparison/labels/frame_{frame}",
                            text=f"frame {frame} {'pre' if frame < self.args.boundary else 'post'}",
                            position=pose[:3, 3],
                            anchor="bottom-center",
                            visible=bool(self.show_cameras_gui.value),
                        )
                    )

                centers = np.stack([pose[:3, 3] for pose in poses])
                segments = self.line_segments(centers)
                if len(segments):
                    self.handles["trajectory"].append(
                        self.server.scene.add_line_segments(
                            "/comparison/candidate_trajectory",
                            segments,
                            colors=color,
                            line_width=3.0,
                            visible=bool(self.show_cameras_gui.value),
                        )
                    )
                finite = np.concatenate(clouds, axis=0)
                finite = finite[np.isfinite(finite).all(axis=1)]
                low = np.quantile(finite, 0.02, axis=0)
                high = np.quantile(finite, 0.98, axis=0)
                self.scene_center = ((low + high) * 0.5).astype(np.float32)
                self.scene_radius = max(float(np.linalg.norm(high - low)), 1.5)

            self.pre_image_gui.image = case.images[1]
            self.post_image_gui.image = case.images[2]
            self.update_metrics(case, result, method)

    def run(self) -> None:
        print(f"V11.1 retained-method viewer: http://127.0.0.1:{self.args.port}", flush=True)
        while True:
            time.sleep(10.0)


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    base.EXAMPLES = EXAMPLES
    base.METHODS = METHODS
    base.CAMERA_COLORS = COLORS
    base.method_result = method_result

    v10, methods = load_maps(args)
    device = torch.device(args.device)
    layer = base.build_smpl_layer(device)
    cache = {}
    for label, name in EXAMPLES.items():
        print(f"Preparing retained-method interactive scene: {label}", flush=True)
        case = base.load_cached_case(name, v10[name], methods[name], layer, device, args)
        local = Path(v10[name]["paths"]["human3r_local_reset"])
        arrays = {key: [] for key in ("rotvec", "shape", "transl", "expression")}
        for frame in range(4):
            with np.load(local / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
                for key in arrays:
                    value = smpl[key]
                    if key == "expression" and (value is None or len(value) == 0):
                        arrays[key].append(np.zeros(10, dtype=np.float32))
                    else:
                        arrays[key].append(np.asarray(value[0], dtype=np.float32))
        with torch.no_grad():
            body = layer(
                torch.from_numpy(np.stack(arrays["rotvec"])).to(device),
                torch.from_numpy(np.stack(arrays["shape"])).to(device),
                torch.from_numpy(np.stack(arrays["transl"])).to(device),
                None,
                None,
                K=torch.from_numpy(case.intrinsics).to(device),
                expression=torch.from_numpy(np.stack(arrays["expression"])).to(device),
            )
        joints = body["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32)
        names = layer.joint_names
        foot_indices = [
            names.index("left_big_toe"),
            names.index("left_small_toe"),
            names.index("left_heel"),
            names.index("right_big_toe"),
            names.index("right_small_toe"),
            names.index("right_heel"),
        ]
        case.smpl_root_camera = np.stack(arrays["transl"]).astype(np.float32)
        case.smpl_pelvis_camera = joints[:, names.index("pelvis")]
        case.smpl_feet_camera = np.median(joints[:, foot_indices], axis=1).astype(np.float32)
        cache[label] = case
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    del layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    viewer = RetainedMethodsViewer(cache, faces, args)
    viewer.run()


if __name__ == "__main__":
    main()
