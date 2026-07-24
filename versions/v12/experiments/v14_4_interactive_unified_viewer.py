#!/usr/bin/env python3
"""Demo-style 10+10 frame 3D viewer for V14.4 alignment variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v12.experiments import v14_3_interactive_continuity_viewer as viewer_base  # noqa: E402
from versions.v12.experiments.v14_3_human_continuity_visualization import load_sequence  # noqa: E402
from boundary_shot_scale_support import scale_pose  # noqa: E402


DEFAULT_REPORT = (
    ROOT
    / "output/v14_4_unified_similarity_reanchoring/full180_final"
    / "v14_4_unified_similarity_reanchoring.json"
)
DEFAULT_V14_3 = (
    ROOT
    / "output/v14_3_projection_consistent_reanchoring/quantitative"
    / "v14_3_projection_consistent_reanchoring.json"
)
DEFAULT_V14_2 = (
    ROOT
    / "output/v14_2_canonical_human_memory/single_cut"
    / "v14_2_canonical_human_memory_probe.json"
)
DEFAULT_CACHE = ROOT / "output/v52_long_sequence_visualization/cache"
DEFAULT_SELECTION = (
    ROOT
    / "output/v14_4_unified_similarity_reanchoring/visualization"
    / "case_selection.json"
)
DEFAULT_CASE = "mvhuman200_120_150_200002_410_22327109_22236235"
METHOD_KEYS = (
    "v11_4_uniform_similarity_conditional_vggt",
    "unified_shared_scale_coupled_root_conditional_vggt",
    "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
    "naive_sequential",
)
METHOD_LABELS = (
    "V11.4 Uniform Similarity",
    "Unified Human Projection",
    "Unified DA3 Root",
    "Naive Sequential",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--v14_3_report", type=Path, default=DEFAULT_V14_3)
    parser.add_argument("--v14_2_report", type=Path, default=DEFAULT_V14_2)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--case", default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--point_stride", type=int, default=24)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--point_size", type=float, default=0.009)
    parser.add_argument("--camera_scale", type=float, default=0.16)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def finite_scene(method: dict) -> float:
    value = float(method["scene"]["trimmed_mean_m"])
    return value if np.isfinite(value) else 0.50


def composite(method: dict) -> float:
    return float(
        method["camera"]["translation_m"]
        + method["human"]["world_root_error_m"]
        + method["human"]["world_joint_mean_error_m"]
        + finite_scene(method)
    )


def select_cases(report: dict, available: set[str]) -> dict:
    rows = [row for row in report["cases"] if row["case_name"] in available]
    by_source = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]

        def scores(row: dict) -> tuple[float, float, float, float]:
            methods = row["methods"]
            v11 = composite(methods[METHOD_KEYS[0]])
            human = composite(methods[METHOD_KEYS[1]])
            da3 = composite(methods[METHOD_KEYS[2]])
            naive = composite(methods[METHOD_KEYS[3]])
            return v11, human, da3, naive

        by_source[source] = {
            "unified_success": max(
                source_rows, key=lambda row: scores(row)[0] - min(scores(row)[1:3])
            )["case_name"],
            "v11_better": max(
                source_rows, key=lambda row: min(scores(row)[1:3]) - scores(row)[0]
            )["case_name"],
            "human_projection_better": max(
                source_rows, key=lambda row: scores(row)[2] - scores(row)[1]
            )["case_name"],
            "da3_better": max(
                source_rows, key=lambda row: scores(row)[1] - scores(row)[2]
            )["case_name"],
            "naive_double_correction": max(
                source_rows, key=lambda row: scores(row)[3] - min(scores(row)[:3])
            )["case_name"],
        }
    all_roles = [name for roles in by_source.values() for name in roles.values()]
    return {
        "available_long_sequences": len(rows),
        "default_case": DEFAULT_CASE if DEFAULT_CASE in available else all_roles[0],
        "methods": dict(zip(METHOD_LABELS, METHOD_KEYS)),
        "by_source": by_source,
    }


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def method_geometry(sequence, method_index: int, frame: int) -> dict:
    row = sequence.v14_4_case["methods"][METHOD_KEYS[method_index]]
    pre = frame < sequence.count
    scale = float(sequence.v14_4_case["scales"]["common_pre"] if pre else row["definition"]["human_scale"])
    local_pose = sequence.local_poses[frame]
    scaled_pose = scale_pose(local_pose, scale)
    boundary = np.eye(4, dtype=np.float32) if pre else np.asarray(row["boundary"], dtype=np.float32)
    camera_pose = (boundary @ scaled_pose).astype(np.float32)

    raw_root = sequence.raw_joints_camera[frame, 0]
    root_local = scaled_pose[:3, :3] @ (raw_root * scale) + scaled_pose[:3, 3]
    if not pre:
        root_local = root_local + sequence.v14_4_root_corrections[method_index]
    root_world = boundary[:3, :3] @ root_local + boundary[:3, 3]
    joints_centered = (sequence.raw_joints_camera[frame] - raw_root) * scale
    vertices_centered = (sequence.raw_vertices_camera[frame] - raw_root) * scale
    joints_world = joints_centered @ camera_pose[:3, :3].T + root_world
    vertices_world = vertices_centered @ camera_pose[:3, :3].T + root_world
    inverse_camera = np.linalg.inv(camera_pose)
    return {
        "camera_pose": camera_pose,
        "root_world": root_world.astype(np.float32),
        "joints_world": joints_world.astype(np.float32),
        "vertices_world": vertices_world.astype(np.float32),
        "joints_camera": transform_points(inverse_camera, joints_world),
        "vertices_camera": transform_points(inverse_camera, vertices_world),
        "points_world": (
            sequence.points_local[frame] * scale
            if pre
            else transform_points(boundary, sequence.points_local[frame] * scale)
        ).astype(np.float32),
    }


def build_sequence(args: argparse.Namespace):
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.4 viewer SMPL-X preparation must run on CUDA")
    torch.cuda.set_device(torch.device(args.device))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    v14_3 = json.loads(args.v14_3_report.read_text(encoding="utf-8"))
    v14_2 = json.loads(args.v14_2_report.read_text(encoding="utf-8"))
    v14_4_map = {row["case_name"]: row for row in report["cases"]}
    v14_3_map = {row["case_name"]: row for row in v14_3["cases"]}
    v14_2_map = {row["case_name"]: row for row in v14_2["cases"]}
    available = {path.parent.name for path in args.cache_dir.glob("*/manifest.json")}
    common = available & v14_4_map.keys() & v14_3_map.keys() & v14_2_map.keys()
    selection = select_cases(report, common)
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    name = str(args.case or selection["default_case"])
    if name not in common:
        raise KeyError(f"No common V14.4 long-sequence cache for {name}")

    device = torch.device(args.device)
    print(
        f">> Building V14.4 10+10 frame body cache on {device}: {name}",
        file=sys.stderr,
        flush=True,
    )
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    args.alignment_method = "v18_coupled_full"
    sequence = load_sequence(
        name,
        args.cache_dir,
        v14_3_map[name],
        v14_2_map[name],
        layer,
        device,
        args,
    )
    del layer
    torch.cuda.empty_cache()
    sequence.v14_4_case = v14_4_map[name]
    sequence.v14_4_root_corrections = []
    first_pose = sequence.local_poses[sequence.count]
    raw_reference = np.asarray(v14_3_map[name]["roots"]["raw_camera"], dtype=np.float32)
    for key in METHOD_KEYS:
        method = v14_4_map[name]["methods"][key]
        scale = float(method["definition"]["human_scale"])
        final_root = np.asarray(method["human"]["root_camera"], dtype=np.float32)
        correction_camera = final_root - raw_reference * scale
        sequence.v14_4_root_corrections.append(
            (first_pose[:3, :3] @ correction_camera).astype(np.float32)
        )
    print(f">> Case roles written to {args.selection}", file=sys.stderr, flush=True)
    return sequence, faces


class UnifiedViewer(viewer_base.ContinuityViewer):
    def _update_metrics(self, frame: int) -> None:
        method = self.method_index()
        paired = viewer_base.PAIR_METHOD[method]
        selected = self.geometries[method][frame]
        comparison = self.geometries[paired][frame]
        selected_row = self.sequence.v14_4_case["methods"][METHOD_KEYS[method]]
        paired_row = self.sequence.v14_4_case["methods"][METHOD_KEYS[paired]]
        mesh_delta = float(
            np.mean(np.linalg.norm(selected["vertices_world"] - comparison["vertices_world"], axis=1))
        )
        root_delta = float(np.linalg.norm(selected["root_world"] - comparison["root_world"]))
        point_delta = float(
            np.max(np.abs(selected["points_world"] - comparison["points_world"]))
        )
        phase = "PRE-CUT" if frame < self.sequence.count else "POST-CUT"
        self.metrics_gui.content = (
            f"### {self.sequence.name}\n"
            f"**Frame {frame}/{self.frame_count - 1} · {phase}**  "
            f"(cut: {self.sequence.count - 1} → {self.sequence.count})\n\n"
            f"- Current: **{METHOD_LABELS[method]}**\n"
            f"- Wireframe: **{METHOD_LABELS[paired]}**\n"
            f"- Current mesh/root/pointmap difference: **{mesh_delta:.3f} / "
            f"{root_delta:.3f} / {point_delta:.3f} m**\n"
            f"- Shared post scale: **{selected_row['definition']['human_scale']:.4f}**\n\n"
            "#### 180-cut unified metrics for this case\n"
            f"- Camera translation: **{selected_row['camera']['translation_m']:.3f} m** "
            f"(wireframe {paired_row['camera']['translation_m']:.3f})\n"
            f"- Human root / joints: **{selected_row['human']['world_root_error_m']:.3f} / "
            f"{selected_row['human']['world_joint_mean_error_m']:.3f} m**\n"
            f"- Scene / foot-scene: **{selected_row['scene']['trimmed_mean_m']:.3f} / "
            f"{selected_row['scene']['foot_scene_mean_m']:.3f} m**\n"
            f"- Torso reprojection: **{selected_row['projection']['torso_mean_px']:.1f} px**\n"
            f"- Camera-human closure: **{selected_row['sanity']['camera_human_equation_closure_m']:.2e} m**"
        )

    def run(self) -> None:
        print(f">> Interactive V14.4 viewer: http://127.0.0.1:{self.args.port}", flush=True)
        super().run()


def main() -> None:
    args = parse_args()
    viewer_base.METHOD_LABELS = METHOD_LABELS
    viewer_base.DISPLAY_METHODS = METHOD_LABELS
    viewer_base.PAIR_METHOD = {0: 1, 1: 0, 2: 3, 3: 2}
    viewer_base.method_geometry = method_geometry
    sequence, faces = build_sequence(args)
    if args.validate_only:
        validation = {}
        for method, label in enumerate(METHOD_LABELS):
            geometries = [method_geometry(sequence, method, frame) for frame in range(len(sequence.images))]
            vertices = np.concatenate([row["vertices_world"] for row in geometries])
            points = np.concatenate([row["points_world"] for row in geometries])
            cameras = np.stack([row["camera_pose"] for row in geometries])
            validation[label] = {
                "frames": len(geometries),
                "point_count": int(len(points)),
                "finite_vertices": bool(np.isfinite(vertices).all()),
                "finite_cameras": bool(np.isfinite(cameras).all()),
                "vertex_extent_m": np.ptp(vertices, axis=0).astype(float).tolist(),
                "point_extent_m": np.ptp(points, axis=0).astype(float).tolist(),
            }
        print(json.dumps(validation, indent=2, ensure_ascii=False), flush=True)
        return
    UnifiedViewer(sequence, faces, args).run()


if __name__ == "__main__":
    main()
