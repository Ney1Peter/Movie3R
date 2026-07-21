#!/usr/bin/env python3
"""Interactive Human3R-style 3D viewer for V18 DA3 metric-depth examples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import v16_interactive_3d_comparison_viewer as base  # noqa: E402


EXAMPLES = {
    "AvatarReX camera-gain audit": "avatarrex_120_150_lbn2_1651_22010714_22070932",
    "THuman camera-gain audit": "thuman_060_090_thuman02_2651_cam17_cam12",
    "MVHuman100 audit A": "mvhuman100_150_180_100004_360_CC32871A022_CC32871A035",
    "MVHuman100 audit B": "mvhuman100_150_180_100005_346_CC32871A037_CC32871A015",
    "MVHuman200 audit A": "mvhuman200_120_150_200003_443_22327118_22327091",
    "MVHuman200 audit B": "mvhuman200_090_120_200001_433_22327118_22327113",
}

METHODS = {
    "Hard Reset": None,
    "Fixed Explicit": None,
    "V18 Human Camera Pose": None,
    "DA3 Camera Pose (raw H3R geometry)": None,
    "Boundary Oracle": None,
}

COLORS = {
    "Hard Reset": (217, 70, 70),
    "Fixed Explicit": (220, 124, 25),
    "V18 Human Camera Pose": (13, 148, 136),
    "DA3 Camera Pose (raw H3R geometry)": (22, 163, 74),
    "Boundary Oracle": (37, 99, 235),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10_report", type=Path, default=base.DEFAULT_V10_REPORT)
    parser.add_argument("--candidate_dir", type=Path, default=base.DEFAULT_CANDIDATE_DIR)
    parser.add_argument(
        "--v18_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "final_candidates"
        / "v18_human_metric_translation_eval.json",
    )
    parser.add_argument(
        "--da3_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "da3_metric_depth"
        / "v18_da3_metric_depth_probe.json",
    )
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--point_stride", type=int, default=6)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--point_size", type=float, default=0.008)
    return parser.parse_args()


def normalize_v18(row: dict) -> dict:
    return {
        **row,
        "camera_translation_error_m": float(row["camera_translation_error_m"]),
        "camera_rotation_error_deg": float(row["camera_rotation_error_deg"]),
        "yaw_error_deg": float(row.get("yaw_error_deg", 0.0)),
        "pitch_error_deg": float(row.get("pitch_error_deg", 0.0)),
        "roll_error_deg": float(row.get("roll_error_deg", 0.0)),
    }


def normalize_da3(row: dict, rotation_reference: dict) -> dict:
    return {
        "transform": row["transform"],
        "camera_translation_error_m": float(row["translation_m"]),
        "camera_rotation_error_deg": float(row["rotation_deg"]),
        "yaw_error_deg": float(rotation_reference.get("yaw_error_deg", 0.0)),
        "pitch_error_deg": float(rotation_reference.get("pitch_error_deg", 0.0)),
        "roll_error_deg": float(rotation_reference.get("roll_error_deg", 0.0)),
        "bounded_residual_deg": 20.0,
    }


def normalize_v16(row: dict) -> dict:
    return {
        **row,
        "camera_translation_error_m": float(row["camera_translation_error_m"]),
        "camera_rotation_error_deg": float(row["camera_rotation_error_deg"]),
        "yaw_error_deg": float(row.get("yaw_error_deg", 0.0)),
        "pitch_error_deg": float(row.get("pitch_error_deg", 0.0)),
        "roll_error_deg": float(row.get("roll_error_deg", 0.0)),
    }


def load_maps(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    v10_report = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v10 = {str(row["case_name"]): row for row in v10_report["cases"]}
    v16 = {}
    for path in sorted(args.candidate_dir.glob("v16_candidates_shard_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            v16[str(row["case_name"])] = row
    v18_report = json.loads(args.v18_report.read_text(encoding="utf-8"))
    v18 = {str(row["case_name"]): row for row in v18_report["cases"]}
    da3_report = json.loads(args.da3_report.read_text(encoding="utf-8"))
    da3 = {str(row["case_name"]): row for row in da3_report["cases"]}

    methods = {}
    for case_name in EXAMPLES.values():
        if case_name not in v10 or case_name not in v16 or case_name not in v18 or case_name not in da3:
            raise KeyError(f"Missing cached viewer case: {case_name}")
        v18_human = normalize_v18(v18[case_name]["candidates"]["human_no_calibration"])
        methods[case_name] = {
            "viewer_methods": {
                "Hard Reset": normalize_v16(v16[case_name]["baselines"]["hard_reset"]),
                "Fixed Explicit": normalize_v18(v18[case_name]["candidates"]["fixed_explicit"]),
                "V18 Human Camera Pose": v18_human,
                "DA3 Camera Pose (raw H3R geometry)": normalize_da3(
                    da3[case_name]["candidates"]["da3_pelvis_depth"], v18_human
                ),
                "Boundary Oracle": normalize_v18(v18[case_name]["candidates"]["boundary_oracle"]),
            }
        }
    return v10, methods


def method_result(case: base.CachedCase, method: str) -> dict:
    return case.candidate["viewer_methods"][method]


class Da3ComparisonViewer(base.ComparisonViewer):
    def human_root_jump_m(self, case: base.CachedCase, result: dict) -> float:
        transform = np.asarray(result["transform"], dtype=np.float32)
        roots = np.asarray(case.visual_root_world, dtype=np.float32)
        pre_root = roots[int(self.args.boundary) - 1]
        post_root = base.transform_points(
            transform, roots[int(self.args.boundary) : int(self.args.boundary) + 1]
        )[0]
        return float(np.linalg.norm(post_root - pre_root))

    def update_metrics(self, case: base.CachedCase, result: dict, method: str) -> None:
        fixed = method_result(case, "Fixed Explicit")
        translation_gain = float(fixed["camera_translation_error_m"]) - float(
            result["camera_translation_error_m"]
        )
        rotation_gain = float(fixed["camera_rotation_error_deg"]) - float(
            result["camera_rotation_error_deg"]
        )
        root_jump = self.human_root_jump_m(case, result)
        fixed_root_jump = self.human_root_jump_m(case, fixed)
        root_jump_delta = root_jump - fixed_root_jump
        oracle_note = (
            "\n- **Oracle only fixes the camera pose; it does not repair Human3R local depth.**"
            if "Oracle" in method
            else ""
        )
        self.metrics_gui.content = (
            f"### {method}\n"
            f"- GT camera translation error: **{float(result['camera_translation_error_m']):.3f} m**\n"
            f"- GT camera rotation error: **{float(result['camera_rotation_error_deg']):.2f} deg**\n"
            f"- Visible Human3R root jump: **{root_jump:.3f} m**\n"
            f"- Yaw / pitch / roll: **{float(result['yaw_error_deg']):.2f} / "
            f"{float(result['pitch_error_deg']):.2f} / {float(result['roll_error_deg']):.2f} deg**\n"
            f"- Camera gain over Fixed: **{translation_gain:+.3f} m T, {rotation_gain:+.2f} deg R**\n"
            f"- Root-jump change vs Fixed: **{root_jump_delta:+.3f} m**"
            f"{oracle_note}"
        )

    def run(self) -> None:
        print(f"V18 DA3 Human3R 3D viewer: http://127.0.0.1:{self.args.port}", flush=True)
        while True:
            time.sleep(10.0)


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SMPL-X cache preparation but is unavailable")
    base.EXAMPLES = EXAMPLES
    base.METHODS = METHODS
    base.CAMERA_COLORS = COLORS
    base.method_result = method_result

    v10_cases, method_cases = load_maps(args)
    device = torch.device(args.device)
    layer = base.build_smpl_layer(device)
    cache: dict[str, base.CachedCase] = {}
    for label, case_name in EXAMPLES.items():
        print(f"Preparing DA3 interactive scene: {label}", flush=True)
        cached_case = base.load_cached_case(
            case_name,
            v10_cases[case_name],
            method_cases[case_name],
            layer,
            device,
            args,
        )
        local_dir = Path(v10_cases[case_name]["paths"]["human3r_local_reset"])
        visual_roots = []
        for frame in range(4):
            with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
                root_camera = np.asarray(smpl["transl"][0], dtype=np.float32)
            pose = cached_case.camera_poses[frame]
            visual_roots.append(pose[:3, :3] @ root_camera + pose[:3, 3])
        cached_case.visual_root_world = np.stack(visual_roots).astype(np.float32)
        cache[label] = cached_case
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    del layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    viewer = Da3ComparisonViewer(cache, faces, args)
    viewer.run()


if __name__ == "__main__":
    main()
