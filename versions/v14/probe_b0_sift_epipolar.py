#!/usr/bin/env python3
"""Test background SIFT epipolar pose as a B0-centered fine cue.

The method reads only the last pre-cut RGB frame, first post-cut RGB frame, predicted
human boxes, nominal intrinsics, cached Human3R poses, and frozen B0. GT is evaluation
only. Essential-matrix translation is used only as a direction: B0 keeps its metric
baseline length and rotation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_residual_observability import (  # noqa: E402
    evaluate_boundary,
    gt_boundary,
    right_residual,
    serializable,
    transform,
    vector_stats,
)
from versions.v14.run_v14_2_multihuman_sequence import solution  # noqa: E402,F401


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/sift_epipolar"
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences", nargs="+", choices=tuple(SEQUENCE_INPUTS), default=("three",)
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--max_features", type=int, default=6000)
    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--ransac_threshold_px", type=float, default=1.5)
    parser.add_argument("--human_margin", type=float, default=0.12)
    parser.add_argument("--skip_identity_rebuild", action="store_true")
    return parser.parse_args()


class FrameReader:
    def __init__(self, args: argparse.Namespace, sequence: str):
        self.args = args
        self.sequence = sequence
        self.captures = {}

    def read(self, camera: int, frame: int) -> np.ndarray:
        if camera not in self.captures:
            helper_args = SimpleNamespace(data_root=self.args.data_root, sequence=self.sequence)
            capture = cv2.VideoCapture(str(geometry.video_path(helper_args, camera)))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video for {self.sequence} camera {camera}")
            self.captures[camera] = capture
        capture = self.captures[camera]
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(
                f"Cannot read {self.sequence} camera {camera} frame {frame}"
            )
        resolution = int(self.args.resolution)
        return cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()


def background_mask(
    humans: dict[str, dict], resolution: int, margin_fraction: float
) -> np.ndarray:
    mask = np.full((resolution, resolution), 255, dtype=np.uint8)
    for human in humans.values():
        bbox = np.asarray(human["bbox"], dtype=np.float64) * (resolution / 512.0)
        width = max(float(bbox[2] - bbox[0]), 1.0)
        height = max(float(bbox[3] - bbox[1]), 1.0)
        margin = margin_fraction * np.asarray((width, height), dtype=np.float64)
        bbox[:2] -= margin
        bbox[2:] += margin
        lower = np.maximum(np.floor(bbox[:2]).astype(int), 0)
        upper = np.minimum(np.ceil(bbox[2:]).astype(int), resolution - 1)
        cv2.rectangle(mask, tuple(lower), tuple(upper), 0, thickness=-1)
    return mask


def nominal_intrinsics(resolution: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * resolution / math.tan(0.5 * math.radians(float(fov_deg)))
    return np.asarray(
        [[focal, 0.0, resolution / 2.0], [0.0, focal, resolution / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def ratio_matches(first, second, ratio: float) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(first, second, k=2)
    return [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


def mutual_ratio_matches(
    first: np.ndarray, second: np.ndarray, ratio: float
) -> list[cv2.DMatch]:
    forward = ratio_matches(first, second, ratio)
    reverse = ratio_matches(second, first, ratio)
    reverse_pairs = {(match.trainIdx, match.queryIdx) for match in reverse}
    return [
        match
        for match in forward
        if (int(match.queryIdx), int(match.trainIdx)) in reverse_pairs
    ]


def estimate_rotation(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    sift = cv2.SIFT_create(
        nfeatures=int(args.max_features),
        contrastThreshold=0.01,
        edgeThreshold=12,
    )
    first_keypoints, first_desc = sift.detectAndCompute(
        cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY), first_mask
    )
    second_keypoints, second_desc = sift.detectAndCompute(
        cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY), second_mask
    )
    if first_desc is None or second_desc is None:
        return {"status": "no_descriptors"}
    matches = mutual_ratio_matches(first_desc, second_desc, float(args.ratio))
    if len(matches) < 8:
        return {
            "status": "too_few_matches",
            "first_keypoints": len(first_keypoints),
            "second_keypoints": len(second_keypoints),
            "matches": len(matches),
        }
    first_points = np.float64([first_keypoints[match.queryIdx].pt for match in matches])
    second_points = np.float64([second_keypoints[match.trainIdx].pt for match in matches])
    intrinsics = nominal_intrinsics(int(args.resolution), float(args.fov_deg))
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    essential, mask = cv2.findEssentialMat(
        first_points,
        second_points,
        intrinsics,
        method=method,
        prob=0.999,
        threshold=float(args.ransac_threshold_px),
    )
    if essential is None or mask is None:
        return {"status": "essential_failed", "matches": len(matches)}
    if essential.shape != (3, 3):
        essential = essential[:3]
    inlier = mask.reshape(-1).astype(bool)
    if int(np.sum(inlier)) < 8:
        return {
            "status": "too_few_inliers",
            "matches": len(matches),
            "essential_inliers": int(np.sum(inlier)),
        }
    count, rotation_21, translation_direction, pose_mask = cv2.recoverPose(
        essential,
        first_points,
        second_points,
        intrinsics,
        mask=mask,
    )
    pose_inlier = pose_mask.reshape(-1).astype(bool)
    selected = first_points[pose_inlier]
    selected_second = second_points[pose_inlier]
    displacement = np.linalg.norm(selected_second - selected, axis=1)
    return {
        "status": "ok",
        "first_keypoints": len(first_keypoints),
        "second_keypoints": len(second_keypoints),
        "matches": len(matches),
        "essential_inliers": int(np.sum(inlier)),
        "pose_inliers": int(count),
        "inlier_ratio": float(count / max(len(matches), 1)),
        "median_pixel_displacement": float(np.median(displacement)),
        "rotation_21": np.asarray(rotation_21, dtype=np.float64),
        "translation_direction_21": np.asarray(translation_direction).reshape(3),
    }


def candidate_boundary_rotation(cache: dict, estimate: dict) -> np.ndarray:
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    rotation_21 = np.asarray(estimate["rotation_21"], dtype=np.float64)
    return pre_pose[:3, :3] @ rotation_21.T @ post_pose[:3, :3].T


def keep_post_camera_center(
    cache: dict, b0: np.ndarray, boundary_rotation: np.ndarray
) -> np.ndarray:
    """Change Boundary rotation while preserving the B0 first-post camera center."""
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    coarse_camera_center = geometry.transform_points(
        b0, post_pose[:3, 3][None]
    )[0]
    translation = coarse_camera_center - boundary_rotation @ post_pose[:3, 3]
    return transform(boundary_rotation, translation)


def slerp_direction(
    first: np.ndarray, second: np.ndarray, maximum_deg: float
) -> np.ndarray:
    """Rotate one unit direction toward another inside an angular trust region."""
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        raise ValueError("Cannot interpolate a zero-length direction")
    first = first / first_norm
    second = second / second_norm
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    angle = float(math.acos(cosine))
    step = min(angle, math.radians(float(maximum_deg)))
    if step < 1e-12:
        return first
    axis = np.cross(first, second)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        # The antipodal case has no unique great circle. Pick a deterministic axis.
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(first)))] = 1.0
        axis = np.cross(first, basis)
        axis_norm = float(np.linalg.norm(axis))
    axis = axis / axis_norm
    rotation = cv2.Rodrigues(axis * step)[0]
    return rotation @ first


def boundary_from_camera_center(
    cache: dict, boundary_rotation: np.ndarray, camera_center: np.ndarray
) -> np.ndarray:
    """Construct Boundary translation for a requested first-post camera center."""
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    camera_center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    translation = camera_center - boundary_rotation @ post_pose[:3, 3]
    return transform(boundary_rotation, translation)


def load_case(
    sequence: str,
    report_case: dict,
    reader: FrameReader,
    args: argparse.Namespace,
) -> dict:
    inputs = SEQUENCE_INPUTS[sequence]
    cache_path = inputs["cache"] / f"{report_case['case']['key']}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    if not args.skip_identity_rebuild:
        cache = geometry.reassign_cache_gt_identities(
            SimpleNamespace(data_root=args.data_root, size=512, sequence=sequence), cache
        )
    case = report_case["case"]
    first = reader.read(int(case["source_camera"]), int(case["pre_frames"][-1]))
    second = reader.read(int(case["target_camera"]), int(case["post_frame"]))
    resolution = int(args.resolution)
    first_mask = background_mask(cache["humans"][-2], resolution, float(args.human_margin))
    second_mask = background_mask(cache["humans"][-1], resolution, float(args.human_margin))
    estimate = estimate_rotation(first, second, first_mask, second_mask, args)
    b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
    target = gt_boundary(cache)
    identities = tuple(
        identity for identity in geometry.IDENTITIES if identity in cache["humans"][-1]
    )
    methods = {"b0": evaluate_boundary(cache, b0, identities)}
    if estimate["status"] == "ok":
        candidate_rotation = candidate_boundary_rotation(cache, estimate)
        right_rotation = b0[:3, :3].T @ candidate_rotation
        right_rotvec = cv2.Rodrigues(right_rotation)[0].reshape(3)
        residual_angle = float(np.degrees(np.linalg.norm(right_rotvec)))
        candidate = b0 @ transform(right_rotation, np.zeros(3))
        methods["sift_rotation"] = evaluate_boundary(cache, candidate, identities)
        centered_candidate = keep_post_camera_center(cache, b0, candidate_rotation)
        methods["sift_center_rotation"] = evaluate_boundary(
            cache, centered_candidate, identities
        )
        for maximum_deg in (1, 2, 3, 5, 10):
            maximum = math.radians(float(maximum_deg))
            norm = float(np.linalg.norm(right_rotvec))
            bounded = right_rotvec * min(1.0, maximum / max(norm, 1e-12))
            bounded_rotation = cv2.Rodrigues(bounded)[0]
            methods[f"sift_rotation_b{maximum_deg}"] = evaluate_boundary(
                cache,
                b0 @ transform(bounded_rotation, np.zeros(3)),
                identities,
            )
            bounded_boundary_rotation = b0[:3, :3] @ bounded_rotation
            methods[f"sift_center_rotation_b{maximum_deg}"] = evaluate_boundary(
                cache,
                keep_post_camera_center(cache, b0, bounded_boundary_rotation),
                identities,
            )

        # recoverPose translation has unknown scale. Use only its camera-baseline
        # direction, retain B0's metric baseline length and Boundary rotation, and
        # bound the angular change of that direction around B0.
        pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
        post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
        coarse_post_center = geometry.transform_points(
            b0, post_pose[:3, 3][None]
        )[0]
        coarse_baseline = coarse_post_center - pre_pose[:3, 3]
        sift_post_orientation = candidate_rotation @ post_pose[:3, :3]
        sift_baseline_direction = -sift_post_orientation @ np.asarray(
            estimate["translation_direction_21"], dtype=np.float64
        )
        if (
            float(np.linalg.norm(coarse_baseline)) > 1e-12
            and float(np.linalg.norm(sift_baseline_direction)) > 1e-12
        ):
            coarse_length = float(np.linalg.norm(coarse_baseline))
            for maximum_deg in (2, 5, 10, 20, 45):
                direction = slerp_direction(
                    coarse_baseline, sift_baseline_direction, maximum_deg
                )
                camera_center = pre_pose[:3, 3] + coarse_length * direction
                candidate = boundary_from_camera_center(
                    cache, b0[:3, :3], camera_center
                )
                methods[f"sift_direction_b{maximum_deg}"] = evaluate_boundary(
                    cache, candidate, identities
                )
            direction = sift_baseline_direction / np.linalg.norm(
                sift_baseline_direction
            )
            camera_center = pre_pose[:3, 3] + coarse_length * direction
            methods["sift_direction"] = evaluate_boundary(
                cache,
                boundary_from_camera_center(cache, b0[:3, :3], camera_center),
                identities,
            )
            estimate["coarse_baseline_length_m"] = coarse_length
            estimate["sift_vs_b0_direction_deg"] = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(coarse_baseline / coarse_length, direction),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        estimate["candidate_boundary_rotation"] = candidate_rotation
        estimate["right_residual_deg"] = residual_angle
        estimate["candidate_vs_gt_rotation_deg"] = geometry.rotation_error_deg(
            candidate, target
        )
        _, target_du = right_residual(b0, target)
        estimate["gt_right_du"] = target_du
    return {
        "sequence": sequence,
        "case": case,
        "camera_span_deg": float(report_case["camera_span_deg"]),
        "estimate": estimate,
        "methods": methods,
    }


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if "sift_rotation" in row["methods"]]
    output = {
        "case_count": len(rows),
        "valid_cases": len(valid),
        "success_rate": float(len(valid) / max(len(rows), 1)),
    }
    method_names = sorted(set().union(*(set(row["methods"]) for row in valid)))
    for method in method_names:
        method_rows = [row for row in valid if method in row["methods"]]
        output[method] = {
            metric: vector_stats([row["methods"][method][metric] for row in method_rows])
            for metric in (
                "camera_translation_error_m",
                "camera_rotation_error_deg",
                "camera_composite",
                "human_root_error_m",
            )
        }
    if valid:
        output["paired"] = {}
        for method in method_names:
            if method == "b0":
                continue
            method_rows = [row for row in valid if method in row["methods"]]
            output["paired"][method] = {
                metric: {
                    "mean_delta": float(
                        np.mean(
                            [
                                row["methods"][method][metric]
                                - row["methods"]["b0"][metric]
                                for row in method_rows
                            ]
                        )
                    ),
                    "improvement_rate": float(
                        np.mean(
                            [
                                row["methods"][method][metric]
                                < row["methods"]["b0"][metric]
                                for row in method_rows
                            ]
                        )
                    ),
                }
                for metric in (
                    "camera_rotation_error_deg",
                    "camera_composite",
                    "human_root_error_m",
                )
            }
        output["reliability"] = {
            key: vector_stats([row["estimate"][key] for row in valid])
            for key in (
                "matches",
                "pose_inliers",
                "inlier_ratio",
                "median_pixel_displacement",
                "right_residual_deg",
                "candidate_vs_gt_rotation_deg",
            )
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    lines = [
        "# V14 B0 SIFT Epipolar Pose Probe",
        "",
        f"Sequences: `{', '.join(report['sequences'])}`. "
        f"Valid: `{summary['valid_cases']}/{summary['case_count']}`.",
        "",
        "GT is evaluation only. The proposal uses background SIFT matches and nominal "
        "intrinsics. Essential translation contributes only a direction; B0 retains "
        "its metric baseline length and rotation.",
        "",
        "| Method | Camera T | Camera R | Composite | Human root |",
        "|---|---:|---:|---:|---:|",
    ]
    method_names = [
        "b0",
        "sift_center_rotation_b1",
        "sift_center_rotation_b2",
        "sift_center_rotation_b3",
        "sift_center_rotation_b5",
        "sift_center_rotation_b10",
        "sift_center_rotation",
        "sift_rotation_b1",
        "sift_rotation_b2",
        "sift_rotation_b3",
        "sift_rotation_b5",
        "sift_rotation_b10",
        "sift_rotation",
        "sift_direction_b2",
        "sift_direction_b5",
        "sift_direction_b10",
        "sift_direction_b20",
        "sift_direction_b45",
        "sift_direction",
    ]
    for method in method_names:
        if method not in summary:
            continue
        values = summary[method]
        lines.append(
            f"| {method} | {values['camera_translation_error_m']['mean']:.4f} | "
            f"{values['camera_rotation_error_deg']['mean']:.3f} | "
            f"{values['camera_composite']['mean']:.4f} | "
            f"{values['human_root_error_m']['mean']:.4f} |"
        )
    if "paired" in summary:
        lines.extend(["", "## Paired Against B0", ""])
        for method, metrics in summary["paired"].items():
            for metric, values in metrics.items():
                lines.append(
                    f"- `{method}` / `{metric}`: mean delta "
                    f"`{values['mean_delta']:.4f}`, improve rate "
                    f"`{100 * values['improvement_rate']:.1f}%`."
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence in args.sequences:
        reader = FrameReader(args, sequence)
        try:
            report = json.loads(SEQUENCE_INPUTS[sequence]["report"].read_text())
            report_cases = report["cases"][: args.max_cases or None]
            for index, report_case in enumerate(report_cases, start=1):
                row = load_case(sequence, report_case, reader, args)
                rows.append(row)
                print(
                    f"[{sequence} {index:03d}/{len(report_cases):03d}] "
                    f"{report_case['case']['key']} {row['estimate']['status']}",
                    flush=True,
                )
        finally:
            reader.close()
    report = {
        "experiment": "v14_b0_sift_epipolar_pose",
        "sequences": list(args.sequences),
        "protocol": {
            "rgb": "last pre-cut plus first post-cut frame",
            "human_mask": "predicted Human3R boxes only",
            "intrinsics": f"nominal {args.fov_deg:g}-degree FOV",
            "gt_usage": "strict identity rebuild and evaluation only",
            "translation": "direction only; B0 baseline length and rotation retained",
        },
        "parameters": {
            "resolution": int(args.resolution),
            "fov_deg": float(args.fov_deg),
            "max_features": int(args.max_features),
            "ratio": float(args.ratio),
            "ransac_threshold_px": float(args.ransac_threshold_px),
            "human_margin": float(args.human_margin),
        },
        "summary": summarize(rows),
        "cases": rows,
    }
    json_path = args.output_dir / "v14_b0_sift_epipolar.json"
    md_path = args.output_dir / "v14_b0_sift_epipolar.md"
    json_path.write_text(json.dumps(serializable(report), indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
