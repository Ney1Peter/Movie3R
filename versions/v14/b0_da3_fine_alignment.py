"""Deployable B0-centered fine Boundary from bidirectional DA3 shared poses.

This module contains no evaluator, dataset, identity, or ground-truth dependency.
It consumes frozen B0, Human3R pre/raw-post camera poses, and two DA3 relative-pose
predictions. It returns exactly one shared SE(3), with exact B0 fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FineAlignmentConfig:
    rotation_cap_deg: float = 3.0
    direction_cap_deg: float = 5.0
    rotation_spread_limit_deg: float = 5.0
    direction_spread_limit_deg: float = 5.0
    right_rotation_limit_deg: float = 15.0
    direction_vs_b0_limit_deg: float = 30.0


DEFAULT_CONFIG = FineAlignmentConfig()


class DA3FineAligner:
    """Deployment adapter from two RGB frames to one safe shared Boundary.

    ``model`` is a loaded frozen ``DepthAnything3`` instance. Keeping model
    construction outside this geometry module avoids a hard import dependency
    for unit tests and for B0-only fallback deployments.
    """

    def __init__(
        self,
        model,
        config: FineAlignmentConfig = DEFAULT_CONFIG,
        process_res: int = 504,
        use_ray_pose: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.process_res = int(process_res)
        self.use_ray_pose = bool(use_ray_pose)

    @staticmethod
    def camera_to_world_from_prediction(prediction) -> np.ndarray:
        extrinsics = getattr(prediction, "extrinsics", None)
        if extrinsics is None:
            raise ValueError("DA3 prediction has no camera extrinsics")
        extrinsics = np.stack([homogeneous(row) for row in np.asarray(extrinsics)])
        if extrinsics.shape != (2, 4, 4):
            raise ValueError(
                f"DA3 pair must contain two extrinsics, got {extrinsics.shape}"
            )
        if not np.isfinite(extrinsics).all():
            raise ValueError("DA3 prediction contains non-finite extrinsics")
        return np.linalg.inv(extrinsics)

    def _predict_pair(self, first_rgb, second_rgb) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        prediction = self.model.inference(
            [first_rgb, second_rgb],
            process_res=self.process_res,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy="first",
        )
        return self.camera_to_world_from_prediction(prediction), (
            time.perf_counter() - started
        )

    def refine_images(
        self,
        b0: np.ndarray,
        pre_pose: np.ndarray,
        raw_post_pose: np.ndarray,
        pre_rgb,
        post_rgb,
    ) -> tuple[np.ndarray, dict]:
        """Run bidirectional frozen DA3 and return a bounded Boundary or B0."""
        original_b0 = np.asarray(b0)
        fallback_b0 = (
            original_b0.copy()
            if original_b0.shape == (4, 4)
            else homogeneous(original_b0)
        )
        try:
            forward, forward_seconds = self._predict_pair(pre_rgb, post_rgb)
            reverse, reverse_seconds = self._predict_pair(post_rgb, pre_rgb)
        except Exception as error:  # external inference failures must be fail-safe
            return fallback_b0.copy(), {
                "accepted": False,
                "selected": "b0",
                "config": asdict(self.config),
                "reason": f"DA3 inference failure: {error}",
            }
        boundary, diagnostics = refine_b0_with_da3(
            fallback_b0,
            pre_pose,
            raw_post_pose,
            forward,
            reverse,
            self.config,
        )
        diagnostics.update(
            {
                "da3_forward_seconds": float(forward_seconds),
                "da3_reverse_seconds": float(reverse_seconds),
            }
        )
        return boundary, diagnostics


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected 3x4 or 4x4 pose, got {matrix.shape}")
    output = np.eye(4, dtype=np.float64)
    output[:3] = matrix
    return output


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    output[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return output


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = homogeneous(first)[:3, :3].T @ homogeneous(second)[:3, :3]
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0])))


def direction_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        raise ValueError("Cannot compare a zero-length direction")
    cosine = float(np.clip((first @ second) / (first_norm * second_norm), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("Rotation consensus requires at least one proposal")
    value = np.sum(
        [np.asarray(rotation, dtype=np.float64).reshape(3, 3) for rotation in rotations],
        axis=0,
    )
    left, _, right = np.linalg.svd(value)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(left @ right)
    return left @ correction @ right


def mean_direction(directions: list[np.ndarray]) -> np.ndarray:
    if not directions:
        raise ValueError("Direction consensus requires at least one proposal")
    rows = []
    for direction in directions:
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            raise ValueError("Direction consensus received a zero vector")
        rows.append(direction / norm)
    output = np.sum(rows, axis=0)
    if float(np.linalg.norm(output)) < 1e-12:
        raise ValueError("Direction proposals cancel")
    return output / np.linalg.norm(output)


def rotate_direction_toward(
    first: np.ndarray, second: np.ndarray, maximum_deg: float
) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        raise ValueError("Cannot interpolate a zero-length direction")
    first = first / first_norm
    second = second / second_norm
    angle = math.acos(float(np.clip(first @ second, -1.0, 1.0)))
    step = min(angle, math.radians(float(maximum_deg)))
    if step < 1e-12:
        return first
    axis = np.cross(first, second)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(first)))] = 1.0
        axis = np.cross(first, basis)
        axis_norm = float(np.linalg.norm(axis))
    return cv2.Rodrigues(axis / axis_norm * step)[0] @ first


def da3_proposal(
    pre_pose: np.ndarray,
    raw_post_pose: np.ndarray,
    camera_to_world: np.ndarray,
    reverse: bool,
) -> dict:
    camera_to_world = np.stack(
        [homogeneous(pose) for pose in np.asarray(camera_to_world)]
    )
    if camera_to_world.shape != (2, 4, 4):
        raise ValueError(
            f"DA3 pair must contain two camera poses, got {camera_to_world.shape}"
        )
    pre_index, post_index = ((1, 0) if reverse else (0, 1))
    da3_pre = camera_to_world[pre_index]
    da3_post = camera_to_world[post_index]
    pre_pose = homogeneous(pre_pose)
    raw_post_pose = homogeneous(raw_post_pose)
    world_from_da3 = pre_pose @ np.linalg.inv(da3_pre)
    desired_post_pose = world_from_da3 @ da3_post
    boundary = desired_post_pose @ np.linalg.inv(raw_post_pose)
    baseline = desired_post_pose[:3, 3] - pre_pose[:3, 3]
    baseline_norm = float(np.linalg.norm(baseline))
    if baseline_norm < 1e-12:
        raise ValueError("DA3 predicts a degenerate camera baseline")
    return {
        "boundary_rotation": boundary[:3, :3],
        "baseline_direction_world": baseline / baseline_norm,
        "da3_baseline_units": baseline_norm,
    }


def gate_decision(diagnostics: dict, config: FineAlignmentConfig) -> bool:
    required = (
        "forward_reverse_rotation_spread_deg",
        "forward_reverse_direction_spread_deg",
        "right_rotation_deg",
        "direction_vs_b0_deg",
    )
    if any(key not in diagnostics for key in required):
        return False
    values = np.asarray([diagnostics[key] for key in required], dtype=np.float64)
    if not np.isfinite(values).all():
        return False
    return bool(
        diagnostics["forward_reverse_rotation_spread_deg"]
        <= config.rotation_spread_limit_deg
        and diagnostics["forward_reverse_direction_spread_deg"]
        <= config.direction_spread_limit_deg
        and diagnostics["right_rotation_deg"] <= config.right_rotation_limit_deg
        and diagnostics["direction_vs_b0_deg"] <= config.direction_vs_b0_limit_deg
    )


def b0_camera_center(b0: np.ndarray, raw_post_pose: np.ndarray) -> np.ndarray:
    b0 = homogeneous(b0)
    raw_post_pose = homogeneous(raw_post_pose)
    return b0[:3, :3] @ raw_post_pose[:3, 3] + b0[:3, 3]


def boundary_from_camera_center(
    raw_post_pose: np.ndarray,
    boundary_rotation: np.ndarray,
    camera_center: np.ndarray,
) -> np.ndarray:
    raw_post_pose = homogeneous(raw_post_pose)
    rotation = np.asarray(boundary_rotation, dtype=np.float64).reshape(3, 3)
    camera_center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    return transform(
        rotation,
        camera_center - rotation @ raw_post_pose[:3, 3],
    )


def refine_b0_with_da3(
    b0: np.ndarray,
    pre_pose: np.ndarray,
    raw_post_pose: np.ndarray,
    forward_camera_to_world: np.ndarray | None,
    reverse_camera_to_world: np.ndarray | None,
    config: FineAlignmentConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, dict]:
    """Return one bounded shared Boundary, or an exact copy of B0 on any failure."""
    original_b0 = np.asarray(b0)
    exact_b0 = (
        original_b0.copy()
        if original_b0.shape == (4, 4)
        else homogeneous(original_b0)
    )
    b0 = homogeneous(original_b0)
    fallback = {
        "accepted": False,
        "selected": "b0",
        "config": asdict(config),
    }
    try:
        pre_pose = homogeneous(pre_pose)
        raw_post_pose = homogeneous(raw_post_pose)
        arrays = np.concatenate(
            [
                b0.reshape(-1),
                pre_pose.reshape(-1),
                raw_post_pose.reshape(-1),
            ]
        )
        if not np.isfinite(arrays).all():
            raise ValueError("Non-finite B0 or Human3R pose")
        if forward_camera_to_world is None or reverse_camera_to_world is None:
            raise ValueError("Missing bidirectional DA3 pose")
        forward = da3_proposal(
            pre_pose, raw_post_pose, forward_camera_to_world, reverse=False
        )
        reverse = da3_proposal(
            pre_pose, raw_post_pose, reverse_camera_to_world, reverse=True
        )
        proposal_rotation = mean_rotation(
            [forward["boundary_rotation"], reverse["boundary_rotation"]]
        )
        proposal_direction = mean_direction(
            [
                forward["baseline_direction_world"],
                reverse["baseline_direction_world"],
            ]
        )
        coarse_center = b0_camera_center(b0, raw_post_pose)
        coarse_baseline = coarse_center - pre_pose[:3, 3]
        coarse_length = float(np.linalg.norm(coarse_baseline))
        if coarse_length < 1e-12:
            raise ValueError("B0 camera baseline is degenerate")
        right_rotation = b0[:3, :3].T @ proposal_rotation
        right_rotvec = cv2.Rodrigues(right_rotation)[0].reshape(3)
        diagnostics = {
            **fallback,
            "forward_reverse_rotation_spread_deg": rotation_angle_deg(
                transform(forward["boundary_rotation"], np.zeros(3)),
                transform(reverse["boundary_rotation"], np.zeros(3)),
            ),
            "forward_reverse_direction_spread_deg": direction_angle_deg(
                forward["baseline_direction_world"],
                reverse["baseline_direction_world"],
            ),
            "right_rotation_deg": float(
                np.degrees(np.linalg.norm(right_rotvec))
            ),
            "direction_vs_b0_deg": direction_angle_deg(
                coarse_baseline, proposal_direction
            ),
            "coarse_baseline_length": coarse_length,
            "da3_forward_baseline_units": float(
                forward["da3_baseline_units"]
            ),
            "da3_reverse_baseline_units": float(
                reverse["da3_baseline_units"]
            ),
        }
        if not gate_decision(diagnostics, config):
            diagnostics["reason"] = "agreement_or_b0_prior_gate"
            return exact_b0.copy(), diagnostics

        rotation_cap = math.radians(float(config.rotation_cap_deg))
        rotation_norm = float(np.linalg.norm(right_rotvec))
        bounded_vector = right_rotvec * min(
            1.0, rotation_cap / max(rotation_norm, 1e-12)
        )
        boundary_rotation = b0[:3, :3] @ cv2.Rodrigues(bounded_vector)[0]
        direction = rotate_direction_toward(
            coarse_baseline, proposal_direction, config.direction_cap_deg
        )
        camera_center = pre_pose[:3, 3] + coarse_length * direction
        boundary = boundary_from_camera_center(
            raw_post_pose, boundary_rotation, camera_center
        )
        diagnostics.update(
            {
                "accepted": True,
                "selected": "da3_bounded_consensus",
                "applied_rotation_deg": float(
                    np.degrees(np.linalg.norm(bounded_vector))
                ),
                "applied_direction_deg": direction_angle_deg(
                    coarse_baseline, direction
                ),
            }
        )
        return boundary, diagnostics
    except (
        TypeError,
        ValueError,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
        cv2.error,
    ) as error:
        fallback["reason"] = str(error)
        return exact_b0.copy(), fallback


def apply_boundary_to_pose(boundary: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return homogeneous(boundary) @ homogeneous(pose)


def apply_boundary_to_points(
    boundary: np.ndarray, points: np.ndarray
) -> np.ndarray:
    boundary = homogeneous(boundary)
    points = np.asarray(points, dtype=np.float64)
    return points @ boundary[:3, :3].T + boundary[:3, 3]
