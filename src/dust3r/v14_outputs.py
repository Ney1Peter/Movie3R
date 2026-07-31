"""Output-side shared Boundary operations for causal V14 segment alignment."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _batched_boundary(boundary: torch.Tensor, batch_size: int) -> torch.Tensor:
    if boundary.ndim == 2:
        boundary = boundary.unsqueeze(0)
    if boundary.ndim != 3 or boundary.shape[-2:] != (4, 4):
        raise ValueError(f"Boundary must have shape [4,4] or [B,4,4], got {boundary.shape}")
    if boundary.shape[0] == 1 and batch_size > 1:
        boundary = boundary.expand(batch_size, -1, -1)
    if boundary.shape[0] != batch_size:
        raise ValueError(
            f"Boundary batch {boundary.shape[0]} does not match prediction batch {batch_size}"
        )
    return boundary


def boundary_from_camera_predictions(
    shadow_prediction: Mapping[str, torch.Tensor],
    raw_prediction: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return B with ``C_shadow = B @ C_raw`` in camera-to-world convention."""

    from dust3r.utils.camera import pose_encoding_to_camera

    if "camera_pose" not in shadow_prediction or "camera_pose" not in raw_prediction:
        raise KeyError("Both predictions must contain camera_pose")
    shadow_camera = pose_encoding_to_camera(shadow_prediction["camera_pose"].float())
    raw_camera = pose_encoding_to_camera(raw_prediction["camera_pose"].float())
    if shadow_camera.shape != raw_camera.shape:
        raise ValueError(
            f"Shadow/raw camera shapes differ: {shadow_camera.shape} vs {raw_camera.shape}"
        )
    return shadow_camera @ torch.linalg.inv(raw_camera)


def apply_boundary_to_camera_encoding(
    camera_pose: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    """Left-multiply a camera-to-world pose encoding by a shared Boundary."""

    from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera

    camera = pose_encoding_to_camera(camera_pose.float())
    boundary = _batched_boundary(boundary.to(camera), camera.shape[0])
    transformed = boundary @ camera
    return camera_to_pose_encoding(transformed).to(dtype=camera_pose.dtype)


def apply_boundary_to_prediction(
    prediction: Mapping[str, torch.Tensor],
    boundary: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply one Boundary to world camera/pointmap while preserving local SMPL-X."""

    from dust3r.utils.geometry import geotrf

    if "camera_pose" not in prediction:
        raise KeyError("Prediction must contain camera_pose")
    transformed = dict(prediction)
    transformed["camera_pose"] = apply_boundary_to_camera_encoding(
        prediction["camera_pose"], boundary
    )

    world_points = prediction.get("pts3d_in_other_view")
    if torch.is_tensor(world_points):
        batched = _batched_boundary(boundary.to(world_points), world_points.shape[0])
        transformed["pts3d_in_other_view"] = geotrf(batched, world_points.float()).to(
            dtype=world_points.dtype
        )

    transformed["v14_shared_boundary"] = _batched_boundary(
        boundary.to(transformed["camera_pose"]), transformed["camera_pose"].shape[0]
    ).clone()
    return transformed
