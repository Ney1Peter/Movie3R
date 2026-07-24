import numpy as np
from scipy.spatial.transform import Rotation

from versions.v12.experiments.v14_2_canonical_human_memory_probe import (
    blend_rotations,
    boundary_from_camera_pose,
    camera_pose_from_human,
    direction_error_deg,
    ema,
    physical_scale,
    transform_point,
)


def test_ema_is_causal_and_uses_requested_update_weight():
    values = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32)
    assert np.allclose(ema(values, 0.25), np.asarray([1.375], dtype=np.float32))


def test_physical_scale_changes_linearly_with_uniform_body_scale():
    joints = np.zeros((22, 3), dtype=np.float32)
    for index in range(len(joints)):
        joints[index] = np.asarray([0.1 * index, 0.2 * index, -0.05 * index])
    assert np.isclose(physical_scale(2.5 * joints), 2.5 * physical_scale(joints))


def test_camera_translation_equation_recovers_world_root():
    rotation = Rotation.from_euler("xyz", [15.0, -20.0, 5.0], degrees=True).as_matrix()
    world_root = np.asarray([1.2, -0.4, 3.1], dtype=np.float32)
    camera_root = np.asarray([0.2, 0.1, 2.0], dtype=np.float32)
    camera_pose = camera_pose_from_human(rotation, world_root, camera_root)
    assert np.allclose(transform_point(camera_pose, camera_root), world_root, atol=1e-6)


def test_boundary_is_left_multiplication_in_camera_to_world_convention():
    local = np.eye(4, dtype=np.float32)
    local[:3, 3] = np.asarray([0.2, 0.3, 0.4])
    target = np.eye(4, dtype=np.float32)
    target[:3, 3] = np.asarray([1.0, -2.0, 3.0])
    boundary = boundary_from_camera_pose(target, local)
    assert np.allclose(boundary @ local, target, atol=1e-6)


def test_rotation_blend_projects_to_so3():
    first = np.eye(3, dtype=np.float32)[None]
    second = Rotation.from_euler("y", 35.0, degrees=True).as_matrix().astype(np.float32)[None]
    blended = blend_rotations(first, second, 0.25)
    assert np.allclose(blended.transpose(0, 2, 1) @ blended, np.eye(3)[None], atol=1e-5)
    assert np.all(np.linalg.det(blended) > 0.999)


def test_direction_error_handles_parallel_vectors():
    assert np.isclose(direction_error_deg(np.asarray([1.0, 0.0, 0.0]), np.asarray([2.0, 0.0, 0.0])), 0.0)
