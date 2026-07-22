import numpy as np
from scipy.spatial.transform import Rotation

from scripts.v14_3_projection_consistent_reanchoring_probe import (
    boundary_from_camera_pose,
    camera_pose_from_human,
    mesh_bbox_metrics,
    root_from_pixel_depth,
    safe_pearson,
    transform_point,
)


def test_coupled_camera_and_root_close_exactly_in_world():
    rotation = Rotation.from_euler("xyz", [8.0, -41.0, 3.0], degrees=True).as_matrix()
    world_root = np.asarray([0.4, 1.2, -0.7], dtype=np.float32)
    camera_root = np.asarray([0.2, -0.1, 2.4], dtype=np.float32)
    camera_pose = camera_pose_from_human(rotation, world_root, camera_root)
    assert np.allclose(transform_point(camera_pose, camera_root), world_root, atol=1e-6)


def test_boundary_maps_fresh_camera_pose_to_coupled_pose():
    fresh = np.eye(4, dtype=np.float32)
    fresh[:3, 3] = np.asarray([0.2, 0.3, -0.1])
    coupled = np.eye(4, dtype=np.float32)
    coupled[:3, 3] = np.asarray([-1.0, 0.4, 2.0])
    boundary = boundary_from_camera_pose(coupled, fresh)
    assert np.allclose(boundary @ fresh, coupled, atol=1e-6)


def test_pixel_depth_backprojection_matches_requested_pixel():
    K = np.asarray([[500.0, 0.0, 184.0], [0.0, 510.0, 256.0], [0.0, 0.0, 1.0]])
    pixel = np.asarray([204.0, 236.0])
    root = root_from_pixel_depth(pixel, 2.5, K)
    projected = np.asarray(
        [K[0, 0] * root[0] / root[2] + K[0, 2], K[1, 1] * root[1] / root[2] + K[1, 2]]
    )
    assert np.allclose(projected, pixel, atol=1e-6)


def test_safe_pearson_handles_constant_metric_without_warning():
    first = np.asarray([1.0, 2.0, 3.0])
    second = np.asarray([0.5, 0.5, 0.5])
    assert np.isnan(safe_pearson(first, second))


def test_mesh_bbox_metrics_matches_identical_projected_box():
    vertices = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    metrics = mesh_bbox_metrics(vertices, np.zeros(3), np.asarray([0.0, 0.0, 1.0, 1.0]), np.eye(3))
    assert np.isclose(metrics["iou"], 1.0)
    assert np.isclose(metrics["width_ratio"], 1.0)
    assert np.isclose(metrics["height_ratio"], 1.0)
