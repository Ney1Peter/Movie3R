import torch

# The repository's camera module expects model/heads registration to have run.
import dust3r.model  # noqa: F401
from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera
from dust3r.utils.geometry import geotrf
from dust3r.v14_outputs import (
    apply_boundary_to_prediction,
    boundary_from_camera_predictions,
)
from versions.v14.run_v14_2_single_sequence import root_anchored_boundary


def _pose(translation):
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor(translation, dtype=torch.float32)
    return pose


def test_boundary_maps_raw_camera_to_shadow_camera_by_left_multiplication():
    raw_camera = _pose([0.25, -0.5, 1.0])
    boundary = _pose([2.0, 3.0, -1.0])
    shadow_camera = boundary @ raw_camera
    estimated = boundary_from_camera_predictions(
        {"camera_pose": camera_to_pose_encoding(shadow_camera)},
        {"camera_pose": camera_to_pose_encoding(raw_camera)},
    )
    assert torch.allclose(estimated, boundary, atol=1e-6)


def test_camera_and_world_pointmap_receive_the_same_boundary():
    raw_camera = _pose([0.25, -0.5, 1.0])
    boundary = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    boundary[:, :3, :3] = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    boundary[:, :3, 3] = torch.tensor([[2.0, 3.0, -1.0]])
    local_points = torch.tensor([[[[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]]])
    raw_world = geotrf(raw_camera, local_points)

    transformed = apply_boundary_to_prediction(
        {
            "camera_pose": camera_to_pose_encoding(raw_camera),
            "pts3d_in_self_view": local_points,
            "pts3d_in_other_view": raw_world,
        },
        boundary,
    )
    transformed_camera = pose_encoding_to_camera(transformed["camera_pose"])
    expected_world = geotrf(boundary, raw_world)

    assert torch.allclose(transformed_camera, boundary @ raw_camera, atol=1e-6)
    assert torch.allclose(transformed["pts3d_in_other_view"], expected_world, atol=1e-6)
    assert torch.allclose(
        transformed["pts3d_in_other_view"],
        geotrf(transformed_camera, local_points),
        atol=1e-6,
    )


def test_boundary_does_not_modify_camera_local_smpl_parameters():
    prediction = {
        "camera_pose": camera_to_pose_encoding(_pose([0.0, 0.0, 0.0])),
        "smpl_transl": torch.tensor([[[0.1, 0.2, 2.0]]]),
        "smpl_shape": torch.arange(10, dtype=torch.float32).reshape(1, 1, 10),
        "smpl_rotmat": torch.eye(3).reshape(1, 1, 1, 3, 3),
    }
    transformed = apply_boundary_to_prediction(prediction, _pose([1.0, 2.0, 3.0]))
    for key in ("smpl_transl", "smpl_shape", "smpl_rotmat"):
        assert transformed[key] is prediction[key]
        assert torch.equal(transformed[key], prediction[key])


def test_camera_local_human_receives_the_same_world_boundary_via_camera():
    raw_camera = _pose([0.25, -0.5, 1.0])
    boundary = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    boundary[:, :3, :3] = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    boundary[:, :3, 3] = torch.tensor([[2.0, 3.0, -1.0]])
    local_root = torch.tensor([[[0.1, 0.2, 2.0]]])
    prediction = {
        "camera_pose": camera_to_pose_encoding(raw_camera),
        "smpl_transl": local_root,
    }

    transformed = apply_boundary_to_prediction(prediction, boundary)
    raw_world_root = geotrf(raw_camera, local_root)
    transformed_world_root = geotrf(
        pose_encoding_to_camera(transformed["camera_pose"]), local_root
    )

    assert torch.allclose(
        transformed_world_root, geotrf(boundary, raw_world_root), atol=1e-6
    )


def test_root_anchored_boundary_maps_post_root_to_pre_world_root():
    raw_camera = torch.eye(4, dtype=torch.float32).numpy()
    raw_camera[:3, 3] = [1.0, -2.0, 0.5]
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).numpy()
    raw_root_camera = torch.tensor([0.25, 0.5, 2.0]).numpy()
    target_root_world = torch.tensor([4.0, 3.0, 1.5]).numpy()

    boundary = root_anchored_boundary(
        rotation, raw_camera, raw_root_camera, target_root_world
    )
    raw_root_world = raw_camera[:3, :3] @ raw_root_camera + raw_camera[:3, 3]
    aligned_root = boundary[:3, :3] @ raw_root_world + boundary[:3, 3]

    assert torch.allclose(
        torch.from_numpy(aligned_root), torch.from_numpy(target_root_world), atol=1e-6
    )
