import torch

from dust3r.shot_human_memory import (
    HumanMemoryConfig,
    StreamingHumanMemory,
    project_to_rotation,
    transform_points,
)


def identity_rotations(batch=1, humans=1, joints=3):
    return torch.eye(3).reshape(1, 1, 1, 3, 3).expand(batch, humans, joints, -1, -1).clone()


def test_projection_returns_valid_rotations():
    matrix = torch.randn(2, 4, 3, 3)
    rotation = project_to_rotation(matrix)
    identity = rotation.transpose(-1, -2) @ rotation
    expected = torch.eye(3).expand_as(identity)
    assert torch.allclose(identity, expected, atol=1e-5)
    assert torch.all(torch.det(rotation) > 0.999)


def test_memory_conditions_only_human_fields():
    config = HumanMemoryConfig(token_alpha=0.25, shape_alpha=1.0, local_pose_alpha=0.0)
    memory = StreamingHumanMemory(config)
    token = torch.tensor([[[1.0, 0.0, 0.0]]])
    prediction = {
        "smpl_shape": torch.ones(1, 1, 2),
        "smpl_rotmat": identity_rotations(),
    }
    memory.commit(token, prediction, world_root=torch.zeros(1, 1, 3))

    current = torch.tensor([[[0.0, 1.0, 0.0]]])
    conditioned, weights, _ = memory.condition_token(current)
    output = {
        "camera_pose": torch.arange(7, dtype=torch.float32).reshape(1, 7),
        "pts3d_in_self_view": torch.randn(1, 2, 2, 3),
        "smpl_shape": torch.zeros(1, 1, 2),
        "smpl_rotmat": identity_rotations(),
    }
    camera_before = output["camera_pose"].clone()
    points_before = output["pts3d_in_self_view"].clone()
    memory.stabilize_prediction(output, weights)

    assert torch.allclose(conditioned, torch.tensor([[[0.25, 0.75, 0.0]]]))
    assert torch.allclose(output["smpl_shape"], torch.ones(1, 1, 2))
    assert torch.equal(output["camera_pose"], camera_before)
    assert torch.equal(output["pts3d_in_self_view"], points_before)


def test_boundary_transform_is_applied_before_world_commit():
    transform = torch.eye(4).unsqueeze(0)
    transform[:, :3, 3] = torch.tensor([2.0, -1.0, 0.5])
    point = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert torch.allclose(transform_points(transform, point), torch.tensor([[[3.0, 1.0, 3.5]]]))


def test_verify_rejects_large_world_root_jump():
    config = HumanMemoryConfig(
        min_detection_score=0.1,
        max_shape_delta=10.0,
        max_world_root_jump=0.5,
    )
    memory = StreamingHumanMemory(config)
    token = torch.tensor([[[1.0, 0.0]]])
    prediction = {
        "smpl_shape": torch.zeros(1, 1, 2),
        "smpl_rotmat": identity_rotations(),
    }
    memory.commit(token, prediction, world_root=torch.zeros(1, 1, 3))
    _, weights, _ = memory.condition_token(token)
    valid, diagnostics = memory.quality(
        prediction,
        detection_score=torch.ones(1, 1),
        world_root=torch.tensor([[[1.0, 0.0, 0.0]]]),
        weights=weights,
    )
    assert not bool(valid.item())
    assert diagnostics["world_root_jump"].item() == 1.0
