import torch
from types import SimpleNamespace

from dust3r.inference import (
    _compose_v14_1_dual_path_scene,
    _compute_v14_1_shadow_geometry_loss,
    _make_v14_1_event_off_batch,
    _make_v8_image_only_model_batch,
)


def test_image_only_batch_preserves_deployable_shot_label():
    batch = []
    for label in (0.0, 0.0, 1.0):
        batch.append(
            {
                "img": torch.zeros(1, 3, 16, 16),
                "img_mask": torch.ones(1, dtype=torch.bool),
                "ray_mask": torch.ones(1, dtype=torch.bool),
                "ray_map": torch.ones(1, 16, 16, 6),
                "shot_label": torch.tensor([label]),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "raw_camera_pose": torch.eye(4).unsqueeze(0),
                "smpl_transl": torch.zeros(1, 1, 3),
            }
        )

    model_batch = _make_v8_image_only_model_batch(batch)

    assert [float(view["shot_label"].item()) for view in model_batch] == [0.0, 0.0, 1.0]
    assert all(not bool(view["ray_mask"].any()) for view in model_batch)
    assert all(not bool(view["ray_map"].any()) for view in model_batch)
    assert all("camera_pose" not in view for view in model_batch)
    assert all("raw_camera_pose" not in view for view in model_batch)
    assert all("smpl_transl" not in view for view in model_batch)


def test_event_off_batch_does_not_mutate_source_labels():
    source = [
        {"shot_label": torch.tensor([0.0])},
        {"shot_label": torch.tensor([1.0])},
    ]
    event_off = _make_v14_1_event_off_batch(source)

    assert [float(view["shot_label"].item()) for view in event_off] == [0.0, 0.0]
    assert [float(view["shot_label"].item()) for view in source] == [0.0, 1.0]


def test_shadow_geometry_loss_is_event_only_and_uses_shared_transform():
    camera_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    local = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    event_on_local = local.clone().requires_grad_(True)
    event_on_world = (local + 1.0).requires_grad_(True)
    context_wrong = torch.full_like(local, 100.0, requires_grad=True)

    common_human = {
        "smpl_shape": torch.zeros(1, 1, 10),
        "smpl_rotmat": torch.eye(3).reshape(1, 1, 1, 3, 3),
        "smpl_expression": torch.zeros(1, 1, 10),
    }
    preds_on = [
        {
            "camera_pose": camera_pose,
            "pts3d_in_self_view": context_wrong,
            "pts3d_in_other_view": context_wrong,
            **common_human,
        },
        {
            "camera_pose": camera_pose,
            "pts3d_in_self_view": event_on_local,
            "pts3d_in_other_view": event_on_world,
            **common_human,
        },
    ]
    preds_off = [
        {
            "camera_pose": camera_pose,
            "pts3d_in_self_view": local,
            "pts3d_in_other_view": local,
            **common_human,
        },
        {
            "camera_pose": camera_pose,
            "pts3d_in_self_view": local,
            "pts3d_in_other_view": local,
            **common_human,
        },
    ]
    batch = [
        {"shot_label": torch.tensor([0.0])},
        {"shot_label": torch.tensor([1.0])},
    ]
    model = SimpleNamespace(
        v14_1_self_pointmap_keep_loss_weight=1.0,
        v14_1_shared_pointmap_loss_weight=1.0,
        v14_1_human_param_keep_loss_weight=1.0,
    )

    loss, details = _compute_v14_1_shadow_geometry_loss(
        batch, preds_on, preds_off, model
    )

    assert loss > 0
    assert details["v14_1_self_pointmap_keep_loss"] == 0.0
    assert details["v14_1_shared_pointmap_loss"] > 0.0
    loss.backward()
    assert event_on_world.grad is not None
    assert event_on_world.grad.abs().sum() > 0
    assert context_wrong.grad is None


def test_dual_path_scene_keeps_corrected_pose_and_raw_event_geometry():
    corrected_camera = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    corrected_local = torch.tensor([[[[9.0, 9.0, 9.0]]]])
    raw_local = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    corrected_human = torch.tensor([[[4.0, 5.0, 6.0]]])
    corrected_preds = [
        {
            "camera_pose": corrected_camera,
            "pts3d_in_self_view": corrected_local,
            "pts3d_in_other_view": corrected_local,
            "conf_self": torch.tensor([[[2.0]]]),
            "smpl_transl": corrected_human,
        }
    ]
    raw_preds = [
        {
            "camera_pose": torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
            ),
            "pts3d_in_self_view": raw_local,
            "pts3d_in_other_view": raw_local,
            "conf_self": torch.tensor([[[3.0]]]),
            "smpl_transl": torch.zeros_like(corrected_human),
        }
    ]

    composed = _compose_v14_1_dual_path_scene(
        [{"shot_label": torch.tensor([1.0])}], corrected_preds, raw_preds
    )[0]

    assert torch.equal(composed["camera_pose"], corrected_camera)
    assert torch.equal(composed["smpl_transl"], corrected_human)
    assert torch.equal(composed["pts3d_in_self_view"], raw_local)
    assert torch.equal(
        composed["pts3d_in_other_view"],
        torch.tensor([[[[2.0, 2.0, 3.0]]]]),
    )
    assert torch.equal(composed["conf_self"], torch.tensor([[[3.0]]]))


if __name__ == "__main__":
    test_image_only_batch_preserves_deployable_shot_label()
    test_event_off_batch_does_not_mutate_source_labels()
    test_shadow_geometry_loss_is_event_only_and_uses_shared_transform()
    test_dual_path_scene_keeps_corrected_pose_and_raw_event_geometry()
