import torch

from dust3r.inference import _make_v8_image_only_model_batch


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


if __name__ == "__main__":
    test_image_only_batch_preserves_deployable_shot_label()
