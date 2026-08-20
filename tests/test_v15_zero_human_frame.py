from __future__ import annotations

import numpy as np

from versions.v15.harmony4d.run_harmony_case import (
    frame_people_vertices,
    prediction_person_count,
)


def test_prediction_without_smpl_fields_is_a_zero_human_frame() -> None:
    assert prediction_person_count({"camera_pose": object()}) == 0


def test_prediction_person_count_uses_the_detection_axis() -> None:
    assert prediction_person_count({"smpl_transl": np.zeros((1, 3, 3))}) == 3


def test_zero_human_frame_materializes_as_empty_mesh_batch() -> None:
    value = frame_people_vertices({"people": []})

    assert value.shape == (0, 0, 3)
