from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from versions.v14.probe_brtc_person_orientation_observable_selector import (
    predict_tree,
    rotate_about_native_root,
    selection,
    tree_dict,
)


def test_serialized_tree_predictor_matches_sklearn() -> None:
    values = np.asarray(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    targets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(values, targets)
    assert np.array_equal(predict_tree(tree_dict(model), values), model.predict(values))


def test_rotation_preserves_native_root_exactly() -> None:
    person = {
        "root": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        "joints": np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], dtype=np.float64),
        "vertices": np.asarray([[1.0, 3.0, 3.0], [1.0, 2.0, 4.0]], dtype=np.float64),
    }
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output = rotate_about_native_root(person, rotation)
    assert np.array_equal(output["root"], person["root"])
    assert not np.array_equal(output["vertices"], person["vertices"])


def test_selector_defaults_to_torso4_for_low_confidence_and_ood() -> None:
    predictions = np.asarray([-0.003, -0.001, -0.003], dtype=np.float64)
    values = np.asarray([[0.0], [0.0], [6.0]], dtype=np.float64)
    selected, zmax = selection(
        predictions,
        values,
        threshold_m=0.002,
        mean=np.asarray([0.0]),
        std=np.asarray([1.0]),
        max_abs_z=5.0,
    )
    assert selected.tolist() == [True, False, False]
    assert zmax.tolist() == [0.0, 0.0, 6.0]
