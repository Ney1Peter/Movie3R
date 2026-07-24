import numpy as np
from scipy.spatial.transform import Rotation

from versions.v13.experiments.fusion_optimization import (
    normalize_weights,
    strict_cache_from_report,
    weighted_solution,
)


def candidate(rotation_deg, translation):
    rotation = Rotation.from_euler("y", rotation_deg, degrees=True).as_matrix()
    root = np.asarray([0.2, -0.1, 2.0], dtype=np.float64)
    return {
        "rotation": rotation,
        "translation": np.asarray(translation, dtype=np.float64),
        "anchor": rotation @ root + np.asarray(translation, dtype=np.float64),
        "post_root": root,
    }


def test_uniform_raw_translation_matches_naive_definition():
    candidates = {
        "person0": candidate(-10.0, [0.2, 0.0, 0.3]),
        "person1": candidate(10.0, [0.4, 0.2, 0.1]),
    }
    solution = weighted_solution(
        candidates, ("person0", "person1"), np.asarray([0.5, 0.5]), "raw"
    )
    assert np.allclose(solution["translation"], [0.3, 0.1, 0.2])
    assert np.allclose(solution["rotation"], np.eye(3), atol=1e-7)


def test_soft_weights_remain_positive_and_normalized():
    weights = normalize_weights(np.asarray([0.0, np.nan, 2.0]))
    assert np.all(weights > 0.0)
    assert np.isclose(weights.sum(), 1.0)


def test_saved_assignment_relabels_detections_by_detection_index():
    cache = {
        "humans": [{
            "legacy0": {"detection_index": 0, "value": "left"},
            "legacy1": {"detection_index": 1, "value": "right"},
        }]
    }
    assignment = [{
        "assignments": [
            {"identity": "person1", "detection_index": 0},
            {"identity": "person0", "detection_index": 1},
        ]
    }]
    relabeled = strict_cache_from_report(cache, assignment)
    assert relabeled["humans"][0]["person1"]["value"] == "left"
    assert relabeled["humans"][0]["person0"]["value"] == "right"
