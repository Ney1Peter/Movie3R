import numpy as np

from versions.v13.causal_who_where import (
    hypothesis_state_cache,
    transform_shot,
    update_geometry_history,
)


def detection(index: int, root: list[float]) -> dict:
    root = np.asarray(root, dtype=np.float64)
    return {
        "detection_index": index,
        "score": 1.0,
        "completeness": 1.0,
        "bbox": np.zeros(4),
        "bbox_area": 1.0,
        "root": root,
        "torso": np.eye(3),
        "root_rotation": np.eye(3),
        "joints": root[None],
        "vertices": root[None],
    }


def shot(track_id: int = 7) -> dict:
    return {
        "camera": 0,
        "frames": [
            {
                "dataset_frame": 10,
                "camera": 0,
                "pose": np.eye(4),
                "cloud": np.asarray([[0.0, 0.0, 1.0]]),
                "detections": [detection(0, [1.0, 0.0, 0.0])],
                "external_ids": np.asarray([track_id]),
            }
        ],
    }


def test_one_boundary_transforms_camera_cloud_and_every_human():
    boundary = np.eye(4)
    boundary[:3, 3] = [2.0, 3.0, 4.0]
    aligned = transform_shot(shot(), boundary)
    frame = aligned["frames"][0]

    assert np.allclose(frame["pose"][:3, 3], [2.0, 3.0, 4.0])
    assert np.allclose(frame["cloud"][0], [2.0, 3.0, 5.0])
    assert np.allclose(frame["detections"][0]["root"], [3.0, 3.0, 4.0])


def test_persistent_geometry_history_supplies_inactive_track_hypothesis():
    history = {}
    aligned = shot(track_id=7)
    update_geometry_history(history, aligned, history_size=5)
    post = {
        "dataset_frame": 20,
        "camera": 1,
        "pose": np.eye(4),
        "cloud": np.asarray([[0.0, 0.0, 1.0]]),
        "detections": [detection(0, [1.1, 0.0, 0.0])],
    }
    result = {
        "bank": {"track_ids": np.asarray([7])},
        "accepted_pairs": [{"source_index": 0, "target_index": 0}],
    }
    cache = hypothesis_state_cache(history, shot(track_id=99), post, result, ("person0",))

    assert cache["case"]["pre_frames"] == [10]
    assert "person0" in cache["humans"][0]
    assert "person0" in cache["humans"][-1]
    assert cache["accepted"][0]["track_id"] == 7
