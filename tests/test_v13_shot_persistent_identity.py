import copy

import numpy as np

from versions.v13.identity_bridge import evaluate_assignment
from versions.v13.shot_persistent_identity import (
    ShotPersistentIdentityState,
    enumerate_topk_hypotheses,
)


def frame(values: list[list[float]]) -> dict:
    values = np.asarray(values, dtype=np.float32)
    count = len(values)
    return {
        "count": count,
        "head_scores": np.ones(count, dtype=np.float32),
        "appearance_valid": np.ones(count, dtype=bool),
        "features": {
            "appearance": values,
            "smpl_beta": values,
            "local_pose": values,
            "appearance_beta_pose": values,
        },
    }


def pair_track_targets(result: dict) -> set[tuple[int, int]]:
    track_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    return {
        (int(track_ids[int(row["source_index"])]), int(row["target_index"]))
        for row in result["accepted_pairs"]
    }


def test_running_state_query_is_read_only_and_commit_is_explicit():
    state = ShotPersistentIdentityState(ttl=8)
    ids = state.bootstrap(frame([[1.0, 0.0], [0.0, 1.0]]), timestamp=0)
    before = copy.deepcopy(state.snapshot())

    result = state.hungarian(
        frame([[0.0, 1.0], [1.0, 0.0]]),
        feature="appearance",
        prototype="mean",
        timestamp=1,
    )

    assert state.snapshot() == before
    assert pair_track_targets(result) == {(int(ids[0]), 1), (int(ids[1]), 0)}
    committed = state.commit(
        frame([[0.0, 1.0], [1.0, 0.0]]), result, timestamp=1
    )
    assert committed.tolist() == [int(ids[1]), int(ids[0])]
    assert state.snapshot()["commit_count"] == 2


def test_topk_contains_correct_permutation():
    state = ShotPersistentIdentityState(ttl=8)
    state.bootstrap(
        frame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        timestamp=0,
    )
    post = frame([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    payload, cost = state.cost_matrix(post, "appearance", "mean", timestamp=1)
    hypotheses = enumerate_topk_hypotheses(payload, cost, top_k=6)

    metrics = [
        evaluate_assignment(
            row["result"], {0: 0, 1: 1, 2: 2}, np.asarray([2, 0, 1])
        )
        for row in hypotheses
    ]
    assert metrics[0]["idf1"] == 1.0
    assert any(row["idf1"] == 1.0 for row in metrics)


def test_detection_order_permutation_maps_back_to_same_tracks():
    state = ShotPersistentIdentityState(ttl=8)
    state.bootstrap(frame([[1.0, 0.0], [0.0, 1.0]]), timestamp=0)
    original = frame([[0.9, 0.1], [0.1, 0.9]])
    reversed_frame = frame([[0.1, 0.9], [0.9, 0.1]])

    direct = state.hungarian(original, "appearance", "mean", timestamp=1)
    reversed_result = state.hungarian(
        reversed_frame, "appearance", "mean", timestamp=1
    )
    remapped = {
        (track_id, 1 - target)
        for track_id, target in pair_track_targets(reversed_result)
    }
    assert remapped == pair_track_targets(direct)
