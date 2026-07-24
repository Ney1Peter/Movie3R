import numpy as np

from versions.v13.identity_bridge import (
    CausalIdentityMemory,
    MatchConfig,
    build_identity_bank,
    evaluate_assignment,
    feature_distance,
    match_identity_bank,
)


def frame(track_ids, refined, beta=None):
    refined = np.asarray(refined, dtype=np.float32)
    count = len(refined)
    if beta is None:
        beta = np.zeros((count, 2), dtype=np.float32)
    return {
        "count": count,
        "track_ids": np.asarray(track_ids, dtype=np.int64),
        "features": {
            "refined_human_tokens": refined,
            "cut3r_head_tokens": refined,
            "mhmr_head_tokens": refined,
            "fused_human_prompts": refined,
            "smpl_beta": np.asarray(beta, dtype=np.float32),
            "local_pose": refined,
        },
    }


def test_mean_bank_is_grouped_by_track_id_not_detection_index():
    frames = [
        frame([10, 20], [[1.0, 0.0], [0.0, 1.0]]),
        frame([20, 10], [[0.0, 0.8], [0.8, 0.0]]),
    ]
    bank = build_identity_bank(frames, "refined_human_tokens", "mean", "raw_l2")
    assert bank["track_ids"].tolist() == [10, 20]
    assert np.allclose(bank["prototypes"], [[0.9, 0.0], [0.0, 0.9]])


def test_hungarian_bridge_recovers_swapped_detection_order():
    pre = [frame([10, 20], [[1.0, 0.0], [0.0, 1.0]])]
    post = frame([0, 1], [[0.0, 0.9], [0.9, 0.0]])
    result = match_identity_bank(
        pre,
        post,
        MatchConfig(
            feature="refined_human_tokens",
            prototype="last",
            distance="raw_l2",
            matcher="hungarian",
        ),
    )
    pairs = {
        row["source_index"]: row["target_index"] for row in result["accepted_pairs"]
    }
    assert pairs == {0: 1, 1: 0}
    metrics = evaluate_assignment(result, {10: 0, 20: 1}, np.asarray([1, 0]))
    assert metrics["assignment_accuracy"] == 1.0
    assert metrics["idf1"] == 1.0


def test_dustbin_rejects_cost_above_frozen_threshold():
    pre = [frame([10], [[1.0, 0.0]])]
    post = frame([0], [[0.0, 1.0]])
    result = match_identity_bank(
        pre,
        post,
        MatchConfig(
            feature="refined_human_tokens",
            prototype="last",
            distance="raw_l2",
            matcher="hungarian",
            max_cost=0.5,
        ),
    )
    assert not result["accepted_pairs"]
    assert result["unmatched_source"].tolist() == [0]
    assert result["unmatched_target"].tolist() == [0]


def test_combined_feature_equalizes_token_and_beta_cues():
    first = np.asarray([[1000.0, 0.0], [0.0, 1000.0]])
    second = np.asarray([[999.0, 0.0], [0.0, 999.0]])
    raw = feature_distance(first, second, "normalized_l2")
    assert raw[0, 0] < raw[0, 1]


def test_cosine_distance_is_finite_for_zero_memory_control():
    distance = feature_distance(
        np.zeros((2, 3), dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "cosine",
    )
    assert np.isfinite(distance).all()
    assert np.allclose(distance, 1.0)


def test_tentative_match_does_not_commit_or_pollute_memory():
    memory = CausalIdentityMemory(ttl=2, prototype_window=2)
    memory.bootstrap(frame([0, 1], [[1.0, 0.0], [0.0, 1.0]]), timestamp=0)
    before = memory.snapshot()
    post = frame([0, 1], [[0.0, 0.9], [0.9, 0.0]])
    result = memory.tentative_match(
        post,
        MatchConfig("refined_human_tokens", "mean", "raw_l2", "hungarian"),
        timestamp=1,
    )
    assert memory.snapshot() == before
    committed = memory.commit(post, result, timestamp=1)
    assert committed.tolist() == [1, 0]
    assert memory.snapshot()["commit_count"] == 2


def test_inactive_tracklet_survives_until_ttl_and_new_person_gets_new_id():
    memory = CausalIdentityMemory(ttl=2, prototype_window=2)
    memory.bootstrap(frame([0], [[1.0, 0.0]]), timestamp=0)
    empty = frame([], np.empty((0, 2), dtype=np.float32))
    result = memory.tentative_match(
        empty,
        MatchConfig("refined_human_tokens", "last", "raw_l2", "hungarian"),
        timestamp=1,
    )
    memory.commit(empty, result, timestamp=1)
    assert 0 in memory.snapshot()["tracks"]
    assert not memory.snapshot()["tracks"][0]["active"]
    memory.expire(timestamp=3)
    assert 0 not in memory.snapshot()["tracks"]
    newcomer = frame([0], [[0.0, 1.0]])
    result = memory.tentative_match(
        newcomer,
        MatchConfig("refined_human_tokens", "last", "raw_l2", "hungarian"),
        timestamp=3,
    )
    assigned = memory.commit(newcomer, result, timestamp=3)
    assert assigned.tolist() == [1]


def test_normal_frame_observation_updates_known_ids_without_matching():
    memory = CausalIdentityMemory(ttl=2, prototype_window=2)
    memory.bootstrap(frame([4], [[1.0, 0.0]]), timestamp=0)
    memory.observe(frame([9], [[0.9, 0.0]]), np.asarray([4]), timestamp=1)
    snapshot = memory.snapshot()
    assert snapshot["tracks"][4]["history_count"] == 2
    assert snapshot["tracks"][4]["last_seen"] == 1


def test_source_camera_and_path_metadata_do_not_change_matching():
    pre = [frame([10, 20], [[1.0, 0.0], [0.0, 1.0]])]
    post = frame([0, 1], [[0.0, 0.9], [0.9, 0.0]])
    config = MatchConfig(
        "refined_human_tokens", "last", "raw_l2", "hungarian"
    )
    baseline = match_identity_bank(pre, post, config)
    pre[0].update(
        source_id="perturbed-source",
        camera_id=999,
        camera_pair_id="wrong-pair",
        file_path="/renamed/sequence/frame.jpg",
    )
    post.update(
        source_id="another-source",
        camera_id=-1,
        camera_pair_id="another-pair",
        file_path="/different/name.jpg",
    )
    perturbed = match_identity_bank(pre, post, config)
    assert baseline["accepted_pairs"] == perturbed["accepted_pairs"]
    assert np.array_equal(baseline["cost"], perturbed["cost"])


def test_rejected_match_creates_new_track_without_polluting_old_track():
    memory = CausalIdentityMemory(ttl=3, prototype_window=2)
    memory.bootstrap(frame([7], [[1.0, 0.0]]), timestamp=0)
    before = memory.snapshot()
    post = frame([0], [[0.0, 1.0]])
    result = memory.tentative_match(
        post,
        MatchConfig(
            "refined_human_tokens",
            "last",
            "raw_l2",
            "hungarian",
            max_cost=0.1,
        ),
        timestamp=1,
    )
    assigned = memory.commit(post, result, timestamp=1)
    after = memory.snapshot()
    assert assigned.tolist() == [8]
    assert after["tracks"][7]["history_count"] == before["tracks"][7]["history_count"]
    assert not after["tracks"][7]["active"]
    assert after["tracks"][8]["active"]
