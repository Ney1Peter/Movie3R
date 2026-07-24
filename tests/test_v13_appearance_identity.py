import numpy as np

from versions.v13.appearance_identity import (
    PrecisionGateConfig,
    apply_precision_gate,
    crop_predicted_bbox,
    precision_signals,
)
from versions.v13.identity_bridge import CausalIdentityMemory


def identity_frame(track_ids, appearance, valid=None):
    appearance = np.asarray(appearance, dtype=np.float32)
    count = len(appearance)
    if valid is None:
        valid = np.ones(count, dtype=bool)
    return {
        "count": count,
        "track_ids": np.asarray(track_ids, dtype=np.int64),
        "features": {
            "appearance": appearance,
            "smpl_beta": appearance.copy(),
            "local_pose": appearance.copy(),
        },
        "appearance_valid": np.asarray(valid, dtype=bool),
    }


def test_predicted_bbox_crop_scales_without_cropping_the_input_frame():
    image = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3)
    image = (image % 255).astype(np.uint8)
    crop, row = crop_predicted_bbox(
        image,
        np.asarray([25.0, 10.0, 75.0, 40.0]),
        processed_shape=(50, 100),
        padding_ratio=0.0,
    )
    assert row["valid"]
    assert row["bbox_rgb"].tolist() == [50, 20, 150, 80]
    assert np.array_equal(crop, image[20:80, 50:150])


def test_precision_gate_accepts_mutual_margin_and_five_frame_vote():
    pre = [
        identity_frame([10, 20], [[1.0, 0.0], [0.0, 1.0]])
        for _ in range(5)
    ]
    post = identity_frame([0, 1], [[0.0, 1.0], [1.0, 0.0]])
    config = PrecisionGateConfig(
        feature="appearance",
        prototype="mean",
        distance="cosine",
        max_primary_distance=0.01,
        min_primary_margin=0.5,
        min_vote_fraction=1.0,
        max_beta_distance=0.01,
        max_pose_distance=0.01,
        min_valid_observations=5,
        require_mutual=True,
    )
    base, signals = precision_signals(pre, post, config)
    result = apply_precision_gate(base, signals, config)
    assert len(result["accepted_pairs"]) == 2
    assert all(signal["mutual_nearest"] for signal in signals)
    assert all(signal["vote_fraction"] == 1.0 for signal in signals)


def test_ambiguous_or_invalid_appearance_is_sent_to_dustbin():
    pre = [identity_frame([10], [[1.0, 0.0]]) for _ in range(5)]
    post = identity_frame([0], [[0.0, 1.0]], valid=[False])
    config = PrecisionGateConfig(
        feature="appearance",
        prototype="mean",
        distance="cosine",
        max_primary_distance=0.1,
        min_valid_observations=3,
    )
    base, signals = precision_signals(pre, post, config)
    result = apply_precision_gate(base, signals, config)
    assert not result["accepted_pairs"]
    assert result["unmatched_source"].tolist() == [0]
    assert result["unmatched_target"].tolist() == [0]


def test_identity_memory_preserves_appearance_validity():
    memory = CausalIdentityMemory(ttl=8, prototype_window=5)
    memory.bootstrap(identity_frame([7], [[1.0, 0.0]], valid=[False]), timestamp=0)
    observations = memory.prototype_frames(timestamp=1)
    assert len(observations) == 1
    assert observations[0]["appearance_valid"].tolist() == [False]
