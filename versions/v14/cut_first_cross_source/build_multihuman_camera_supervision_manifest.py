#!/usr/bin/env python3
"""Freeze the P0 MultiHuman camera-supervision development manifest.

Only ``three`` is used here because it was already opened repeatedly during
development.  It is explicitly *not* a pristine final test.  EgoHumans 5-chain
confirmation data and the previously opened dance/box diagnostics never enter
this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "config/manifests/v14_multihuman_camera_supervision_20260803.json"
SEQUENCE = "three"
FRAME_MIN, FRAME_MAX = 379, 1555
# These unordered camera pairs are held completely out of the development
# training events.  Their timestamp set is also removed from training.
DEV_PAIRS = ((0, 3), (1, 4), (2, 5))
DEV_TIMESTAMPS = (500, 700, 900, 1100, 1300, 1500)
SEED = 20260803


def event(sequence: str, pre_camera: int, post_camera: int, frame: int, prefix: str) -> dict:
    return {
        "event_id": f"{prefix}_{sequence}_t{frame:04d}_c{pre_camera}_c{post_camera}",
        "sequence": sequence,
        "pre_camera": int(pre_camera),
        "post_camera": int(post_camera),
        "frame": int(frame),
    }


def sha256_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build() -> dict:
    generator = np.random.default_rng(SEED)
    all_directed = [(first, second) for first in range(6) for second in range(6) if first != second]
    dev_unordered = {tuple(sorted(pair)) for pair in DEV_PAIRS}
    train_pairs = [pair for pair in all_directed if tuple(sorted(pair)) not in dev_unordered]
    excluded_frames = {timestamp + offset for timestamp in DEV_TIMESTAMPS for offset in (-2, -1, 0, 1, 2)}
    candidate_frames = np.asarray(
        [frame for frame in range(FRAME_MIN + 3, FRAME_MAX - 2) if frame not in excluded_frames],
        dtype=np.int64,
    )
    # 192 unique real-multi-person events: enough to have a substantive
    # intervention but small enough to keep the first ablation reproducible.
    train = []
    pair_order = generator.permutation(len(train_pairs))
    frame_order = generator.permutation(candidate_frames)
    for index in range(192):
        pre, post = train_pairs[int(pair_order[index % len(train_pairs)])]
        frame = int(frame_order[index])
        train.append(event(SEQUENCE, pre, post, frame, "train"))

    dev = []
    for timestamp in DEV_TIMESTAMPS:
        for first, second in DEV_PAIRS:
            dev.append(event(SEQUENCE, first, second, timestamp, "dev"))
            dev.append(event(SEQUENCE, second, first, timestamp, "dev"))

    records = train + dev
    event_keys = [row["event_id"] for row in records]
    if len(event_keys) != len(set(event_keys)):
        raise AssertionError("duplicate event id")
    train_frames = {row["frame"] for row in train}
    if train_frames.intersection(DEV_TIMESTAMPS):
        raise AssertionError("train/dev timestamp overlap")
    if any(tuple(sorted((row["pre_camera"], row["post_camera"]))) in dev_unordered for row in train):
        raise AssertionError("train/dev camera pair overlap")

    return {
        "schema_version": 1,
        "created": "2026-08-03",
        "purpose": "P0 unified B0 development-only real multi-human camera supervision",
        "seed": SEED,
        "source": {
            "dataset": "MultiHuman Real-World-Capture",
            "sequence": SEQUENCE,
            "rgb": "original calibrated six-camera video, full frame",
            "targets": "per-frame official camera extrinsics/intrinsics only",
            "human_supervision": "none; SMPL-X meshes/GT identities never enter training",
        },
        "split_policy": {
            "training": "192 events; pair-disjoint from dev and excludes dev timestamps +/- 2 frames",
            "development": "36 opposite-camera events at six fixed timestamps; no hyperparameters in final confirmation may be chosen from it",
            "excluded": [
                "EgoHumans 5-chain confirmation (never train/select)",
                "MultiHuman dance and box (previously read diagnostics, not used here)",
            ],
            "pristine_status": "development only: the three sequence was previously opened",
        },
        "train": train,
        "dev": dev,
        "content_sha256": sha256_json({"train": train, "dev": dev}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "train": len(payload["train"]), "dev": len(payload["dev"]), "sha256": payload["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
