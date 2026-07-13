#!/usr/bin/env python3
"""Deployable image-only shot-boundary detector for V10 streaming.

The detector is trained from the saved V10 detector feature CSV and predicts
online boundaries from adjacent input frames only.  It intentionally does not
use SMPL, camera, GT angles, or future frames.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v10_detector_feature_probe import BASIC_FEATURES, MATCH_FEATURES, pair_features


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ONLY_FEATURES = BASIC_FEATURES + MATCH_FEATURES
DEFAULT_FEATURE_CSV = (
    REPO_ROOT
    / "output"
    / "v10_detector_probe"
    / "image_feature_round1"
    / "detector_pair_features.csv"
)


def read_feature_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class StreamingImageOnlyShotDetector:
    """Online detector using only adjacent-frame image features."""

    def __init__(
        self,
        feature_csv: Path = DEFAULT_FEATURE_CSV,
        image_size: int = 192,
        orb_size: int = 384,
        threshold: float = 0.5,
    ) -> None:
        self.feature_csv = Path(feature_csv)
        self.image_size = int(image_size)
        self.orb_size = int(orb_size)
        self.threshold = float(threshold)
        self.feature_names = list(IMAGE_ONLY_FEATURES)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        self._fit()

    def _fit(self) -> None:
        rows = read_feature_rows(self.feature_csv)
        if not rows:
            raise ValueError(f"No detector feature rows found: {self.feature_csv}")
        x = np.asarray([[float(row[name]) for name in self.feature_names] for row in rows], dtype=np.float64)
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        self.model.fit(x, y)

    def predict_pair(self, prev_image: Path, cur_image: Path) -> dict:
        feats = pair_features(Path(prev_image), Path(cur_image), self.image_size, self.orb_size)
        x = np.asarray([[float(feats[name]) for name in self.feature_names]], dtype=np.float64)
        prob = float(self.model.predict_proba(x)[0, 1])
        pred = int(prob >= self.threshold)
        return {
            "pred": pred,
            "prob": prob,
            "threshold": self.threshold,
            "features": feats,
        }

    def predict_sequence(self, image_paths: list[Path]) -> tuple[list[int], list[dict]]:
        labels = [0 for _ in image_paths]
        rows = []
        for idx in range(1, len(image_paths)):
            out = self.predict_pair(image_paths[idx - 1], image_paths[idx])
            labels[idx] = int(out["pred"])
            rows.append(
                {
                    "pair_idx": idx,
                    "prev_image": str(image_paths[idx - 1]),
                    "cur_image": str(image_paths[idx]),
                    "pred": int(out["pred"]),
                    "prob": float(out["prob"]),
                    "threshold": float(out["threshold"]),
                }
            )
        return labels, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Ordered image paths for online boundary prediction.")
    parser.add_argument("--feature_csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--orb_size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = StreamingImageOnlyShotDetector(
        feature_csv=args.feature_csv,
        image_size=args.image_size,
        orb_size=args.orb_size,
        threshold=args.threshold,
    )
    labels, rows = detector.predict_sequence([Path(p) for p in args.images])
    print(json.dumps({"shot_labels": labels, "pairs": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
