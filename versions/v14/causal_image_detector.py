#!/usr/bin/env python3
"""Runtime loader for the audited causal GRU shot detector.

This module contains no geometry or supervision.  It consumes adjacent RGB
frames, keeps a bounded history of pair statistics, and emits ``p_cut`` for
the current frame.  The artifact is produced by ``train_causal_detector.py``;
the legacy logistic detector remains the default fallback in the main demo.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.v10_detector_feature_probe import pair_features


class _TemporalHead(nn.Module):
    def __init__(self, dim: int, kind: str) -> None:
        super().__init__()
        if kind != "gru":
            raise ValueError(f"Unsupported runtime detector kind: {kind}")
        self.rnn = nn.GRU(dim, 32, batch_first=True)
        self.net = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.rnn(x)[0][:, -1]).squeeze(-1)


class CausalGRUShotDetector:
    """Causal image-only detector with no future-frame access."""

    def __init__(self, model_path: Path, image_size: int = 192, orb_size: int = 384, threshold: float = .5):
        payload = torch.load(Path(model_path), map_location="cpu", weights_only=False)
        if payload.get("kind") != "gru":
            raise ValueError("The selected runtime artifact is not a GRU detector")
        self.feature_names = tuple(payload["feature_names"])
        self.history = int(payload["history"])
        self.image_size = int(image_size); self.orb_size = int(orb_size); self.threshold = float(threshold)
        self.mean = np.asarray(payload["mean"], dtype=np.float32).reshape(self.history, len(self.feature_names))
        self.std = np.asarray(payload["std"], dtype=np.float32).reshape(self.history, len(self.feature_names))
        self.std[self.std < 1e-6] = 1.
        self.model = _TemporalHead(len(self.feature_names), "gru")
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self._history: deque[np.ndarray] = deque(maxlen=self.history)

    def reset(self) -> None:
        self._history.clear()

    def predict_pair(self, prev_image: Path, cur_image: Path) -> dict:
        raw = pair_features(Path(prev_image), Path(cur_image), self.image_size, self.orb_size)
        vector = np.asarray([float(raw[name]) for name in self.feature_names], dtype=np.float32)
        self._history.append(vector)
        padded = np.zeros((self.history, len(self.feature_names)), dtype=np.float32)
        values = np.asarray(self._history, dtype=np.float32)
        padded[-len(values):] = values
        with torch.no_grad():
            prob = float(torch.sigmoid(self.model(torch.from_numpy(((padded - self.mean) / self.std)[None])))[0])
        return {"pred": int(prob >= self.threshold), "prob": prob, "threshold": self.threshold, "features": raw}

    def predict_sequence(self, image_paths: list[Path]) -> tuple[list[int], list[dict]]:
        self.reset(); labels = [0 for _ in image_paths]; rows = []
        for idx in range(1, len(image_paths)):
            result = self.predict_pair(image_paths[idx - 1], image_paths[idx])
            labels[idx] = int(result["pred"])
            rows.append({"pair_idx": idx, "pred": labels[idx], "prob": result["prob"], "threshold": self.threshold})
        return labels, rows
