#!/usr/bin/env python3
"""Frozen common-SMPL surface and 24-joint topology for Harmony4D."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SMPLX_TO_SMPL = REPO_ROOT / "src/models/smplx/smplx2smpl.pkl"
SMPL_NEUTRAL = REPO_ROOT / "src/models/smpl/SMPL_NEUTRAL.pkl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CommonTopology:
    indices: np.ndarray
    weights: np.ndarray
    joint_regressor: np.ndarray
    transfer_sha256: str
    smpl_model_sha256: str

    @classmethod
    def load(cls) -> "CommonTopology":
        with SMPLX_TO_SMPL.open("rb") as handle:
            matrix = pickle.load(handle, encoding="latin1")["matrix"]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix)
        rows, columns = np.nonzero(matrix)
        counts = np.bincount(rows, minlength=matrix.shape[0])
        width = int(counts.max())
        indices = np.full((matrix.shape[0], width), -1, dtype=np.int64)
        weights = np.zeros((matrix.shape[0], width), dtype=np.float32)
        offsets = np.zeros(matrix.shape[0], dtype=np.int64)
        for row, column in zip(rows.tolist(), columns.tolist()):
            slot = int(offsets[row])
            indices[row, slot] = column
            weights[row, slot] = float(matrix[row, column])
            offsets[row] += 1
        del matrix, rows, columns
        with SMPL_NEUTRAL.open("rb") as handle:
            model = pickle.load(handle, encoding="latin1")
        regressor = model["J_regressor"]
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        regressor = np.asarray(regressor, dtype=np.float64)[:24]
        return cls(
            indices=indices,
            weights=weights,
            joint_regressor=regressor,
            transfer_sha256=file_sha256(SMPLX_TO_SMPL),
            smpl_model_sha256=file_sha256(SMPL_NEUTRAL),
        )

    def smplx_vertices_to_smpl(self, vertices: np.ndarray) -> np.ndarray:
        value = np.asarray(vertices)
        if value.shape[-2:] != (10475, 3):
            raise ValueError(f"Expected SMPL-X vertices (...,10475,3), got {value.shape}")
        safe = np.maximum(self.indices, 0)
        gathered = value[..., safe, :]
        return (gathered * self.weights[..., None]).sum(axis=-2)

    def joints_from_smpl(self, vertices: np.ndarray) -> np.ndarray:
        value = np.asarray(vertices)
        if value.shape[-2:] != (6890, 3):
            raise ValueError(f"Expected SMPL vertices (...,6890,3), got {value.shape}")
        return np.einsum("jv,...vc->...jc", self.joint_regressor, value)

    def metadata(self) -> dict:
        return {
            "name": "Harmony4D-Movie3R-common-SMPL24-v1",
            "surface_vertices": 6890,
            "joints": 24,
            "pelvis_definition": "mean(SMPL joints 1,2)",
            "smplx_to_smpl_sha256": self.transfer_sha256,
            "smpl_neutral_sha256": self.smpl_model_sha256,
        }

