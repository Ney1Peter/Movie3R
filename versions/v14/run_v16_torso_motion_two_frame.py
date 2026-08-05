#!/usr/bin/env python3
"""Two-frame V16-style torso-motion human-only probe.

The pre-cut torso orientation history predicts the torso orientation at the
first post-cut time.  The residual from the fresh post torso is applied only
to the post human around its BRTC root; camera/background remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import smplx
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pre-index", type=int, default=4)
    p.add_argument("--post-index", type=int, default=5)
    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--max-angle-deg", type=float, default=180.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load(path: Path, subdir: str, index: int, suffix: str) -> dict[str, np.ndarray]:
    with np.load(path / subdir / f"{index:06d}{suffix}", allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def copy_frame(source: Path, output: Path, source_index: int, output_index: int) -> None:
    for subdir, suffix in (("camera", ".npz"), ("color", ".png"), ("depth", ".npy"), ("conf", ".npy"), ("smpl", ".npz")):
        src = source / subdir / f"{source_index:06d}{suffix}"
        (output / subdir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, output / subdir / f"{output_index:06d}{suffix}")


def normalize(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def torso_frame(joints: np.ndarray) -> np.ndarray:
    pelvis, head, left_hip, right_hip = joints[0], joints[15], joints[1], joints[2]
    up = normalize(head - pelvis)
    right = normalize(right_hip - left_hip)
    forward = normalize(np.cross(right, up))
    right = normalize(np.cross(up, forward))
    return np.stack((right, up, forward), axis=1)


def main() -> None:
    a = args()
    source, output = a.source.resolve(), a.output.resolve()
    if output.exists():
        if not a.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    model = smplx.create(str(REPO_ROOT / "src" / "models"), "smplx", gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10).eval()
    reg = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    frames = []
    vertices = []
    for index in range(int(a.pre_index) + 1):
        values = load(source, "smpl", index, ".npz")
        mesh = np.asarray(values["verts_world"][0], dtype=np.float64)
        vertices.append(mesh)
        frames.append(torso_frame(reg @ mesh))
    post_values = load(source, "smpl", int(a.post_index), ".npz")
    post_vertices = np.asarray(post_values["verts_world"][0], dtype=np.float64)
    post_joints = reg @ post_vertices

    history_start = max(1, int(a.pre_index) - int(a.history_frames) + 1)
    deltas = [Rotation.from_matrix(frames[i] @ frames[i - 1].T).as_rotvec() for i in range(history_start, int(a.pre_index) + 1)]
    if deltas:
        delta_array = np.asarray(deltas)
        center = np.median(delta_array, axis=0)
        distances = np.linalg.norm(delta_array - center[None], axis=1)
        threshold = max(np.radians(10.0), float(np.median(distances) + 2.5 * np.median(np.abs(distances - np.median(distances)))))
        keep = distances <= threshold
        omega = delta_array[keep].mean(axis=0) if np.any(keep) else center
    else:
        delta_array = np.zeros((0, 3), dtype=np.float64)
        keep = np.zeros((0,), dtype=bool)
        omega = np.zeros(3, dtype=np.float64)
    target_frame = Rotation.from_rotvec(omega).as_matrix() @ frames[int(a.pre_index)]
    residual = target_frame @ torso_frame(post_joints).T
    raw_residual_vec = Rotation.from_matrix(residual).as_rotvec()
    raw_angle = float(np.linalg.norm(raw_residual_vec))
    applied_vec = raw_residual_vec.copy()
    max_angle = np.radians(max(float(a.max_angle_deg), 0.0))
    if np.linalg.norm(applied_vec) > max_angle > 0:
        applied_vec *= max_angle / np.linalg.norm(applied_vec)
    applied_rotation = Rotation.from_rotvec(applied_vec).as_matrix()
    root = post_joints[0]
    corrected_vertices = (post_vertices - root) @ applied_rotation.T + root

    copy_frame(source, output, int(a.pre_index), 0)
    copy_frame(source, output, int(a.post_index), 1)
    corrected = dict(post_values)
    corrected["verts_world"] = corrected_vertices[None].astype(np.float32)
    np.savez(output / "smpl" / "000001.npz", **corrected)
    diagnostics = {
        "method": "v16_torso_motion_human_only",
        "source": str(source),
        "output": str(output),
        "pre_index": int(a.pre_index),
        "post_index": int(a.post_index),
        "history_start": history_start,
        "history_frames": int(a.history_frames),
        "history_delta_deg": [float(np.degrees(np.linalg.norm(value))) for value in delta_array],
        "history_inlier_count": int(keep.sum()),
        "predicted_omega_deg": float(np.degrees(np.linalg.norm(omega))),
        "raw_residual_deg": float(np.degrees(raw_angle)),
        "applied_residual_deg": float(np.degrees(np.linalg.norm(applied_vec))),
        "max_angle_deg": float(a.max_angle_deg),
        "camera_unchanged": True,
        "runtime_contract": "GT-free; only causal pre-cut torso history and first post body are used",
    }
    (output / "v16_torso_motion_two_frame.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
