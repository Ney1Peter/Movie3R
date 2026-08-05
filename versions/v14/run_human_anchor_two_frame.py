#!/usr/bin/env python3
"""Two-frame single-person human-anchor camera/body probe.

This is an isolated experiment for the first post frame.  It does not change
the frozen B0 pipeline.  Given a B0 payload and the same checkpoint's raw
payload, it estimates a rotation from corresponding SMPL-X body joints, uses
the B0/raw root rays to solve camera translation, and applies the same
rotation around the B0 root to the post human.  The result is written as a
small demo.py --save-compatible payload containing only ``pre`` and ``post``.

No GT is read by the correction itself; GT evaluation is intentionally kept in
the separate evaluator/command used after this script finishes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="B0+BRTC+C1 saved payload")
    p.add_argument("--human-source", type=Path, required=True, help="Same-checkpoint raw saved payload")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pre-index", type=int, default=4)
    p.add_argument("--post-index", type=int, default=5)
    p.add_argument(
        "--anchor-set",
        choices=("stable_feet", "body22", "torso"),
        default="stable_feet",
        help="SMPL-X joint subset used for the human-anchor rotation",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


ANCHORS = {
    # Pelvis/hips/torso/feet/head/shoulders.  These are the low-dimensional
    # anchors used by the earlier streaming human-geometry probes.
    "stable_feet": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 15, 16, 17],
    "body22": list(range(22)),
    "torso": [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17],
}


def load_smpl(path: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def load_camera(path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64), np.asarray(z["intrinsics"])


def weighted_kabsch_rotation(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return column-vector R mapping source points to target points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    source_mean = (source * weights[:, None]).sum(axis=0)
    target_mean = (target * weights[:, None]).sum(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    left, _, right = np.linalg.svd(
        (source_centered * weights[:, None]).T @ target_centered
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    mapped = source_centered @ rotation.T
    rms = float(np.sqrt(np.sum(weights * np.sum((mapped - target_centered) ** 2, axis=1))))
    return rotation, rms


def copy_frame(source: Path, output: Path, source_index: int, output_index: int) -> None:
    for subdir, suffix in (("camera", ".npz"), ("color", ".png"), ("depth", ".npy"), ("conf", ".npy"), ("smpl", ".npz")):
        src = source / subdir / f"{source_index:06d}{suffix}"
        if not src.is_file():
            raise FileNotFoundError(src)
        (output / subdir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, output / subdir / f"{output_index:06d}{suffix}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    raw = args.human_source.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    pre_i, post_i = int(args.pre_index), int(args.post_index)
    pre_smpl = load_smpl(source, pre_i)
    post_smpl = load_smpl(source, post_i)
    raw_smpl = load_smpl(raw, post_i)
    pre_pose, _ = load_camera(source, pre_i)
    post_pose, post_intrinsics = load_camera(source, post_i)
    raw_pose, _ = load_camera(raw, post_i)

    # The saved mesh is already in the payload's world gauge.  SMPL-X's
    # regressor maps its 10,475 vertices to the canonical joint set.
    import smplx

    model = smplx.create(
        str(REPO_ROOT / "src" / "models"),
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    ).eval()
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    pre_vertices = np.asarray(pre_smpl["verts_world"][0], dtype=np.float64)
    post_vertices = np.asarray(post_smpl["verts_world"][0], dtype=np.float64)
    raw_vertices = np.asarray(raw_smpl["verts_world"][0], dtype=np.float64)
    pre_joints = regressor @ pre_vertices
    post_joints = regressor @ post_vertices
    raw_joints = regressor @ raw_vertices

    anchor_ids = np.asarray(ANCHORS[args.anchor_set], dtype=np.int64)
    weights = np.ones(anchor_ids.shape[0], dtype=np.float64)
    # Give the pelvis and the two hip joints more influence for the shared
    # camera/body root while retaining feet and torso orientation cues.
    for pos, joint_id in enumerate(anchor_ids.tolist()):
        weights[pos] = {0: 3.0, 1: 2.0, 2: 2.0, 9: 2.0, 15: 2.0}.get(joint_id, 1.0)
    human_rotation, anchor_rms = weighted_kabsch_rotation(
        post_joints[anchor_ids], pre_joints[anchor_ids], weights
    )

    root_current = post_joints[0]
    root_raw = raw_joints[0]
    q_current = post_pose[:3, :3].T @ (root_current - post_pose[:3, 3])
    q_raw = raw_pose[:3, :3].T @ (root_raw - raw_pose[:3, 3])
    q_mean = 0.5 * (q_current + q_raw)

    # Joint camera/body update.  The body is rotated around the already BRTC
    # corrected root; camera translation is solved from the same root ray.
    corrected_pose = np.array(post_pose, copy=True)
    corrected_pose[:3, :3] = human_rotation @ post_pose[:3, :3]
    corrected_pose[:3, 3] = root_current - corrected_pose[:3, :3] @ q_mean
    corrected_vertices = (post_vertices - root_current) @ human_rotation.T + root_current

    copy_frame(source, output, pre_i, 0)
    copy_frame(source, output, post_i, 1)
    np.savez(output / "camera" / "000001.npz", pose=corrected_pose.astype(np.float32), intrinsics=post_intrinsics)
    corrected_smpl = dict(post_smpl)
    corrected_smpl["verts_world"] = corrected_vertices[None].astype(np.float32)
    np.savez(output / "smpl" / "000001.npz", **corrected_smpl)

    diagnostics = {
        "method": "two_frame_human_anchor_keypoint_camera_body",
        "source": str(source),
        "human_source": str(raw),
        "output": str(output),
        "pre_index": pre_i,
        "post_index": post_i,
        "anchor_set": args.anchor_set,
        "anchor_joint_ids": anchor_ids.tolist(),
        "anchor_rms_m": anchor_rms,
        "human_rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(human_rotation).as_rotvec()))),
        "q_current_m": q_current.tolist(),
        "q_raw_m": q_raw.tolist(),
        "q_mean_m": q_mean.tolist(),
        "runtime_contract": "GT-free; only B0/raw payloads at the two boundary frames are used",
        "formula": "R_new=R_human R_B0; t_new=root_BRTC-R_new*mean(q_B0,q_raw); V_new=root_BRTC+R_human(V_B0-root_BRTC)",
    }
    (output / "human_anchor_two_frame.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
