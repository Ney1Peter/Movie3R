#!/usr/bin/env python3
"""Two-frame human-only alignment probe.

The camera and background stay bit-exactly equal to the B0+BRTC+C1 payload.
Only the post-shot SMPL-X mesh is aligned to the pre-shot human using a
weighted human-keypoint rigid transform.  This isolates the user's proposed
alternative from the joint camera--human correction.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]

ANCHORS = {
    "stable_feet": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 15, 16, 17],
    "body22": list(range(22)),
    "torso": [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17],
    # V14 frozen person-local orientation candidate: hip/shoulder torso4.
    "torso4": [1, 2, 16, 17],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pre-index", type=int, default=4)
    p.add_argument("--post-index", type=int, default=5)
    p.add_argument("--anchor-set", choices=tuple(ANCHORS), default="torso")
    p.add_argument(
        "--body-transform",
        choices=("root_rotation", "full_rigid"),
        default="root_rotation",
        help="root_rotation preserves the BRTC root; full_rigid also copies the Kabsch translation",
    )
    p.add_argument("--rotation-fraction", type=float, default=1.0,
                   help="fraction of raw SO(3) rotvec (V14 uses 0.5)")
    p.add_argument("--max-angle-deg", type=float, default=180.0,
                   help="maximum applied angle (V14 uses 25 degrees)")
    p.add_argument("--observable-gate", action="store_true",
                   help="apply only if the root-centred anchor residual decreases")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_npz(path: Path, subdir: str, index: int, suffix: str) -> dict[str, np.ndarray] | None:
    with np.load(path / subdir / f"{index:06d}{suffix}", allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def rigid_transform(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights /= max(float(weights.sum()), 1e-12)
    source_mean = (source * weights[:, None]).sum(axis=0)
    target_mean = (target * weights[:, None]).sum(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    left, _, right = np.linalg.svd((source_centered * weights[:, None]).T @ target_centered)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_mean - rotation @ source_mean
    mapped = source @ rotation.T + translation
    rms = float(np.sqrt(np.sum(weights * np.sum((mapped - target) ** 2, axis=1))))
    return rotation, translation, rms


def copy_frame(source: Path, output: Path, source_index: int, output_index: int) -> None:
    for subdir, suffix in (("camera", ".npz"), ("color", ".png"), ("depth", ".npy"), ("conf", ".npy"), ("smpl", ".npz")):
        src = source / subdir / f"{source_index:06d}{suffix}"
        if not src.is_file():
            raise FileNotFoundError(src)
        (output / subdir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, output / subdir / f"{output_index:06d}{suffix}")


def main() -> None:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    import smplx

    model = smplx.create(
        str(REPO_ROOT / "src" / "models"), "smplx", gender="neutral",
        use_pca=False, flat_hand_mean=True, num_betas=10,
    ).eval()
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    pre = load_npz(source, "smpl", args.pre_index, ".npz")
    post = load_npz(source, "smpl", args.post_index, ".npz")
    pre_vertices = np.asarray(pre["verts_world"][0], dtype=np.float64)
    post_vertices = np.asarray(post["verts_world"][0], dtype=np.float64)
    pre_joints = regressor @ pre_vertices
    post_joints = regressor @ post_vertices
    anchor_ids = np.asarray(ANCHORS[args.anchor_set], dtype=np.int64)
    weights = np.ones(anchor_ids.shape[0], dtype=np.float64)
    for pos, joint_id in enumerate(anchor_ids.tolist()):
        weights[pos] = {0: 3.0, 1: 2.0, 2: 2.0, 9: 2.0, 15: 2.0}.get(joint_id, 1.0)
    raw_rotation, translation, rms = rigid_transform(
        post_joints[anchor_ids], pre_joints[anchor_ids], weights
    )
    raw_rotvec = Rotation.from_matrix(raw_rotation).as_rotvec()
    raw_angle_deg = float(np.degrees(np.linalg.norm(raw_rotvec)))
    fraction = float(np.clip(args.rotation_fraction, 0.0, 1.0))
    applied_rotvec = raw_rotvec * fraction
    applied_angle = float(np.linalg.norm(applied_rotvec))
    max_angle_rad = np.radians(max(float(args.max_angle_deg), 0.0))
    if applied_angle > max_angle_rad > 0.0:
        applied_rotvec *= max_angle_rad / applied_angle
    rotation = Rotation.from_rotvec(applied_rotvec).as_matrix()
    before_residual = float(np.linalg.norm(
        (post_joints[anchor_ids] - post_joints[0])
        - (pre_joints[anchor_ids] - pre_joints[0]), axis=1
    ).mean())
    after_residual = float(np.linalg.norm(
        (post_joints[anchor_ids] - post_joints[0]) @ rotation.T
        - (pre_joints[anchor_ids] - pre_joints[0]), axis=1
    ).mean())
    applied = True
    if args.observable_gate and not (after_residual < before_residual):
        rotation = np.eye(3)
        applied = False
    root = post_joints[0]
    if args.body_transform == "root_rotation":
        corrected_vertices = (post_vertices - root) @ rotation.T + root
        applied_translation = np.zeros(3, dtype=np.float64)
    else:
        corrected_vertices = post_vertices @ rotation.T + translation
        applied_translation = translation

    copy_frame(source, output, args.pre_index, 0)
    copy_frame(source, output, args.post_index, 1)
    corrected = dict(post)
    corrected["verts_world"] = corrected_vertices[None].astype(np.float32)
    np.savez(output / "smpl" / "000001.npz", **corrected)
    # camera/colour/depth/conf remain the original B0 post frame by design.

    diagnostics = {
        "method": "human_only_two_frame",
        "source": str(source),
        "output": str(output),
        "pre_index": int(args.pre_index),
        "post_index": int(args.post_index),
        "anchor_set": args.anchor_set,
        "anchor_joint_ids": anchor_ids.tolist(),
        "body_transform": args.body_transform,
        "anchor_rms_m": rms,
        "raw_rotation_deg": raw_angle_deg,
        "rotation_fraction": fraction,
        "max_angle_deg": float(args.max_angle_deg),
        "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()))),
        "anchor_residual_before_m": before_residual,
        "anchor_residual_after_m": after_residual,
        "observable_gate": bool(args.observable_gate),
        "applied": bool(applied),
        "applied_translation_m": applied_translation.tolist(),
        "camera_unchanged": True,
        "runtime_contract": "GT-free; camera, background, depth and confidence are copied bit-exactly from B0",
    }
    (output / "human_only_two_frame.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
