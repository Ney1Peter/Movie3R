#!/usr/bin/env python3
"""Convert raw AvatarReX folders into grouped Movie3R/Human3R training format.

The original Movie3R-dataset AvatarReX converter assumes every sequence has the
same complete frame list as smpl_params.npz and enumerates images as SMPL frame
indices. That is unsafe for partially missing raw frames. This V8.1 helper uses
the numeric image stem as the SMPL index, so a missing frame is skipped without
shifting all later SMPL annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def smpl_to_smplx_format(smpl_data: dict[str, np.ndarray], frame_idx: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["smplx_gender_id"] = np.asarray(0, dtype=np.int32)
    out["smplx_shape"] = np.concatenate([smpl_data["betas"][0], [0.0]]).astype(np.float32)
    out["smplx_root_pose"] = smpl_data["global_orient"][frame_idx : frame_idx + 1].astype(np.float32)
    out["smplx_transl"] = smpl_data["transl"][frame_idx].astype(np.float32)
    out["smplx_body_pose"] = smpl_data["body_pose"][frame_idx].reshape(21, 3).astype(np.float32)
    out["smplx_jaw_pose"] = smpl_data["jaw_pose"][frame_idx : frame_idx + 1].astype(np.float32)
    out["smplx_leye_pose"] = np.zeros((1, 3), dtype=np.float32)
    out["smplx_reye_pose"] = np.zeros((1, 3), dtype=np.float32)
    out["smplx_left_hand_pose"] = smpl_data["left_hand_pose"][frame_idx].reshape(15, 3).astype(np.float32)
    out["smplx_right_hand_pose"] = smpl_data["right_hand_pose"][frame_idx].reshape(15, 3).astype(np.float32)
    return out


def build_camera_pose(R: np.ndarray, T: np.ndarray, person_transl: np.ndarray) -> np.ndarray:
    R = R.astype(np.float32)
    T = T.astype(np.float32)
    person_transl = person_transl.astype(np.float32)
    cam_rel_person = T - person_transl
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = -R @ cam_rel_person
    return c2w


def convert_one(task: tuple) -> tuple[str, str, str | None]:
    (
        seq_id,
        frame_path,
        raw_root,
        out_seq,
        R,
        T,
        K,
        smpl_data,
        smpl_frames,
        overwrite,
    ) = task
    frame_stem = frame_path.stem
    frame_idx = int(frame_stem)
    if frame_idx < 0 or frame_idx >= smpl_frames:
        return seq_id, "skipped", f"frame index out of SMPL range: {frame_stem}"

    out_rgb = out_seq / "rgb" / f"{frame_stem}.png"
    out_cam = out_seq / "cam" / f"{frame_stem}.npz"
    out_smpl = out_seq / "smpl" / f"{frame_stem}.pkl"
    out_mask = out_seq / "mask" / f"{frame_stem}.png"
    if (
        not overwrite
        and out_rgb.exists()
        and out_cam.exists()
        and out_smpl.exists()
        and out_mask.exists()
    ):
        return seq_id, "skipped", None

    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        return seq_id, "error", f"failed to read image: {frame_path}"
    if not cv2.imwrite(str(out_rgb), img):
        return seq_id, "error", f"failed to write image: {out_rgb}"

    person_transl = smpl_data["transl"][frame_idx].astype(np.float32)
    c2w = build_camera_pose(R, T, person_transl)
    np.savez_compressed(out_cam, pose=c2w, intrinsics=K)

    smplx = smpl_to_smplx_format(smpl_data, frame_idx)
    key_shapes = {
        "smplx_root_pose": (1, 3),
        "smplx_body_pose": (21, 3),
        "smplx_jaw_pose": (1, 3),
        "smplx_leye_pose": (1, 3),
        "smplx_reye_pose": (1, 3),
        "smplx_left_hand_pose": (15, 3),
        "smplx_right_hand_pose": (15, 3),
        "smplx_shape": (11,),
        "smplx_transl": (3,),
    }
    human = {}
    for key, shape in key_shapes.items():
        value = smplx[key]
        human[key] = value.reshape(-1) if len(shape) > 1 else np.asarray(value)
    with out_smpl.open("wb") as f:
        pickle.dump([human], f)

    mask_path = raw_root / seq_id / "mask" / "pha" / f"{frame_stem}.jpg"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is not None:
            cv2.imwrite(str(out_mask), mask)

    return seq_id, "done", None


def convert_dataset(raw_root: Path, out_training_root: Path, group: str, workers: int, overwrite: bool) -> None:
    smpl_file = raw_root / "smpl_params.npz"
    cal_file = raw_root / "calibration_full.json"
    if not smpl_file.exists():
        raise FileNotFoundError(smpl_file)
    if not cal_file.exists():
        raise FileNotFoundError(cal_file)

    smpl_npz = np.load(smpl_file)
    smpl_data = {key: smpl_npz[key].copy() for key in smpl_npz.keys()}
    smpl_npz.close()
    smpl_frames = int(smpl_data["global_orient"].shape[0])

    with cal_file.open("r", encoding="utf-8") as f:
        cal_data = json.load(f)

    group_root = out_training_root / group
    group_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    missing_seqs = []
    for seq_id in sorted(cal_data):
        seq_root = raw_root / seq_id
        if not seq_root.exists():
            missing_seqs.append(seq_id)
            continue
        out_seq = group_root / seq_id
        for sub in ("rgb", "cam", "smpl", "mask"):
            (out_seq / sub).mkdir(parents=True, exist_ok=True)

        cal = cal_data[seq_id]
        R = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
        T = np.asarray(cal["T"], dtype=np.float32)
        K = np.asarray(cal["K"], dtype=np.float32).reshape(3, 3)
        frame_paths = sorted(seq_root.glob("*.jpg"), key=lambda p: int(p.stem))
        for frame_path in frame_paths:
            tasks.append((seq_id, frame_path, raw_root, out_seq, R, T, K, smpl_data, smpl_frames, overwrite))

    print(f"group={group}")
    print(f"raw_root={raw_root}")
    print(f"out_root={group_root}")
    print(f"smpl_frames={smpl_frames} seqs={len(cal_data)} tasks={len(tasks)} workers={workers}")
    if missing_seqs:
        print(f"missing seq dirs: {missing_seqs}")

    done = skipped = errors = 0
    error_msgs = []
    with Pool(processes=workers) as pool:
        for seq_id, status, msg in tqdm(pool.imap_unordered(convert_one, tasks), total=len(tasks), desc=f"convert {group}"):
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
                if msg and len(error_msgs) < 20:
                    error_msgs.append(f"SKIP {seq_id}: {msg}")
            else:
                errors += 1
                if msg and len(error_msgs) < 20:
                    error_msgs.append(f"ERROR {seq_id}: {msg}")

    print(f"summary group={group}: done={done} skipped={skipped} errors={errors}")
    for msg in error_msgs:
        print(msg)
    if errors:
        raise RuntimeError(f"{errors} conversion errors in group={group}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--out_training_root", type=Path, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--workers", type=int, default=min(32, max(4, cpu_count() - 2)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_dataset(args.raw_root, args.out_training_root, args.group, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
