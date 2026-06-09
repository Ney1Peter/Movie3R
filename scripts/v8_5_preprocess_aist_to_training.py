#!/usr/bin/env python3
"""Convert AIST organized-by-motion videos to Movie3R training layout.

Input layout:
  asit/<motion>/<motion>_chXX/{camera,gt,videos}

Output layout:
  Training/asit/<motion>_chXX/cYY/{rgb,mask,smpl,cam}

Only the first N seconds are extracted. AIST videos are 60000/1001 FPS, but the
SMPL fits are stored as 60 Hz arrays. ``--frame_stride 4`` keeps source frames
0, 4, 8, ... and reindexes outputs to contiguous filenames 00000000, 00000001,
... so the Movie3R dataloader can still build consecutive clips.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float32)


def camera_json_to_c2w(camera_json: dict) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(camera_json["matrix"], dtype=np.float32).reshape(3, 3)
    R_w2c = rodrigues(np.asarray(camera_json["rotation"], dtype=np.float32))
    T_w2c = np.asarray(camera_json["translation"], dtype=np.float32).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_w2c
    T[:3, 3] = T_w2c
    c2w = np.linalg.inv(T).astype(np.float32)
    return c2w, K


def load_smpl(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    required = {"smpl_poses", "smpl_scaling", "smpl_trans"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return data


def smpl_frame_to_training_human(smpl_data: dict[str, np.ndarray], frame_idx: int) -> dict[str, np.ndarray]:
    pose = np.asarray(smpl_data["smpl_poses"][frame_idx], dtype=np.float32).reshape(72)
    scale = np.asarray(smpl_data["smpl_scaling"], dtype=np.float32).reshape(-1)[0]
    transl = np.asarray(smpl_data["smpl_trans"][frame_idx], dtype=np.float32).reshape(3)
    return {
        "smpl_root_pose": pose[:3].reshape(1, 3).astype(np.float32),
        "smpl_body_pose": pose[3:].reshape(23, 3).astype(np.float32),
        "smpl_shape": np.zeros((10,), dtype=np.float32),
        "smpl_transl": transl.astype(np.float32),
        "smpl_scale": np.asarray([scale], dtype=np.float32),
        "smpl_gender_id": np.asarray(0, dtype=np.int32),
    }


def discover_sequences(input_root: Path) -> list[Path]:
    seqs = []
    for motion_dir in sorted(input_root.iterdir()):
        if not motion_dir.is_dir():
            continue
        for seq_dir in sorted(motion_dir.iterdir()):
            if (
                seq_dir.is_dir()
                and (seq_dir / "gt" / "smpl.pkl").is_file()
                and (seq_dir / "camera").is_dir()
                and (seq_dir / "videos").is_dir()
            ):
                seqs.append(seq_dir)
    return seqs


def resolve_sequence(input_root: Path, sequence: str) -> Path:
    seq_path = input_root / sequence
    if seq_path.is_dir():
        return seq_path
    matches = [p for p in discover_sequences(input_root) if p.name == sequence or str(p.relative_to(input_root)) == sequence]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve sequence {sequence!r}; matches={matches[:5]}")
    return matches[0]


def camera_ids_for_sequence(seq_dir: Path) -> list[str]:
    cams = sorted(p.name.split("_")[0] for p in (seq_dir / "camera").glob("*_camera.json"))
    vids = sorted(p.stem for p in (seq_dir / "videos").glob("*.mp4"))
    return [cam for cam in cams if cam in vids]


def write_full_mask(path: Path, shape: tuple[int, int, int]) -> None:
    h, w = shape[:2]
    mask = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


def convert_camera(
    seq_dir: Path,
    cam_id: str,
    smpl_data: dict[str, np.ndarray],
    output_root: Path,
    max_frames: int,
    source_fps: int,
    frame_stride: int,
    overwrite: bool,
    clean_output: bool,
    write_mask_placeholder: bool,
    image_ext: str,
) -> dict:
    video_path = seq_dir / "videos" / f"{cam_id}.mp4"
    camera_path = seq_dir / "camera" / f"{cam_id}_camera.json"
    with camera_path.open("r", encoding="utf-8") as f:
        camera_data = json.load(f)
    c2w, K = camera_json_to_c2w(camera_data)

    out_seq = output_root / "asit" / seq_dir.name / cam_id
    if clean_output and out_seq.exists():
        shutil.rmtree(out_seq)
    for subdir in ("rgb", "mask", "smpl", "cam"):
        (out_seq / subdir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    smpl_frames = int(np.asarray(smpl_data["smpl_poses"]).shape[0])
    source_frame_limit = min(max_frames, video_frames, smpl_frames)
    source_frame_indices = list(range(0, source_frame_limit, frame_stride))
    written = 0

    for out_idx, source_frame_idx in enumerate(source_frame_indices):
        frame_name = f"{out_idx:08d}"
        rgb_path = out_seq / "rgb" / f"{frame_name}{image_ext}"
        mask_path = out_seq / "mask" / f"{frame_name}.png"
        smpl_path = out_seq / "smpl" / f"{frame_name}.pkl"
        cam_path = out_seq / "cam" / f"{frame_name}.npz"
        mask_ready = (not write_mask_placeholder) or mask_path.exists()
        if not overwrite and rgb_path.exists() and mask_ready and smpl_path.exists() and cam_path.exists():
            written += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if image_ext.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(rgb_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(str(rgb_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if write_mask_placeholder:
            write_full_mask(mask_path, frame.shape)
        with smpl_path.open("wb") as f:
            pickle.dump([smpl_frame_to_training_human(smpl_data, int(source_frame_idx))], f)
        np.savez_compressed(
            cam_path,
            pose=c2w,
            intrinsics=K,
            source_frame_idx=np.asarray(int(source_frame_idx), dtype=np.int32),
            source_timestamp_sec=np.asarray(float(source_frame_idx) / float(source_fps), dtype=np.float32),
        )
        written += 1

    cap.release()
    meta = {
        "source_sequence": str(seq_dir),
        "camera_id": cam_id,
        "video_frames": video_frames,
        "smpl_frames": smpl_frames,
        "source_frame_limit": source_frame_limit,
        "source_frame_first": int(source_frame_indices[0]) if source_frame_indices else None,
        "source_frame_last": int(source_frame_indices[-1]) if source_frame_indices else None,
        "source_frame_stride": int(frame_stride),
        "written_frames": written,
        "fps_nominal": int(source_fps),
        "output_fps_nominal": float(source_fps) / float(frame_stride),
        "seconds_requested": max_frames / float(source_fps),
        "output_sequence": str(out_seq),
        "image_ext": image_ext,
        "mask": "full_placeholder" if write_mask_placeholder else "empty_dir",
    }
    with (out_seq / "source_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/asit"))
    parser.add_argument("--output_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/Training"))
    parser.add_argument("--sequence", default=None, help="Sequence name or relative path under input_root.")
    parser.add_argument("--all", action="store_true", help="Convert all sequences.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Keep every Nth source frame. Use 4 to turn 60 FPS into 15 FPS outputs.",
    )
    parser.add_argument("--cams", default=None, help="Comma-separated camera ids; default uses all available cameras.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Remove each output camera sequence before writing; useful when changing frame_stride.",
    )
    parser.add_argument(
        "--write_mask_placeholder",
        action="store_true",
        help="Write full-white placeholder masks. By default mask dirs are left empty for later YOLO masks.",
    )
    parser.add_argument("--image_ext", default=".png", choices=(".png", ".jpg", ".jpeg"))
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.all == (args.sequence is not None):
        raise ValueError("Use exactly one of --all or --sequence")

    seqs = discover_sequences(args.input_root) if args.all else [resolve_sequence(args.input_root, args.sequence)]
    max_frames = int(round(args.seconds * args.fps))
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")
    requested_cams = None if args.cams is None else {cam.strip() for cam in args.cams.split(",") if cam.strip()}

    all_meta = []
    for seq_dir in tqdm(seqs, desc="AIST sequences"):
        smpl_data = load_smpl(seq_dir / "gt" / "smpl.pkl")
        cams = camera_ids_for_sequence(seq_dir)
        if requested_cams is not None:
            cams = [cam for cam in cams if cam in requested_cams]
        for cam_id in tqdm(cams, desc=seq_dir.name, leave=False):
            all_meta.append(
                convert_camera(
                    seq_dir=seq_dir,
                    cam_id=cam_id,
                    smpl_data=smpl_data,
                    output_root=args.output_root,
                    max_frames=max_frames,
                    source_fps=int(args.fps),
                    frame_stride=int(args.frame_stride),
                    overwrite=args.overwrite,
                    clean_output=bool(args.clean_output),
                    write_mask_placeholder=bool(args.write_mask_placeholder),
                    image_ext=args.image_ext,
                )
            )

    manifest = args.manifest
    if manifest is None:
        manifest = args.output_root / "asit_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"wrote {manifest} with {len(all_meta)} camera sequences")


if __name__ == "__main__":
    main()
