#!/usr/bin/env python3
"""Convert one MVHuman sequence to the Movie3R AvatarReX-style training layout.

The MVHuman release stores low-resolution images but full-resolution camera
intrinsics/2D annotations. This converter writes one Movie3R sequence per
camera under:

  data/Training/mvhuman/<seq>/<camera>/{rgb,mask,cam,smpl}

Images are symlinked by default to avoid duplicating the original dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SMPLX_VERTEX_COUNT = 10475


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/MVHuman/100001"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Training/mvhuman"),
    )
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="symlink keeps storage small; copy materializes PNG/JPG files.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="write the old flat layout <seq>_<camera> instead of <seq>/<camera>.",
    )
    parser.add_argument(
        "--check-dir",
        type=Path,
        default=Path("output/mvhuman_100001_conversion_check"),
    )
    return parser.parse_args()


def ensure_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def camera_key_to_name(key: str) -> str:
    # Examples:
    #   1_CC32871A004.png -> CC32871A004
    #   0_0_22236222.png  -> 22236222
    return key.rsplit(".", 1)[0].split("_")[-1]


def load_cameras(root: Path, image_size: tuple[int, int]) -> dict[str, dict[str, np.ndarray]]:
    intrinsics_json = json.loads((root / "camera_intrinsics.json").read_text())
    extrinsics_json = json.loads((root / "camera_extrinsics.json").read_text())
    full_k = np.asarray(intrinsics_json["intrinsics"], dtype=np.float32)

    first_ext = next(iter(extrinsics_json.values()))
    image_center = first_ext["image_center"]
    if len(image_center) >= 4:
        full_w = float(image_center[2])
        full_h = float(image_center[3])
    else:
        # Some MVHuman 200xxx captures only store principal point [cx, cy].
        # Their full image size follows the standard center convention
        # cx=(W-1)/2, cy=(H-1)/2.
        full_w = float(image_center[0]) * 2.0 + 1.0
        full_h = float(image_center[1]) * 2.0 + 1.0
    lr_w, lr_h = image_size
    sx = float(lr_w) / full_w
    sy = float(lr_h) / full_h

    k_lr = full_k.copy()
    k_lr[0, :] *= sx
    k_lr[1, :] *= sy

    cameras: dict[str, dict[str, np.ndarray]] = {}
    for key, value in extrinsics_json.items():
        cam = camera_key_to_name(key)
        r_w2c = np.asarray(value["rotation"], dtype=np.float32)
        t_w2c = np.asarray(value["translation"], dtype=np.float32).reshape(3) / 1000.0
        pose_c2w = np.eye(4, dtype=np.float32)
        pose_c2w[:3, :3] = r_w2c.T
        pose_c2w[:3, 3] = -(r_w2c.T @ t_w2c)
        cameras[cam] = {
            "pose_c2w": pose_c2w,
            "intrinsics": k_lr.astype(np.float32),
            "r_w2c": r_w2c.astype(np.float32),
            "t_w2c": t_w2c.astype(np.float32),
            "full_intrinsics": full_k.astype(np.float32),
            "full_to_lr": np.asarray([sx, sy], dtype=np.float32),
        }
    return cameras


def convert_smplx_json(path: Path, camera_scale: float) -> list[dict[str, np.ndarray]]:
    people = json.loads(path.read_text())
    converted = []
    for person in people:
        poses = np.asarray(person["poses"], dtype=np.float32).reshape(-1)
        if poses.size < 66:
            raise ValueError(f"Unexpected MVHuman pose length {poses.size} in {path}")

        # EasyMocap-style SMPL-X stores global orientation in Rh and keeps a
        # zero root slot at poses[0:3]. Human3R expects axis-angle body pose
        # without that root slot.
        shape = np.asarray(person["shapes"], dtype=np.float32).reshape(-1)
        if shape.size == 10:
            shape = np.concatenate([shape, np.zeros(1, dtype=np.float32)], axis=0)

        def pose_slice(start: int, end: int, fallback_shape: tuple[int, ...]) -> np.ndarray:
            if poses.size >= end:
                return poses[start:end].reshape(fallback_shape).astype(np.float32)
            return np.zeros(fallback_shape, dtype=np.float32)

        converted.append(
            {
                "smplx_root_pose": np.asarray(person["Rh"], dtype=np.float32).reshape(1, 3),
                "smplx_body_pose": poses[3:66].reshape(21, 3).astype(np.float32),
                "smplx_jaw_pose": pose_slice(66, 69, (1, 3)),
                "smplx_leye_pose": pose_slice(69, 72, (1, 3)),
                "smplx_reye_pose": pose_slice(72, 75, (1, 3)),
                # MVHuman stores 6-D PCA hand coefficients. The current
                # Human3R path uses use_pca=False, so keep hands neutral rather
                # than silently treating PCA values as axis-angle joints.
                "smplx_left_hand_pose": np.zeros((15, 3), dtype=np.float32),
                "smplx_right_hand_pose": np.zeros((15, 3), dtype=np.float32),
                "smplx_shape": shape.astype(np.float32),
                "smplx_transl": np.asarray(person["Th"], dtype=np.float32).reshape(3),
                "smplx_world_scale": np.asarray(1.0 / float(camera_scale), dtype=np.float32),
                "smplx_gender_id": np.asarray(0, dtype=np.float32),
            }
        )
    return converted


def load_mvhuman_body25(path: Path, camera_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    people = json.loads(path.read_text())
    if not people:
        body25 = np.zeros((25, 3), dtype=np.float32)
        valid = np.zeros((25,), dtype=np.float32)
        return body25, valid, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    keypoints = np.asarray(people[0]["keypoints3d"], dtype=np.float32)
    body25 = keypoints[:25, :3] / float(camera_scale)
    valid = (keypoints[:25, 3] > 0.2).astype(np.float32)

    head_ids = [0, 15, 16, 17, 18]
    head_valid = [idx for idx in head_ids if valid[idx] > 0.5]
    if head_valid:
        head = body25[head_valid].mean(axis=0)
    elif valid[1] > 0.5:
        head = body25[1]
    else:
        head = np.zeros(3, dtype=np.float32)

    if valid[8] > 0.5:
        pelvis = body25[8]
    else:
        hip_ids = [idx for idx in [9, 12] if valid[idx] > 0.5]
        pelvis = body25[hip_ids].mean(axis=0) if hip_ids else np.zeros(3, dtype=np.float32)

    return (
        body25.astype(np.float32),
        valid.astype(np.float32),
        head.astype(np.float32),
        pelvis.astype(np.float32),
    )


def load_mvhuman_smplx_mesh(path: Path, camera_scale: float) -> tuple[np.ndarray, bool]:
    verts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            vals = line.strip().split()
            if len(vals) < 4:
                continue
            verts.append([float(vals[1]), float(vals[2]), float(vals[3])])

    verts_np = np.asarray(verts, dtype=np.float32) / float(camera_scale)
    exact = verts_np.shape == (SMPLX_VERTEX_COUNT, 3)
    if verts_np.shape[0] < SMPLX_VERTEX_COUNT:
        pad_count = SMPLX_VERTEX_COUNT - verts_np.shape[0]
        pad_value = verts_np[-1:] if verts_np.size else np.zeros((1, 3), dtype=np.float32)
        verts_np = np.concatenate([verts_np, np.repeat(pad_value, pad_count, axis=0)], axis=0)
    elif verts_np.shape[0] > SMPLX_VERTEX_COUNT:
        verts_np = verts_np[:SMPLX_VERTEX_COUNT]
    if verts_np.shape != (SMPLX_VERTEX_COUNT, 3):
        raise ValueError(f"Unexpected MVHuman mesh shape {verts_np.shape} in {path}")
    return verts_np.astype(np.float32), exact


def smpl_index_from_image(path: Path) -> int:
    image_id = int(path.name.split("_", 1)[0])
    if image_id % 5 != 0:
        raise ValueError(f"Unexpected MVHuman image id, expected multiple of 5: {path.name}")
    return image_id // 5 - 1


def write_projection_check(
    root: Path,
    output_dir: Path,
    cam: str,
    image_path: Path,
    smpl_idx: int,
    camera: dict[str, np.ndarray],
    camera_scale: float,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_id = image_path.name.split("_", 1)[0]
    ann_path = root / "annots" / cam / f"{frame_id}_img.json"
    kpt_path = root / "smplx" / "keypoints3d" / f"{smpl_idx:06d}.json"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    ann = np.asarray(json.loads(ann_path.read_text())["annots"][0]["keypoints"], dtype=np.float32)[:25]
    points_3d = (
        np.asarray(json.loads(kpt_path.read_text())[0]["keypoints3d"], dtype=np.float32)[:25, :3]
        / float(camera_scale)
    )
    r_w2c = camera["r_w2c"]
    t_w2c = camera["t_w2c"]
    k_lr = camera["intrinsics"]
    x_cam = (r_w2c @ points_3d.T).T + t_w2c.reshape(1, 3)
    uv = (k_lr @ x_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    sx, sy = camera["full_to_lr"]
    ann_lr = ann.copy()
    ann_lr[:, 0] *= sx
    ann_lr[:, 1] *= sy
    valid = (ann_lr[:, 2] > 0.2) & np.isfinite(uv).all(axis=1) & (x_cam[:, 2] > 1e-6)
    err = np.linalg.norm(uv[valid] - ann_lr[valid, :2], axis=1)

    for point in ann_lr:
        if point[2] > 0.2 and point[0] > 0 and point[1] > 0:
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 0, 0), width=2)
    for x, y in uv:
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 255, 0), width=2)

    out_path = output_dir / f"{cam}_{frame_id}_projected_keypoints.jpg"
    image.save(out_path, quality=95)
    return {
        "projection_check": str(out_path),
        "valid_keypoints": int(valid.sum()),
        "mean_px_error_lr": float(err.mean()) if err.size else float("nan"),
        "median_px_error_lr": float(np.median(err)) if err.size else float("nan"),
    }


def main() -> None:
    args = parse_args()
    root = args.input
    sequence_id = args.sequence_id or root.name
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    first_image = next(iter(sorted((root / "images_lr").glob("*/*_img.jpg"))))
    image_size = Image.open(first_image).size
    cameras = load_cameras(root, image_size)
    camera_scale = float(pickle.load(open(root / "camera_scale.pkl", "rb")))

    summary = {
        "input": str(root),
        "output_root": str(output_root),
        "sequence_id": sequence_id,
        "image_size": list(image_size),
        "camera_count": len(cameras),
        "camera_scale": camera_scale,
        "link_mode": args.link_mode,
        "smplx_mesh_vertex_count": SMPLX_VERTEX_COUNT,
        "smplx_mesh_padded_or_truncated_frames": 0,
        "missing_annotation_images": 0,
        "sequences": [],
    }

    # SMPL-X annotations are shared by all cameras. Cache them per raw SMPL
    # frame so 48-view subjects do not re-parse the same OBJ mesh 48 times.
    smpl_frame_cache: dict[int, tuple[list[dict[str, np.ndarray]], np.ndarray, bool]] = {}

    for cam in sorted(cameras):
        image_dir = root / "images_lr" / cam
        mask_dir = root / "fmask_lr" / cam
        images = sorted(image_dir.glob("*_img.jpg"))
        filtered_images = []
        for image_path in images:
            smpl_idx = smpl_index_from_image(image_path)
            raw_frame = image_path.name.split("_", 1)[0]
            required = [
                mask_dir / f"{raw_frame}_img_fmask.png",
                root / "smplx" / "smpl" / f"{smpl_idx:06d}.json",
                root / "smplx" / "keypoints3d" / f"{smpl_idx:06d}.json",
                root / "smplx" / "smplx_mesh" / f"{smpl_idx:06d}.obj",
            ]
            if all(path.exists() for path in required):
                filtered_images.append(image_path)
            else:
                summary["missing_annotation_images"] += 1
        images = filtered_images
        if args.max_frames is not None:
            images = images[: args.max_frames]

        if args.flat_output:
            seq_name = f"{sequence_id}_{cam}"
            seq_dir = output_root / seq_name
        else:
            seq_name = f"{sequence_id}/{cam}"
            seq_dir = output_root / sequence_id / cam
        for sub in ("rgb", "mask", "cam", "smpl"):
            (seq_dir / sub).mkdir(parents=True, exist_ok=True)

        for out_idx, image_path in enumerate(images):
            smpl_idx = smpl_index_from_image(image_path)
            frame_name = f"{out_idx:08d}"
            raw_frame = image_path.name.split("_", 1)[0]
            mask_path = mask_dir / f"{raw_frame}_img_fmask.png"
            if smpl_idx not in smpl_frame_cache:
                smpl_path = root / "smplx" / "smpl" / f"{smpl_idx:06d}.json"
                keypoints3d_path = root / "smplx" / "keypoints3d" / f"{smpl_idx:06d}.json"
                mesh_path = root / "smplx" / "smplx_mesh" / f"{smpl_idx:06d}.obj"
                body25_world, body25_mask, head_world, pelvis_world = load_mvhuman_body25(
                    keypoints3d_path, camera_scale
                )
                mesh_world, mesh_exact = load_mvhuman_smplx_mesh(mesh_path, camera_scale)
                if not mesh_exact:
                    summary["smplx_mesh_padded_or_truncated_frames"] += 1

                people = convert_smplx_json(smpl_path, camera_scale)
                if people:
                    people[0]["smplx_body25_world"] = body25_world
                    people[0]["smplx_body25_mask"] = body25_mask
                    people[0]["smplx_head_world"] = head_world
                    people[0]["smplx_pelvis_world"] = pelvis_world
                    people[0]["smplx_has_precomputed_keypoints"] = np.asarray(1.0, dtype=np.float32)
                    people[0]["smplx_mesh_world"] = mesh_world
                    people[0]["smplx_has_precomputed_mesh"] = np.asarray(1.0, dtype=np.float32)
                smpl_frame_cache[smpl_idx] = (people, mesh_world, mesh_exact)
            people = smpl_frame_cache[smpl_idx][0]

            ensure_link_or_copy(image_path, seq_dir / "rgb" / f"{frame_name}.png", args.link_mode)
            ensure_link_or_copy(mask_path, seq_dir / "mask" / f"{frame_name}.png", args.link_mode)
            np.savez(
                seq_dir / "cam" / f"{frame_name}.npz",
                pose=cameras[cam]["pose_c2w"],
                intrinsics=cameras[cam]["intrinsics"],
            )
            with open(seq_dir / "smpl" / f"{frame_name}.pkl", "wb") as f:
                pickle.dump(people, f)

        summary["sequences"].append({"name": seq_name, "frames": len(images)})

    check_cam = sorted(cameras)[0]
    check_image = sorted((root / "images_lr" / check_cam).glob("*_img.jpg"))[0]
    summary.update(
        write_projection_check(
            root=root,
            output_dir=args.check_dir,
            cam=check_cam,
            image_path=check_image,
            smpl_idx=smpl_index_from_image(check_image),
            camera=cameras[check_cam],
            camera_scale=camera_scale,
        )
    )

    summary_path = args.check_dir / "conversion_summary.json"
    args.check_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
