#!/usr/bin/env python3
"""Export GT-isolation viewers for one saved MultiHuman Human3R sequence.

Two independent original-demo payloads are produced in the evaluation gauge:

1. exact GT camera extrinsics with the original camera-local Human3R bodies;
2. exact GT camera extrinsics with the dataset GT SMPL-X meshes.

The second payload retains the Human3R background depth only as visual context;
the displayed people and camera poses are ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path(
    "/data/wangzheng/codex_tmp/b0_da3_visuals/"
    "multihuman_three_t0900_c0_c3_original_demo/raw_reset"
)
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_OUTPUT = Path(
    "/data/wangzheng/codex_tmp/b0_da3_visuals/"
    "multihuman_three_t0900_c0_c3_gt_demo"
)
DEFAULT_PRE_FRAMES = (897, 898, 899, 900)
DEFAULT_POST_FRAMES = (900, 901, 902, 903, 904, 905)
IDENTITIES = ("person0", "person1", "person2")
VARIANTS = ("gt_camera_human3r", "gt_camera_gt_humans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_payload", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence", default="three")
    parser.add_argument("--source_camera", type=int, default=0)
    parser.add_argument("--target_camera", type=int, default=3)
    parser.add_argument(
        "--pre_frames", type=int, nargs="+", default=DEFAULT_PRE_FRAMES
    )
    parser.add_argument(
        "--post_frames", type=int, nargs="+", default=DEFAULT_POST_FRAMES
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parameter_root(data_root: Path, sequence: str, frame: int) -> Path:
    return data_root / sequence / sequence / "person0" / "parameter" / str(frame)


def gt_c2w(
    data_root: Path, sequence: str, camera: int, frame: int
) -> np.ndarray:
    path = parameter_root(data_root, sequence, frame) / f"{camera}_extrinsic.npy"
    value = np.load(path)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3] = np.asarray(value, dtype=np.float64)
    return np.linalg.inv(world_to_camera)


def gt_intrinsics(data_root: Path, sequence: str, camera: int) -> np.ndarray:
    path = data_root / f"{sequence}_original_video" / "calibration_new.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    value = np.asarray(calibration[str(camera)]["K"], dtype=np.float64).reshape(3, 3)
    # The dataset calibration is for 2048x2048 frames; the saved demo RGB is 512x512.
    value[:2] *= 0.25
    return value.astype(np.float32)


def mesh_path(
    data_root: Path, sequence: str, identity: str, frame: int
) -> Path:
    return (
        data_root
        / sequence
        / sequence
        / identity
        / "smplx"
        / str(frame)
        / "smplx.obj"
    )


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(
                    (float(fields[1]), float(fields[2]), float(fields[3]))
                )
    value = np.asarray(vertices, dtype=np.float32)
    if value.shape != (10475, 3):
        raise ValueError(f"Unexpected SMPL-X mesh {value.shape}: {path}")
    return value


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.einsum("ij,...j->...i", transform[:3, :3], points)
        + transform[:3, 3]
    ).astype(np.float32)


def replace_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".new")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def copy_variant(raw_payload: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        shutil.rmtree(destination)
    shutil.copytree(raw_payload, destination)


def main() -> None:
    args = parse_args()
    raw_payload = args.raw_payload.resolve()
    output_root = args.output_root.resolve()
    if not raw_payload.is_dir():
        raise FileNotFoundError(raw_payload)
    if str(output_root) in ("/", "/data"):
        raise ValueError(f"Refusing broad output target: {output_root}")

    pre_frames = [int(value) for value in args.pre_frames]
    post_frames = [int(value) for value in args.post_frames]
    frames = pre_frames + post_frames
    cameras = [int(args.source_camera)] * len(pre_frames) + [
        int(args.target_camera)
    ] * len(post_frames)
    camera_files = sorted((raw_payload / "camera").glob("*.npz"))
    if len(camera_files) != len(frames):
        raise RuntimeError(
            f"Payload has {len(camera_files)} frames, specification has {len(frames)}"
        )

    with np.load(raw_payload / "camera" / f"{len(pre_frames)-1:06d}.npz") as value:
        predicted_pre_pose = np.asarray(value["pose"], dtype=np.float64)
    gt_pre_pose = gt_c2w(
        args.data_root, args.sequence, int(args.source_camera), pre_frames[-1]
    )
    evaluation_gauge = predicted_pre_pose @ np.linalg.inv(gt_pre_pose)

    faces_path = REPO_ROOT / "src/models/smplx/SMPLX_NEUTRAL.npz"
    with np.load(faces_path, allow_pickle=True) as model:
        faces = np.asarray(model["f"], dtype=np.int32)

    output_root.mkdir(parents=True, exist_ok=True)
    destinations = {variant: output_root / variant for variant in VARIANTS}
    for destination in destinations.values():
        copy_variant(raw_payload, destination, bool(args.overwrite))

    camera_poses = []
    for index, (camera, frame) in enumerate(zip(cameras, frames)):
        pose = evaluation_gauge @ gt_c2w(
            args.data_root, args.sequence, camera, frame
        )
        camera_poses.append(pose)
        for variant, destination in destinations.items():
            path = destination / "camera" / f"{index:06d}.npz"
            with np.load(path) as source:
                values = {key: source[key] for key in source.files}
            values["pose"] = pose.astype(np.float32)
            if variant == "gt_camera_gt_humans":
                values["intrinsics"] = gt_intrinsics(
                    args.data_root, args.sequence, camera
                )
            replace_npz(path, values)

        gt_vertices = np.stack(
            [
                transform_points(
                    evaluation_gauge,
                    load_obj_vertices(
                        mesh_path(
                            args.data_root, args.sequence, identity, frame
                        )
                    ),
                )
                for identity in IDENTITIES
            ]
        )
        smpl_path = (
            destinations["gt_camera_gt_humans"]
            / "smpl"
            / f"{index:06d}.npz"
        )
        with np.load(smpl_path, allow_pickle=True) as source:
            smpl_values = {key: source[key] for key in source.files}
        smpl_values["verts_world"] = gt_vertices
        smpl_values["faces"] = faces
        smpl_values["smpl_id"] = np.arange(len(IDENTITIES), dtype=np.int64)
        replace_npz(smpl_path, smpl_values)
        # This Real-World-Capture split has no GT scene depth. Suppress the
        # Human3R pointmap in the all-GT page instead of presenting it as GT.
        confidence_path = (
            destinations["gt_camera_gt_humans"]
            / "conf"
            / f"{index:06d}.npy"
        )
        confidence = np.load(confidence_path)
        np.save(confidence_path, np.zeros_like(confidence))

    manifest = {
        "case": "three_t0900_c0_c3",
        "sequence": args.sequence,
        "source_camera": int(args.source_camera),
        "target_camera": int(args.target_camera),
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "cut_index": len(pre_frames),
        "identities": list(IDENTITIES),
        "evaluation_gauge": evaluation_gauge.tolist(),
        "camera_poses": [pose.tolist() for pose in camera_poses],
        "variants": {
            "gt_camera_human3r": {
                "path": str(destinations["gt_camera_human3r"]),
                "camera": "dataset GT extrinsics in the evaluation gauge",
                "humans": "original Human3R camera-local predictions",
                "background": "Human3R depth, visual context only",
            },
            "gt_camera_gt_humans": {
                "path": str(destinations["gt_camera_gt_humans"]),
                "camera": "dataset GT extrinsics and intrinsics in the evaluation gauge",
                "humans": "dataset GT SMPL-X OBJ meshes",
                "background": "disabled because this split has no dataset GT scene depth",
            },
        },
        "source_payload": str(raw_payload),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f">> GT camera + Human3R humans: {destinations['gt_camera_human3r']}")
    print(f">> GT camera + GT humans: {destinations['gt_camera_gt_humans']}")
    print(f">> manifest: {manifest_path}")


if __name__ == "__main__":
    main()
