#!/usr/bin/env python3
"""V13.1 apply one streaming Fixed Explicit transform to a real video.

The pre-cut shot comes from the original Human3R run.  The post-cut shot comes
from a fresh Human3R run that starts at the known cut.  A single transform is
estimated from the last two pre-cut frames and the first post-cut frame, then
applied to every post-cut camera, pointmap, and SMPL-X mesh.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.utils import SMPL_Layer  # noqa: E402
from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    history_background_cloud,
    root_pose_world,
)
from v10_1_fixed_explicit_candidate_probe import (  # noqa: E402
    FIXED_EXPLICIT_NAME,
    combine_clouds,
    human_initial,
    refine_candidate,
)
from viser_utils import SceneHumanViewer  # noqa: E402


KINDS = {
    "camera": ".npz",
    "depth": ".npy",
    "conf": ".npy",
    "color": ".png",
    "smpl": ".npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre_dir", type=Path, required=True)
    parser.add_argument("--post_dir", type=Path, required=True)
    parser.add_argument("--cut_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cloud_points_per_frame", type=int, default=5000)
    parser.add_argument("--viewer_point_downsample", type=int, default=20)
    parser.add_argument("--viewer_smpl_downsample", type=int, default=5)
    parser.add_argument("--viewer_camera_downsample", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def frame_count(path: Path) -> int:
    return len(list((path / "camera").glob("*.npz")))


def frame_path(path: Path, kind: str, index: int) -> Path:
    return path / kind / f"{index:06d}{KINDS[kind]}"


def link_boundary_frame(cache: Path, target_index: int, source: Path, source_index: int) -> None:
    for kind in KINDS:
        src = frame_path(source, kind, source_index)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = frame_path(cache, kind, target_index)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


def build_boundary_cache(args: argparse.Namespace) -> Path:
    if args.cut_idx < 2:
        raise ValueError("Fixed Explicit requires at least two pre-cut frames")
    cache = args.output_dir / "boundary_cache"
    link_boundary_frame(cache, 0, args.pre_dir, args.cut_idx - 2)
    link_boundary_frame(cache, 1, args.pre_dir, args.cut_idx - 1)
    link_boundary_frame(cache, 2, args.post_dir, 0)
    return cache


def rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3] @ right[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def estimate_fixed_transform(cache: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    initial = human_initial(cache, [0, 1], 2, mode="mean")
    target, target_debug = history_background_cloud(
        cache,
        [0, 1],
        int(args.cloud_points_per_frame),
    )
    source, source_debug = combine_clouds(
        cache,
        [2],
        int(args.cloud_points_per_frame),
        seed=20260721,
    )
    candidate = refine_candidate(
        FIXED_EXPLICIT_NAME,
        initial,
        source,
        target,
        "standard",
        {"source": source_debug, "target": target_debug},
    )

    _, target_root = root_pose_world(cache, 1)
    _, current_root = root_pose_world(cache, 2)
    corrected_root = (
        candidate.transform[:3, :3] @ current_root + candidate.transform[:3, 3]
    )
    with np.load(cache / "camera" / "000001.npz") as pre_camera:
        pre_pose = pre_camera["pose"].astype(np.float32)
    with np.load(cache / "camera" / "000002.npz") as post_camera:
        post_pose = post_camera["pose"].astype(np.float32)
    corrected_post_pose = candidate.transform @ post_pose

    diagnostics = {
        "method": candidate.name,
        "streaming_window": {
            "pre_frames": [int(args.cut_idx - 2), int(args.cut_idx - 1)],
            "post_frames": [int(args.cut_idx)],
            "post_wait_frames": 0,
        },
        "transform": candidate.transform.tolist(),
        "transform_translation_m": float(np.linalg.norm(candidate.transform[:3, 3])),
        "transform_rotation_deg": rotation_distance_deg(
            candidate.transform,
            np.eye(4, dtype=np.float32),
        ),
        "last_pre_to_first_post_camera_center_m_before": float(
            np.linalg.norm(pre_pose[:3, 3] - post_pose[:3, 3])
        ),
        "last_pre_to_first_post_camera_center_m_after": float(
            np.linalg.norm(pre_pose[:3, 3] - corrected_post_pose[:3, 3])
        ),
        "last_pre_to_first_post_camera_rotation_deg_before": rotation_distance_deg(
            pre_pose,
            post_pose,
        ),
        "last_pre_to_first_post_camera_rotation_deg_after": rotation_distance_deg(
            pre_pose,
            corrected_post_pose,
        ),
        "human_root_jump_m_before": float(np.linalg.norm(target_root - current_root)),
        "human_root_jump_m_after": float(np.linalg.norm(target_root - corrected_root)),
        "candidate_diagnostics": candidate.diagnostics,
    }
    return candidate.transform.astype(np.float32), diagnostics


def load_record(path: Path, index: int) -> dict:
    with np.load(frame_path(path, "camera", index)) as camera:
        pose = camera["pose"].astype(np.float32)
        intrinsics = camera["intrinsics"].astype(np.float32)
    with np.load(frame_path(path, "smpl", index), allow_pickle=True) as archive:
        smpl = {key: archive[key] for key in archive.files}
    return {
        "pose": pose,
        "intrinsics": intrinsics,
        "depth": np.load(frame_path(path, "depth", index)).astype(np.float32),
        "confidence": np.load(frame_path(path, "conf", index)).astype(np.float32),
        "color": np.asarray(Image.open(frame_path(path, "color", index)).convert("RGB")),
        "smpl": smpl,
    }


def reconstruct_world_pointmap(pose: np.ndarray, intrinsics: np.ndarray, depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    points_camera = np.stack(
        [
            (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
            (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
            depth,
        ],
        axis=-1,
    )
    return (
        np.einsum("ij,hwj->hwi", pose[:3, :3], points_camera)
        + pose[:3, 3]
    ).astype(np.float32)


def normalize_expression(value: np.ndarray | None, count: int) -> np.ndarray:
    if value is None or getattr(value, "dtype", None) == object or len(value) == 0:
        return np.zeros((count, 10), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(count, -1)


def build_smpl_vertices(records: list[dict], device: torch.device) -> tuple[list[np.ndarray], np.ndarray]:
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    vertices_by_frame: list[np.ndarray] = []

    with torch.no_grad():
        for record in records:
            smpl = record["smpl"]
            count = int(len(smpl["shape"]))
            if count == 0:
                vertices_by_frame.append(np.empty((0, 0, 3), dtype=np.float32))
                continue
            pose = torch.from_numpy(np.asarray(smpl["rotvec"], dtype=np.float32)).to(device)
            shape = torch.from_numpy(np.asarray(smpl["shape"], dtype=np.float32)).to(device)
            translation = torch.from_numpy(np.asarray(smpl["transl"], dtype=np.float32)).to(device)
            intrinsics = torch.from_numpy(record["intrinsics"]).to(device).expand(count, -1, -1)
            expression = torch.from_numpy(normalize_expression(smpl.get("expression"), count)).to(device)
            output = layer(
                pose,
                shape,
                translation,
                None,
                None,
                K=intrinsics,
                expression=expression,
            )
            vertices_camera = output["smpl_v3d"].detach().float().cpu().numpy()
            world = (
                np.einsum("ij,nvj->nvi", record["pose"][:3, :3], vertices_camera)
                + record["pose"][None, :3, 3]
            ).astype(np.float32)
            vertices_by_frame.append(world)

    del layer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vertices_by_frame, faces


def load_aligned_sequence(
    args: argparse.Namespace,
    transform: np.ndarray,
) -> tuple[list[dict], list[int]]:
    pre_count = frame_count(args.pre_dir)
    post_count = frame_count(args.post_dir)
    expected_post = pre_count - int(args.cut_idx)
    if post_count != expected_post:
        raise ValueError(
            f"Post-cut frame count mismatch: got {post_count}, expected {expected_post} "
            f"from pre_count={pre_count}, cut_idx={args.cut_idx}"
        )

    records = []
    original_indices = []
    for index in range(args.cut_idx):
        records.append(load_record(args.pre_dir, index))
        original_indices.append(index)
    for index in range(post_count):
        record = load_record(args.post_dir, index)
        record["pose"] = (transform @ record["pose"]).astype(np.float32)
        records.append(record)
        original_indices.append(args.cut_idx + index)
    return records, original_indices


def build_viewer(args: argparse.Namespace, records: list[dict], original_indices: list[int]) -> SceneHumanViewer:
    device = torch.device(args.device)
    vertices, faces = build_smpl_vertices(records, device)
    pointmaps = []
    colors = []
    confidences = []
    masks = []
    ids = []
    rotations = []
    translations = []
    focals = []
    principal_points = []

    for record in records:
        depth = record["depth"]
        smpl = record["smpl"]
        pointmaps.append(
            reconstruct_world_pointmap(record["pose"], record["intrinsics"], depth)[None]
        )
        colors.append((record["color"].astype(np.float32) / 255.0)[None])
        confidences.append(record["confidence"][None])
        mask = smpl.get("msk")
        if mask is None or getattr(mask, "dtype", None) == object:
            masks.append(np.zeros((1, *depth.shape), dtype=np.float32))
        else:
            mask_array = np.asarray(mask, dtype=np.float32)
            masks.append(mask_array if mask_array.ndim == 3 else mask_array[None])
        ids.append(np.arange(len(smpl["shape"]), dtype=np.int64))
        rotations.append(record["pose"][:3, :3])
        translations.append(record["pose"][:3, 3])
        focals.append(float(record["intrinsics"][0, 0]))
        principal_points.append(record["intrinsics"][:2, 2])

    cam_dict = {
        "focal": np.asarray(focals, dtype=np.float32),
        "pp": np.stack(principal_points).astype(np.float32),
        "R": np.stack(rotations).astype(np.float32),
        "t": np.stack(translations).astype(np.float32),
    }
    viewer = SceneHumanViewer(
        pointmaps,
        colors,
        confidences,
        cam_dict,
        vertices,
        faces,
        ids,
        masks,
        edge_color_list=[None] * len(records),
        device="cpu",
        port=args.port,
        show_camera=True,
        vis_threshold=1.0,
        msk_threshold=0.1,
        size=512,
        downsample_factor=args.viewer_point_downsample,
        smpl_downsample_factor=args.viewer_smpl_downsample,
        camera_downsample_factor=args.viewer_camera_downsample,
    )
    viewer.original_frame_indices = original_indices
    return viewer


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = build_boundary_cache(args)
    transform, diagnostics = estimate_fixed_transform(cache, args)
    diagnostics_path = args.output_dir / "fixed_explicit_alignment.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)

    records, original_indices = load_aligned_sequence(args, transform)
    viewer = build_viewer(args, records, original_indices)
    print(
        f"Fixed Explicit aligned viewer: http://127.0.0.1:{args.port} "
        f"(cut at original frame {args.cut_idx}; diagnostics: {diagnostics_path})",
        flush=True,
    )
    viewer.run()


if __name__ == "__main__":
    main()
