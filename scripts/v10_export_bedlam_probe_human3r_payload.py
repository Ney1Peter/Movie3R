#!/usr/bin/env python3
"""Export BEDLAM integrator probe variants as Human3R saved-output payloads.

The synthetic integrator probe is trajectory-only.  This helper wraps those
trajectories with BEDLAM RGB/SMPL-X annotations and dummy depth/conf files so
they can be inspected with scripts/view_human3r_saved_output.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from v10_bedlam_motion_integrator_probe import (
    load_bedlam_trajectory,
    make_episodes,
    make_items,
    stream_apply_variant,
)
from v10_visualize_bedlam_motion_integrator_probe import load_model, ns_from_saved_args


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "output" / "v10_bedlam_motion_integrator_probe" / "train_method_probe_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "v10_bedlam_motion_integrator_probe" / "train_method_probe_v1_human3r_payload"


VARIANTS = [
    "target_gt",
    "raw_perturbed",
    "fixed_explicit_se3",
    "current_only_mlp",
    "history_current_integrator",
    "explicit_se3_residual_integrator",
    "oracle_se3_upper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_probe_episode(run_dir: Path, episode_index: int, device: torch.device):
    metrics_json = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    saved_args = ns_from_saved_args(metrics_json["args"])
    traj = load_bedlam_trajectory(saved_args.manifest)
    episodes = make_episodes(traj, max(episode_index + 1, 1), saved_args, seed_offset=100000)
    episode = episodes[episode_index]
    items = make_items([episode])
    models = {
        "current_only_mlp": load_model(run_dir, "current_only_mlp", items[0].feature_current.numel(), saved_args, device),
        "history_current_integrator": load_model(
            run_dir, "history_current_integrator", items[0].feature_history.numel(), saved_args, device
        ),
        "explicit_se3_residual_integrator": load_model(
            run_dir,
            "explicit_se3_residual_integrator",
            items[0].feature_residual.numel(),
            saved_args,
            device,
        ),
    }
    return saved_args, episode, models


def variant_pose(episode, models, saved_args, variant: str, device: torch.device):
    if variant == "target_gt":
        return (
            episode.target_root_t,
            episode.target_root_R,
            episode.target_cam_t,
            episode.target_cam_R,
        )
    return stream_apply_variant(episode, variant, models, saved_args, device)


def clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    for name in ("camera", "smpl", "color", "depth", "conf"):
        (path / name).mkdir(parents=True, exist_ok=True)


def load_body_masks(meta: dict, frame: int, people: int, hw: tuple[int, int]) -> np.ndarray:
    masks = []
    for person in range(people):
        mask_path = Path(meta["body_mask_pattern"].format(frame=int(frame), person=person))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(hw, dtype=np.uint8)
        if mask.shape[:2] != hw:
            mask = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(mask.astype(np.float32) / 255.0)
    return np.stack(masks, axis=0)


def write_variant(
    variant_dir: Path,
    variant: str,
    episode,
    models,
    saved_args,
    meta: dict,
    npz,
    device: torch.device,
    overwrite: bool,
) -> None:
    clean_dir(variant_dir, overwrite)
    _, _, cam_t, cam_R = variant_pose(episode, models, saved_args, variant, device)
    frames = list(meta["kept_frames"])
    for out_idx, frame in enumerate(frames):
        image_path = Path(meta["image_pattern"].format(frame=int(frame)))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        shutil.copyfile(image_path, variant_dir / "color" / f"{out_idx:06d}.png")
        np.save(variant_dir / "depth" / f"{out_idx:06d}.npy", np.zeros((h, w), dtype=np.float32))
        np.save(variant_dir / "conf" / f"{out_idx:06d}.npy", np.zeros((h, w), dtype=np.float32))

        indices = meta["npz_indices_by_frame"][f"{int(frame):04d}"]
        K = np.asarray(npz["cam_int"][indices[0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = cam_R[out_idx].detach().cpu().numpy().astype(np.float32)
        pose[:3, 3] = cam_t[out_idx].detach().cpu().numpy().astype(np.float32)
        np.savez(variant_dir / "camera" / f"{out_idx:06d}.npz", pose=pose, intrinsics=K)

        pose_cam = np.asarray(npz["pose_cam"][indices], dtype=np.float32).reshape(len(indices), 55, 3)
        rotvec = pose_cam[:, :53, :]
        shape = np.asarray(npz["shape"][indices], dtype=np.float32)
        transl = np.asarray(npz["trans_cam"][indices], dtype=np.float32)
        expression = np.zeros((len(indices), 10), dtype=np.float32)
        masks = load_body_masks(meta, int(frame), len(indices), (h, w))
        np.savez(
            variant_dir / "smpl" / f"{out_idx:06d}.npz",
            scores=np.zeros((h, w), dtype=np.float32),
            msk=masks,
            shape=shape,
            rotvec=rotvec,
            transl=transl,
            expression=expression,
            smpl_id=np.arange(len(indices), dtype=np.int64),
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    saved_args, episode, models = load_probe_episode(args.run_dir, args.episode_index, device)
    meta = json.loads(Path(saved_args.manifest).read_text(encoding="utf-8"))
    npz = np.load(meta["npz_path"], allow_pickle=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"variants": {}, "run_dir": str(args.run_dir), "episode_index": int(args.episode_index)}
    for variant in VARIANTS:
        variant_dir = args.output_dir / variant
        write_variant(variant_dir, variant, episode, models, saved_args, meta, npz, device, args.overwrite)
        manifest["variants"][variant] = str(variant_dir)
        print(f"Wrote {variant}: {variant_dir}")
    (args.output_dir / "viewer_payload_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
