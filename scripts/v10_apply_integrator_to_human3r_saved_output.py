#!/usr/bin/env python3
"""Apply a trained V10 integrator probe to a saved Human3R output.

This creates Human3R-compatible saved-output directories with real depth/color/
conf/SMPL payloads.  Only camera poses are changed, so the viewer shows the
point cloud, camera frustums, and people in the transformed world gauge.

The current probe setting is synthetic: start from a continuous Human3R output,
apply random segment gauge perturbations, then use the trained integrator heads
to attach perturbed segments back to the original world.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from v10_bedlam_motion_integrator_probe import (
    load_human3r_saved_trajectory,
    make_episode,
    make_items,
    stream_apply_variant,
)
from v10_visualize_bedlam_motion_integrator_probe import load_model, ns_from_saved_args


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "output" / "v10_bedlam_motion_integrator_probe" / "train_method_probe_v1"
DEFAULT_INPUT_DIR = REPO_ROOT / "output" / "v10_bedlam_seq21_original_human3r" / "original_human3r_demo"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "v10_bedlam_seq21_original_human3r" / "v10_integrator_train_method_probe_v1"

VARIANTS = [
    "target_original_human3r",
    "raw_perturbed",
    "fixed_explicit_se3",
    "history_current_integrator",
    "history_direct_residual_integrator",
    "history_bidir_consistency_integrator",
    "history_bidir_direct_residual_integrator",
    "explicit_se3_residual_integrator",
    "oracle_se3_upper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_frames", type=int, default=29)
    parser.add_argument("--max_people", type=int, default=4)
    parser.add_argument("--segment_boundaries", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--perturb_rot_deg", type=float, default=None)
    parser.add_argument("--perturb_trans", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_run_args(run_dir: Path):
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    return ns_from_saved_args(metrics["args"])


def load_models(run_dir: Path, episode, saved_args, device: torch.device):
    items = make_items([episode])
    models = {
        "current_only_mlp": load_model(run_dir, "current_only_mlp", items[0].feature_current.numel(), saved_args, device),
        "history_current_integrator": load_model(
            run_dir, "history_current_integrator", items[0].feature_history.numel(), saved_args, device
        ),
        "history_direct_residual_integrator": load_model(
            run_dir,
            "history_direct_residual_integrator",
            _history_direct_residual_dim(run_dir, episode, saved_args, device),
            saved_args,
            device,
        ),
        "explicit_se3_residual_integrator": load_model(
            run_dir,
            "explicit_se3_residual_integrator",
            items[0].feature_residual.numel(),
            saved_args,
            device,
        ),
    }
    if (run_dir / "history_bidir_consistency_integrator.pth").is_file():
        models["history_bidir_consistency_integrator"] = load_model(
            run_dir,
            "history_bidir_consistency_integrator",
            items[0].feature_history.numel(),
            saved_args,
            device,
        )
    if (run_dir / "history_bidir_direct_residual_integrator.pth").is_file():
        models["history_bidir_direct_residual_integrator"] = load_model(
            run_dir,
            "history_bidir_direct_residual_integrator",
            _history_direct_residual_dim(
                run_dir,
                episode,
                saved_args,
                device,
                direct_name="history_bidir_consistency_integrator",
            ),
            saved_args,
            device,
        )
    return models


def _history_direct_residual_dim(
    run_dir: Path,
    episode,
    saved_args,
    device: torch.device,
    direct_name: str = "history_current_integrator",
) -> int:
    direct_model = load_model(
        run_dir,
        direct_name,
        make_items([episode])[0].feature_history.numel(),
        saved_args,
        device,
    )
    items = make_items([episode])
    from v10_bedlam_motion_integrator_probe import (
        apply_transform_batch,
        build_history_direct_residual_features,
    )

    with torch.no_grad():
        features = items[0].feature_history[None].to(device)
        R_direct, t_direct = direct_model(features)
        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
            items[0].local_root_t[None].to(device),
            items[0].local_root_R[None].to(device),
            items[0].local_cam_t[None].to(device),
            items[0].local_cam_R[None].to(device),
            R_direct,
            t_direct,
        )
        residual_features = build_history_direct_residual_features(
            features,
            R_direct,
            t_direct,
            coarse_root_t,
            coarse_root_R,
            coarse_cam_t,
            coarse_cam_R,
        )
    return int(residual_features.shape[-1])


def clean_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    for name in ("camera", "smpl", "depth", "conf", "color"):
        (path / name).mkdir(parents=True, exist_ok=True)


def copy_payload_with_camera(input_dir: Path, output_dir: Path, cam_t: torch.Tensor, cam_R: torch.Tensor, overwrite: bool) -> None:
    clean_output_dir(output_dir, overwrite)
    num_frames = cam_t.shape[0]
    for idx in range(num_frames):
        for name, suffix in (("depth", ".npy"), ("conf", ".npy")):
            shutil.copyfile(input_dir / name / f"{idx:06d}{suffix}", output_dir / name / f"{idx:06d}{suffix}")
        for name, suffix in (("color", ".png"), ("smpl", ".npz")):
            shutil.copyfile(input_dir / name / f"{idx:06d}{suffix}", output_dir / name / f"{idx:06d}{suffix}")

        src = np.load(input_dir / "camera" / f"{idx:06d}.npz")
        pose = src["pose"].astype("float32").copy()
        pose[:3, :3] = cam_R[idx].detach().cpu().numpy().astype("float32")
        pose[:3, 3] = cam_t[idx].detach().cpu().numpy().astype("float32")
        np.savez(
            output_dir / "camera" / f"{idx:06d}.npz",
            pose=pose,
            intrinsics=src["intrinsics"].astype("float32"),
        )


def variant_camera(episode, models, saved_args, variant: str, device: torch.device):
    if variant == "target_original_human3r":
        return episode.target_cam_t, episode.target_cam_R
    pred = stream_apply_variant(episode, variant, models, saved_args, device)
    return pred[2], pred[3]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    saved_args = load_run_args(args.run_dir)
    saved_args.segment_boundaries = list(args.segment_boundaries)
    saved_args.global_rot_deg = 0.0
    saved_args.global_trans = 0.0
    if args.perturb_rot_deg is not None:
        saved_args.perturb_rot_deg = float(args.perturb_rot_deg)
    if args.perturb_trans is not None:
        saved_args.perturb_trans = float(args.perturb_trans)
    saved_args.seed = int(args.seed)

    traj = load_human3r_saved_trajectory(args.input_dir, args.num_frames, args.max_people)

    rng = np.random.default_rng(saved_args.seed + 100000)
    episode = make_episode(
        traj,
        list(saved_args.segment_boundaries),
        float(saved_args.perturb_rot_deg),
        float(saved_args.perturb_trans),
        0.0,
        0.0,
        rng,
    )
    models = load_models(args.run_dir, episode, saved_args, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_dir": str(args.input_dir),
        "run_dir": str(args.run_dir),
        "num_frames": int(args.num_frames),
        "max_people": int(args.max_people),
        "segment_boundaries": list(saved_args.segment_boundaries),
        "perturb_rot_deg": float(saved_args.perturb_rot_deg),
        "perturb_trans": float(saved_args.perturb_trans),
        "variants": {},
    }
    for variant in VARIANTS:
        if variant.endswith("_integrator") and variant not in models:
            print(f"Skip {variant}: model not found in {args.run_dir}")
            continue
        out_dir = args.output_dir / variant
        cam_t, cam_R = variant_camera(episode, models, saved_args, variant, device)
        copy_payload_with_camera(args.input_dir, out_dir, cam_t, cam_R, args.overwrite)
        manifest["variants"][variant] = str(out_dir)
        print(f"Wrote {variant}: {out_dir}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
