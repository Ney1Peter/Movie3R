#!/usr/bin/env python3
"""Train the V7 implicit pose adapter from a multi-case token manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dust3r.v7_pose_adapter import HumanSceneTokenPoseAdapter, apply_left_se3_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/data-V7-stage-a/ms-aist/shot2_30f_floor_locked_human35/usable_cases_floor_locked_human_single_human.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/data-V7-training/ms-aist/shot2_30f_boundary_singlehuman_adapter"),
    )
    parser.add_argument("--input_mode", choices=sorted(HumanSceneTokenPoseAdapter.VALID_INPUT_MODES), default="human")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--max_delta_t", type=float, default=20.0)
    parser.add_argument("--max_delta_r", type=float, default=3.3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_batch_fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--target_weight", type=float, default=4.0)
    parser.add_argument("--noop_weight", type=float, default=0.05)
    parser.add_argument("--rot_loss_weight", type=float, default=4.0)
    parser.add_argument("--alpha_loss_weight", type=float, default=0.1)
    parser.add_argument("--reliability_loss_weight", type=float, default=0.05)
    parser.add_argument("--target_only", action="store_true")
    parser.add_argument(
        "--zero_raw_camera_pose_input",
        action="store_true",
        help="Hide explicit raw camera pose from the adapter by feeding zeros as the pose prior; corrections are still applied to the real raw pose.",
    )
    parser.add_argument("--case_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(manifest_path: Path, case_limit: int) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if case_limit > 0:
        cases = cases[:case_limit]
    if not cases:
        raise ValueError(f"No cases found in {manifest_path}")
    return cases


def load_dataset(cases: list[dict]) -> dict[str, np.ndarray | list[str]]:
    arrays: dict[str, list[np.ndarray]] = {
        "frame_ids": [],
        "pose_tokens": [],
        "scene_tokens": [],
        "human_tokens": [],
        "human_token_mask": [],
        "memory_tokens": [],
        "raw_camera_pose": [],
        "target_mask": [],
        "target_delta_t": [],
        "target_delta_rotvec": [],
        "target_alpha": [],
        "target_r_human": [],
        "target_r_scene": [],
    }
    case_ids = []
    case_names = []
    for case_id, case in enumerate(cases):
        token_path = Path(case["tokens_npz"])
        data = np.load(token_path)
        num_frames = int(data["frame_ids"].shape[0])
        case_names.append(case["name"])
        case_ids.append(np.full((num_frames,), case_id, dtype=np.int32))
        for key in arrays:
            arrays[key].append(data[key])
    # **========== 原始代码 ==========**
    # merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    # **========== 新代码 ==========**
    merged = {}
    for key, values in arrays.items():
        if key == "human_tokens":
            max_humans = max(value.shape[1] for value in values)
            dim = values[0].shape[2]
            padded = []
            for value in values:
                out = np.zeros((value.shape[0], max_humans, dim), dtype=value.dtype)
                out[:, : value.shape[1]] = value
                padded.append(out)
            merged[key] = np.concatenate(padded, axis=0)
        elif key == "human_token_mask":
            max_humans = max(value.shape[1] for value in values)
            padded = []
            for value in values:
                out = np.zeros((value.shape[0], max_humans), dtype=value.dtype)
                out[:, : value.shape[1]] = value
                padded.append(out)
            merged[key] = np.concatenate(padded, axis=0)
        else:
            merged[key] = np.concatenate(values, axis=0)
    # **========== 结束 ==========**
    merged["case_ids"] = np.concatenate(case_ids, axis=0)
    merged["case_names"] = case_names
    return merged


def to_tensor(array: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def make_tensors(data: dict[str, np.ndarray | list[str]], device: torch.device) -> dict[str, torch.Tensor]:
    tensors = {
        "pose_tokens": to_tensor(data["pose_tokens"], device),
        "scene_tokens": to_tensor(data["scene_tokens"], device),
        "human_tokens": to_tensor(data["human_tokens"], device),
        "memory_tokens": to_tensor(data["memory_tokens"], device),
        "raw_camera_pose": to_tensor(data["raw_camera_pose"], device),
        "target_delta_t": to_tensor(data["target_delta_t"], device),
        "target_delta_rotvec": to_tensor(data["target_delta_rotvec"], device),
        "target_alpha": to_tensor(data["target_alpha"].astype(np.float32), device),
        "target_r_human": to_tensor(data["target_r_human"].astype(np.float32), device),
        "target_r_scene": to_tensor(data["target_r_scene"].astype(np.float32), device),
        "target_mask": to_tensor(data["target_mask"].astype(np.float32), device),
        "human_token_mask": torch.from_numpy(data["human_token_mask"].astype(np.bool_)).to(device=device),
    }
    return tensors


def sample_batch(target_indices: torch.Tensor, noop_indices: torch.Tensor, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    batch_size = int(args.batch_size)
    if args.target_only or noop_indices.numel() == 0:
        choice = torch.randint(0, target_indices.numel(), (batch_size,), device=device)
        return target_indices[choice]
    num_target = max(1, min(batch_size, int(round(batch_size * float(args.target_batch_fraction)))))
    num_noop = batch_size - num_target
    target_choice = torch.randint(0, target_indices.numel(), (num_target,), device=device)
    noop_choice = torch.randint(0, noop_indices.numel(), (num_noop,), device=device)
    batch = torch.cat([target_indices[target_choice], noop_indices[noop_choice]], dim=0)
    return batch[torch.randperm(batch.numel(), device=device)]


def adapter_forward(adapter: HumanSceneTokenPoseAdapter, tensors: dict[str, torch.Tensor], batch_idx: torch.Tensor, args: argparse.Namespace):
    raw_camera_pose = tensors["raw_camera_pose"][batch_idx]
    camera_pose_input = torch.zeros_like(raw_camera_pose) if args.zero_raw_camera_pose_input else raw_camera_pose
    corrected_pose, info = adapter(
        pose_token=tensors["pose_tokens"][batch_idx],
        scene_tokens=tensors["scene_tokens"][batch_idx],
        human_tokens=tensors["human_tokens"][batch_idx],
        memory_tokens=tensors["memory_tokens"][batch_idx],
        camera_pose=camera_pose_input,
        human_token_mask=tensors["human_token_mask"][batch_idx],
    )
    if args.zero_raw_camera_pose_input:
        corrected_pose = apply_left_se3_delta(
            raw_camera_pose,
            info["v7_pose_delta_t"],
            info["v7_pose_delta_rotvec"],
            info["v7_pose_alpha"],
        )
    return corrected_pose, info


def batch_loss(adapter: HumanSceneTokenPoseAdapter, tensors: dict[str, torch.Tensor], batch_idx: torch.Tensor, args: argparse.Namespace):
    corrected_pose, info = adapter_forward(adapter, tensors, batch_idx, args)
    target_mask = tensors["target_mask"][batch_idx]
    sample_weight = torch.where(
        target_mask > 0.5,
        target_mask.new_full(target_mask.shape, float(args.target_weight)),
        target_mask.new_full(target_mask.shape, float(args.noop_weight)),
    )
    if args.target_only:
        sample_weight = target_mask.new_full(target_mask.shape, float(args.target_weight))

    target_t = tensors["target_delta_t"][batch_idx]
    target_r = tensors["target_delta_rotvec"][batch_idx]
    t_loss = F.smooth_l1_loss(info["v7_pose_delta_t"], target_t, reduction="none").mean(dim=-1)
    r_loss = F.smooth_l1_loss(info["v7_pose_delta_rotvec"], target_r, reduction="none").mean(dim=-1)
    fit_loss = (sample_weight * (t_loss + float(args.rot_loss_weight) * r_loss)).mean()

    alpha_target = tensors["target_alpha"][batch_idx].clamp(0.0, 1.0)
    alpha_pred = info["v7_pose_alpha"].reshape(-1).clamp(1e-4, 1.0 - 1e-4)
    alpha_loss = F.binary_cross_entropy(alpha_pred, alpha_target, reduction="none")
    alpha_loss = (sample_weight * alpha_loss).mean()

    r_human_loss = (info["v7_pose_r_human"].reshape(-1) - tensors["target_r_human"][batch_idx]).pow(2)
    r_scene_loss = (info["v7_pose_r_scene"].reshape(-1) - tensors["target_r_scene"][batch_idx]).pow(2)
    reliability_loss = (sample_weight * (r_human_loss + r_scene_loss)).mean()

    loss = fit_loss
    loss = loss + float(args.alpha_loss_weight) * alpha_loss
    loss = loss + float(args.reliability_loss_weight) * reliability_loss
    parts = {
        "fit_loss": fit_loss.detach(),
        "alpha_loss": alpha_loss.detach(),
        "reliability_loss": reliability_loss.detach(),
    }
    return loss, corrected_pose, info, parts


def evaluate(adapter: HumanSceneTokenPoseAdapter, tensors: dict[str, torch.Tensor], eval_indices: torch.Tensor, args: argparse.Namespace) -> dict:
    adapter.eval()
    infos = []
    losses = []
    parts = []
    with torch.no_grad():
        for start in range(0, eval_indices.numel(), int(args.batch_size)):
            batch_idx = eval_indices[start : start + int(args.batch_size)]
            loss, _, info, part = batch_loss(adapter, tensors, batch_idx, args)
            losses.append(loss.detach())
            infos.append({k: v.detach() for k, v in info.items()})
            parts.append(part)
    merged = {key: torch.cat([info[key] for info in infos], dim=0) for key in infos[0]}
    target_mask = tensors["target_mask"][eval_indices] > 0.5
    noop_mask = ~target_mask
    target_t = tensors["target_delta_t"][eval_indices]
    target_r = tensors["target_delta_rotvec"][eval_indices]
    pred_t = merged["v7_pose_delta_t"]
    pred_r = merged["v7_pose_delta_rotvec"]
    if target_mask.any():
        target_err_t = torch.linalg.norm(pred_t[target_mask] - target_t[target_mask], dim=-1).mean()
        target_err_r = torch.rad2deg(torch.linalg.norm(pred_r[target_mask] - target_r[target_mask], dim=-1)).mean()
        target_alpha = merged["v7_pose_alpha"].reshape(-1)[target_mask].mean()
        target_r_human = merged["v7_pose_r_human"].reshape(-1)[target_mask].mean()
    else:
        target_err_t = torch.zeros((), device=eval_indices.device)
        target_err_r = torch.zeros((), device=eval_indices.device)
        target_alpha = torch.zeros((), device=eval_indices.device)
        target_r_human = torch.zeros((), device=eval_indices.device)
    if noop_mask.any():
        noop_delta_t = torch.linalg.norm(pred_t[noop_mask], dim=-1).mean()
        noop_delta_r = torch.rad2deg(torch.linalg.norm(pred_r[noop_mask], dim=-1)).mean()
        noop_alpha = merged["v7_pose_alpha"].reshape(-1)[noop_mask].mean()
    else:
        noop_delta_t = torch.zeros((), device=eval_indices.device)
        noop_delta_r = torch.zeros((), device=eval_indices.device)
        noop_alpha = torch.zeros((), device=eval_indices.device)
    return {
        "loss": float(torch.stack(losses).mean().detach().cpu()),
        "fit_loss": float(torch.stack([p["fit_loss"] for p in parts]).mean().detach().cpu()),
        "alpha_loss": float(torch.stack([p["alpha_loss"] for p in parts]).mean().detach().cpu()),
        "reliability_loss": float(torch.stack([p["reliability_loss"] for p in parts]).mean().detach().cpu()),
        "target_err_t": float(target_err_t.detach().cpu()),
        "target_err_r_deg": float(target_err_r.detach().cpu()),
        "target_alpha_mean": float(target_alpha.detach().cpu()),
        "target_r_human_mean": float(target_r_human.detach().cpu()),
        "noop_delta_t_norm": float(noop_delta_t.detach().cpu()),
        "noop_delta_r_deg": float(noop_delta_r.detach().cpu()),
        "noop_alpha_mean": float(noop_alpha.detach().cpu()),
    }


# **========== 原始代码 ==========**
# def predict_all(adapter: HumanSceneTokenPoseAdapter, tensors: dict[str, torch.Tensor], indices: torch.Tensor, batch_size: int):
# **========== 新代码 ==========**
def predict_all(adapter: HumanSceneTokenPoseAdapter, tensors: dict[str, torch.Tensor], indices: torch.Tensor, batch_size: int, args: argparse.Namespace):
# **========== 结束 ==========**
    adapter.eval()
    poses = []
    infos = []
    with torch.no_grad():
        for start in range(0, indices.numel(), int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            corrected_pose, info = adapter_forward(adapter, tensors, batch_idx, args)
            poses.append(corrected_pose.detach().cpu())
            infos.append({k: v.detach().cpu() for k, v in info.items()})
    merged = {key: torch.cat([info[key] for info in infos], dim=0) for key in infos[0]}
    return torch.cat(poses, dim=0), merged


def checkpoint_payload(adapter, args, dec_dim, metrics, history, case_names):
    return {
        "adapter": adapter.state_dict(),
        "input_mode": args.input_mode,
        "dec_dim": int(dec_dim),
        "hidden_dim": int(args.hidden_dim),
        "max_delta_t": float(args.max_delta_t),
        "max_delta_r": float(args.max_delta_r),
        "metrics": metrics,
        "history": history,
        "case_names": list(case_names),
        "zero_raw_camera_pose_input": bool(args.zero_raw_camera_pose_input),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cases = load_cases(args.manifest, int(args.case_limit))
    data = load_dataset(cases)
    tensors = make_tensors(data, device)
    target_indices = torch.where(tensors["target_mask"] > 0.5)[0]
    noop_indices = torch.where(tensors["target_mask"] <= 0.5)[0]
    eval_indices = target_indices if args.target_only else torch.arange(tensors["target_mask"].numel(), device=device)
    dec_dim = int(data["pose_tokens"].shape[-1])
    adapter = HumanSceneTokenPoseAdapter(
        dec_dim=dec_dim,
        hidden_dim=int(args.hidden_dim),
        input_mode=args.input_mode,
        max_delta_t=float(args.max_delta_t),
        max_delta_r=float(args.max_delta_r),
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    setup = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "num_cases": len(cases),
        "num_frames": int(tensors["target_mask"].numel()),
        "num_target_frames": int(target_indices.numel()),
        "num_noop_frames": int(noop_indices.numel()),
        "input_mode": args.input_mode,
        "max_delta_t": float(args.max_delta_t),
        "max_delta_r": float(args.max_delta_r),
        "target_only": bool(args.target_only),
        "zero_raw_camera_pose_input": bool(args.zero_raw_camera_pose_input),
        "case_names": data["case_names"],
    }
    (args.output_dir / "train_setup.json").write_text(json.dumps(setup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(setup, sort_keys=True), flush=True)

    history = []
    best = {"loss": float("inf"), "step": -1, "metrics": None, "state_dict": None}
    for step in range(int(args.steps) + 1):
        adapter.train()
        batch_idx = sample_batch(target_indices, noop_indices, args, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = batch_loss(adapter, tensors, batch_idx, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        if step % int(args.log_every) == 0 or step == int(args.steps):
            metrics = evaluate(adapter, tensors, eval_indices, args)
            metrics["step"] = int(step)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            history.append(metrics)
            if metrics["loss"] < best["loss"]:
                best = {
                    "loss": metrics["loss"],
                    "step": int(step),
                    "metrics": metrics,
                    "state_dict": {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()},
                }
                torch.save(
                    checkpoint_payload(adapter, args, dec_dim, metrics, history, data["case_names"]),
                    args.output_dir / "checkpoint_best.pt",
                )

    final_metrics = evaluate(adapter, tensors, eval_indices, args)
    final_metrics["step"] = int(args.steps)
    torch.save(
        checkpoint_payload(adapter, args, dec_dim, final_metrics, history, data["case_names"]),
        args.output_dir / "checkpoint_last.pt",
    )
    if best["state_dict"] is not None:
        adapter.load_state_dict(best["state_dict"])
    all_indices = torch.arange(tensors["target_mask"].numel(), device=device)
    # **========== 原始代码 ==========**
    # corrected_pose, info = predict_all(adapter, tensors, all_indices, int(args.batch_size))
    # **========== 新代码 ==========**
    corrected_pose, info = predict_all(adapter, tensors, all_indices, int(args.batch_size), args)
    # **========== 结束 ==========**
    np.savez_compressed(
        args.output_dir / "train_predictions_best.npz",
        frame_ids=data["frame_ids"].astype(np.int32),
        case_ids=data["case_ids"].astype(np.int32),
        corrected_camera_pose=corrected_pose.numpy().astype(np.float32),
        pred_delta_t=info["v7_pose_delta_t"].numpy().astype(np.float32),
        pred_delta_rotvec=info["v7_pose_delta_rotvec"].numpy().astype(np.float32),
        pred_alpha=info["v7_pose_alpha"].numpy().astype(np.float32),
        pred_r_human=info["v7_pose_r_human"].numpy().astype(np.float32),
        pred_r_scene=info["v7_pose_r_scene"].numpy().astype(np.float32),
        target_mask=data["target_mask"].astype(np.bool_),
        target_delta_t=data["target_delta_t"].astype(np.float32),
        target_delta_rotvec=data["target_delta_rotvec"].astype(np.float32),
        target_alpha=data["target_alpha"].astype(np.float32),
    )
    metrics_out = {
        **setup,
        "best_step": int(best["step"]),
        "best_loss": float(best["loss"]),
        "best_metrics": best["metrics"],
        "final_metrics": final_metrics,
        "history": history,
    }
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics_out, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
