#!/usr/bin/env python3
"""Overfit the V7 implicit Human-Scene Token Adapter on dumped token features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dust3r.v7_pose_adapter import HumanSceneTokenPoseAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens_npz", type=Path, required=True)
    parser.add_argument("--labels_npz", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--input_mode",
        choices=sorted(HumanSceneTokenPoseAdapter.VALID_INPUT_MODES),
        default="human_scene",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--max_delta_t", type=float, default=3.0)
    parser.add_argument("--max_delta_r", type=float, default=0.75)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--target_weight", type=float, default=4.0)
    parser.add_argument("--noop_weight", type=float, default=0.25)
    parser.add_argument("--rot_loss_weight", type=float, default=4.0)
    parser.add_argument("--alpha_loss_weight", type=float, default=0.2)
    parser.add_argument("--reliability_loss_weight", type=float, default=0.1)
    parser.add_argument("--include_noop", action="store_true")
    parser.add_argument("--train_start", type=int, default=None)
    parser.add_argument("--train_end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_array(data, key, required=True):
    if key in data.files:
        return data[key]
    if required:
        raise KeyError(f"missing {key!r} in {data.filename}")
    return None


def _align_labels(frame_ids: np.ndarray, labels_npz: Path) -> dict[str, np.ndarray]:
    labels = np.load(labels_npz)
    label_frames = labels["frame_ids"].astype(np.int64)
    index = {int(frame): i for i, frame in enumerate(label_frames)}
    target_mask = np.zeros((len(frame_ids),), dtype=np.bool_)
    target_delta_t = np.zeros((len(frame_ids), 3), dtype=np.float32)
    target_delta_rotvec = np.zeros((len(frame_ids), 3), dtype=np.float32)
    target_alpha = np.zeros((len(frame_ids),), dtype=np.float32)
    target_r_human = np.zeros((len(frame_ids),), dtype=np.float32)
    target_r_scene = np.zeros((len(frame_ids),), dtype=np.float32)
    for i, frame in enumerate(frame_ids.tolist()):
        j = index.get(int(frame))
        if j is None:
            continue
        target_mask[i] = True
        target_delta_t[i] = labels["delta_t"][j]
        target_delta_rotvec[i] = labels["delta_rotvec"][j]
        target_alpha[i] = labels["alpha"][j]
        target_r_human[i] = labels["r_human"][j]
        target_r_scene[i] = labels["r_scene"][j]
    return {
        "target_mask": target_mask,
        "target_delta_t": target_delta_t,
        "target_delta_rotvec": target_delta_rotvec,
        "target_alpha": target_alpha,
        "target_r_human": target_r_human,
        "target_r_scene": target_r_scene,
    }


def load_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    data = np.load(args.tokens_npz)
    frame_ids = _load_array(data, "frame_ids").astype(np.int64)
    result = {
        "frame_ids": frame_ids,
        "pose_tokens": _load_array(data, "pose_tokens").astype(np.float32),
        "scene_tokens": _load_array(data, "scene_tokens").astype(np.float32),
        "human_tokens": _load_array(data, "human_tokens").astype(np.float32),
        "memory_tokens": _load_array(data, "memory_tokens").astype(np.float32),
        "raw_camera_pose": _load_array(data, "raw_camera_pose").astype(np.float32),
        "human_token_mask": _load_array(data, "human_token_mask", required=False),
    }
    if result["human_token_mask"] is not None:
        result["human_token_mask"] = result["human_token_mask"].astype(np.bool_)
    if "target_mask" in data.files:
        for key in [
            "target_mask",
            "target_delta_t",
            "target_delta_rotvec",
            "target_alpha",
            "target_r_human",
            "target_r_scene",
        ]:
            result[key] = data[key]
        result["target_mask"] = result["target_mask"].astype(np.bool_)
    elif args.labels_npz is not None:
        result.update(_align_labels(frame_ids, args.labels_npz))
    else:
        raise ValueError("No labels found in token dump and --labels_npz was not provided")
    return result


def select_indices(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    frame_ids = data["frame_ids"]
    keep = np.ones((len(frame_ids),), dtype=np.bool_)
    if args.train_start is not None:
        keep &= frame_ids >= int(args.train_start)
    if args.train_end is not None:
        keep &= frame_ids <= int(args.train_end)
    if not args.include_noop:
        keep &= data["target_mask"].astype(np.bool_)
    indices = np.where(keep)[0]
    if len(indices) == 0:
        raise ValueError("No training frames selected")
    return indices.astype(np.int64)


def to_tensor(array: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def batch_loss(adapter, tensors, batch_idx, args):
    human_mask = tensors.get("human_token_mask")
    if human_mask is not None:
        human_mask = human_mask[batch_idx]
    corrected_pose, info = adapter(
        pose_token=tensors["pose_tokens"][batch_idx],
        scene_tokens=tensors["scene_tokens"][batch_idx],
        human_tokens=tensors["human_tokens"][batch_idx],
        memory_tokens=tensors["memory_tokens"][batch_idx],
        camera_pose=tensors["raw_camera_pose"][batch_idx],
        human_token_mask=human_mask,
    )
    target_t = tensors["target_delta_t"][batch_idx]
    target_r = tensors["target_delta_rotvec"][batch_idx]
    target_mask = tensors["target_mask"][batch_idx]
    sample_weight = torch.where(
        target_mask > 0.5,
        target_mask.new_full(target_mask.shape, float(args.target_weight)),
        target_mask.new_full(target_mask.shape, float(args.noop_weight)),
    )
    if not args.include_noop:
        sample_weight = target_mask.new_full(target_mask.shape, float(args.target_weight))

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
    return loss, corrected_pose, info, {
        "fit_loss": fit_loss.detach(),
        "alpha_loss": alpha_loss.detach(),
        "reliability_loss": reliability_loss.detach(),
    }


def evaluate(adapter, tensors, indices, args):
    adapter.eval()
    preds = []
    infos = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), int(args.batch_size)):
            batch_idx = indices[start : start + int(args.batch_size)]
            loss, corrected_pose, info, _ = batch_loss(adapter, tensors, batch_idx, args)
            losses.append(loss.detach())
            preds.append(corrected_pose.detach().cpu())
            infos.append({k: v.detach().cpu() for k, v in info.items()})
    merged = {key: torch.cat([info[key] for info in infos], dim=0) for key in infos[0]}
    return torch.stack(losses).mean(), torch.cat(preds, dim=0), merged


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

    data = load_dataset(args)
    indices_np = select_indices(data, args)
    dec_dim = int(data["pose_tokens"].shape[-1])
    adapter = HumanSceneTokenPoseAdapter(
        dec_dim=dec_dim,
        hidden_dim=int(args.hidden_dim),
        input_mode=args.input_mode,
        max_delta_t=float(args.max_delta_t),
        max_delta_r=float(args.max_delta_r),
        dropout=float(args.dropout),
    ).to(device)
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
    }
    if data.get("human_token_mask") is not None:
        tensors["human_token_mask"] = torch.from_numpy(data["human_token_mask"]).to(device=device)

    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    best = {
        "loss": float("inf"),
        "step": -1,
        "state_dict": None,
    }
    for step in range(int(args.steps) + 1):
        adapter.train()
        perm = indices[torch.randperm(indices.numel(), device=device)]
        running = []
        for start in range(0, perm.numel(), int(args.batch_size)):
            batch_idx = perm[start : start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = batch_loss(adapter, tensors, batch_idx, args)
            loss.backward()
            optimizer.step()
            running.append(loss.detach())
        if step % int(args.log_every) == 0 or step == int(args.steps):
            eval_loss, _, eval_info = evaluate(adapter, tensors, indices, args)
            target_mask = tensors["target_mask"][indices] > 0.5
            pred_t = eval_info["v7_pose_delta_t"].to(device)
            pred_r = eval_info["v7_pose_delta_rotvec"].to(device)
            target_t = tensors["target_delta_t"][indices]
            target_r = tensors["target_delta_rotvec"][indices]
            if target_mask.any():
                target_err_t = torch.linalg.norm(pred_t[target_mask] - target_t[target_mask], dim=-1).mean()
                target_err_r = torch.rad2deg(torch.linalg.norm(pred_r[target_mask] - target_r[target_mask], dim=-1)).mean()
            else:
                target_err_t = torch.zeros((), device=device)
                target_err_r = torch.zeros((), device=device)
            noop_mask = ~target_mask
            noop_delta_t = torch.linalg.norm(pred_t[noop_mask], dim=-1).mean() if noop_mask.any() else torch.zeros((), device=device)
            record = {
                "step": int(step),
                "loss": float(eval_loss.cpu()),
                "train_loss": float(torch.stack(running).mean().cpu()),
                "target_err_t": float(target_err_t.detach().cpu()),
                "target_err_r_deg": float(target_err_r.detach().cpu()),
                "noop_delta_t_norm": float(noop_delta_t.detach().cpu()),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            history.append(record)
            if record["loss"] < best["loss"]:
                best["loss"] = record["loss"]
                best["step"] = int(step)
                best["state_dict"] = {
                    key: value.detach().cpu().clone()
                    for key, value in adapter.state_dict().items()
                }

    if best["state_dict"] is not None:
        adapter.load_state_dict(best["state_dict"])
    final_loss, corrected_pose, info = evaluate(adapter, tensors, indices, args)
    selected_frame_ids = data["frame_ids"][indices_np].astype(np.int32)
    np.savez_compressed(
        args.output_dir / "v7_implicit_student_predictions.npz",
        frame_ids=selected_frame_ids,
        corrected_camera_pose=corrected_pose.numpy().astype(np.float32),
        pred_delta_t=info["v7_pose_delta_t"].numpy().astype(np.float32),
        pred_delta_rotvec=info["v7_pose_delta_rotvec"].numpy().astype(np.float32),
        pred_alpha=info["v7_pose_alpha"].numpy().astype(np.float32),
        pred_r_human=info["v7_pose_r_human"].numpy().astype(np.float32),
        pred_r_scene=info["v7_pose_r_scene"].numpy().astype(np.float32),
        target_mask=data["target_mask"][indices_np].astype(np.bool_),
        target_delta_t=data["target_delta_t"][indices_np].astype(np.float32),
        target_delta_rotvec=data["target_delta_rotvec"][indices_np].astype(np.float32),
        target_alpha=data["target_alpha"][indices_np].astype(np.float32),
    )
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "input_mode": args.input_mode,
            "dec_dim": dec_dim,
            "hidden_dim": int(args.hidden_dim),
            "max_delta_t": float(args.max_delta_t),
            "max_delta_r": float(args.max_delta_r),
            "best_step": int(best["step"]),
            "best_loss": float(best["loss"]),
        },
        args.output_dir / "v7_implicit_student_adapter.pt",
    )
    metrics = {
        "tokens_npz": str(args.tokens_npz),
        "labels_npz": str(args.labels_npz) if args.labels_npz else None,
        "output_dir": str(args.output_dir),
        "input_mode": args.input_mode,
        "num_train_frames": int(len(indices_np)),
        "num_target_frames": int(data["target_mask"][indices_np].sum()),
        "final_loss": float(final_loss.cpu()),
        "best_step": int(best["step"]),
        "best_loss": float(best["loss"]),
        "history": history,
        "causal_inputs_only": True,
    }
    with open(args.output_dir / "v7_implicit_student_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
