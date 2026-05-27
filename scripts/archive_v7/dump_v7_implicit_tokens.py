#!/usr/bin/env python3
"""Dump causal Human3R internal tokens for V7 implicit adapter overfit.

The dumped inputs are limited to current-frame Human3R tokens and recurrent
memory. They intentionally do not include decoded SMPL bodies, explicit scene
planes, or any future-frame teacher evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from add_ckpt_path import add_path_to_dust3r
from demo import parse_seq_path, prepare_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--seq_path", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--pseudo_labels", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pool_scene_tokens",
        action="store_true",
        help="Store only mean-pooled scene tokens as (frames, 1, C) to reduce disk usage.",
    )
    parser.add_argument(
        "--pool_memory_tokens",
        action="store_true",
        help="Store only mean-pooled memory tokens as (frames, 1, C) to reduce disk usage.",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=10000000)
    parser.add_argument("--use_ttt3r", action="store_true")
    parser.add_argument(
        "--enable_v7_pose_adapter",
        action="store_true",
        help="Keep an already-trained V7 adapter enabled while dumping its inputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _frame_id(view: dict, fallback: int) -> int:
    value = view.get("idx", fallback)
    if torch.is_tensor(value):
        value = value.reshape(-1)[0].item()
    return int(value)


def _stack_required(preds: list[dict], key: str) -> np.ndarray:
    values = []
    for i, pred in enumerate(preds):
        if key not in pred:
            raise KeyError(f"missing {key!r} in prediction {i}; set model.return_v7_pose_adapter_inputs=True")
        values.append(_to_numpy(pred[key]))
    return np.concatenate(values, axis=0).astype(np.float32)


def _maybe_pool_tokens(tokens: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected token array with shape (frames, tokens, dim), got {tokens.shape}")
    return tokens.mean(axis=1, keepdims=True).astype(np.float32)


def _stack_human_tokens(preds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    max_humans = 0
    dim = None
    for pred in preds:
        value = pred.get("v7_human_tokens_input", None)
        if value is None:
            arr = np.zeros((1, 0, 0), dtype=np.float32)
        else:
            arr = _to_numpy(value).astype(np.float32)
        arrays.append(arr)
        max_humans = max(max_humans, arr.shape[1])
        if arr.shape[1] > 0:
            dim = arr.shape[2]
    if dim is None:
        pose = preds[0]["v7_pose_token_input"]
        dim = int(pose.shape[-1])
    tokens = np.zeros((len(arrays), max_humans, dim), dtype=np.float32)
    mask = np.zeros((len(arrays), max_humans), dtype=np.bool_)
    for i, arr in enumerate(arrays):
        if arr.shape[1] == 0:
            continue
        n = arr.shape[1]
        tokens[i, :n] = arr[0]
        mask[i, :n] = True
    return tokens, mask


def _align_labels(frame_ids: np.ndarray, labels_path: Path) -> dict[str, np.ndarray]:
    labels = np.load(labels_path)
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
        "label_frame_ids": label_frames.astype(np.int32),
        "target_mask": target_mask,
        "target_delta_t": target_delta_t,
        "target_delta_rotvec": target_delta_rotvec,
        "target_alpha": target_alpha,
        "target_r_human": target_r_human,
        "target_r_scene": target_r_scene,
    }


def main() -> None:
    args = parse_args()
    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(args.output_npz)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    add_path_to_dust3r(str(args.model_path))
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    img_paths, tmpdirname = parse_seq_path(str(args.seq_path))
    if args.max_frames is not None:
        img_paths = img_paths[: int(args.max_frames)]
    img_paths = img_paths[:: int(args.subsample)]
    if not img_paths:
        raise FileNotFoundError(args.seq_path)

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
    model.return_v7_pose_adapter_inputs = True
    if hasattr(model, "enable_v7_pose_adapter") and not args.enable_v7_pose_adapter:
        model.enable_v7_pose_adapter = False
    model.eval()

    img_res = getattr(model, "mhmr_img_res", None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, args.device, use_ttt3r=args.use_ttt3r
        )
    preds = outputs["pred"]
    out_views = outputs["views"]
    frame_ids = np.asarray(
        [_frame_id(view, i) for i, view in enumerate(out_views)], dtype=np.int32
    )

    human_tokens, human_mask = _stack_human_tokens(preds)
    # **========== 原始代码 ==========**
    # arrays = {
    #     "frame_ids": frame_ids,
    #     "pose_tokens": _stack_required(preds, "v7_pose_token_input"),
    #     "scene_tokens": _stack_required(preds, "v7_scene_tokens_input"),
    #     "human_tokens": human_tokens,
    #     "human_token_mask": human_mask,
    #     "memory_tokens": _stack_required(preds, "v7_memory_tokens_input"),
    #     "raw_camera_pose": _stack_required(preds, "v7_raw_camera_pose_input"),
    # }
    # **========== 新代码 ==========**
    scene_tokens = _maybe_pool_tokens(
        _stack_required(preds, "v7_scene_tokens_input"), args.pool_scene_tokens
    )
    memory_tokens = _maybe_pool_tokens(
        _stack_required(preds, "v7_memory_tokens_input"), args.pool_memory_tokens
    )
    arrays = {
        "frame_ids": frame_ids,
        "pose_tokens": _stack_required(preds, "v7_pose_token_input"),
        "scene_tokens": scene_tokens,
        "human_tokens": human_tokens,
        "human_token_mask": human_mask,
        "memory_tokens": memory_tokens,
        "raw_camera_pose": _stack_required(preds, "v7_raw_camera_pose_input"),
    }
    # **========== 结束 ==========**
    if args.pseudo_labels is not None:
        arrays.update(_align_labels(frame_ids, args.pseudo_labels))

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    manifest = {
        "output_npz": str(args.output_npz),
        "model_path": str(args.model_path),
        "seq_path": str(args.seq_path),
        "pseudo_labels": str(args.pseudo_labels) if args.pseudo_labels else None,
        "num_frames": int(len(frame_ids)),
        "frame_start": int(frame_ids.min()) if len(frame_ids) else None,
        "frame_end": int(frame_ids.max()) if len(frame_ids) else None,
        "keys": sorted(arrays.keys()),
        "pool_scene_tokens": bool(args.pool_scene_tokens),
        "pool_memory_tokens": bool(args.pool_memory_tokens),
        "causal_inputs_only": True,
    }
    with open(args.output_npz.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
