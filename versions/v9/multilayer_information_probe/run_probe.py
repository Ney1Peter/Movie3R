#!/usr/bin/env python3
"""Isolated multi-layer information probe for the V9 correction architecture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


DEFAULT_MODEL = REPO_ROOT / "src/human3r_896L.pth"
DEFAULT_OUTPUT = REPO_ROOT / "output/v9_multilayer_information_probe"
TEN_ROOT = REPO_ROOT / "config/manifests/v14_1_cut_event/ten"
SINGLE_MANIFEST = REPO_ROOT / "config/manifests/v14_1_cut_event/single/lbn1_1192.jsonl"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")

TRAIN_MANIFESTS = {
    "avatarrex": REPO_ROOT
    / "config/manifests/v9_4source_baseline_avatarrex_lbn1_lbn2_zzr_angle60_manifests/train_aabb_60k.jsonl",
    "thuman": REPO_ROOT
    / "config/manifests/v9_4source_baseline_thuman00_02_angle60_manifests/train_aabb_60k.jsonl",
    "mvhuman100": REPO_ROOT
    / "config/manifests/v9_4source_baseline_mvhuman100_angle60_manifests/train_aabb_60k.jsonl",
    "mvhuman200": REPO_ROOT
    / "config/manifests/v9_4source_baseline_mvhuman200_angle60_manifests/train_aabb_60k.jsonl",
}

SOURCE_SPLITS = {
    "avatarrex": "Training",
    "thuman": "Training",
    "mvhuman100": "Training/mvhuman",
    "mvhuman200": "Training/mvhuman",
}

ENCODER_LAYERS = (5, 11, 17, 23)
DINO_LAYERS = (5, 11, 17, 23)
DECODER_LAYERS = (2, 5, 8, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--train-per-source", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--descriptor-tokens", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--mlp-steps", type=int, default=400)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def raw_calibration_roots() -> dict[str, str]:
    root = DATA_ROOT / "AvatarReX_raw_meta"
    return {name: str(root / name) for name in ("lbn1", "lbn2", "zzr", "zxc")}


def source_for_record(record: dict[str, Any], fallback: str | None = None) -> str:
    if fallback is not None:
        return fallback
    group = str(record.get("group", ""))
    if group.startswith("thuman"):
        return "thuman"
    if group.startswith("100"):
        return "mvhuman100"
    if group.startswith("200"):
        return "mvhuman200"
    return "avatarrex"


def aabb_to_pattern(record: dict[str, Any], source: str) -> dict[str, Any]:
    frame = int(record["start_frame"])
    seq_a = str(record["seqA"])
    seq_b = str(record["seqB"])
    return {
        "clip_type": "cut_event",
        "group": str(record.get("group", "")),
        "seqs": [seq_a, seq_a, seq_b],
        "frames": [frame - 1, frame, frame],
        "shot_labels": [0, 0, 1],
        "transition_angles_deg": [0.0, 0.0, float(record.get("view_angle_deg", 0.0))],
        "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
        "angle_bucket": str(record.get("angle_bucket", "unknown")),
        "pattern_id": f"probe_train_{source}_{record.get('group', '')}_{seq_a.split('/')[-1]}_{seq_b.split('/')[-1]}_{frame}",
    }


def record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["seqs"][-2]),
        str(record["seqs"][-1]),
        int(record["frames"][-1]),
    )


def collect_records(train_per_source: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_records: list[dict[str, Any]] = []
    for source, filename in (
        ("avatarrex", "avatarrex.jsonl"),
        ("thuman", "thuman.jsonl"),
        ("mvhuman100", "mvhuman100.jsonl"),
        ("mvhuman200", "mvhuman200.jsonl"),
    ):
        for record in read_jsonl(TEN_ROOT / filename):
            item = dict(record)
            item["source"] = source
            item["split"] = "eval10"
            eval_records.append(item)

    excluded_pairs = {
        (str(record["seqs"][-2]), str(record["seqs"][-1]))
        for record in eval_records
    }
    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    for source, manifest in TRAIN_MANIFESTS.items():
        candidates = read_jsonl(manifest)
        rng.shuffle(candidates)
        accepted = 0
        for candidate in candidates:
            pair = (str(candidate["seqA"]), str(candidate["seqB"]))
            if pair in excluded_pairs:
                continue
            pattern = aabb_to_pattern(candidate, source)
            pattern["source"] = source
            pattern["split"] = "train"
            train_records.append(pattern)
            accepted += 1
            if accepted >= train_per_source:
                break
        if accepted < train_per_source:
            raise RuntimeError(f"Only selected {accepted}/{train_per_source} records for {source}")
    return train_records, eval_records


def make_dataset(records: list[dict[str, Any]], source: str, seed: int):
    from dust3r.datasets.avatarrex import AvatarReX_Pattern

    return AvatarReX_Pattern(
        allow_repeat=True,
        split=SOURCE_SPLITS[source],
        ROOT=str(DATA_ROOT),
        aug_crop=0,
        resolution=512,
        resize_mode="human3r_demo",
        num_views=3,
        seed=seed,
        n_corres=0,
        fixed_samples=records,
        load_da3_depth=False,
        raw_calibration_root=raw_calibration_roots() if source == "avatarrex" else None,
        max_humans=1,
    )


def prepare_batch(dataset, record_index: int) -> tuple[list[dict], list[dict]]:
    from dust3r.inference import _make_v8_image_only_model_batch
    from dust3r.utils.geometry import resize_camera_intrinsics
    from dust3r.utils.image import pad_image

    loader = DataLoader(
        torch.utils.data.Subset(dataset, [record_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    gt_batch = next(iter(loader))
    images = torch.stack([view["img"] for view in gt_batch], dim=0)
    images = images.view(-1, *images.shape[2:])
    intrinsics = torch.stack([view["camera_intrinsics"] for view in gt_batch], dim=0)
    intrinsics = intrinsics.view(-1, *intrinsics.shape[2:])
    intrinsics_mhmr = resize_camera_intrinsics(intrinsics, *images.shape[2:], 896)
    images_mhmr = pad_image(images, 896)
    for view, image_mhmr, intrinsic_mhmr in zip(
        gt_batch,
        images_mhmr.chunk(len(gt_batch), dim=0),
        intrinsics_mhmr.chunk(len(gt_batch), dim=0),
    ):
        view["img_mhmr"] = image_mhmr
        view["K_mhmr"] = intrinsic_mhmr
    model_batch = _make_v8_image_only_model_batch(copy.deepcopy(gt_batch))
    for view in model_batch:
        reference = view["img_mask"]
        view["reset"] = torch.zeros_like(reference, dtype=torch.bool)
        for key in ("update", "update_state", "update_mem", "update_v8_history"):
            view[key] = torch.ones_like(reference, dtype=torch.bool)
        view["shot_label"] = torch.zeros_like(reference, dtype=torch.float32)
    return gt_batch, model_batch


def tensor_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


@dataclass
class Capture:
    values: dict[str, list[torch.Tensor]]
    handles: list[Any]

    @classmethod
    def attach(cls, model) -> "Capture":
        values: dict[str, list[torch.Tensor]] = defaultdict(list)
        handles = []

        def hook(name: str):
            def save(_module, _inputs, output):
                value = tensor_from_output(output).detach().float().cpu()
                values[name].append(value)

            return save

        for index in ENCODER_LAYERS:
            handles.append(model.enc_blocks[index].register_forward_hook(hook(f"cut3r_l{index:02d}")))
        for index in DINO_LAYERS:
            handles.append(
                model.backbone.encoder.blocks[index].register_forward_hook(hook(f"dino_l{index:02d}"))
            )
        for index in DECODER_LAYERS:
            handles.append(model.dec_blocks[index].register_forward_hook(hook(f"decoder_l{index:02d}")))
        return cls(values=values, handles=handles)

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def descriptor_for_pair(pre: torch.Tensor, post: torch.Tensor, max_tokens: int) -> torch.Tensor:
    pre = pre[0] if pre.ndim == 3 else pre
    post = post[0] if post.ndim == 3 else post
    count = min(max_tokens, pre.shape[0], post.shape[0])

    def pool(tokens: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(tokens.T.unsqueeze(0), count).squeeze(0).T
        return F.normalize(pooled, dim=-1, eps=1e-6)

    pre_pool = pool(pre)
    post_pool = pool(post)
    similarity = post_pool @ pre_pool.T
    row_values, row_indices = similarity.topk(k=min(2, count), dim=1)
    col_values, col_indices = similarity.topk(k=min(2, count), dim=0)
    nearest_pre = row_indices[:, 0]
    nearest_post = col_indices[0]
    post_ids = torch.arange(count)
    mutual = (nearest_post[nearest_pre] == post_ids).float()

    stats = []
    for tensor in (pre, post, post - pre[: post.shape[0]] if pre.shape == post.shape else None):
        if tensor is None:
            continue
        stats.extend([tensor.mean(dim=0), tensor.std(dim=0, unbiased=False)])
    summary = torch.cat(stats)
    summary_size = 256
    if summary.numel() > summary_size:
        summary = F.adaptive_avg_pool1d(summary.reshape(1, 1, -1), summary_size).reshape(-1)
    else:
        summary = F.pad(summary, (0, summary_size - summary.numel()))

    relation_stats = torch.stack(
        [
            similarity.mean(),
            similarity.std(unbiased=False),
            row_values[:, 0].mean(),
            row_values[:, 0].std(unbiased=False),
            row_values[:, 0].amin(),
            row_values[:, 0].amax(),
            (row_values[:, 0] - row_values[:, -1]).mean(),
            col_values[0].mean(),
            col_values[0].std(unbiased=False),
            mutual.mean(),
        ]
    )
    return torch.cat([summary, similarity.flatten(), relation_stats]).float()


def camera_matrix_from_prediction(prediction: dict) -> np.ndarray:
    from dust3r.utils.camera import pose_encoding_to_camera

    return (
        pose_encoding_to_camera(prediction["camera_pose"].detach().float())[0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def gt_camera(view: dict) -> np.ndarray:
    value = view.get("raw_camera_pose", view["camera_pose"])
    return value.detach().float()[0].cpu().numpy().astype(np.float32)


def relative_pose(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    return np.linalg.inv(pre) @ post


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, :2].T.reshape(-1).astype(np.float32)


def rotation_from_6d(value: np.ndarray) -> np.ndarray:
    vectors = np.asarray(value, dtype=np.float64).reshape(2, 3)
    first = vectors[0] / max(np.linalg.norm(vectors[0]), 1e-8)
    second = vectors[1] - np.dot(first, vectors[1]) * first
    second = second / max(np.linalg.norm(second), 1e-8)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1).astype(np.float32)


def target_vector(relative: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation_to_6d(relative[:3, :3]), relative[:3, 3]]).astype(np.float32)


def vector_to_pose(value: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation_from_6d(value[:6])
    pose[:3, 3] = value[6:9]
    return pose


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    relative = estimated[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_error(estimated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    translation = float(np.linalg.norm(estimated[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(estimated, target)
    return {
        "translation_m": translation,
        "rotation_deg": rotation,
        "composite": translation + 0.02 * rotation,
    }


def configure_raw_model(model) -> None:
    for name, value in (
        ("enable_shot_adaptation", False),
        ("enable_v8_pose_prompt", False),
        ("enable_v8_human_latent_corr", False),
        ("enable_v8_human_trans_corr", False),
        ("enable_v8_head_lora", False),
        ("v9_raw_pose_step_gate_enabled", False),
        ("v9_clean_raw_pose_step_gate_enabled", False),
        ("v9_oracle_correction_gate_enabled", False),
    ):
        if hasattr(model, name):
            setattr(model, name, value)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def select_frame_pair(values: list[torch.Tensor], expected_frames: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if len(values) != expected_frames:
        raise RuntimeError(f"{name} captured {len(values)} calls, expected {expected_frames}")
    return values[-2], values[-1]


def extract_records(args: argparse.Namespace, records: list[dict[str, Any]], model, capture: Capture) -> list[dict]:
    datasets: dict[str, Any] = {}
    dataset_indices: dict[tuple[str, tuple[str, str, int]], int] = {}
    source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_records[source_for_record(record, record.get("source"))].append(record)
    for source, items in source_records.items():
        dataset = make_dataset(items, source, args.seed)
        datasets[source] = dataset
        for local_index, sample in enumerate(dataset.samples):
            dataset_indices[(source, record_key(sample))] = local_index

    outputs = []
    valid_records = []
    for record in records:
        source = source_for_record(record, record.get("source"))
        lookup_key = (source, record_key(record))
        if lookup_key not in dataset_indices:
            print(f"skip incomplete record: {record.get('pattern_id', lookup_key)}", flush=True)
            continue
        valid_records.append((record, source, dataset_indices[lookup_key]))

    for global_index, (record, source, local_index) in enumerate(valid_records):
        gt_batch, model_batch = prepare_batch(datasets[source], local_index)
        capture.clear()
        started = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            predictions, _ = model.forward_recurrent_lighter(
                model_batch,
                args.device,
                ret_state=False,
                use_ttt3r=False,
            )
        elapsed = time.perf_counter() - started
        descriptors = {}
        for name, values in sorted(capture.values.items()):
            pre, post = select_frame_pair(values, len(model_batch), name)
            descriptors[name] = descriptor_for_pair(pre, post, args.descriptor_tokens)

        gt_relative = relative_pose(gt_camera(gt_batch[-2]), gt_camera(gt_batch[-1]))
        raw_relative = relative_pose(
            camera_matrix_from_prediction(predictions[-2]),
            camera_matrix_from_prediction(predictions[-1]),
        )
        row = {
            "pattern_id": str(record.get("pattern_id", f"record_{global_index}")),
            "source": source,
            "split": str(record.get("split", "unknown")),
            "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
            "seqs": list(record["seqs"]),
            "frames": list(map(int, record["frames"])),
            "descriptors": descriptors,
            "target": torch.from_numpy(target_vector(gt_relative)),
            "gt_relative": torch.from_numpy(gt_relative),
            "raw_relative": torch.from_numpy(raw_relative),
            "raw_error": pose_error(raw_relative, gt_relative),
            "elapsed_s": elapsed,
        }
        outputs.append(row)
        print(
            f"[{global_index + 1:03d}/{len(valid_records):03d}] {row['pattern_id']} "
            f"{source} {elapsed:.2f}s raw={row['raw_error']['composite']:.3f}",
            flush=True,
        )
    return outputs


def stack_features(
    rows: list[dict],
    names: tuple[str, ...],
    include_raw_pose: bool = False,
) -> np.ndarray:
    features = []
    for row in rows:
        parts = [row["descriptors"][name] for name in names]
        if include_raw_pose:
            parts.append(torch.from_numpy(target_vector(row["raw_relative"].numpy())))
        features.append(torch.cat(parts).numpy())
    return np.stack(features, axis=0).astype(np.float64)


def stack_targets(rows: list[dict], target_mode: str) -> np.ndarray:
    targets = []
    for row in rows:
        if target_mode == "absolute":
            pose = row["gt_relative"].numpy()
        elif target_mode == "residual":
            pose = row["gt_relative"].numpy() @ np.linalg.inv(row["raw_relative"].numpy())
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
        targets.append(target_vector(pose))
    return np.stack(targets, axis=0).astype(np.float64)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_scaler = Standardizer.fit(train_x)
    y_scaler = Standardizer.fit(train_y)
    x = x_scaler.transform(train_x)
    y = y_scaler.transform(train_y)
    xt = x_scaler.transform(test_x)
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    xt = np.concatenate([xt, np.ones((xt.shape[0], 1))], axis=1)
    gram = x @ x.T
    dual = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), y)
    prediction = xt @ x.T @ dual
    return y_scaler.inverse(prediction)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = min(128, max(64, input_dim // 8))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def mlp_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    steps: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    x_scaler = Standardizer.fit(train_x)
    y_scaler = Standardizer.fit(train_y)
    probe_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_scaler.transform(train_x)).float().to(probe_device)
    y = torch.from_numpy(y_scaler.transform(train_y)).float().to(probe_device)
    xt = torch.from_numpy(x_scaler.transform(test_x)).float().to(probe_device)
    model = MLPProbe(x.shape[1]).to(probe_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(steps):
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_prediction = model(x).cpu().numpy()
        test_prediction = model(xt).cpu().numpy()
    del model, x, y, xt
    return y_scaler.inverse(train_prediction), y_scaler.inverse(test_prediction)


def summarize_predictions(
    rows: list[dict],
    predictions: np.ndarray,
    target_mode: str = "absolute",
) -> dict[str, Any]:
    per_case = []
    for row, prediction in zip(rows, predictions):
        estimated = vector_to_pose(prediction)
        if target_mode == "residual":
            estimated = estimated @ row["raw_relative"].numpy()
        target = row["gt_relative"].numpy()
        error = pose_error(estimated, target)
        per_case.append({"pattern_id": row["pattern_id"], "source": row["source"], **error})
    summary = {}
    for key in ("translation_m", "rotation_deg", "composite"):
        values = np.asarray([item[key] for item in per_case], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
        }
    summary["per_case"] = per_case
    return summary


def feature_groups(rows: list[dict]) -> dict[str, tuple[str, ...]]:
    available = tuple(sorted(rows[0]["descriptors"]))
    groups = {name: (name,) for name in available}
    groups["cut3r_multi"] = tuple(name for name in available if name.startswith("cut3r_l"))
    groups["dino_multi"] = tuple(name for name in available if name.startswith("dino_l"))
    groups["decoder_multi"] = tuple(name for name in available if name.startswith("decoder_l"))
    groups["cut3r_dino_multi"] = groups["cut3r_multi"] + groups["dino_multi"]
    groups["cut3r_decoder_multi"] = groups["cut3r_multi"] + groups["decoder_multi"]
    groups["all_multi"] = groups["cut3r_multi"] + groups["dino_multi"] + groups["decoder_multi"]
    return groups


def evaluate_task(
    args: argparse.Namespace,
    rows: list[dict],
    train_rows: list[dict],
    eval_rows: list[dict],
    single_rows: list[dict],
    target_mode: str,
) -> dict[str, Any]:
    groups = feature_groups(rows)
    include_raw_pose = target_mode == "residual"
    if include_raw_pose:
        groups = {"raw_pose_only": (), **groups}
    report: dict[str, Any] = {"target_mode": target_mode, "groups": {}}

    train_y = stack_targets(train_rows, target_mode)
    eval_y = stack_targets(eval_rows, target_mode)
    for group_name, names in groups.items():
        train_x = stack_features(train_rows, names, include_raw_pose)
        eval_x = stack_features(eval_rows, names, include_raw_pose)
        ridge_eval = ridge_fit_predict(train_x, train_y, eval_x, args.ridge)
        mlp_train, mlp_eval = mlp_fit_predict(
            train_x,
            train_y,
            eval_x,
            args.mlp_steps,
            args.seed + int(hashlib.sha1(group_name.encode()).hexdigest()[:6], 16),
            args.device,
        )
        ten_overfit_train, ten_overfit_eval = mlp_fit_predict(
            eval_x,
            eval_y,
            eval_x,
            args.mlp_steps,
            args.seed + 1000 + int(hashlib.sha1(group_name.encode()).hexdigest()[:6], 16),
            args.device,
        )
        group_report = {
            "features": (["raw_relative_pose"] if include_raw_pose else []) + list(names),
            "dimension": int(train_x.shape[1]),
            "ridge_heldout": summarize_predictions(eval_rows, ridge_eval, target_mode),
            "mlp_train_fit": summarize_predictions(train_rows, mlp_train, target_mode),
            "mlp_heldout": summarize_predictions(eval_rows, mlp_eval, target_mode),
            "ten_case_overfit": summarize_predictions(eval_rows, ten_overfit_eval, target_mode),
        }
        if single_rows:
            single_x = stack_features(single_rows, names, include_raw_pose)
            single_y = stack_targets(single_rows, target_mode)
            _, single_prediction = mlp_fit_predict(
                single_x,
                single_y,
                single_x,
                max(500, args.mlp_steps // 2),
                args.seed + 2000,
                args.device,
            )
            group_report["single_case_overfit"] = summarize_predictions(
                single_rows, single_prediction, target_mode
            )
        report["groups"][group_name] = group_report
        metric = group_report["mlp_heldout"]["composite"]["mean"]
        print(f"fit {target_mode:8s} {group_name:24s} heldout composite={metric:.4f}", flush=True)

    ranking = sorted(
        (
            {
                "group": name,
                "mlp_heldout_composite": values["mlp_heldout"]["composite"]["mean"],
                "ridge_heldout_composite": values["ridge_heldout"]["composite"]["mean"],
                "ten_overfit_composite": values["ten_case_overfit"]["composite"]["mean"],
            }
            for name, values in report["groups"].items()
        ),
        key=lambda item: item["mlp_heldout_composite"],
    )
    report["ranking"] = ranking
    return report


def evaluate(args: argparse.Namespace, rows: list[dict]) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval10"]
    single_rows = [row for row in eval_rows if "lbn1_1192" in row["pattern_id"]]
    return {
        "protocol": {
            "train_cases": len(train_rows),
            "eval_cases": len(eval_rows),
            "single_case": single_rows[0]["pattern_id"] if single_rows else None,
            "encoder_layers": ENCODER_LAYERS,
            "dino_layers": DINO_LAYERS,
            "decoder_layers": DECODER_LAYERS,
            "descriptor_tokens": args.descriptor_tokens,
        },
        "raw_human3r": summarize_predictions(
            eval_rows,
            np.stack([target_vector(row["raw_relative"].numpy()) for row in eval_rows]),
        ),
        "tasks": {
            target_mode: evaluate_task(
                args, rows, train_rows, eval_rows, single_rows, target_mode
            )
            for target_mode in ("absolute", "residual")
        },
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_markdown(report: dict[str, Any], path: Path) -> None:
    raw = report["raw_human3r"]["composite"]["mean"]
    lines = [
        "# V9 Multi-Layer Information Probe",
        "",
        "## Protocol",
        "",
        f"- Train cases: {report['protocol']['train_cases']}",
        f"- Frozen evaluation cuts: {report['protocol']['eval_cases']}",
        f"- Single overfit: `{report['protocol']['single_case']}`",
        f"- Raw Human3R relative-camera composite: `{raw:.4f}`",
    ]
    for target_mode, task in report["tasks"].items():
        lines.extend(
            [
                "",
                f"## {target_mode.title()} Target Ranking",
                "",
                "| Feature group | Held-out MLP | Held-out ridge | 10-cut overfit |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in task["ranking"]:
            lines.append(
                f"| {row['group']} | {row['mlp_heldout_composite']:.4f} | "
                f"{row['ridge_heldout_composite']:.4f} | {row['ten_overfit_composite']:.4f} |"
            )
        best = task["ranking"][0]
        lines.extend(
            [
                "",
                f"Best `{target_mode}` feature group: `{best['group']}` with "
                f"MLP composite `{best['mlp_heldout_composite']:.4f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This direct head is an information-capacity diagnostic. A positive result "
            "does not by itself establish the final V9 decoder-in evidence-token design.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "descriptor_cache.pt"
    metadata_path = args.output_dir / "protocol.json"
    train_records, eval_records = collect_records(args.train_per_source, args.seed)
    records = train_records + eval_records
    metadata_path.write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "train_records": train_records,
                "eval_records": eval_records,
                "args": vars(args),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.fit_only and (args.overwrite_cache or not cache_path.is_file()):
        from dust3r.model import ARCroco3DStereo

        device = torch.device(args.device)
        model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).float()
        configure_raw_model(model)
        capture = Capture.attach(model)
        try:
            rows = extract_records(args, records, model, capture)
        finally:
            capture.close()
        torch.save(rows, cache_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        rows = torch.load(cache_path, map_location="cpu", weights_only=False)

    if args.extract_only:
        return
    report = evaluate(args, rows)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(json_ready(report), indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, args.output_dir / "report.md")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
