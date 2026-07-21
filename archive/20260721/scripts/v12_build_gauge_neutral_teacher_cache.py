#!/usr/bin/env python3
"""Build V12 first-write Oracle caches for real and pseudo camera cuts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import AvatarReX_Pattern  # noqa: E402
from v10_causal_state_query_prompt_validation import (  # noqa: E402
    image_paths,
    prepare_views,
    tensor_summary,
)
from v10_latent_activation_patching_probe import PatchSpec, run_branch  # noqa: E402
from v10_token_alignment_4source_probe import raw_roots_for_record, source_split_and_scope  # noqa: E402
from v11_gauge_neutral_first_write_oracle import (  # noqa: E402
    build_loss_targets,
    camera_pose_gpu,
    gauge_neutral_loss,
    optimize_first_write,
)
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    read_jsonl,
    record_spec,
    texture_score,
)


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v12_gated_first_write" / "teacher_cache"
DEFAULT_V11 = (
    REPO_ROOT
    / "output"
    / "v11_gauge_neutral_first_write"
    / "stage2_final_full"
    / "merged"
    / "stage2_merged.json"
)
DEFAULT_PSEUDO = (
    REPO_ROOT / "output" / "v10_latent_token_probe" / "causal_state_transition_inputs" / "clip01",
    REPO_ROOT / "output" / "v10_latent_token_probe" / "causal_state_transition_inputs" / "clip02",
    REPO_ROOT / "output" / "v10_latent_token_probe" / "causal_state_transition_inputs" / "clip03",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "pseudo"), required=True)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--v11_report", type=Path, default=DEFAULT_V11)
    parser.add_argument("--pseudo_dirs", type=Path, nargs="*", default=DEFAULT_PSEUDO)
    parser.add_argument("--pseudo_boundaries", type=int, nargs="*", default=tuple(range(4, 16)))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--point_sample", type=int, default=4096)
    parser.add_argument("--loss_point_sample", type=int, default=2048)
    parser.add_argument("--opt_steps", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.06)
    parser.add_argument("--max_state_residual_std", type=float, default=0.50)
    parser.add_argument("--state_regularization", type=float, default=1e-3)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=24)
    return parser.parse_args()


def load_v11_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case_rows = {}
    csv_path = path.parent / "stage2_cases.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    for shard in sorted(path.parents[1].glob("stage2_shard_*_of_*.json")):
        for case in json.loads(shard.read_text(encoding="utf-8"))["cases"]:
            case_rows[case["case_name"]] = case
    if int(payload["case_count"]) != len(case_rows):
        raise RuntimeError("V11 case map is incomplete")
    return case_rows


def gate_targets(reset: dict, oracle: dict) -> dict:
    metrics = (
        ("camera_relative_translation_m", 0.05, 0.35),
        ("camera_relative_rotation_deg", 1.0, 0.35),
        ("camera_frame_pointmap_m", 0.10, 0.20),
        ("human_relative_root_m", 0.10, 0.10),
    )
    gain = 0.0
    worst_degradation = 0.0
    for key, floor, weight in metrics:
        before = float(reset[key])
        after = float(oracle[key])
        normalized = (before - after) / max(abs(before), floor)
        gain += weight * normalized
        worst_degradation = max(worst_degradation, -normalized)
    score = gain - 2.0 * worst_degradation
    gate = float(np.clip((score - 0.05) / 0.50, 0.0, 1.0))
    difficult = bool(
        float(reset["camera_relative_translation_m"]) > 0.10
        or float(reset["camera_relative_rotation_deg"]) > 2.0
        or float(reset["camera_frame_pointmap_m"]) > 0.20
    )
    return {
        "gate_target": gate,
        "gain_target": float(score),
        "wait_target": 1.0 if difficult and gate > 0.25 else 0.0,
        "oracle_helpful": bool(gate > 0.05),
    }


def old_a_dataset(spec: dict, args: argparse.Namespace) -> AvatarReX_Pattern:
    record = spec["record"]
    split, _ = source_split_and_scope(record)
    seq_a = str(record["seqA"])
    boundary = int(spec["post_frames"][0])
    rgb_dir = args.data_root / split / seq_a / "rgb"
    available = sorted(int(path.stem) for path in rgb_dir.glob("*.png") if int(path.stem) < boundary)
    frames = available[-int(args.warmup_frames) :]
    if not frames:
        raise RuntimeError(f"No A-camera history for {record['pattern_id']}")
    sample = {
        "clip_type": "v12_old_a_history",
        "group": str(record.get("group", "")),
        "seqs": [seq_a] * len(frames),
        "frames": frames,
        "shot_labels": [0] * len(frames),
        "transition_angles_deg": [0.0] * len(frames),
        "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
        "angle_bucket": str(record.get("angle_bucket", "unknown")),
        "pattern_id": f"{record['pattern_id']}_old_a",
    }
    return AvatarReX_Pattern(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=len(frames),
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=[sample],
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(record),
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def cpu_half(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float16)


def flat_camera(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(-1).cpu().to(torch.float16)


def diagnostics(
    old_state: torch.Tensor,
    fresh_state: torch.Tensor,
    image_summary: torch.Tensor,
    human_summary: torch.Tensor,
    memory_summary: torch.Tensor,
    texture: float,
) -> torch.Tensor:
    old = old_state.detach().float()
    fresh = fresh_state.detach().float()
    relative_l2 = torch.linalg.vector_norm(old - fresh) / torch.linalg.vector_norm(fresh).clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(old.reshape(1, -1), fresh.reshape(1, -1)).squeeze()
    values = torch.tensor(
        [
            float(texture),
            float(relative_l2),
            float(old.std(unbiased=False)),
            float(fresh.std(unbiased=False)),
            float(image_summary.float().std(unbiased=False)),
            float(human_summary.float().norm() / math.sqrt(max(human_summary.numel(), 1))),
            float(memory_summary.float().std(unbiased=False)),
            float(cosine),
        ],
        dtype=torch.float16,
    )
    return values


def serialize_targets(targets: dict) -> dict:
    human_valid = [root is not None for root in targets["gt_roots"]]
    roots = [torch.zeros(3, device=targets["gt_poses"][0].device) if root is None else root for root in targets["gt_roots"]]
    rotations = [
        torch.eye(3, device=targets["gt_poses"][0].device).repeat(53, 1, 1) if value is None else value
        for value in targets["gt_rotmats"]
    ]
    scales = {
        key: torch.tensor([float(row[key]) for row in targets["baseline_scales"]], dtype=torch.float32)
        for key in targets["baseline_scales"][0]
    }
    return {
        "gt_poses": torch.stack(targets["gt_poses"]).detach().cpu().float(),
        "point_ids": [value.detach().cpu().to(torch.int32) for value in targets["point_ids"]],
        "teacher_point_samples": [
            targets["teacher_points"][idx][ids].detach().cpu().to(torch.float16)
            for idx, ids in enumerate(targets["point_ids"])
        ],
        "gt_roots": torch.stack(roots).detach().cpu().to(torch.float16),
        "gt_rotmats": torch.stack(rotations).detach().cpu().to(torch.float16),
        "human_valid": torch.tensor(human_valid, dtype=torch.bool),
        "baseline_scales": scales,
    }


def save_pair(
    path: Path,
    metadata: dict,
    old_latents: dict,
    reset_latents: dict,
    best_state: torch.Tensor,
    targets: dict,
    labels: dict,
    texture: float,
) -> dict:
    old_state = old_latents["new_state"][0]
    old_memory = old_latents["pose_memory_after"][0]
    fresh_state = reset_latents["new_state"][0]
    image_summary = tensor_summary(reset_latents["encoder_final"], 2048)
    human_summary = tensor_summary(reset_latents.get("human_prompt"), 1536)
    memory_summary = tensor_summary(old_latents["pose_memory_after"], 3072)
    payload = {
        "metadata": metadata,
        "old_state": cpu_half(old_state),
        "old_pose_memory": cpu_half(old_memory),
        "fresh_state": cpu_half(fresh_state),
        "oracle_state": best_state[0].to(torch.float16),
        "oracle_residual": (best_state[0].float() - fresh_state.float()).to(torch.float16),
        "image_summary": image_summary,
        "human_summary": human_summary,
        "camera_token": flat_camera(reset_latents["camera_initial"]),
        "memory_summary": memory_summary,
        "diagnostics": diagnostics(
            old_state,
            fresh_state,
            image_summary,
            human_summary,
            memory_summary,
            texture,
        ),
        "labels": labels,
        "loss_targets": serialize_targets(targets),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), **metadata, **labels}


def prepare_pseudo_target_views(views: list[dict], teacher: list[dict]) -> None:
    for view, prediction in zip(views, teacher):
        pose = camera_pose_gpu(prediction).detach().float()
        view["raw_camera_pose"] = pose.unsqueeze(0).to(view["img"].device)
        valid = prediction.get("smpl_transl") is not None and prediction["smpl_transl"].shape[1] > 0
        view["smpl_mask"] = torch.tensor([[valid]], device=view["img"].device, dtype=torch.bool)
        root = torch.zeros(3, device=view["img"].device)
        rotations = torch.eye(3, device=view["img"].device).repeat(53, 1, 1)
        if valid:
            root = prediction["smpl_transl"][0, 0].detach().to(view["img"].device).float()
            rotations = prediction["smpl_rotmat"][0, 0].detach().to(view["img"].device).float()
        view["smpl_j3d"] = root.reshape(1, 1, 1, 3)
        view["smpl_rotmat"] = rotations.reshape(1, 1, *rotations.shape)


def pseudo_gate_labels(reset_predictions: list[dict], oracle_predictions: list[dict], targets: dict, args) -> dict:
    with torch.no_grad():
        reset_loss, _ = gauge_neutral_loss(
            [{key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in row.items()} for row in reset_predictions],
            targets,
            args,
        )
        oracle_loss, _ = gauge_neutral_loss(
            [{key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in row.items()} for row in oracle_predictions],
            targets,
            args,
        )
    gain = float((reset_loss - oracle_loss) / reset_loss.clamp_min(1e-8))
    return {
        "gate_target": float(np.clip((gain - 0.05) / 0.50, 0.0, 1.0)),
        "gain_target": gain,
        "wait_target": 1.0 if gain > 0.35 else 0.0,
        "oracle_helpful": bool(gain > 0.05),
    }


def build_real(args: argparse.Namespace, model, gt_model, device: torch.device) -> list[dict]:
    records = read_jsonl(args.records)
    selected = [row for idx, row in enumerate(records) if idx % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    v11 = load_v11_cases(args.v11_report)
    rows = []
    for local_index, record in enumerate(selected):
        spec = record_spec(record, args)
        path = args.output_dir / "pairs" / "real" / f"{record['pattern_id']}.pt"
        if path.is_file() and not args.overwrite:
            rows.append({"path": str(path), "case_name": record["pattern_id"], "source": record["source"], "kind": "real"})
            continue
        reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, model.mhmr_img_res)
        teacher_views = configure_views(one_batch(build_dataset([spec], True, args)), device, model.mhmr_img_res)
        old_views = configure_views(one_batch(old_a_dataset(spec, args)), device, model.mhmr_img_res)
        with torch.no_grad():
            gt_model.update_smpl_gt(reset_views)
        reset_predictions, reset_latents, _, _ = run_branch(model, reset_views, device, 0, capture=True)
        teacher_predictions, _, _, _ = run_branch(
            model, teacher_views, device, spec["warmup_count"], capture=False
        )
        old_predictions, old_latents, _, _ = run_branch(
            model, old_views, device, len(old_views) - 1, capture=True
        )
        del old_predictions
        teacher_post = teacher_predictions[spec["warmup_count"] :]
        targets = build_loss_targets(
            reset_predictions,
            teacher_post,
            reset_views,
            device,
            args.loss_point_sample,
        )
        best_state, optimization = optimize_first_write(
            model, reset_views, reset_latents["new_state"], targets, args
        )
        case = v11[record["pattern_id"]]
        labels = gate_targets(
            case["variants"]["reset_gt_boundary"]["mean_future"],
            case["variants"]["gauge_neutral_oracle_gt_boundary"]["mean_future"],
        )
        labels["oracle_optimization_loss"] = float(optimization["best_loss"])
        metadata = {
            "case_name": record["pattern_id"],
            "kind": "real",
            "source": record["source"],
            "group": record["group"],
            "seqA": record["seqA"],
            "seqB": record["seqB"],
            "start_frame": int(record["start_frame"]),
            "angle_bucket": record["angle_bucket"],
            "view_angle_deg": float(record["view_angle_deg"]),
            "post_frames": spec["post_frames"],
        }
        rows.append(
            save_pair(
                path,
                metadata,
                old_latents,
                reset_latents,
                best_state,
                targets,
                labels,
                texture_score(reset_views[0]),
            )
        )
        print(f">> [{local_index + 1}/{len(selected)}] cached {record['pattern_id']} gate={labels['gate_target']:.3f}", flush=True)
        del reset_views, teacher_views, old_views, reset_predictions, teacher_predictions
        torch.cuda.empty_cache()
    return rows


def build_pseudo(args: argparse.Namespace, model, device: torch.device) -> list[dict]:
    tasks = []
    for input_dir in args.pseudo_dirs:
        paths = image_paths(input_dir, args.max_frames)
        for boundary in args.pseudo_boundaries:
            if boundary > 0 and boundary + args.max_post_frames <= len(paths):
                tasks.append((input_dir, paths, boundary))
    tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        tasks = tasks[: args.max_cases]
    rows = []
    for index, (input_dir, paths, boundary) in enumerate(tasks):
        name = f"{input_dir.name}_b{boundary:03d}"
        path = args.output_dir / "pairs" / "pseudo" / f"{name}.pt"
        if path.is_file() and not args.overwrite:
            rows.append({"path": str(path), "case_name": name, "source": "pseudo", "kind": "pseudo"})
            continue
        views = prepare_views(paths, model, args, device)
        teacher_predictions, teacher_latents, _, _ = run_branch(
            model, views, device, boundary, capture=True
        )
        post_views = views[boundary : boundary + args.max_post_frames]
        reset_predictions, reset_latents, _, _ = run_branch(model, post_views, device, 0, capture=True)
        teacher_post = teacher_predictions[boundary : boundary + args.max_post_frames]
        prepare_pseudo_target_views(post_views, teacher_post)
        targets = build_loss_targets(
            reset_predictions,
            teacher_post,
            post_views,
            device,
            args.loss_point_sample,
        )
        best_state, optimization = optimize_first_write(
            model, post_views, reset_latents["new_state"], targets, args
        )
        oracle_predictions, _, _, _ = run_branch(
            model,
            post_views,
            device,
            0,
            capture=False,
            patch=PatchSpec("v12_pseudo_oracle", ("first_write_state",)),
            source={"new_state": best_state},
        )
        labels = pseudo_gate_labels(reset_predictions, oracle_predictions, targets, args)
        labels["oracle_optimization_loss"] = float(optimization["best_loss"])
        old_latents = dict(teacher_latents)
        old_latents["new_state"] = teacher_latents["persistent_state"]
        old_latents["pose_memory_after"] = teacher_latents["pose_memory_before"]
        metadata = {
            "case_name": name,
            "kind": "pseudo",
            "source": "pseudo",
            "group": input_dir.name,
            "input_dir": str(input_dir),
            "boundary": int(boundary),
            "post_frames": list(range(boundary, boundary + len(post_views))),
        }
        rows.append(
            save_pair(
                path,
                metadata,
                old_latents,
                reset_latents,
                best_state,
                targets,
                labels,
                texture_score(post_views[0]),
            )
        )
        print(f">> [{index + 1}/{len(tasks)}] cached {name} gate={labels['gate_target']:.3f}", flush=True)
        del views, post_views, teacher_predictions, reset_predictions, oracle_predictions
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V12 teacher-cache generation requires CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args)
    gt_model, _ = build_smpl_models(model, device)
    started = time.perf_counter()
    rows = build_real(args, model, gt_model, device) if args.mode == "real" else build_pseudo(args, model, device)
    index_path = args.output_dir / f"index_{args.mode}_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    index_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {index_path} cases={len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
