#!/usr/bin/env python3
"""Train and evaluate a minimal causal read-old/write-fresh Shot Prompt.

This is an experiment-only path.  Human3R remains frozen and its default
inference code is not changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_causal_state_transition_oracle_probe import (  # noqa: E402
    image_paths,
    offset_recovery,
    prepare_views,
)
from v10_latent_activation_patching_probe import (  # noqa: E402
    PatchSpec,
    add_recovery,
    build_model,
    evaluate_branch,
    finite_mean,
    run_branch,
    safe_name,
    source_dict,
)
from dust3r.utils.camera import camera_to_pose_encoding, pose_encoding_to_camera  # noqa: E402
from dust3r.v10_causal_state_query_prompt import CausalStateQueryFirstWritePrompt  # noqa: E402


DEFAULT_INPUT_ROOT = REPO_ROOT / "output" / "v10_latent_token_probe" / "causal_state_transition_inputs"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_latent_token_probe" / "causal_state_query_prompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dirs",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT_ROOT / "clip01", DEFAULT_INPUT_ROOT / "clip02", DEFAULT_INPUT_ROOT / "clip03"],
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--boundaries", type=int, nargs="+", default=(6, 9, 12, 15))
    parser.add_argument("--train_clips", nargs="+", default=("clip01", "clip02"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--point_sample", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--overwrite_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def tensor_summary(value: torch.Tensor | None, dim: int) -> torch.Tensor:
    if value is None or value.numel() == 0:
        return torch.zeros(dim, dtype=torch.float16)
    value = value.detach().float()
    pooled = torch.cat([value.mean(dim=1), value.std(dim=1, unbiased=False)], dim=-1)
    pooled = pooled.reshape(-1).cpu().to(torch.float16)
    if pooled.numel() != dim:
        raise ValueError(f"Expected summary dim {dim}, got {pooled.numel()}")
    return pooled


def flat_camera(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(-1).cpu().to(torch.float16)


def save_pair(
    path: Path,
    clip: str,
    boundary: int,
    teacher_latents: dict,
    reset_latents: dict,
    early_latents: dict,
) -> dict:
    pair = {
        "clip": clip,
        "boundary": int(boundary),
        "old_state": teacher_latents["persistent_state"][0].clone(),
        "old_pose_memory": teacher_latents["pose_memory_before"][0].clone(),
        "target_state": teacher_latents["new_state"][0].clone(),
        "raw_fresh_state": reset_latents["new_state"][0].clone(),
        "early_fresh_state": early_latents["new_state"][0].clone(),
        "image_summary": tensor_summary(reset_latents["encoder_final"], 2048),
        "human_summary": tensor_summary(reset_latents.get("human_prompt"), 1536),
        "raw_camera": flat_camera(reset_latents["camera_initial"]),
        "early_camera": flat_camera(teacher_latents["camera_initial"]),
        "memory_summary": tensor_summary(teacher_latents["pose_memory_before"], 3072),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pair, path)
    return {"path": str(path), "clip": clip, "boundary": int(boundary)}


def build_cache(model, args: argparse.Namespace, device: torch.device) -> list[dict]:
    cache_dir = args.output_dir / "state_pair_cache"
    index_path = cache_dir / "index.json"
    if index_path.is_file() and not args.overwrite_cache:
        return json.loads(index_path.read_text(encoding="utf-8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip_index, input_dir in enumerate(args.input_dirs):
        paths = image_paths(input_dir, int(args.max_frames))
        views = prepare_views(paths, model, args, device)
        for boundary in args.boundaries:
            if boundary <= 0 or boundary + 8 >= len(views):
                continue
            teacher_predictions, teacher_latents, _seconds, _skipped = run_branch(
                model,
                views,
                device,
                int(boundary),
                capture=True,
                seed=args.seed + clip_index * 100 + int(boundary),
            )
            del teacher_predictions
            post_first = views[int(boundary) : int(boundary) + 1]
            reset_predictions, reset_latents, _seconds, _skipped = run_branch(
                model,
                post_first,
                device,
                0,
                capture=True,
                seed=args.seed + clip_index * 100 + int(boundary),
            )
            del reset_predictions
            teacher_source = source_dict(teacher_latents)
            early_predictions, early_latents, _seconds, _skipped = run_branch(
                model,
                post_first,
                device,
                0,
                capture=True,
                patch=PatchSpec("read_old_pose_memory", ("read_old_pose_memory",)),
                source=teacher_source,
                seed=args.seed + clip_index * 100 + int(boundary),
            )
            del early_predictions
            clip = safe_name(input_dir.name)
            pair_path = cache_dir / f"{clip}_b{int(boundary):03d}.pt"
            rows.append(
                save_pair(
                    pair_path,
                    clip,
                    int(boundary),
                    teacher_latents,
                    reset_latents,
                    early_latents,
                )
            )
        del views
        torch.cuda.empty_cache()
    index_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


class StatePairDataset(Dataset):
    def __init__(self, rows: list[dict], fresh_key: str, camera_key: str):
        self.rows = rows
        self.fresh_key = fresh_key
        self.camera_key = camera_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        pair = torch.load(self.rows[index]["path"], map_location="cpu", weights_only=False)
        return {
            "old_state": pair["old_state"],
            "fresh_state": pair[self.fresh_key],
            "target_state": pair["target_state"],
            "image_summary": pair["image_summary"],
            "human_summary": pair["human_summary"],
            "camera_token": pair[self.camera_key],
            "memory_summary": pair["memory_summary"],
        }


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


def latent_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    baseline_sq = []
    predicted_sq = []
    gates = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(
                batch["old_state"],
                batch["fresh_state"],
                batch["image_summary"],
                batch["human_summary"],
                batch["camera_token"],
                batch["memory_summary"],
            )
            baseline_sq.append((batch["fresh_state"] - batch["target_state"]).square().mean(dim=(1, 2)))
            predicted_sq.append((output.corrected_state - batch["target_state"]).square().mean(dim=(1, 2)))
            gates.append(output.state_gate.mean(dim=(1, 2)))
    baseline = torch.cat(baseline_sq).mean().sqrt().item()
    predicted = torch.cat(predicted_sq).mean().sqrt().item()
    return {
        "baseline_rmse": baseline,
        "predicted_rmse": predicted,
        "recovery": 1.0 - predicted / max(baseline, 1e-8),
        "mean_gate": torch.cat(gates).mean().item(),
    }


def train_one_adapter(
    name: str,
    rows: list[dict],
    fresh_key: str,
    camera_key: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CausalStateQueryFirstWritePrompt, dict]:
    checkpoint = args.output_dir / "checkpoints" / f"{name}.pth"
    train_rows = [row for row in rows if row["clip"] in set(args.train_clips)]
    val_rows = [row for row in rows if row["clip"] not in set(args.train_clips)]
    if not val_rows:
        val_rows = train_rows
    train_loader = DataLoader(
        StatePairDataset(train_rows, fresh_key, camera_key),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
    )
    train_eval_loader = DataLoader(StatePairDataset(train_rows, fresh_key, camera_key), batch_size=1)
    val_loader = DataLoader(StatePairDataset(val_rows, fresh_key, camera_key), batch_size=1)
    model = CausalStateQueryFirstWritePrompt(hidden_dim=int(args.hidden_dim)).to(device)
    if checkpoint.is_file() and not args.overwrite_train:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        return model, payload["report"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    iterator = iter(train_loader)
    history = []
    model.train()
    for step in range(1, int(args.steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["old_state"],
            batch["fresh_state"],
            batch["image_summary"],
            batch["human_summary"],
            batch["camera_token"],
            batch["memory_summary"],
        )
        baseline_mse = (batch["fresh_state"] - batch["target_state"]).square().mean().detach()
        state_mse = (output.corrected_state - batch["target_state"]).square().mean()
        difficulty_target = torch.log1p(
            (batch["fresh_state"] - batch["target_state"]).square().mean(dim=(1, 2)).sqrt()
        )
        difficulty_loss = nn.functional.mse_loss(output.predicted_difficulty, difficulty_target)
        gate_loss = output.state_gate.mean()
        loss = state_mse / baseline_mse.clamp_min(1e-8) + 0.05 * difficulty_loss + 1e-4 * gate_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(args.steps):
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "relative_state_mse": float((state_mse / baseline_mse.clamp_min(1e-8)).detach()),
                "gate": float(output.state_gate.mean().detach()),
            }
            history.append(row)
            print(f">> {name} {row}", flush=True)

    report = {
        "name": name,
        "train_cases": len(train_rows),
        "val_cases": len(val_rows),
        "fresh_key": fresh_key,
        "camera_key": camera_key,
        "train": latent_metrics(model, train_eval_loader, device),
        "validation": latent_metrics(model, val_loader, device),
        "history": history,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report}, checkpoint)
    return model, report


class LearnedFirstWriteController(AbstractContextManager):
    def __init__(
        self,
        model,
        adapter: CausalStateQueryFirstWritePrompt,
        old_state: torch.Tensor,
        old_pose_memory: torch.Tensor,
        use_early_query: bool,
        source_mode: str = "correct",
        seed: int = 0,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.old_state = old_state
        self.old_pose_memory = old_pose_memory
        self.use_early_query = bool(use_early_query)
        self.source_mode = source_mode
        self.seed = int(seed)
        self.original_rollout = None
        self.rollout_index = -1

    def _source_tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.old_state.to(device=device, dtype=dtype).unsqueeze(0)
        memory = self.old_pose_memory.to(device=device, dtype=dtype).unsqueeze(0)
        if self.source_mode == "zero":
            return torch.zeros_like(state), torch.zeros_like(memory)
        if self.source_mode == "shuffle":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            state_order = torch.randperm(state.shape[1], generator=generator).to(device)
            memory_order = torch.randperm(memory.shape[1], generator=generator).to(device)
            return state[:, state_order], memory[:, memory_order]
        return state, memory

    def __enter__(self):
        self.original_rollout = self.model._recurrent_rollout

        def rollout_wrapper(_model, *args, **kwargs):
            self.rollout_index += 1
            args = list(args)
            if self.rollout_index != 0:
                return self.original_rollout(*args, **kwargs)
            old_state, old_memory = self._source_tensors(args[0].device, args[0].dtype)
            if self.use_early_query:
                image_query = self.model._get_img_level_feat(args[2])
                args[4] = self.model.pose_retriever.inquire(image_query, old_memory)
            result = self.original_rollout(*args, **kwargs)
            image_summary = torch.cat(
                [args[2].mean(dim=1), args[2].std(dim=1, unbiased=False)], dim=-1
            ).float()
            if args[6] is None or args[6].numel() == 0:
                human_summary = image_summary.new_zeros(image_summary.shape[0], 1536)
            else:
                human_summary = torch.cat(
                    [args[6].mean(dim=1), args[6].std(dim=1, unbiased=False)], dim=-1
                ).float()
            memory_summary = torch.cat(
                [old_memory.mean(dim=1), old_memory.std(dim=1, unbiased=False)], dim=-1
            ).float()
            with torch.no_grad():
                output = self.adapter(
                    old_state.float(),
                    result[0].float(),
                    image_summary,
                    human_summary,
                    args[4].reshape(args[4].shape[0], -1).float(),
                    memory_summary,
                )
            corrected = output.corrected_state.to(dtype=result[0].dtype)
            return corrected, result[1], result[2]

        self.model._recurrent_rollout = MethodType(rollout_wrapper, self.model)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.model._recurrent_rollout = self.original_rollout
        return False


def run_learned_branch(
    model,
    adapter,
    views,
    device: torch.device,
    pair: dict,
    use_early_query: bool,
    source_mode: str = "correct",
    seed: int = 0,
) -> list[dict]:
    with LearnedFirstWriteController(
        model,
        adapter,
        pair["old_state"],
        pair["old_pose_memory"],
        use_early_query,
        source_mode,
        seed,
    ):
        with torch.no_grad():
            predictions, _views = model.forward_recurrent_lighter(
                views,
                str(device),
                ret_state=False,
                use_ttt3r=False,
                return_token_debug=False,
            )
    return predictions


def boundary_camera_align(predictions: list[dict], teacher: list[dict]) -> list[dict]:
    pred_boundary = pose_encoding_to_camera(predictions[0]["camera_pose"].detach().float())
    teacher_boundary = pose_encoding_to_camera(teacher[0]["camera_pose"].detach().float())
    transform = teacher_boundary @ torch.linalg.inv(pred_boundary)
    aligned = []
    for prediction in predictions:
        row = dict(prediction)
        camera = pose_encoding_to_camera(prediction["camera_pose"].detach().float())
        row["camera_pose"] = camera_to_pose_encoding(transform @ camera).to(
            device=prediction["camera_pose"].device,
            dtype=prediction["camera_pose"].dtype,
        )
        aligned.append(row)
    return aligned


def generic_aggregate(cases: list[dict]) -> dict:
    if not cases:
        return {}
    names = sorted(set.intersection(*(set(case["variants"]) for case in cases)))
    result = {}
    for name in names:
        mean_keys = cases[0]["variants"][name]["metrics"]["mean"].keys()
        recovery_keys = cases[0]["variants"][name]["recovery"].keys()
        result[name] = {
            "mean_error": {
                key: finite_mean([case["variants"][name]["metrics"]["mean"][key] for case in cases])
                for key in mean_keys
            },
            "mean_recovery": {
                key: finite_mean([case["variants"][name]["recovery"][key] for case in cases])
                for key in recovery_keys
            },
            "offset_recovery": {},
        }
        offsets = sorted(cases[0]["variants"][name]["offset_recovery"], key=int)
        for offset in offsets:
            result[name]["offset_recovery"][offset] = {
                key: finite_mean(
                    [case["variants"][name]["offset_recovery"][offset][key] for case in cases]
                )
                for key in cases[0]["variants"][name]["offset_recovery"][offset]
            }
    return result


def evaluate_adapters(
    human3r,
    raw_adapter,
    early_adapter,
    rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    row_by_key = {(row["clip"], int(row["boundary"])): row for row in rows}
    cases = []
    for clip_index, input_dir in enumerate(args.input_dirs):
        paths = image_paths(input_dir, int(args.max_frames))
        views = prepare_views(paths, human3r, args, device)
        clip = safe_name(input_dir.name)
        for boundary in args.boundaries:
            key = (clip, int(boundary))
            if key not in row_by_key or boundary + 8 >= len(views):
                continue
            pair = torch.load(row_by_key[key]["path"], map_location="cpu", weights_only=False)
            teacher_predictions, teacher_latents, _seconds, _skipped = run_branch(
                human3r, views, device, int(boundary), capture=True, seed=args.seed + clip_index
            )
            teacher_post = teacher_predictions[int(boundary) :]
            post_views = views[int(boundary) :]
            reset_predictions, _latents, _seconds, _skipped = run_branch(
                human3r, post_views, device, 0, capture=False, seed=args.seed + clip_index
            )
            raw_learned = run_learned_branch(
                human3r, raw_adapter, post_views, device, pair, False, seed=args.seed + clip_index
            )
            early_learned = run_learned_branch(
                human3r, early_adapter, post_views, device, pair, True, seed=args.seed + clip_index
            )
            early_zero = run_learned_branch(
                human3r, early_adapter, post_views, device, pair, True, "zero", args.seed + clip_index
            )
            early_shuffle = run_learned_branch(
                human3r, early_adapter, post_views, device, pair, True, "shuffle", args.seed + clip_index
            )
            teacher_source = source_dict(teacher_latents)
            oracle_predictions, _latents, _seconds, _skipped = run_branch(
                human3r,
                post_views,
                device,
                0,
                capture=False,
                patch=PatchSpec("oracle_first_write", ("first_write_state",)),
                source=teacher_source,
                seed=args.seed + clip_index,
            )
            variants = {
                "A_continuous_teacher": teacher_post,
                "B_reset_baseline": reset_predictions,
                "C_boundary_output_oracle": boundary_camera_align(reset_predictions, teacher_post),
                "D_raw_state_query_prompt": raw_learned,
                "E_early_state_query_prompt": early_learned,
                "E_early_prompt_zero_old_state": early_zero,
                "E_early_prompt_shuffled_old_state": early_shuffle,
                "F_early_prompt_plus_output_oracle": boundary_camera_align(early_learned, teacher_post),
                "G_first_write_state_oracle": oracle_predictions,
                "G_first_write_plus_output_oracle": boundary_camera_align(oracle_predictions, teacher_post),
            }
            metric_rows = {
                name: {
                    "metrics": evaluate_branch(predictions, teacher_post, args, args.seed + clip_index),
                    "patch": [],
                }
                for name, predictions in variants.items()
            }
            metric_rows["reset_raw"] = metric_rows["B_reset_baseline"]
            add_recovery(metric_rows)
            metric_rows.pop("reset_raw")
            offset_recovery(metric_rows, (0, 1, 2, 4, 8))
            cases.append(
                {
                    "case_name": f"{clip}_b{int(boundary):03d}",
                    "clip": clip,
                    "boundary": int(boundary),
                    "split": "train" if clip in set(args.train_clips) else "validation",
                    "variants": metric_rows,
                }
            )
        del views
        torch.cuda.empty_cache()
    return {
        "overall": generic_aggregate(cases),
        "train": generic_aggregate([case for case in cases if case["split"] == "train"]),
        "validation": generic_aggregate([case for case in cases if case["split"] == "validation"]),
        "cases": cases,
    }


def write_summary_csv(path: Path, overall: dict) -> None:
    rows = []
    for name, values in overall.items():
        row = {"variant": name}
        row.update({f"error_{key}": value for key, value in values["mean_error"].items()})
        row.update({f"recovery_{key}": value for key, value in values["mean_recovery"].items()})
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda"):
        raise ValueError("This experiment must run on CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    human3r = build_model(args)
    rows = build_cache(human3r, args, device)

    raw_adapter, raw_report = train_one_adapter(
        "raw_first_write",
        rows,
        "raw_fresh_state",
        "raw_camera",
        args,
        device,
    )
    early_adapter, early_report = train_one_adapter(
        "early_query_first_write",
        rows,
        "early_fresh_state",
        "early_camera",
        args,
        device,
    )
    report = {
        "experiment": "Causal State-query Shot Prompt Validation",
        "constraints": {
            "human3r_frozen": True,
            "old_state_read_only": True,
            "fresh_state_only_committed": True,
            "gpu": str(device),
        },
        "cache_cases": len(rows),
        "training": {"raw": raw_report, "early": early_report},
    }
    if not args.skip_eval:
        report["rollout"] = evaluate_adapters(
            human3r,
            raw_adapter.eval(),
            early_adapter.eval(),
            rows,
            args,
            device,
        )
        write_summary_csv(args.output_dir / "rollout_summary.csv", report["rollout"]["overall"])
    (args.output_dir / "causal_state_query_prompt_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
