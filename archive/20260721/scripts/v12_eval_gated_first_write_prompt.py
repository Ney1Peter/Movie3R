#!/usr/bin/env python3
"""Evaluate V12 gated first-write adapters on unseen real camera cuts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.v12_gated_gauge_neutral_prompt import GatedGaugeNeutralFirstWritePrompt  # noqa: E402
from v10_latent_activation_patching_probe import PatchSpec, run_branch  # noqa: E402
from v11_gauge_neutral_first_write_oracle import fixed_explicit_transform, max_boundary_difference  # noqa: E402
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    evaluate_case,
    record_spec,
)
from v12_gated_first_write_runtime import GatedFirstWriteController  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "output" / "v12_gated_first_write" / "teacher_cache_loso_mvhuman200"
DEFAULT_TRAINING = REPO_ROOT / "output" / "v12_gated_first_write" / "training_loso_mvhuman200"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v12_gated_first_write" / "eval_loso_mvhuman200"
DEFAULT_CANDIDATES = REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--training_dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate_root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--point_sample", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def record_from_pair(pair: dict) -> dict:
    metadata = pair["metadata"]
    return {
        "angle_bucket": metadata["angle_bucket"],
        "clip_type": "aabb",
        "group": metadata["group"],
        "seqA": metadata["seqA"],
        "seqB": metadata["seqB"],
        "start_frame": int(metadata["start_frame"]),
        "view_angle_deg": float(metadata["view_angle_deg"]),
        "source": metadata["source"],
        "pattern_id": metadata["case_name"],
    }


def load_adapter(path: Path, hidden_dim: int, device: torch.device):
    model = GatedGaugeNeutralFirstWritePrompt(hidden_dim=hidden_dim).to(device).eval()
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model


def run_prompt(human3r, adapter, views, device, pair, variant, source_mode="correct", donor=None, seed=0):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with GatedFirstWriteController(
        human3r,
        adapter,
        pair,
        variant,
        source_mode=source_mode,
        donor_pair=donor,
        seed=seed,
    ) as controller:
        with torch.no_grad():
            predictions, _ = human3r.forward_recurrent_lighter(
                views,
                str(device),
                ret_state=False,
                use_ttt3r=False,
                return_token_debug=False,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return predictions, controller.output, elapsed, peak_memory_mb


def run_plain(human3r, views, device, target_frame, **kwargs):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    predictions, latents, _, skipped = run_branch(
        human3r,
        views,
        device,
        target_frame,
        **kwargs,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return predictions, latents, elapsed, peak_memory_mb, skipped


def eval_variant(spec, predictions, teacher, views, warmup, gt_model, pred_layer, args, index, transform=None):
    return evaluate_case(
        spec,
        predictions,
        teacher,
        views,
        warmup,
        gt_model,
        pred_layer,
        args,
        index,
        world_transform=transform,
        update_smpl_gt=False,
    )


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V12 evaluation requires CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.cache_dir / "index_test.json").read_text(encoding="utf-8"))
    selected = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    output = args.output_dir / f"eval_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if output.is_file() and not args.overwrite:
        print(f">> exists {output}")
        return
    device = torch.device(args.device)
    human3r = build_model(args)
    gt_model, pred_layer = build_smpl_models(human3r, device)
    gated = load_adapter(args.training_dir / "checkpoints" / "gated_rollout.pth", args.hidden_dim, device)
    ungated = load_adapter(args.training_dir / "checkpoints" / "ungated_rollout.pth", args.hidden_dim, device)
    no_old = load_adapter(args.training_dir / "checkpoints" / "no_old_rollout.pth", args.hidden_dim, device)
    all_pairs = [torch.load(row["path"], map_location="cpu", weights_only=False) for row in rows]
    donor_by_name = {
        pair["metadata"]["case_name"]: all_pairs[(idx + 1) % len(all_pairs)]
        for idx, pair in enumerate(all_pairs)
    }
    cases = []
    started = time.perf_counter()
    for index, row in enumerate(selected):
        pair = torch.load(row["path"], map_location="cpu", weights_only=False)
        record = record_from_pair(pair)
        spec = record_spec(record, args)
        reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, human3r.mhmr_img_res)
        teacher_views = configure_views(one_batch(build_dataset([spec], True, args)), device, human3r.mhmr_img_res)
        with torch.no_grad():
            gt_model.update_smpl_gt(reset_views)
        reset_predictions, _, reset_seconds, reset_peak, _ = run_plain(
            human3r, reset_views, device, 0, capture=False
        )
        teacher_predictions, _, teacher_seconds, teacher_peak, _ = run_plain(
            human3r, teacher_views, device, spec["warmup_count"], capture=False
        )
        oracle_predictions, _, oracle_seconds, oracle_peak, _ = run_plain(
            human3r,
            reset_views,
            device,
            0,
            capture=False,
            patch=PatchSpec("v12_oracle", ("first_write_state",)),
            source={"new_state": pair["oracle_state"].unsqueeze(0)},
        )
        gated_predictions, gated_output, gated_seconds, gated_peak = run_prompt(
            human3r, gated, reset_views, device, pair, "gated"
        )
        ungated_predictions, ungated_output, ungated_seconds, ungated_peak = run_prompt(
            human3r, ungated, reset_views, device, pair, "ungated"
        )
        no_old_predictions, no_old_output, no_old_seconds, no_old_peak = run_prompt(
            human3r, no_old, reset_views, device, pair, "no_old"
        )
        zero_predictions, zero_output, zero_seconds, zero_peak = run_prompt(
            human3r, gated, reset_views, device, pair, "gated", "zero"
        )
        shuffle_predictions, shuffle_output, shuffle_seconds, shuffle_peak = run_prompt(
            human3r, gated, reset_views, device, pair, "gated", "shuffle", seed=args.seed + index
        )
        wrong_predictions, wrong_output, wrong_seconds, wrong_peak = run_prompt(
            human3r,
            gated,
            reset_views,
            device,
            pair,
            "gated",
            "wrong",
            donor=donor_by_name[pair["metadata"]["case_name"]],
            seed=args.seed + index,
        )
        explicit, explicit_name = fixed_explicit_transform(args.candidate_root, record["pattern_id"])
        variants = {
            "hard_reset": eval_variant(spec, reset_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "oracle": eval_variant(spec, oracle_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "ungated": eval_variant(spec, ungated_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "gated": eval_variant(spec, gated_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "no_old": eval_variant(spec, no_old_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "zero_old": eval_variant(spec, zero_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "shuffle_old": eval_variant(spec, shuffle_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "wrong_old": eval_variant(spec, wrong_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index),
            "explicit_only": eval_variant(spec, reset_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index, explicit),
            "gated_explicit": eval_variant(spec, gated_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index, explicit),
            "oracle_explicit": eval_variant(spec, oracle_predictions, teacher_predictions, reset_views, spec["warmup_count"], gt_model, pred_layer, args, index, explicit),
        }
        variants["boundary_output_only"] = copy.deepcopy(variants["hard_reset"])
        variants["gate_only"] = copy.deepcopy(variants["hard_reset"])
        boundary = {
            "oracle": max_boundary_difference(reset_predictions[0], oracle_predictions[0]),
            "gated": max_boundary_difference(reset_predictions[0], gated_predictions[0]),
            "ungated": max_boundary_difference(reset_predictions[0], ungated_predictions[0]),
            "no_old": max_boundary_difference(reset_predictions[0], no_old_predictions[0]),
        }
        cases.append(
            {
                "case_name": record["pattern_id"],
                "record": record,
                "labels": pair["labels"],
                "gate_predictions": {
                    "gated": float(gated_output.gate.item()),
                    "predicted_gain": float(gated_output.predicted_gain.item()),
                    "wait_score": float(gated_output.wait_score.item()),
                    "ungated_internal_gate": float(ungated_output.gate_logit.sigmoid().item()),
                    "no_old": float(no_old_output.gate.item()),
                    "zero_old": float(zero_output.gate.item()),
                    "shuffle_old": float(shuffle_output.gate.item()),
                    "wrong_old": float(wrong_output.gate.item()),
                },
                "boundary_lock": boundary,
                "explicit": {"name": explicit_name, "transform": explicit.tolist()},
                "timing_seconds": {
                    "reset": reset_seconds,
                    "teacher": teacher_seconds,
                    "oracle": oracle_seconds,
                    "gated": gated_seconds,
                    "ungated": ungated_seconds,
                    "no_old": no_old_seconds,
                    "zero_old": zero_seconds,
                    "shuffle_old": shuffle_seconds,
                    "wrong_old": wrong_seconds,
                },
                "peak_memory_mb": {
                    "reset": reset_peak,
                    "teacher": teacher_peak,
                    "oracle": oracle_peak,
                    "gated": gated_peak,
                    "ungated": ungated_peak,
                    "no_old": no_old_peak,
                    "zero_old": zero_peak,
                    "shuffle_old": shuffle_peak,
                    "wrong_old": wrong_peak,
                },
                "variants": variants,
            }
        )
        before = variants["hard_reset"]["mean_future"]
        after = variants["gated"]["mean_future"]
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} gate={gated_output.gate.item():.3f} "
            f"R={before['camera_relative_rotation_deg']:.3f}->{after['camera_relative_rotation_deg']:.3f}",
            flush=True,
        )
        del reset_views, teacher_views, reset_predictions, teacher_predictions, oracle_predictions
        del gated_predictions, ungated_predictions, no_old_predictions, zero_predictions, shuffle_predictions, wrong_predictions
        torch.cuda.empty_cache()
    report = {
        "experiment": "V12 Learned Gated Gauge-Neutral First-Write Prompt",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "test_source": "mvhuman200",
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
