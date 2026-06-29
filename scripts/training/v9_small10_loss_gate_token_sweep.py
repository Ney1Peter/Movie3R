#!/usr/bin/env python3
"""Run the V9 small10 loss/gate/correct-token sweep.

The script intentionally keeps orchestration simple:
- generated Hydra configs live under config/v9_loss_gate_token_sweep/
- train/eval outputs live under output/v9_loss_gate_token_sweep/
- each worker owns a disjoint experiment subset, selected by modulo index
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = REPO_ROOT / "output/v9_loss_gate_token_sweep"
CONFIG_ROOT = REPO_ROOT / "config/v9_loss_gate_token_sweep"
BASELINE_EVAL = (
    REPO_ROOT
    / "output/v9_small10_ablation/eval_mixed_small18_baseline_best/checkpoint-best/summary.json"
)

RAW_ROOTS = (
    "{'lbn1':'/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1',"
    "'lbn2':'/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn2',"
    "'zzr':'/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr',"
    "'zxc':'/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zxc'}"
)

MODEL_TEMPLATE = """ARCroco3DStereo(ARCroco3DStereoConfig(freeze='v8_pose_prompt_head_lora',
  state_size=768, state_pe='2d', pos_embed='RoPE100',
  rgb_head=True, pose_head=True, msk_head=True, patch_embed_cls='ManyAR_PatchEmbed', img_size=(512,
  512), head_type='dpt', output_mode='pts3d+pose+smpl', depth_mode=('exp', -inf, inf),
  conf_mode=('exp', 1, inf), pose_mode=('exp', -inf, inf), enc_embed_dim=1024, enc_depth=24,
  enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, landscape_only=False,
  backbone='dinov2_vitl14', mhmr_img_res=896, lora_rank=64, shot_loss_weight=0.0,
  shot_q0_loss_weight=0.0, shot_scale_init=0.0, shot_noop_loss_weight=0.0,
  pose_delta_t_max=0.5, pose_align_delta_t_max=0.25, pose_align_delta_q_max=0.05,
  shot_pointmap_keep_loss_weight=0.0, shot_pose_residual_loss_weight=0.0,
  shot_pose_layers='none', layerwise_pose_shot_scale_init=0.0,
  v8_pose_prompt_variant='relation_v8_2',
  v8_pose_prompt_num_heads=8, v8_pose_prompt_dropout=0.0,
  v8_pose_prompt_gate_bias=0.0,
  v8_pose_prompt_use_history=True,
  v8_pose_prompt_use_pose_memory=True,
  v8_pose_prompt_use_reliability=True,
  v8_pose_prompt_use_gate=True,
  v8_pose_prompt_image_only=True,
  v8_pose_prompt_use_human_alignment={use_human_alignment},
  v8_human_trans_corr=False,
  v8_human_trans_corr_gate_bias=0.0,
  v8_human_trans_corr_use_gate=True,
  v8_human_trans_corr_gate_mode='shared',
  v8_human_trans_corr_max_delta=3.0,
  v8_human_trans_corr_apply_from_view=-1,
  v8_human_latent_corr=True,
  v8_human_latent_corr_gate_bias=0.0,
  v8_human_latent_corr_use_gate=True,
  v8_human_latent_corr_gate_mode='{human_gate_mode}',
  v8_human_latent_corr_max_delta=1.0,
  v8_human_latent_corr_apply_from_view=-1,
  v8_pose_head_lora=True,
  v8_human_head_lora=True,
  v8_head_lora_rank=8,
  v8_head_lora_alpha=8.0,
  v8_head_lora_dropout=0.0))"""


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    title: str
    drift_weight: float = 0.05
    improvement_weight: float = 0.05
    improvement_margin: float = 0.0
    drift_target_deadzone: float = 0.0
    drift_target_scale: float = 1.0
    human_trans_weight: float = 10.0
    human_trans_delta_weight: float = 1.0e-5
    use_human_alignment: bool = False
    human_gate_mode: str = "shared"
    extra_notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


EXPERIMENTS: list[Experiment] = [
    Experiment("g1_drift_x2", "Gate: drift weight x2", drift_weight=0.10, tags=("gate",)),
    Experiment("g2_drift_x4", "Gate: drift weight x4", drift_weight=0.20, tags=("gate",)),
    Experiment(
        "g3_deadzone_target",
        "Gate: deadzone/stretch drift target",
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        tags=("gate",),
    ),
    Experiment(
        "g4_deadzone_target_drift_x2",
        "Gate: deadzone/stretch target + drift weight x2",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        tags=("gate",),
    ),
    Experiment("i1_improve_x2", "Improvement: weight x2", improvement_weight=0.10, tags=("improve",)),
    Experiment(
        "i2_margin_005",
        "Improvement: positive margin 0.05",
        improvement_margin=0.05,
        tags=("improve",),
    ),
    Experiment(
        "i3_improve_x2_margin_005",
        "Improvement: weight x2 + margin 0.05",
        improvement_weight=0.10,
        improvement_margin=0.05,
        tags=("improve",),
    ),
    Experiment(
        "i4_improve_x2_margin_010",
        "Improvement: weight x2 + margin 0.10",
        improvement_weight=0.10,
        improvement_margin=0.10,
        tags=("improve",),
    ),
    Experiment(
        "c1_drift_x2_improve_x2",
        "Combined: drift x2 + improvement x2",
        drift_weight=0.10,
        improvement_weight=0.10,
        tags=("combined",),
    ),
    Experiment(
        "c2_deadzone_drift_x2_margin_005",
        "Combined: deadzone target + drift x2 + margin 0.05",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        tags=("combined",),
    ),
    Experiment(
        "c3_deadzone_drift_x2_improve_x2_margin_005",
        "Combined: deadzone target + drift x2 + improvement x2 + margin 0.05",
        drift_weight=0.10,
        improvement_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        tags=("combined",),
    ),
    Experiment("h1_human_x2", "Human: human translation weight x2", human_trans_weight=20.0, tags=("human",)),
    Experiment(
        "h2_c2_human_x2",
        "Human: C2 + human translation weight x2",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        human_trans_weight=20.0,
        tags=("human",),
    ),
    Experiment(
        "h3_c2_human_delta_weak",
        "Human: C2 + weaker human delta regularization",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        human_trans_delta_weight=1.0e-6,
        tags=("human",),
    ),
    Experiment(
        "t1_c2_human_alignment_token",
        "Token: C2 + human-alignment correction token",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        use_human_alignment=True,
        tags=("token",),
    ),
    Experiment(
        "t2_c2_independent_human_gate",
        "Token/head: C2 + independent human gate",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        human_gate_mode="independent",
        tags=("token",),
    ),
    Experiment(
        "t3_c2_product_human_gate",
        "Token/head: C2 + product human gate",
        drift_weight=0.10,
        drift_target_deadzone=0.05,
        drift_target_scale=0.45,
        improvement_margin=0.05,
        human_gate_mode="product",
        tags=("token",),
    ),
]


def criterion_string(exp: Experiment) -> str:
    return (
        "V82PoseRelationLoss("
        "translation_weight=1.0, rotation_weight=5.0, "
        "residual_weight=1.0e-5, "
        f"drift_weight={exp.drift_weight}, "
        f"improvement_weight={exp.improvement_weight}, "
        "pose_key='raw_camera_pose', "
        "drift_trans_scale=0.5, drift_rot_scale_deg=45.0, "
        f"drift_target_deadzone={exp.drift_target_deadzone}, "
        f"drift_target_scale={exp.drift_target_scale}, "
        f"improvement_margin={exp.improvement_margin}, "
        f"human_trans_weight={exp.human_trans_weight}, "
        f"human_trans_delta_weight={exp.human_trans_delta_weight}, "
        "pose_lora_norm_weight=0.0, human_lora_norm_weight=0.0)"
    )


def eval_loss_args(exp: Experiment) -> list[str]:
    return [
        "--translation_weight",
        "1.0",
        "--rotation_weight",
        "5.0",
        "--residual_weight",
        "1.0e-5",
        "--drift_weight",
        str(exp.drift_weight),
        "--improvement_weight",
        str(exp.improvement_weight),
        "--improvement_margin",
        str(exp.improvement_margin),
        "--drift_target_deadzone",
        str(exp.drift_target_deadzone),
        "--drift_target_scale",
        str(exp.drift_target_scale),
        "--human_trans_weight",
        str(exp.human_trans_weight),
        "--human_trans_delta_weight",
        str(exp.human_trans_delta_weight),
    ]


def model_string(exp: Experiment) -> str:
    return MODEL_TEMPLATE.format(
        use_human_alignment="True" if exp.use_human_alignment else "False",
        human_gate_mode=exp.human_gate_mode,
    )


def config_text(exp: Experiment) -> str:
    exp_root = SWEEP_ROOT / "train" / exp.exp_id
    lines = [
        "# @package _global_",
        f"# Auto-generated V9 small10 sweep config: {exp.exp_id}",
        f"# {exp.title}",
        "defaults:",
        "  - /train_v9_small10_pose_human_lora_baseline",
        "  - _self_",
        "",
        f"exp_name: v9_sweep_{exp.exp_id}",
        f"logdir: {exp_root}/logs",
        f"output_dir: {exp_root}",
        "",
        "epochs: 100",
        "eval_freq: 10",
        "early_stopping_patience: 30",
        "save_freq: 10",
        "keep_freq: 0",
        "save_last_checkpoint: true",
        "save_final_checkpoint: false",
        "print_freq: 1",
        "structured_log_freq: 1",
        "print_img_freq: 1000000",
        "save_code: false",
        "cuda_cache_reserve_mb: 0",
        "",
        f"train_criterion: {criterion_string(exp)}",
        f"test_criterion: {criterion_string(exp)}",
    ]
    if exp.use_human_alignment or exp.human_gate_mode != "shared":
        lines.extend(["", "model: " + model_string(exp)])
    return "\n".join(lines) + "\n"


def ensure_dirs() -> None:
    for path in [
        SWEEP_ROOT,
        SWEEP_ROOT / "configs",
        SWEEP_ROOT / "train",
        SWEEP_ROOT / "eval",
        SWEEP_ROOT / "logs",
        SWEEP_ROOT / "summary",
        CONFIG_ROOT,
        REPO_ROOT / "output/tmp/mpl",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_configs() -> None:
    ensure_dirs()
    for exp in EXPERIMENTS:
        text = config_text(exp)
        (CONFIG_ROOT / f"{exp.exp_id}.yaml").write_text(text, encoding="utf-8")
        (SWEEP_ROOT / "configs" / f"{exp.exp_id}.yaml").write_text(text, encoding="utf-8")
    manifest = [
        {
            "index": idx,
            "exp_id": exp.exp_id,
            "title": exp.title,
            "tags": list(exp.tags),
            "params": {
                "drift_weight": exp.drift_weight,
                "improvement_weight": exp.improvement_weight,
                "improvement_margin": exp.improvement_margin,
                "drift_target_deadzone": exp.drift_target_deadzone,
                "drift_target_scale": exp.drift_target_scale,
                "human_trans_weight": exp.human_trans_weight,
                "human_trans_delta_weight": exp.human_trans_delta_weight,
                "use_human_alignment": exp.use_human_alignment,
                "human_gate_mode": exp.human_gate_mode,
            },
        }
        for idx, exp in enumerate(EXPERIMENTS)
    ]
    (SWEEP_ROOT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_command(cmd: list[str], log_file: Path, env: dict[str, str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write("\n$ " + " ".join(cmd) + "\n")
        fp.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def checkpoint_for_eval(exp: Experiment) -> Path:
    out_dir = SWEEP_ROOT / "train" / exp.exp_id
    best = out_dir / "checkpoint-best.pth"
    last = out_dir / "checkpoint-last.pth"
    final = out_dir / "checkpoint-final.pth"
    if best.is_file():
        return best
    if last.is_file():
        return last
    if final.is_file():
        return final
    raise FileNotFoundError(f"no checkpoint found for {exp.exp_id} in {out_dir}")


def eval_summary_path(exp: Experiment) -> Path:
    ckpt_name = checkpoint_for_eval(exp).stem
    return SWEEP_ROOT / "eval" / exp.exp_id / ckpt_name / "summary.json"


def run_one(exp: Experiment, gpu: str, force: bool = False) -> None:
    ensure_dirs()
    write_configs()
    done_path = SWEEP_ROOT / "eval" / exp.exp_id / "DONE.json"
    if done_path.is_file() and not force:
        print(f"[skip] {exp.exp_id}: DONE exists", flush=True)
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / "output/tmp/mpl")
    env["PYTHONPATH"] = "src:." + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    log_file = SWEEP_ROOT / "logs" / f"{exp.exp_id}.log"
    status_path = SWEEP_ROOT / "eval" / exp.exp_id / "STATUS.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({"exp_id": exp.exp_id, "status": "running_train", "gpu": gpu}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[train] {exp.exp_id} on GPU {gpu}", flush=True)
    run_command(
        [
            ".venv/bin/python",
            "src/train.py",
            "--config-name",
            f"v9_loss_gate_token_sweep/{exp.exp_id}",
        ],
        log_file,
        env,
    )

    ckpt = checkpoint_for_eval(exp)
    status_path.write_text(
        json.dumps(
            {"exp_id": exp.exp_id, "status": "running_eval", "gpu": gpu, "checkpoint": str(ckpt)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[eval] {exp.exp_id} using {ckpt.name}", flush=True)
    run_command(
        [
            ".venv/bin/python",
            "scripts/v8_4_eval_pose_benchmark.py",
            "--model_path",
            str(ckpt),
            "--benchmark_dir",
            str(SWEEP_ROOT.parent / "v9_small10_ablation/benchmark_mixed_small18"),
            "--output_dir",
            str(SWEEP_ROOT / "eval" / exp.exp_id),
            "--subsets",
            "test_aabb,test_aaaa",
            "--test_split",
            "Training",
            "--raw_roots",
            RAW_ROOTS,
            "--resize_mode",
            "resize_only_16",
            "--batch_size",
            "1",
            "--num_workers",
            "0",
            *eval_loss_args(exp),
            "--dump_poses",
        ],
        log_file,
        env,
    )

    summary = eval_summary_path(exp)
    done_path.write_text(
        json.dumps(
            {
                "exp_id": exp.exp_id,
                "status": "done",
                "gpu": gpu,
                "checkpoint": str(ckpt),
                "summary": str(summary),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status_path.unlink(missing_ok=True)
    print(f"[done] {exp.exp_id}", flush=True)


def run_worker(worker_id: int, num_workers: int, gpu: str, force: bool = False) -> None:
    ensure_dirs()
    write_configs()
    assigned = [exp for idx, exp in enumerate(EXPERIMENTS) if idx % num_workers == worker_id]
    print(
        f"worker {worker_id}/{num_workers} on GPU {gpu}: {[exp.exp_id for exp in assigned]}",
        flush=True,
    )
    for exp in assigned:
        run_one(exp, gpu=gpu, force=force)
    collect()


def flatten_metrics(exp_id: str, summary_path: Path, title: str = "") -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row: dict[str, object] = {"exp_id": exp_id, "title": title, "summary": str(summary_path)}
    for prefix, data in [
        ("overall", summary.get("overall", {})),
        ("aabb", summary.get("subsets", {}).get("test_aabb", {})),
        ("aaaa", summary.get("subsets", {}).get("test_aaaa", {})),
    ]:
        for key in [
            "count",
            "v82_raw_trans_err_mean",
            "v82_trans_err_mean",
            "v82_trans_improvement_mean",
            "v82_raw_rot_err_deg_mean",
            "v82_rot_err_deg_mean",
            "v82_rot_improvement_deg_mean",
            "v82_raw_human_trans_err_mean",
            "v82_human_trans_err_mean",
            "v82_gate_mean_mean",
            "v82_drift_target_mean_mean",
            "v82_metric_mpjpe_mm_mean",
            "v82_mpjpe_mm_mean",
            "v82_pa_mpjpe_mm_mean",
            "loss_mean",
        ]:
            if key in data:
                row[f"{prefix}_{key}"] = data[key]
    return row


def collect() -> None:
    ensure_dirs()
    rows: list[dict[str, object]] = []
    if BASELINE_EVAL.is_file():
        rows.append(flatten_metrics("b0_current_baseline", BASELINE_EVAL, "Current baseline"))
    title_by_id = {exp.exp_id: exp.title for exp in EXPERIMENTS}
    for exp in EXPERIMENTS:
        done = SWEEP_ROOT / "eval" / exp.exp_id / "DONE.json"
        if not done.is_file():
            continue
        done_data = json.loads(done.read_text(encoding="utf-8"))
        summary_path = Path(done_data["summary"])
        if summary_path.is_file():
            rows.append(flatten_metrics(exp.exp_id, summary_path, title_by_id.get(exp.exp_id, "")))
    if not rows:
        return

    columns = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "exp_id",
        "title",
        "overall_v82_trans_err_mean",
        "overall_v82_trans_improvement_mean",
        "overall_v82_human_trans_err_mean",
        "overall_v82_gate_mean_mean",
        "aabb_v82_trans_err_mean",
        "aabb_v82_trans_improvement_mean",
        "aabb_v82_human_trans_err_mean",
        "aabb_v82_gate_mean_mean",
        "aaaa_v82_trans_err_mean",
        "aaaa_v82_human_trans_err_mean",
        "aaaa_v82_gate_mean_mean",
        "aabb_v82_pa_mpjpe_mm_mean",
        "summary",
    ]
    columns = [c for c in preferred if c in columns] + [c for c in columns if c not in preferred]
    csv_path = SWEEP_ROOT / "summary" / "sweep_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = SWEEP_ROOT / "summary" / "sweep_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = SWEEP_ROOT / "summary" / "sweep_metrics.md"
    md_cols = [
        "exp_id",
        "aabb_v82_trans_err_mean",
        "aabb_v82_trans_improvement_mean",
        "aabb_v82_human_trans_err_mean",
        "aabb_v82_gate_mean_mean",
        "aaaa_v82_trans_err_mean",
        "aaaa_v82_human_trans_err_mean",
        "aaaa_v82_gate_mean_mean",
    ]
    lines = ["# V9 Small10 Sweep Metrics", ""]
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(md_cols)) + "|")
    for row in rows:
        values = []
        for col in md_cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}", flush=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")

    one = sub.add_parser("run-one")
    one.add_argument("exp_id")
    one.add_argument("--gpu", required=True)
    one.add_argument("--force", action="store_true")

    worker = sub.add_parser("worker")
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--num-workers", type=int, required=True)
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--force", action="store_true")

    sub.add_parser("collect")
    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        write_configs()
        print(f"prepared {len(EXPERIMENTS)} configs under {CONFIG_ROOT}", flush=True)
        return 0
    if args.cmd == "run-one":
        exp_by_id = {exp.exp_id: exp for exp in EXPERIMENTS}
        if args.exp_id not in exp_by_id:
            raise SystemExit(f"unknown exp_id: {args.exp_id}")
        run_one(exp_by_id[args.exp_id], gpu=args.gpu, force=args.force)
        return 0
    if args.cmd == "worker":
        run_worker(args.worker_id, args.num_workers, gpu=args.gpu, force=args.force)
        return 0
    if args.cmd == "collect":
        collect()
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
