#!/usr/bin/env python3
"""Run V8.1 round-1 pose prompt ablations and summarize camera-pose metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOTS = "{'lbn1':'/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1','zxc':'/data/wangzheng/iJCV-CODE/data/avatarrex_zxc','zzr':'/data/wangzheng/iJCV-CODE/data/avatarrex_zzr'}"


ABLATIONS = [
    {"id": "e1_body_only_nogate", "history": False, "pose_memory": False, "gate": False},
    {"id": "e2_body_only_gate", "history": False, "pose_memory": False, "gate": True},
    {"id": "e3_body_history_nogate", "history": True, "pose_memory": False, "gate": False},
    {"id": "e4_body_history_gate", "history": True, "pose_memory": False, "gate": True},
    {"id": "e5_body_posemem_nogate", "history": False, "pose_memory": True, "gate": False},
    {"id": "e6_body_posemem_gate", "history": False, "pose_memory": True, "gate": True},
    {"id": "e7_body_history_posemem_nogate", "history": True, "pose_memory": True, "gate": False},
    {"id": "e8_body_history_posemem_gate", "history": True, "pose_memory": True, "gate": True},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "output" / "v8_1_ablation_round1")
    parser.add_argument("--manifest_root", type=Path, default=REPO_ROOT / "output" / "v8_1_aabb_manifests" / "round1_ablation_small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda_visible_devices", default="7")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def py_bool(value: bool) -> str:
    return "True" if value else "False"


def model_expr(ablation: dict) -> str:
    use_reliability = bool(ablation["gate"])
    return (
        "ARCroco3DStereo(ARCroco3DStereoConfig(freeze='v8_pose_prompt',"
        "state_size=768,state_pe='2d',pos_embed='RoPE100',"
        "rgb_head=True,pose_head=True,msk_head=True,patch_embed_cls='ManyAR_PatchEmbed',img_size=(512,512),"
        "head_type='dpt',output_mode='pts3d+pose+smpl',depth_mode=('exp',-inf,inf),"
        "conf_mode=('exp',1,inf),pose_mode=('exp',-inf,inf),enc_embed_dim=1024,enc_depth=24,"
        "enc_num_heads=16,dec_embed_dim=768,dec_depth=12,dec_num_heads=12,landscape_only=False,"
        "backbone='dinov2_vitl14',mhmr_img_res=896,lora_rank=64,shot_loss_weight=0.0,"
        "shot_q0_loss_weight=0.0,shot_scale_init=0.0,shot_noop_loss_weight=0.0,"
        "pose_delta_t_max=0.5,pose_align_delta_t_max=0.25,pose_align_delta_q_max=0.05,"
        "shot_pointmap_keep_loss_weight=0.0,shot_pose_residual_loss_weight=0.0,"
        "shot_pose_layers='none',layerwise_pose_shot_scale_init=0.0,"
        "v8_pose_prompt_num_body_queries=4,v8_pose_prompt_num_heads=8,"
        "v8_pose_prompt_dropout=0.0,v8_pose_prompt_gate_bias=0.0,"
        f"v8_pose_prompt_use_history={py_bool(ablation['history'])},"
        f"v8_pose_prompt_use_pose_memory={py_bool(ablation['pose_memory'])},"
        f"v8_pose_prompt_use_reliability={py_bool(use_reliability)},"
        f"v8_pose_prompt_use_gate={py_bool(ablation['gate'])}))"
    )


def run_command(cmd: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(line, end="")
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def train_one(ablation: dict, args: argparse.Namespace, env: dict) -> Path:
    run_dir = args.output_root / ablation["id"]
    cmd = [
        sys.executable,
        "src/train.py",
        "--config-name",
        "train_v8_pose_prompt_ablation_small",
        f"model={model_expr(ablation)}",
        f"exp_name={ablation['id']}",
        f"logdir={run_dir / 'logs'}",
        f"output_dir={run_dir}",
    ]
    run_command(cmd, run_dir / "train_stdout.log", env)
    ckpt = run_dir / "checkpoint-final.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    return ckpt


def eval_one(name: str, model_path: Path, manifest: Path, output_json: Path, args: argparse.Namespace, env: dict) -> dict:
    cmd = [
        sys.executable,
        "scripts/v8_1_eval_avatarrex_pose_batch.py",
        "--model_path",
        str(model_path),
        "--manifest_path",
        str(manifest),
        "--name",
        name,
        "--output_json",
        str(output_json),
        "--avatarrex_root",
        "/data/wangzheng/iJCV-CODE/data/Avatarrex_output",
        "--avatarrex_raw_root",
        RAW_ROOTS,
        "--device",
        args.device,
    ]
    run_command(cmd, output_json.with_suffix(".log"), env)
    return json.loads(output_json.read_text(encoding="utf-8"))


def format_float(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_summary(rows: list[dict], output_root: Path) -> None:
    csv_path = output_root / "round1_ablation_summary.csv"
    md_path = output_root / "round1_ablation_summary.md"
    headers = [
        "id",
        "gate",
        "history",
        "pose_memory",
        "pose_head",
        "human_head",
        "contact_history",
        "val_trans",
        "val_rot",
        "test_trans",
        "test_rot",
        "test_trans_improve",
        "test_rot_improve",
    ]
    csv_lines = [",".join(headers)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(h, "")) for h in headers))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md = [
        "# V8.1 Round-1 Pose Prompt Ablation",
        "",
        "All runs use the same small mixed lbn1/zxc/zzr AABB manifests, no DA3 depth, raw calibration camera GT, one epoch, pose head frozen, human head frozen.",
        "",
        "| ID | Gate | History | Pose Memory | Contact History | Val Trans | Val Rot deg | Test Trans | Test Rot deg | Test Trans Improve | Test Rot Improve |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            "| {id} | {gate} | {history} | {pose_memory} | {contact_history} | {val_trans} | {val_rot} | {test_trans} | {test_rot} | {test_trans_improve} | {test_rot_improve} |".format(
                **row
            )
        )
    md.append("")
    md.append("Positive improvement means the ablation is better than the raw Human3R baseline on the same test manifest.")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    env["HYDRA_FULL_ERROR"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    if args.device == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    val_manifest = args.manifest_root / "round1_val.jsonl"
    test_manifest = args.manifest_root / "round1_test.jsonl"
    if not val_manifest.is_file() or not test_manifest.is_file():
        raise FileNotFoundError(f"Missing round1 manifests under {args.manifest_root}")

    eval_dir = args.output_root / "eval"
    baseline_val = baseline_test = None
    if not args.skip_eval:
        baseline_val = eval_one(
            "raw_human3r_val",
            REPO_ROOT / "src" / "human3r_896L.pth",
            val_manifest,
            eval_dir / "raw_human3r_val.json",
            args,
            env,
        )
        baseline_test = eval_one(
            "raw_human3r_test",
            REPO_ROOT / "src" / "human3r_896L.pth",
            test_manifest,
            eval_dir / "raw_human3r_test.json",
            args,
            env,
        )

    rows = []
    for ablation in ABLATIONS:
        run_dir = args.output_root / ablation["id"]
        ckpt = run_dir / "checkpoint-final.pth"
        if not args.skip_train or not ckpt.is_file():
            ckpt = train_one(ablation, args, env)
        if args.skip_eval:
            continue
        val_result = eval_one(ablation["id"] + "_val", ckpt, val_manifest, eval_dir / f"{ablation['id']}_val.json", args, env)
        test_result = eval_one(ablation["id"] + "_test", ckpt, test_manifest, eval_dir / f"{ablation['id']}_test.json", args, env)

        val = val_result["summary"]
        test = test_result["summary"]
        base_test = baseline_test["summary"]
        rows.append(
            {
                "id": ablation["id"],
                "gate": str(ablation["gate"]),
                "history": str(ablation["history"]),
                "pose_memory": str(ablation["pose_memory"]),
                "pose_head": "frozen",
                "human_head": "frozen",
                "contact_history": "False",
                "val_trans": format_float(val["mean_trans_err"]),
                "val_rot": format_float(val["mean_rot_err_deg"]),
                "test_trans": format_float(test["mean_trans_err"]),
                "test_rot": format_float(test["mean_rot_err_deg"]),
                "test_trans_improve": format_float(base_test["mean_trans_err"] - test["mean_trans_err"]),
                "test_rot_improve": format_float(base_test["mean_rot_err_deg"] - test["mean_rot_err_deg"]),
            }
        )
        write_summary(rows, args.output_root)

    if rows:
        write_summary(rows, args.output_root)


if __name__ == "__main__":
    main()
