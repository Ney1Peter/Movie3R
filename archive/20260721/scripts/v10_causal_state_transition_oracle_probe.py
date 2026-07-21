#!/usr/bin/env python3
"""Stage-0/1 oracle validation for a causal read-old/write-fresh Shot Prompt."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from demo import prepare_input  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
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


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_latent_token_probe" / "causal_state_transition_oracle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:5" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--boundary", type=int, default=12)
    parser.add_argument("--point_sample", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--decoder_layers", type=int, nargs="*", default=None)
    return parser.parse_args()


def image_paths(input_dir: Path, max_frames: int) -> list[str]:
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(input_dir)
    return [str(path) for path in paths]


def prepare_views(paths: list[str], model, args: argparse.Namespace, device: torch.device) -> list[dict]:
    views = prepare_input(
        paths,
        [True] * len(paths),
        int(args.size),
        revisit=1,
        update=True,
        img_res=model.mhmr_img_res,
        reset_interval=1000000,
    )
    return todevice(views, device)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-8))


def stage0_audit(model, teacher_latents: dict, reset_latents: dict) -> dict:
    shared = model.pose_token.detach().cpu().to(torch.float16)
    teacher_initial = teacher_latents["camera_initial"]
    reset_initial = reset_latents["camera_initial"]
    pose_query_output = teacher_latents.get("pose_query_output")
    return {
        "persistent_state_time": "teacher S_{t-1}, captured before current-frame decoder and before first state write",
        "teacher_new_state_time": "teacher S_t, captured after current-frame decoder; oracle-only for current-frame intervention",
        "camera_initial_location": "pose_feat_i immediately before _recurrent_rollout / decoder layer 0",
        "camera_initial_source_teacher": "pose_retriever.inquire(current global image token, pre-update pose memory)",
        "camera_initial_source_reset": "shared learnable model.pose_token because reset frame is local index 0",
        "camera_initial_before_state_write": True,
        "reset_initial_vs_shared_relative_l2": relative_l2(reset_initial, shared),
        "teacher_initial_vs_shared_relative_l2": relative_l2(teacher_initial, shared),
        "teacher_initial_vs_pose_query_output_relative_l2": None
        if pose_query_output is None
        else relative_l2(teacher_initial, pose_query_output),
        "pre_vs_post_state_relative_l2": relative_l2(
            teacher_latents["persistent_state"], teacher_latents["new_state"]
        ),
        "pose_memory_before_available": "pose_memory_before" in teacher_latents,
        "state_shape": list(teacher_latents["persistent_state"].shape),
        "pose_memory_shape": None
        if "pose_memory_before" not in teacher_latents
        else list(teacher_latents["pose_memory_before"].shape),
    }


def variant_specs() -> list[PatchSpec]:
    specs = [
        PatchSpec("C_late_camera_refined", ("camera_refined",)),
        PatchSpec(
            "C_late_camera_refined_plus_first_write",
            ("camera_refined", "first_write_state"),
        ),
        PatchSpec("D_early_camera_initial", ("camera_initial",)),
        PatchSpec("E_first_write_state", ("first_write_state",)),
        PatchSpec("F_initial_camera_plus_first_write", ("camera_initial", "first_write_state")),
        PatchSpec("G_read_old_pose_memory_write_fresh", ("read_old_pose_memory",)),
        PatchSpec("G_random_pose_memory_write_fresh", ("read_old_pose_memory",), "random"),
        PatchSpec("G_shuffled_pose_memory_write_fresh", ("read_old_pose_memory",), "shuffle"),
        PatchSpec("legacy_pre_state_as_active", ("persistent_state",)),
        PatchSpec(
            "legacy_pre_state_as_active_plus_camera_initial",
            ("persistent_state", "camera_initial"),
        ),
        PatchSpec(
            "legacy_pre_state_as_active_plus_camera_refined",
            ("persistent_state", "camera_refined"),
        ),
        PatchSpec("H_post_update_state_input", ("post_update_state_input",)),
        PatchSpec(
            "H_post_update_state_input_plus_camera",
            ("post_update_state_input", "camera_initial"),
        ),
    ]
    for layer in range(4):
        specs.extend(
            [
                PatchSpec(f"D_early_camera_l{layer}", (f"camera_l{layer}",)),
                PatchSpec(
                    f"F_early_camera_l{layer}_plus_first_write",
                    (f"camera_l{layer}", "first_write_state"),
                ),
                PatchSpec(
                    f"G_read_old_write_fresh_l{layer}",
                    (f"read_old_write_fresh_l{layer}",),
                ),
            ]
        )
    specs.extend(
        [
            PatchSpec("G_read_random_old_write_fresh_l1", ("read_old_write_fresh_l1",), "random"),
            PatchSpec("G_read_shuffled_old_write_fresh_l1", ("read_old_write_fresh_l1",), "shuffle"),
        ]
    )
    return specs


def offset_recovery(variants: dict[str, dict], offsets: tuple[int, ...]) -> None:
    reset_frames = variants["B_reset_baseline"]["metrics"]["per_frame"]
    for name, variant in variants.items():
        frames = variant["metrics"]["per_frame"]
        result = {}
        for offset in offsets:
            if offset >= len(frames) or offset >= len(reset_frames):
                continue
            result[str(offset)] = {}
            for key, reset_value in reset_frames[offset].items():
                if key == "post_index":
                    continue
                value = frames[offset][key]
                if not np.isfinite(reset_value) or abs(float(reset_value)) < 1e-8 or not np.isfinite(value):
                    recovery = float("nan")
                else:
                    recovery = float((reset_value - value) / reset_value)
                result[str(offset)][key] = recovery
        variant["offset_recovery"] = result


def run_case(model, input_dir: Path, args: argparse.Namespace, device: torch.device, case_index: int) -> dict:
    paths = image_paths(input_dir, int(args.max_frames))
    boundary = int(args.boundary)
    if boundary <= 0 or boundary + 8 >= len(paths):
        raise ValueError(f"Need boundary+8 < frames, got boundary={boundary}, frames={len(paths)}")
    views = prepare_views(paths, model, args, device)
    teacher_predictions, teacher_latents, teacher_seconds, _ = run_branch(
        model, views, device, boundary, capture=True, seed=args.seed + case_index
    )
    post_views = views[boundary:]
    reset_predictions, reset_latents, reset_seconds, _ = run_branch(
        model, post_views, device, 0, capture=True, seed=args.seed + case_index
    )
    teacher_post = teacher_predictions[boundary:]
    variants = {
        "A_continuous_teacher": {
            "metrics": evaluate_branch(teacher_post, teacher_post, args, args.seed + case_index),
            "seconds": teacher_seconds,
            "patch": [],
        },
        "B_reset_baseline": {
            "metrics": evaluate_branch(reset_predictions, teacher_post, args, args.seed + case_index),
            "seconds": reset_seconds,
            "patch": [],
        },
    }
    source = source_dict(teacher_latents)
    for spec_index, spec in enumerate(variant_specs()):
        predictions, _latents, seconds, skipped = run_branch(
            model,
            post_views,
            device,
            0,
            capture=False,
            patch=spec,
            source=source,
            seed=args.seed + case_index * 1000 + spec_index,
        )
        variants[spec.name] = {
            "metrics": evaluate_branch(predictions, teacher_post, args, args.seed + case_index),
            "seconds": seconds,
            "patch": list(spec.components),
            "source_mode": spec.source_mode,
            "skipped_replacements": skipped,
        }
    # add_recovery expects this legacy key.
    variants["reset_raw"] = variants["B_reset_baseline"]
    add_recovery(variants)
    variants.pop("reset_raw")
    offset_recovery(variants, (0, 1, 2, 4, 8))
    report = {
        "case_name": safe_name(input_dir.name),
        "input_dir": str(input_dir),
        "input_images": paths,
        "boundary": boundary,
        "post_frames": len(post_views),
        "stage0_audit": stage0_audit(model, teacher_latents, reset_latents),
        "variants": variants,
    }
    del views, post_views
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def aggregate(cases: list[dict]) -> dict:
    names = sorted(set.intersection(*(set(case["variants"]) for case in cases)))
    overall = {}
    for name in names:
        mean_keys = cases[0]["variants"][name]["metrics"]["mean"].keys()
        recovery_keys = cases[0]["variants"][name]["recovery"].keys()
        offset_keys = sorted(
            set.intersection(*(set(case["variants"][name]["offset_recovery"]) for case in cases)),
            key=int,
        )
        overall[name] = {
            "mean_error": {
                key: finite_mean([case["variants"][name]["metrics"]["mean"][key] for case in cases])
                for key in mean_keys
            },
            "mean_recovery": {
                key: finite_mean([case["variants"][name]["recovery"][key] for case in cases])
                for key in recovery_keys
            },
            "offset_recovery": {
                offset: {
                    key: finite_mean(
                        [case["variants"][name]["offset_recovery"][offset][key] for case in cases]
                    )
                    for key in cases[0]["variants"][name]["offset_recovery"][offset]
                }
                for offset in offset_keys
            },
        }
    return overall


def write_csv(path: Path, overall: dict) -> None:
    rows = []
    for name, row in overall.items():
        output = {"variant": name}
        output.update({f"error_{key}": value for key, value in row["mean_error"].items()})
        output.update({f"recovery_{key}": value for key, value in row["mean_recovery"].items()})
        for offset, metrics in row["offset_recovery"].items():
            for key, value in metrics.items():
                output[f"off{offset}_recovery_{key}"] = value
        rows.append(output)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_recovery(path: Path, overall: dict) -> None:
    names = [
        "B_reset_baseline",
        "C_late_camera_refined",
        "C_late_camera_refined_plus_first_write",
        "D_early_camera_initial",
        "D_early_camera_l1",
        "E_first_write_state",
        "F_initial_camera_plus_first_write",
        "F_early_camera_l1_plus_first_write",
        "G_read_old_pose_memory_write_fresh",
        "G_read_old_write_fresh_l1",
        "legacy_pre_state_as_active",
        "H_post_update_state_input",
    ]
    names = [name for name in names if name in overall]
    metrics = [
        "camera_translation_m",
        "camera_rotation_deg",
        "pointmap_world_mean_m",
        "human_world_root_m",
        "human_global_orientation_deg",
    ]
    matrix = np.asarray([[overall[name]["mean_recovery"].get(key, np.nan) for key in metrics] for name in names])
    fig, ax = plt.subplots(figsize=(11, max(5, 0.48 * len(names))))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)), [key.replace("_", "\n") for key in metrics], fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for y in range(len(names)):
        for x in range(len(metrics)):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Causal State Transition Oracle Recovery")
    fig.colorbar(image, ax=ax, label="Recovery ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_offsets(path: Path, overall: dict) -> None:
    names = [
        "C_late_camera_refined",
        "D_early_camera_initial",
        "E_first_write_state",
        "F_initial_camera_plus_first_write",
        "G_read_old_pose_memory_write_fresh",
        "G_read_old_write_fresh_l1",
        "H_post_update_state_input",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in (
        (axes[0], "camera_rotation_deg", "Camera rotation"),
        (axes[1], "pointmap_world_mean_m", "World pointmap"),
        (axes[2], "human_world_root_m", "Human root"),
    ):
        for name in names:
            if name not in overall:
                continue
            offsets = sorted(overall[name]["offset_recovery"], key=int)
            values = [overall[name]["offset_recovery"][offset][metric] for offset in offsets]
            ax.plot([int(offset) for offset in offsets], values, marker="o", label=name)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Post-cut offset")
        ax.set_ylabel("Recovery")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args)
    cases = []
    for index, input_dir in enumerate(args.input_dirs):
        print(f">> [{index + 1}/{len(args.input_dirs)}] {input_dir}", flush=True)
        cases.append(run_case(model, input_dir, args, device, index))
    overall = aggregate(cases)
    report = {
        "experiment": "Causal State Transition Oracle",
        "case_count": len(cases),
        "constraints": {
            "same_boundary_rgb": True,
            "teacher_post_update_state_is_oracle_only": True,
            "read_old_write_fresh_discards_old_state_write": True,
            "human3r_frozen": True,
        },
        "overall": overall,
        "cases": cases,
    }
    (args.output_dir / "causal_state_transition_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "causal_state_transition_metrics.csv", overall)
    plot_recovery(args.output_dir / "causal_state_transition_recovery.png", overall)
    plot_offsets(args.output_dir / "causal_state_transition_offsets.png", overall)
    print(f">> wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
