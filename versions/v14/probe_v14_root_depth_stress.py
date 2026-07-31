#!/usr/bin/env python3
"""Evaluate Human3R-internal root depth on mined near/far stress cuts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import AvatarReX_AABB  # noqa: E402
from dust3r.inference import _make_v8_image_only_model_batch  # noqa: E402
from dust3r.utils.geometry import get_camera_parameters  # noqa: E402
from dust3r.utils.image import pad_image  # noqa: E402
from scripts.boundary_human3r_reset_support import build_smpl_models  # noqa: E402
from versions.v13.native_token_probe import jsonable, tensor_numpy  # noqa: E402
from versions.v14.probe_v14_internal_root_depth import (  # noqa: E402
    conservative_fusion,
    decode_local_humans,
    mask_translation_candidate,
    pointmap_candidate,
    prediction_mask,
    prediction_pointmap,
    quantile_box,
    split_person_mask,
)
from versions.v14.run_v14_2_single_sequence import configure_model  # noqa: E402


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_MINING = (
    REPO_ROOT
    / "output/v14/root_depth_stress_mining/root_depth_stress_mining.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/root_depth_stress_probe"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
AVATAR_GROUPS = ("lbn1", "lbn2", "zzr", "zxc")
METHODS = (
    "raw",
    "pointmap_z",
    "persistent_mask_ratio_z",
    "mask_translation",
    "candidate_mean",
    "conservative_gate",
    "oracle_candidate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mining_report", type=Path, default=DEFAULT_MINING)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases_per_source", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask_threshold", type=float, default=0.50)
    parser.add_argument("--point_radius", type=float, default=0.025)
    parser.add_argument("--max_point_iqr_m", type=float, default=0.45)
    parser.add_argument("--max_relative_shift", type=float, default=0.30)
    parser.add_argument("--agreement_m", type=float, default=0.22)
    parser.add_argument("--agreement_relative", type=float, default=0.12)
    parser.add_argument("--gate_multiscale_range_m", type=float, default=0.005)
    parser.add_argument("--gate_point_iqr_m", type=float, default=0.10)
    parser.add_argument("--gate_relative_shift", type=float, default=0.08)
    parser.add_argument("--gate_absolute_shift_m", type=float, default=0.20)
    return parser.parse_args()


def raw_calibration_roots(data_root: Path) -> dict[str, str]:
    return {
        name: str(data_root / "AvatarReX_raw_meta" / name)
        for name in AVATAR_GROUPS
    }


def dataset_for_rows(
    args: argparse.Namespace, source: str, rows: list[dict]
) -> AvatarReX_AABB:
    samples = [
        (str(row["seqA"]), str(row["seqB"]), int(row["start_frame"]))
        for row in rows
    ]
    is_mvhuman = str(source).startswith("mvhuman")
    return AvatarReX_AABB(
        allow_repeat=True,
        split="Training/mvhuman" if is_mvhuman else "Training",
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=int(args.size),
        resize_mode="human3r_demo",
        num_views=4,
        seed=14201,
        n_corres=0,
        fixed_samples=samples,
        load_da3_depth=False,
        raw_calibration_root=(
            None if is_mvhuman else raw_calibration_roots(args.data_root)
        ),
        max_humans=1,
    )


def prepare_post_view(view: dict, model) -> dict:
    clean = _make_v8_image_only_model_batch([view])[0]
    clean["img_mhmr"] = pad_image(clean["img"], int(model.mhmr_img_res))
    clean["K_mhmr"] = get_camera_parameters(
        int(model.mhmr_img_res), device="cpu"
    )
    reference = clean["img_mask"]
    clean["reset"] = torch.zeros_like(reference, dtype=torch.bool)
    for key in ("update", "update_state", "update_mem", "update_v8_history"):
        clean[key] = torch.ones_like(reference, dtype=torch.bool)
    clean["shot_label"] = torch.zeros_like(reference, dtype=torch.float32)
    return clean


def gt_local_root(view: dict, smpl_layer) -> np.ndarray:
    mask = view["smpl_mask"][0].detach().cpu().bool()
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    if not len(ids):
        raise ValueError("No valid GT human")
    index = int(ids[0])
    device = next(smpl_layer.parameters()).device

    def parameter(key: str) -> torch.Tensor:
        return view[key][0, index].detach().float().to(device)

    with torch.no_grad():
        output = smpl_layer.bm_x(
            global_orient=parameter("smplx_root_pose").reshape(1, 3),
            body_pose=parameter("smplx_body_pose").reshape(1, -1),
            left_hand_pose=parameter("smplx_left_hand_pose").reshape(1, -1),
            right_hand_pose=parameter("smplx_right_hand_pose").reshape(1, -1),
            jaw_pose=parameter("smplx_jaw_pose").reshape(1, 3),
            betas=parameter("smplx_shape")[: smpl_layer.num_betas].reshape(1, -1),
            transl=parameter("smplx_transl").reshape(1, 3),
        )
    pelvis_index = smpl_layer.joint_names.index("pelvis")
    root = output.joints[0, pelvis_index].detach().float().cpu().numpy()
    params_are_world = bool(view["human_params_are_world"][0].detach().cpu())
    if params_are_world:
        w2c = tensor_numpy(view["T_w2c"])[0]
        root = root @ w2c[:3, :3].T + w2c[:3, 3]
    return np.asarray(root, dtype=np.float64)


def run_prediction(model, view: dict, args: argparse.Namespace):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        predictions, output_views, debug = model.forward_recurrent_lighter(
            [prepare_post_view(view, model)],
            str(args.device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    return predictions[0], output_views[0], debug[0]


def evaluate_case(
    model,
    pred_layer,
    args: argparse.Namespace,
    stress: dict,
    batch: list[dict],
) -> dict:
    post_view = batch[2]
    target_root = gt_local_root(post_view, pred_layer)
    started = time.perf_counter()
    pre_prediction, pre_output_view, pre_debug = run_prediction(
        model, batch[1], args
    )
    prediction, output_view, debug = run_prediction(model, post_view, args)
    pre_humans = decode_local_humans(
        pre_prediction, pre_output_view, pre_debug, pred_layer
    )
    humans = decode_local_humans(prediction, output_view, debug, pred_layer)
    if not humans or not pre_humans:
        return {
            "stress": stress,
            "status": "no_detection" if not humans else "no_pre_detection",
            "runtime_seconds": time.perf_counter() - started,
        }
    pre_human = max(pre_humans, key=lambda row: float(row["score"]))
    human = max(humans, key=lambda row: float(row["score"]))
    points, confidence = prediction_pointmap(prediction)
    mask = prediction_mask(prediction)
    mhmr_size = int(output_view["img_mhmr"].shape[-1])
    if mask is not None and mask.shape != (mhmr_size, mhmr_size):
        mask = cv2.resize(mask, (mhmr_size, mhmr_size), interpolation=cv2.INTER_LINEAR)
    locations = np.stack([row["location_mhmr"] for row in humans])
    person_masks = (
        split_person_mask(mask, locations, float(args.mask_threshold))
        if mask is not None
        else [None for _ in humans]
    )
    pre_mask = prediction_mask(pre_prediction)
    pre_mhmr_size = int(pre_output_view["img_mhmr"].shape[-1])
    if pre_mask is not None and pre_mask.shape != (pre_mhmr_size, pre_mhmr_size):
        pre_mask = cv2.resize(
            pre_mask,
            (pre_mhmr_size, pre_mhmr_size),
            interpolation=cv2.INTER_LINEAR,
        )
    pre_locations = np.stack([row["location_mhmr"] for row in pre_humans])
    pre_person_masks = (
        split_person_mask(pre_mask, pre_locations, float(args.mask_threshold))
        if pre_mask is not None
        else [None for _ in pre_humans]
    )
    selected_index = int(human["detection_index"])
    pre_selected_index = int(pre_human["detection_index"])
    point_shift, point_debug = pointmap_candidate(
        human,
        points,
        confidence,
        person_masks[selected_index],
        mhmr_size,
        args,
    )
    intrinsic = tensor_numpy(output_view["K_mhmr"])[0].astype(np.float64)
    mask_shift, mask_debug = mask_translation_candidate(
        human,
        person_masks[selected_index]
        if person_masks[selected_index] is not None
        else np.zeros((mhmr_size, mhmr_size), dtype=bool),
        intrinsic,
        mhmr_size,
    )
    valid = [value for value in (point_shift, mask_shift) if value is not None]
    mean_shift = np.mean(np.stack(valid), axis=0) if valid else np.zeros(3)
    pre_box = (
        quantile_box(pre_person_masks[pre_selected_index])
        if pre_person_masks[pre_selected_index] is not None
        else None
    )
    post_box = (
        quantile_box(person_masks[selected_index])
        if person_masks[selected_index] is not None
        else None
    )
    persistent_shift = None
    persistent_debug = {"status": "missing_mask_box"}
    if pre_box is not None and post_box is not None:
        pre_height = max(float(pre_box[3] - pre_box[1]), 1.0)
        post_height = max(float(post_box[3] - post_box[1]), 1.0)
        target_depth = float(pre_human["root"][2]) * pre_height / post_height
        persistent_shift = np.asarray(
            (0.0, 0.0, target_depth - float(human["root"][2])),
            dtype=np.float64,
        )
        persistent_debug = {
            "status": "ok",
            "pre_mask_height_px": pre_height,
            "post_mask_height_px": post_height,
            "height_ratio_pre_over_post": pre_height / post_height,
            "pre_root_depth_m": float(pre_human["root"][2]),
            "target_post_depth_m": target_depth,
            "root_shift_m": persistent_shift,
        }
    gated_shift, gate_debug = conservative_fusion(
        human, point_shift, point_debug, args
    )
    shifts = {
        "raw": np.zeros(3, dtype=np.float64),
        "pointmap_z": point_shift if point_shift is not None else np.zeros(3),
        "persistent_mask_ratio_z": (
            persistent_shift if persistent_shift is not None else np.zeros(3)
        ),
        "mask_translation": mask_shift if mask_shift is not None else np.zeros(3),
        "candidate_mean": mean_shift,
        "conservative_gate": gated_shift,
    }
    metrics = {}
    for name, shift in shifts.items():
        predicted = np.asarray(human["root"], dtype=np.float64) + shift
        metrics[name] = {
            "root_error_m": float(np.linalg.norm(predicted - target_root)),
            "depth_error_m": float(abs(predicted[2] - target_root[2])),
            "predicted_root": predicted,
            "target_root": target_root,
        }
    choices = ["raw"]
    if point_shift is not None:
        choices.append("pointmap_z")
    if persistent_shift is not None:
        choices.append("persistent_mask_ratio_z")
    if mask_shift is not None:
        choices.append("mask_translation")
    if valid:
        choices.append("candidate_mean")
    best = min(choices, key=lambda name: metrics[name]["root_error_m"])
    shifts["oracle_candidate"] = shifts[best]
    metrics["oracle_candidate"] = dict(metrics[best])
    return {
        "stress": stress,
        "status": "ok",
        "runtime_seconds": time.perf_counter() - started,
        "detection_count": len(humans),
        "selected_detection": selected_index,
        "raw_root_depth_m": float(human["root"][2]),
        "gt_root_depth_m": float(target_root[2]),
        "pointmap": point_debug,
        "persistent_mask_ratio": persistent_debug,
        "mask": mask_debug,
        "gate": gate_debug,
        "oracle_candidate_name": best,
        "shifts": shifts,
        "metrics": metrics,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["status"] == "ok"]
    raw = np.asarray([row["metrics"]["raw"]["root_error_m"] for row in valid])
    output = {
        "case_count": len(rows),
        "valid_count": len(valid),
        "no_detection_count": len(rows) - len(valid),
    }
    for method in METHODS:
        current = np.asarray(
            [row["metrics"][method]["root_error_m"] for row in valid]
        )
        output[method] = {
            "root_error_m": distribution(current.tolist()),
            "depth_error_m": distribution(
                [row["metrics"][method]["depth_error_m"] for row in valid]
            ),
            "improved_fraction": float(np.mean(current < raw - 1e-8)),
            "harmed_over_5cm_fraction": float(np.mean(current > raw + 0.05)),
        }
    output["gate_coverage"] = float(
        np.mean([row["gate"]["status"] == "accepted" for row in valid])
    )
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 Root-Depth Stress Probe",
        "",
        "Selected cases are the strongest GT near/far or occupancy changes from each held-out manifest.",
        "Inference uses a fresh single post-cut frame, pseudo intrinsics, and no external depth model.",
        "",
    ]
    for source, summary in report["summary"].items():
        lines.extend(
            [
                f"## {source}",
                "",
                f"Valid: `{summary['valid_count']}/{summary['case_count']}`; gate coverage: "
                f"`{100.0 * summary['gate_coverage']:.1f}%`.",
                "",
                "| Method | Root mean | Root median | Root P90 | Depth mean | Improved | >5cm harm |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = summary[method]
            lines.append(
                f"| {method} | {row['root_error_m']['mean']:.3f} | "
                f"{row['root_error_m']['median']:.3f} | {row['root_error_m']['p90']:.3f} | "
                f"{row['depth_error_m']['mean']:.3f} | "
                f"{100.0 * row['improved_fraction']:.1f}% | "
                f"{100.0 * row['harmed_over_5cm_fraction']:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mining = json.loads(args.mining_report.read_text(encoding="utf-8"))
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
    flags = configure_model(model)
    _, pred_layer = build_smpl_models(model, torch.device(args.device))
    all_rows = {}
    for source, selected in mining["selected"].items():
        selected = selected[: int(args.max_cases_per_source)]
        dataset = dataset_for_rows(args, source, selected)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        rows = []
        for index, (stress, batch) in enumerate(zip(selected, loader), start=1):
            output_path = args.output_dir / "cases" / source / f"case_{index:02d}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.is_file() and not args.overwrite:
                row = json.loads(output_path.read_text(encoding="utf-8"))
            else:
                row = evaluate_case(model, pred_layer, args, stress, batch)
                output_path.write_text(
                    json.dumps(jsonable(row), indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
            rows.append(row)
            print(
                f">> [{source} {index}/{len(selected)}] {row['status']}", flush=True
            )
        all_rows[source] = rows
    report = {
        "experiment": "v14_root_depth_stress_probe",
        "model_path": str(args.model_path),
        "model_flags": flags,
        "protocol": {
            "mining_report": str(args.mining_report),
            "gt_use": "stress selection and metrics only",
            "pseudo_intrinsics": True,
            "external_depth_model": False,
            "fresh_post_cut_frame": True,
            "gate_threshold_source": "MultiHuman three development only",
        },
        "summary": {source: summarize(rows) for source, rows in all_rows.items()},
        "cases": all_rows,
    }
    json_path = args.output_dir / "root_depth_stress_probe.json"
    md_path = args.output_dir / "README.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f">> wrote {json_path}", flush=True)
    print(f">> wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
