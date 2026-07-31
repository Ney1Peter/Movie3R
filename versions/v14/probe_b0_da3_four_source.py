#!/usr/bin/env python3
"""Frozen four-source evaluation of deployable B0 + DA3 fine alignment.

The proposal path receives only the frozen V14 B0 inputs plus the last pre-cut
and first post-cut RGB images. Ground-truth camera/SMPL-X fields are retained on
CPU and are accessed only after both B0 and the refined Boundary are finalized.

This is a source-diversity diagnostic over the already selected 180 AABB cuts,
not a parameter-selection script. FineAlignmentConfig remains frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
DA3_ROOT = REPO_ROOT.parent / "Movie3R-dataset" / "Depth-Anything-3"
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT, DA3_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from depth_anything_3.api import DepthAnything3  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from v10_token_alignment_4source_probe import (  # noqa: E402
    load_aabb_views_for_record,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from versions.v14.b0_da3_fine_alignment import (  # noqa: E402
    DEFAULT_CONFIG,
    refine_b0_with_da3,
)
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    gt_head_world,
    model_batch_from_gt,
    predicted_head_world,
    set_event_indices,
)


DEFAULT_CHECKPOINT = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_DA3 = DA3_ROOT / "checkpoints" / "DAE-base"
DEFAULT_RECORDS = (
    REPO_ROOT
    / "output/v10_candidate_selection/oracle_gt_4source/selected_records.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/da3_shared_pose_four_source"
)
SOURCE_ORDER = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--da3_path", type=Path, default=DEFAULT_DA3)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_ORDER, default=SOURCE_ORDER)
    parser.add_argument("--max_cases_per_source", type=int, default=0)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("_")


def image_path(view: dict) -> Path | None:
    value = view.get("instance")
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, str) and Path(value).is_file():
        return Path(value)
    return None


def rgb_from_view(view: dict) -> tuple[np.ndarray, str]:
    path = image_path(view)
    if path is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), str(path)
    image = view["img"].detach().float().cpu().numpy()[0]
    image = np.moveaxis(image, 0, -1)
    image = np.clip(np.rint((image + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return image, "normalized_model_input_fallback"


def add_mhmr_inputs(views: list[dict], model_resolution: int = 896) -> None:
    """Match the current V14 demo preprocessing for the MHMR image branch."""
    from dust3r.utils.geometry import resize_camera_intrinsics
    from dust3r.utils.image import pad_image

    images = torch.stack([view["img"] for view in views], dim=0)
    images = images.view(-1, *images.shape[2:])
    intrinsics = torch.stack([view["camera_intrinsics"] for view in views], dim=0)
    intrinsics = intrinsics.view(-1, *intrinsics.shape[2:])
    intrinsics_mhmr = resize_camera_intrinsics(
        intrinsics, *images.shape[2:], int(model_resolution)
    )
    images_mhmr = pad_image(images, int(model_resolution))
    for view, image_mhmr, intrinsic_mhmr in zip(
        views,
        images_mhmr.chunk(len(views), dim=0),
        intrinsics_mhmr.chunk(len(views), dim=0),
    ):
        view["img_mhmr"] = image_mhmr
        view["K_mhmr"] = intrinsic_mhmr


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (4, 4):
        return value
    if value.shape != (3, 4):
        raise ValueError(f"Expected 3x4/4x4 matrix, got {value.shape}")
    output = np.eye(4, dtype=np.float64)
    output[:3] = value
    return output


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_human3r(
    model: ARCroco3DStereo,
    gt_views: list[dict],
    boundary: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], float]:
    clean = todevice(model_batch_from_gt(gt_views), device)
    shadow_views = set_event_indices(clean[: boundary + 1], {boundary})
    raw_views = set_event_indices(clean[boundary : boundary + 1], set())
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow, _ = model.forward_recurrent_lighter(
            shadow_views, str(device), ret_state=False, use_ttt3r=False
        )
        raw, _ = model.forward_recurrent_lighter(
            raw_views, str(device), ret_state=False, use_ttt3r=False
        )
    synchronize(device)
    return shadow, raw, time.perf_counter() - started


def run_da3(
    model: DepthAnything3,
    first: np.ndarray,
    second: np.ndarray,
    process_res: int,
    device: torch.device,
) -> dict:
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model.inference(
            [first, second],
            process_res=int(process_res),
            use_ray_pose=False,
            ref_view_strategy="first",
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    if prediction.extrinsics is None:
        return {"status": "no_extrinsics", "elapsed_seconds": elapsed}
    extrinsics = np.stack([homogeneous(row) for row in prediction.extrinsics])
    camera_to_world = np.linalg.inv(extrinsics)
    output = {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "camera_to_world": camera_to_world,
        "baseline_units": float(
            np.linalg.norm(camera_to_world[1, :3, 3] - camera_to_world[0, :3, 3])
        ),
    }
    if prediction.conf is not None:
        confidence = np.asarray(prediction.conf, dtype=np.float64)
        output["confidence_mean"] = float(np.mean(confidence))
        output["confidence_p10"] = float(np.percentile(confidence, 10))
    return output


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    relative = estimated[:3, :3].T @ target[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def evaluate_boundary(
    boundary: np.ndarray,
    raw_prediction: dict,
    raw_camera: np.ndarray,
    target_camera: np.ndarray,
    target_head: np.ndarray | None,
) -> dict:
    estimated_camera = np.asarray(boundary, dtype=np.float64) @ raw_camera
    translation = float(
        np.linalg.norm(estimated_camera[:3, 3] - target_camera[:3, 3])
    )
    rotation = rotation_error_deg(estimated_camera, target_camera)
    estimated_head = predicted_head_world(raw_prediction, estimated_camera)
    human = (
        float(np.linalg.norm(estimated_head - target_head))
        if estimated_head is not None and target_head is not None
        else float("nan")
    )
    return {
        "boundary": np.asarray(boundary, dtype=np.float64),
        "estimated_camera": estimated_camera,
        "camera_translation_error_m": translation,
        "camera_rotation_error_deg": rotation,
        "camera_composite": translation + 0.02 * rotation,
        "catastrophic": bool(translation > 1.0 or rotation > 30.0),
        "human_head_error_m": human,
    }


def evaluate_record(
    record: dict,
    args: argparse.Namespace,
    human3r: ARCroco3DStereo,
    da3: DepthAnything3,
    smpl_layer: SMPL_Layer,
    device: torch.device,
) -> dict:
    loader_args = SimpleNamespace(
        data_root=args.data_root,
        resolution=tuple(args.resolution),
        resize_mode=args.resize_mode,
        boundary=int(args.boundary),
    )
    gt_views = load_aabb_views_for_record(record, loader_args, torch.device("cpu"))
    add_mhmr_inputs(gt_views)
    boundary_index = int(args.boundary)
    first_rgb, first_path = rgb_from_view(gt_views[boundary_index - 1])
    second_rgb, second_path = rgb_from_view(gt_views[boundary_index])

    shadow, raw, human3r_seconds = run_human3r(
        human3r, gt_views, boundary_index, device
    )
    pre_camera = camera_matrix(shadow[boundary_index - 1]).astype(np.float64)
    raw_camera = camera_matrix(raw[0]).astype(np.float64)
    b0 = (
        boundary_from_camera_predictions(shadow[-1], raw[0])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    forward = run_da3(da3, first_rgb, second_rgb, args.process_res, device)
    reverse = run_da3(da3, second_rgb, first_rgb, args.process_res, device)
    fine, diagnostics = refine_b0_with_da3(
        b0,
        pre_camera,
        raw_camera,
        forward.get("camera_to_world") if forward["status"] == "ok" else None,
        reverse.get("camera_to_world") if reverse["status"] == "ok" else None,
    )

    # Evaluation starts here. None of the fields below influence B0, DA3, or the gate.
    gt_pre = gt_pose_from_view(gt_views[boundary_index - 1]).cpu().numpy().astype(np.float64)
    gt_post = gt_pose_from_view(gt_views[boundary_index]).cpu().numpy().astype(np.float64)
    evaluation_gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_camera = evaluation_gauge @ gt_post
    target_head = gt_head_world(
        gt_views[boundary_index], evaluation_gauge, smpl_layer
    )
    methods = {
        "b0": evaluate_boundary(b0, raw[0], raw_camera, target_camera, target_head),
        "da3_safe": evaluate_boundary(
            fine, raw[0], raw_camera, target_camera, target_head
        ),
    }
    return {
        "status": "ok",
        "case_name": safe_name(record["pattern_id"]),
        "record": record,
        "inputs": {
            "last_pre_rgb": first_path,
            "first_post_rgb": second_path,
            "deployment_input_audit": (
                "RGB + Human3R pre/raw-post poses + frozen B0 + frozen DA3 only"
            ),
            "gt_usage": "evaluation after Boundary finalization only",
        },
        "timing_seconds": {
            "human3r": human3r_seconds,
            "da3_forward": forward["elapsed_seconds"],
            "da3_reverse": reverse["elapsed_seconds"],
        },
        "poses": {
            "human3r_pre": pre_camera,
            "human3r_raw_post": raw_camera,
            "target_post_evaluation_only": target_camera,
        },
        "da3": {"forward": forward, "reverse": reverse},
        "fine_diagnostics": diagnostics,
        "methods": methods,
    }


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(rows: list[dict]) -> dict:
    output = {"case_count": len(rows), "gate_acceptance": float("nan"), "methods": {}}
    if not rows:
        return output
    output["gate_acceptance"] = float(
        np.mean([row["fine_diagnostics"]["accepted"] for row in rows])
    )
    metrics = (
        "camera_translation_error_m",
        "camera_rotation_error_deg",
        "camera_composite",
        "human_head_error_m",
    )
    for method in ("b0", "da3_safe"):
        result = {
            metric: stats([row["methods"][method][metric] for row in rows])
            for metric in metrics
        }
        result["catastrophic_count"] = int(
            sum(row["methods"][method]["catastrophic"] for row in rows)
        )
        result["catastrophic_rate"] = float(
            np.mean([row["methods"][method]["catastrophic"] for row in rows])
        )
        if method != "b0":
            result["paired"] = {
                metric: {
                    "mean_delta": float(
                        np.nanmean(
                            [
                                row["methods"][method][metric]
                                - row["methods"]["b0"][metric]
                                for row in rows
                            ]
                        )
                    ),
                    "improvement_rate": float(
                        np.nanmean(
                            [
                                row["methods"][method][metric]
                                < row["methods"]["b0"][metric]
                                for row in rows
                                if np.isfinite(row["methods"][method][metric])
                                and np.isfinite(row["methods"]["b0"][metric])
                            ]
                        )
                    ),
                }
                for metric in metrics
            }
        output["methods"][method] = result
    output["timing_seconds"] = {
        key: stats([row["timing_seconds"][key] for row in rows])
        for key in ("human3r", "da3_forward", "da3_reverse")
    }
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 Frozen B0 + DA3 Four-Source Evaluation",
        "",
        "Parameters and gate were frozen on `three`; this run does not tune them.",
        "GT is used only after each output Boundary is finalized.",
        "",
        "| Split | N | Gate | Method | Camera T | Camera R | Composite | P95 comp. | Human head | Catastrophic |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("overall", report["summary"])] + [
        (source, report["by_source"][source]) for source in SOURCE_ORDER if source in report["by_source"]
    ]
    for name, summary in groups:
        if not summary["methods"]:
            continue
        for method in ("b0", "da3_safe"):
            row = summary["methods"][method]
            lines.append(
                f"| {name} | {summary['case_count']} | {summary['gate_acceptance']:.1%} | "
                f"{method} | {row['camera_translation_error_m']['mean']:.4f} | "
                f"{row['camera_rotation_error_deg']['mean']:.3f} | "
                f"{row['camera_composite']['mean']:.4f} | "
                f"{row['camera_composite']['p95']:.4f} | "
                f"{row['human_head_error_m']['mean']:.4f} | "
                f"{row['catastrophic_count']} |"
            )
    lines.extend(
        [
            "",
            f"Completed cases: `{report['summary']['case_count']}`; failures: `{len(report['failures'])}`.",
            "",
            "`human_head` is a single-person translation/head proxy available on these AABB records; "
            "the held-out MultiHuman evaluation remains the stronger multi-person layout test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if not (args.da3_path / "model.safetensors").is_file():
        raise FileNotFoundError(args.da3_path / "model.safetensors")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    counts: defaultdict[str, int] = defaultdict(int)
    for record in read_jsonl(args.records):
        source = str(record["source"])
        if source not in args.sources:
            continue
        if args.max_cases_per_source and counts[source] >= args.max_cases_per_source:
            continue
        selected.append(record)
        counts[source] += 1

    human3r = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(human3r)
    da3 = DepthAnything3.from_pretrained(str(args.da3_path)).to(device).eval()
    smpl_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()

    rows, failures = [], []
    for index, record in enumerate(selected, start=1):
        name = safe_name(record["pattern_id"])
        path = cases_dir / f"{name}.json"
        cached = None
        if path.is_file() and not args.overwrite:
            cached = json.loads(path.read_text(encoding="utf-8"))
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                row = evaluate_record(
                    record, args, human3r, da3, smpl_layer, device
                )
            except Exception as error:  # persist data/model failures and continue
                row = {
                    "status": "failed",
                    "case_name": name,
                    "record": record,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            path.write_text(
                json.dumps(jsonable(row), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if row["status"] == "ok":
            rows.append(row)
            method = row["methods"]["da3_safe"]
            print(
                f"[{index:03d}/{len(selected):03d}] {name} "
                f"gate={row['fine_diagnostics']['accepted']} "
                f"comp={row['methods']['b0']['camera_composite']:.4f}->"
                f"{method['camera_composite']:.4f}",
                flush=True,
            )
        else:
            failures.append(row)
            print(
                f"[{index:03d}/{len(selected):03d}] {name} FAILED {row['error']}",
                flush=True,
            )
        if device.type == "cuda" and index % 10 == 0:
            torch.cuda.empty_cache()

    by_source = {
        source: summarize([row for row in rows if row["record"]["source"] == source])
        for source in SOURCE_ORDER
        if any(row["record"]["source"] == source for row in rows)
    }
    by_angle = {
        bucket: summarize(
            [row for row in rows if row["record"].get("angle_bucket") == bucket]
        )
        for bucket in sorted({row["record"].get("angle_bucket") for row in rows})
    }
    report = {
        "experiment": "v14_b0_da3_four_source_frozen",
        "protocol": {
            "records": str(args.records),
            "requested_case_count": len(selected),
            "sources": list(args.sources),
            "boundary": int(args.boundary),
            "fine_alignment_config": DEFAULT_CONFIG.__dict__,
            "parameter_selection": "none; frozen on MultiHuman three",
            "gt_usage": "camera/head evaluation after Boundary finalization only",
            "proposal_inputs": (
                "last pre RGB, first post RGB, Human3R pre/raw-post camera, B0, DA3"
            ),
        },
        "models": {
            "human3r_checkpoint": str(args.model_path),
            "human3r_flags": flags,
            "da3_checkpoint": str(args.da3_path),
            "device": str(device),
            "process_res": int(args.process_res),
        },
        "summary": summarize(rows),
        "by_source": by_source,
        "by_angle_bucket": by_angle,
        "failures": failures,
        "cases": rows,
    }
    json_path = args.output_dir / "v14_b0_da3_four_source.json"
    md_path = args.output_dir / "v14_b0_da3_four_source.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json_path, flush=True)
    print(md_path, flush=True)


if __name__ == "__main__":
    main()
