#!/usr/bin/env python3
"""Compare an event-only checkpoint with raw Human3R on a no-event segment."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
THIS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, THIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from evaluate_cut_events import (  # noqa: E402
    SOURCE_SPECS,
    add_mhmr_inputs,
    raw_calibration_roots,
    read_jsonl,
)
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    configure_model,
    model_batch_from_gt,
    set_event_indices,
)


DEFAULT_EVENT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_HUMAN3R = SRC_ROOT / "human3r_896L.pth"
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "config/manifests/v14_1_cut_event/ten"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14_cut_first_cross_source/reset_only_parity"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
COMPARE_KEYS = (
    "camera_pose",
    "pts3d_in_self_view",
    "pts3d_in_other_view",
    "conf_self",
    "conf",
    "smpl_transl",
    "smpl_root_pose",
    "smpl_body_pose",
    "smpl_shape",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model", type=Path, default=DEFAULT_EVENT_MODEL)
    parser.add_argument("--human3r-model", type=Path, default=DEFAULT_HUMAN3R)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--source", choices=tuple(SOURCE_SPECS), default="avatarrex")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def make_no_event_batch(args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    from dust3r.datasets.avatarrex import AvatarReX_Pattern

    filename, split = SOURCE_SPECS[args.source]
    source_record = read_jsonl(args.manifest_root / filename)[args.record_index]
    target_sequence = str(source_record["seqs"][-1])
    target_frame = int(source_record["frames"][-1])
    if target_frame < 2:
        raise ValueError(f"Need two past frames for parity audit, got {target_frame}")
    record = {
        "angle_bucket": "no_event",
        "clip_type": "cut_event",
        "frames": [target_frame - 2, target_frame - 1, target_frame],
        "group": source_record.get("group", ""),
        "pattern_id": f"reset_parity_{args.source}_{target_sequence}_{target_frame}",
        "seqs": [target_sequence, target_sequence, target_sequence],
        "shot_labels": [0, 0, 0],
        "transition_angles_deg": [0.0, 0.0, 0.0],
        "view_angle_deg": 0.0,
    }
    dataset = AvatarReX_Pattern(
        allow_repeat=True,
        split=split,
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=512,
        resize_mode="human3r_demo",
        num_views=3,
        seed=14991,
        n_corres=0,
        fixed_samples=[record],
        load_da3_depth=False,
        raw_calibration_root=(
            raw_calibration_roots() if args.source in ("avatarrex", "thuman") else None
        ),
        max_humans=1,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    add_mhmr_inputs(batch)
    return batch, record


def run_model(
    checkpoint: Path,
    gt_views: list[dict],
    device: torch.device,
    configure_event_only: bool,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    flags = configure_model(model) if configure_event_only else {"raw_human3r": True}
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    clean = todevice(model_batch_from_gt(copy.deepcopy(gt_views)), device)
    views = set_event_indices(clean, set())
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        predictions, _ = model.forward_recurrent_lighter(
            views, str(device), ret_state=False, use_ttt3r=False
        )
    selected = []
    for prediction in predictions:
        selected.append(
            {
                key: prediction[key].detach().float().cpu()
                for key in COMPARE_KEYS
                if key in prediction and torch.is_tensor(prediction[key])
            }
        )
    del predictions, views, clean, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected, flags


def compare(
    event_outputs: list[dict[str, torch.Tensor]],
    raw_outputs: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if len(event_outputs) != len(raw_outputs):
        raise RuntimeError("Output frame counts differ")
    rows = []
    for frame, (event, raw) in enumerate(zip(event_outputs, raw_outputs)):
        shared = sorted(set(event) & set(raw))
        missing = sorted(set(raw) - set(event))
        for key in shared:
            if event[key].shape != raw[key].shape:
                rows.append(
                    {
                        "frame": frame,
                        "key": key,
                        "status": "shape_mismatch",
                        "event_shape": list(event[key].shape),
                        "raw_shape": list(raw[key].shape),
                    }
                )
                continue
            difference = (event[key] - raw[key]).abs()
            finite = torch.isfinite(difference)
            rows.append(
                {
                    "frame": frame,
                    "key": key,
                    "status": "ok",
                    "count": int(finite.sum()),
                    "max_abs": float(difference[finite].max()) if finite.any() else float("nan"),
                    "mean_abs": float(difference[finite].mean()) if finite.any() else float("nan"),
                }
            )
        for key in missing:
            rows.append({"frame": frame, "key": key, "status": "missing_in_event"})
    valid = [row for row in rows if row["status"] == "ok"]
    return {
        "rows": rows,
        "max_abs": max(row["max_abs"] for row in valid),
        "mean_abs": float(np.mean([row["mean_abs"] for row in valid])),
        "all_shapes_match": all(row["status"] == "ok" for row in rows),
    }


def main() -> None:
    args = parse_args()
    for path in (args.event_model, args.human3r_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_views, record = make_no_event_batch(args)
    event_outputs, event_flags = run_model(
        args.event_model, gt_views, device, configure_event_only=True
    )
    raw_outputs, raw_flags = run_model(
        args.human3r_model, gt_views, device, configure_event_only=False
    )
    result = compare(event_outputs, raw_outputs)
    report = {
        "experiment": "v14_cut_first_reset_only_parity",
        "event_model": str(args.event_model),
        "human3r_model": str(args.human3r_model),
        "record": record,
        "event_flags": event_flags,
        "raw_flags": raw_flags,
        "tolerance": args.atol,
        "passed": bool(result["all_shapes_match"] and result["max_abs"] <= args.atol),
        "comparison": result,
    }
    path = args.output_dir / "reset_only_parity.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "tolerance")}, indent=2))
    print(json.dumps({key: result[key] for key in ("max_abs", "mean_abs", "all_shapes_match")}, indent=2))
    print(path)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
