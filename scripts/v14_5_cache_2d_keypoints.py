#!/usr/bin/env python3
"""Cache frozen V18 2D keypoints for a holdout with an arbitrary case count.

The detector and person-selection function are imported from the archived V18
implementation.  This wrapper only removes its hard-coded 180-case assertion.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_V18 = ROOT / "archive/20260721/scripts/v18_cache_2d_keypoints.py"

from torch_cache_support import configure_torch_cache  # noqa: E402

configure_torch_cache()


def frozen_select_person():
    spec = importlib.util.spec_from_file_location("v14_5_frozen_v18_keypoints", FROZEN_V18)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen V18 keypoint code from {FROZEN_V18}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_person


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--detection_threshold", type=float, default=0.50)
    parser.add_argument("--expected_cases", type=int, default=60)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(stream_dir: Path, expected: int) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(stream_dir / "v18_stream_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    rows = sorted(rows, key=lambda row: str(row["case_name"]))
    unique = {str(row["case_name"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise RuntimeError(f"Expected {expected} unique stream-cache cases, got {len(rows)}/{len(unique)}")
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Frozen 2D keypoint detector requires CUDA")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("Invalid shard index")
    select_person = frozen_select_person()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / (
        f"v18_keypoints_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    )
    if manifest_path.exists() and not args.overwrite:
        print(f">> exists {manifest_path}")
        return

    cases = load_cases(args.stream_dir, int(args.expected_cases))
    selected = [
        case
        for index, case in enumerate(cases)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(args.device).eval()
    output_rows = []
    for case_index, case in enumerate(selected):
        with np.load(case["cache_path"]) as cache:
            images = [*cache["old_images"], cache["new_image"]]
        results = []
        for start in range(0, len(images), int(args.batch_size)):
            batch = [
                torch.from_numpy(np.asarray(image).copy())
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .to(args.device)
                for image in images[start : start + int(args.batch_size)]
            ]
            with torch.no_grad():
                results.extend(model(batch))
        detections = [
            select_person(row, float(args.detection_threshold)) for row in results
        ]
        keypoints = np.stack([row[0] for row in detections])
        confidence = np.stack([row[1] for row in detections])
        boxes = np.stack([row[2] for row in detections])
        scores = np.asarray([row[3] for row in detections], dtype=np.float32)
        output_path = args.output_dir / "cases" / f"{case['case_name']}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            old_keypoints=keypoints[:-1],
            old_confidence=confidence[:-1],
            old_boxes=boxes[:-1],
            old_scores=scores[:-1],
            new_keypoints=keypoints[-1],
            new_confidence=confidence[-1],
            new_box=boxes[-1],
            new_score=scores[-1],
        )
        output_rows.append(
            {
                "case_name": case["case_name"],
                "source": case["source"],
                "cache_path": str(output_path),
                "new_detection_score": float(scores[-1]),
                "new_valid_keypoints": int(np.sum(confidence[-1] >= 0.30)),
            }
        )
        print(
            f">> [{case_index + 1}/{len(selected)}] {case['case_name']} "
            f"score={scores[-1]:.3f} joints={output_rows[-1]['new_valid_keypoints']}",
            flush=True,
        )

    payload = {
        "experiment": "V14.5 frozen V18 RGB 2D keypoint cache",
        "case_count": len(output_rows),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "model": "torchvision Keypoint R-CNN ResNet50-FPN default weights",
        "frozen_selection_source": str(FROZEN_V18),
        "cases": output_rows,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
