#!/usr/bin/env python3
"""Run the frozen 2D keypoint detector on V20 five-plus-three frame caches."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAM = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "stream3_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "keypoint3_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--detection_threshold", type=float, default=0.50)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(root: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v20_stream3_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    rows = sorted(rows, key=lambda row: str(row["case_name"]))
    if len(rows) != 180 or len({row["case_name"] for row in rows}) != 180:
        raise RuntimeError(f"Expected 180 V20 stream cases, got {len(rows)}")
    return rows


def select_person(output: dict, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = output["scores"].detach().float().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()
    keypoints = output["keypoints"].detach().float().cpu().numpy()
    keypoint_scores = output.get("keypoints_scores")
    if keypoint_scores is not None:
        keypoint_scores = keypoint_scores.detach().float().cpu().numpy()
    valid = np.flatnonzero(scores >= float(threshold))
    if not len(valid):
        valid = np.arange(min(len(scores), 1))
    if not len(valid):
        return (
            np.full((22, 2), np.nan, dtype=np.float32),
            np.zeros(22, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            0.0,
        )
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    index = int(valid[np.argmax(scores[valid] * np.sqrt(np.maximum(area[valid], 1.0)))])
    coco = keypoints[index, :, :2]
    confidence = keypoints[index, :, 2]
    if keypoint_scores is not None:
        confidence = 1.0 / (1.0 + np.exp(-keypoint_scores[index]))
    joints = np.full((22, 2), np.nan, dtype=np.float32)
    conf = np.zeros(22, dtype=np.float32)
    mapping = {0: 15, 5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
    for coco_index, smpl_index in mapping.items():
        joints[smpl_index] = coco[coco_index]
        conf[smpl_index] = confidence[coco_index]
    for smpl_index, pair in ((0, (1, 2)), (12, (16, 17))):
        if np.isfinite(joints[list(pair)]).all():
            joints[smpl_index] = joints[list(pair)].mean(axis=0)
            conf[smpl_index] = float(min(conf[pair[0]], conf[pair[1]]))
    return joints, conf, boxes[index].astype(np.float32), float(scores[index])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V20 keypoint cache requires CUDA")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"v20_keypoints3_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if manifest.exists() and not args.overwrite:
        print(f">> exists {manifest}")
        return
    cases = load_cases(args.stream_dir)
    selected = [
        case
        for index, case in enumerate(cases)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(
        args.device
    ).eval()
    rows = []
    for case_index, case in enumerate(selected):
        with np.load(case["cache_path"]) as cache:
            images = [*cache["old_images"], *cache["new_images"]]
        detections = []
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
                detections.extend(model(batch))
        selected_people = [select_person(row, float(args.detection_threshold)) for row in detections]
        keypoints = np.stack([row[0] for row in selected_people])
        confidence = np.stack([row[1] for row in selected_people])
        boxes = np.stack([row[2] for row in selected_people])
        scores = np.asarray([row[3] for row in selected_people], dtype=np.float32)
        output = args.output_dir / "cases" / f"{case['case_name']}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            old_keypoints=keypoints[:5],
            old_confidence=confidence[:5],
            old_boxes=boxes[:5],
            old_scores=scores[:5],
            new_keypoints=keypoints[5:],
            new_confidence=confidence[5:],
            new_boxes=boxes[5:],
            new_scores=scores[5:],
        )
        rows.append(
            {
                "case_name": case["case_name"],
                "source": case["source"],
                "cache_path": str(output),
                "new_detection_scores": scores[5:].astype(float).tolist(),
                "new_valid_keypoints": [int(np.sum(row >= 0.30)) for row in confidence[5:]],
            }
        )
        print(f">> [{case_index + 1}/{len(selected)}] {case['case_name']}", flush=True)
    payload = {
        "experiment": "V20 frozen 2D keypoints for five-plus-three frames",
        "case_count": len(rows),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "cases": rows,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
