#!/usr/bin/env python3
"""Visualize selected V8.4 mixed AABB/AAAA manifest clips as contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=Path("output/v8_4_mixed_aabb_aaaa_manifests"),
    )
    parser.add_argument(
        "--training_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Training"),
    )
    parser.add_argument(
        "--test_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Test/v8_4_mixed_aabb_aaaa"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/v8_4_mixed_aabb_aaaa_visual_samples"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_by_bucket(records: list[dict], buckets: list[str]) -> list[dict]:
    chosen = []
    used_keys = set()
    for bucket in buckets:
        candidates = [r for r in records if r.get("angle_bucket") == bucket]
        if not candidates:
            continue
        record = candidates[len(candidates) // 2]
        key = json.dumps(record, sort_keys=True)
        if key not in used_keys:
            chosen.append(record)
            used_keys.add(key)
    if len(chosen) >= len(buckets):
        return chosen[: len(buckets)]
    for record in records:
        key = json.dumps(record, sort_keys=True)
        if key in used_keys:
            continue
        chosen.append(record)
        used_keys.add(key)
        if len(chosen) >= len(buckets):
            break
    return chosen


def choose_by_group(records: list[dict], groups: list[str], n: int) -> list[dict]:
    chosen = []
    used_keys = set()
    for group in groups:
        candidates = [r for r in records if r.get("group") == group]
        if not candidates:
            continue
        record = candidates[len(candidates) // 2]
        key = json.dumps(record, sort_keys=True)
        if key not in used_keys:
            chosen.append(record)
            used_keys.add(key)
    if len(chosen) >= n:
        return chosen[:n]
    for record in records:
        key = json.dumps(record, sort_keys=True)
        if key in used_keys:
            continue
        chosen.append(record)
        used_keys.add(key)
        if len(chosen) >= n:
            break
    return chosen


def frame_specs(record: dict) -> list[tuple[str, int]]:
    start = int(record["start_frame"])
    if record["clip_type"] == "aaaa":
        return [(record["seq"], start + i) for i in range(4)]
    return [
        (record["seqA"], start),
        (record["seqA"], start + 1),
        (record["seqB"], start + 2),
        (record["seqB"], start + 3),
    ]


def read_frame(root: Path, seq: str, frame: int, width: int = 320) -> np.ndarray:
    path = root / seq / "rgb" / f"{int(frame):08d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    scale = width / max(w, 1)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def draw_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0)
    y = 19
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 19
    return canvas


def make_sheet(root: Path, record: dict, title: str, output_path: Path) -> None:
    frames = []
    for view_idx, (seq, frame) in enumerate(frame_specs(record)):
        image = read_frame(root, seq, frame)
        lines = [
            f"view {view_idx}: {seq}",
            f"frame {frame:08d}",
        ]
        frames.append(draw_label(image, lines))
    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < max_h:
            pad = np.zeros((max_h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            frame = np.concatenate([frame, pad], axis=0)
        padded.append(frame)
    sheet = np.concatenate(padded, axis=1)
    header = np.zeros((72, sheet.shape[1], 3), dtype=np.uint8)
    angle = float(record.get("view_angle_deg", 0.0))
    text1 = f"{title} | {record['clip_type'].upper()} | group={record.get('group')} | start={record['start_frame']} | angle={angle:.1f}"
    text2 = "AABB: views 0-1 from camera A, views 2-3 from camera B; AAAA: all 4 views from same camera"
    cv2.putText(header, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, text2, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 220, 255), 1, cv2.LINE_AA)
    output = np.concatenate([header, sheet], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_aabb = load_jsonl(args.manifest_root / "train_aabb_60k.jsonl")
    train_aaaa = load_jsonl(args.manifest_root / "train_aaaa_20k.jsonl")
    test_aabb = load_jsonl(args.manifest_root / "test_aabb_2k.jsonl")
    test_aaaa = load_jsonl(args.manifest_root / "test_aaaa_1k.jsonl")

    selections = []
    for i, record in enumerate(choose_by_bucket(train_aabb, ["030_060", "090_120", "150_180"])):
        selections.append(("train_aabb", args.training_root, record, f"train_aabb_{i+1:02d}.png"))
    for i, record in enumerate(choose_by_group(train_aaaa, ["lbn1", "zzr", "thuman00"], 3)):
        selections.append(("train_aaaa", args.training_root, record, f"train_aaaa_{i+1:02d}.png"))
    test_records = choose_by_bucket(test_aabb, ["015_030", "150_180"]) + choose_by_group(
        test_aaaa, ["thuman02"], 1
    )
    for i, record in enumerate(test_records[:3]):
        selections.append(("test_mixed", args.test_root, record, f"test_mixed_{i+1:02d}.png"))

    manifest = []
    for title, root, record, filename in selections:
        output_path = args.output_dir / filename
        make_sheet(root, record, title, output_path)
        item = dict(record)
        item["visualization"] = str(output_path)
        item["source_root"] = str(root)
        manifest.append(item)
        print(output_path)

    (args.output_dir / "selected_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
