#!/usr/bin/env python3
"""Build a fixed V8.4 pose benchmark from the existing held-out test split.

The benchmark is intentionally sampled from:
  output/v8_4_mixed_aabb_aaaa_manifests_no_zxc/test_*.jsonl
and the explicit test folder:
  /data/wangzheng/iJCV-CODE/data/Test/v8_4_mixed_aabb_aaaa

It also writes a small train-sanity subset. Use the test subset for reporting
generalization and the train-sanity subset only to check whether training can
fit familiar distribution samples.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ANGLE_BUCKETS = ["015_030", "030_060", "060_090", "090_120", "120_150", "150_180"]
GROUPS = ["lbn1", "zzr", "thuman00", "thuman02"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=Path("output/v8_4_mixed_aabb_aaaa_manifests_no_zxc"),
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
        default=Path("output/v8_4_pose_benchmark"),
    )
    parser.add_argument("--test_aabb_per_group_bucket", type=int, default=1)
    parser.add_argument("--test_aaaa_per_group", type=int, default=2)
    parser.add_argument("--train_aabb_per_group_bucket", type=int, default=1)
    parser.add_argument("--train_aaaa_per_group", type=int, default=1)
    parser.add_argument("--sheet_width", type=int, default=280)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def choose_evenly(records: list[dict], n: int) -> list[dict]:
    if n <= 0 or not records:
        return []
    records = sorted(records, key=lambda r: json.dumps(r, sort_keys=True))
    if len(records) <= n:
        return records
    if n == 1:
        return [records[len(records) // 2]]
    indices = np.linspace(0, len(records) - 1, n).round().astype(int).tolist()
    out = []
    used = set()
    for idx in indices:
        idx = int(idx)
        key = json.dumps(records[idx], sort_keys=True)
        if key not in used:
            out.append(records[idx])
            used.add(key)
    for record in records:
        if len(out) >= n:
            break
        key = json.dumps(record, sort_keys=True)
        if key not in used:
            out.append(record)
            used.add(key)
    return out


def select_test_aabb(records: list[dict], per_group_bucket: int) -> list[dict]:
    selected = []
    for group in GROUPS:
        for bucket in ANGLE_BUCKETS:
            candidates = [
                r for r in records
                if r.get("group") == group and r.get("angle_bucket") == bucket
            ]
            selected.extend(choose_evenly(candidates, per_group_bucket))
    return selected


def select_test_aaaa(records: list[dict], per_group: int) -> list[dict]:
    selected = []
    for group in GROUPS:
        candidates = [r for r in records if r.get("group") == group]
        selected.extend(choose_evenly(candidates, per_group))
    return selected


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


def read_frame(root: Path, seq: str, frame: int, width: int) -> np.ndarray:
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
        cv2.putText(canvas, line[:60], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 19
    return canvas


def make_sheet(root: Path, record: dict, title: str, output_path: Path, width: int) -> None:
    frames = []
    for view_idx, (seq, frame) in enumerate(frame_specs(record)):
        image = read_frame(root, seq, frame, width)
        frames.append(draw_label(image, [f"view {view_idx}: {seq}", f"frame {frame:08d}"]))
    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < max_h:
            pad = np.zeros((max_h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            frame = np.concatenate([frame, pad], axis=0)
        padded.append(frame)
    sheet = np.concatenate(padded, axis=1)
    header = np.zeros((76, sheet.shape[1], 3), dtype=np.uint8)
    angle = float(record.get("view_angle_deg", 0.0))
    line1 = f"{title} | {record['clip_type'].upper()} | group={record.get('group')} | bucket={record.get('angle_bucket')}"
    line2 = f"start={record['start_frame']} | angle={angle:.1f}"
    cv2.putText(header, line1, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, line2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 220, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate([header, sheet], axis=0))


def summarize(records: list[dict]) -> dict:
    by_group = defaultdict(int)
    by_bucket = defaultdict(int)
    for record in records:
        by_group[str(record.get("group", ""))] += 1
        by_bucket[str(record.get("angle_bucket", ""))] += 1
    return {
        "count": len(records),
        "by_group": dict(sorted(by_group.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
    }


def annotate(records: list[dict], subset: str, root_key: str) -> list[dict]:
    out = []
    for idx, record in enumerate(records):
        item = dict(record)
        item["benchmark_subset"] = subset
        item["benchmark_index"] = idx
        item["source_root_key"] = root_key
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.output_dir / "sheets", ignore_errors=True)
    for filename in [
        "test_aabb.jsonl",
        "test_aaaa.jsonl",
        "train_sanity_aabb.jsonl",
        "train_sanity_aaaa.jsonl",
        "all_benchmark.jsonl",
        "benchmark_manifest.json",
    ]:
        path = args.output_dir / filename
        if path.exists():
            path.unlink()

    train_aabb = load_jsonl(args.manifest_root / "train_aabb_no_zxc.jsonl")
    train_aaaa = load_jsonl(args.manifest_root / "train_aaaa_no_zxc.jsonl")
    test_aabb = load_jsonl(args.manifest_root / "test_aabb_no_zxc.jsonl")
    test_aaaa = load_jsonl(args.manifest_root / "test_aaaa_no_zxc.jsonl")

    subsets = {
        "test_aabb": annotate(
            select_test_aabb(test_aabb, args.test_aabb_per_group_bucket),
            "test_aabb",
            "test",
        ),
        "test_aaaa": annotate(
            select_test_aaaa(test_aaaa, args.test_aaaa_per_group),
            "test_aaaa",
            "test",
        ),
        "train_sanity_aabb": annotate(
            select_test_aabb(train_aabb, args.train_aabb_per_group_bucket),
            "train_sanity_aabb",
            "training",
        ),
        "train_sanity_aaaa": annotate(
            select_test_aaaa(train_aaaa, args.train_aaaa_per_group),
            "train_sanity_aaaa",
            "training",
        ),
    }

    roots = {"test": args.test_root, "training": args.training_root}
    manifest = {
        "manifest_root": str(args.manifest_root),
        "training_root": str(args.training_root),
        "test_root": str(args.test_root),
        "subsets": {},
    }

    all_records = []
    for subset_name, records in subsets.items():
        out_path = args.output_dir / f"{subset_name}.jsonl"
        write_jsonl(out_path, records)
        manifest["subsets"][subset_name] = {
            "path": str(out_path),
            **summarize(records),
        }
        all_records.extend(records)
        for record in records:
            root = roots[record["source_root_key"]]
            filename = f"{record['benchmark_index']:03d}_{subset_name}_{record.get('group')}_{record.get('angle_bucket')}.png"
            sheet_path = args.output_dir / "sheets" / subset_name / filename
            make_sheet(root, record, subset_name, sheet_path, args.sheet_width)
            record["sheet_path"] = str(sheet_path)

    write_jsonl(args.output_dir / "all_benchmark.jsonl", all_records)
    manifest["all_benchmark"] = str(args.output_dir / "all_benchmark.jsonl")
    manifest["summary"] = summarize(all_records)
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["subsets"], indent=2, sort_keys=True))
    print(f"Wrote benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
