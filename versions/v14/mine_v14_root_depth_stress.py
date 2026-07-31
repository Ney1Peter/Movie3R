#!/usr/bin/env python3
"""Mine near/far camera-cut stress cases from existing converted datasets."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import (  # noqa: E402
    _load_avatarrex_raw_calibration,
    _raw_calibration_c2w,
)


DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/root_depth_stress_mining"
MANIFESTS = {
    "avatarrex": REPO_ROOT
    / "config/manifests/v9_120h_eval_benchmark_manifests/avatarrex/test_aabb.jsonl",
    "thuman": REPO_ROOT
    / "config/manifests/v9_120h_eval_benchmark_manifests/thuman/test_aabb.jsonl",
    "mvhuman100": REPO_ROOT
    / "config/manifests/v9_4source_baseline_mvhuman100_10k_manifests/test_aabb_2k.jsonl",
    "mvhuman200": REPO_ROOT
    / "config/manifests/v9_4source_baseline_mvhuman200_10k_manifests/test_aabb_2k.jsonl",
}
AVATAR_GROUPS = ("lbn1", "lbn2", "zzr", "zxc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_n", type=int, default=8)
    return parser.parse_args()


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def sequence_root(data_root: Path, group: str, sequence: str) -> Path:
    if group[:1].isdigit():
        return data_root / "Training/mvhuman" / sequence
    return data_root / "Training" / sequence


def frame_payload(
    data_root: Path,
    calibration: dict,
    group: str,
    sequence: str,
    frame: int,
) -> dict:
    root = sequence_root(data_root, group, sequence)
    stem = f"{int(frame):08d}"
    with (root / "smpl" / f"{stem}.pkl").open("rb") as handle:
        annotations = pickle.load(handle)
    if not annotations:
        raise ValueError("empty_smpl")
    translation = np.asarray(annotations[0]["smplx_transl"], dtype=np.float64)
    if group in AVATAR_GROUPS:
        camera = _raw_calibration_c2w(calibration, sequence)
    else:
        camera = np.asarray(
            np.load(root / "cam" / f"{stem}.npz")["pose"], dtype=np.float64
        )
    local = np.linalg.inv(camera) @ np.r_[translation, 1.0]
    mask = cv2.imread(str(root / "mask" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    occupancy = float(np.mean(mask > 25)) if mask is not None else float("nan")
    return {
        "sequence": sequence,
        "frame": int(frame),
        "distance_m": float(np.linalg.norm(local[:3])),
        "depth_m": float(local[2]),
        "occupancy": occupancy,
    }


def ratio(first: float, second: float) -> float:
    if not np.isfinite(first) or not np.isfinite(second):
        return float("nan")
    return float(max(first / max(second, 1e-9), second / max(first, 1e-9)))


def scan_source(
    source: str,
    manifest: Path,
    args: argparse.Namespace,
    calibration: dict,
) -> tuple[list[dict], dict]:
    rows = []
    skipped = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        group = str(record.get("group", str(record["seqA"]).split("/", 1)[0]))
        start = int(record["start_frame"])
        try:
            pre = frame_payload(
                args.data_root, calibration, group, str(record["seqA"]), start + 1
            )
            post = frame_payload(
                args.data_root, calibration, group, str(record["seqB"]), start + 2
            )
        except (FileNotFoundError, KeyError, ValueError, OSError):
            skipped += 1
            continue
        distance_ratio = ratio(pre["distance_m"], post["distance_m"])
        occupancy_ratio = ratio(pre["occupancy"], post["occupancy"])
        score = max(distance_ratio, math_sqrt(occupancy_ratio))
        rows.append(
            {
                "source": source,
                "group": group,
                "seqA": str(record["seqA"]),
                "seqB": str(record["seqB"]),
                "start_frame": start,
                "view_angle_deg": float(record.get("view_angle_deg", float("nan"))),
                "pre": pre,
                "post": post,
                "distance_ratio": distance_ratio,
                "occupancy_ratio": occupancy_ratio,
                "stress_score": score,
            }
        )
    summary = {
        "manifest": str(manifest),
        "valid_records": len(rows),
        "skipped_records": skipped,
        "distance_ratio": distribution([row["distance_ratio"] for row in rows]),
        "occupancy_ratio": distribution([row["occupancy_ratio"] for row in rows]),
    }
    return rows, summary


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value)) if np.isfinite(value) and value >= 0.0 else 0.0


def selected_rows(rows: list[dict], count: int) -> list[dict]:
    selected = []
    seen = set()
    rankings = (
        sorted(rows, key=lambda row: row["distance_ratio"], reverse=True),
        sorted(rows, key=lambda row: row["occupancy_ratio"], reverse=True),
        sorted(rows, key=lambda row: row["stress_score"], reverse=True),
    )
    for ranking in rankings:
        for row in ranking:
            key = (row["seqA"], row["seqB"], int(row["start_frame"]))
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= int(count):
                return selected
    return selected


def markdown(report: dict) -> str:
    lines = [
        "# V14 Root-Depth Stress Mining",
        "",
        "GT is used only to select near/far and apparent-size stress cases.",
        "The cut is the AABB transition from pre index 1 to post index 2.",
        "",
        "| Source | Records | Distance ratio median/P95/max | Occupancy ratio median/P95/max |",
        "|---|---:|---:|---:|",
    ]
    for source, summary in report["summary"].items():
        distance = summary["distance_ratio"]
        occupancy = summary["occupancy_ratio"]
        lines.append(
            f"| {source} | {summary['valid_records']} | "
            f"{distance['median']:.3f}/{distance['p95']:.3f}/{distance['maximum']:.3f} | "
            f"{occupancy['median']:.3f}/{occupancy['p95']:.3f}/{occupancy['maximum']:.3f} |"
        )
    lines.extend(["", "## Selected", ""])
    for source, rows in report["selected"].items():
        lines.append(f"### {source}")
        lines.append("")
        for row in rows:
            lines.append(
                f"- `{row['seqA']} -> {row['seqB']} @ {row['start_frame']}`: "
                f"distance ratio `{row['distance_ratio']:.3f}`, occupancy ratio "
                f"`{row['occupancy_ratio']:.3f}`, view `{row['view_angle_deg']:.1f} deg`."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _load_avatarrex_raw_calibration(
        {
            group: str(args.data_root / "AvatarReX_raw_meta" / group)
            for group in AVATAR_GROUPS
        }
    )
    all_rows = {}
    summaries = {}
    selected = {}
    for source, manifest in MANIFESTS.items():
        rows, summary = scan_source(source, manifest, args, calibration)
        all_rows[source] = rows
        summaries[source] = summary
        selected[source] = selected_rows(rows, int(args.top_n))
        print(
            f">> {source}: {len(rows)} valid, "
            f"max distance ratio={summary['distance_ratio']['maximum']:.3f}",
            flush=True,
        )
    report = {
        "experiment": "v14_root_depth_stress_mining",
        "protocol": {
            "gt_candidate_generation": "selection only",
            "sources": list(MANIFESTS),
            "top_n_per_source": int(args.top_n),
        },
        "summary": summaries,
        "selected": selected,
        "rows": all_rows,
    }
    json_path = args.output_dir / "root_depth_stress_mining.json"
    md_path = args.output_dir / "README.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f">> wrote {json_path}", flush=True)
    print(f">> wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
