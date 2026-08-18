#!/usr/bin/env python3
"""Build a compact, deterministic capture index for one extracted archive.

This stage reads filenames and required coordinate metadata only.  It never
uses a Movie3R prediction.  The SHA-256 rank provides a reproducible capture
order before any GPU forward or test metric is observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.dataset import frame_numbers  # noqa: E402
from versions.v15.harmony4d.protocol import PROTOCOL_SEED  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--archive-entry", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=150)
    parser.add_argument("--select-count", type=int, default=1)
    return parser.parse_args()


def capture_roots(extracted_root: Path) -> list[Path]:
    return sorted(
        path.parent.parent.parent
        for path in extracted_root.rglob("colmap/workplace/cameras.txt")
    )


def main() -> None:
    args = parse_args()
    root = args.extracted_root.resolve()
    if args.clip_length < 2 or args.select_count < 1:
        raise ValueError("clip length and selection count must be positive")
    rows = []
    for capture in capture_roots(root):
        relative = str(capture.relative_to(root))
        transform = capture / "colmap/workplace/aria_from_colmap_transforms.pkl"
        has_pnp_annotations = bool(
            list((capture / "processed_data/smpl").glob("*.npy"))
            and list((capture / "processed_data/poses2d").glob("cam*/*.npy"))
        )
        reasons = []
        frames: list[int] = []
        if not transform.is_file() and not has_pnp_annotations:
            reasons.append("missing_coordinate_transform_and_pnp_annotations")
        try:
            frames = frame_numbers(capture)
        except Exception as error:  # compact audit must preserve every failure
            reasons.append(f"frame_index_error:{type(error).__name__}:{error}")
        if len(frames) < args.clip_length:
            reasons.append("fewer_than_clip_length_synchronized_frames")
        contiguous = bool(frames) and frames == list(range(min(frames), max(frames) + 1))
        if frames and not contiguous:
            reasons.append("noncontiguous_synchronized_frames")
        rank = hashlib.sha256(
            f"{PROTOCOL_SEED}:{args.archive_entry}:{relative}".encode("utf-8")
        ).hexdigest()
        rows.append({
            "capture_relative": relative,
            "capture_name": capture.name,
            "frame_count": len(frames),
            "frame_min": min(frames) if frames else None,
            "frame_max": max(frames) if frames else None,
            "frames_contiguous": contiguous,
            "exo_camera_count": len(list((capture / "exo").glob("cam*"))),
            "has_coordinate_transform": transform.is_file(),
            "has_pnp_annotations": has_pnp_annotations,
            "coordinate_adapter": (
                "colmap_plus_aria_similarity_with_reprojection_guard"
                if transform.is_file()
                else "published_smpl45_to_poses2d45_static_pnp"
            ),
            "eligible_structural": not reasons,
            "exclusion_reasons": reasons,
            "selection_rank_sha256": rank,
        })
    eligible = sorted(
        (row for row in rows if row["eligible_structural"]),
        key=lambda row: (row["selection_rank_sha256"], row["capture_relative"]),
    )
    selected = [row["capture_relative"] for row in eligible[: args.select_count]]
    report = {
        "schema_version": "Harmony4D-Movie3R-archive-index-v1",
        "archive_entry": args.archive_entry,
        "extracted_root": str(root),
        "protocol_seed": PROTOCOL_SEED,
        "clip_length": args.clip_length,
        "selection_rule": "lowest SHA256(seed:archive_entry:capture_relative) among structurally eligible captures",
        "capture_count": len(rows),
        "eligible_count": len(eligible),
        "selected_capture_relatives": selected,
        "eligible_ranked": eligible,
        "all_captures": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "capture_count": len(rows),
        "eligible_count": len(eligible),
        "selected": selected,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
