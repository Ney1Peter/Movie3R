#!/usr/bin/env python3
"""Build a V7 Stage-A input manifest from refined 30-frame clip manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined_manifest", type=Path, action="append", required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--split_name", action="append", default=None)
    parser.add_argument("--require_existing_videos", action="store_true")
    parser.add_argument("--boundary", type=int, default=None, help="Override local boundary frame for every case.")
    return parser.parse_args()


def infer_split_name(path: Path, manifest: dict) -> str:
    value = manifest.get("split_name")
    if value:
        return str(value)
    name = path.name
    suffix = "_manifest.json"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def load_cases(path: Path, split_name: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    manifest = json.loads(path.read_text())
    cases = []
    skipped = []
    for record in manifest.get("accepted", []):
        video = Path(record["output_path"])
        if args.require_existing_videos and not video.is_file():
            skipped.append({"split_name": split_name, "source_video": str(video), "reason": "missing_video"})
            continue
        boundary = int(args.boundary) if args.boundary is not None else int(record.get("refined_boundary_local_frame", record.get("pre_count", 10)))
        cases.append(
            {
                "name": video.stem,
                "split_name": split_name,
                "source_video": str(video),
                "boundary": boundary,
                "target_frames": list(range(boundary, boundary + 3)),
                "refined_manifest": str(path),
                "source_clip": record.get("source_clip"),
                "preview_path": record.get("preview_path"),
                "jump_absdiff": record.get("jump_absdiff"),
                "jump_ratio": record.get("jump_ratio"),
                "person_status": record.get("person_status"),
            }
        )
    return cases, skipped


def main() -> None:
    args = parse_args()
    if args.split_name is not None and len(args.split_name) != len(args.refined_manifest):
        raise ValueError("--split_name count must match --refined_manifest count when provided")

    all_cases = []
    skipped = []
    inputs = []
    for idx, path in enumerate(args.refined_manifest):
        manifest = json.loads(path.read_text())
        split_name = args.split_name[idx] if args.split_name is not None else infer_split_name(path, manifest)
        cases, skipped_here = load_cases(path, split_name, args)
        all_cases.extend(cases)
        skipped.extend(skipped_here)
        inputs.append({"refined_manifest": str(path), "split_name": split_name, "num_cases": len(cases)})

    all_cases.sort(key=lambda item: (item["split_name"], item["name"]))
    manifest = {
        "description": "V7 Stage-A input manifest built from refined MS-AIST boundary clips.",
        "inputs": inputs,
        "num_cases": len(all_cases),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "cases": all_cases,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_manifest": str(args.output_manifest), "num_cases": len(all_cases), "num_skipped": len(skipped)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
