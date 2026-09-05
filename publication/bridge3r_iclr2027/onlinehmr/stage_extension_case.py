#!/usr/bin/env python3
"""Materialize one frozen extension case as OnlineHMR's exact JPEG timeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_FRAMES = 150


def read_row(path: Path, line_number: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(line_number)
    return rows[line_number - 1]


def validate(path: Path) -> None:
    images = sorted(path.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(EXPECTED_FRAMES)]
    if [item.name for item in images] != expected or any(item.stat().st_size <= 0 for item in images):
        raise ValueError(f"invalid 150-frame JPEG timeline at {path}")


def remove_partial(path: Path, root: Path) -> None:
    if not path.exists():
        return
    resolved, parent = path.resolve(), root.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"unsafe partial stage path: {resolved}")
    shutil.rmtree(resolved)


def harmony_source(root: Path, row: dict[str, Any], evaluator: dict[str, Any]) -> list[Path]:
    capture = root / f"train_{evaluator['sequence']}" / str(evaluator["capture_relative"])
    output = []
    for camera, frames in zip(evaluator["shot_cameras"], evaluator["shot_frame_numbers"]):
        for frame in frames:
            directory = capture / "exo" / str(camera) / "images"
            candidates = [directory / f"{int(frame):05d}.jpg", directory / f"{int(frame):06d}.jpg"]
            matches = [path for path in candidates if path.is_file()]
            if len(matches) != 1:
                raise FileNotFoundError(candidates[0])
            output.append(matches[0])
    if len(output) != int(row["clip_length"]):
        raise ValueError("Harmony4D evaluator timeline differs from runtime length")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    row = read_row(args.runtime_manifest.resolve(), int(args.line))
    evaluator = read_row(args.evaluator_manifest.resolve(), int(args.line))
    if row["case_id"] != evaluator["case_id"] or row.get("runtime_gt_access") is not False:
        raise ValueError("runtime/evaluator identity or prediction-only contract mismatch")
    root = args.output_root.resolve()
    final = root / str(row["case_id"]) / "images"
    if final.is_dir():
        validate(final)
        print(json.dumps({"status": "reused", "case_id": row["case_id"], "images": str(final)}))
        return
    partial_case = root / (str(row["case_id"]) + ".partial")
    remove_partial(partial_case, root)
    partial = partial_case / "images"
    partial.mkdir(parents=True)
    dataset = str(row["dataset"])
    source_root = args.source_root.resolve()
    if dataset == "AIST++":
        video = source_root / str(row["input_video"])
        if not video.is_file():
            raise FileNotFoundError(video)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-vsync", "0", "-q:v", "2", "-start_number", "0", str(partial / "%06d.jpg")],
            check=True,
        )
    elif dataset == "MVHuman":
        video = source_root / "derived" / str(row["input_video"])
        if not video.is_file():
            raise FileNotFoundError(video)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
             "-vsync", "0", "-q:v", "2", "-start_number", "0", str(partial / "%06d.jpg")],
            check=True,
        )
    elif dataset == "Harmony4D":
        for index, source in enumerate(harmony_source(source_root, row, evaluator)):
            shutil.copy2(source, partial / f"{index:06d}.jpg")
    else:
        raise ValueError(f"unsupported extension dataset {dataset}")
    validate(partial)
    final.parent.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_case, final.parent)
    print(json.dumps({"status": "staged", "case_id": row["case_id"], "images": str(final)}))


if __name__ == "__main__":
    main()
