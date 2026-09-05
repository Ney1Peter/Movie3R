#!/usr/bin/env python3
"""Freeze prediction-only OnlineHMR manifests for the extension protocols."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE_ROOT = WORKSPACE / "Movie3R"
DATA_ROOT = WORKSPACE / "data"
IMAGE_MEMBERS = [f"images/{index:06d}.jpg" for index in range(150)]

PROTOCOLS = {
    "aist_cs150": {
        "runtime": DATA_ROOT / "bridge3r_singleperson_v1/manifests/aist_cs150_test.runtime.jsonl",
        "evaluator": DATA_ROOT / "bridge3r_singleperson_v1/manifests/aist_cs150_test.evaluator.jsonl",
        "source_root": DATA_ROOT / "bridge3r_singleperson_v1",
        "count": 100,
    },
    "aist_mc150_3": {
        "runtime": DATA_ROOT / "bridge3r_singleperson_v1/manifests/aist_mc150-3_test.runtime.jsonl",
        "evaluator": DATA_ROOT / "bridge3r_singleperson_v1/manifests/aist_mc150-3_test.evaluator.jsonl",
        "source_root": DATA_ROOT / "bridge3r_singleperson_v1",
        "count": 100,
    },
    "aist_mc150_4": {
        "runtime": DATA_ROOT / "bridge3r_singleperson_v1/manifests/aist_mc150-4_test.runtime.jsonl",
        "evaluator": DATA_ROOT / "bridge3r_singleperson_v1/manifests/aist_mc150-4_test.evaluator.jsonl",
        "source_root": DATA_ROOT / "bridge3r_singleperson_v1",
        "count": 100,
    },
    "mvhuman_mvh150": {
        "runtime": MOVIE_ROOT / "output/bridge3r_mvhuman_v1/protocol_freeze/manifests/test_runtime.jsonl",
        "evaluator": MOVIE_ROOT / "output/bridge3r_mvhuman_v1/protocol_freeze/manifests/test_evaluator.jsonl",
        "source_root": MOVIE_ROOT / "output/bridge3r_mvhuman_v1/protocol_freeze",
        "count": 50,
    },
    "harmony4d_multicut": {
        "runtime": None,
        "evaluator": MOVIE_ROOT / "publication/bridge3r_iclr2027/multicut/manifests/harmony4d_multicut_v1.jsonl",
        "source_root": DATA_ROOT / "Bridge3R_multicut_harmony4d/staging",
        "count": 4,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def runtime_row(source: dict[str, Any]) -> dict[str, Any]:
    """Retain only fields available before inference, plus exact RGB names."""

    frames = int(source.get("num_frames", source.get("clip_length", -1)))
    if frames != 150:
        raise ValueError(f"expected 150 frames for {source.get('case_id')}, found {frames}")
    return {
        "case_id": str(source["case_id"]),
        "dataset": str(source["dataset"]),
        "protocol": str(source["protocol"]),
        "split": str(source.get("role", source.get("split", "test"))),
        "role": str(source.get("role", source.get("split", "test"))),
        "fps": float(source["fps"]),
        "num_frames": frames,
        "clip_length": frames,
        "input_video": source.get("input_video"),
        "image_members": IMAGE_MEMBERS,
        "runtime_gt_access": False,
    }


def validate_source(protocol: str, row: dict[str, Any], evaluator: dict[str, Any], root: Path) -> None:
    if protocol.startswith("aist_"):
        video = root / str(row["input_video"])
        label = root / str(evaluator["label"])
        if not video.is_file() or not label.is_file():
            raise FileNotFoundError(video if not video.is_file() else label)
    elif protocol == "mvhuman_mvh150":
        video = root / "derived" / str(row["input_video"])
        if not video.is_file():
            raise FileNotFoundError(video)
    elif protocol == "harmony4d_multicut":
        capture = root / f"train_{evaluator['sequence']}" / str(evaluator["capture_relative"])
        for camera, frames in zip(evaluator["shot_cameras"], evaluator["shot_frame_numbers"]):
            for frame in (frames[0], frames[-1]):
                path = capture / "exo" / str(camera) / "images" / f"{int(frame):05d}.jpg"
                if not path.is_file():
                    raise FileNotFoundError(path)
    else:
        raise ValueError(protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    inventory: dict[str, Any] = {
        "schema_version": "Bridge3R-OnlineHMR-extension-manifests-v1",
        "runtime_contract": {
            "rgb_only": True,
            "ordered_jpeg_names": "000000.jpg...000149.jpg",
            "gt_camera_identity_count_cut_mask_depth": False,
        },
        "protocols": {},
    }
    for name, spec in PROTOCOLS.items():
        evaluator_path = Path(spec["evaluator"]).resolve()
        evaluator_rows = read_jsonl(evaluator_path)
        source_runtime = Path(spec["runtime"]).resolve() if spec["runtime"] else evaluator_path
        source_rows = read_jsonl(source_runtime)
        if len(source_rows) != int(spec["count"]) or len(evaluator_rows) != int(spec["count"]):
            raise ValueError(f"{name} count mismatch: {len(source_rows)} / {len(evaluator_rows)}")
        if [row["case_id"] for row in source_rows] != [row["case_id"] for row in evaluator_rows]:
            raise ValueError(f"{name} runtime/evaluator case order differs")
        rows = []
        for source, evaluator in zip(source_rows, evaluator_rows):
            if name == "harmony4d_multicut":
                source = {
                    "case_id": evaluator["case_id"],
                    "dataset": "Harmony4D",
                    "protocol": evaluator["protocol"],
                    "role": "test",
                    "fps": evaluator["fps"],
                    "clip_length": evaluator["clip_length"],
                }
            row = runtime_row(source)
            validate_source(name, row, evaluator, Path(spec["source_root"]).resolve())
            rows.append(row)
        runtime_output = output / f"{name}.runtime.jsonl"
        evaluator_output = output / f"{name}.evaluator.jsonl"
        atomic_text(runtime_output, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        atomic_text(evaluator_output, "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluator_rows))
        inventory["protocols"][name] = {
            "case_count": len(rows),
            "runtime_manifest": str(runtime_output),
            "evaluator_manifest": str(evaluator_output),
            "source_root": str(Path(spec["source_root"]).resolve()),
            "prediction_fields": sorted(rows[0]),
            "runtime_gt_access": False,
        }
    atomic_text(output / "inventory.json", json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
