#!/usr/bin/env python3
"""Export separate original-demo payloads for frozen B0 + DA3 boundaries.

The input is a saved raw-reset Human3R payload. Each requested method receives
its own complete output directory in the same format written by
``demo.py::prepare_output``. Pre-cut frames are copied unchanged; one frozen
Boundary is applied to every post-cut camera pose. Camera-local depth and SMPL-X
parameters remain unchanged, preserving the original Human3R reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path(
    "/dev/shm/movie3r_v14_2/multihuman_three_t0900_c0_c3/raw_reset"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/da3_shared_pose_three_dev/"
    "v14_b0_da3_shared_pose.json"
)
DEFAULT_OUTPUT = Path(
    "/data/wangzheng/codex_tmp/b0_da3_visuals/"
    "multihuman_three_t0900_c0_c3_original_demo"
)
DEFAULT_CASE = "three_t0900_c0_c3_k0"
METHODS = ("raw_reset", "b0", "da3_safe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_payload", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--case_json",
        type=Path,
        default=None,
        help="Optional standalone frozen case JSON; bypasses --report/--case_key.",
    )
    parser.add_argument("--case_key", default=DEFAULT_CASE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cut_index", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_case(report_path: Path, case_key: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matches = [row for row in report["cases"] if row["case"]["key"] == case_key]
    if len(matches) != 1:
        raise KeyError(f"Expected one case {case_key!r}, found {len(matches)}")
    return matches[0]


def frame_count(payload: Path) -> int:
    cameras = sorted((payload / "camera").glob("*.npz"))
    if not cameras:
        raise FileNotFoundError(f"No camera payloads under {payload}")
    expected = [f"{index:06d}.npz" for index in range(len(cameras))]
    if [path.name for path in cameras] != expected:
        raise RuntimeError(f"Non-contiguous camera payload under {payload}")
    return len(cameras)


def human_count(payload: Path) -> int:
    path = payload / "smpl" / "000000.npz"
    with np.load(path, allow_pickle=True) as values:
        return int(values["shape"].shape[0])


def replace_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".new")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def transform_cached_vertices(path: Path, boundary: np.ndarray) -> None:
    if not path.is_file():
        return
    with np.load(path, allow_pickle=True) as source:
        if "verts_world" not in source.files:
            return
        values = {key: source[key] for key in source.files}
    vertices = np.asarray(values["verts_world"], dtype=np.float32)
    values["verts_world"] = (
        np.einsum("ij,...j->...i", boundary[:3, :3], vertices)
        + boundary[:3, 3]
    ).astype(np.float32)
    replace_npz(path, values)


def export_method(
    raw_payload: Path,
    destination: Path,
    boundary: np.ndarray,
    cut_index: int,
    overwrite: bool,
) -> int:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
        shutil.rmtree(destination)
    shutil.copytree(raw_payload, destination)
    count = frame_count(destination)
    if not 0 < cut_index < count:
        raise ValueError(f"cut_index={cut_index} is invalid for {count} frames")

    for index in range(cut_index, count):
        camera_path = destination / "camera" / f"{index:06d}.npz"
        with np.load(camera_path) as source:
            values = {key: source[key] for key in source.files}
        values["pose"] = (
            boundary @ np.asarray(values["pose"], dtype=np.float64)
        ).astype(np.float32)
        replace_npz(camera_path, values)
        transform_cached_vertices(
            destination / "smpl" / f"{index:06d}.npz", boundary
        )
    return count


def main() -> None:
    args = parse_args()
    raw_payload = args.raw_payload.resolve()
    output_root = args.output_root.resolve()
    if not raw_payload.is_dir():
        raise FileNotFoundError(raw_payload)
    if str(output_root) in ("/", "/data"):
        raise ValueError(f"Refusing broad output target: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    row = (
        json.loads(args.case_json.read_text(encoding="utf-8"))
        if args.case_json is not None
        else load_case(args.report, args.case_key)
    )
    boundaries = {
        "raw_reset": np.eye(4, dtype=np.float64),
        "b0": np.asarray(row["methods"]["b0"]["boundary"], dtype=np.float64),
        "da3_safe": np.asarray(
            row["methods"]["da3_safe"]["boundary"], dtype=np.float64
        ),
    }
    outputs = {}
    for method in args.methods:
        destination = output_root / method
        count = export_method(
            raw_payload,
            destination,
            boundaries[method],
            int(args.cut_index),
            bool(args.overwrite),
        )
        outputs[method] = str(destination)
        print(f">> {method}: {count} frames -> {destination}", flush=True)

    diagnostics = row.get("fine_diagnostics")
    if diagnostics is None:
        diagnostics = row["proposal_diagnostics"]["da3_safe"]
    case_metadata = row.get("case", row.get("record", {"case_name": row.get("case_name")}))
    manifest = {
        "format": "original Human3R demo.py saved payload",
        "case": case_metadata,
        "cut_index": int(args.cut_index),
        "human_count": human_count(raw_payload),
        "methods": {
            method: {
                "output": outputs[method],
                "boundary": boundaries[method].tolist(),
                "metrics": (
                    None
                    if method == "raw_reset"
                    else row["methods"][method]
                ),
            }
            for method in args.methods
        },
        "da3_gate": diagnostics,
        "source_payload": str(raw_payload),
        "source_report": str(
            (args.case_json if args.case_json is not None else args.report).resolve()
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
