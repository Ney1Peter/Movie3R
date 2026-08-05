#!/usr/bin/env python3
"""Front shot-detector + adaptive shared human--camera correction.

This is the event-first ablation of the Movie3R boundary transaction.  A
causal RGB-only detector proposes cut indices from adjacent input images.  At
each proposal, predicted human geometry supplies a second confidence gate;
only accepted proposals trigger the shared post-cut SE(3).  Thus an image
false positive is harmless when the human residual is small or ambiguous.

The script is CPU-only and consumes an already saved baseline payload.  It is
also the reference implementation used by ``demo.py`` when
``--adaptive_joint_mode detector`` is selected.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.dust3r.adaptive_joint import AdaptiveJointConfig, apply_to_arrays, apply_with_raw_reference
from versions.v14.adaptive_post_human_boundary import (
    frame_count,
    load_camera,
    load_mesh,
    replace_npz,
)


DEFAULT_FEATURE_CSV = Path(
    "output/archive/20260721/v10_detector_probe/image_feature_round1/"
    "detector_pair_features.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--raw-source", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--images", type=Path, nargs="*", default=None)
    p.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    p.add_argument("--detector-threshold", type=float, default=0.5)
    p.add_argument("--min-rotation-deg", type=float, default=20.0)
    p.add_argument("--max-vertex-rms-m", type=float, default=0.20)
    p.add_argument("--max-normalized-rms", type=float, default=0.20)
    p.add_argument("--min-permutation-margin-m", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        images = [Path(value).resolve() for value in args.images]
    elif args.manifest is not None:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        images = [Path(value).resolve() for value in data["input_paths"]]
    else:
        raise ValueError("Provide --images or --manifest")
    if len(images) < 2 or any(not path.is_file() for path in images):
        missing = [str(path) for path in images if not path.is_file()]
        raise FileNotFoundError(f"Invalid image sequence; missing={missing[:3]}")
    return images


def detect(images: list[Path], csv_path: Path, threshold: float) -> tuple[list[int], list[dict]]:
    # Keep the legacy detector isolated from the model import path.  It uses
    # only OpenCV/scikit-learn and never touches GPU/model weights.
    import sys

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from v10_image_only_detector import StreamingImageOnlyShotDetector

    detector = StreamingImageOnlyShotDetector(csv_path, threshold=threshold)
    labels, rows = detector.predict_sequence(images)
    return [i for i, label in enumerate(labels) if int(label) == 1], rows


def main() -> None:
    a = parse_args()
    source = a.source.resolve()
    output = a.output.resolve()
    images = read_images(a)
    cut_indices, detector_rows = detect(images, a.feature_csv.resolve(), a.detector_threshold)
    if output.exists():
        if not a.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        if output in {Path("/"), Path("/data"), Path("/data/wangzheng")}:  # safety
            raise ValueError(f"Refusing broad output deletion: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)
    n = frame_count(source)
    if len(images) != n:
        raise ValueError(f"Detector sequence length {len(images)} != payload frame count {n}")
    cameras = np.stack([load_camera(source, i) for i in range(n)], axis=0)
    meshes = [load_mesh(source, i) for i in range(n)]
    cfg = AdaptiveJointConfig(
        min_rotation_deg=a.min_rotation_deg,
        max_vertex_rms_m=a.max_vertex_rms_m,
        max_normalized_rms=a.max_normalized_rms,
        min_permutation_margin_m=a.min_permutation_margin_m,
        alpha=a.alpha,
    )
    raw_cameras = raw_meshes = None
    if a.raw_source is not None:
        raw_source = a.raw_source.resolve()
        raw_n = frame_count(raw_source)
        if raw_n != n:
            raise ValueError(f"Raw payload frame count {raw_n} != source frame count {n}")
        raw_cameras = np.stack([load_camera(raw_source, i) for i in range(n)], axis=0)
        raw_meshes = [load_mesh(raw_source, i) for i in range(n)]
    if raw_cameras is None or raw_meshes is None:
        cameras_new, meshes_new, _, records = apply_to_arrays(cameras, meshes, None, cut_indices, cfg)
    else:
        cameras_new, meshes_new, _, records = apply_with_raw_reference(
            cameras, meshes, raw_cameras, raw_meshes, None, cut_indices, cfg
        )
    for index in range(n):
        cpath = output / "camera" / f"{index:06d}.npz"
        with np.load(cpath) as z:
            values = {key: z[key] for key in z.files}
        values["pose"] = cameras_new[index].astype(np.float32)
        replace_npz(cpath, values)
        spath = output / "smpl" / f"{index:06d}.npz"
        with np.load(spath, allow_pickle=True) as z:
            values = {key: z[key] for key in z.files}
        values["verts_world"] = meshes_new[index].astype(np.float32)
        replace_npz(spath, values)

    diagnostics = {
        "method": "causal_image_detector_then_adaptive_shared_boundary_v1",
        "source": str(source),
        "raw_source": str(a.raw_source.resolve()) if a.raw_source is not None else None,
        "output": str(output),
        "frame_count": n,
        "detector": {
            "feature_csv": str(a.feature_csv.resolve()),
            "threshold": float(a.detector_threshold),
            "predicted_cut_indices": cut_indices,
            "pairs": detector_rows,
        },
        "geometry_gate": {
            "min_rotation_deg": float(a.min_rotation_deg),
            "max_vertex_rms_m": float(a.max_vertex_rms_m),
            "max_normalized_rms": float(a.max_normalized_rms),
            "min_permutation_margin_m": float(a.min_permutation_margin_m),
            "alpha": float(a.alpha),
        },
        "runtime_contract": "RGB-only causal proposal followed by GT-free human gate; pre frames unchanged",
        "records": records,
    }
    (output / "adaptive_joint_boundary.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
