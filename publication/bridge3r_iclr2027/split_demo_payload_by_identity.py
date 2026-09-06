#!/usr/bin/env python3
"""Split a demo.py-compatible payload into one mesh-only payload per identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", required=True)
    return parser.parse_args()


def replace_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def main() -> None:
    args = parse_args()
    source = args.payload.resolve()
    colours = sorted((source / "color").glob("*.png"))
    cameras = sorted((source / "camera").glob("*.npz"))
    meshes = sorted((source / "smpl").glob("*.npz"))
    if not colours or not (len(colours) == len(cameras) == len(meshes)):
        raise ValueError(
            f"Payload mismatch: color={len(colours)}, camera={len(cameras)}, "
            f"smpl={len(meshes)}"
        )

    first_image = cv2.imread(str(colours[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise OSError(f"Could not read {colours[0]}")
    zeros = np.zeros(first_image.shape[:2], dtype=np.float32)

    for identity in args.ids:
        root = args.output.resolve() / f"person_id_{identity}"
        for name in ("color", "depth", "conf", "camera", "smpl"):
            (root / name).mkdir(parents=True, exist_ok=True)
        support = []
        for index, (colour, camera, mesh) in enumerate(zip(colours, cameras, meshes)):
            name = f"{index:06d}"
            replace_link(colour, root / "color" / f"{name}.png")
            replace_link(camera, root / "camera" / f"{name}.npz")
            np.save(root / "depth" / f"{name}.npy", zeros)
            np.save(root / "conf" / f"{name}.npy", zeros)
            with np.load(mesh, allow_pickle=False) as value:
                ids = np.asarray(value["smpl_id"], dtype=np.int64)
                keep = ids == int(identity)
                vertices = np.asarray(value["verts_world"], dtype=np.float32)[keep]
                faces = np.asarray(value["faces"], dtype=np.int32)
            support.append(int(keep.sum()))
            np.savez(
                root / "smpl" / f"{name}.npz",
                smpl_id=ids[keep],
                verts_world=vertices,
                faces=faces,
                msk=np.zeros((1, 1), dtype=np.float32),
            )
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "Bridge3R-isolated-person-demo-payload-v1",
                    "source_payload": str(source),
                    "persistent_identity": int(identity),
                    "frame_support": support,
                    "scene_pointcloud_available": False,
                    "camera_display_intended": False,
                    "human_geometry_modified": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"identity": identity, "payload": str(root), "support": support}))


if __name__ == "__main__":
    main()
