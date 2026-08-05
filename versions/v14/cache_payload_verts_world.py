#!/usr/bin/env python3
"""Add demo-compatible world-space SMPL-X vertices to a saved payload."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dust3r.utils.geometry import geotrf
from dust3r.utils.smpl_layer import SMPL_Layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    payload = args.payload.resolve()
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").eval()
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    for camera_path in sorted((payload / "camera").glob("*.npz")):
        index = int(camera_path.stem)
        smpl_path = payload / "smpl" / f"{index:06d}.npz"
        with np.load(smpl_path, allow_pickle=True) as source:
            values = {key: source[key] for key in source.files}
        with np.load(camera_path) as camera:
            pose = torch.from_numpy(np.asarray(camera["pose"], dtype=np.float32))
            intrinsics = torch.from_numpy(np.asarray(camera["intrinsics"], dtype=np.float32))
        pose_params = torch.from_numpy(np.asarray(values["rotvec"], dtype=np.float32))
        shape = torch.from_numpy(np.asarray(values["shape"], dtype=np.float32))
        transl = torch.from_numpy(np.asarray(values["transl"], dtype=np.float32))
        expression = values.get("expression")
        expression_tensor = None if expression is None else torch.from_numpy(np.asarray(expression, dtype=np.float32))
        with torch.no_grad():
            output = layer(
                pose_params,
                shape,
                transl,
                None,
                None,
                intrinsics.unsqueeze(0) if intrinsics.ndim == 2 else intrinsics,
                expression=expression_tensor,
            )
        vertices_world = geotrf(pose.unsqueeze(0), output["smpl_v3d"].unsqueeze(0))[0]
        values["verts_world"] = vertices_world.cpu().numpy().astype(np.float32)
        values["faces"] = faces
        np.savez(smpl_path, **values)
        print(index, values["verts_world"].shape)


if __name__ == "__main__":
    main()
