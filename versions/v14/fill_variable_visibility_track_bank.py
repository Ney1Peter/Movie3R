#!/usr/bin/env python3
"""Causal visualization/evaluation ablation for variable human visibility.

When the detector temporarily drops a track, keep its last accepted SMPL mesh
under the same persistent ``smpl_id``.  This does not alter cameras and is
explicitly marked as a hold-last policy; it is useful to distinguish an ID
bank failure from a detector visibility failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(); source, output = a.source.resolve(), a.output.resolve()
    if output.exists():
        if not a.overwrite: raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(source, output)
    first = None; last = {}; filled = []
    for index in range(int(a.boundary), len(list((source / "camera").glob("*.npz")))):
        path = output / "smpl" / f"{index:06d}.npz"
        with np.load(path, allow_pickle=True) as z: values = {key: z[key] for key in z.files}
        ids = np.asarray(values["smpl_id"], dtype=np.int64).reshape(-1)
        verts = np.asarray(values["verts_world"], dtype=np.float32)
        if first is None: first = ids.copy()
        rows = {int(identity): verts[row].copy() for row, identity in enumerate(ids)}
        missing = []
        for identity in first.tolist():
            if int(identity) in rows:
                last[int(identity)] = rows[int(identity)]
            elif int(identity) in last:
                rows[int(identity)] = last[int(identity)].copy(); missing.append(int(identity))
        order = [int(identity) for identity in first.tolist() if int(identity) in rows]
        values["smpl_id"] = np.asarray(order, dtype=np.int64)
        values["verts_world"] = np.stack([rows[int(identity)] for identity in order]).astype(np.float32)
        np.savez(path, **values)
        filled.append({"index": index, "observed_ids": ids.tolist(), "output_ids": order, "held_ids": missing})
    report = {"source": str(source), "output": str(output), "boundary_index": int(a.boundary), "policy": "causal_hold_last_mesh_per_persistent_id", "frames": filled}
    (output / "track_bank.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
