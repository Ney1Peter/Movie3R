#!/usr/bin/env python3
"""Save the raw post-shot Human3R branch from the current Movie3R checkpoint.

This is a CPU-only helper for the joint camera-human probe.  It uses no GT and
does not load another checkpoint: the raw post stream is the same checkpoint
with a clean reset, before applying B0/BRTC.  The five pre files are copied
only to keep the standard 30-frame payload indexing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from dust3r.model import ARCroco3DStereo
from dust3r.utils.smpl_layer import SMPL_Layer
from versions.v13 import gt_id_consensus as gt
from versions.v14.export_p5_brtc_demo_payload import FACES, as_array, background_from_prediction
from versions.v14.export_report_multihuman_comparison import configure_model, decode_frames, set_event_indices, write_demo_payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True, help="Full demo case root")
    p.add_argument("--current-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(); case, out = a.case.resolve(), a.output.resolve()
    manifest = json.loads((case / "manifest.json").read_text())
    paths = [Path(value) for value in manifest["input_paths"]]
    boundary = int(a.boundary); post_paths = paths[boundary:]
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = ARCroco3DStereo.from_pretrained(str(a.current_checkpoint.resolve())).to("cpu")
    flags = configure_model(model); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to("cpu").eval()
    views = set_event_indices(gt.prepare_full_square_input(model, post_paths, SimpleNamespace(size=int(a.size))), set())
    with torch.no_grad():
        predictions, returned, debug = model.forward_recurrent_lighter(views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True)
    frames = decode_frames(predictions, returned, debug, layer)
    temp = out / "_post_tmp"
    write_demo_payload(temp, frames, np.load(ROOT / FACES, allow_pickle=False)["f"], True)
    # Copy pre files from the already saved B0 payload and move raw post files
    # into their original global indices.
    baseline = case / "movie3r_b0_brtc_c1"
    for sub in ("camera", "color", "depth", "conf", "smpl"):
        (out / sub).mkdir(exist_ok=True)
        for index in range(boundary):
            suffix = "npz" if sub in ("camera", "smpl") else "npy" if sub in ("depth", "conf") else "png"
            src = baseline / sub / f"{index:06d}.{suffix}"
            shutil.copy2(src, out / sub / src.name)
        suffix = "npz" if sub in ("camera", "smpl") else "npy" if sub in ("depth", "conf") else "png"
        for local in range(len(frames)):
            src = temp / sub / f"{local:06d}.{suffix}"
            shutil.copy2(src, out / sub / f"{boundary + local:06d}.{suffix}")
    shutil.rmtree(temp)
    (out / "raw_current_manifest.json").write_text(json.dumps({"case": str(case), "checkpoint": str(a.current_checkpoint.resolve()), "boundary": boundary, "people_per_post_frame": [len(frame["people"]) for frame in frames], "gt_used": False, "flags": flags}, indent=2, ensure_ascii=False) + "\n")
    print(out)


if __name__ == "__main__": main()
