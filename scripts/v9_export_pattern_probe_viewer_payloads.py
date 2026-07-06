#!/usr/bin/env python3
"""Export one training clip per explicit shot pattern for viewer inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from dust3r.inference import loss_of_one_batch
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import todevice
from scripts.v8_4_view_pose_benchmark_scene import (
    case_name,
    load_manifest,
    make_single_record_dataset,
    write_one_record_manifest,
)
from scripts.v9_export_viewer_payload_light import save_payload_light


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_root", type=Path, required=True)
    parser.add_argument("--source", default="avatarrex")
    parser.add_argument("--patterns", nargs="+", default=["aaaa", "aabb", "abab", "abba", "aabc", "abcd"])
    parser.add_argument("--entry", type=int, default=0)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--case_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="resize_only_16")
    parser.add_argument("--raw_roots", default="null")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.case_root.mkdir(parents=True, exist_ok=True)

    add_path_to_dust3r(str(args.model_path))
    print(f"Loading model once: {args.model_path}", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).float().eval()
    smpl_model = SMPLModel(
        torch.device(args.device),
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )

    exported = []
    for pattern in args.patterns:
        manifest = args.manifest_root / args.source / f"train_{pattern}.jsonl"
        records = load_manifest(manifest)
        record = records[args.entry]
        case_dir = args.case_root / f"{args.source}_{case_name(record)}"
        corrected_dir = case_dir / "corrected"
        raw_dir = case_dir / "raw_human3r"
        needed = [corrected_dir / "camera" / "000000.npz", raw_dir / "camera" / "000000.npz"]
        if all(path.is_file() for path in needed) and not args.overwrite:
            print(json.dumps({"pattern": pattern, "case_dir": str(case_dir), "status": "exists"}), flush=True)
            exported.append({"pattern": pattern, "case_dir": str(case_dir), "record": record})
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "viewer_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        one_record_manifest = case_dir / "one_record_manifest.jsonl"
        write_one_record_manifest(one_record_manifest, record)

        dataset_args = argparse.Namespace(
            data_root=args.data_root,
            test_split="Test/v8_4_mixed_aabb_aaaa",
            resolution=tuple(args.resolution),
            resize_mode=args.resize_mode,
            raw_roots=args.raw_roots,
        )
        print(f"[{pattern}] building dataloader sample: {record.get('seqs')} frames={record.get('frames')}", flush=True)
        dataset = make_single_record_dataset(dataset_args, record, one_record_manifest)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        batch = todevice(next(iter(loader)), args.device)

        print(f"[{pattern}] running forward", flush=True)
        with torch.no_grad():
            result = loss_of_one_batch(
                batch,
                model,
                criterion=None,
                accelerator=None,
                symmetrize_batch=False,
                inference=False,
                smpl_model=smpl_model,
            )
        outputs = todevice({"views": result["views"], "pred": result["pred"]}, "cpu")

        print(f"[{pattern}] saving corrected: {corrected_dir}", flush=True)
        save_payload_light(outputs, corrected_dir, "camera_pose", smpl_model=smpl_model)
        print(f"[{pattern}] saving raw: {raw_dir}", flush=True)
        save_payload_light(outputs, raw_dir, "v8_raw_camera_pose", smpl_model=smpl_model)
        exported.append({"pattern": pattern, "case_dir": str(case_dir), "record": record})
        print(json.dumps({"pattern": pattern, "case_dir": str(case_dir), "status": "exported"}), flush=True)

    manifest_path = args.case_root / "viewer_cases.json"
    manifest_path.write_text(json.dumps(exported, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
