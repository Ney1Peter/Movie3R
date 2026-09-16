#!/usr/bin/env python3
"""Disk-bounded EgoHumans inference worker with one persistent model per GPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from experiments.registration_baselines import infer  # noqa: E402
from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "src/human3r_896L.pth")
    parser.add_argument("--scene-grid-stride", type=int, default=8)
    parser.add_argument("--scene-cache-voxel", type=float, default=0.03)
    parser.add_argument("--scene-max-points", type=int, default=250000)
    return parser.parse_args()


def capture_name(entry: str) -> str:
    name = Path(entry).name
    stem = name[:-7] if name.endswith(".tar.gz") else name
    return re.sub(r"-\d{3}$", "", stem)


def slug(entry: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", entry.removesuffix(".tar.gz").replace("/", "__"))


def safe_cleanup(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_parent.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"unsafe cleanup target: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def stage_capture(outer: Path, entry: str, worker_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    token = slug(entry)
    archive = worker_root / "archives" / f"{token}.tar.gz"
    stage = worker_root / "staging" / token
    capture = stage / capture_name(entry)
    archive.parent.mkdir(parents=True, exist_ok=True)
    stage.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    safe_cleanup(stage, worker_root / "staging")
    partial = archive.with_suffix(archive.suffix + ".partial")
    with zipfile.ZipFile(outer) as source:
        info = source.getinfo(entry)
        with source.open(info) as reader, partial.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
    if partial.stat().st_size != info.file_size:
        raise ValueError(f"inner tar size mismatch for {entry}")
    os.replace(partial, archive)
    stage.mkdir(parents=True)
    command = [
        "tar", "-xzf", str(archive), "-C", str(stage), "--strip-components=8",
        "--no-same-owner", "--no-same-permissions",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"tar extraction failed: {completed.stderr[-2000:]}")
    if not (capture / "exo").is_dir():
        raise FileNotFoundError(capture / "exo")
    if any(path.is_symlink() for path in capture.rglob("*")):
        raise ValueError(f"capture contains symlink: {entry}")
    return stage, archive, {
        "entry": entry, "crc32": f"{info.CRC:08x}",
        "inner_tar_size_bytes": int(info.file_size),
    }


def main() -> None:
    args = parse_args()
    rows = infer.read_rows(args.manifest.resolve())
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line, row in enumerate(rows, start=1):
        grouped[str(row["archive_entry"])].append((line, row))
    entries = sorted(grouped)
    selected_entries = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard_index]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size != 4670554642:
        raise FileNotFoundError(checkpoint)
    topology = CommonTopology.load()
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    frozen.strict_original(model)
    model.eval()
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False,
        person_center="head",
    ).to(device).eval()
    worker_root = args.output_root / "work/egohumans_runtime" / args.split / f"shard{args.shard_index:02d}"
    completed_cases, reused_cases, failures, captures = [], [], [], []
    infer_args = SimpleNamespace(
        device=args.device, size=args.size, output_root=args.output_root,
        dataset="egohumans", split=args.split, manifest=args.manifest,
        checkpoint=checkpoint, scene_grid_stride=args.scene_grid_stride,
        scene_cache_voxel=args.scene_cache_voxel,
        scene_max_points=args.scene_max_points,
    )
    for entry_position, entry in enumerate(selected_entries, start=1):
        pending = []
        for line, record in grouped[entry]:
            case_id = str(record["case_id"])
            cache = args.output_root / "predictions/egohumans" / args.split / f"{case_id}.r0_scene.npz"
            report = args.output_root / "logs/inference/egohumans" / args.split / f"{case_id}.runtime.json"
            if infer.reusable(cache, report, case_id):
                reused_cases.append(case_id)
            else:
                pending.append((line, record))
        if not pending:
            print(f"[{entry_position}/{len(selected_entries)}] reusable capture {entry}", flush=True)
            continue
        stage = archive = None
        try:
            stage, archive, metadata = stage_capture(args.outer.resolve(), entry, worker_root)
            capture_results = []
            for line, record in pending:
                try:
                    paths = infer.image_paths(record, stage)
                    result = infer.save_case(record, paths, model, layer, topology, infer_args, line)
                    completed_cases.append(result)
                    capture_results.append(result["case_id"])
                    print(f"[{entry_position}/{len(selected_entries)}] complete {result['case_id']} {result['seconds']:.1f}s", flush=True)
                except Exception as exc:
                    failure = {"case_id": record["case_id"], "entry": entry, "error": f"{type(exc).__name__}: {exc}"}
                    failures.append(failure)
                    print(f"FAILED {failure}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            captures.append({**metadata, "completed_cases": capture_results})
        except Exception as exc:
            for _, record in pending:
                failures.append({"case_id": record["case_id"], "entry": entry, "error": f"{type(exc).__name__}: {exc}"})
            print(f"CAPTURE FAILED {entry}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if stage is not None:
                safe_cleanup(stage, worker_root / "staging")
            if archive is not None and archive.exists():
                archive.unlink()
    ledger = args.output_root / "logs" / f"infer_egohumans_{args.split}_shard{args.shard_index:02d}.json"
    infer.atomic_json(ledger, {
        "schema_version": "Shot3R-registration-EgoHumans-inference-worker-v1",
        "split": args.split, "device": args.device,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "selected_captures": selected_entries, "captures": captures,
        "completed": completed_cases, "reused": reused_cases, "failures": failures,
        "gt_opened_by_runtime": False,
    })
    if failures:
        raise SystemExit(f"{len(failures)} cases failed; see {ledger}")


if __name__ == "__main__":
    main()

