#!/usr/bin/env python3
"""Run one frozen Movie3R-v15 case and emit standard demo payloads.

The heavy model forward is delegated to the audited v14 exporter.  This
wrapper adds the v15 release contract and the optional adaptive camera-human
post-gate.  It deliberately forces CPU execution and keeps all generated
files below ``output/v15``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).resolve().parent / "FINAL_RUNTIME_SPEC.json"
EXPORTER = REPO_ROOT / "versions/v14/export_report_multihuman_comparison.py"
ADAPTIVE = REPO_ROOT / "versions/v14/adaptive_post_human_boundary.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/v15/cases"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-json", help="One JSON object from a JSONL manifest")
    source.add_argument("--case", type=Path, help="JSONL manifest path")
    p.add_argument("--line", type=int, default=1, help="1-based non-empty JSONL line when --case is used")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--original-checkpoint", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--disable-adaptive-joint", action="store_true")
    p.add_argument("--size", type=int, default=512)
    return p.parse_args()


def repo_path(value: Path) -> Path:
    raw = value.expanduser()
    path = (REPO_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Path must stay inside Movie3R: {path}")
    return path


def read_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.record_json:
        record = json.loads(args.record_json)
    else:
        if args.line < 1:
            raise ValueError("--line is 1-based")
        rows = [line for line in args.case.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.line > len(rows):
            raise IndexError(f"Manifest has {len(rows)} non-empty lines; requested {args.line}")
        record = json.loads(rows[args.line - 1])
    if not isinstance(record, dict):
        raise TypeError("A case must be a JSON object")
    required = ("sequence", "frame", "pre_camera", "post_camera")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing case fields: {missing}")
    record.setdefault("case_id", f"{record['sequence']}_t{int(record['frame']):04d}_c{record['pre_camera']}_c{record['post_camera']}")
    record.setdefault("pre_frames", 5)
    record.setdefault("post_frames", 25)
    record.setdefault("enable_adaptive_joint", True)
    if int(record["pre_frames"]) < 1 or int(record["post_frames"]) < 3:
        raise ValueError("pre_frames >= 1 and post_frames >= 3 are required")
    return record


def safe_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid case_id: {value!r}")
    return cleaned[:180]


def exporter_case_name(record: dict[str, Any]) -> str:
    return (
        f"{record['sequence']}_t{int(record['frame']):04d}_c{record['pre_camera']}"
        f"_c{record['post_camera']}_pre{int(record['pre_frames'])}_post{int(record['post_frames'])}"
    )


def default_checkpoint() -> Path:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return REPO_ROOT / spec["checkpoints"]["primary_multihuman"]["path"]


def default_original_checkpoint() -> Path:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return REPO_ROOT / spec["checkpoints"]["original_human3r"]["path"]


def run_command(command: list[str], env: dict[str, str], log_path: Path) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    payload = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}\n{completed.stderr[-4000:]}")


def main() -> None:
    args = parse_args()
    record = read_record(args)
    output_root = repo_path(args.output_root)
    if output_root == REPO_ROOT or output_root in {Path("/"), Path("/data"), Path("/data/wangzheng")}:
        raise ValueError(f"Refusing broad output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    case_id = safe_case_id(str(record["case_id"]))
    case_container = output_root / case_id
    case_root = case_container / exporter_case_name(record)
    if case_root.exists() and not args.overwrite:
        raise FileExistsError(f"Case exists; pass --overwrite: {case_root}")

    record_checkpoint = record.get("checkpoint")
    checkpoint = (
        repo_path(args.checkpoint)
        if args.checkpoint
        else (repo_path(Path(str(record_checkpoint))) if record_checkpoint else default_checkpoint())
    )
    original = repo_path(args.original_checkpoint) if args.original_checkpoint else default_original_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not original.is_file():
        raise FileNotFoundError(original)

    # Temporary files produced by image/model helpers are kept in the release
    # output tree rather than in / or an implicit system directory.
    tmp_root = REPO_ROOT / "output/v15/tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(tmp_root), "PYTHONPATH": str(REPO_ROOT)})
    command = [
        sys.executable, str(EXPORTER),
        "--sequence", str(record["sequence"]),
        "--frame", str(int(record["frame"])),
        "--pre-camera", str(record["pre_camera"]),
        "--post-camera", str(record["post_camera"]),
        "--pre-frames", str(int(record["pre_frames"])),
        "--post-frames", str(int(record["post_frames"])),
        "--size", str(int(args.size)),
        "--current-checkpoint", str(checkpoint),
        "--original-checkpoint", str(original),
        "--output-root", str(case_container),
    ]
    if str(record["sequence"]) == "avatarrex":
        command.extend(["--avatarrex-group", str(record.get("avatarrex_group", "lbn1"))])
    if args.overwrite:
        command.append("--overwrite")

    run_command(command, env, case_container / "v15_exporter.log.json")
    if not case_root.is_dir():
        raise RuntimeError(f"Exporter completed but case directory is missing: {case_root}")

    adaptive_enabled = bool(record.get("enable_adaptive_joint", True)) and not args.disable_adaptive_joint
    final_root = case_root / "movie3r_final_adaptive_joint"
    adaptive_command: list[str] | None = None
    adaptive_log = case_root / "v15_adaptive.log.json"
    if adaptive_enabled:
        adaptive_command = [
            sys.executable, str(ADAPTIVE),
            "--source", str(case_root / "movie3r_b0_brtc_c1"),
            "--raw-source", str(case_root / "movie3r_raw_current_human3r"),
            "--output", str(final_root),
            "--boundary", str(int(record["pre_frames"])),
        ]
        if args.overwrite:
            adaptive_command.append("--overwrite")
        run_command(adaptive_command, env, adaptive_log)
    else:
        # Keep a clearly named final artifact even for an ablation.  Use the
        # audited adaptive script only when enabled; copying is done by the
        # exporter-compatible Python standard library here.
        import shutil
        if final_root.exists():
            if not args.overwrite:
                raise FileExistsError(final_root)
            shutil.rmtree(final_root)
        shutil.copytree(case_root / "movie3r_b0_brtc_c1", final_root)
        (final_root / "adaptive_joint_boundary.json").write_text(
            json.dumps({"accepted": False, "reason": "disabled_by_case_manifest"}, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest_path = case_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["v15"] = {
        "release": "Movie3R-v15-final",
        "case_id": case_id,
        "record": record,
        "checkpoint": str(checkpoint),
        "adaptive_joint_enabled": adaptive_enabled,
        "adaptive_command": adaptive_command,
        "output_paths": {
            "original_human3r": str(case_root / "original_human3r"),
            "raw_current_human3r": str(case_root / "movie3r_raw_current_human3r"),
            "b0_brtc_c1": str(case_root / "movie3r_b0_brtc_c1"),
            "final_adaptive_joint": str(final_root),
        },
        "runtime_contract": {
            "device": "cpu", "gt_used": False, "future_frames_used_at_boundary": 0,
            "pre_frames_unchanged": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": case_id, "case_root": str(case_root), "final": str(final_root), "adaptive": adaptive_enabled}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
