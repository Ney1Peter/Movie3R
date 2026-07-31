#!/usr/bin/env python3
"""Evaluate a V14 cut-first checkpoint on the frozen 180 AABB source audit."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
THIS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT, THIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from evaluate_cut_events import evaluate_batch, stats, summarize  # noqa: E402
from v10_token_alignment_4source_probe import load_aabb_views_for_record  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_RECORDS = (
    REPO_ROOT / "output/v10_candidate_selection/oracle_gt_4source/selected_records.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14_cut_first_cross_source/eval_current_single_180"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
SOURCE_ORDER = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
METHODS = ("raw_reset", "shadow_event", "b0_runtime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_ORDER, default=SOURCE_ORDER)
    parser.add_argument("--max-cases-per-source", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("_")


def load_views(record: dict[str, Any], args: argparse.Namespace) -> list[dict]:
    loader_args = SimpleNamespace(
        data_root=args.data_root,
        resolution=(512, 288),
        resize_mode="human3r_demo",
        boundary=2,
    )
    views = load_aabb_views_for_record(record, loader_args, torch.device("cpu"))
    if len(views) < 3:
        raise RuntimeError(f"Expected at least three AABB views, got {len(views)}")
    return views[:3]


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V14 Frozen 180-Case B0 Evaluation",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "| Split | N | Method | Camera T | Camera R | Composite | P90 comp. | P95 comp. | Human head | Catastrophic |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("overall", report["summary"])] + [
        (source, report["by_source"][source])
        for source in SOURCE_ORDER
        if source in report["by_source"]
    ]
    for split, summary in groups:
        for method in METHODS:
            row = summary["methods"][method]
            lines.append(
                f"| {split} | {summary['case_count']} | {method} | "
                f"{row['camera_translation_m']['mean']:.4f} | "
                f"{row['camera_rotation_deg']['mean']:.3f} | "
                f"{row['camera_composite']['mean']:.4f} | "
                f"{row['camera_composite']['p90']:.4f} | "
                f"{row['camera_composite']['p95']:.4f} | "
                f"{row['human_head_m']['mean']:.4f} | "
                f"{row['catastrophic_count']} |"
            )
    lines.extend(
        [
            "",
            f"Completed: `{report['summary']['case_count']}`; failures: `{len(report['failures'])}`.",
            "",
            "The 180 records are evaluation-only. Ground-truth camera/human fields are read after the shadow and raw-reset predictions are finalized.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    source_counts: defaultdict[str, int] = defaultdict(int)
    for record in read_jsonl(args.records):
        source = str(record["source"])
        if source not in args.sources:
            continue
        if args.max_cases_per_source and source_counts[source] >= args.max_cases_per_source:
            continue
        selected.append(record)
        source_counts[source] += 1

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    smpl_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()

    rows, failures = [], []
    for index, record in enumerate(selected, start=1):
        name = safe_name(record["pattern_id"])
        path = cases_dir / f"{name}.json"
        cached = None
        if path.is_file() and not args.overwrite:
            cached = json.loads(path.read_text(encoding="utf-8"))
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                row = evaluate_batch(model, smpl_layer, load_views(record, args), device)
                row.update({"status": "ok", "source": record["source"], "record": record})
            except Exception as error:
                row = {
                    "status": "failed",
                    "source": record["source"],
                    "record": record,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        if row["status"] == "ok":
            rows.append(row)
            metric = row["methods"]["b0_runtime"]
            print(
                f"[{index:03d}/{len(selected):03d}] {record['source']} {name} "
                f"comp={metric['camera_composite']:.4f} cat={metric['catastrophic']}",
                flush=True,
            )
        else:
            failures.append(row)
            print(f"[{index:03d}/{len(selected):03d}] {name} FAILED {row['error']}", flush=True)
        if device.type == "cuda" and index % 10 == 0:
            torch.cuda.empty_cache()

    report = {
        "experiment": "v14_cut_first_cross_source_frozen_180_b0",
        "checkpoint": str(args.model_path),
        "records": str(args.records),
        "model_flags": flags,
        "summary": summarize(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in args.sources
        },
        "failures": failures,
        "timing_seconds": stats([row["timing_seconds"] for row in rows]),
        "cases": rows,
    }
    json_path = args.output_dir / "four_source_b0_evaluation.json"
    md_path = args.output_dir / "four_source_b0_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
