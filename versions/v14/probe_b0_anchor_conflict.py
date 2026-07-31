#!/usr/bin/env python3
"""Audit whether human rotation or translation refinement helps learned V14 B0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.run_v14_2_multihuman_sequence import (  # noqa: E402
    b0_human_candidates,
    solution,
)


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/b0_anchor_conflict"
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
SEQUENCE_INPUTS = {
    "three": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching/v14_b0_identity_matching.json",
        "cache": REPO_ROOT
        / "output/v20_phase1_gt_id_multihuman_consensus/case_cache",
    },
    "dance": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
        "cache": REPO_ROOT / "output/v13/dance_phase2/case_cache",
    },
    "box": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
        "cache": REPO_ROOT / "output/v13/box_phase3/case_cache",
    },
}
METHODS = (
    "b0_only",
    "phase2_uniform_multi",
    "b0_rotation_only",
    "b0_translation_only",
    "b0_full_per_candidate_translation",
    "b0_full_shared_rotation_translation",
)
METRICS = (
    "camera_translation_error_m",
    "camera_rotation_error_deg",
    "camera_composite",
    "human_root_error_m",
    "human_joint_error_m",
    "human_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences", nargs="+", choices=tuple(SEQUENCE_INPUTS), default=tuple(SEQUENCE_INPUTS)
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def b0_centered_solutions(cache: dict, b0: np.ndarray) -> dict[str, dict]:
    candidates = b0_human_candidates(cache, b0)
    identities = tuple(sorted(candidates))
    if not identities:
        return {}

    rotations = [candidates[identity]["rotation"] for identity in identities]
    candidate_translations = [
        candidates[identity]["translation"] for identity in identities
    ]
    rotation_mean = geometry.so3_mean(rotations)

    translation_at_b0_rotation = np.mean(
        np.stack(
            [
                candidates[identity]["anchor"]
                - b0[:3, :3] @ candidates[identity]["post_root"]
                for identity in identities
            ]
        ),
        axis=0,
    )
    translation_at_shared_rotation = np.mean(
        np.stack(
            [
                candidates[identity]["anchor"]
                - rotation_mean @ candidates[identity]["post_root"]
                for identity in identities
            ]
        ),
        axis=0,
    )

    phase2_candidates = geometry.human_candidates(cache)
    phase2_identities = tuple(sorted(phase2_candidates))
    phase2_rotations = [
        phase2_candidates[identity]["rotation"] for identity in phase2_identities
    ]
    phase2_translations = [
        phase2_candidates[identity]["translation"] for identity in phase2_identities
    ]

    return {
        "b0_only": solution(b0[:3, :3], b0[:3, 3], identities),
        "phase2_uniform_multi": solution(
            geometry.so3_mean(phase2_rotations),
            np.mean(np.stack(phase2_translations), axis=0),
            phase2_identities,
        ),
        "b0_rotation_only": solution(rotation_mean, b0[:3, 3], identities),
        "b0_translation_only": solution(
            b0[:3, :3], translation_at_b0_rotation, identities
        ),
        "b0_full_per_candidate_translation": solution(
            rotation_mean, np.mean(np.stack(candidate_translations), axis=0), identities
        ),
        "b0_full_shared_rotation_translation": solution(
            rotation_mean, translation_at_shared_rotation, identities
        ),
    }


def finite_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: float("nan") for name in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(cases: list[dict], method: str) -> dict:
    rows = [case["methods"][method] for case in cases if method in case["methods"]]
    return {
        "valid_cases": len(rows),
        **{
            metric: finite_distribution([float(row[metric]) for row in rows])
            for metric in METRICS
        },
        "catastrophic_rate": (
            float(np.mean([row["catastrophic"] for row in rows]))
            if rows
            else float("nan")
        ),
    }


def paired_against_b0(cases: list[dict], method: str) -> dict:
    rows = [
        case
        for case in cases
        if "b0_only" in case["methods"] and method in case["methods"]
    ]
    output = {"valid_cases": len(rows)}
    for metric in METRICS:
        baseline = np.asarray(
            [case["methods"]["b0_only"][metric] for case in rows], dtype=np.float64
        )
        candidate = np.asarray(
            [case["methods"][method][metric] for case in rows], dtype=np.float64
        )
        finite = np.isfinite(baseline) & np.isfinite(candidate)
        delta = candidate[finite] - baseline[finite]
        output[metric] = {
            "method_minus_b0_mean": float(np.mean(delta)) if len(delta) else float("nan"),
            "improvement_rate": float(np.mean(delta < 0.0)) if len(delta) else float("nan"),
            "harmful_rate": float(np.mean(delta > 0.0)) if len(delta) else float("nan"),
        }
    baseline_catastrophic = np.asarray(
        [case["methods"]["b0_only"]["catastrophic"] for case in rows], dtype=bool
    )
    candidate_catastrophic = np.asarray(
        [case["methods"][method]["catastrophic"] for case in rows], dtype=bool
    )
    output["new_catastrophic_count"] = int(
        np.sum(candidate_catastrophic & ~baseline_catastrophic)
    )
    output["resolved_catastrophic_count"] = int(
        np.sum(~candidate_catastrophic & baseline_catastrophic)
    )
    return output


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def load_sequence(
    name: str, max_cases: int, data_root: Path, size: int
) -> list[dict]:
    inputs = SEQUENCE_INPUTS[name]
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[name]
    report = json.loads(inputs["report"].read_text())
    report_cases = report["cases"][: max_cases or None]
    output = []
    for index, report_case in enumerate(report_cases, start=1):
        case = report_case["case"]
        cache_path = inputs["cache"] / f"{case['key']}.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache = geometry.reassign_cache_gt_identities(
            SimpleNamespace(data_root=data_root, size=int(size), sequence=name), cache
        )
        b0 = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
        solutions = b0_centered_solutions(cache, b0)
        methods = {
            method: geometry.evaluate_solution(cache, candidate)
            for method, candidate in solutions.items()
        }
        output.append(
            {
                "sequence": name,
                "case": case,
                "candidate_count": len(next(iter(solutions.values()))["identities"]),
                "methods": methods,
            }
        )
        print(f"[{name} {index:03d}/{len(report_cases):03d}] {case['key']}", flush=True)
    return output


def aggregate(cases: list[dict]) -> dict:
    return {
        "case_count": len(cases),
        "methods": {method: summarize(cases, method) for method in METHODS},
        "paired_against_b0": {
            method: paired_against_b0(cases, method)
            for method in METHODS
            if method != "b0_only"
        },
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V14 B0 Anchor-Conflict Audit",
        "",
        "All refinements use GT identity to isolate the frozen WHERE module. "
        "Detection-to-ID labels are rebuilt with the same GT mesh-projection audit "
        "used by the learned-B0 matching probe.",
        "",
    ]
    for sequence, summary in report["summary"]["by_sequence"].items():
        lines.extend(
            [
                f"## {sequence}",
                "",
                f"Cases: {summary['case_count']}",
                "",
                "| Method | Cam T mean | Cam R mean | Composite mean | Root mean | P95 composite | Catastrophic |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = summary["methods"][method]
            lines.append(
                "| {} | {:.4f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} | {:.2%} |".format(
                    method,
                    row["camera_translation_error_m"]["mean"],
                    row["camera_rotation_error_deg"]["mean"],
                    row["camera_composite"]["mean"],
                    row["human_root_error_m"]["mean"],
                    row["camera_composite"]["p95"],
                    row["catastrophic_rate"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_sequence = {
        sequence: load_sequence(
            sequence, args.max_cases, args.data_root, int(args.size)
        )
        for sequence in args.sequences
    }
    all_cases = [case for cases in by_sequence.values() for case in cases]
    report = {
        "experiment": "v14_b0_anchor_conflict",
        "protocol": {
            "identity": "GT-ID, diagnostic only",
            "identity_audit": "GT mesh-projection reassignment, matching the B0 probe",
            "learned_b0": "precomputed V14.1 shadow/raw camera boundary",
            "geometry": "frozen Fixed Explicit + V16 20-degree bound",
            "sequences": list(args.sequences),
        },
        "summary": {
            "by_sequence": {
                sequence: aggregate(cases) for sequence, cases in by_sequence.items()
            },
            "combined": aggregate(all_cases),
        },
        "cases": all_cases,
    }
    json_path = args.output_dir / "v14_b0_anchor_conflict.json"
    markdown_path = args.output_dir / "v14_b0_anchor_conflict.md"
    json_path.write_text(json.dumps(serializable(report), indent=2, allow_nan=True))
    write_markdown(markdown_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
