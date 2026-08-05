#!/usr/bin/env python3
"""Compare old-B0 and cross96-B0 on the same frozen EgoHumans confirmation.

This is a read-only report generator.  It loads the two saved clean-reset
geometry caches and two report JSON files, verifies that their raw forward
outputs are numerically identical, then computes fixed-world camera error
from the cached B0 compositions and evaluator-only COLMAP poses.  It does not
run Human3R, tune a policy, use a GPU, or inspect future frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from versions.v13.egobody_probe import load_colmap  # noqa: E402
from versions.v14.run_v14_2_single_sequence import rotation_error_deg  # noqa: E402


DEFAULT_CROSS = REPO_ROOT / "output/v14/fine_alignment_research/cross96_brtc_egohumans_confirmation"
DEFAULT_OLD = REPO_ROOT / "output/v14/fine_alignment_research/old_b0_brtc_egohumans_confirmation"
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble")
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/b0_checkpoint_comparison_egohumans"
METRICS = (
    "w_mpjpe_mm", "wa_mpjpe_mm", "fixed_world_root_mm", "fixed_world_joint_mm",
    "fixed_world_vertex_mm", "pairwise_root_vector_mm", "ate_m_sim3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
    }


def raw_parity(cross: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    delta = {"camera": 0.0, "person_count": 0, "native_id": 0}
    if len(cross["chains"]) != len(old["chains"]):
        raise ValueError("different chain count")
    for first, second in zip(cross["chains"], old["chains"]):
        for first_segment, second_segment in zip(first["segments"], second["segments"]):
            for first_frame, second_frame in zip(first_segment["frames"], second_segment["frames"]):
                delta["camera"] = max(delta["camera"], float(np.max(np.abs(
                    np.asarray(first_frame["camera_c2w"]) - np.asarray(second_frame["camera_c2w"])
                ))))
                first_people, second_people = first_frame["people"], second_frame["people"]
                delta["person_count"] = max(delta["person_count"], abs(len(first_people) - len(second_people)))
                delta["native_id"] = max(delta["native_id"], int(
                    [person["native_track_id"] for person in first_people]
                    != [person["native_track_id"] for person in second_people]
                ))
    return {"max_abs_camera": delta["camera"], "person_count_max_difference": delta["person_count"],
            "native_id_any_difference": bool(delta["native_id"]),
            "exact_raw_forward_parity": bool(delta["camera"] == 0.0 and delta["person_count"] == 0 and delta["native_id"] == 0)}


def camera_errors(cache: dict[str, Any], data_root: Path) -> dict[str, Any]:
    _, exo = load_colmap(data_root)
    first_post, propagated = [], []
    per_chain = []
    for chain in cache["chains"]:
        segments = chain["segments"]
        first = segments[0]["frames"][0]
        gauge = np.asarray(first["camera_c2w"], dtype=np.float64) @ np.linalg.inv(
            np.asarray(exo[first["camera_name"]]["c2w_aria01"], dtype=np.float64)
        )
        cumulative = np.eye(4, dtype=np.float64)
        chain_rows = []
        for segment_index, segment in enumerate(segments):
            if segment_index:
                cumulative = cumulative @ np.asarray(chain["b0_boundaries"][segment_index - 1], dtype=np.float64)
            for frame_index, frame in enumerate(segment["frames"]):
                predicted = cumulative @ np.asarray(frame["camera_c2w"], dtype=np.float64)
                target = gauge @ np.asarray(exo[frame["camera_name"]]["c2w_aria01"], dtype=np.float64)
                row = {
                    "chain": int(chain["chain_index"]), "segment": int(segment_index), "frame": int(frame_index),
                    "translation_m": float(np.linalg.norm(predicted[:3, 3] - target[:3, 3])),
                    "rotation_deg": float(rotation_error_deg(predicted, target)),
                }
                row["composite"] = row["translation_m"] + 0.02 * row["rotation_deg"]
                if segment_index:
                    propagated.append(row)
                    if frame_index == 0:
                        first_post.append(row)
                chain_rows.append(row)
        per_chain.append(chain_rows)
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {key: stats([row[key] for row in rows]) for key in ("translation_m", "rotation_deg", "composite")}
    return {"first_post": summarize(first_post), "all_post_shot_frames": summarize(propagated), "per_chain": per_chain}


def markdown(report: dict[str, Any]) -> str:
    checkpoints = report["checkpoint_comparison"]
    lines = [
        "# EgoHumans frozen batch: old B0 vs cross96 B0",
        "",
        "This is a read-only, same-manifest local comparison. It is not the official Multi-THuMBS protocol.",
        "",
        "## Integrity",
        "",
        f"- Raw-reset forward parity: `{report['raw_forward_parity']['exact_raw_forward_parity']}`; camera max abs delta `{report['raw_forward_parity']['max_abs_camera']:.1e}`.",
        f"- Same 5 chains / 10 cuts / 75 frames; coverage `{report['coverage']:.1%}` for both checkpoints.",
        "- No GPU/model forward/policy selection in this comparison script.",
        "",
        "## Main result",
        "",
        "| Method | B0 W | BRTC W | B0 WA | BRTC WA | B0 fixed root | BRTC fixed root | BRTC camera |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("old_b0", "cross96"):
        value = checkpoints[name]
        b0, brtc = value["methods"]["b0"]["metrics"], value["methods"]["b0_brtc_lc"]["metrics"]
        lines.append(
            f"| {name} | {b0['w_mpjpe_mm']:.1f} | {brtc['w_mpjpe_mm']:.1f} | "
            f"{b0['wa_mpjpe_mm']:.1f} | {brtc['wa_mpjpe_mm']:.1f} | "
            f"{b0['fixed_world_root_mm']:.1f} | {brtc['fixed_world_root_mm']:.1f} | "
            f"{value['camera_bit_exact']} |"
        )
    cross, old = checkpoints["cross96"], checkpoints["old_b0"]
    lines.extend([
        "",
        "Cross96 minus old after BRTC: "
        f"W `{cross['methods']['b0_brtc_lc']['metrics']['w_mpjpe_mm'] - old['methods']['b0_brtc_lc']['metrics']['w_mpjpe_mm']:+.1f} mm`, "
        f"WA `{cross['methods']['b0_brtc_lc']['metrics']['wa_mpjpe_mm'] - old['methods']['b0_brtc_lc']['metrics']['wa_mpjpe_mm']:+.1f} mm`, "
        f"fixed root `{cross['methods']['b0_brtc_lc']['metrics']['fixed_world_root_mm'] - old['methods']['b0_brtc_lc']['metrics']['fixed_world_root_mm']:+.1f} mm`.",
        "",
        "## Camera proposal error",
        "",
        "| Scope | Checkpoint | T (m) | R (deg) | Composite |",
        "|---|---|---:|---:|---:|",
    ])
    for scope in ("first_post", "all_post_shot_frames"):
        for name in ("old_b0", "cross96"):
            row = checkpoints[name]["camera_error"][scope]
            lines.append(f"| {scope} | {name} | {row['translation_m']['mean']:.3f} | {row['rotation_deg']['mean']:.2f} | {row['composite']['mean']:.3f} |")
    lines.extend([
        "",
        "## Decision",
        "",
        "`NO_GO_CROSS96_AS_UNIVERSAL_MULTI_HUMAN_B0`: cross96 retains its established controlled four-source camera benefit, but on this independent multi-human EgoHumans confirmation it is worse than the old B0 in W, fixed-world human metrics, camera ATE and camera error. The result cannot be attributed to changed raw Human3R outputs or coverage. It must not be presented as a universal B0 replacement.",
        "",
        "BRTC remains a camera-invariant, low-tail auxiliary verifier; it does not erase the B0 domain gap. Future work must either validate one B0 checkpoint on all claimed domains or explicitly scope controlled-single-person and real-multi-person results to separate checkpoints rather than combining them in one main-table row.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (args.cross_dir, args.old_dir, args.output_dir):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError(f"artifact outside workspace: {path}")
    cross_cache = torch.load(args.cross_dir / "cross96_raw_geometry_cpu.pt", map_location="cpu", weights_only=False)
    old_cache = torch.load(args.old_dir / "cross96_raw_geometry_cpu.pt", map_location="cpu", weights_only=False)
    cross_report = json.loads((args.cross_dir / "report.json").read_text(encoding="utf-8"))
    old_report = json.loads((args.old_dir / "report.json").read_text(encoding="utf-8"))
    parity = raw_parity(cross_cache, old_cache)
    if not parity["exact_raw_forward_parity"]:
        raise RuntimeError(f"raw branches unexpectedly differ: {parity}")
    coverage = float(cross_report["methods"]["b0"]["coverage"])
    if abs(coverage - float(old_report["methods"]["b0"]["coverage"])) > 1e-12:
        raise RuntimeError("coverage mismatch")
    report = {
        "title": "Old B0 versus cross96 B0, frozen EgoHumans batch confirmation",
        "raw_forward_parity": parity,
        "coverage": coverage,
        "checkpoint_comparison": {
            "cross96": {"execution": cross_report["execution"], "methods": cross_report["methods"], "camera_bit_exact": cross_report["camera_exactness"]["bit_exact"], "camera_error": camera_errors(cross_cache, args.data_root)},
            "old_b0": {"execution": old_report["execution"], "methods": old_report["methods"], "camera_bit_exact": old_report["camera_exactness"]["bit_exact"], "camera_error": camera_errors(old_cache, args.data_root)},
        },
        "decision": "NO_GO_CROSS96_AS_UNIVERSAL_MULTI_HUMAN_B0",
        "limitations": ["local transparent protocol, not official Multi-THuMBS", "one EgoHumans capture", "cross96 is nevertheless independently trained and the Ego timestamps were frozen before either result"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
