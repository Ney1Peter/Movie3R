#!/usr/bin/env python3
"""Audit wide-baseline candidates for catastrophic V24 holdout cases."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output" / "v27_consensus_holdout2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_ROOT / "records" / "holdout_records.jsonl",
    )
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_ROOT / "v15")
    parser.add_argument(
        "--evaluation_report",
        type=Path,
        default=DEFAULT_ROOT / "evaluation" / "v25_holdout_rotation_validation.json",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_ROOT / "failure_candidate_audit"
    )
    parser.add_argument("--catastrophic_deg", type=float, default=45.0)
    return parser.parse_args()


def load_shards(root: Path) -> dict[str, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise RuntimeError(f"Invalid V15 cache: {len(rows)} rows, {len(output)} unique")
    return output


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    return float(
        np.degrees(
            np.linalg.norm(Rotation.from_matrix(relative.astype(np.float64)).as_rotvec())
        )
    )


def candidate_summary(
    candidate: dict | None,
    fixed_rotation: np.ndarray,
    v24_rotation: np.ndarray | None,
) -> dict:
    if not candidate or candidate.get("fit_failed") or candidate.get("transform") is None:
        return {"fit_failed": True}
    rotation_error = candidate.get(
        "rotation_error_deg", candidate.get("camera_rotation_error_deg")
    )
    translation_error = candidate.get(
        "translation_error_m", candidate.get("camera_translation_error_m")
    )
    if rotation_error is None or translation_error is None:
        return {"fit_failed": True, "missing_evaluation_fields": True}
    transform = np.asarray(candidate["transform"], dtype=np.float32)
    rotation = transform[:3, :3]
    summary = {
        "fit_failed": False,
        "rotation_error_deg": float(rotation_error),
        "translation_error_m": float(translation_error),
        "rotation_from_fixed_deg": angle_deg(rotation, fixed_rotation),
    }
    if v24_rotation is not None:
        summary["rotation_from_v24_deg"] = angle_deg(rotation, v24_rotation)
    for key in (
        "fit_residual_median_m",
        "robust_inlier_ratio",
        "correspondence_count",
        "epipolar_median_px",
        "translation_direction_error_deg",
        "translation_scale_ratio",
        "translation_scale_log_abs",
        "essential_inlier_ratio",
        "essential_rotation_spread_deg",
        "da3_correspondence_count",
        "da3_fit_residual_median_m",
        "da3_robust_inlier_ratio",
    ):
        if key in candidate:
            value = candidate[key]
            summary[key] = (
                int(value)
                if key in ("correspondence_count", "da3_correspondence_count")
                else float(value)
            )
    return summary


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide_rows = load_shards(args.v15_dir)
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    evaluation_rows = {row["case_name"]: row for row in evaluation["cases"]}
    if set(wide_rows) != set(evaluation_rows):
        raise RuntimeError(
            f"V15/evaluation mismatch: {len(wide_rows)}/{len(evaluation_rows)}"
        )

    candidate_names = (
        "coarse",
        "rotation_fixed_translation",
        "essential_rotation_fixed_translation",
        "da3_metric_full",
        "da3_wide_rotation_metric_translation",
        "metric_full_full",
        "wide_rotation_metric_translation_full",
    )
    cases: list[dict] = []
    all_candidate_errors: dict[str, list[float]] = {}
    best_counts: dict[str, int] = {}
    for name in sorted(wide_rows):
        eval_row = evaluation_rows[name]
        if float(eval_row["v24"]["rotation_error_deg"]) <= args.catastrophic_deg:
            continue
        wide = wide_rows[name]
        fixed_rotation = np.asarray(
            wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32
        )[:3, :3]
        v24_rotation = None
        if "v24_transform" in eval_row:
            v24_rotation = np.asarray(eval_row["v24_transform"], dtype=np.float32)[:3, :3]
        windows = {}
        ranked: list[tuple[float, str]] = []
        for window_name, window in wide["windows"].items():
            candidate_rows = {}
            for candidate_name in candidate_names:
                candidate = window["candidates"].get(candidate_name)
                summary = candidate_summary(candidate, fixed_rotation, v24_rotation)
                candidate_rows[candidate_name] = summary
                if not summary["fit_failed"]:
                    label = f"{window_name}/{candidate_name}"
                    error = float(summary["rotation_error_deg"])
                    ranked.append((error, label))
                    all_candidate_errors.setdefault(label, []).append(error)
            windows[window_name] = {
                "spread_deg": float(window["rotation_consensus"]["spread_deg"]),
                "pair_count": int(window["pair_count"]),
                "candidates": candidate_rows,
            }
        ranked.sort()
        best_label = ranked[0][1] if ranked else None
        if best_label is not None:
            best_counts[best_label] = best_counts.get(best_label, 0) + 1
        cases.append(
            {
                "case_name": name,
                "source": eval_row["source"],
                "texture_score": float(wide["texture_score"]),
                "human_image_ratio": float(wide["human_image_ratio"]),
                "fixed_rotation_error_deg": float(eval_row["fixed"]["rotation_error_deg"]),
                "torso_rotation_error_deg": float(
                    eval_row["torso_gravity"]["rotation_error_deg"]
                ),
                "v24_rotation_error_deg": float(eval_row["v24"]["rotation_error_deg"]),
                "v25_rotation_error_deg": float(eval_row["v25"]["rotation_error_deg"]),
                "v24_diagnostics": eval_row["v24_diagnostics"],
                "v25_diagnostics": eval_row["v25_diagnostics"],
                "best_candidate": (
                    None
                    if not ranked
                    else {"name": ranked[0][1], "rotation_error_deg": ranked[0][0]}
                ),
                "candidate_ranking": [
                    {"name": label, "rotation_error_deg": error}
                    for error, label in ranked
                ],
                "windows": windows,
            }
        )

    failure_names = {row["case_name"] for row in cases}
    records = load_jsonl(args.records)
    subset = [
        row
        for row in records
        if row.get("case_name", row.get("pattern_id")) in failure_names
    ]
    if len(subset) != len(failure_names):
        raise RuntimeError(
            f"Failed to materialize subset: {len(subset)}/{len(failure_names)}"
        )

    report = {
        "experiment": "V27 holdout catastrophic candidate audit",
        "protocol": {
            "catastrophic_threshold_deg": args.catastrophic_deg,
            "gt_used_for_offline_diagnosis_only": True,
            "runtime_selection_rule": False,
        },
        "total_case_count": len(wide_rows),
        "catastrophic_case_count": len(cases),
        "best_candidate_counts": best_counts,
        "candidate_rotation_error_on_catastrophes": {
            label: distribution(values)
            for label, values in sorted(all_candidate_errors.items())
        },
        "cases": cases,
    }
    report_path = args.output_dir / "failure_candidate_audit.json"
    subset_path = args.output_dir / "failure_records.jsonl"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    subset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "total_case_count": len(wide_rows),
                "catastrophic_case_count": len(cases),
                "best_candidate_counts": best_counts,
                "failures": [
                    {
                        "case_name": row["case_name"],
                        "source": row["source"],
                        "v24_rotation_error_deg": row["v24_rotation_error_deg"],
                        "best_candidate": row["best_candidate"],
                    }
                    for row in cases
                ],
            },
            indent=2,
        )
    )
    print(f">> wrote {report_path}")
    print(f">> wrote {subset_path}")


if __name__ == "__main__":
    main()
