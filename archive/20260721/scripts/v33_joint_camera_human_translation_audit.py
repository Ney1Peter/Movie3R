#!/usr/bin/env python3
"""Audit camera-pose gains against visible Human3R root continuity."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETS = (
    f"holdout3={REPO_ROOT / 'output/v29_rotation_rule_holdout3/v15'}",
    f"holdout4={REPO_ROOT / 'output/v31_metric_fit_holdout4/v15'}",
    f"holdout5={REPO_ROOT / 'output/v32_texture_safe_holdout5/v15'}",
)
METHODS = {
    "fixed_explicit": ("baseline", "fixed_explicit"),
    "boundary_oracle": ("baseline", "boundary_oracle"),
    "vggt_coarse": ("candidate", "coarse"),
    "da3_vggt_rotation_translation": (
        "candidate",
        "da3_wide_rotation_metric_translation",
    ),
    "human3r_vggt_rotation_translation": (
        "candidate",
        "wide_rotation_metric_translation_full",
    ),
    "da3_human3r_metric_rotation_translation": (
        "candidate",
        "da3_metric_rotation_metric_translation_full",
    ),
    "human3r_metric_full": ("candidate", "metric_full_full"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sets",
        nargs="+",
        default=DEFAULT_SETS,
        help="Named V15 caches in label=/path form.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "output/v33_joint_camera_human_translation"
        / "v33_joint_camera_human_translation_audit.json",
    )
    parser.add_argument("--window", default="full_rgb_1p1")
    parser.add_argument("--max_rotation_deg", type=float, default=45.0)
    return parser.parse_args()


def parse_set(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected label=/path, got {value!r}")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    return label, path if path.is_absolute() else REPO_ROOT / path


def load_cases(path: Path) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(glob.glob(str(path / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(shard).read_text(encoding="utf-8"))["cases"])
    if not rows or len(rows) != len({row["case_name"] for row in rows}):
        raise RuntimeError(f"Invalid V15 cache {path}: {len(rows)} rows")
    return rows


def valid_candidate(row: dict | None) -> bool:
    if row is None or row.get("transform") is None:
        return False
    if row.get("fit_failed") or row.get("da3_fit_failed"):
        return False
    required = (
        "camera_translation_error_m",
        "camera_rotation_error_deg",
        "human_root_jump_m",
    )
    return all(key in row and np.isfinite(float(row[key])) for key in required)


def distribution(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def aggregate(rows: list[dict], method: str) -> dict | None:
    selected = [row for row in rows if method in row["methods"]]
    if not selected:
        return None
    camera = np.asarray(
        [row["methods"][method]["camera_translation_m"] for row in selected],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [row["methods"][method]["camera_rotation_deg"] for row in selected],
        dtype=np.float64,
    )
    root = np.asarray(
        [row["methods"][method]["human_root_jump_m"] for row in selected],
        dtype=np.float64,
    )
    fixed_camera = np.asarray(
        [row["methods"]["fixed_explicit"]["camera_translation_m"] for row in selected],
        dtype=np.float64,
    )
    fixed_root = np.asarray(
        [row["methods"]["fixed_explicit"]["human_root_jump_m"] for row in selected],
        dtype=np.float64,
    )
    camera_gain = fixed_camera - camera
    root_gain = fixed_root - root
    return {
        "camera_translation_m": distribution(camera),
        "camera_rotation_deg": distribution(rotation),
        "visible_human_root_jump_m": distribution(root),
        "camera_improved_rate": float(np.mean(camera_gain > 0.0)),
        "root_jump_improved_rate": float(np.mean(root_gain > 0.0)),
        "both_improved_rate": float(np.mean((camera_gain > 0.0) & (root_gain > 0.0))),
        "camera_gain_but_root_harmed_010m_rate": float(
            np.mean((camera_gain > 0.10) & (root_gain < -0.10))
        ),
        "camera_gain_but_root_harmed_025m_rate": float(
            np.mean((camera_gain > 0.10) & (root_gain < -0.25))
        ),
        "joint_success_rate": float(np.mean((camera < 1.0) & (root < 0.25))),
        "camera_root_error_correlation": correlation(camera, root),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for set_value in args.sets:
        set_name, root = parse_set(set_value)
        for case in load_cases(root):
            window = case["windows"][str(args.window)]
            method_rows: dict[str, dict] = {}
            for method, (kind, key) in METHODS.items():
                candidate = (
                    case["baselines"].get(key)
                    if kind == "baseline"
                    else window["candidates"].get(key)
                )
                if not valid_candidate(candidate):
                    continue
                method_rows[method] = {
                    "camera_translation_m": float(candidate["camera_translation_error_m"]),
                    "camera_rotation_deg": float(candidate["camera_rotation_error_deg"]),
                    "human_root_jump_m": float(candidate["human_root_jump_m"]),
                }
            if "fixed_explicit" not in method_rows:
                raise RuntimeError(f"Missing Fixed Explicit for {case['case_name']}")
            rows.append(
                {
                    "set": set_name,
                    "case_name": case["case_name"],
                    "source": case["record"]["source"],
                    "methods": method_rows,
                }
            )

    trusted_rotation_rows = [
        row
        for row in rows
        if "vggt_coarse" in row["methods"]
        and row["methods"]["vggt_coarse"]["camera_rotation_deg"]
        <= float(args.max_rotation_deg)
    ]
    method_names = tuple(METHODS)
    sources = sorted({row["source"] for row in rows})
    sets = sorted({row["set"] for row in rows})
    report = {
        "experiment": "V33 joint camera-human translation candidate audit",
        "protocol": {
            "window": str(args.window),
            "gt_camera_used_for_offline_evaluation_only": True,
            "visible_human_root_jump_is_not_gt_human_error": True,
            "trusted_rotation_threshold_deg": float(args.max_rotation_deg),
            "runtime_selector_used": False,
        },
        "case_count": len(rows),
        "trusted_rotation_case_count": len(trusted_rotation_rows),
        "overall": {method: aggregate(rows, method) for method in method_names},
        "trusted_rotation": {
            method: aggregate(trusted_rotation_rows, method) for method in method_names
        },
        "by_set_trusted_rotation": {
            set_name: {
                method: aggregate(
                    [row for row in trusted_rotation_rows if row["set"] == set_name],
                    method,
                )
                for method in method_names
            }
            for set_name in sets
        },
        "by_source_trusted_rotation": {
            source: {
                method: aggregate(
                    [row for row in trusted_rotation_rows if row["source"] == source],
                    method,
                )
                for method in method_names
            }
            for source in sources
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact = {
        method: {
            "count": value["camera_translation_m"]["count"],
            "camera_mean_m": value["camera_translation_m"]["mean"],
            "root_jump_mean_m": value["visible_human_root_jump_m"]["mean"],
            "both_improved_rate": value["both_improved_rate"],
            "camera_gain_root_harm_010m": value[
                "camera_gain_but_root_harmed_010m_rate"
            ],
        }
        for method, value in report["trusted_rotation"].items()
        if value is not None
    }
    print(json.dumps(compact, indent=2))
    print(f">> wrote {args.output}")


if __name__ == "__main__":
    main()
