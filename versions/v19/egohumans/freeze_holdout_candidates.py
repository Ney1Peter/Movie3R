#!/usr/bin/env python3
"""Freeze at most three development-selected candidates before holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


BASELINE = "v16_0_m15_geometry"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FALLBACK = REPO_ROOT / "versions/v17/harmony4d/frozen_multicue_candidate.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--fallback-candidate", type=Path, default=DEFAULT_FALLBACK)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize(name: str, value: dict[str, Any]) -> dict[str, Any]:
    if "geometry" in value:
        result = dict(value)
        result["name"] = name
        result.setdefault("identity", None)
        return result
    geometry = dict(value)
    geometry["name"] = name
    return {"name": name, "geometry": geometry, "identity": None}


def signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    geometry = candidate["geometry"]
    return (
        geometry.get("boundary_kind", "none"),
        float(geometry.get("camera_alpha", 1.0)),
        bool(geometry.get("use_velocity_target", False)),
    )


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.max_candidates) <= 3:
        raise ValueError("Holdout may receive one to three candidates")
    payload = json.loads(args.development_summary.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "Movie3R-v19-EgoHumans-CS100-summary-v1":
        raise ValueError("Unexpected development summary schema")
    if payload.get("split") != "development" or not payload.get("protocol", {}).get("parameter_selection_allowed"):
        raise ValueError("Candidates may only be frozen from development")
    passing = list(payload.get("passing_development_candidates", []))
    if not passing:
        raise ValueError("No candidate passed the pre-registered development gate")
    normalized = []
    for name in passing:
        config = payload["methods"][name].get("candidate")
        if not isinstance(config, dict):
            continue
        normalized.append(normalize(name, config))
    selected = []
    seen_signatures = set()
    innovation_limit = max(0, int(args.max_candidates) - 1)
    if innovation_limit:
        for candidate in normalized:
            key = signature(candidate)
            if key in seen_signatures:
                continue
            selected.append(candidate)
            seen_signatures.add(key)
            if len(selected) == innovation_limit:
                break
    if len(selected) < innovation_limit:
        selected_names = {row["name"] for row in selected}
        for candidate in normalized:
            if candidate["name"] in selected_names:
                continue
            selected.append(candidate)
            selected_names.add(candidate["name"])
            if len(selected) == innovation_limit:
                break
    fallback_payload = json.loads(args.fallback_candidate.read_text(encoding="utf-8"))
    fallback_rows = [
        row for row in fallback_payload.get("candidates", [])
        if str(row.get("name")) != BASELINE
    ]
    if len(fallback_rows) != 1:
        raise ValueError(f"Expected one v17 fallback in {args.fallback_candidate}")
    fallback = normalize(str(fallback_rows[0]["name"]), fallback_rows[0])
    if fallback["name"] not in {row["name"] for row in selected}:
        selected.append(fallback)
    baseline = {
        "name": BASELINE,
        "geometry": {"name": BASELINE},
        "identity": None,
    }
    output = {
        "schema_version": "Movie3R-v19-EgoHumans-holdout-candidates-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "source_split": "development",
        "development_summary": str(args.development_summary.resolve()),
        "development_summary_sha256": sha256(args.development_summary),
        "selection_rule": (
            "pre-registered passing candidates in composite rank order; prefer distinct "
            "boundary-kind/camera-alpha/velocity signatures; reserve one slot for the "
            "frozen v17 exact-fallback reference; at most three"
        ),
        "selected_names": [row["name"] for row in selected],
        "candidates": [baseline, *selected],
        "frozen_before_holdout": True,
        "holdout_or_test_metrics_read": False,
        "fallback_source": str(args.fallback_candidate.resolve()),
        "fallback_source_sha256": sha256(args.fallback_candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, args.output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
