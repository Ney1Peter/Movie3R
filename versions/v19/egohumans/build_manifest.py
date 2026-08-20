#!/usr/bin/env python3
"""Build deterministic EgoHumans-CS100 manifests from GT-only audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v19.egohumans.dataset import (
    CLIP_LENGTH,
    POST_COUNT,
    PRE_COUNT,
    PROTOCOL_NAME,
    PROTOCOL_SEED,
    atomic_json,
)


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    text = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in rows
    )
    return hashlib.sha256((text + "\n").encode()).hexdigest()


def case_rows(audit: dict[str, Any], split: str, one_pair: bool) -> list[dict[str, Any]]:
    frames = [int(value) for value in audit["clip_frames"]]
    if len(frames) != CLIP_LENGTH or frames != list(range(frames[0], frames[0] + CLIP_LENGTH)):
        raise ValueError(f"Audit clip is not one contiguous CS100 window: {audit['capture_name']}")
    pairs = list(audit["selected_protocol_pairs"])
    if one_pair:
        pairs = pairs[:1]
    action = str(audit["archive_entry"]).split("/", 1)[0]
    capture = str(audit["capture_name"])
    output = []
    for pair in pairs:
        case_id = (
            f"ego_{split}_{action}_{capture}_{pair['angle_stratum']}_"
            f"{pair['pre_camera']}_{pair['post_camera']}_b{frames[PRE_COUNT]:05d}"
        )
        output.append(
            {
                "protocol": PROTOCOL_NAME,
                "protocol_seed": PROTOCOL_SEED,
                "split": split,
                "case_id": case_id,
                "archive_entry": audit["archive_entry"],
                "sequence": action,
                "capture": capture,
                "capture_relative": audit["capture_relative"],
                "pre_camera": pair["pre_camera"],
                "post_camera": pair["post_camera"],
                "pre_frame_numbers": frames[:PRE_COUNT],
                "post_frame_numbers": frames[PRE_COUNT : PRE_COUNT + POST_COUNT],
                "boundary_frame": frames[PRE_COUNT],
                "boundary_index": PRE_COUNT,
                "clip_length": CLIP_LENGTH,
                "fps": float(audit["fps"]),
                "angle_stratum": pair["angle_stratum"],
                "camera_rotation_span_deg_evaluator_only": float(pair["angle_deg"]),
                "camera_center_baseline_m_evaluator_only": float(pair["baseline_m"]),
                "person_count_evaluator_only": int(audit["identity_count"]),
                "identities_evaluator_only": list(audit["identities"]),
                "minimum_visible_people_at_boundary_evaluator_only": int(
                    pair["minimum_visible_people_at_boundary"]
                ),
                "selection_depends_on_model_result": False,
                "gt_available_to_runtime": False,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--one-pair", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.audits)]
    rows = [row for audit in audits for row in case_rows(audit, args.split, args.one_pair)]
    rows.sort(key=lambda row: (row["sequence"], row["capture"], row["angle_stratum"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(args.output)
    digest = hashlib.sha256(text.encode()).hexdigest()
    spec = {
        "schema_version": "Movie3R-v19-EgoHumans-manifest-spec-v1",
        "protocol": PROTOCOL_NAME,
        "seed": PROTOCOL_SEED,
        "split": args.split,
        "manifest": str(args.output.resolve()),
        "manifest_sha256": digest,
        "rows_sha256_recomputed": rows_sha256(rows),
        "case_count": len(rows),
        "capture_count": len({row["capture"] for row in rows}),
        "action_count": len({row["sequence"] for row in rows}),
        "clip_length": CLIP_LENGTH,
        "boundary_index": PRE_COUNT,
        "one_pair": bool(args.one_pair),
        "construction": "GT calibration/visibility only; no model result used",
        "runtime_gt_access": False,
    }
    atomic_json(args.output.with_suffix(".spec.json"), spec)
    print(json.dumps(spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
