#!/usr/bin/env python3
"""Build deterministic cross-shot manifests from evaluator-only schema audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.protocol import (  # noqa: E402
    PROTOCOL_NAME,
    PROTOCOL_SEED,
    manifest_sha256,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-count", type=int, default=75)
    parser.add_argument("--post-count", type=int, default=75)
    parser.add_argument("--one-pair", action="store_true", help="Use only the first selected angle pair")
    return parser.parse_args()


def case_rows(audit: dict[str, Any], split: str, pre_count: int, post_count: int, one_pair: bool) -> list[dict[str, Any]]:
    boundary = int(audit["audit_frame"])
    pre = list(range(boundary - pre_count, boundary))
    post = list(range(boundary, boundary + post_count))
    if min(pre + post) < int(audit["frame_min"]) or max(pre + post) > int(audit["frame_max"]):
        raise ValueError(f"Centred clip exceeds sequence bounds for {audit['archive_entry']}")
    pairs = list(audit["selected_protocol_pairs"])
    if one_pair:
        pairs = pairs[:1]
    sequence = Path(str(audit["archive_entry"])).stem
    capture = str(audit["sequence_root_name"])
    rows = []
    for pair in pairs:
        case_id = (
            f"h4d_{split}_{sequence}_{capture}_{pair['angle_stratum']}_"
            f"{pair['pre_camera']}_{pair['post_camera']}_b{boundary:05d}"
        )
        rows.append({
            "protocol": PROTOCOL_NAME,
            "protocol_seed": PROTOCOL_SEED,
            "split": split,
            "case_id": case_id,
            "archive_entry": audit["archive_entry"],
            "sequence": sequence,
            "capture": capture,
            "capture_relative": audit.get(
                "capture_relative", f"{audit['capture_group_name']}/{audit['sequence_root_name']}"
            ),
            "pre_camera": pair["pre_camera"],
            "post_camera": pair["post_camera"],
            "pre_frame_numbers": pre,
            "post_frame_numbers": post,
            "boundary_frame": boundary,
            "boundary_index": pre_count,
            "clip_length": pre_count + post_count,
            "fps": float(audit["fps"]),
            "angle_stratum": pair["angle_stratum"],
            "camera_rotation_span_deg_evaluator_only": float(pair["angle_deg"]),
            "camera_center_baseline_m_evaluator_only": float(pair["baseline_m"]),
            "person_count_evaluator_only": int(audit["person_count_at_audit"]),
            "identities_evaluator_only": list(audit["identities"]),
            "selection_depends_on_model_result": False,
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.pre_count < 1 or args.post_count < 1:
        raise ValueError("pre/post counts must be positive")
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.audits)]
    rows = []
    for audit in audits:
        rows.extend(case_rows(audit, args.split, args.pre_count, args.post_count, args.one_pair))
    rows.sort(key=lambda row: (row["sequence"], row["angle_stratum"], row["pre_camera"], row["post_camera"]))
    digest = write_jsonl(args.output, rows)
    spec = {
        "protocol": PROTOCOL_NAME,
        "seed": PROTOCOL_SEED,
        "split": args.split,
        "manifest": str(args.output.resolve()),
        "manifest_sha256": digest,
        "case_count": len(rows),
        "sequence_count": len({row["sequence"] for row in rows}),
        "clip_length": args.pre_count + args.post_count,
        "boundary_index": args.pre_count,
        "construction": "GT calibration/visibility only; no model result used",
        "causal_boundary": "last pre plus first post and prior state; no future frame",
        "rows_sha256_recomputed": manifest_sha256(rows),
    }
    spec_path = args.output.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
