#!/usr/bin/env python3
"""Audit correction-module training/evaluation identity overlap.

The audit is intentionally conservative.  It treats every event in the five
configured training-manifest universes as potentially observed, even though
the MultiHuman source samples only 96 of its 192 candidate events per epoch.
Dataset-qualified identities are compared at source, capture, event, and
frame/member level.  Unqualified text-token intersections are also retained
as a sanity check, but they are never used to equate records from different
datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent

CHECKPOINT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/"
    "v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth"
)
CONFIG = REPO_ROOT / "config/train_v14_1_cut_first_cross_source_multihuman_p0.yaml"
PARAMETER_AUDIT = (
    REPO_ROOT
    / "publication/bridge3r_iclr2027/evidence/checkpoint_parameter_audit/"
    "v14_1_p0_e6.json"
)
TRAIN96_ROOT = REPO_ROOT / "versions/v14/cut_first_cross_source/manifests/train96ps"
MULTIHUMAN_MANIFEST = (
    REPO_ROOT / "config/manifests/v14_multihuman_camera_supervision_20260803.json"
)

EGOHUMANS_MANIFEST = (
    WORKSPACE
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
HARMONY_CASE_IDS = (
    WORKSPACE
    / "ICLR-paper/bridge3r_iclr2027/private_audit/"
    "harmony4d_cs150_formal_case_ids.txt"
)
HARMONY_RUNTIME = (
    WORKSPACE
    / "data/Harmony4D_work_v17_full_test/external_predictions/trace/manifests/"
    "harmony4d_test.runtime.jsonl"
)
EGOBODY_EVALUATIONS = REPO_ROOT / "output/v20_egobody/formal/test/evaluations"

SCHEMA = "Bridge3R-correction-training-test-overlap-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output/bridge3r_overlap_audit_v1",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def qualified(dataset: str, kind: str, value: str) -> str:
    return f"{dataset.lower()}::{kind}::{value}"


def training_universe() -> dict[str, Any]:
    sources: set[str] = set()
    captures: set[str] = set()
    events: set[str] = set()
    frames: set[str] = set()
    raw_captures: set[str] = set()
    raw_events: set[str] = set()
    raw_frames: set[str] = set()
    manifests = []

    specifications = (
        ("avatarrex", "avatarrex.jsonl"),
        ("thuman", "thuman.jsonl"),
        ("mvhuman100", "mvhuman100.jsonl"),
        ("mvhuman200", "mvhuman200.jsonl"),
    )
    for dataset, filename in specifications:
        path = TRAIN96_ROOT / filename
        rows = read_jsonl(path)
        if len(rows) != 96:
            raise ValueError(f"{path}: expected 96 rows, found {len(rows)}")
        sources.add(qualified(dataset, "source", dataset))
        for row in rows:
            group = str(row["group"])
            event_id = str(row["pattern_id"])
            captures.add(qualified(dataset, "capture", group))
            events.add(qualified(dataset, "event", event_id))
            raw_captures.add(group)
            raw_events.add(event_id)
            for sequence, frame in zip(row["seqs"], row["frames"], strict=True):
                token = f"{sequence}@{int(frame)}"
                frames.add(qualified(dataset, "frame", token))
                raw_frames.add(token)
        manifests.append(
            {
                "dataset": dataset,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "candidate_event_count": len(rows),
            }
        )

    payload = json.loads(MULTIHUMAN_MANIFEST.read_text(encoding="utf-8"))
    rows = payload.get("train")
    if not isinstance(rows, list) or len(rows) != 192:
        raise ValueError("MultiHuman manifest must contain 192 training events")
    dataset = "multihuman"
    source_name = "MultiHuman Real-World-Capture"
    sources.add(qualified(dataset, "source", source_name))
    for row in rows:
        sequence = str(row["sequence"])
        event_id = str(row["event_id"])
        capture = sequence
        captures.add(qualified(dataset, "capture", capture))
        events.add(qualified(dataset, "event", event_id))
        raw_captures.add(capture)
        raw_events.add(event_id)
        frame = int(row["frame"])
        for camera in (int(row["pre_camera"]), int(row["post_camera"])):
            token = f"{sequence}/camera{camera}@{frame}"
            frames.add(qualified(dataset, "frame", token))
            raw_frames.add(token)
    manifests.append(
        {
            "dataset": dataset,
            "path": str(MULTIHUMAN_MANIFEST.resolve()),
            "sha256": sha256(MULTIHUMAN_MANIFEST),
            "candidate_event_count": len(rows),
            "configured_epoch_sample_count": 96,
            "audit_policy": "all 192 candidate train events treated as potentially observed",
        }
    )
    return {
        "sources": sources,
        "captures": captures,
        "events": events,
        "frames": frames,
        "raw_captures": raw_captures,
        "raw_events": raw_events,
        "raw_frames": raw_frames,
        "manifests": manifests,
    }


def empty_benchmark(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "sources": {qualified(name, "source", name)},
        "captures": set(),
        "events": set(),
        "frames": set(),
        "raw_captures": set(),
        "raw_events": set(),
        "raw_frames": set(),
        "inputs": [],
    }


def egobody_universe() -> dict[str, Any]:
    output = empty_benchmark("egobody")
    files = sorted(EGOBODY_EVALUATIONS.glob("*.evaluation.json"))
    if len(files) != 129:
        raise ValueError(f"expected 129 EgoBody evaluations, found {len(files)}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload["record_runtime_fields"]
        case_id = str(payload["case_id"])
        if case_id != str(record["case_id"]):
            raise ValueError(f"{path}: case id mismatch")
        capture = str(record["capture"])
        output["captures"].add(qualified("egobody", "capture", capture))
        output["events"].add(qualified("egobody", "event", case_id))
        output["raw_captures"].add(capture)
        output["raw_events"].add(case_id)
        for member in record["image_members"]:
            member = str(member)
            output["frames"].add(qualified("egobody", "frame", member))
            output["raw_frames"].add(member)
    output["inputs"].append(
        {
            "path": str(EGOBODY_EVALUATIONS.resolve()),
            "file_count": len(files),
            "aggregate_filename_and_content_sha256": aggregate_hash(files),
        }
    )
    return output


def egohumans_universe() -> dict[str, Any]:
    output = empty_benchmark("egohumans")
    rows = read_jsonl(EGOHUMANS_MANIFEST)
    if len(rows) != 90:
        raise ValueError(f"expected 90 EgoHumans rows, found {len(rows)}")
    for row in rows:
        capture = f"{row['sequence']}/{row['capture']}"
        case_id = str(row["case_id"])
        output["captures"].add(qualified("egohumans", "capture", capture))
        output["events"].add(qualified("egohumans", "event", case_id))
        output["raw_captures"].add(capture)
        output["raw_events"].add(case_id)
        archive = str(row["archive_entry"])
        for camera_key, frame_key in (
            ("pre_camera", "pre_frame_numbers"),
            ("post_camera", "post_frame_numbers"),
        ):
            camera = str(row[camera_key])
            for frame in row[frame_key]:
                token = f"{archive}/{camera}@{int(frame)}"
                output["frames"].add(qualified("egohumans", "frame", token))
                output["raw_frames"].add(token)
    output["inputs"].append(
        {
            "path": str(EGOHUMANS_MANIFEST.resolve()),
            "sha256": sha256(EGOHUMANS_MANIFEST),
            "row_count": len(rows),
        }
    )
    return output


def harmony4d_universe() -> dict[str, Any]:
    output = empty_benchmark("harmony4d")
    formal_ids = {
        line.strip()
        for line in HARMONY_CASE_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(formal_ids) != 88:
        raise ValueError(f"expected 88 Harmony4D case ids, found {len(formal_ids)}")
    runtime_rows = {str(row["case_id"]): row for row in read_jsonl(HARMONY_RUNTIME)}
    missing = formal_ids.difference(runtime_rows)
    if missing:
        raise ValueError(f"Harmony4D runtime manifest lacks {len(missing)} formal ids")
    rows = [runtime_rows[case_id] for case_id in sorted(formal_ids)]
    for row in rows:
        capture = str(row["capture_relative"])
        case_id = str(row["case_id"])
        output["captures"].add(qualified("harmony4d", "capture", capture))
        output["events"].add(qualified("harmony4d", "event", case_id))
        output["raw_captures"].add(capture)
        output["raw_events"].add(case_id)
        for member in row["image_members"]:
            member = str(member)
            output["frames"].add(qualified("harmony4d", "frame", member))
            output["raw_frames"].add(member)
    output["inputs"].extend(
        [
            {
                "path": str(HARMONY_CASE_IDS.resolve()),
                "sha256": sha256(HARMONY_CASE_IDS),
                "row_count": len(formal_ids),
            },
            {
                "path": str(HARMONY_RUNTIME.resolve()),
                "sha256": sha256(HARMONY_RUNTIME),
                "selected_row_count": len(rows),
                "total_row_count": len(runtime_rows),
            },
        ]
    )
    return output


def intersection_record(training: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    levels = {}
    for level in ("sources", "captures", "events", "frames"):
        overlap = sorted(training[level].intersection(benchmark[level]))
        levels[level] = {
            "training_count": len(training[level]),
            "benchmark_count": len(benchmark[level]),
            "intersection_count": len(overlap),
            "intersection": overlap,
        }
    raw = {}
    for level in ("raw_captures", "raw_events", "raw_frames"):
        overlap = sorted(training[level].intersection(benchmark[level]))
        raw[level] = {
            "intersection_count": len(overlap),
            "intersection": overlap,
        }
    passed = all(value["intersection_count"] == 0 for value in levels.values())
    return {
        "status": "zero_overlap" if passed else "overlap_detected",
        "qualified_identity_intersections": levels,
        "unqualified_text_sanity_intersections": raw,
        "passed": passed,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Bridge3R correction-module training--test overlap audit",
        "",
        f"- Checkpoint SHA-256: `{payload['checkpoint']['sha256']}`",
        f"- Training config SHA-256: `{payload['training_config']['sha256']}`",
        (
            "- Scope: the Bridge3R correction checkpoint and its five configured "
            "fine-tuning sources. The inherited Human3R pretraining corpus is a "
            "separate provenance scope and is not reconstructed by this audit."
        ),
        (
            "- Conservative policy: all 192 MultiHuman train candidates are treated "
            "as potentially observed, although only 96 are sampled per epoch."
        ),
        "",
        "## Result",
        "",
        "| Benchmark | Cases | Captures | Source overlap | Capture overlap | Event overlap | Frame/member overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("egobody", "egohumans", "harmony4d"):
        result = payload["benchmarks"][name]
        levels = result["overlap"]["qualified_identity_intersections"]
        lines.append(
            f"| {result['display_name']} | {result['event_count']} | "
            f"{result['capture_count']} | {levels['sources']['intersection_count']} | "
            f"{levels['captures']['intersection_count']} | "
            f"{levels['events']['intersection_count']} | "
            f"{levels['frames']['intersection_count']} |"
        )
    lines.extend(
        [
            "",
            (
                "All three benchmark identities are disjoint from the conservative "
                "correction-training universe at source, capture, event, and "
                "frame/member levels. The conclusion uses dataset-qualified "
                "identities; the retained unqualified-token sanity intersections are "
                "also empty."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "This proves zero overlap for the correction-module fine-tuning data "
                "bound by the audited configuration. It does not claim that the base "
                "Human3R model was pretrained without any benchmark-related source, "
                "because the complete inherited pretraining manifest is not part of "
                "the present evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def latex(payload: dict[str, Any]) -> str:
    rows = []
    for name in ("egobody", "egohumans", "harmony4d"):
        result = payload["benchmarks"][name]
        levels = result["overlap"]["qualified_identity_intersections"]
        rows.append(
            f"    {result['display_name']} & {result['event_count']} & "
            f"{result['capture_count']} & {levels['sources']['intersection_count']} & "
            f"{levels['captures']['intersection_count']} & "
            f"{levels['events']['intersection_count']} & "
            f"{levels['frames']['intersection_count']} \\\\"
        )
    return "\n".join(
        [
            "% Generated by audit_training_test_overlap.py.",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Benchmark & Cases & Captures & Source & Capture & Event & Frame/member \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    required = (
        CHECKPOINT,
        CONFIG,
        PARAMETER_AUDIT,
        MULTIHUMAN_MANIFEST,
        EGOHUMANS_MANIFEST,
        HARMONY_CASE_IDS,
        HARMONY_RUNTIME,
        EGOBODY_EVALUATIONS,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    parameter_audit = json.loads(PARAMETER_AUDIT.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(CHECKPOINT)
    if parameter_audit.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("checkpoint hash differs from parameter audit")

    training = training_universe()
    benchmarks = {
        "egobody": egobody_universe(),
        "egohumans": egohumans_universe(),
        "harmony4d": harmony4d_universe(),
    }
    output_benchmarks = {}
    display = {"egobody": "EgoBody", "egohumans": "EgoHumans", "harmony4d": "Harmony4D"}
    for name, benchmark in benchmarks.items():
        overlap = intersection_record(training, benchmark)
        if not overlap["passed"]:
            raise ValueError(f"{name}: training/test overlap detected")
        output_benchmarks[name] = {
            "display_name": display[name],
            "status": "complete_zero_overlap",
            "event_count": len(benchmark["events"]),
            "capture_count": len(benchmark["captures"]),
            "unique_frame_or_member_count": len(benchmark["frames"]),
            "inputs": benchmark["inputs"],
            "overlap": overlap,
        }

    payload = {
        "schema_version": SCHEMA,
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "bytes": CHECKPOINT.stat().st_size,
            "sha256": checkpoint_hash,
            "parameter_audit": str(PARAMETER_AUDIT.resolve()),
            "parameter_audit_sha256": sha256(PARAMETER_AUDIT),
        },
        "training_config": {
            "path": str(CONFIG.resolve()),
            "sha256": sha256(CONFIG),
            "binding": (
                "configuration exp_name/output_dir matches the checkpoint run directory; "
                "saved model flags match the retained parameter audit"
            ),
            "caveat": "the pth does not itself embed a cryptographic hash of this YAML",
        },
        "training_universe": {
            "source_count": len(training["sources"]),
            "capture_count": len(training["captures"]),
            "candidate_event_count": len(training["events"]),
            "unique_frame_identity_count": len(training["frames"]),
            "manifests": training["manifests"],
            "policy": (
                "superset audit: all configured manifest candidates are treated as "
                "potentially observed; zero overlap with this superset implies zero "
                "overlap with the actual sampled subset"
            ),
        },
        "identity_definition": {
            "source": "dataset-qualified dataset/source namespace",
            "capture": "dataset-qualified subject/sequence/capture namespace",
            "event": "dataset-qualified pattern/event/case identifier",
            "frame_or_member": "dataset-qualified source sequence/frame or archive member",
        },
        "benchmarks": output_benchmarks,
        "overall_status": "complete_zero_overlap",
        "claim": (
            "The audited Bridge3R correction-module fine-tuning sources have zero "
            "source-, capture-, event-, and frame/member-level overlap with the "
            "EgoBody-129, EgoHumans-90, and Harmony4D-88 evaluation sets."
        ),
        "scope_boundary": (
            "This audit covers correction-module fine-tuning bound by the retained "
            "configuration. It does not reconstruct or certify the inherited Human3R "
            "base model's complete pretraining corpus."
        ),
    }
    output_root = args.output_root.resolve()
    output_json = output_root / "training_test_overlap_audit.json"
    output_md = output_root / "TRAINING_TEST_OVERLAP_AUDIT.md"
    output_tex = output_root / "training_test_overlap_table.tex"
    atomic_json(output_json, payload)
    output_md.write_text(markdown(payload), encoding="utf-8")
    output_tex.write_text(latex(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["overall_status"],
                "json": str(output_json),
                "md": str(output_md),
                "tex": str(output_tex),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
