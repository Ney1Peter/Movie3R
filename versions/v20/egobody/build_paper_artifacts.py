#!/usr/bin/env python3
"""Build audited ICLR-facing artifacts for the completed EgoBody v20 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
FORMAL_ROOT = REPO_ROOT / "output/v20_egobody/formal"
DEFAULT_FINAL = (
    WORKSPACE_ROOT / "data/EgoBody_work_v20/frozen/frozen_final_candidate.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "ICLR-paper/movie3r_iclr2027_draft/artifacts/egobody_v20"
)
DEFAULT_MULTI_THUMBS = WORKSPACE_ROOT / "paper/Multi-THuMBS.pdf"

AGGREGATE_SCHEMA = "Bridge3R-EgoBody-CS150-aggregate-v1"
STATE_SCHEMA = "Bridge3R-EgoBody-CS150-protocol-state-v1"
FINAL_SCHEMA = "Bridge3R-EgoBody-frozen-final-candidate-v1"
RUNTIME_SCHEMA = "Bridge3R-EgoBody-CS150-runtime-cache-v1"
PROBE_SCHEMA = "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1"
ARTIFACT_SCHEMA = "Bridge3R-EgoBody-CS150-paper-artifacts-v2"
PROTOCOL = "Bridge3R-EgoBody-CS150-v1"
PARENT = "v16_0_m15_geometry"
FROZEN_FINAL_METHOD = "v19_ungated_translation_b050"
TEST_LEDGER_SCHEMA = "Bridge3R-EgoBody-test-consumption-v1"
FORMAL_SCOPE = {
    "development": {"case_count": 195, "recording_count": 65},
    "holdout": {"case_count": 48, "recording_count": 16},
    "test": {"case_count": 129, "recording_count": 43},
}
ANGLE_STRATA = ("small", "medium", "extreme")
EXPECTED_METHODS = (
    "m0_strict_human3r",
    "m0r_original_clean_reset",
    "m1_current_clean_reset",
    "m3_b0_only",
    "m15_v17_gated_parent",
    PARENT,
    FROZEN_FINAL_METHOD,
)

TABLE_METRICS = (
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "PA-MPJPE_mm",
    "MPVPE_mm",
    "Accel_mm_frame2",
    "RTE_H3R_percent",
    "ATE_Sim3_m",
    "ATE_SE3_m",
    "Boundary_camera_t_m",
    "Boundary_camera_R_deg",
    "Boundary_root_m",
    "Post_root_m",
    "Seam_camera_t_m",
    "Seam_camera_R_deg",
    "IDF1",
    "IDs",
    "Coverage",
    "Seam_root_m",
    "CHRGE_m",
    "Seam_CHRGE_m",
    "Pair_vector_m",
    "ROE_joint_proxy_deg",
    "Jitter_H3R",
    "Foot_sliding_cm",
    "Detection_precision",
)
PRIMARY_TABLE_METRICS = (
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "RTE_H3R_percent",
    "ATE_Sim3_m",
    "IDF1",
    "IDs",
)
LOCAL_TABLE_METRICS = (
    "MPJPE_mm",
    "PA-MPJPE_mm",
    "MPVPE_mm",
    "ATE_SE3_m",
)
TEMPORAL_TABLE_METRICS = (
    "Accel_mm_frame2",
    "ROE_joint_proxy_deg",
    "Jitter_H3R",
    "Foot_sliding_cm",
    "Coverage",
    "Detection_precision",
)
BOUNDARY_CAMERA_TABLE_METRICS = (
    "Boundary_camera_t_m",
    "Boundary_camera_R_deg",
    "Seam_camera_t_m",
    "Seam_camera_R_deg",
)
BOUNDARY_HUMAN_TABLE_METRICS = (
    "Boundary_root_m",
    "Post_root_m",
    "Seam_root_m",
    "CHRGE_m",
    "Seam_CHRGE_m",
    "Pair_vector_m",
)
BOUNDARY_TABLE_METRICS = (
    *BOUNDARY_CAMERA_TABLE_METRICS,
    *BOUNDARY_HUMAN_TABLE_METRICS,
)
REQUIRED_PAPER_METRICS = tuple(
    dict.fromkeys(
        (
            *PRIMARY_TABLE_METRICS,
            *LOCAL_TABLE_METRICS,
            *TEMPORAL_TABLE_METRICS,
            *BOUNDARY_TABLE_METRICS,
        )
    )
)
METRIC_LABELS = {
    "W-MPJPE_mm": "W (mm) $\\downarrow$",
    "WA-MPJPE_mm": "WA (mm) $\\downarrow$",
    "MPJPE_mm": "MPJPE (mm) $\\downarrow$",
    "PA-MPJPE_mm": "PA-MPJPE (mm) $\\downarrow$",
    "MPVPE_mm": "MPVPE (mm) $\\downarrow$",
    "Accel_mm_frame2": "Accel (mm/frame$^2$) $\\downarrow$",
    "RTE_H3R_percent": "RTE-H3R (\\%) $\\downarrow$",
    "ATE_Sim3_m": "ATE-Sim3 (m) $\\downarrow$",
    "ATE_SE3_m": "ATE-SE3 (m) $\\downarrow$",
    "Boundary_camera_t_m": "B-Cam. T (m) $\\downarrow$",
    "Boundary_camera_R_deg": "B-Cam. R ($^\\circ$) $\\downarrow$",
    "Boundary_root_m": "B-Root (m) $\\downarrow$",
    "Post_root_m": "Post-root (m) $\\downarrow$",
    "Seam_camera_t_m": "Seam-Cam. T (m) $\\downarrow$",
    "Seam_camera_R_deg": "Seam-Cam. R ($^\\circ$) $\\downarrow$",
    "IDF1": "IDF1 $\\uparrow$",
    "IDs": "IDs $\\downarrow$",
    "Coverage": "Coverage $\\uparrow$",
    "Seam_root_m": "Seam-root (m) $\\downarrow$",
    "CHRGE_m": "CHRGE (m) $\\downarrow$",
    "Seam_CHRGE_m": "Seam-CHRGE (m) $\\downarrow$",
    "Pair_vector_m": "Pair vector (m) $\\downarrow$",
    "ROE_joint_proxy_deg": "ROE proxy ($^\\circ$) $\\downarrow$",
    "Jitter_H3R": "Jitter-H3R $\\downarrow$",
    "Foot_sliding_cm": "Foot sliding (cm) $\\downarrow$",
    "Detection_precision": "Person precision $\\uparrow$",
}
DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "m0r_original_clean_reset": "Original Human3R reset (oracle cut)",
    "m1_current_clean_reset": "Bridge3R clean reset (oracle cut)",
    "m3_b0_only": "Bridge3R B0 (oracle cut)",
    "m15_v17_gated_parent": "Bridge3R runtime parent (oracle cut)",
    PARENT: "Bridge3R geometry parent (causal detector)",
    "v17_harmony_multicue_safe": "Bridge3R-v17 MultiCue-Safe",
    "v19_ungated_translation_b050": "Bridge3R-v19 ungated",
}
ANGLE_ORDER = {value: index for index, value in enumerate(ANGLE_STRATA)}
MULTI_THUMBS_EGOBODY = {
    "W-MPJPE_mm": 99.2,
    "WA-MPJPE_mm": 72.8,
    "MPJPE_mm": 72.0,
    "MPVPE_mm": 94.9,
    "Accel_mm_frame2": 6.0,
    "IDs": 0.0,
    "ATE_source_defined": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development", type=Path, default=FORMAL_ROOT / "development/aggregate"
    )
    parser.add_argument(
        "--holdout", type=Path, default=FORMAL_ROOT / "holdout/aggregate"
    )
    parser.add_argument("--test", type=Path, default=FORMAL_ROOT / "test/aggregate")
    parser.add_argument("--final-candidate", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--multi-thumbs-pdf", type=Path, default=DEFAULT_MULTI_THUMBS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def average(values: Iterable[Any]) -> float | None:
    numbers = [value for item in values if (value := finite(item)) is not None]
    return mean(numbers) if numbers else None


def summary_path(value: Path) -> Path:
    path = value.resolve()
    return path / "summary.json" if path.is_dir() else path


def checked_path(
    payload: dict[str, Any], path_key: str, sha_key: str, owner: Path
) -> Path:
    path = Path(str(payload.get(path_key, ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{owner}: missing {path_key}: {path}")
    observed = str(payload.get(sha_key, ""))
    expected = sha256(path)
    if observed != expected:
        raise ValueError(f"{owner}: {sha_key} mismatch: {observed} vs {expected}")
    return path


def checked_state_manifest(
    state: dict[str, Any], path_key: str, sha_key: str, owner: Path
) -> Path:
    """Validate one state/run-identity manifest path and its current bytes."""

    run_identity = state.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError(f"{owner}: protocol state lacks run_identity")
    state_raw = state.get(path_key)
    identity_raw = run_identity.get(path_key)
    if not isinstance(state_raw, str) or not state_raw:
        raise ValueError(f"{owner}: protocol state lacks {path_key}")
    if not isinstance(identity_raw, str) or not identity_raw:
        raise ValueError(f"{owner}: run_identity lacks {path_key}")
    state_path = Path(state_raw).resolve()
    identity_path = Path(identity_raw).resolve()
    if state_path != identity_path:
        raise ValueError(
            f"{owner}: state/run_identity {path_key} path differs: "
            f"{state_path} vs {identity_path}"
        )
    if not state_path.is_file():
        raise FileNotFoundError(f"{owner}: missing {path_key}: {state_path}")
    declared_sha = str(run_identity.get(sha_key, ""))
    observed_sha = sha256(state_path)
    if declared_sha != observed_sha:
        raise ValueError(
            f"{owner}: run_identity {sha_key} mismatch: "
            f"{declared_sha} vs {observed_sha}"
        )
    return state_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL manifest: {path}")
    return rows


def validate_formal_scope(
    split: str,
    summary: dict[str, Any],
    state: dict[str, Any],
    runtime_manifest: Path,
    evaluator_manifest: Path,
    owner: Path,
) -> dict[str, int]:
    """Fail closed on the preregistered formal split and three-angle scope."""

    expected = FORMAL_SCOPE.get(split)
    if expected is None:
        raise ValueError(f"unregistered formal split: {split}")
    runtime_rows = read_jsonl(runtime_manifest)
    evaluator_rows = read_jsonl(evaluator_manifest)
    expected_cases = int(expected["case_count"])
    if len(runtime_rows) != expected_cases or len(evaluator_rows) != expected_cases:
        raise ValueError(
            f"{owner}: formal {split} manifest count mismatch: "
            f"runtime={len(runtime_rows)}, evaluator={len(evaluator_rows)}, "
            f"expected={expected_cases}"
        )
    runtime_ids = [str(row.get("case_id", "")) for row in runtime_rows]
    evaluator_ids = [str(row.get("case_id", "")) for row in evaluator_rows]
    if (
        any(not value for value in runtime_ids)
        or len(set(runtime_ids)) != expected_cases
        or runtime_ids != evaluator_ids
    ):
        raise ValueError(f"{owner}: runtime/evaluator case identity mismatch")
    run_identity = state.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError(f"{owner}: protocol state lacks run_identity")
    selected_ids = [str(value) for value in run_identity.get("selected_case_ids", [])]
    if selected_ids != runtime_ids:
        raise ValueError(f"{owner}: selected case IDs differ from frozen manifests")
    if (
        int(state.get("selected_case_count", -1)) != expected_cases
        or int(summary.get("selected_case_count", -1)) != expected_cases
    ):
        raise ValueError(f"{owner}: formal {split} selected case count mismatch")

    recordings: dict[str, set[str]] = defaultdict(set)
    for runtime_row, evaluator_row in zip(runtime_rows, evaluator_rows):
        if (
            runtime_row.get("protocol") != PROTOCOL
            or evaluator_row.get("protocol") != PROTOCOL
            or runtime_row.get("split") != split
            or evaluator_row.get("split") != split
        ):
            raise ValueError(f"{owner}: manifest protocol/split mismatch")
        recording = str(runtime_row.get("recording", ""))
        if not recording or str(evaluator_row.get("recording", "")) != recording:
            raise ValueError(f"{owner}: manifest recording mismatch")
        angle = str(evaluator_row.get("angle_stratum_evaluator_only", ""))
        if angle not in ANGLE_STRATA:
            raise ValueError(f"{owner}: invalid/missing angle stratum: {angle!r}")
        if angle in recordings[recording]:
            raise ValueError(f"{owner}: duplicate {angle} case for {recording}")
        recordings[recording].add(angle)
    expected_recordings = int(expected["recording_count"])
    if len(recordings) != expected_recordings or any(
        values != set(ANGLE_STRATA) for values in recordings.values()
    ):
        raise ValueError(
            f"{owner}: formal {split} recording/angle scope mismatch: "
            f"recordings={len(recordings)}, expected={expected_recordings}"
        )
    unavailable = int(summary.get("evaluator_unavailable_case_count", -1))
    evaluable = int(summary.get("case_count", -1))
    if unavailable < 0 or evaluable != expected_cases - unavailable:
        raise ValueError(f"{owner}: case accounting mismatch")
    reported_recordings = int(summary.get("recording_count", -1))
    if reported_recordings < 1 or reported_recordings > expected_recordings:
        raise ValueError(f"{owner}: invalid evaluable recording count")
    return {
        "selected_case_count": expected_cases,
        "structural_recording_count": expected_recordings,
        "evaluable_case_count": evaluable,
        "evaluator_unavailable_case_count": unavailable,
    }


def validate_final_candidate(payload: dict[str, Any], path: Path) -> str:
    expected_candidate = {
        "name": FROZEN_FINAL_METHOD,
        "geometry": {
            "name": FROZEN_FINAL_METHOD,
            "camera_alpha": 1.0,
            "boundary_kind": "translation",
            "boundary_blend": 0.5,
        },
        "identity": None,
    }
    if (
        payload.get("schema_version") != FINAL_SCHEMA
        or payload.get("protocol") != PROTOCOL
        or payload.get("frozen_before_test") is not True
        or payload.get("test_metrics_read") is not False
        or payload.get("source_candidate_name") != FROZEN_FINAL_METHOD
        or payload.get("fallback_to_parent") is not False
        or payload.get("candidates") != [expected_candidate]
        or Path(str(payload.get("frozen_artifact_path", ""))).resolve()
        != path.resolve()
    ):
        raise ValueError("invalid frozen final candidate contract")
    return FROZEN_FINAL_METHOD


def validate_test_ledger(
    ledger_path: Path,
    final_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    expected_path = final_path.with_suffix(
        final_path.suffix + ".test-consumption.json"
    ).resolve()
    if ledger_path.resolve() != expected_path:
        raise ValueError("Test consumption ledger is not at the frozen final suffix")
    ledger = load(ledger_path)
    run_identity = state.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("Test state lacks run_identity")
    output_root = Path(str(run_identity.get("output_root", ""))).resolve()
    if output_root.name != "test" or output_root.parent.name != "formal":
        raise ValueError("Test ledger output_root is not formal/test")
    expected = {
        "schema_version": TEST_LEDGER_SCHEMA,
        "candidate_json": str(final_path.resolve()),
        "candidate_json_sha256": sha256(final_path.resolve()),
        "run_identity_sha256": state.get("run_identity_sha256"),
        "output_root": str(output_root),
    }
    if ledger != expected:
        raise ValueError("Test consumption ledger content mismatch")
    candidates = state.get("candidate_json")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Test state does not contain exactly one frozen candidate")
    candidate = candidates[0]
    if (
        not isinstance(candidate, dict)
        or Path(str(candidate.get("path", ""))).resolve() != final_path.resolve()
        or candidate.get("sha256") != expected["candidate_json_sha256"]
        or candidate.get("source_candidate_name") != FROZEN_FINAL_METHOD
        or candidate.get("frozen_before_test") is not True
        or candidate.get("test_metrics_read") is not False
    ):
        raise ValueError("Test state candidate binding mismatch")
    return ledger


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    partial.write_bytes(value)
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def write_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def validate_bundle(value: Path, split: str) -> dict[str, Any]:
    path = summary_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = load(path)
    if summary.get("schema_version") != AGGREGATE_SCHEMA:
        raise ValueError(f"unexpected aggregate schema in {path}")
    if summary.get("split") != split or summary.get("parent") != PARENT:
        raise ValueError(f"split/parent mismatch in {path}")
    state_path = checked_path(summary, "protocol_state", "protocol_state_sha256", path)
    state = load(state_path)
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("status") != "complete"
        or state.get("split") != split
        or state.get("smoke_subset") is not False
        or state.get("max_cases") is not None
    ):
        raise ValueError(f"{path} is not bound to a completed formal {split} run")
    if summary.get("run_identity_sha256") != state.get("run_identity_sha256"):
        raise ValueError(f"aggregate/state identity mismatch in {path}")
    runtime_manifest = checked_state_manifest(
        state, "runtime_manifest", "runtime_manifest_sha256", state_path
    )
    evaluator_manifest = checked_state_manifest(
        state, "evaluator_manifest", "evaluator_manifest_sha256", state_path
    )
    case_metrics = checked_path(summary, "case_metrics", "case_metrics_sha256", path)
    recording_metrics = checked_path(
        summary, "recording_metrics", "recording_metrics_sha256", path
    )
    candidate_report = checked_path(
        summary, "candidate_report", "candidate_report_sha256", path
    )
    candidate_source = checked_path(
        summary, "candidate_source", "candidate_source_sha256", path
    )
    report = load(candidate_report)
    if report.get("schema_version") != PROBE_SCHEMA or report.get("errors") != []:
        raise ValueError(f"candidate report is incomplete: {candidate_report}")
    report_sha = sha256(candidate_report)
    state_reports = state.get("candidate_reports", {})
    if not isinstance(state_reports, dict) or not any(
        isinstance(value, dict)
        and value.get("status") == "complete"
        and value.get("output_sha256") == report_sha
        for value in state_reports.values()
    ):
        raise ValueError(
            f"candidate report SHA is absent from completed protocol state: "
            f"{candidate_report}"
        )
    if Path(str(report.get("candidate_source", ""))).resolve() != candidate_source:
        raise ValueError(f"candidate source mismatch in {candidate_report}")
    selected = int(summary.get("selected_case_count", -1))
    unavailable = int(summary.get("evaluator_unavailable_case_count", -1))
    if selected < 1 or unavailable < 0 or int(summary.get("case_count", -1)) != selected - unavailable:
        raise ValueError(f"case accounting mismatch in {path}")
    formal_scope = validate_formal_scope(
        split, summary, state, runtime_manifest, evaluator_manifest, path
    )
    return {
        "split": split,
        "summary_path": path,
        "summary": summary,
        "state_path": state_path,
        "state": state,
        "runtime_manifest": runtime_manifest,
        "evaluator_manifest": evaluator_manifest,
        "case_metrics": case_metrics,
        "recording_metrics": recording_metrics,
        "candidate_report": candidate_report,
        "candidate_source": candidate_source,
        "report": report,
        "formal_scope": formal_scope,
    }


def candidate_runtime_bindings(bundle: dict[str, Any]) -> dict[str, tuple[Path, str]]:
    expected = {
        str(value)
        for value in bundle["state"].get("run_identity", {}).get(
            "selected_case_ids", []
        )
    }
    bindings: dict[str, tuple[Path, str]] = {}
    rows = bundle["report"].get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"candidate report has no rows in {bundle['split']}")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"candidate report has a non-object row in {bundle['split']}")
        case_id = str(row.get("case_id", ""))
        diagnostics = row.get("diagnostics") or {}
        provenance = diagnostics.get("provenance") or {}
        path_raw = provenance.get("runtime_report")
        digest = str(provenance.get("runtime_report_sha256", ""))
        if (
            case_id not in expected
            or path_raw in (None, "")
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(
                f"candidate row lacks a valid runtime binding in {bundle['split']}: "
                f"{case_id!r}"
            )
        binding = (Path(str(path_raw)).resolve(), digest)
        previous = bindings.get(case_id)
        if previous is not None and previous != binding:
            raise ValueError(
                f"candidate report has inconsistent runtime bindings for {case_id}"
            )
        bindings[case_id] = binding
    if set(bindings) != expected:
        raise ValueError(
            f"candidate/runtime provenance coverage mismatch in {bundle['split']}: "
            f"bound={len(bindings)}, expected={len(expected)}"
        )
    return bindings


def runtime_reports(bundle: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    state = bundle["state"]
    output_root = Path(
        str(state.get("run_identity", {}).get("output_root", bundle["state_path"].parent))
    ).resolve()
    prediction_root = output_root / "predictions"
    paths = sorted(prediction_root.glob("*.runtime.json"))
    expected = {
        str(value)
        for value in state.get("run_identity", {}).get("selected_case_ids", [])
    }
    bindings = candidate_runtime_bindings(bundle)
    output: list[tuple[Path, dict[str, Any]]] = []
    observed: set[str] = set()
    for path in paths:
        payload = load(path)
        if payload.get("schema_version") != RUNTIME_SCHEMA:
            raise ValueError(f"unexpected runtime schema: {path}")
        case_id = str(payload.get("record", {}).get("case_id", ""))
        if not case_id or case_id in observed:
            raise ValueError(f"missing/duplicate runtime case: {case_id!r}")
        declared_path, declared_sha = bindings.get(case_id, (Path(), ""))
        if path.resolve() != declared_path:
            raise ValueError(
                f"runtime path differs from candidate report for {case_id}: "
                f"{path.resolve()} vs {declared_path}"
            )
        current_sha = sha256(path.resolve())
        if current_sha != declared_sha:
            raise ValueError(
                f"runtime SHA differs from candidate report for {case_id}: "
                f"{current_sha} vs {declared_sha}"
            )
        inference = state.get("inference", {}).get(case_id, {})
        if (
            not isinstance(inference, dict)
            or inference.get("status") != "complete"
            or inference.get("runtime_report_sha256") != declared_sha
        ):
            raise ValueError(
                f"runtime SHA/status differs from completed inference state for "
                f"{case_id}"
            )
        observed.add(case_id)
        output.append((path, payload))
    if observed != expected:
        raise ValueError(
            f"runtime coverage mismatch in {bundle['split']}: "
            f"observed={len(observed)}, expected={len(expected)}"
        )
    return output


def metric_mean(summary: dict[str, Any], method: str, metric: str) -> float | None:
    return finite(
        summary.get("methods", {})
        .get(method, {})
        .get("metrics", {})
        .get(metric, {})
        .get("mean")
    )


def display(method: str, final_method: str) -> str:
    if method == final_method:
        return "Bridge3R (causal, frozen)"
    return DISPLAY.get(method, method)


def boundary_mode(method: str, final_method: str) -> str:
    if method == final_method or method == PARENT:
        return "causal_detector"
    if method == "m0_strict_human3r":
        return "none"
    if method in {
        "m0r_original_clean_reset",
        "m1_current_clean_reset",
        "m3_b0_only",
        "m15_v17_gated_parent",
    }:
        return "oracle_cut"
    raise ValueError(f"unregistered boundary operating point: {method}")


def chosen_methods(summary: dict[str, Any], final_method: str) -> list[str]:
    if final_method != FROZEN_FINAL_METHOD:
        raise ValueError(f"unexpected frozen Test method: {final_method}")
    methods = summary.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("Test aggregate methods are malformed")
    available = set(methods)
    expected = set(EXPECTED_METHODS)
    if available != expected:
        raise ValueError(
            "Test aggregate method inventory mismatch: "
            f"missing={sorted(expected - available)}, extra={sorted(available - expected)}"
        )
    return list(EXPECTED_METHODS)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def fmt(value: Any, metric: str) -> str:
    number = finite(value)
    if number is None:
        return "--"
    if metric.endswith("_m") or metric in {
        "IDF1",
        "Coverage",
        "Detection_precision",
    }:
        return f"{number:.3f}"
    if metric == "IDs":
        return f"{number:.2f}"
    return f"{number:.1f}"


TEXT_COLUMNS = {
    "angle_stratum",
    "case_id",
    "display_name",
    "gpu",
    "gate_state",
    "fallback_array_exactness_observed",
    "fallback_metric_exactness_observed",
    "method",
    "metric",
    "metric_display",
    "definition",
    "precision",
    "split",
    "status",
}


def latex_value(value: Any, key: str) -> str:
    if value in (None, ""):
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key in METRIC_LABELS:
        return fmt(value, key)
    if isinstance(value, str):
        return latex_escape(value)
    number = finite(value)
    if number is None:
        return "--"
    if key.endswith("_count") or key in {
        "cases",
        "evaluation_boundary",
        "proposal_boundary",
        "accepted_W_harm_over_5pct",
        "accepted_W_harm_over_10pct",
        "accepted_W_harm_over_20pct",
    }:
        return f"{number:.0f}"
    if key.endswith("_rate") or key.endswith("_ratio") or key in {
        "boundary_recall",
        "directly_comparable",
        "detector_precision",
        "detector_recall",
        "detector_f1",
        "brier",
    }:
        return f"{number:.3f}"
    if "harm_percent" in key:
        return f"{number:.1f}"
    if "fps" in key:
        return f"{number:.1f}"
    if "seconds" in key or "rss_gib" in key:
        return f"{number:.3f}"
    return f"{number:.2f}"


def latex_table(
    rows: list[dict[str, Any]], leading: list[tuple[str, str]], metrics: tuple[str, ...]
) -> str:
    columns = "".join(
        "l" if key in TEXT_COLUMNS else "r" for key, _ in leading
    ) + "r" * len(metrics)
    header = [label for _, label in leading] + [METRIC_LABELS[value] for value in metrics]
    lines = [
        "% Auto-generated by versions/v20/egobody/build_paper_artifacts.py.",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        values = [latex_value(row.get(key), key) for key, _ in leading]
        values.extend(fmt(row.get(metric), metric) for metric in metrics)
        lines.append(" & ".join(values) + r" \\")
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def latex_panels(*tables: str) -> str:
    return "\n\\par\\smallskip\n".join(table.rstrip() for table in tables) + "\n"


def main_table(
    test: dict[str, Any], final_method: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = chosen_methods(test, final_method)
    rows = []
    full = {}
    for method in methods:
        value = test["methods"][method]
        required = {
            metric: metric_mean(test, method, metric)
            for metric in REQUIRED_PAPER_METRICS
        }
        missing = [metric for metric, result in required.items() if result is None]
        if missing:
            raise ValueError(
                f"missing required paper metric for {method}: {sorted(missing)}"
            )
        row = {
            "method": method,
            "display_name": display(method, final_method),
            "boundary_mode": boundary_mode(method, final_method),
            "recording_count": int(value.get("recording_count", 0)),
            "case_count": int(value.get("case_count", 0)),
        }
        row.update({metric: metric_mean(test, method, metric) for metric in TABLE_METRICS})
        rows.append(row)
        full[method] = {
            "display_name": row["display_name"],
            "boundary_mode": row["boundary_mode"],
            "recording_count": row["recording_count"],
            "case_count": row["case_count"],
            "metrics": value.get("metrics", {}),
        }
    return rows, full


def angle_rows(
    summary: dict[str, Any], methods: list[str], final_method: str
) -> list[dict[str, Any]]:
    if tuple(methods) != EXPECTED_METHODS:
        raise ValueError("angle table method inventory differs from frozen Test inventory")
    rows = []
    strata = summary.get("angle_strata", {})
    if not isinstance(strata, dict) or set(strata) != set(ANGLE_STRATA):
        raise ValueError(
            f"angle strata mismatch: observed={sorted(strata) if isinstance(strata, dict) else type(strata).__name__}"
        )
    counts_by_method = {method: 0 for method in methods}
    for angle in ANGLE_STRATA:
        angle_payload = strata.get(angle)
        if not isinstance(angle_payload, dict) or set(angle_payload) != set(methods):
            raise ValueError(f"angle method coverage mismatch for {angle}")
        angle_counts: set[int] = set()
        for method in methods:
            value = angle_payload.get(method)
            if not isinstance(value, dict) or int(value.get("case_count", 0)) <= 0:
                raise ValueError(f"empty/malformed angle row for {method}/{angle}")
            case_count = int(value["case_count"])
            counts_by_method[method] += case_count
            angle_counts.add(case_count)
            missing = [
                metric
                for metric in PRIMARY_TABLE_METRICS
                if finite(value.get("metrics", {}).get(metric)) is None
            ]
            if missing:
                raise ValueError(
                    f"missing required paper metric for {method}/{angle}: "
                    f"{sorted(missing)}"
                )
            row = {
                "angle_stratum": angle,
                "method": method,
                "display_name": display(method, final_method),
                "boundary_mode": boundary_mode(method, final_method),
                "case_count": case_count,
            }
            row.update(
                {metric: finite(value.get("metrics", {}).get(metric)) for metric in TABLE_METRICS}
            )
            rows.append(row)
        if len(angle_counts) != 1:
            raise ValueError(f"method case counts differ within angle stratum {angle}")
    for method in methods:
        expected_count = int(summary["methods"][method].get("case_count", -1))
        if counts_by_method[method] != expected_count:
            raise ValueError(
                f"angle case count does not reconcile for {method}: "
                f"{counts_by_method[method]} vs {expected_count}"
            )
    return rows


def detector_artifacts(
    bundles: list[dict[str, Any]], runtimes: dict[str, list[tuple[Path, dict[str, Any]]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        split = bundle["split"]
        for path, payload in runtimes[split]:
            record = payload["record"]
            detector = payload.get("runtime", {}).get("causal_gru_detector", {})
            boundary = int(record["boundary_index"])
            clip_length = int(record.get("clip_length", 0))
            if boundary <= 0 or boundary >= clip_length:
                raise ValueError(f"invalid detector boundary: {path}")
            labels_raw = detector.get("labels")
            if not isinstance(labels_raw, list) or len(labels_raw) != clip_length:
                raise ValueError(f"detector label count mismatch: {path}")
            if any(value not in (0, 1, False, True) for value in labels_raw):
                raise ValueError(f"detector labels are not binary: {path}")
            labels = [bool(value) for value in labels_raw]
            if labels[0]:
                raise ValueError(f"detector cannot emit a label for frame zero: {path}")

            probability_rows = detector.get("rows")
            if not isinstance(probability_rows, list):
                raise ValueError(f"detector probability rows are missing: {path}")
            by_index: dict[int, dict[str, Any]] = {}
            thresholds: set[float] = set()
            brier_terms: list[float] = []
            for row in probability_rows:
                if not isinstance(row, dict):
                    raise ValueError(f"malformed detector probability row: {path}")
                index = int(row.get("pair_idx", -1))
                if index in by_index:
                    raise ValueError(f"duplicate detector pair_idx {index}: {path}")
                probability = finite(row.get("prob"))
                threshold = finite(row.get("threshold"))
                if (
                    index < 1
                    or index >= clip_length
                    or probability is None
                    or not 0.0 <= probability <= 1.0
                    or threshold is None
                    or not 0.0 <= threshold <= 1.0
                ):
                    raise ValueError(f"invalid detector probability row: {path}")
                prediction = int(row.get("pred", -1))
                if prediction not in (0, 1):
                    raise ValueError(f"invalid detector prediction: {path}")
                if prediction != int(labels[index]) or prediction != int(
                    probability >= threshold
                ):
                    raise ValueError(f"detector probability/label mismatch: {path}")
                by_index[index] = row
                thresholds.add(threshold)
                target = 1.0 if index == boundary else 0.0
                brier_terms.append((probability - target) ** 2)
            if set(by_index) != set(range(1, clip_length)) or len(thresholds) != 1:
                raise ValueError(f"detector probability coverage mismatch: {path}")

            positive_indices = [index for index, value in enumerate(labels) if value]
            derived_proposal = positive_indices[0] if positive_indices else None
            proposal_raw = detector.get("proposal_boundary")
            proposal = None if proposal_raw is None else int(proposal_raw)
            declared_first_raw = detector.get("first_positive_index", proposal)
            declared_first = (
                None if declared_first_raw is None else int(declared_first_raw)
            )
            if proposal != derived_proposal or declared_first != derived_proposal:
                raise ValueError(f"detector proposal is not the first positive: {path}")
            error = None if proposal is None else proposal - boundary
            exact = proposal == boundary
            status = "missed" if proposal is None else ("exact" if exact else ("early" if error < 0 else "late"))
            tp = int(labels[boundary])
            fp = sum(value for index, value in enumerate(labels) if index != boundary)
            fn = 1 - tp
            cases.append(
                {
                    "split": split,
                    "case_id": record["case_id"],
                    "evaluation_boundary": boundary,
                    "proposal_boundary": proposal,
                    "signed_error_frames": error,
                    "absolute_error_frames": None if error is None else abs(error),
                    "status": status,
                    "exact": exact,
                    "boundary_label_positive": labels[boundary],
                    "positive_frame_count": sum(labels),
                    "off_boundary_positive_count": fp,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "frame_count": clip_length,
                    "evaluated_pair_count": clip_length - 1,
                    "brier": average(brier_terms),
                    "detector_seconds": finite(detector.get("seconds")),
                }
            )
        values = [row for row in cases if row["split"] == split]
        if not values:
            raise ValueError(f"no detector cases for {split}")
        proposed = [row for row in values if row["signed_error_frames"] is not None]
        tp = sum(int(row["tp"]) for row in values)
        fp = sum(int(row["fp"]) for row in values)
        fn = sum(int(row["fn"]) for row in values)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        total_frames = sum(int(row["frame_count"]) for row in values)
        summaries.append(
            {
                "split": split,
                "case_count": len(values),
                "exact_count": sum(bool(row["exact"]) for row in values),
                "error_count": sum(not bool(row["exact"]) for row in values),
                "exact_rate": average(bool(row["exact"]) for row in values),
                "missed_count": sum(row["status"] == "missed" for row in values),
                "early_count": sum(row["status"] == "early" for row in values),
                "late_count": sum(row["status"] == "late" for row in values),
                "boundary_recall": average(row["boundary_label_positive"] for row in values),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detector_precision": precision,
                "detector_recall": recall,
                "detector_f1": f1,
                "off_boundary_positive_count": fp,
                "false_positives_per_100_frames": 100.0 * fp / max(total_frames, 1),
                "brier": average(row["brier"] for row in values),
                "mean_signed_first_positive_offset_frames": average(
                    row["signed_error_frames"] for row in proposed
                ),
                "mae_frames_given_proposal": average(
                    row["absolute_error_frames"] for row in proposed
                ),
                "max_absolute_error_frames": max(
                    (int(row["absolute_error_frames"]) for row in proposed), default=None
                ),
            }
        )
    return cases, summaries


def safety_artifacts(
    bundles: list[dict[str, Any]], final_method: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    accepted_cases = []
    worst_rows = []
    for bundle in bundles:
        split = bundle["split"]
        summary = bundle["summary"]
        safety = summary.get("safety", {})
        for method, value in sorted(safety.items()):
            ratio = finite(value.get("worst_accepted_W_ratio"))
            case_count = int(value.get("case_count", 0))
            accepted_count = int(value.get("accepted_count", 0))
            fallback_count = int(value.get("fallback_count", 0))
            gate_enabled = value.get("gate_enabled") is True
            missing_gate_count = len(value.get("missing_gate_cases", []))
            if accepted_count + fallback_count + missing_gate_count != case_count:
                raise ValueError(
                    f"safety case accounting mismatch for {split}/{method}"
                )
            if method == final_method and gate_enabled:
                raise ValueError("frozen ungated final is marked gate-enabled")
            array_exact = value.get("fallback_array_exactness_passed") is True
            metric_exact = value.get("fallback_metric_exactness_passed") is True
            summary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "display_name": display(method, final_method),
                    "is_final_method": method == final_method,
                    "case_count": case_count,
                    "gate_enabled": gate_enabled,
                    "gate_state": "Enabled" if gate_enabled else "Disabled",
                    "accepted_count": accepted_count,
                    "fallback_count": fallback_count,
                    "materialized_count": accepted_count,
                    "detector_miss_parent_reuse_count": fallback_count,
                    "acceptance_rate": finite(value.get("acceptance_rate")),
                    "missing_gate_case_count": missing_gate_count,
                    "fallback_array_exactness_passed": array_exact,
                    "fallback_array_exactness_observed": (
                        "N/A (0 reuse)" if fallback_count == 0
                        else ("Yes" if array_exact else "No")
                    ),
                    "fallback_array_audit_missing_count": len(
                        value.get("fallback_array_audit_missing_cases", [])
                    ),
                    "fallback_array_mismatch_count": len(
                        value.get("fallback_array_mismatches", [])
                    ),
                    "fallback_metric_exactness_passed": metric_exact,
                    "fallback_metric_exactness_observed": (
                        "N/A (0 reuse)" if fallback_count == 0
                        else ("Yes" if metric_exact else "No")
                    ),
                    "fallback_metric_mismatch_count": len(
                        value.get("fallback_metric_mismatches", [])
                    ),
                    "accepted_W_harm_over_5pct": int(
                        value.get("accepted_W_harm_over_5pct", 0)
                    ),
                    "accepted_W_harm_over_10pct": int(
                        value.get("accepted_W_harm_over_10pct", 0)
                    ),
                    "accepted_W_harm_over_20pct": int(
                        value.get("accepted_W_harm_over_20pct", 0)
                    ),
                    "worst_accepted_W_ratio": ratio,
                    "worst_accepted_W_harm_percent": None
                    if ratio is None
                    else 100.0 * (ratio - 1.0),
                    "accepted_W_improvement_rate": finite(
                        value.get("accepted_W_improvement_rate")
                    ),
                }
            )
        report = bundle["report"]
        parent_rows = {
            str(row["case_id"]): row
            for row in report.get("rows", [])
            if row.get("candidate") == PARENT and row.get("status") == "complete"
        }
        for row in report.get("rows", []):
            if row.get("status") != "complete":
                continue
            gate = (row.get("diagnostics") or {}).get("reliability_gate")
            if not isinstance(gate, dict) or gate.get("accepted") is not True:
                continue
            case_id = str(row["case_id"])
            parent = parent_rows.get(case_id)
            candidate_w = finite(row.get("metrics", {}).get("W-MPJPE_mm"))
            parent_w = finite(None if parent is None else parent.get("metrics", {}).get("W-MPJPE_mm"))
            if candidate_w is None or parent_w is None or parent_w <= 0:
                continue
            ratio = candidate_w / parent_w
            accepted_cases.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "angle_stratum": row.get("angle_stratum"),
                    "method": row.get("candidate"),
                    "is_final_method": row.get("candidate") == final_method,
                    "gate_enabled": gate.get("enabled") is True,
                    "candidate_W-MPJPE_mm": candidate_w,
                    "parent_W-MPJPE_mm": parent_w,
                    "candidate_to_parent_W_ratio": ratio,
                    "W_harm_percent": 100.0 * (ratio - 1.0),
                }
            )
    accepted_cases.sort(
        key=lambda row: (
            -float(row["candidate_to_parent_W_ratio"]),
            str(row["split"]),
            str(row["method"]),
            str(row["case_id"]),
        )
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_cases:
        grouped[(str(row["split"]), str(row["method"]))].append(row)
    for (split, method), values in sorted(grouped.items()):
        worst = max(values, key=lambda row: float(row["candidate_to_parent_W_ratio"]))
        worst_rows.append(
            {
                "split": split,
                "method": method,
                "display_name": display(method, final_method),
                "accepted_case_count": len(values),
                "worst_case_id": worst["case_id"],
                "worst_accepted_W_ratio": worst["candidate_to_parent_W_ratio"],
                "worst_accepted_W_harm_percent": worst["W_harm_percent"],
            }
        )
    return summary_rows, accepted_cases, worst_rows


def runtime_artifacts(
    bundles: list[dict[str, Any]],
    runtimes: dict[str, list[tuple[Path, dict[str, Any]]]],
    final_method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = []
    summaries = []
    for bundle in bundles:
        split = bundle["split"]
        postprocess = {
            str(row["case_id"]): finite(
                (row.get("diagnostics") or {}).get("postprocess_seconds")
            )
            for row in bundle["report"].get("rows", [])
            if row.get("candidate") == final_method and row.get("status") == "complete"
        }
        for _, payload in runtimes[split]:
            record = payload["record"]
            runtime = payload.get("runtime", {})
            forward = runtime.get("m0_forward", {})
            detector = runtime.get("causal_gru_detector", {})
            frames = int(forward.get("frames", record.get("clip_length", 0)))
            total = finite(payload.get("total_process_seconds"))
            detector_seconds = finite(detector.get("seconds"))
            post_seconds = postprocess.get(str(record["case_id"]))
            cases.append(
                {
                    "split": split,
                    "case_id": record["case_id"],
                    "frames": frames,
                    "strict_human3r_forward_seconds": finite(forward.get("seconds")),
                    "strict_human3r_forward_fps": finite(forward.get("fps")),
                    "causal_detector_seconds": detector_seconds,
                    "causal_detector_fps": None
                    if detector_seconds in (None, 0)
                    else frames / detector_seconds,
                    "whole_multi_method_protocol_seconds": total,
                    "frozen_candidate_postprocess_seconds": post_seconds,
                    "frozen_candidate_postprocess_fps": None
                    if post_seconds in (None, 0)
                    else frames / post_seconds,
                    "process_peak_host_rss_gib": (
                        finite(payload.get("environment", {}).get("process_peak_rss_bytes"))
                        / (1024**3)
                        if finite(payload.get("environment", {}).get("process_peak_rss_bytes"))
                        is not None
                        else None
                    ),
                    "gpu": payload.get("environment", {}).get("gpu"),
                    "precision": payload.get("environment", {}).get("precision"),
                }
            )
        values = [row for row in cases if row["split"] == split]
        fps = [
            row["strict_human3r_forward_fps"]
            for row in values
            if row["strict_human3r_forward_fps"] is not None
        ]
        post_fps = [
            row["frozen_candidate_postprocess_fps"]
            for row in values
            if row["frozen_candidate_postprocess_fps"] is not None
        ]
        summaries.append(
            {
                "split": split,
                "case_count": len(values),
                "strict_human3r_forward_fps_mean": average(fps),
                "strict_human3r_forward_fps_median": median(fps) if fps else None,
                "causal_detector_seconds_mean": average(
                    row["causal_detector_seconds"] for row in values
                ),
                "causal_detector_fps_mean": average(
                    row["causal_detector_fps"] for row in values
                ),
                "whole_multi_method_protocol_seconds_mean": average(
                    row["whole_multi_method_protocol_seconds"] for row in values
                ),
                "frozen_candidate_postprocess_case_count": len(post_fps),
                "frozen_candidate_postprocess_fps_mean": average(post_fps),
                "frozen_candidate_postprocess_seconds_mean": average(
                    row["frozen_candidate_postprocess_seconds"] for row in values
                ),
                "process_peak_host_rss_gib_max": max(
                    (
                        float(row["process_peak_host_rss_gib"])
                        for row in values
                        if row["process_peak_host_rss_gib"] is not None
                    ),
                    default=None,
                ),
                "deployed_single_method_fps_available": False,
                "gpu_peak_memory_available": False,
                "timing_contract": (
                    "component and whole multi-method protocol wall-clock diagnostics "
                    "on recorded GPUs/shared server; the whole protocol includes model "
                    "loads, multiple baselines, oracle/detector paths and I/O, so it is "
                    "not deployed single-method end-to-end throughput; not "
                    "hardware-normalized; host RSS only; GPU peak memory unmeasured"
                ),
            }
        )
    return cases, summaries


def multi_thumbs_context(pdf: Path) -> dict[str, Any]:
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    return {
        "source": {
            "paper": "Multi-THuMBS: Multi-person Tracking of 3D Human Meshes Beyond Video Shots",
            "path": str(pdf.resolve()),
            "sha256": sha256(pdf.resolve()),
            "table": "main-paper Tables 1 and 2, EgoBody Ours rows",
        },
        "multi_thumbs_egobody_values": MULTI_THUMBS_EGOBODY,
        "bridge3r_protocol": {
            "name": PROTOCOL,
            "clip": "150 frames: 75 pre-cut + 75 post-cut at 30 FPS",
            "camera_pairs": "three fixed small/medium/extreme cross-camera cases per recording",
            "w_wa": "one shared clip-level Sim(3) across both people",
            "aggregation": "three-case recording means, then equal-weight recording macro",
        },
        "multi_thumbs_protocol": {
            "description": (
                "paper-constructed multi-shot EgoBody benchmark with its own edited clips, "
                "camera/cut sampling, optimization and evaluator"
            ),
            "reported_runtime": (
                "approximately 10 minutes for a 150-frame 1920x1080 video on one RTX 3090"
            ),
            "ate_definition": (
                "source-defined ATE; the available main-paper PDF does not establish "
                "its alignment convention or unit"
            ),
        },
        "comparability": {
            "direct_leaderboard_comparison": False,
            "reason": (
                "same-named metrics are literature-scale context only: exact cases, cuts, "
                "camera pairs, alignment scope and evaluator are not the Bridge3R-CS150 protocol"
            ),
        },
    }


def source_entry(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def verify_artifact_manifest(
    manifest_path: Path,
    sidecar_path: Path | None = None,
    *,
    reject_extra_files: bool = True,
) -> dict[str, Any]:
    """Independently verify every declared input/output and release marker."""

    manifest_path = manifest_path.resolve()
    sidecar_path = (
        sidecar_path.resolve()
        if sidecar_path is not None
        else manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    )
    manifest = load(manifest_path)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("protocol") != PROTOCOL
    ):
        raise ValueError("unexpected paper-artifact manifest schema/protocol")
    expected_sidecar = f"{sha256(manifest_path)}  {manifest_path.name}\n"
    if not sidecar_path.is_file() or sidecar_path.read_text(
        encoding="utf-8"
    ) != expected_sidecar:
        raise ValueError("paper-artifact manifest SHA sidecar mismatch")
    for section in ("sources", "outputs"):
        rows = manifest.get(section)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"paper-artifact manifest has no {section}")
        observed_paths: set[Path] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"malformed paper-artifact {section} entry")
            path = Path(str(row.get("path", ""))).resolve()
            if path in observed_paths:
                raise ValueError(f"duplicate paper-artifact {section} path: {path}")
            observed_paths.add(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if row.get("sha256") != sha256(path) or int(row.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"paper-artifact {section} SHA/size mismatch: {path}")
    output_names = {
        Path(str(row["path"])).name for row in manifest.get("outputs", [])
    }
    expected_names = set(manifest.get("expected_output_names", []))
    if output_names != expected_names:
        raise ValueError("paper-artifact expected output set mismatch")
    if any(
        Path(str(row["path"])).resolve().parent != manifest_path.parent
        for row in manifest["outputs"]
    ):
        raise ValueError("paper-artifact output escaped the release directory")
    if reject_extra_files:
        actual_names = {path.name for path in manifest_path.parent.iterdir() if path.is_file()}
        allowed = expected_names | {manifest_path.name, sidecar_path.name}
        if actual_names != allowed:
            raise ValueError(
                "paper-artifact release directory contains missing/stale files: "
                f"missing={sorted(allowed - actual_names)}, extra={sorted(actual_names - allowed)}"
            )
    return manifest


def main() -> None:
    args = parse_args()
    bundles = [
        validate_bundle(args.development, "development"),
        validate_bundle(args.holdout, "holdout"),
        validate_bundle(args.test, "test"),
    ]
    final_path = args.final_candidate.resolve()
    final = load(final_path)
    final_method = validate_final_candidate(final, final_path)
    test_bundle = bundles[-1]
    if (
        test_bundle["candidate_source"] != final_path
        or test_bundle["summary"].get("candidate_source_sha256") != sha256(final_path)
    ):
        raise ValueError("Test aggregate is not bound to the frozen final candidate")
    ledger = checked_path(
        test_bundle["state"],
        "test_consumption_ledger",
        "test_consumption_ledger_sha256",
        test_bundle["state_path"],
    )
    validate_test_ledger(ledger, final_path, test_bundle["state"])
    runtimes = {bundle["split"]: runtime_reports(bundle) for bundle in bundles}

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A failed regeneration must never leave an older valid release marker over
    # partially replaced files.  Individual artifact writes remain atomic.
    (output / "artifact_manifest.json").unlink(missing_ok=True)
    (output / "artifact_manifest.json.sha256").unlink(missing_ok=True)
    test_summary = test_bundle["summary"]
    main_rows, main_json = main_table(test_summary, final_method)
    main_fields = [
        "method", "display_name", "boundary_mode", "recording_count",
        "case_count", *TABLE_METRICS
    ]
    write_csv(output / "recording_macro_main.csv", main_rows, main_fields)
    write_json(
        output / "recording_macro_main.json",
        {
            "aggregation": test_summary.get("aggregation"),
            "final_method": final_method,
            "methods": main_json,
        },
    )
    primary_table = latex_table(
        main_rows, [("display_name", "Method")], PRIMARY_TABLE_METRICS
    )
    write_text(output / "recording_macro_main.tex", primary_table)
    write_text(output / "recording_macro_primary.tex", primary_table)
    write_text(
        output / "recording_macro_local.tex",
        latex_panels(
            latex_table(
                main_rows, [("display_name", "Method")], LOCAL_TABLE_METRICS
            ),
            latex_table(
                main_rows, [("display_name", "Method")], TEMPORAL_TABLE_METRICS
            ),
        ),
    )
    write_text(
        output / "recording_macro_boundary.tex",
        latex_panels(
            latex_table(
                main_rows,
                [("display_name", "Method")],
                BOUNDARY_CAMERA_TABLE_METRICS,
            ),
            latex_table(
                main_rows,
                [("display_name", "Method")],
                BOUNDARY_HUMAN_TABLE_METRICS,
            ),
        ),
    )

    methods = [row["method"] for row in main_rows]
    angles = angle_rows(test_summary, methods, final_method)
    angle_fields = [
        "angle_stratum", "method", "display_name", "boundary_mode", "case_count",
        *TABLE_METRICS
    ]
    write_csv(output / "angle_strata.csv", angles, angle_fields)
    write_text(
        output / "angle_strata.tex",
        latex_table(
            angles,
            [("angle_stratum", "Angle"), ("display_name", "Method")],
            PRIMARY_TABLE_METRICS,
        ),
    )

    detector_cases, detector_summary = detector_artifacts(bundles, runtimes)
    detector_case_fields = [
        "split", "case_id", "evaluation_boundary", "proposal_boundary",
        "signed_error_frames", "absolute_error_frames", "status", "exact",
        "boundary_label_positive", "positive_frame_count",
        "off_boundary_positive_count", "tp", "fp", "fn", "frame_count",
        "evaluated_pair_count", "brier", "detector_seconds",
    ]
    detector_summary_fields = [
        "split", "case_count", "exact_count", "error_count", "exact_rate",
        "missed_count", "early_count", "late_count", "boundary_recall",
        "tp", "fp", "fn", "detector_precision", "detector_recall",
        "detector_f1", "off_boundary_positive_count",
        "false_positives_per_100_frames", "brier",
        "mean_signed_first_positive_offset_frames",
        "mae_frames_given_proposal", "max_absolute_error_frames",
    ]
    write_csv(output / "detector_cases.csv", detector_cases, detector_case_fields)
    write_csv(output / "detector_summary.csv", detector_summary, detector_summary_fields)
    write_json(output / "detector_summary.json", detector_summary)
    test_detector_table = [row for row in detector_summary if row["split"] == "test"]
    if len(test_detector_table) != 1:
        raise ValueError("detector paper table must contain exactly one Test row")
    write_text(
        output / "detector_table.tex",
        latex_panels(
            latex_table(
                test_detector_table,
                [
                    ("case_count", "Cases"),
                    ("exact_count", "Exact"),
                    ("early_count", "Early"),
                    ("late_count", "Late"),
                    ("missed_count", "Missed"),
                ],
                (),
            ),
            latex_table(
                test_detector_table,
                [
                    ("detector_precision", "Precision $\\uparrow$"),
                    ("detector_recall", "Recall $\\uparrow$"),
                    ("detector_f1", "F1 $\\uparrow$"),
                    (
                        "false_positives_per_100_frames",
                        "FP/100 frames $\\downarrow$",
                    ),
                    ("brier", "Brier $\\downarrow$"),
                    (
                        "mean_signed_first_positive_offset_frames",
                        "First-positive offset (frames) $\\rightarrow 0$",
                    ),
                ],
                (),
            ),
        ),
    )

    safety, accepted, worst = safety_artifacts(bundles, final_method)
    safety_fields = [
        "split", "method", "display_name", "is_final_method", "case_count",
        "gate_enabled", "gate_state", "accepted_count", "fallback_count",
        "materialized_count", "detector_miss_parent_reuse_count",
        "acceptance_rate", "missing_gate_case_count",
        "fallback_array_exactness_passed",
        "fallback_array_exactness_observed",
        "fallback_array_audit_missing_count", "fallback_array_mismatch_count",
        "fallback_metric_exactness_passed", "fallback_metric_exactness_observed",
        "fallback_metric_mismatch_count", "accepted_W_harm_over_5pct",
        "accepted_W_harm_over_10pct", "accepted_W_harm_over_20pct",
        "worst_accepted_W_ratio", "worst_accepted_W_harm_percent",
        "accepted_W_improvement_rate",
    ]
    accepted_fields = [
        "split", "case_id", "angle_stratum", "method", "is_final_method",
        "gate_enabled", "candidate_W-MPJPE_mm", "parent_W-MPJPE_mm",
        "candidate_to_parent_W_ratio", "W_harm_percent",
    ]
    worst_fields = [
        "split", "method", "display_name", "accepted_case_count", "worst_case_id",
        "worst_accepted_W_ratio", "worst_accepted_W_harm_percent",
    ]
    write_csv(output / "safety_summary.csv", safety, safety_fields)
    write_json(output / "safety_summary.json", safety)
    final_safety_table = [
        row
        for row in safety
        if row["split"] == "test" and row["method"] == final_method
    ]
    if len(final_safety_table) != 1:
        raise ValueError("safety paper table must contain exactly one frozen Test row")
    write_text(
        output / "safety_table.tex",
        latex_panels(
            latex_table(
                final_safety_table,
                [
                    ("gate_state", "Reliability gate"),
                    ("materialized_count", "Materialized"),
                    (
                        "detector_miss_parent_reuse_count",
                        "Detector-miss parent reuse",
                    ),
                    (
                        "fallback_array_exactness_observed",
                        "Reuse array exact",
                    ),
                    (
                        "fallback_metric_exactness_observed",
                        "Reuse metric exact",
                    ),
                ],
                (),
            ),
            latex_table(
                final_safety_table,
                [
                    ("accepted_W_improvement_rate", "Materialized W improve rate"),
                    ("accepted_W_harm_over_5pct", "$>5\\%$ W harm"),
                    ("accepted_W_harm_over_10pct", "$>10\\%$ W harm"),
                    ("accepted_W_harm_over_20pct", "$>20\\%$ W harm"),
                    ("worst_accepted_W_harm_percent", "Worst W harm (\\%)"),
                ],
                (),
            ),
        ),
    )
    write_csv(output / "accepted_harm_cases.csv", accepted, accepted_fields)
    write_csv(output / "worst_accepted_harm.csv", worst, worst_fields)
    write_json(output / "worst_accepted_harm.json", worst)

    runtime_cases, runtime_summary = runtime_artifacts(
        bundles, runtimes, final_method
    )
    runtime_case_fields = [
        "split", "case_id", "frames", "strict_human3r_forward_seconds",
        "strict_human3r_forward_fps", "causal_detector_seconds",
        "causal_detector_fps", "whole_multi_method_protocol_seconds",
        "frozen_candidate_postprocess_seconds",
        "frozen_candidate_postprocess_fps", "process_peak_host_rss_gib",
        "gpu", "precision",
    ]
    runtime_summary_fields = [
        "split", "case_count", "strict_human3r_forward_fps_mean",
        "strict_human3r_forward_fps_median", "causal_detector_seconds_mean",
        "causal_detector_fps_mean", "whole_multi_method_protocol_seconds_mean",
        "frozen_candidate_postprocess_case_count",
        "frozen_candidate_postprocess_fps_mean",
        "frozen_candidate_postprocess_seconds_mean",
        "process_peak_host_rss_gib_max", "deployed_single_method_fps_available",
        "gpu_peak_memory_available", "timing_contract",
    ]
    write_csv(output / "runtime_cases.csv", runtime_cases, runtime_case_fields)
    write_csv(output / "runtime_summary.csv", runtime_summary, runtime_summary_fields)
    write_json(output / "runtime_summary.json", runtime_summary)
    write_text(
        output / "runtime_components.tex",
        latex_table(
            runtime_summary,
            [
                ("split", "Split"),
                ("case_count", "Cases"),
                ("strict_human3r_forward_fps_mean", "Strict FPS"),
                ("causal_detector_seconds_mean", "Detector (s)"),
                (
                    "frozen_candidate_postprocess_seconds_mean",
                    "Frozen post. (s)",
                ),
                (
                    "whole_multi_method_protocol_seconds_mean",
                    "Whole protocol (s)",
                ),
                ("process_peak_host_rss_gib_max", "Host RSS (GiB)"),
            ],
            (),
        ),
    )

    context = multi_thumbs_context(args.multi_thumbs_pdf.resolve())
    bridge_context = {
        metric: metric_mean(test_summary, final_method, metric)
        for metric in (
            "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm",
            "Accel_mm_frame2", "IDs", "ATE_Sim3_m",
        )
    }
    context["bridge3r_cs150_values"] = bridge_context
    context_rows = [
        {
            "metric": metric,
            "metric_display": metric,
            "definition": "same-named literature-scale context; protocols differ",
            "Bridge3R_CS150": bridge_context.get(metric),
            "Multi-THuMBS_EgoBody_Ours": value,
            "directly_comparable": False,
        }
        for metric, value in MULTI_THUMBS_EGOBODY.items()
        if metric != "ATE_source_defined"
    ]
    context_rows.extend(
        (
            {
                "metric": "ATE_Sim3_m",
                "metric_display": "Bridge3R ATE-Sim3 (m)",
                "definition": "Bridge3R-CS150 Sim(3)-aligned camera trajectory error",
                "Bridge3R_CS150": bridge_context["ATE_Sim3_m"],
                "Multi-THuMBS_EgoBody_Ours": None,
                "directly_comparable": False,
            },
            {
                "metric": "ATE_source_defined",
                "metric_display": "Multi-THuMBS ATE (source-defined)",
                "definition": (
                    "source reports 0.1; alignment convention/unit are not "
                    "established by the available main-paper PDF"
                ),
                "Bridge3R_CS150": None,
                "Multi-THuMBS_EgoBody_Ours": MULTI_THUMBS_EGOBODY[
                    "ATE_source_defined"
                ],
                "directly_comparable": False,
            },
        )
    )
    write_csv(
        output / "multithumbs_context.csv",
        context_rows,
        [
            "metric", "metric_display", "definition", "Bridge3R_CS150",
            "Multi-THuMBS_EgoBody_Ours", "directly_comparable",
        ],
    )
    write_json(output / "multithumbs_context.json", context)
    context_tex_rows = []
    for row in context_rows:
        metric = str(row["metric"])
        display_metric = "ATE_Sim3_m" if metric == "ATE_Sim3_m" else metric
        context_tex_rows.append(
            {
                **row,
                "Bridge3R_CS150_display": fmt(
                    row["Bridge3R_CS150"], display_metric
                ),
                "Multi-THuMBS_EgoBody_Ours_display": fmt(
                    row["Multi-THuMBS_EgoBody_Ours"], display_metric
                ),
            }
        )
    write_text(
        output / "multithumbs_context.tex",
        latex_table(
            context_tex_rows,
            [
                ("metric_display", "Metric / definition"),
                ("Bridge3R_CS150_display", "Bridge3R-CS150"),
                (
                    "Multi-THuMBS_EgoBody_Ours_display",
                    "Multi-THuMBS",
                ),
                ("directly_comparable", "Directly comparable"),
            ],
            (),
        ),
    )

    final_safety = [
        row for row in safety if row["split"] == "test" and row["method"] == final_method
    ]
    test_detector = next(row for row in detector_summary if row["split"] == "test")
    test_runtime = next(row for row in runtime_summary if row["split"] == "test")
    report_lines = [
        "# EgoBody v20 paper-artifact summary",
        "",
        f"- Protocol: `{PROTOCOL}`.",
        f"- Frozen Test candidate: `{final_method}`.",
        f"- Test recording/case count: {test_summary['recording_count']} / {test_summary['case_count']} evaluable; {test_summary['evaluator_unavailable_case_count']} evaluator-unavailable.",
        f"- Detector exact/error: {test_detector['exact_count']} / {test_detector['error_count']} over {test_detector['case_count']} Test cases.",
        f"- Strict Human3R forward FPS: {latex_value(test_runtime['strict_human3r_forward_fps_mean'], 'strict_human3r_forward_fps_mean')} mean.",
        "- Runtime values are component/whole-protocol wall-clock diagnostics on a shared server, not deployed single-method or hardware-normalized throughput.",
        "- The recorded memory value is host RSS; deployed GPU peak memory is unmeasured.",
        "- Multi-THuMBS numbers are different-protocol literature context, not a direct leaderboard comparison.",
    ]
    if final_safety:
        row = final_safety[0]
        report_lines.extend(
            (
                f"- Reliability gate: {row['gate_state']}.",
                f"- Materialized detector events / detector-miss parent reuse: {row['materialized_count']} / {row['detector_miss_parent_reuse_count']}.",
                f"- Reuse array/metric exactness: {row['fallback_array_exactness_observed']} / {row['fallback_metric_exactness_observed']}.",
                f"- Materialized W harm over 5/10/20%: {row['accepted_W_harm_over_5pct']} / {row['accepted_W_harm_over_10pct']} / {row['accepted_W_harm_over_20pct']}.",
                f"- Worst materialized W harm: {fmt(row['worst_accepted_W_harm_percent'], 'MPJPE_mm')}%.",
            )
        )
    write_text(output / "ARTIFACT_SUMMARY.md", "\n".join(report_lines) + "\n")

    source_roles: dict[Path, set[str]] = defaultdict(set)
    source_roles[Path(__file__).resolve()].add("artifact_builder")
    source_roles[final_path].add("frozen_final_candidate")
    source_roles[ledger].add("test_consumption_ledger")
    source_roles[args.multi_thumbs_pdf.resolve()].add("literature_context")
    for bundle in bundles:
        for key in (
            "summary_path", "state_path", "runtime_manifest",
            "evaluator_manifest", "case_metrics", "recording_metrics",
            "candidate_report", "candidate_source",
        ):
            source_roles[bundle[key]].add(f"{bundle['split']}:{key}")
        for path, _ in runtimes[bundle["split"]]:
            source_roles[path].add(f"{bundle['split']}:runtime_report")
    generated_names = (
        "recording_macro_main.csv", "recording_macro_main.json",
        "recording_macro_main.tex", "recording_macro_primary.tex",
        "recording_macro_local.tex", "recording_macro_boundary.tex",
        "angle_strata.csv", "angle_strata.tex",
        "detector_cases.csv", "detector_summary.csv", "detector_summary.json",
        "detector_table.tex", "safety_summary.csv", "safety_summary.json",
        "safety_table.tex", "accepted_harm_cases.csv", "worst_accepted_harm.csv",
        "worst_accepted_harm.json", "runtime_cases.csv", "runtime_summary.csv",
        "runtime_summary.json", "runtime_components.tex", "multithumbs_context.csv",
        "multithumbs_context.json", "multithumbs_context.tex",
        "ARTIFACT_SUMMARY.md",
    )
    output_files = [output / name for name in generated_names]
    manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "protocol": PROTOCOL,
        "final_method": final_method,
        "source_candidate_name": final_method,
        "test_selected_case_count": int(test_summary["selected_case_count"]),
        "test_evaluable_case_count": int(test_summary["case_count"]),
        "test_evaluator_unavailable_case_count": int(
            test_summary["evaluator_unavailable_case_count"]
        ),
        "test_structural_recording_count": int(
            test_bundle["formal_scope"]["structural_recording_count"]
        ),
        "test_evaluable_recording_count": int(test_summary["recording_count"]),
        "expected_output_names": list(generated_names),
        "sources": [
            source_entry(path, ",".join(sorted(roles)))
            for path, roles in sorted(source_roles.items(), key=lambda value: str(value[0]))
        ],
        "outputs": [source_entry(path, "generated_artifact") for path in output_files],
        "contracts": {
            "main_table_aggregation": test_summary.get("aggregation"),
            "test_candidate_frozen_before_metrics": True,
            "frozen_final_geometry": {
                "camera_alpha": 1.0,
                "boundary_kind": "translation",
                "boundary_blend": 0.5,
                "identity": None,
                "reliability_gate_enabled": False,
                "fallback_to_parent": False,
                "parent_reuse_trigger": "detector_miss_only",
            },
            "test_consumption_ledger_sha256": sha256(ledger),
            "multi_thumbs_directly_comparable": False,
            "timing_hardware_normalized": False,
            "deployed_single_method_fps_available": False,
            "gpu_peak_memory_available": False,
            "recorded_memory_scope": "process peak host RSS only",
        },
    }
    artifact_manifest = output / "artifact_manifest.json"
    write_json(artifact_manifest, manifest)
    artifact_manifest_sha = sha256(artifact_manifest)
    write_text(
        output / "artifact_manifest.json.sha256",
        f"{artifact_manifest_sha}  {artifact_manifest.name}\n",
    )
    try:
        verify_artifact_manifest(artifact_manifest)
    except Exception:
        artifact_manifest.unlink(missing_ok=True)
        (output / "artifact_manifest.json.sha256").unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "final_method": final_method,
                "test_selected_cases": int(test_summary["selected_case_count"]),
                "test_evaluable_cases": int(test_summary["case_count"]),
                "artifacts": len(output_files) + 2,
                "manifest": str(artifact_manifest.resolve()),
                "manifest_sha256": artifact_manifest_sha,
                "manifest_sha256_sidecar": str(
                    (output / "artifact_manifest.json.sha256").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
