from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from versions.v20.egobody import build_paper_artifacts as artifacts
from versions.v20.egobody.build_paper_artifacts import (
    ANGLE_STRATA,
    BOUNDARY_CAMERA_TABLE_METRICS,
    BOUNDARY_HUMAN_TABLE_METRICS,
    BOUNDARY_TABLE_METRICS,
    AGGREGATE_SCHEMA,
    ARTIFACT_SCHEMA,
    EXPECTED_METHODS,
    FINAL_SCHEMA,
    FORMAL_SCOPE,
    FROZEN_FINAL_METHOD,
    LOCAL_TABLE_METRICS,
    MULTI_THUMBS_EGOBODY,
    PARENT,
    PRIMARY_TABLE_METRICS,
    PROBE_SCHEMA,
    RUNTIME_SCHEMA,
    STATE_SCHEMA,
    TABLE_METRICS,
    TEMPORAL_TABLE_METRICS,
    TEST_LEDGER_SCHEMA,
    angle_rows,
    boundary_mode,
    detector_artifacts,
    display,
    latex_escape,
    latex_table,
    latex_value,
    main_table,
    multi_thumbs_context,
    runtime_artifacts,
    runtime_reports,
    safety_artifacts,
    sha256,
    validate_bundle,
    validate_final_candidate,
    validate_formal_scope,
    validate_test_ledger,
    verify_artifact_manifest,
)


def test_paper_metric_groups_cover_primary_local_and_boundary_claims() -> None:
    assert PRIMARY_TABLE_METRICS == (
        "W-MPJPE_mm",
        "WA-MPJPE_mm",
        "RTE_H3R_percent",
        "ATE_Sim3_m",
        "IDF1",
        "IDs",
    )
    assert LOCAL_TABLE_METRICS == (
        "MPJPE_mm",
        "PA-MPJPE_mm",
        "MPVPE_mm",
        "ATE_SE3_m",
    )
    assert TEMPORAL_TABLE_METRICS == (
        "Accel_mm_frame2",
        "ROE_joint_proxy_deg",
        "Jitter_H3R",
        "Foot_sliding_cm",
        "Coverage",
        "Detection_precision",
    )
    assert BOUNDARY_TABLE_METRICS == (
        *BOUNDARY_CAMERA_TABLE_METRICS,
        *BOUNDARY_HUMAN_TABLE_METRICS,
    )
    assert len(TABLE_METRICS) == len(set(TABLE_METRICS))
    assert (
        set(PRIMARY_TABLE_METRICS)
        | set(LOCAL_TABLE_METRICS)
        | set(TEMPORAL_TABLE_METRICS)
        | set(BOUNDARY_TABLE_METRICS)
    ) <= set(TABLE_METRICS)
    assert {
        "Post_root_m",
        "Seam_camera_t_m",
        "Seam_camera_R_deg",
    } <= set(TABLE_METRICS)

    row = {"display_name": "Method", **{metric: 1.0 for metric in TABLE_METRICS}}
    primary = latex_table(
        [row], [("display_name", "Method")], PRIMARY_TABLE_METRICS
    )
    local = latex_table([row], [("display_name", "Method")], LOCAL_TABLE_METRICS)
    boundary = latex_table(
        [row], [("display_name", "Method")], BOUNDARY_TABLE_METRICS
    )
    assert "W (mm) $\\downarrow$" in primary
    assert "ATE-Sim3 (m) $\\downarrow$" in primary
    assert "MPJPE (mm) $\\downarrow$" in local
    assert "B-Cam. T (m) $\\downarrow$" in boundary
    assert "Seam-CHRGE (m) $\\downarrow$" in boundary


def test_latex_escape_and_table_format_all_special_value_types() -> None:
    assert latex_escape("\\&%$#_{}~^") == (
        r"\textbackslash{}\&\%\$\#\_\{\}\textasciitilde{}"
        r"\textasciicircum{}"
    )
    rows = [
        {
            "display_name": "A&B_1",
            "case_count": 3,
            "fallback_array_exactness_passed": True,
            "exact_rate": 0.5,
            "accepted_W_harm_over_5pct": 3,
            "RTE_H3R_percent": 12.345,
            "ATE_Sim3_m": 0.12345,
        }
    ]
    table = latex_table(
        rows,
        [
            ("display_name", "Method"),
            ("case_count", "Cases"),
            ("fallback_array_exactness_passed", "Exact"),
            ("exact_rate", "Rate"),
            ("accepted_W_harm_over_5pct", "Harm"),
        ],
        ("RTE_H3R_percent", "ATE_Sim3_m"),
    )
    assert r"\begin{tabular}{lrrrrrr}" in table
    assert r"A\&B\_1 & 3 & Yes & 0.500 & 3 & 12.3 & 0.123" in table
    assert latex_value(False, "fallback_metric_exactness_passed") == "No"
    assert latex_value(None, "exact_rate") == "--"


def test_display_names_do_not_conflate_oracle_and_causal_operating_points() -> None:
    final = FROZEN_FINAL_METHOD
    assert "oracle cut" in display("m3_b0_only", final)
    assert "oracle cut" in display("m15_v17_gated_parent", final)
    assert "causal detector" in display(PARENT, final)
    assert display(final, final) == "Bridge3R (causal, frozen)"
    assert boundary_mode("m0_strict_human3r", final) == "none"
    assert boundary_mode("m3_b0_only", final) == "oracle_cut"
    assert boundary_mode(PARENT, final) == "causal_detector"
    assert boundary_mode(final, final) == "causal_detector"
    with pytest.raises(ValueError, match="unregistered boundary operating point"):
        boundary_mode("unknown_method", final)


def _method_summary(
    *, missing_metric: str | None = None, case_count: int = 3
) -> dict[str, Any]:
    metrics = {
        metric: {"mean": float(index + 1)}
        for index, metric in enumerate(TABLE_METRICS)
        if metric != missing_metric
    }
    return {
        method: {
            "recording_count": 1,
            "case_count": case_count,
            "metrics": metrics,
        }
        for method in EXPECTED_METHODS
    }


@pytest.mark.parametrize("missing_metric", ["RTE_H3R_percent", "Boundary_root_m"])
def test_main_table_fails_closed_when_a_required_metric_is_missing(
    missing_metric: str,
) -> None:
    summary = {"methods": _method_summary(missing_metric=missing_metric)}
    with pytest.raises(ValueError, match="missing required paper metric"):
        main_table(summary, FROZEN_FINAL_METHOD)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_main_table_requires_the_exact_seven_method_inventory(mutation: str) -> None:
    methods = _method_summary()
    if mutation == "missing":
        methods.pop("m0_strict_human3r")
    else:
        methods["posthoc_extra"] = next(iter(methods.values()))
    with pytest.raises(ValueError, match="method inventory mismatch"):
        main_table({"methods": methods}, FROZEN_FINAL_METHOD)


def _write_runtime(path: Path, case_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": RUNTIME_SCHEMA,
                "record": {"case_id": case_id, "clip_length": 150},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _runtime_bundle(runtime_path: Path, declared_sha: str) -> dict[str, Any]:
    case_id = "case_001"
    return {
        "split": "test",
        "state_path": runtime_path.parent.parent / "protocol_state.json",
        "state": {
            "run_identity": {
                "output_root": str(runtime_path.parent.parent.resolve()),
                "selected_case_ids": [case_id],
            },
            "inference": {
                case_id: {
                    "status": "complete",
                    "runtime_report_sha256": declared_sha,
                }
            },
        },
        "report": {
            "rows": [
                {
                    "case_id": case_id,
                    "diagnostics": {
                        "provenance": {
                            "runtime_report": str(runtime_path.resolve()),
                            "runtime_report_sha256": declared_sha,
                        }
                    },
                },
                {
                    "case_id": case_id,
                    "diagnostics": {
                        "provenance": {
                            "runtime_report": str(runtime_path.resolve()),
                            "runtime_report_sha256": declared_sha,
                        }
                    },
                },
            ]
        },
    }


def test_runtime_reports_are_bound_to_candidate_report_sha(tmp_path: Path) -> None:
    runtime_path = tmp_path / "formal" / "predictions" / "case_001.runtime.json"
    _write_runtime(runtime_path, "case_001")
    bundle = _runtime_bundle(runtime_path, sha256(runtime_path))
    reports = runtime_reports(bundle)
    assert [(path, payload["record"]["case_id"]) for path, payload in reports] == [
        (runtime_path.resolve(), "case_001")
    ]

    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    payload["mutated_after_candidate_report"] = True
    runtime_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime SHA differs from candidate report"):
        runtime_reports(bundle)


def test_runtime_report_binding_rejects_inconsistent_candidate_rows(
    tmp_path: Path,
) -> None:
    runtime_path = tmp_path / "formal" / "predictions" / "case_001.runtime.json"
    _write_runtime(runtime_path, "case_001")
    bundle = _runtime_bundle(runtime_path, sha256(runtime_path))
    bundle["report"]["rows"][1]["diagnostics"]["provenance"][
        "runtime_report_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="inconsistent runtime bindings"):
        runtime_reports(bundle)


def test_every_candidate_row_requires_a_runtime_binding(tmp_path: Path) -> None:
    runtime_path = tmp_path / "formal" / "predictions" / "case_001.runtime.json"
    _write_runtime(runtime_path, "case_001")
    bundle = _runtime_bundle(runtime_path, sha256(runtime_path))
    del bundle["report"]["rows"][1]["diagnostics"]["provenance"]
    with pytest.raises(ValueError, match="lacks a valid runtime binding"):
        runtime_reports(bundle)


def test_runtime_report_sha_must_also_match_completed_inference_state(
    tmp_path: Path,
) -> None:
    runtime_path = tmp_path / "formal" / "predictions" / "case_001.runtime.json"
    _write_runtime(runtime_path, "case_001")
    bundle = _runtime_bundle(runtime_path, sha256(runtime_path))
    bundle["state"]["inference"]["case_001"]["runtime_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="completed inference state"):
        runtime_reports(bundle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _formal_manifest_rows(split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_rows = []
    evaluator_rows = []
    for recording_index in range(FORMAL_SCOPE[split]["recording_count"]):
        recording = f"recording_{recording_index:03d}"
        for angle in ANGLE_STRATA:
            case_id = f"{split}_{recording}_{angle}"
            common = {
                "case_id": case_id,
                "recording": recording,
                "protocol": artifacts.PROTOCOL,
                "split": split,
            }
            runtime_rows.append(dict(common))
            evaluator_rows.append(
                {**common, "angle_stratum_evaluator_only": angle}
            )
    return runtime_rows, evaluator_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("mutation", ["wrong_count", "duplicate_angle", "selected_ids"])
def test_formal_scope_is_exact_and_three_angle_complete(
    tmp_path: Path, mutation: str
) -> None:
    runtime_rows, evaluator_rows = _formal_manifest_rows("test")
    selected_ids = [row["case_id"] for row in runtime_rows]
    if mutation == "wrong_count":
        runtime_rows.pop()
        evaluator_rows.pop()
        selected_ids.pop()
    elif mutation == "duplicate_angle":
        evaluator_rows[-1]["angle_stratum_evaluator_only"] = "small"
    else:
        selected_ids[-1] = "different_case"
    runtime_manifest = tmp_path / "runtime.jsonl"
    evaluator_manifest = tmp_path / "evaluator.jsonl"
    _write_jsonl(runtime_manifest, runtime_rows)
    _write_jsonl(evaluator_manifest, evaluator_rows)
    summary = {
        "selected_case_count": FORMAL_SCOPE["test"]["case_count"],
        "evaluator_unavailable_case_count": 0,
        "case_count": FORMAL_SCOPE["test"]["case_count"],
        "recording_count": FORMAL_SCOPE["test"]["recording_count"],
    }
    state = {
        "selected_case_count": FORMAL_SCOPE["test"]["case_count"],
        "run_identity": {"selected_case_ids": selected_ids},
    }
    with pytest.raises(ValueError, match="formal|angle|selected case IDs"):
        validate_formal_scope(
            "test",
            summary,
            state,
            runtime_manifest,
            evaluator_manifest,
            tmp_path / "summary.json",
        )


def test_validate_bundle_requires_candidate_report_sha_in_completed_state(
    tmp_path: Path,
) -> None:
    candidate_source = tmp_path / "candidate.json"
    _write_json(candidate_source, {"schema_version": FINAL_SCHEMA})
    report_path = tmp_path / "candidate_report.json"
    _write_json(
        report_path,
        {
            "schema_version": PROBE_SCHEMA,
            "errors": [],
            "candidate_source": str(candidate_source.resolve()),
        },
    )
    case_metrics = tmp_path / "case_metrics.csv"
    recording_metrics = tmp_path / "recording_metrics.csv"
    case_metrics.write_text("case_id\ncase_001\n", encoding="utf-8")
    recording_metrics.write_text("recording\nrecording_001\n", encoding="utf-8")
    runtime_manifest = tmp_path / "test.runtime.jsonl"
    evaluator_manifest = tmp_path / "test.evaluator.jsonl"
    runtime_rows, evaluator_rows = _formal_manifest_rows("test")
    _write_jsonl(runtime_manifest, runtime_rows)
    _write_jsonl(evaluator_manifest, evaluator_rows)
    state_path = tmp_path / "protocol_state.json"
    state = {
        "schema_version": STATE_SCHEMA,
        "status": "complete",
        "split": "test",
        "smoke_subset": False,
        "max_cases": None,
        "run_identity_sha256": "identity",
        "selected_case_count": FORMAL_SCOPE["test"]["case_count"],
        "runtime_manifest": str(runtime_manifest.resolve()),
        "evaluator_manifest": str(evaluator_manifest.resolve()),
        "run_identity": {
            "runtime_manifest": str(runtime_manifest.resolve()),
            "runtime_manifest_sha256": sha256(runtime_manifest),
            "evaluator_manifest": str(evaluator_manifest.resolve()),
            "evaluator_manifest_sha256": sha256(evaluator_manifest),
            "selected_case_ids": [row["case_id"] for row in runtime_rows],
        },
        "candidate_reports": {
            "candidate": {
                "status": "complete",
                "output_sha256": sha256(report_path),
            }
        },
    }
    _write_json(state_path, state)
    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": AGGREGATE_SCHEMA,
            "split": "test",
            "parent": PARENT,
            "protocol_state": str(state_path.resolve()),
            "protocol_state_sha256": sha256(state_path),
            "run_identity_sha256": "identity",
            "case_metrics": str(case_metrics.resolve()),
            "case_metrics_sha256": sha256(case_metrics),
            "recording_metrics": str(recording_metrics.resolve()),
            "recording_metrics_sha256": sha256(recording_metrics),
            "candidate_report": str(report_path.resolve()),
            "candidate_report_sha256": sha256(report_path),
            "candidate_source": str(candidate_source.resolve()),
            "candidate_source_sha256": sha256(candidate_source),
            "selected_case_count": FORMAL_SCOPE["test"]["case_count"],
            "evaluator_unavailable_case_count": 0,
            "case_count": FORMAL_SCOPE["test"]["case_count"],
            "recording_count": FORMAL_SCOPE["test"]["recording_count"],
        },
    )
    bundle = validate_bundle(summary_path, "test")
    assert bundle["candidate_report"] == report_path
    assert bundle["runtime_manifest"] == runtime_manifest.resolve()
    assert bundle["evaluator_manifest"] == evaluator_manifest.resolve()

    original_runtime = runtime_manifest.read_text(encoding="utf-8")
    runtime_manifest.write_text('{"case_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_manifest_sha256 mismatch"):
        validate_bundle(summary_path, "test")
    runtime_manifest.write_text(original_runtime, encoding="utf-8")

    state["runtime_manifest"] = str(evaluator_manifest.resolve())
    _write_json(state_path, state)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["protocol_state_sha256"] = sha256(state_path)
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="runtime_manifest path differs"):
        validate_bundle(summary_path, "test")
    state["runtime_manifest"] = str(runtime_manifest.resolve())

    state["candidate_reports"]["candidate"]["output_sha256"] = "0" * 64
    _write_json(state_path, state)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["protocol_state_sha256"] = sha256(state_path)
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="absent from completed protocol state"):
        validate_bundle(summary_path, "test")


def _valid_final_payload(path: Path) -> dict[str, Any]:
    return {
        "schema_version": FINAL_SCHEMA,
        "protocol": artifacts.PROTOCOL,
        "frozen_before_test": True,
        "test_metrics_read": False,
        "source_candidate_name": FROZEN_FINAL_METHOD,
        "fallback_to_parent": False,
        "candidates": [
            {
                "name": FROZEN_FINAL_METHOD,
                "geometry": {
                    "name": FROZEN_FINAL_METHOD,
                    "camera_alpha": 1.0,
                    "boundary_kind": "translation",
                    "boundary_blend": 0.5,
                },
                "identity": None,
            }
        ],
        "frozen_artifact_path": str(path.resolve()),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_alpha", 0.5),
        ("boundary_kind", "rotation"),
        ("boundary_blend", 1.0),
        ("identity", {"enabled": True}),
        ("fallback_to_parent", True),
    ],
)
def test_final_candidate_contract_is_exact(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "frozen_final_candidate.json"
    valid = _valid_final_payload(path)
    _write_json(path, valid)
    assert validate_final_candidate(valid, path) == FROZEN_FINAL_METHOD
    mutated = json.loads(json.dumps(valid))
    if field in {"fallback_to_parent"}:
        mutated[field] = value
    elif field == "identity":
        mutated["candidates"][0][field] = value
    else:
        mutated["candidates"][0]["geometry"][field] = value
    with pytest.raises(ValueError, match="frozen final candidate contract"):
        validate_final_candidate(mutated, path)


@pytest.mark.parametrize(
    "mutated_key",
    ["schema_version", "candidate_json_sha256", "run_identity_sha256", "output_root"],
)
def test_test_consumption_ledger_is_semantically_bound(
    tmp_path: Path, mutated_key: str
) -> None:
    final_path = tmp_path / "frozen" / "frozen_final_candidate.json"
    _write_json(final_path, _valid_final_payload(final_path))
    output_root = tmp_path / "formal" / "test"
    state = {
        "run_identity_sha256": "identity-sha",
        "run_identity": {"output_root": str(output_root.resolve())},
        "candidate_json": [
            {
                "path": str(final_path.resolve()),
                "sha256": sha256(final_path),
                "source_candidate_name": FROZEN_FINAL_METHOD,
                "frozen_before_test": True,
                "test_metrics_read": False,
            }
        ],
    }
    ledger_path = final_path.with_suffix(
        final_path.suffix + ".test-consumption.json"
    )
    ledger = {
        "schema_version": TEST_LEDGER_SCHEMA,
        "candidate_json": str(final_path.resolve()),
        "candidate_json_sha256": sha256(final_path),
        "run_identity_sha256": "identity-sha",
        "output_root": str(output_root.resolve()),
    }
    _write_json(ledger_path, ledger)
    assert validate_test_ledger(ledger_path, final_path, state) == ledger
    ledger[mutated_key] = "wrong"
    _write_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="ledger content mismatch"):
        validate_test_ledger(ledger_path, final_path, state)


def test_runtime_artifacts_never_label_multi_method_time_as_end_to_end() -> None:
    final = "selected_candidate"
    bundle = {
        "split": "test",
        "report": {
            "rows": [
                {
                    "case_id": "case_001",
                    "candidate": final,
                    "status": "complete",
                    "diagnostics": {"postprocess_seconds": 0.25},
                }
            ]
        },
    }
    runtime = {
        "record": {"case_id": "case_001", "clip_length": 150},
        "runtime": {
            "m0_forward": {"frames": 150, "seconds": 15.0, "fps": 10.0},
            "causal_gru_detector": {"seconds": 3.0},
        },
        "total_process_seconds": 120.0,
        "environment": {
            "process_peak_rss_bytes": 2 * 1024**3,
            "gpu": "Synthetic GPU",
            "precision": "FP32",
        },
    }
    cases, summaries = runtime_artifacts(
        [bundle], {"test": [(Path("case_001.runtime.json"), runtime)]}, final
    )
    assert cases[0]["whole_multi_method_protocol_seconds"] == 120.0
    assert "end_to_end_fps" not in cases[0]
    assert "gpu_peak_memory" not in cases[0]
    assert summaries[0]["deployed_single_method_fps_available"] is False
    assert summaries[0]["gpu_peak_memory_available"] is False
    assert "not deployed single-method end-to-end throughput" in summaries[0][
        "timing_contract"
    ]
    assert summaries[0]["process_peak_host_rss_gib_max"] == 2.0
    assert summaries[0]["whole_multi_method_protocol_seconds_mean"] == 120.0
    assert all("end_to_end" not in key for key in summaries[0])
    assert "whole_multi_method_protocol_fps" not in summaries[0]


def _angle_summary() -> dict[str, Any]:
    raw_metrics = {
        metric: float(index + 1) for index, metric in enumerate(TABLE_METRICS)
    }
    return {
        "methods": _method_summary(case_count=3),
        "angle_strata": {
            angle: {
                method: {"case_count": 1, "metrics": dict(raw_metrics)}
                for method in EXPECTED_METHODS
            }
            for angle in ANGLE_STRATA
        },
    }


@pytest.mark.parametrize("mutation", ["missing_stratum", "missing_method", "bad_count"])
def test_angle_rows_fail_closed_on_incomplete_coverage(mutation: str) -> None:
    summary = _angle_summary()
    if mutation == "missing_stratum":
        summary["angle_strata"].pop("extreme")
    elif mutation == "missing_method":
        summary["angle_strata"]["small"].pop("m0_strict_human3r")
    else:
        summary["angle_strata"]["small"]["m0_strict_human3r"]["case_count"] = 2
    with pytest.raises(ValueError, match="angle"):
        angle_rows(summary, list(EXPECTED_METHODS), FROZEN_FINAL_METHOD)
    valid = _angle_summary()
    assert len(
        angle_rows(valid, list(EXPECTED_METHODS), FROZEN_FINAL_METHOD)
    ) == 3 * len(EXPECTED_METHODS)


def _detector_runtime(
    positive_indices: set[int], *, case_id: str = "case_001"
) -> dict[str, Any]:
    labels = [int(index in positive_indices) for index in range(150)]
    rows = [
        {
            "pair_idx": index,
            "pred": labels[index],
            "prob": 0.9 if labels[index] else 0.1,
            "threshold": 0.5,
        }
        for index in range(1, 150)
    ]
    proposal = min(positive_indices) if positive_indices else None
    return {
        "record": {"case_id": case_id, "clip_length": 150, "boundary_index": 75},
        "runtime": {
            "causal_gru_detector": {
                "labels": labels,
                "rows": rows,
                "proposal_boundary": proposal,
                "first_positive_index": proposal,
                "seconds": 1.0,
            }
        },
    }


def test_detector_artifacts_report_frozen_full_metric_contract() -> None:
    bundle = {"split": "test"}
    runtime = _detector_runtime({70, 75})
    cases, summaries = detector_artifacts(
        [bundle], {"test": [(Path("case.runtime.json"), runtime)]}
    )
    assert cases[0]["status"] == "early"
    assert cases[0]["signed_error_frames"] == -5
    summary = summaries[0]
    assert summary["early_count"] == 1
    assert summary["detector_precision"] == pytest.approx(0.5)
    assert summary["detector_recall"] == pytest.approx(1.0)
    assert summary["detector_f1"] == pytest.approx(2 / 3)
    assert summary["false_positives_per_100_frames"] == pytest.approx(2 / 3)
    assert summary["mean_signed_first_positive_offset_frames"] == -5
    assert summary["brier"] == pytest.approx((148 * 0.01 + 0.81) / 149)


@pytest.mark.parametrize("mutation", ["missing_pair", "duplicate_pair", "bad_prob", "bad_proposal"])
def test_detector_probability_evidence_fails_closed(mutation: str) -> None:
    runtime = _detector_runtime({75})
    detector = runtime["runtime"]["causal_gru_detector"]
    if mutation == "missing_pair":
        detector["rows"].pop()
    elif mutation == "duplicate_pair":
        detector["rows"][-1]["pair_idx"] = 1
    elif mutation == "bad_prob":
        detector["rows"][0]["prob"] = 2.0
    else:
        detector["proposal_boundary"] = 76
    with pytest.raises(ValueError, match="detector"):
        detector_artifacts(
            [{"split": "test"}],
            {"test": [(Path("case.runtime.json"), runtime)]},
        )


def test_safety_uses_ungated_materialization_and_vacuous_reuse_semantics() -> None:
    value = {
        "gate_enabled": False,
        "case_count": 2,
        "accepted_count": 2,
        "fallback_count": 0,
        "acceptance_rate": 1.0,
        "missing_gate_cases": [],
        "fallback_array_exactness_passed": True,
        "fallback_array_audit_missing_cases": [],
        "fallback_array_mismatches": [],
        "fallback_metric_exactness_passed": True,
        "fallback_metric_mismatches": [],
        "accepted_W_harm_over_5pct": 0,
        "accepted_W_harm_over_10pct": 0,
        "accepted_W_harm_over_20pct": 0,
        "accepted_W_improvement_rate": 1.0,
        "worst_accepted_W_ratio": 0.9,
    }
    rows, _, _ = safety_artifacts(
        [{"split": "test", "summary": {"safety": {FROZEN_FINAL_METHOD: value}}, "report": {"rows": []}}],
        FROZEN_FINAL_METHOD,
    )
    row = rows[0]
    assert row["gate_state"] == "Disabled"
    assert row["materialized_count"] == 2
    assert row["detector_miss_parent_reuse_count"] == 0
    assert row["fallback_array_exactness_observed"] == "N/A (0 reuse)"
    value["gate_enabled"] = True
    with pytest.raises(ValueError, match="ungated final"):
        safety_artifacts(
            [{"split": "test", "summary": {"safety": {FROZEN_FINAL_METHOD: value}}, "report": {"rows": []}}],
            FROZEN_FINAL_METHOD,
        )


def test_main_writes_every_declared_table_and_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_name = FROZEN_FINAL_METHOD
    final_path = tmp_path / "frozen" / "frozen_final_candidate.json"
    _write_json(final_path, _valid_final_payload(final_path))
    method_names = EXPECTED_METHODS
    metric_values = {
        metric: float(index + 1) / (100.0 if metric.endswith("_m") else 1.0)
        for index, metric in enumerate(TABLE_METRICS)
    }
    method_summary = {
        method: {
            "recording_count": 1,
            "case_count": 3,
            "metrics": {
                metric: {"mean": value, "ci95": [value, value]}
                for metric, value in metric_values.items()
            },
        }
        for method in method_names
    }
    bundles: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for split in ("development", "holdout", "test"):
        root = tmp_path / "formal" / split
        state_path = root / "protocol_state.json"
        ledger = final_path.with_suffix(final_path.suffix + ".test-consumption.json")
        if split == "test":
            _write_json(
                ledger,
                {
                    "schema_version": TEST_LEDGER_SCHEMA,
                    "candidate_json": str(final_path.resolve()),
                    "candidate_json_sha256": sha256(final_path),
                    "run_identity_sha256": "test-identity",
                    "output_root": str(root.resolve()),
                },
            )
        state = {
            "split": split,
            "run_identity_sha256": f"{split}-identity",
            "run_identity": {"output_root": str(root.resolve())},
            "candidate_json": [
                {
                    "path": str(final_path.resolve()),
                    "sha256": sha256(final_path),
                    "source_candidate_name": final_name,
                    "frozen_before_test": True,
                    "test_metrics_read": False,
                }
            ],
            **(
                {
                    "test_consumption_ledger": str(ledger.resolve()),
                    "test_consumption_ledger_sha256": sha256(ledger),
                }
                if split == "test"
                else {}
            ),
        }
        _write_json(state_path, state)
        report_path = root / "candidate_report.json"
        report = {
            "rows": [
                {
                    "case_id": f"{split}_case",
                    "candidate": final_name,
                    "status": "complete",
                    "diagnostics": {"postprocess_seconds": 0.25},
                }
            ]
        }
        _write_json(report_path, report)
        case_metrics = root / "case_metrics.csv"
        recording_metrics = root / "recording_metrics.csv"
        runtime_manifest = root / "runtime_manifest.jsonl"
        evaluator_manifest = root / "evaluator_manifest.jsonl"
        case_metrics.write_text("case_id\nsynthetic\n", encoding="utf-8")
        recording_metrics.write_text("recording\nsynthetic\n", encoding="utf-8")
        runtime_manifest.write_text('{"case_id":"synthetic"}\n', encoding="utf-8")
        evaluator_manifest.write_text('{"case_id":"synthetic"}\n', encoding="utf-8")
        summary_path = root / "aggregate" / "summary.json"
        summary = {
            "split": split,
            "recording_count": 1,
            "case_count": 3,
            "selected_case_count": 1,
            "evaluator_unavailable_case_count": 0,
            "aggregation": "synthetic recording macro",
            "candidate_source_sha256": sha256(final_path),
            "methods": method_summary,
            "angle_strata": {
                angle: {
                    method: {"case_count": 1, "metrics": metric_values}
                    for method in method_names
                }
                for angle in ANGLE_STRATA
            },
            "safety": {
                final_name: {
                    "gate_enabled": False,
                    "case_count": 1,
                    "accepted_count": 1,
                    "fallback_count": 0,
                    "acceptance_rate": 1.0,
                    "missing_gate_cases": [],
                    "fallback_array_exactness_passed": True,
                    "fallback_array_audit_missing_cases": [],
                    "fallback_array_mismatches": [],
                    "fallback_metric_exactness_passed": True,
                    "fallback_metric_mismatches": [],
                    "accepted_W_harm_over_5pct": 0,
                    "accepted_W_harm_over_10pct": 0,
                    "accepted_W_harm_over_20pct": 0,
                    "worst_accepted_W_ratio": 0.9,
                    "accepted_W_improvement_rate": 1.0,
                }
            },
        }
        _write_json(summary_path, summary)
        runtime_path = root / "predictions" / f"{split}_case.runtime.json"
        labels = [False] * 150
        labels[75] = True
        detector_rows = [
            {
                "pair_idx": index,
                "pred": int(labels[index]),
                "prob": 0.9 if labels[index] else 0.1,
                "threshold": 0.5,
            }
            for index in range(1, 150)
        ]
        runtime = {
            "schema_version": RUNTIME_SCHEMA,
            "record": {
                "case_id": f"{split}_case",
                "clip_length": 150,
                "boundary_index": 75,
            },
            "runtime": {
                "m0_forward": {"frames": 150, "seconds": 15.0, "fps": 10.0},
                "causal_gru_detector": {
                    "seconds": 3.0,
                    "labels": labels,
                    "rows": detector_rows,
                    "proposal_boundary": 75,
                    "first_positive_index": 75,
                },
            },
            "total_process_seconds": 120.0,
            "environment": {
                "process_peak_rss_bytes": 2 * 1024**3,
                "gpu": "Synthetic GPU",
                "precision": "FP32",
            },
        }
        _write_json(runtime_path, runtime)
        bundles[split] = {
            "split": split,
            "summary_path": summary_path.resolve(),
            "summary": summary,
            "state_path": state_path.resolve(),
            "state": state,
            "runtime_manifest": runtime_manifest.resolve(),
            "evaluator_manifest": evaluator_manifest.resolve(),
            "case_metrics": case_metrics.resolve(),
            "recording_metrics": recording_metrics.resolve(),
            "candidate_report": report_path.resolve(),
            "candidate_source": final_path.resolve(),
            "report": report,
            "formal_scope": {
                "selected_case_count": 1,
                "structural_recording_count": 43 if split == "test" else 1,
                "evaluable_case_count": 1,
                "evaluator_unavailable_case_count": 0,
            },
        }
        runtimes[split] = [(runtime_path.resolve(), runtime)]

    literature = tmp_path / "Multi-THuMBS.pdf"
    literature.write_bytes(b"synthetic PDF placeholder")
    output = tmp_path / "paper_artifacts"
    monkeypatch.setattr(
        artifacts,
        "parse_args",
        lambda: Namespace(
            development=Path("development"),
            holdout=Path("holdout"),
            test=Path("test"),
            final_candidate=final_path,
            multi_thumbs_pdf=literature,
            output=output,
        ),
    )
    monkeypatch.setattr(
        artifacts, "validate_bundle", lambda _value, split: bundles[split]
    )
    monkeypatch.setattr(
        artifacts, "runtime_reports", lambda bundle: runtimes[bundle["split"]]
    )
    artifacts.main()

    required = {
        "recording_macro_primary.tex",
        "recording_macro_local.tex",
        "recording_macro_boundary.tex",
        "runtime_components.tex",
        "artifact_manifest.json",
        "artifact_manifest.json.sha256",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert "RTE-H3R" in (output / "recording_macro_primary.tex").read_text()
    assert "ATE-Sim3" in (output / "recording_macro_primary.tex").read_text()
    assert "PA-MPJPE" in (output / "recording_macro_local.tex").read_text()
    assert "ATE-SE3" in (output / "recording_macro_local.tex").read_text()
    assert "B-Cam. T" in (output / "recording_macro_boundary.tex").read_text()
    assert "Seam-root" in (output / "recording_macro_boundary.tex").read_text()
    assert "Whole protocol (s)" in (output / "runtime_components.tex").read_text()
    assert "end-to-end" not in (output / "runtime_components.tex").read_text().lower()
    detector_tex = (output / "detector_table.tex").read_text(encoding="utf-8")
    assert "development" not in detector_tex.lower()
    assert "holdout" not in detector_tex.lower()
    assert "FP/100 frames" in detector_tex
    assert "First-positive offset (frames)" in detector_tex
    safety_tex = (output / "safety_table.tex").read_text(encoding="utf-8")
    assert "development" not in safety_tex.lower()
    assert "holdout" not in safety_tex.lower()
    assert "Disabled" in safety_tex
    assert "Materialized" in safety_tex
    assert "Detector-miss parent reuse" in safety_tex
    assert "N/A (0 reuse)" in safety_tex
    multithumbs_tex = (output / "multithumbs_context.tex").read_text(
        encoding="utf-8"
    )
    bridge_ate_row = next(
        line for line in multithumbs_tex.splitlines() if "Bridge3R ATE-Sim3 (m)" in line
    )
    source_ate_row = next(
        line
        for line in multithumbs_tex.splitlines()
        if "Multi-THuMBS ATE (source-defined)" in line
    )
    assert bridge_ate_row != source_ate_row
    assert " & -- & " in bridge_ate_row
    assert " & -- & 0.1 & " in source_ate_row
    with (output / "recording_macro_main.csv").open(newline="") as handle:
        fields = next(csv.reader(handle))
    assert "boundary_mode" in fields
    assert "RTE_H3R_percent" in fields
    assert "Boundary_camera_t_m" in fields
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["schema_version"] == ARTIFACT_SCHEMA
    declared = {Path(row["path"]).name for row in manifest["outputs"]}
    assert required - {
        "artifact_manifest.json",
        "artifact_manifest.json.sha256",
    } <= declared
    assert manifest["contracts"]["deployed_single_method_fps_available"] is False
    assert manifest["contracts"]["gpu_peak_memory_available"] is False
    sources = {(row["path"], row["role"]) for row in manifest["sources"]}
    for split, bundle in bundles.items():
        assert (
            str(bundle["runtime_manifest"]),
            f"{split}:runtime_manifest",
        ) in sources
        assert (
            str(bundle["evaluator_manifest"]),
            f"{split}:evaluator_manifest",
        ) in sources
    manifest_path = output / "artifact_manifest.json"
    sidecar = output / "artifact_manifest.json.sha256"
    assert sidecar.read_text(encoding="utf-8") == (
        f"{sha256(manifest_path)}  artifact_manifest.json\n"
    )
    verified = verify_artifact_manifest(manifest_path)
    assert verified["expected_output_names"]
    primary_path = output / "recording_macro_primary.tex"
    original_primary = primary_path.read_text(encoding="utf-8")
    primary_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outputs SHA/size mismatch"):
        verify_artifact_manifest(manifest_path)
    primary_path.write_text(original_primary, encoding="utf-8")
    verify_artifact_manifest(manifest_path)

    stale_path = output / "stale_release_file.txt"
    stale_path.write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing/stale files"):
        verify_artifact_manifest(manifest_path)
    with pytest.raises(ValueError, match="missing/stale files"):
        artifacts.main()
    assert not manifest_path.exists()
    assert not sidecar.exists()


def test_multithumbs_metadata_and_ate_definitions_remain_separate(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "Multi-THuMBS.pdf"
    pdf.write_bytes(b"paper")
    context = multi_thumbs_context(pdf)
    assert context["source"]["paper"].endswith("Beyond Video Shots")
    assert MULTI_THUMBS_EGOBODY == {
        "W-MPJPE_mm": 99.2,
        "WA-MPJPE_mm": 72.8,
        "MPJPE_mm": 72.0,
        "MPVPE_mm": 94.9,
        "Accel_mm_frame2": 6.0,
        "IDs": 0.0,
        "ATE_source_defined": 0.1,
    }
    assert "does not establish" in context["multi_thumbs_protocol"][
        "ate_definition"
    ]
