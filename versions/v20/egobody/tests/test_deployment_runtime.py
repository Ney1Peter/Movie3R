from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from versions.v20.egobody import benchmark_deployment_runtime as benchmark
from versions.v20.egobody import deployment_runtime as deployment


SELECTED_LINES = (1, 2, 3, 97, 98, 99, 193, 194, 195)
STRATA = ("extreme", "medium", "small")


def _write(path: Path, value: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        value = value.encode("utf-8")
    path.write_bytes(value)
    return path


def _json(path: Path, value: Any) -> Path:
    return _write(path, json.dumps(value, sort_keys=True) + "\n")


def _frozen_candidate(tmp_path: Path) -> deployment.FrozenCandidate:
    frozen = tmp_path / "frozen" / "final.json"
    provenance: dict[str, str] = {}
    for path_key, sha_key in deployment.PROVENANCE_BINDINGS:
        source = _write(tmp_path / "provenance" / f"{path_key}.bin", path_key)
        provenance[path_key] = str(source.resolve())
        provenance[sha_key] = deployment.file_sha256(source)
    payload = {
        "schema_version": deployment.FINAL_SCHEMA,
        "protocol": deployment.PROTOCOL,
        "frozen_artifact_path": str(frozen.resolve()),
        "frozen_before_test": True,
        "test_metrics_read": False,
        "source_candidate_name": deployment.EXPECTED_CANDIDATE_NAME,
        "fallback_to_parent": False,
        "qualified_holdout_candidates": [deployment.EXPECTED_CANDIDATE_NAME],
        "candidates": [
            {
                "name": deployment.EXPECTED_CANDIDATE_NAME,
                "geometry": {
                    "name": deployment.EXPECTED_CANDIDATE_NAME,
                    "camera_alpha": 1.0,
                    "boundary_kind": "translation",
                    "boundary_blend": 0.5,
                },
                "identity": None,
            }
        ],
        "provenance": provenance,
    }
    _json(frozen, payload)
    return deployment.validate_frozen_candidate(
        frozen, expected_sha256=deployment.file_sha256(frozen)
    )


def _case_metadata(line: int) -> tuple[str, str, str]:
    if line in (1, 2, 3):
        recording, offset = "recording_a", line - 1
    elif line in (97, 98, 99):
        recording, offset = "recording_b", line - 97
    elif line in (193, 194, 195):
        recording, offset = "recording_c", line - 193
    elif line == 4:
        return "warmup_case_extreme_kinect_rgb", "recording_warmup", "extreme"
    else:
        return f"unused_{line:03d}", f"unused_recording_{line:03d}", "unused"
    stratum = STRATA[offset]
    return f"{recording}_{stratum}_kinect_rgb", recording, stratum


def _runtime_row(line: int) -> dict[str, Any]:
    case_id, recording, stratum = _case_metadata(line)
    count = 150 if line in (*SELECTED_LINES, 4) else 2
    return {
        "case_id": case_id,
        "recording": recording,
        "image_members": [f"rgb/{case_id}/{index:03d}.jpg" for index in range(count)],
        "boundary_index": 75,
        "evaluator_only_decoy": {"must_not_be_read": True},
    }


def _preregistration(tmp_path: Path) -> benchmark.FrozenPreregistration:
    manifest = tmp_path / "manifests" / "development.runtime.jsonl"
    rows = [_runtime_row(line) for line in range(1, 196)]
    _write(manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    case_rows = []
    for line in SELECTED_LINES:
        case_id, recording, stratum = _case_metadata(line)
        case_rows.append(
            {
                "manifest_line": line,
                "case_id": case_id,
                "recording": recording,
                "angle_stratum": stratum,
            }
        )
    warm_id, warm_recording, warm_stratum = _case_metadata(4)
    payload = {
        "schema_version": benchmark.PREREGISTRATION_SCHEMA,
        "protocol": deployment.PROTOCOL,
        "frozen_before_holdout_and_test": True,
        "holdout_metrics_read": False,
        "test_metrics_read": False,
        "final_candidate_available_at_freeze": False,
        "source_split": "development",
        "selection_rule": {
            "case_count": 9,
            "no_replacement": True,
        },
        "cases": case_rows,
        "warmup": {
            "manifest_line": 4,
            "case_id": warm_id,
            "recording": warm_recording,
            "angle_stratum": warm_stratum,
            "included_in_reported_cases": False,
            "included_in_steady_state_timing": False,
        },
        "source": {
            "development_runtime_manifest": str(manifest.resolve()),
            "development_runtime_manifest_sha256": deployment.file_sha256(manifest),
        },
        "execution_contract": {
            "steady_state_repeats_per_case": 3,
            "ground_truth_access": False,
            "evaluator_access": False,
        },
    }
    path = _json(tmp_path / "frozen" / "runtime_prereg.json", payload)
    return benchmark.validate_preregistration(
        path, expected_sha256=deployment.file_sha256(path)
    )


class FakeRuntime:
    def __init__(self, selection: benchmark.BenchmarkSelection):
        self.case_index = {
            case.case_id: index for index, case in enumerate(selection.cases)
        }
        self.calls: list[tuple[str, str, int | None]] = []
        self.load_metrics = {
            "seconds": 10.0,
            "cuda_max_memory_allocated_bytes": 1000,
            "cuda_max_memory_reserved_bytes": 2000,
            "process_peak_rss_bytes": 3000,
            "forward_calls": 0,
            "forward_frames": 0,
            "branch": "model_detector_topology_load",
        }
        self.artifacts = {"synthetic": True}

    def measure_case(
        self,
        *,
        case_id: str,
        paths: list[Path],
        phase: str,
        repeat_index: int | None,
    ) -> dict[str, Any]:
        self.calls.append((phase, case_id, repeat_index))
        if phase == "warmup":
            seconds = 99.0
            branch = "detector_missed_lazy_exact_parent"
            calls, frames = 1, 150
        else:
            seconds = float(self.case_index[case_id] + 1 + int(repeat_index))
            branch = (
                "detector_missed_lazy_exact_parent"
                if self.case_index[case_id] % 2
                else "detected_b0_ungated_translation_b050"
            )
            calls = 1 if "missed" in branch else 2
            frames = 150 if calls == 1 else 151
        return {
            "case_id": case_id,
            "phase": phase,
            "repeat_index": repeat_index,
            "seconds": seconds,
            "cuda_max_memory_allocated_bytes": 100 + calls,
            "cuda_max_memory_reserved_bytes": 200 + calls,
            "process_peak_rss_bytes": 300 + calls,
            "branch": branch,
            "forward_calls": calls,
            "forward_frames": frames,
            "input_frames": len(paths),
            "output_frames": len(paths),
        }


def _synthetic_resolver(
    deployment_input: Mapping[str, Any], staged_root: Path
) -> list[Path]:
    assert staged_root.name == "synthetic_staged"
    return [Path(value) for value in deployment_input["image_paths"]]


def test_frozen_selection_and_provenance_are_fail_closed(tmp_path: Path) -> None:
    candidate = _frozen_candidate(tmp_path / "candidate")
    assert candidate.name == deployment.EXPECTED_CANDIDATE_NAME
    assert len(candidate.provenance_files) == 4

    tampered = Path(candidate.provenance_files[0]["path"])
    tampered.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        deployment.validate_frozen_candidate(
            candidate.path, expected_sha256=candidate.sha256
        )

    prereg = _preregistration(tmp_path / "prereg")
    selection = benchmark.select_preregistered_cases(prereg)
    assert [case.manifest_line for case in selection.cases] == list(SELECTED_LINES)
    assert selection.warmup.manifest_line == 4
    assert [case.angle_stratum for case in selection.cases[:3]] == list(STRATA)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        benchmark.validate_preregistration(prereg.path, expected_sha256="0" * 64)


def test_deployment_interface_is_invariant_to_boundary_annotation_mutation() -> None:
    record = _runtime_row(1)
    first = deployment.deployment_input_from_record(record)
    record["boundary_index"] = 1
    second = deployment.deployment_input_from_record(record)
    del record["boundary_index"]
    third = deployment.deployment_input_from_record(record)
    assert first == second == third
    assert set(first) == {"case_id", "recording", "image_paths", "frame_count"}


def test_preregistered_execution_fps_formula_and_output_contract(tmp_path: Path) -> None:
    candidate = _frozen_candidate(tmp_path / "candidate")
    prereg = _preregistration(tmp_path / "prereg")
    selection = benchmark.select_preregistered_cases(prereg)
    runtime = FakeRuntime(selection)

    warmup, runs = benchmark.execute_benchmark_plan(
        runtime,
        selection,
        tmp_path / "synthetic_staged",
        path_resolver=_synthetic_resolver,
    )
    assert runtime.calls[0] == ("warmup", selection.warmup.case_id, None)
    assert len(runtime.calls) == 1 + 9 * 3
    expected_order = [
        ("steady", case.case_id, repeat)
        for repeat, case in benchmark.rotated_execution_order(selection.cases)
    ]
    assert runtime.calls[1:] == expected_order
    assert warmup["seconds"] == 99.0

    aggregate = benchmark.aggregate_steady_runs(runs)
    expected_sum_of_medians = sum(float(index + 2) for index in range(9))
    assert aggregate["sum_case_median_seconds"] == expected_sum_of_medians
    assert aggregate["nonreportable_diagnostic_fps"] == pytest.approx(
        9 * 150 / expected_sum_of_medians
    )
    assert aggregate["reported_fps"] is None

    support_file = _write(tmp_path / "support" / "deployment_source.py", "source\n")
    support = benchmark.build_support_hash_inventory(
        {support_file: {"synthetic_deployment_source"}},
        workspace_root=tmp_path,
    )

    report = benchmark.build_benchmark_report(
        candidate=candidate,
        preregistration=prereg,
        selection=selection,
        load_metrics=runtime.load_metrics,
        warmup=warmup,
        runs=runs,
        gpu_isolation={"verified": True, "unrelated_processes": [], "reason": None},
        runtime_artifacts=runtime.artifacts,
        post_test_support_provenance=support,
    )
    benchmark.validate_benchmark_report(report)
    assert report["reportable_fps"] is False
    assert report["reporting_gate"]["reported_fps"] is None
    assert report["reporting_gate"]["equivalence_audit"]["status"] == "not_proven"
    assert len(report["steady_state"]["runs"]) == 27
    assert report["load"]["forward_calls"] == 0
    assert report["warmup"]["manifest_line"] == 4
    assert report["post_test_support_provenance"]["provenance_timing"] == (
        "post_test_supporting"
    )
    assert report["post_test_support_provenance"][
        "test_metric_artifacts_read_or_inventoried"
    ] is False

    invalid = copy.deepcopy(report)
    invalid["reportable_fps"] = True
    with pytest.raises(ValueError, match="must not be reportable"):
        benchmark.validate_benchmark_report(invalid)

    invalid_support = copy.deepcopy(report)
    invalid_support["post_test_support_provenance"]["replaces_pre_test_lock"] = True
    with pytest.raises(ValueError, match="post-Test provenance"):
        benchmark.validate_benchmark_report(invalid_support)


def test_post_test_support_hash_inventory_is_canonical_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path / "code" / "a.py", "A = 1\n")
    second = _write(tmp_path / "config" / "b.json", "{}\n")
    roles = {
        second: {"config"},
        first: {"runtime", "source"},
    }
    initial = benchmark.build_support_hash_inventory(roles, workspace_root=tmp_path)
    repeated = benchmark.build_support_hash_inventory(roles, workspace_root=tmp_path)
    assert initial == repeated
    assert [row["path"] for row in initial["files"]] == ["code/a.py", "config/b.json"]
    assert initial["entered_protocol_run_identity"] is False
    assert initial["replaces_pre_test_lock"] is False
    assert initial["test_metric_artifact_roles"] == []

    first.write_text("A = 2\n", encoding="utf-8")
    changed = benchmark.build_support_hash_inventory(roles, workspace_root=tmp_path)
    assert changed["root_sha256"] != initial["root_sha256"]


def test_declared_real_post_test_support_paths_exist() -> None:
    assert all(
        (benchmark.PROJECT_ROOT / relative).is_file()
        for relative in benchmark.POST_TEST_DIRECT_SOURCE_PATHS
    )
    assert all(
        (benchmark.PROJECT_ROOT / relative).is_dir()
        for relative in benchmark.POST_TEST_RECURSIVE_SOURCE_ROOTS
    )
    assert all(
        (benchmark.PROJECT_ROOT / relative).is_file()
        for relative in benchmark.POST_TEST_TOPOLOGY_ASSETS
    )


@pytest.mark.parametrize(
    ("parser", "base"),
    [
        (
            deployment.parse_args,
            [
                "--case-id",
                "case",
                "--images",
                "a.jpg",
                "b.jpg",
                "--final-candidate",
                "final.json",
                "--output",
                "out.json",
                "--device",
                "cuda:0",
            ],
        ),
        (
            benchmark.parse_args,
            [
                "--final-candidate",
                "final.json",
                "--preregistration",
                "prereg.json",
                "--staged-root",
                "rgb",
                "--output",
                "out.json",
                "--device",
                "cuda:0",
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "forbidden",
    ["--gt-root", "--evaluator-manifest", "--holdout-manifest", "--test-manifest"],
)
def test_cli_rejects_supervision_and_evaluation_arguments(
    parser: Any, base: list[str], forbidden: str
) -> None:
    with pytest.raises(SystemExit):
        parser([*base, forbidden, "forbidden"])
