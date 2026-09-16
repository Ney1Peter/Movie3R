#!/usr/bin/env python3
"""Verify the completed protocol and build its paper-facing return package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("r0", "r1", "r2", "r3")
ARRAY_KEYS = (
    "cameras_c2w", "vertices_world", "joints_world",
    "persistent_ids", "native_ids", "valid",
)
EXPECTED = {"egobody": 129, "egohumans": 90}
EXPECTED_HOLDOUT = {"egobody": 48, "egohumans": 24}
EXPECTED_QUALITATIVE = {"egobody": 6, "egohumans": 8}
R0_REFERENCES = {
    "egobody": {"w_mpjpe_mm": 752.6069711928769, "wa_mpjpe_mm": 589.4449092128459, "ate_m": 1.1407098598246705},
    "egohumans": {"w_mpjpe_mm": 1248.1, "wa_mpjpe_mm": 491.3},
}

REQUIRED_RETURN_FILES = (
    "protocol_lock.json",
    "environment.md",
    "commands.log",
    "config/human_registration.json",
    "config/scene_registration.json",
    "config/frozen_egobody.json",
    "config/frozen_egohumans.json",
    "config/holdout_gate_egobody.json",
    "config/holdout_gate_egohumans.json",
    "config/qualitative_cases_pre_test.json",
    "provenance/git_status.txt",
    "provenance/input_sha256.txt",
    "provenance/egobody_test_prediction_seal.json",
    "provenance/egohumans_test_prediction_seal.json",
    "metrics/aggregate.json",
    "metrics/case_metrics.csv",
    "metrics/paired_bootstrap.json",
    "metrics/angle_strata.json",
    "metrics/failures.csv",
    "tables/registration_main.md",
    "tables/registration_main.csv",
    "tables/registration_main.tex",
    "tables/registration_angle_strata.md",
    "tables/registration_paired.md",
    "tables/registration_secondary.md",
    "tables/registration_failures.md",
    "figures/registration_viewpoint_curve.pdf",
    "figures/registration_failure_breakdown.pdf",
    "figures/qualitative/QUALITATIVE_INDEX.md",
)

TRANSFORM_KEYS = (
    "schema_version", "dataset", "case_id", "method_id",
    "runtime_manifest_sha256", "checkpoint_sha256", "boundary_index",
    "temporal_access", "gt_used_for_registration", "transform_kind",
    "transform_B_to_A", "scale", "status", "failure_reason",
    "person_count_pre", "person_count_post", "matched_person_count",
    "scene_point_count_pre", "scene_point_count_post", "correspondence_count",
    "inlier_ratio", "fit_residual", "runtime_seconds", "random_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def metric(aggregate: dict[str, Any], dataset: str, method: str, name: str) -> float | None:
    return aggregate["datasets"][dataset][method]["metrics"][name]["mean"]


def validate_completion(root: Path, aggregate: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless every protocol-mandated artifact and gate exists."""
    missing = [relative for relative in REQUIRED_RETURN_FILES if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"missing mandatory return files: {missing}")

    if aggregate.get("bootstrap_samples") != 100_000 or aggregate.get("bootstrap_seed") != 20260915:
        raise ValueError("final aggregate must use 100,000 bootstrap samples and seed 20260915")

    gates: dict[str, Any] = {}
    dataset_files: dict[str, Any] = {}
    for dataset, expected in EXPECTED.items():
        gate_path = root / "config" / f"holdout_gate_{dataset}.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("passed") is not True
            or gate.get("expected_case_count") != EXPECTED_HOLDOUT[dataset]
            or gate.get("evaluated_case_count") != EXPECTED_HOLDOUT[dataset]
        ):
            raise ValueError(f"Holdout gate did not pass completely: {gate_path}")
        gates[dataset] = {
            "passed": True,
            "case_count": gate["evaluated_case_count"],
            "parameter_changes_after_gate_allowed": gate.get("parameter_changes_after_gate_allowed"),
        }

        prediction_dir = root / "predictions" / dataset / "test"
        transform_dir = root / "transforms" / dataset / "test"
        predictions = sorted(prediction_dir.glob("*.npz"))
        runtimes = sorted(prediction_dir.glob("*.runtime.json"))
        scene_caches = sorted(prediction_dir.glob("*.r0_scene.npz"))
        combined = [path for path in predictions if not path.name.endswith(".r0_scene.npz")]
        case_dirs = sorted(path for path in transform_dir.iterdir() if path.is_dir())
        if not (len(combined) == len(runtimes) == len(scene_caches) == len(case_dirs) == expected):
            raise ValueError(
                f"{dataset} Test artifact counts disagree: combined={len(combined)}, "
                f"runtime={len(runtimes)}, scene={len(scene_caches)}, transforms={len(case_dirs)}, "
                f"expected={expected}"
            )
        for case_dir in case_dirs:
            names = {path.name for path in case_dir.glob("*.json")}
            if names != {f"{method}.json" for method in METHODS}:
                raise ValueError(f"incomplete transform set: {case_dir}: {sorted(names)}")
            for method in METHODS:
                transform_path = case_dir / f"{method}.json"
                transform = json.loads(transform_path.read_text(encoding="utf-8"))
                absent = [key for key in TRANSFORM_KEYS if key not in transform]
                if absent:
                    raise ValueError(f"missing transform fields {absent}: {transform_path}")
                if transform.get("gt_used_for_registration") is not False:
                    raise ValueError(f"GT leakage declaration is not false: {transform_path}")
        dataset_files[dataset] = {
            "test_case_count": expected,
            "combined_prediction_count": len(combined),
            "runtime_record_count": len(runtimes),
            "scene_cache_count": len(scene_caches),
            "transform_json_count": 4 * len(case_dirs),
        }

    lock = json.loads((root / "config/qualitative_cases_pre_test.json").read_text(encoding="utf-8"))
    if lock.get("locked_before_registration_test_metrics") is not True:
        raise ValueError("qualitative case list is not declared pre-Test locked")
    locked_count = sum(len(lock["datasets"].get(dataset, [])) for dataset in EXPECTED)
    expected_locked_count = sum(EXPECTED_QUALITATIVE.values())
    for dataset, expected in EXPECTED_QUALITATIVE.items():
        if len(lock["datasets"].get(dataset, [])) != expected:
            raise ValueError(f"qualitative lock count mismatch for {dataset}")
    qualitative_dir = root / "figures/qualitative"
    pdf_count = len(list(qualitative_dir.glob("*.pdf")))
    png_count = len(list(qualitative_dir.glob("*.png")))
    if locked_count != expected_locked_count or pdf_count != expected_locked_count or png_count != expected_locked_count:
        raise ValueError(
            f"qualitative artifact count mismatch: locked={locked_count}, pdf={pdf_count}, png={png_count}"
        )

    return {
        "required_return_file_count": len(REQUIRED_RETURN_FILES),
        "holdout_gates": gates,
        "test_artifact_counts": dataset_files,
        "qualitative_case_count": locked_count,
        "qualitative_pdf_count": pdf_count,
        "qualitative_png_count": png_count,
        "bootstrap_samples": aggregate["bootstrap_samples"],
        "bootstrap_seed": aggregate["bootstrap_seed"],
    }


def validate_test(root: Path, dataset: str) -> dict[str, Any]:
    seal_path = root / "provenance" / f"{dataset}_test_prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("case_count") != EXPECTED[dataset] or seal.get("sealed_before_evaluator") is not True:
        raise ValueError(f"invalid {dataset} prediction seal")
    seal_hash = sha256(seal_path)
    evaluations = sorted((root / "metrics/evaluations" / dataset / "test").glob("*.evaluation.json"))
    if len(evaluations) != EXPECTED[dataset]:
        raise ValueError(f"{dataset} evaluator count mismatch")
    fallback_checks, pre_checks, local_checks = 0, 0, 0
    statuses = {method: {} for method in METHODS}
    for evaluation in evaluations:
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        if report.get("errors"):
            raise ValueError(f"evaluation errors: {evaluation}")
        if report["inputs"].get("prediction_seal_sha256") != seal_hash:
            raise ValueError(f"evaluation not bound to exact seal: {evaluation}")
        if report["inputs"].get("evaluator_manifest_opened_after_prediction_seal") is not True:
            raise ValueError(f"evaluation does not declare post-seal GT access: {evaluation}")
        prediction = Path(report["inputs"]["prediction"])
        runtime = json.loads(Path(report["inputs"]["runtime_report"]).read_text(encoding="utf-8"))
        boundary = int(runtime["record"]["boundary_index"])
        with np.load(prediction, allow_pickle=False) as cache:
            for method in METHODS:
                transform = report["registration"][method]
                status = str(transform["status"])
                statuses[method][status] = statuses[method].get(status, 0) + 1
                for key in ARRAY_KEYS:
                    base = np.asarray(cache[f"r0__{key}"])
                    value = np.asarray(cache[f"{method}__{key}"])
                    if not np.array_equal(base[:boundary], value[:boundary], equal_nan=True):
                        raise ValueError(f"pre-shot modified: {dataset}/{report['case_id']}/{method}/{key}")
                    pre_checks += 1
                    if status != "ok" and not np.array_equal(base, value, equal_nan=True):
                        raise ValueError(f"failed fit is not exact R0 fallback: {dataset}/{report['case_id']}/{method}/{key}")
                        
                if status != "ok":
                    fallback_checks += 1
                pre_scene = np.asarray(cache[f"{method}__scene_pre_points"])
                r0_pre_scene = np.asarray(cache["r0__scene_pre_points"])
                if not np.array_equal(pre_scene, r0_pre_scene, equal_nan=True):
                    raise ValueError(f"pre-shot scene modified: {dataset}/{report['case_id']}/{method}")
                if status != "ok" and not np.array_equal(
                    np.asarray(cache[f"{method}__scene_post_points"]),
                    np.asarray(cache["r0__scene_post_points"]), equal_nan=True,
                ):
                    raise ValueError(f"failed fit scene is not exact R0 fallback: {dataset}/{report['case_id']}/{method}")
                source_scene = np.asarray(cache["r0__scene_post_points"])
                mapped_scene = np.asarray(cache[f"{method}__scene_post_points"])
                matrix = np.asarray(transform["transform_B_to_A"], dtype=np.float64)
                sample = np.unique(np.linspace(0, max(len(source_scene) - 1, 0), min(128, len(source_scene)), dtype=np.int64))
                expected_scene = source_scene[sample] @ matrix[:3, :3].T + matrix[:3, 3]
                if not np.allclose(mapped_scene[sample], expected_scene, rtol=1e-6, atol=1e-6):
                    raise ValueError(f"scene does not share transform: {dataset}/{report['case_id']}/{method}")
            # One shared rigid transform of both camera and human must leave
            # the camera-local pose/shape metrics unchanged up to evaluator noise.
            r0 = report["methods"]["r0"]["multi_thumbs_named_provisional"]
            for method in ("r1", "r2", "r3"):
                current = report["methods"][method]["multi_thumbs_named_provisional"]
                for key in ("mpjpe_mm", "mpvpe_mm"):
                    a, b = r0[key]["mean"], current[key]["mean"]
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        raise ValueError(f"local {key} support changed: {dataset}/{report['case_id']}/{method}")
                    if not math.isclose(float(a), float(b), rel_tol=2e-5, abs_tol=2e-3):
                        raise ValueError(f"shared rigid transform changed local {key}: {dataset}/{report['case_id']}/{method}")
                    local_checks += 1
    return {
        "dataset": dataset, "case_count": len(evaluations), "prediction_seal": str(seal_path),
        "prediction_seal_sha256": seal_hash, "pre_shot_array_checks": pre_checks,
        "exact_failure_fallback_cases": fallback_checks, "local_metric_invariance_checks": local_checks,
        "status_counts": statuses,
    }


def result_summary(root: Path, aggregate: dict[str, Any]) -> str:
    labels = {
        "r1": "pelvis translation",
        "r2": "human-joint robust SE(3)",
        "r3": "scene registration + ICP",
    }

    def value(dataset: str, method: str, name: str) -> float:
        result = metric(aggregate, dataset, method, name)
        if result is None:
            raise ValueError(f"missing summary metric {dataset}/{method}/{name}")
        return float(result)

    def paired(dataset: str, method: str, name: str, comparison: str, digits: int) -> str:
        item = aggregate["paired"][dataset][method][f"{name}_vs_{comparison}"]
        return (
            f"{item['mean']:.{digits}f} "
            f"(95% CI {item['ci95'][0]:.{digits}f} to {item['ci95'][1]:.{digits}f}; "
            f"n={item['common_case_count']}, units={item['unit_count']})"
        )

    lines = [
        "# Shot3R traditional-registration experiment: paper summary", "",
        "## Frozen protocol at a glance", "",
        "R0 reconstructs both shots independently with the public Human3R checkpoint. R1 estimates one shared pelvis translation, R2 one prediction-only robust human-joint SE(3), and R3 an offline FPFH/RANSAC + point-to-plane ICP transform from both complete shot point clouds. Every failed fit is an exact R0 fallback and remains in the fixed denominator. Traditional methods receive annotated boundaries; Shot3R uses its streaming boundary detector.", "",
        "| Dataset | Cases / units | Method | W (mm) | WA (mm) | ATE (m) | IDF1 | Coverage | Registration success |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, units in (("egobody", "129 / 43 recordings"), ("egohumans", "90 / 27 captures")):
        for method in ("r0", "r1", "r2", "r3", "shot3r"):
            summary = aggregate["datasets"][dataset][method]
            success = summary["registration_success_rate"]
            lines.append(
                f"| {dataset} | {units} | {('Shot3R' if method == 'shot3r' else ('Human3R per-shot reset' if method == 'r0' else labels[method]))} | "
                f"{value(dataset,method,'w_mpjpe_mm'):.1f} | {value(dataset,method,'wa_mpjpe_mm'):.1f} | "
                f"{value(dataset,method,'ate_m'):.3f} | {value(dataset,method,'idf1'):.3f} | "
                f"{value(dataset,method,'coverage'):.3f} | "
                f"{'N/A' if success is None else f'{100*success:.1f}%'} |"
            )

    lines.extend(("", "## Answers to the ten protocol questions", ""))
    for dataset in EXPECTED:
        config = json.loads((root / "config" / f"frozen_{dataset}.json").read_text(encoding="utf-8"))
        lines.append(
            f"1. **Frozen Development settings ({dataset}).** R1 `{config['selected_candidate_ids']['r1']}`, "
            f"R2 `{config['selected_candidate_ids']['r2']}`, and R3 `{config['selected_candidate_ids']['r3']}` were selected once by the declared complete-Development aggregate objective. The settings are dataset-level, passed the one-shot Holdout gate, and were never selected case by case."
        )
    lines.extend((
        "",
        "**Aggregation note.** EgoBody headline values are recording-macro over 43 recordings. To reproduce the frozen paper table exactly, EgoHumans headline values retain its established case-macro convention; capture-macro values are stored alongside every metric in `aggregate.json`, and all EgoHumans paired intervals resample the 27 captures. Thus no confidence interval treats the 90 cases as independent.",
        "",
        f"2. **R0 reproduction.** EgoBody reproduces W/WA/ATE-Sim3 = {value('egobody','r0','w_mpjpe_mm'):.1f}/{value('egobody','r0','wa_mpjpe_mm'):.1f} mm/{value('egobody','r0','ate_m'):.3f} m; the preregistered references are 752.6/589.4 mm/1.141 m. EgoHumans reproduces W/WA = {value('egohumans','r0','w_mpjpe_mm'):.1f}/{value('egohumans','r0','wa_mpjpe_mm'):.1f} mm versus 1248.1/491.3 mm. All numerical gates pass within the frozen tolerance.",
        "",
        f"3. **What pelvis translation fixes.** On EgoBody, R1 reduces W from {value('egobody','r0','w_mpjpe_mm'):.1f} to {value('egobody','r1','w_mpjpe_mm'):.1f} mm and ATE from {value('egobody','r0','ate_m'):.3f} to {value('egobody','r1','ate_m'):.3f} m, but boundary rotation error stays {value('egobody','r1','boundary_camera_rotation_deg'):.1f} degrees. On EgoHumans, W changes only from {value('egohumans','r0','w_mpjpe_mm'):.1f} to {value('egohumans','r1','w_mpjpe_mm'):.1f} mm, and its paired W gain is not significant: {paired('egohumans','r1','w_mpjpe_mm','r0',1)}. Translation cannot resolve viewpoint rotation or an incorrect identity match.",
        "",
        f"4. **Does robust human SE(3) improve on translation?** Yes in the fixed-denominator aggregate. Relative to R1, R2 changes W/WA/ATE from {value('egobody','r1','w_mpjpe_mm'):.1f}/{value('egobody','r1','wa_mpjpe_mm'):.1f} mm/{value('egobody','r1','ate_m'):.3f} m to {value('egobody','r2','w_mpjpe_mm'):.1f}/{value('egobody','r2','wa_mpjpe_mm'):.1f} mm/{value('egobody','r2','ate_m'):.3f} m on EgoBody and reduces boundary rotation from {value('egobody','r1','boundary_camera_rotation_deg'):.1f} to {value('egobody','r2','boundary_camera_rotation_deg'):.1f} degrees. On EgoHumans it changes W/WA/ATE from {value('egohumans','r1','w_mpjpe_mm'):.1f}/{value('egohumans','r1','wa_mpjpe_mm'):.1f} mm/{value('egohumans','r1','ate_m'):.3f} m to {value('egohumans','r2','w_mpjpe_mm'):.1f}/{value('egohumans','r2','wa_mpjpe_mm'):.1f} mm/{value('egohumans','r2','ate_m'):.3f} m and reduces boundary rotation from {value('egohumans','r1','boundary_camera_rotation_deg'):.1f} to {value('egohumans','r2','boundary_camera_rotation_deg'):.1f} degrees. Its W gain over R0 is {paired('egobody','r2','w_mpjpe_mm','r0',1)} and {paired('egohumans','r2','w_mpjpe_mm','r0',1)}, respectively. The limitation is robustness: prediction-only RANSAC fails on 16/90 EgoHumans cases, which remain exact R0 fallbacks, and R2 retains lower IDF1 than Shot3R on both datasets.",
        "",
        f"5. **How scene registration fails.** R3 reports 129/129 algorithmic successes on EgoBody but W remains {value('egobody','r3','w_mpjpe_mm'):.1f} mm even though ATE falls to {value('egobody','r3','ate_m'):.3f} m; a numerically accepted camera alignment therefore does not imply correct human placement. On EgoHumans it succeeds on only 37/90 cases (41.1%), with 17 global-RANSAC failures and 36 ICP-gate failures; W is {value('egohumans','r3','w_mpjpe_mm'):.1f} mm, worse than R0. Its prediction-only scene correspondence inlier ratio has median 0.055, consistent with weak overlap, and the fixed extreme/at-least-150-degree stratum has 59.1% algorithmic failure.",
        "",
        f"6. **Do traditional methods improve R0 significantly?** R2 does on W for both datasets because its 95% CIs exclude zero: {paired('egobody','r2','w_mpjpe_mm','r0',1)} and {paired('egohumans','r2','w_mpjpe_mm','r0',1)}. R1 is significant on EgoBody but not EgoHumans W. R3 has no significant EgoBody W gain and significantly degrades EgoHumans W (see `registration_paired.md`).",
        "",
        f"7. **Shot3R versus the strongest traditional method.** R2 is the lowest-W traditional method on both datasets. Shot3R versus R2 is {value('egobody','shot3r','w_mpjpe_mm'):.1f} vs {value('egobody','r2','w_mpjpe_mm'):.1f} mm W, {value('egobody','shot3r','wa_mpjpe_mm'):.1f} vs {value('egobody','r2','wa_mpjpe_mm'):.1f} mm WA, and {value('egobody','shot3r','ate_m'):.3f} vs {value('egobody','r2','ate_m'):.3f} m ATE on EgoBody. The R2-minus-Shot3R paired differences are W {paired('egobody','r2','w_mpjpe_mm','shot3r',1)}, WA {paired('egobody','r2','wa_mpjpe_mm','shot3r',1)}, and ATE {paired('egobody','r2','ate_m','shot3r',3)}. On EgoHumans the corresponding headline values are W {value('egohumans','shot3r','w_mpjpe_mm'):.1f} vs {value('egohumans','r2','w_mpjpe_mm'):.1f} mm, WA {value('egohumans','shot3r','wa_mpjpe_mm'):.1f} vs {value('egohumans','r2','wa_mpjpe_mm'):.1f} mm, and ATE {value('egohumans','shot3r','ate_m'):.3f} vs {value('egohumans','r2','ate_m'):.3f} m. The paired differences are W {paired('egohumans','r2','w_mpjpe_mm','shot3r',1)}, WA {paired('egohumans','r2','wa_mpjpe_mm','shot3r',1)}, and ATE {paired('egohumans','r2','ate_m','shot3r',3)}.",
        "",
        "8. **Viewpoint dependence.** The advantage is not monotonic in W. R2 is lower than Shot3R on EgoBody small W (194.1 vs 201.4 mm) and extreme W (341.2 vs 354.3 mm), while Shot3R is lower on medium W and on extreme WA/ATE. On EgoHumans, R2 is lower on large W (928.2 vs 930.7 mm) and extreme W (803.5 vs 886.1 mm), whereas Shot3R is lower on medium W and on ATE in every listed stratum. These are the pre-defined strata; no result-dependent angle subset was introduced.",
        "",
        "9. **Does this prove Shot3R is not simple post-hoc registration?** It supports a qualified, not absolute, statement. Pelvis translation and scene ICP are not interchangeable substitutes: they leave large human or rotation errors, and scene ICP is offline and unstable on EgoHumans. Robust human SE(3) is much stronger and sometimes matches or exceeds Shot3R in stratum-level W, but it receives the annotated boundary, fails on some crowded cases, does not recover Shot3R's IDF1, and uses a separate post-hoc geometric fit. The appropriate claim is that Shot3R jointly provides streaming cross-shot state handling and association rather than merely applying scene ICP or a shared translation to independent reconstructions.",
        "",
        f"10. **Results that do not support the strongest initial expectation.** Shot3R's overall W advantage over R2 is small and statistically inconclusive on both datasets: {paired('egobody','r2','w_mpjpe_mm','shot3r',1)} and {paired('egohumans','r2','w_mpjpe_mm','shot3r',1)}. R2 is better in several fixed W strata, and on EgoBody R3 attains ATE {value('egobody','r3','ate_m'):.3f} m, close to Shot3R's {value('egobody','shot3r','ate_m'):.3f} m. These outcomes must remain visible; the experiment falls under protocol Decision Rule B, not Rule A.",
        "",
        "## Recommended paper placement", "",
        "Use the compact main table or one short paragraph in the main paper, emphasizing access and joint W/WA/ATE/IDF1 rather than claiming universal W dominance. Put the complete pre-defined angle strata, common-support 100,000-sample bootstrap, fitting diagnostics, valid support, failure taxonomy, and locked qualitative cases in the supplement. State explicitly that R3 is offline post-hoc and that every traditional baseline receives annotated cut boundaries, which favors the baselines.",
        "",
    ))
    return "\n".join(lines)


def readme(root: Path) -> str:
    return """# Shot3R traditional-registration baselines

This directory is the frozen return package for the protocol in
`Shot3R_传统配准基线实验执行协议_20260915.md`.

R0 reconstructs each shot independently with the public Human3R checkpoint.
R1 adds prediction-only pelvis translation, R2 adds prediction-only robust
human-joint SE(3), and R3 performs offline FPFH/RANSAC plus point-to-plane ICP
on predicted background point clouds. Every failed fit remains in the fixed
denominator and uses an exact R0 fallback. The same transform is applied to
the post-cut cameras, joints, meshes, and cached scene geometry.

Parameters were selected once per dataset on complete Development splits,
then passed through a one-shot Holdout execution gate. Frozen Test prediction
and transform files were sealed before the independent GT evaluator was
opened. `provenance/verification.json` audits this order and verifies exact
failure fallback, unchanged pre-cut arrays, and local-pose invariance.

Paper-facing outputs are in `tables/`, `figures/`, `metrics/aggregate.json`,
and `RESULT_SUMMARY_FOR_PAPER.md`. Case-level evidence and failure diagnostics
remain available in `metrics/case_metrics.csv` and `metrics/failures.csv`.

Important aggregation note: EgoBody headline values are recording macro over
43 recordings. EgoHumans keeps the existing paper's case-macro headline so
the frozen Shot3R values remain exact; capture-macro values are retained in
the aggregate JSON and all paired confidence intervals resample captures.
All paired intervals use 100,000 bootstrap samples with seed 20260915.
"""


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    aggregate_path = root / "metrics/aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    completion = validate_completion(root, aggregate)
    checks = [validate_test(root, dataset) for dataset in EXPECTED]
    r0_gate = {}
    for dataset, expected in R0_REFERENCES.items():
        observed = {key: metric(aggregate, dataset, "r0", key) for key in expected}
        tolerances = {"w_mpjpe_mm": 2.0, "wa_mpjpe_mm": 2.0, "ate_m": 0.02}
        passed = {
            key: abs(observed[key] - target) <= tolerances[key]
            for key, target in expected.items()
        }
        r0_gate[dataset] = {"expected": expected, "observed": observed, "tolerances": tolerances, "passed": passed, "all_passed": all(passed.values())}
        if not r0_gate[dataset]["all_passed"]:
            raise ValueError(f"R0 reproduction gate failed for {dataset}: {r0_gate[dataset]}")
    verification = {
        "schema_version": "Shot3R-traditional-registration-final-verification-v1",
        "completion_checks": completion,
        "test_checks": checks, "r0_reproduction_gate": r0_gate,
        "fixed_denominators": EXPECTED, "test_gt_used_for_registration": False,
        "per_case_oracle": False, "decision_rule": "B",
    }
    atomic_json(root / "provenance/verification.json", verification)
    atomic_text(root / "RESULT_SUMMARY_FOR_PAPER.md", result_summary(root, aggregate))
    atomic_text(root / "README.md", readme(root))
    # Hash the returnable evidence once, after all generated summaries exist.
    excluded = {"RETURN_MANIFEST.sha256", "provenance/output_sha256.txt"}
    files = [
        path for path in root.rglob("*") if path.is_file()
        and "work" not in path.relative_to(root).parts
        and str(path.relative_to(root)) not in excluded
        and not path.name.endswith(".partial")
        and not path.name.endswith(".zip")
    ]
    lines = []
    for index, path in enumerate(sorted(files), start=1):
        lines.append(f"{sha256(path)}  {path.relative_to(root)}\n")
        if index % 100 == 0:
            print(f"hashed {index}/{len(files)} return files", flush=True)
    manifest = "".join(lines)
    atomic_text(root / "RETURN_MANIFEST.sha256", manifest)
    atomic_text(root / "provenance/output_sha256.txt", manifest)
    print(json.dumps({"verification": str(root / 'provenance/verification.json'), "return_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
