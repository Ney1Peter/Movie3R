#!/usr/bin/env python3
"""Merge V54 LOSO reports and record the route decision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "output" / "v54_synthetic_explicit_shot_bridge"
OUTPUT = MODEL_ROOT / "v54_explicit_shot_bridge_report.json"
DOC = REPO_ROOT / "docs" / "movie3r" / "v54" / "V54_SYNTHETIC_EXPLICIT_SHOT_BRIDGE.md"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def metric(cases: list[dict], key: str) -> dict[str, float]:
    return summary([float(case[key]) for case in cases])


def main() -> None:
    by_source, all_cases = {}, []
    for source in SOURCES:
        path = MODEL_ROOT / "models_factorized_eval" / f"heldout_{source}" / "da3_se3_human" / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        cases = report["real_held_out"]["cases"]
        all_cases.extend(cases)
        by_source[source] = {
            "count": len(cases),
            "fixed": {
                "rotation_deg": metric(cases, "fixed_rotation_deg"),
                "translation_m": metric(cases, "fixed_translation_m"),
            },
            "learned_full_residual": {
                "rotation_deg": metric(cases, "after_fixed_rotation_deg"),
                "translation_m": metric(cases, "after_fixed_translation_m"),
            },
            "factorized_torso_rotation_learned_translation": {
                "rotation_deg": metric(cases, "torso_learned_translation_rotation_deg"),
                "translation_m": metric(cases, "torso_learned_translation_m"),
            },
            "factorized_translation_improvement_rate": float(
                np.mean(
                    [
                        case["torso_learned_translation_m"] < case["fixed_translation_m"]
                        for case in cases
                    ]
                )
            ),
        }

    v53 = json.loads(
        (REPO_ROOT / "output" / "v53_uniform_similarity_integrity" / "v53_uniform_similarity_integrity_probe.json").read_text(
            encoding="utf-8"
        )
    )
    mv200_ablation = {}
    for variant in ("raw_se3", "raw_sim3", "da3_se3", "da3_se3_human"):
        path = MODEL_ROOT / "models" / "heldout_mvhuman200" / variant / "report.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            mv200_ablation[variant] = report["real_held_out"]["overall"]

    merged = {
        "experiment": "V54 synthetic explicit geometry shot bridge probe",
        "protocol": {
            "training_geometry": "frozen Human3R pointmaps, confidence, masks, camera, and SMPL-X anchors",
            "synthetic_data": "known random SE(3)/Sim(3) perturbations on continuous-shot geometry",
            "real_mix": "training-source real cuts pre-aligned by Fixed plus known random residual perturbations",
            "generalization": "four-fold leave-one-source-out",
            "scale_variants": ["raw Human3R SE(3)", "raw Human3R Sim(3)", "DA3-normalized SE(3)"],
            "runtime_gt": False,
            "training_gt": "camera relative pose on training sources; synthetic perturbation is analytically known",
            "raw_tokens_used": False,
            "post_cut_frames": 1,
        },
        "mvhuman200_scale_ablation": mv200_ablation,
        "overall": {
            "fixed": {
                "rotation_deg": metric(all_cases, "fixed_rotation_deg"),
                "translation_m": metric(all_cases, "fixed_translation_m"),
            },
            "learned_full_residual": {
                "rotation_deg": metric(all_cases, "after_fixed_rotation_deg"),
                "translation_m": metric(all_cases, "after_fixed_translation_m"),
            },
            "factorized_torso_rotation_learned_translation": {
                "rotation_deg": metric(all_cases, "torso_learned_translation_rotation_deg"),
                "translation_m": metric(all_cases, "torso_learned_translation_m"),
            },
            "v53_reference": v53["overall"]["v47_uniform_scene"],
        },
        "by_source": by_source,
        "findings": [
            "Synthetic perturbation recovery generalizes to held-out sources, but direct transfer to real cuts does not.",
            "DA3 normalization is materially better than raw Human3R scale; learning Sim(3) does not replace metric normalization.",
            "SMPL-X anchors improve translation, but an unconstrained learned rotation is unsafe.",
            "Fixing V16 torso rotation and learning only explicit correspondences/translation improves three sources but catastrophically harms THuman.",
            "The learned bridge remains source-dependent and is worse than V53 uniform explicit similarity on aggregate and worst-source metrics.",
        ],
        "decision": {
            "full_learned_se3": "reject",
            "factorized_learned_correspondence_translation": "diagnostically promising but reject as deployable",
            "da3_metric_normalization": "necessary and should remain an explicit preprocessing/measurement component",
            "current_best": "V53 V47 rotation + uniform DA3 scene scale + explicit translation",
            "next_training_requirement": "more realistic synchronized multi-camera geometry with explicit overlap/correspondence supervision; simple rigid perturbations are insufficient",
        },
    }
    OUTPUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    factor = merged["overall"]["factorized_torso_rotation_learned_translation"]
    fixed = merged["overall"]["fixed"]
    v53_ref = merged["overall"]["v53_reference"]
    lines = [
        "# V54 Synthetic Explicit Shot Bridge",
        "",
        "## Design",
        "",
        "- Train on explicit Human3R pointmaps and SMPL-X anchors, never raw tokens.",
        "- Create known small/medium/large SE(3) or Sim(3) perturbations on continuous frames.",
        "- Compare raw scale, learned Sim(3), and DA3-normalized geometry.",
        "- Evaluate real cuts with four-fold leave-one-source-out.",
        "- Final factorized version keeps V16 torso rotation and learns point correspondence for explicit translation solving.",
        "",
        "## Main Result",
        "",
        f"- Fixed: {fixed['translation_m']['mean']:.3f} m, {fixed['rotation_deg']['mean']:.2f} deg.",
        f"- Learned factorized: {factor['translation_m']['mean']:.3f} m, {factor['rotation_deg']['mean']:.2f} deg.",
        f"- V53 reference: {v53_ref['camera_translation_m']['mean']:.3f} m, {v53_ref['camera_rotation_deg']['mean']:.2f} deg.",
        "- Factorized learning improves AvatarReX and both MVHuman sources, but THuman translation degrades from 0.483 m to 2.210 m.",
        "",
        "## Decision",
        "",
        "The HumanMM-style synthetic perturbation idea is useful for generating supervision, but simple transformed continuous-frame pointclouds do not reproduce real camera-cut reconstruction mismatch. Full learned SE(3) is rejected. The factorized learned correspondence branch has real signal but is not source-safe and is currently inferior to V53. Keep V53 as the main method; revisit training only with substantially more realistic synchronized multi-camera overlap/correspondence data.",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "doc": str(DOC), "decision": merged["decision"]}, indent=2))


if __name__ == "__main__":
    main()
