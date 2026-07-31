#!/usr/bin/env python3
"""Audit cached evidence for a deployable BRTC person-to-scene residual.

This script never loads a model, image, GT geometry, or GPU.  It inspects only
the schemas and non-GT predicted geometry already present in the frozen
MultiHuman/Ego caches, plus previously written diagnostic reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "versions/v14",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402


DEFAULT_EGO_CACHE = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/person_scene_contact_availability_audit"
)
DEFAULT_DOC = (
    REPO_ROOT
    / "versions/v14/docs/V14_BRTC_PERSON_SCENE_CONTACT_AVAILABILITY_20260801.md"
)
V46_REPORT = (
    REPO_ROOT
    / "output/v46_contact_preserving_metric_bridge/"
    "v46_contact_preserving_metric_bridge_probe.json"
)
RESIDUAL_REPORT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/residual_observability_three_dev/"
    "v14_b0_residual_observability.json"
)
ORIGINAL_DEMO = REPO_ROOT / "output/v14/brtc_lc_original_demo"
VIRTUAL_DEPTH = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/cut3r_virtual_person_depth"
)
FOOT_JOINTS = np.asarray((7, 8, 10, 11, 60, 61, 62, 63, 64, 65))
DISTANCE_THRESHOLDS_M = (0.10, 0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego_cache", type=Path, default=DEFAULT_EGO_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def count_stats(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()) if len(array) else 0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "mean": float(array.mean()) if len(array) else 0.0,
        "max": int(array.max()) if len(array) else 0,
    }


def split_rows() -> dict[str, tuple[str, list[dict[str, Any]]]]:
    return {
        "three_dev_offset0": ("three", legacy.report_rows(("three",))),
        "three_heldout_offset1": (
            "three",
            json.loads(harness.DEFAULT_CONFIRM_REPORT.read_text(encoding="utf-8"))[
                "cases"
            ],
        ),
        "dance": ("dance", legacy.report_rows(("dance",))),
        "box": ("box", legacy.report_rows(("box",))),
    }


def multihuman_availability() -> dict[str, Any]:
    result = {}
    for split, (sequence, rows) in split_rows().items():
        pre_counts: list[int] = []
        post_counts: list[int] = []
        root_keys: set[str] = set()
        person_keys: set[str] = set()
        post_nonempty = 0
        people = 0
        person_with_near_foot = {threshold: 0 for threshold in DISTANCE_THRESHOLDS_M}
        feet_with_cloud = 0
        near_feet = {threshold: 0 for threshold in DISTANCE_THRESHOLDS_M}
        missing = []
        for row in rows:
            key = str(row["case"]["key"])
            cache_path = SEQUENCE_INPUTS[sequence]["cache"] / f"{key}.pt"
            if not cache_path.is_file():
                missing.append(key)
                continue
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Do not inspect cache['gt']; it is evaluator-only and irrelevant to
            # runtime evidence availability.
            root_keys.update(cache.keys())
            pre_cloud = np.asarray(cache["clouds"][-2], dtype=np.float64)
            post_cloud = np.asarray(cache["clouds"][-1], dtype=np.float64)
            pre_counts.append(len(pre_cloud))
            post_counts.append(len(post_cloud))
            post_nonempty += int(len(post_cloud) > 0)
            tree = cKDTree(post_cloud) if len(post_cloud) else None
            for person in cache["humans"][-1].values():
                people += 1
                person_keys.update(person.keys())
                if tree is None:
                    continue
                joints = np.asarray(person["joints"], dtype=np.float64)
                ids = FOOT_JOINTS[FOOT_JOINTS < len(joints)]
                distances, _ = tree.query(joints[ids], k=1)
                feet_with_cloud += len(distances)
                for threshold in DISTANCE_THRESHOLDS_M:
                    count = int(np.sum(distances <= threshold))
                    near_feet[threshold] += count
                    person_with_near_foot[threshold] += int(count > 0)
        loaded_cases = len(pre_counts)
        result[split] = {
            "sequence": sequence,
            "requested_case_count": len(rows),
            "loaded_case_count": loaded_cases,
            "missing_cache_count": len(missing),
            "person_count": people,
            "cache_root_keys": sorted(root_keys),
            "cached_person_keys": sorted(person_keys),
            "pre_sparse_cloud_point_count": count_stats(pre_counts),
            "post_sparse_cloud_point_count": count_stats(post_counts),
            "post_cloud_nonempty_case_count": post_nonempty,
            "post_cloud_nonempty_case_rate": float(
                post_nonempty / max(loaded_cases, 1)
            ),
            "feet_with_nonempty_post_cloud_count": feet_with_cloud,
            "foot_nearest_sparse_cloud_fraction": {
                str(threshold): float(near_feet[threshold] / max(feet_with_cloud, 1))
                for threshold in DISTANCE_THRESHOLDS_M
            },
            "people_with_any_foot_near_sparse_cloud_count": {
                str(threshold): int(person_with_near_foot[threshold])
                for threshold in DISTANCE_THRESHOLDS_M
            },
            "confidence_preserved": False,
            "pixel_coordinates_preserved": False,
            "foot_visibility_preserved": False,
            "human_mask_preserved": False,
            "dense_or_foot_local_scene_patch_preserved": False,
        }
    return result


def ego_availability(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    frame_keys: set[str] = set()
    person_keys: set[str] = set()
    frame_count = 0
    person_frame_count = 0
    for chain in cache["chains"]:
        for segment in chain["segments"]:
            for frame in segment["frames"]:
                frame_count += 1
                frame_keys.update(frame.keys())
                for person in frame["people"]:
                    person_frame_count += 1
                    person_keys.update(person.keys())
    possible_scene_fields = {
        "pointmap",
        "points",
        "pts3d_in_self_view",
        "pts3d_in_other_view",
        "depth",
        "confidence",
        "conf_self",
        "conf",
        "cloud",
        "clouds",
        "scene",
        "mask",
        "msk",
        "foot_uv",
        "foot_visibility",
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "chain_count": len(cache["chains"]),
        "frame_count": frame_count,
        "person_frame_count": person_frame_count,
        "frame_keys": sorted(frame_keys),
        "person_keys": sorted(person_keys),
        "scene_or_visibility_fields_present": sorted(
            possible_scene_fields & (frame_keys | person_keys)
        ),
        "has_runtime_scene_evidence": False,
        "has_foot_visibility": False,
        "can_evaluate_nonzero_contact_candidate": False,
        "can_evaluate_contact_candidate_acceleration": False,
    }


def saved_artifact_availability() -> dict[str, Any]:
    demo_cases = sorted(path for path in ORIGINAL_DEMO.iterdir() if path.is_dir())
    demos = []
    for case in demo_cases:
        methods = []
        for method in sorted(path for path in case.iterdir() if path.is_dir()):
            counts = {
                subdir: len(list((method / subdir).glob("*")))
                if (method / subdir).is_dir()
                else 0
                for subdir in ("camera", "color", "conf", "depth", "smpl")
            }
            methods.append({"name": method.name, "file_counts": counts})
        demos.append({"case": case.name, "methods": methods})
    virtual_cases = sorted(
        path
        for path in VIRTUAL_DEPTH.iterdir()
        if path.is_dir() and (path / "virtual_pts3d_self.npy").is_file()
    )
    virtual = []
    for case in virtual_cases:
        arrays = {}
        for path in sorted(case.glob("*.npy")):
            value = np.load(path, mmap_mode="r")
            arrays[path.name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        virtual.append({"case": case.name, "arrays": arrays})
    return {
        "original_demo": {
            "case_count": len(demo_cases),
            "cases": demos,
            "scope": "one selected MultiHuman demo, not the split caches or EgoHumans",
        },
        "virtual_person_depth": {
            "case_count": len(virtual_cases),
            "cases": virtual,
            "scope": "one virtual ray-query case, not ordinary B0 per-frame scene evidence",
        },
    }


def prior_failure_evidence() -> dict[str, Any]:
    v46 = json.loads(V46_REPORT.read_text(encoding="utf-8"))
    contact = v46["overall"]["contact_v32"]
    residual = json.loads(RESIDUAL_REPORT.read_text(encoding="utf-8"))
    methods = residual["summary"]["methods"]
    return {
        "v11_2_contact_preserving": {
            "source": str(V46_REPORT),
            "case_count": int(v46["case_count"]),
            "mean_root_contact_correction_m": float(
                contact["contact_correction_m"]["mean"]
            ),
            "mean_human_reprojection_shift_px": float(
                contact["human_reprojection_shift_px"]["mean"]
            ),
            "p95_human_reprojection_shift_px": float(
                contact["human_reprojection_shift_px"]["p95"]
            ),
            "combined_catastrophic_rate": float(
                contact["combined_catastrophic_rate"]
            ),
            "lesson": (
                "forcing a contact proxy to zero by post-hoc root translation can "
                "destroy Human3R image-space consistency"
            ),
        },
        "v14_scene_residual_observability_three_dev": {
            "source": str(RESIDUAL_REPORT),
            "case_count": int(residual["summary"]["case_count"]),
            "b0_camera_composite_mean": float(
                methods["b0"]["camera_composite"]["mean"]
            ),
            "scene_icp_camera_composite_mean": float(
                methods["scene_icp_full"]["camera_composite"]["mean"]
            ),
            "scene_mutual_camera_composite_mean": float(
                methods["scene_mutual_translation_b025"]["camera_composite"]["mean"]
            ),
            "scene_icp_human_root_mean": float(
                methods["scene_icp_full"]["human_root_error_m"]["mean"]
            ),
            "scene_mutual_human_root_mean": float(
                methods["scene_mutual_translation_b025"]["human_root_error_m"]["mean"]
            ),
            "lesson": (
                "same-forward sparse pointmap scene residuals were not a reliable "
                "independent correction cue"
            ),
        },
    }


def minimal_extension() -> dict[str, Any]:
    return {
        "model_rerun": (
            "required once on CPU or a separately authorized device because the current "
            "cache builder discarded the needed forward outputs"
        ),
        "new_pretrained_model": False,
        "gt_runtime_use": False,
        "retain_from_existing_human3r_forward": [
            "pts3d_in_self_view (camera-local, not pre-baked world points)",
            "conf_self",
            "human mask msk when present",
            "projected SMPL-X foot UV/depth and in-frame visibility",
            "raw predicted camera c2w and intrinsics needed to transform local patches",
        ],
        "recommended_compact_representation": {
            "per_person_per_frame": (
                "two 33x33 foot-centred pointmap/confidence patches plus UV, validity, "
                "human-exclusion mask, and foot camera depth"
            ),
            "coordinate_system": (
                "camera-local points; apply the frozen B0/method camera only at runtime"
            ),
            "why_not_existing_sparse_cloud": (
                "it removes the complete person bbox with an 8% margin and drops the "
                "confidence/pixel correspondence required to identify local support"
            ),
        },
        "foot_local_mask_construction": {
            "patch": "33x33 pixels centred at each projected foot anchor",
            "foot_anchors": (
                "SMPL-X ankles/feet/toes/heels; keep per-anchor camera depth and "
                "in-frame flag"
            ),
            "valid_pointmap": "finite xyz, 0.05m < z < 50m, finite confidence",
            "human_exclusion": (
                "exclude the emitted Human3R person mask dilated by 3 pixels; do not "
                "exclude the complete person bbox"
            ),
            "support_annulus": (
                "retain non-human pixels 4..16 pixels from the foot centre and record "
                "their exact UV so spatial extent can be audited"
            ),
            "confidence": (
                "store raw conf_self and use a deterministic within-patch rank; never "
                "tune a confidence threshold on held-out GT"
            ),
            "occlusion": (
                "foot is visible only when in frame, positive depth, and not farther "
                "behind the predicted depth than a frozen tolerance"
            ),
        },
        "future_candidate_contract": {
            "inputs": "last-pre/current-post predicted local foot patches only",
            "gate": (
                "BRTC accepted AND visible foot AND enough high-confidence non-human "
                "surface samples AND robust local plane residual/extent checks"
            ),
            "action": (
                "one shared translation for root/joints/vertices, preserving the last-pre "
                "signed foot-to-local-surface offset rather than forcing contact to zero"
            ),
            "bounds": "start with 30 mm cap and fixed shrinkage; freeze on dev only",
            "fallback": "rejected, unmatched, unobservable, or inconsistent feet are exact B0/BRTC",
            "camera_update": "none",
            "future_frames": 0,
        },
        "first_round_deterministic_gate": {
            "development_only_initial_values": True,
            "minimum_valid_surface_samples_per_foot": 24,
            "minimum_uv_quadrants": 3,
            "minimum_surface_extent_m": 0.05,
            "maximum_weighted_plane_median_residual_m": 0.02,
            "maximum_pre_post_normal_disagreement_deg": 25.0,
            "maximum_contact_like_signed_distance_m": 0.20,
            "maximum_left_right_proposal_disagreement_m": 0.02,
            "observable_relative_improvement_minimum": 0.10,
            "proposal": (
                "fit robust local planes; orient each normal from plane toward foot; "
                "only if stable last-pre and raw current-post signed offsets agree, "
                "move the BRTC-corrected body toward that agreed offset"
            ),
            "action": "clip(0.5 * robust_shared_proposal, norm <= 0.03m)",
            "otherwise": "exact frozen BRTC for accepted; exact B0 for rejected/unmatched",
            "warning": (
                "these values are a first dev policy, not frozen facts; after dev the "
                "entire policy JSON must be checksummed before held-out evaluation"
            ),
        },
        "cpu_replay_reproduction": {
            "cache_build": (
                "extend eval_brtc_multithumbs_egohumans.build_geometry_cache, then run "
                "once with --build_cache --device cpu to a new cache path under /data"
            ),
            "candidate_replay": (
                "load the extended .pt with map_location=cpu; no model/image access; "
                "form the gate before loading evaluator GT"
            ),
            "determinism": [
                "fixed foot IDs, patch size, masks, rank selection, plane solver, and dtype",
                "canonical policy SHA256 recorded before held-out splits",
                "extended cache SHA256 and source checkpoint SHA256 recorded",
                "per-person action/fallback reason and observable residuals serialized",
            ],
            "not_run_in_this_audit": True,
        },
        "required_evaluation": [
            "freeze on three offset0; validate on three offset1, dance, and box",
            "EgoHumans full chain spatial W/WA/pelvis/fixed/layout metrics",
            "root and joint Accel across both cuts",
            "camera bit-exact B0",
            "rejected/unmatched exact B0",
            "accepted action bounded and coverage reported",
            "predicted foot/scene residual and image reprojection displacement audits",
            "GT loaded only after candidate/gate is frozen",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V14 BRTC person-to-scene contact cache availability audit",
        "",
        f"Decision: **{report['decision']['status']}**.",
        "",
        "No model, GPU, image, or GT geometry was loaded. This is an availability audit, not a candidate evaluation.",
        "",
        "## Current cache verdict",
        "",
        "The current caches are insufficient for a defensible foot-local scene residual. "
        "The Ego cache has no pointmap/depth/confidence/scene/foot-visibility fields. "
        "The MultiHuman caches retain only a sparse background cloud; the generator "
        "explicitly removes every complete human bbox plus an 8% margin, then drops "
        "confidence and pixel coordinates. It therefore removes exactly the local support "
        "region needed for a foot-contact gate.",
        "",
        "| Split | Cases | Post cloud nonempty | Post points median | Foot nearest <=10cm | <=25cm | <=50cm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["multihuman_cache"].items():
        fractions = value["foot_nearest_sparse_cloud_fraction"]
        lines.append(
            f"| {name} | {value['loaded_case_count']} | "
            f"{value['post_cloud_nonempty_case_rate']:.1%} | "
            f"{value['post_sparse_cloud_point_count']['median']:.1f} | "
            f"{fractions['0.1']:.1%} | {fractions['0.25']:.1%} | "
            f"{fractions['0.5']:.1%} |"
        )
    ego = report["egohumans_cache"]
    lines.extend(
        [
            "",
            "## EgoHumans cache",
            "",
            f"Frames/person-frames: `{ego['frame_count']}` / `{ego['person_frame_count']}`.",
            f"Frame keys: `{', '.join(ego['frame_keys'])}`.",
            f"Person keys: `{', '.join(ego['person_keys'])}`.",
            f"Scene/visibility fields present: `{ego['scene_or_visibility_fields_present']}`.",
            "",
            "A fallback-only run would be numerically identical to frozen BRTC but have zero "
            "contact coverage. That is not a valid GO result, and spatial/Accel effects of a "
            "nonzero candidate cannot be reported from this cache.",
            "",
            "## Prior failure controls",
            "",
        ]
    )
    v46 = report["prior_failures"]["v11_2_contact_preserving"]
    residual = report["prior_failures"]["v14_scene_residual_observability_three_dev"]
    lines.extend(
        [
            f"- V11.2 forced contact with a mean `{v46['mean_root_contact_correction_m']:.3f} m` "
            f"root correction and caused `{v46['mean_human_reprojection_shift_px']:.1f} px` mean "
            f"reprojection displacement (`{v46['p95_human_reprojection_shift_px']:.1f} px` P95).",
            f"- The earlier three-dev scene probe changed camera composite from "
            f"`{residual['b0_camera_composite_mean']:.4f}` to "
            f"`{residual['scene_icp_camera_composite_mean']:.4f}` (ICP) or "
            f"`{residual['scene_mutual_camera_composite_mean']:.4f}` (bounded mutual translation).",
            "- Human3R pointmap and SMPL-X come from the same forward pass. Their agreement is "
            "a consistency proxy, not independent metric evidence.",
            "",
            "## Saved exceptions are not split coverage",
            "",
            f"The original-demo depth/conf export covers "
            f"`{report['saved_artifacts']['original_demo']['case_count']}` selected case; "
            f"the virtual ray-query artifact covers "
            f"`{report['saved_artifacts']['virtual_person_depth']['case_count']}` case. "
            "Neither supplies ordinary B0 foot-local evidence for MultiHuman validation and Ego chains.",
            "",
            "## Minimal cache extension",
            "",
            "Reuse the existing Human3R forward and retain compact camera-local 33x33 patches "
            "around the projected feet, including pointmap, confidence, UV, validity, human mask, "
            "foot depth/visibility, camera, and intrinsics. Do not store only world points: the "
            "frozen B0 camera must be applied at runtime.",
            "",
            "Foot-local masks must use a 33x33 patch, keep a 4..16 px support annulus, "
            "and remove only the emitted human mask dilated by 3 px. The current whole-bbox "
            "removal is unsuitable. Preserve raw confidence and exact UV rather than only "
            "the selected 3D points.",
            "",
            "The first candidate should preserve the last-pre signed foot-to-surface offset, not "
            "force distance to zero. It must be BRTC-accepted-only, plane-quality/visibility gated, "
            "bounded initially to 30 mm, camera-free, strictly causal, and exact fallback for every "
            "unobservable/rejected/unmatched person.",
            "",
            "### First deterministic development gate",
            "",
            "Require at least 24 valid samples, three UV quadrants, 5 cm 3D extent, <=2 cm "
            "weighted plane residual, <=25 deg pre/post normal disagreement, <=20 cm "
            "contact-like signed distance, and <=2 cm left/right proposal disagreement. "
            "Apply only if the predicted contact residual improves by at least 10%; the action is "
            "`clip(0.5 * proposal, 30 mm)`. These are initial dev values and must be frozen and "
            "checksummed before any held-out run.",
            "",
            "The reference offset must be supported by both stable past-only observations and the "
            "raw current-post Human3R person/scene pair. This prevents blindly treating a moving "
            "or airborne foot as contact. Because both outputs are still same-forward Human3R, the "
            "gate remains a consistency safeguard, not independent metric depth evidence.",
            "",
            "### CPU replay",
            "",
            "Extend the existing CPU-only Ego cache builder and write to a new path under `/data`; "
            "do not overwrite the frozen cache. After that one forward pass, all policy scans and "
            "held-out evaluation must load the `.pt` with `map_location=cpu`, access no images or "
            "model, form actions before GT is loaded, and record cache/checkpoint/policy SHA256 plus "
            "per-person fallback reasons.",
            "",
            "After extending the cache, freeze on three offset0 and validate unchanged on three "
            "offset1, dance, box, and full Ego chains with spatial, layout, reprojection, harm, "
            "root/joint Accel, camera, and fallback audits.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (args.ego_cache, args.output_dir, args.doc.parent):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain under Movie3R on /data")
    report = {
        "experiment": "v14_brtc_person_scene_contact_cache_availability",
        "protocol": {
            "model_loaded": False,
            "gpu_used": False,
            "images_loaded": False,
            "gt_geometry_read": False,
            "candidate_generated": False,
            "runtime_gate_evaluated": False,
            "scope": "schema and predicted non-GT geometry availability only",
        },
        "forward_availability": {
            "human3r_outputs_exist_before_compaction": [
                "pts3d_in_self_view",
                "pts3d_in_other_view",
                "conf_self",
                "conf",
                "msk (when emitted)",
            ],
            "evidence": [
                "versions/v13/gt_id_consensus.py::sampled_background_cloud",
                "versions/v14/probe_v14_internal_root_depth.py::prediction_pointmap",
                "versions/v14/probe_cut3r_virtual_person_depth.py",
            ],
            "current_ego_builder_discards_scene_outputs": True,
            "current_ego_builder": (
                "versions/v14/eval_brtc_multithumbs_egohumans.py::compact_person/"
                "build_geometry_cache"
            ),
        },
        "multihuman_cache": multihuman_availability(),
        "egohumans_cache": ego_availability(args.ego_cache),
        "saved_artifacts": saved_artifact_availability(),
        "prior_failures": prior_failure_evidence(),
        "minimal_cache_extension": minimal_extension(),
        "decision": {
            "current_cache_sufficient": False,
            "candidate_executed": False,
            "spatial_metrics_available_for_nonzero_candidate": False,
            "acceleration_metrics_available_for_nonzero_candidate": False,
            "status": "NO_GO_CURRENT_CACHE_FOR_PERSON_SCENE_CONTACT_RESIDUAL",
            "reason": (
                "missing Ego scene evidence and missing foot-local confidence/visibility/pixel "
                "evidence in MultiHuman caches"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "availability.json"
    text = markdown(report)
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    args.doc.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
