"""Causal person-local body-scale consistency after frozen BRTC v1.

For every BRTC-accepted anonymous match, stable torso/limb bone lengths from
the last pre-cut and current post-cut Human3R joints provide a robust log-scale
ratio.  A frozen fraction and relative cap scale post joints and vertices about
that person's (already BRTC-corrected) native root.  Roots and cameras never
change.  BRTC-rejected and unmatched people remain exact B0/v1 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG as BRTC_CONFIG,
    PersonTriangulationConfig,
    refine_matched_people,
)


# SMPL/SMPL-X body topology through wrists/feet.  Face, fingers, and toes are
# deliberately excluded because their prediction and indexing are less stable.
STABLE_BODY_EDGES = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
)


@dataclass(frozen=True)
class BodyScaleConfig:
    edges: tuple[tuple[int, int], ...] = STABLE_BODY_EDGES
    fraction: float = 0.5
    relative_cap: float = 0.10
    max_log_mad: float = 0.10
    min_valid_edges: int = 8
    min_bone_length_m: float = 0.03
    max_bone_length_m: float = 1.20


DEFAULT_BODY_SCALE_CONFIG = BodyScaleConfig()


def config_dict(config: BodyScaleConfig) -> dict[str, Any]:
    return {
        "edges": [list(edge) for edge in config.edges],
        "fraction": float(config.fraction),
        "relative_cap": float(config.relative_cap),
        "max_log_mad": float(config.max_log_mad),
        "min_valid_edges": int(config.min_valid_edges),
        "min_bone_length_m": float(config.min_bone_length_m),
        "max_bone_length_m": float(config.max_bone_length_m),
    }


def config_from_dict(value: dict[str, Any]) -> BodyScaleConfig:
    restored = dict(value)
    restored["edges"] = tuple(
        (int(edge[0]), int(edge[1])) for edge in restored["edges"]
    )
    return BodyScaleConfig(**restored)


def robust_body_scale_evidence(
    pre_person: dict[str, Any],
    post_person: dict[str, Any],
    config: BodyScaleConfig = DEFAULT_BODY_SCALE_CONFIG,
) -> dict[str, Any]:
    """Estimate observable pre/post body scale using only stable bone lengths."""

    pre = np.asarray(pre_person["joints"], dtype=np.float64)
    post = np.asarray(post_person["joints"], dtype=np.float64)
    if pre.ndim != 2 or post.ndim != 2 or pre.shape[1:] != (3,) or post.shape[1:] != (3,):
        raise ValueError("Body-scale joints must have shape [J,3]")
    log_ratios, edge_rows = [], []
    for first, second in config.edges:
        if max(first, second) >= min(len(pre), len(post)):
            continue
        pre_length = float(np.linalg.norm(pre[first] - pre[second]))
        post_length = float(np.linalg.norm(post[first] - post[second]))
        valid = bool(
            np.isfinite(pre_length)
            and np.isfinite(post_length)
            and config.min_bone_length_m <= pre_length <= config.max_bone_length_m
            and config.min_bone_length_m <= post_length <= config.max_bone_length_m
        )
        if valid:
            log_ratio = float(np.log(pre_length / post_length))
            if np.isfinite(log_ratio):
                log_ratios.append(log_ratio)
                edge_rows.append(
                    {
                        "edge": (first, second),
                        "pre_length_m": pre_length,
                        "post_length_m": post_length,
                        "log_pre_over_post": log_ratio,
                    }
                )
    values = np.asarray(log_ratios, dtype=np.float64)
    if len(values):
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
    else:
        center, mad = float("nan"), float("inf")
    return {
        "valid_edge_count": int(len(values)),
        "median_log_pre_over_post": center,
        "log_ratio_mad": mad,
        "edge_rows": edge_rows,
    }


def bounded_scale_factor(
    evidence: dict[str, Any],
    config: BodyScaleConfig = DEFAULT_BODY_SCALE_CONFIG,
) -> tuple[float, bool, str | None]:
    center = float(evidence["median_log_pre_over_post"])
    valid = int(evidence["valid_edge_count"])
    mad = float(evidence["log_ratio_mad"])
    accepted = bool(
        valid >= config.min_valid_edges
        and np.isfinite(center)
        and np.isfinite(mad)
        and mad <= config.max_log_mad
    )
    if not accepted:
        if valid < config.min_valid_edges:
            reason = "insufficient_valid_stable_edges"
        elif not np.isfinite(center) or not np.isfinite(mad):
            reason = "nonfinite_scale_evidence"
        else:
            reason = "log_ratio_mad_gate"
        return 1.0, False, reason
    raw = float(np.exp(config.fraction * center))
    scale = float(
        np.clip(raw, 1.0 - config.relative_cap, 1.0 + config.relative_cap)
    )
    return scale, True, None


def scale_person_about_root(
    person: dict[str, Any], scale: float
) -> dict[str, Any]:
    """Copy and scale joints/vertices about native root while preserving root."""

    output = dict(person)
    root = np.asarray(person["root"], dtype=np.float64)
    output["root"] = root.copy()
    for key in ("joints", "vertices"):
        if key in person:
            points = np.asarray(person[key], dtype=np.float64)
            # Preserve the exact fallback contract bit-for-bit.  Even the
            # algebraic identity root + (points-root) can round by one ulp.
            output[key] = (
                points.copy()
                if float(scale) == 1.0
                else root + float(scale) * (points - root)
            )
    return output


def refine_brtc_output_body_scale(
    pre_people: list[dict[str, Any]],
    brtc_post_people: list[dict[str, Any]],
    accepted_matches: Iterable[tuple[int, int]],
    config: BodyScaleConfig = DEFAULT_BODY_SCALE_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scale an already-frozen BRTC output without changing any root."""

    matches = tuple((int(first), int(second)) for first, second in accepted_matches)
    if len({first for first, _ in matches}) != len(matches) or len(
        {second for _, second in matches}
    ) != len(matches):
        raise ValueError("Accepted scale matches must be one-to-one")
    corrected = [scale_person_about_root(person, 1.0) for person in brtc_post_people]
    records = []
    for pre_index, post_index in matches:
        if not (0 <= pre_index < len(pre_people) and 0 <= post_index < len(brtc_post_people)):
            raise IndexError("Body-scale match index outside person arrays")
        evidence = robust_body_scale_evidence(
            pre_people[pre_index], brtc_post_people[post_index], config
        )
        scale, accepted, reason = bounded_scale_factor(evidence, config)
        corrected[post_index] = scale_person_about_root(
            brtc_post_people[post_index], scale
        )
        records.append(
            {
                "pre_index": pre_index,
                "post_index": post_index,
                "accepted": accepted,
                "scale_factor": scale,
                "fallback_reason": reason,
                "evidence": evidence,
            }
        )
    root_change = max(
        (
            float(
                np.max(
                    np.abs(
                        np.asarray(after["root"], dtype=np.float64)
                        - np.asarray(before["root"], dtype=np.float64)
                    )
                )
            )
            for before, after in zip(brtc_post_people, corrected)
        ),
        default=0.0,
    )
    return corrected, {
        "camera_update": "none",
        "body_scale_consistency": True,
        "matched_count": len(matches),
        "accepted_count": sum(bool(row["accepted"]) for row in records),
        "root_max_abs_change": root_change,
        "people": records,
        "config": config_dict(config),
    }


def refine_matched_people_body_scale_consistency(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    brtc_config: PersonTriangulationConfig = BRTC_CONFIG,
    scale_config: BodyScaleConfig = DEFAULT_BODY_SCALE_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run frozen BRTC v1, then scale only BRTC-accepted matched people."""

    materialized = tuple((int(first), int(second)) for first, second in matches)
    brtc_corrected, brtc_debug = refine_matched_people(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        materialized,
        brtc_config,
    )
    accepted_matches = [
        (int(row["pre_index"]), int(row["post_index"]))
        for row in brtc_debug["people"]
        if bool(row["accepted"])
    ]
    corrected, scale_debug = refine_brtc_output_body_scale(
        pre_people, brtc_corrected, accepted_matches, scale_config
    )
    debug = dict(brtc_debug)
    debug.update(
        {
            "camera_update": "none",
            "body_scale_consistency": True,
            "brtc_debug": brtc_debug,
            "body_scale_debug": scale_debug,
            "body_scale_config": config_dict(scale_config),
            "root_max_abs_change_vs_brtc": scale_debug["root_max_abs_change"],
        }
    )
    return corrected, debug
