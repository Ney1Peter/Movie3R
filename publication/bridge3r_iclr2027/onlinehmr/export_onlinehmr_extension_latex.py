#!/usr/bin/env python3
"""Export auditable LaTeX fragments for the OnlineHMR extension campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ORDER = (
    "harmony4d_multicut",
    "aist_cs150",
    "mvhuman_mvh150",
    "aist_mc150_3",
    "aist_mc150_4",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def metric(payload: dict[str, Any], name: str) -> tuple[float | None, int]:
    value = payload.get("overall", {}).get(name, {})
    mean = value.get("mean")
    return (None if mean is None else float(mean), int(value.get("support", 0)))


def number(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def conditional(value: float | None, support: int, denominator: int, digits: int = 1) -> str:
    rendered = number(value, digits)
    return rendered if value is None or support == denominator else f"{rendered} ({support})"


def percent(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{100.0 * value:.{digits}f}"


def require_complete(root: Path) -> dict[str, dict[str, Any]]:
    summary = read_json(root / "formal/onlinehmr_extension_campaign_summary.json")
    if not summary.get("all_protocols_complete"):
        raise RuntimeError("campaign summary is not complete")
    result: dict[str, dict[str, Any]] = {}
    for name in ORDER:
        path = root / "formal" / name / "onlinehmr_extension_aggregate.json"
        payload = read_json(path)
        expected = int(payload["fixed_manifest_denominator"])
        if (
            int(payload.get("reported_case_count", -1)) != expected
            or int(payload.get("missing_case_count", -1)) != 0
        ):
            raise RuntimeError(f"incomplete fixed denominator: {path}")
        result[name] = payload
    return result


def paired_bridge(root: Path, protocol: str) -> dict[str, Any]:
    payload = read_json(
        root / "formal" / protocol / "paired_internal/onlinehmr_internal_paired.json"
    )
    candidates = [
        value
        for value in payload.get("internal_methods", {}).values()
        if value.get("label") == "BRIDGE3R"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one BRIDGE3R pairing for {protocol}")
    return candidates[0]


def paired_cell(method: dict[str, Any], name: str, denominator: int, digits: int = 1) -> str:
    value = method.get("overall", {}).get(name, {}).get("internal_advantage", {})
    mean = value.get("mean")
    interval = value.get("ci95")
    support = int(value.get("case_support", 0))
    if mean is None or not interval:
        return "--"
    rendered = f"{float(mean):.{digits}f} [{float(interval[0]):.{digits}f},{float(interval[1]):.{digits}f}]"
    return rendered if support == denominator else f"{rendered} ({support})"


def export_cs150(payload: dict[str, Any]) -> str:
    n = int(payload["fixed_manifest_denominator"])
    pa, pa_n = metric(payload, "PA-MPJPE_mm")
    anchor, anchor_n = metric(payload, "Anchor-MPJPE_mm")
    seam, seam_n = metric(payload, "Seam-root_mm")
    orient, orient_n = metric(payload, "Seam-orientation_deg")
    camera, camera_n = metric(payload, "Camera-rotation_deg")
    coverage, _ = metric(payload, "Coverage")
    return (
        "% Auto-generated from the fixed-denominator OnlineHMR CS150 aggregate.\n"
        "Same-input semi-online & OnlineHMR"
        f" & {conditional(pa, pa_n, n)}"
        f" & {conditional(anchor, anchor_n, n)}"
        f" & {conditional(seam, seam_n, n)}"
        f" & {conditional(orient, orient_n, n)}"
        f" & {conditional(camera, camera_n, n)}"
        f" & {percent(coverage)} \\\\\n"
    )


def export_multicut(payload: dict[str, Any]) -> str:
    n = int(payload["fixed_manifest_denominator"])
    values = []
    for name in (
        "PA-MPJPE_mm",
        "Anchor-MPJPE_mm",
        "Seam-root_mm",
        "Seam-orientation_deg",
        "Camera-rotation_deg",
    ):
        value, support = metric(payload, name)
        values.append(conditional(value, support, n))
    coverage, _ = metric(payload, "Coverage")
    return (
        "% Auto-generated from the fixed-denominator OnlineHMR multi-cut aggregate.\n"
        "OnlineHMR"
        + "".join(f" & {value}" for value in values)
        + f" & {percent(coverage)} \\\\\n"
    )


def export_mvhuman(payload: dict[str, Any]) -> str:
    n = int(payload["fixed_manifest_denominator"])
    values = []
    for name, digits in (
        ("PA-MPJPE_mm", 1),
        ("Anchor-MPJPE_mm", 1),
        ("Seam-root_mm", 1),
        ("Camera-rotation_deg", 1),
        ("Camera-translation_m", 3),
    ):
        value, support = metric(payload, name)
        values.append(conditional(value, support, n, digits))
    coverage, _ = metric(payload, "Coverage")
    return (
        "% Auto-generated from the fixed-denominator OnlineHMR MVH150 aggregate.\n"
        "OnlineHMR"
        + "".join(f" & {value}" for value in values)
        + f" & {percent(coverage)} \\\\\n"
    )


def export_harmony(payload: dict[str, Any]) -> str:
    n = int(payload["fixed_manifest_denominator"])
    w, w_n = metric(payload, "W-MPJPE_mm")
    wa, wa_n = metric(payload, "WA-MPJPE_mm")
    ate, ate_n = metric(payload, "ATE-Sim3_m")
    idf1, _ = metric(payload, "IDF1")
    coverage, _ = metric(payload, "Coverage")
    seam, seam_n = metric(payload, "Seam-root_m")
    return (
        "% Auto-generated from the fixed-denominator OnlineHMR Harmony4D aggregate.\n"
        "OnlineHMR"
        f" & {conditional(w, w_n, n)}"
        f" & {conditional(wa, wa_n, n)}"
        f" & {conditional(ate, ate_n, n, 3)}"
        f" & {number(idf1, 3)}"
        f" & {number(coverage, 3)}"
        f" & {conditional(seam, seam_n, n, 3)} \\\\\n"
    )


def export_boundary_scaling(payloads: dict[str, dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated OnlineHMR repeated-cut boundary diagnostics.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Protocol & Boundary & Seam-root $\\downarrow$ & Seam-orient. $\\downarrow$ & Cam. rot. $\\downarrow$ & Cam. trans. $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for key, label in (("aist_mc150_3", "Two cuts"), ("aist_mc150_4", "Three cuts")):
        payload = payloads[key]
        for order, summary in sorted(payload.get("boundary_order", {}).items(), key=lambda item: int(item[0])):
            metrics = summary.get("metrics", {})
            cells = []
            for name, digits in (
                ("seam_root_excess_mm", 1),
                ("seam_orientation_excess_deg", 1),
                ("camera_relative_rotation_deg", 1),
                ("camera_relative_translation_m", 3),
            ):
                item = metrics.get(name, {})
                cells.append(conditional(item.get("mean"), int(item.get("support", 0)), int(summary["boundary_count"]), digits))
            lines.append(
                f"{label} & {order}"
                + "".join(f" & {cell}" for cell in cells)
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def export_angle_strata(payload: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated OnlineHMR MVHuman viewpoint-stratified results.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Viewpoint stratum & $N$ & PA & Anchor & Seam-root & Cam. rot. & Coverage \\\\",
        "\\midrule",
    ]
    for key, label in (
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
        ("very_large", "very-large"),
        ("extreme", "extreme"),
    ):
        group = payload.get("angle_strata", {}).get(key)
        if not group:
            continue
        n = int(group["case_count"])
        cells = []
        for name in ("PA-MPJPE_mm", "Anchor-MPJPE_mm", "Seam-root_mm", "Camera-rotation_deg"):
            item = group.get("metrics", {}).get(name, {})
            cells.append(conditional(item.get("mean"), int(item.get("support", 0)), n))
        coverage = group.get("metrics", {}).get("Coverage", {}).get("mean")
        lines.append(
            f"{label} & {n}"
            + "".join(f" & {cell}" for cell in cells)
            + f" & {percent(coverage)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def export_paired_bridge(root: Path, payloads: dict[str, dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated paired BRIDGE3R advantage over OnlineHMR.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Protocol & PA & Anchor & Seam-root & Cam. rot. & Coverage \\\\",
        "\\midrule",
    ]
    for key, label in (
        ("aist_cs150", "AIST++ single cut"),
        ("aist_mc150_3", "AIST++ two cuts"),
        ("aist_mc150_4", "AIST++ three cuts"),
        ("mvhuman_mvh150", "MVHuman single cut"),
    ):
        method = paired_bridge(root, key)
        n = int(payloads[key]["fixed_manifest_denominator"])
        cells = [
            paired_cell(method, "PA-MPJPE_mm", n),
            paired_cell(method, "Anchor-MPJPE_mm", n),
            paired_cell(method, "Seam-root_mm", n),
            paired_cell(method, "Camera-rotation_deg", n),
            paired_cell(method, "Coverage", n, 3),
        ]
        lines.append(
            label + "".join(f" & {cell}" for cell in cells) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def export_harmony_paired(root: Path, payload: dict[str, Any]) -> str:
    method = paired_bridge(root, "harmony4d_multicut")
    n = int(payload["fixed_manifest_denominator"])
    lines = [
        "% Auto-generated paired BRIDGE3R advantage over OnlineHMR on Harmony4D multi-cut.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Comparison & W & WA & ATE-Sim3 & IDF1 & Coverage \\\\",
        "\\midrule",
        "BRIDGE3R advantage"
        f" & {paired_cell(method, 'W-MPJPE_mm', n)}"
        f" & {paired_cell(method, 'WA-MPJPE_mm', n)}"
        f" & {paired_cell(method, 'ATE-Sim3_m', n, 3)}"
        f" & {paired_cell(method, 'IDF1', n, 3)}"
        f" & {paired_cell(method, 'Coverage', n, 3)} \\\\ ".rstrip(),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def export_mvhuman_viewpoint_paired(root: Path) -> str:
    method = paired_bridge(root, "mvhuman_mvh150")
    lines = [
        "% Auto-generated paired MVHuman viewpoint analysis.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Stratum & Online Anchor & BRIDGE3R Anchor & Online cam. rot. & BRIDGE3R cam. rot. & Cam. gain [95\\% CI] \\\\",
        "\\midrule",
    ]
    for key, label in (
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
        ("very_large", "very-large"),
        ("extreme", "extreme"),
    ):
        group = method.get("angle_strata", {}).get(key, {})
        anchor = group.get("Anchor-MPJPE_mm", {})
        camera = group.get("Camera-rotation_deg", {})
        online_anchor = anchor.get("online", {}).get("mean")
        bridge_anchor = anchor.get("internal", {}).get("mean")
        online_camera = camera.get("online", {}).get("mean")
        bridge_camera = camera.get("internal", {}).get("mean")
        gain = camera.get("internal_advantage", {})
        interval = gain.get("ci95")
        gain_text = (
            "--"
            if gain.get("mean") is None or not interval
            else f"{float(gain['mean']):.1f} [{float(interval[0]):.1f},{float(interval[1]):.1f}]"
        )
        lines.append(
            f"{label}"
            f" & {number(online_anchor)} & {number(bridge_anchor)}"
            f" & {number(online_camera)} & {number(bridge_camera)}"
            f" & {gain_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    output = args.output_dir.resolve()
    payloads = require_complete(root)
    fragments = {
        "aist_cs150_onlinehmr_row.tex": export_cs150(payloads["aist_cs150"]),
        "aist_mc150-3_onlinehmr_row.tex": export_multicut(payloads["aist_mc150_3"]),
        "aist_mc150-4_onlinehmr_row.tex": export_multicut(payloads["aist_mc150_4"]),
        "mvhuman_onlinehmr_row.tex": export_mvhuman(payloads["mvhuman_mvh150"]),
        "harmony4d_multicut_onlinehmr_row.tex": export_harmony(payloads["harmony4d_multicut"]),
        "aist_multicut_onlinehmr_boundaries.tex": export_boundary_scaling(payloads),
        "mvhuman_onlinehmr_angle_strata.tex": export_angle_strata(payloads["mvhuman_mvh150"]),
        "onlinehmr_extension_paired_bridge3r.tex": export_paired_bridge(root, payloads),
        "harmony4d_multicut_onlinehmr_paired.tex": export_harmony_paired(
            root, payloads["harmony4d_multicut"]
        ),
        "mvhuman_onlinehmr_viewpoint_paired.tex": export_mvhuman_viewpoint_paired(root),
    }
    for name, value in fragments.items():
        atomic_text(output / name, value)
    print(json.dumps({"output_dir": str(output), "fragments": sorted(fragments)}, indent=2))


if __name__ == "__main__":
    main()
