#!/usr/bin/env python3
"""Audit and aggregate the frozen five-case runtime records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SELECTION = ROOT / "selected_cases.json"
METHOD_ORDER = ("human3r", "shot3r", "prompthmr", "onlinehmr", "trace", "tram", "josh")
DISPLAY = {
    "human3r": "Human3R",
    "shot3r": "Shot3R",
    "prompthmr": "PromptHMR (SPEC)",
    "onlinehmr": "OnlineHMR",
    "trace": "TRACE",
    "tram": "TRAM",
    "josh": "JOSH",
}
ACCESS = {
    "human3r": "online",
    "shot3r": "online",
    "prompthmr": "offline",
    "onlinehmr": "semi-online",
    "trace": "online",
    "tram": "offline",
    "josh": "offline",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def record_by_case(root: Path, filename: str, case_id: str) -> Path:
    """Resolve a runtime record by immutable case id, never manifest position."""
    matches = []
    for path in sorted(root.glob(f"line*/{filename}")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if str(value.get("case_id")) == case_id:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {filename} for {case_id} below {root}, found {matches}"
        )
    return matches[0]


def runtime_path(method: str, line: int, case_id: str) -> Path:
    if method in {"human3r", "shot3r"}:
        return ROOT / "runs" / method / case_id / "runtime.json"
    if method == "prompthmr":
        return record_by_case(
            WORKSPACE
            / "data/Harmony4D_work_v17_full_test/external_predictions/"
            / "prompthmr_harmony4d/test/spec/harmony4d_test_spec",
            "prompthmr_spec.runtime.json",
            case_id,
        )
    if method == "onlinehmr":
        return record_by_case(
            WORKSPACE / "data/OnlineHMR_work_v1/runs/harmony4d/attempt04",
            "onlinehmr.runtime.json",
            case_id,
        )
    if method == "trace":
        return record_by_case(
            WORKSPACE / "external_baselines/ROMP/outputs/harmony4d_trace_v2/test",
            "trace.runtime.json",
            case_id,
        )
    if method == "tram":
        return ROOT / "runs" / "tram" / case_id / "runtime.json"
    if method == "josh":
        return record_by_case(
            WORKSPACE / "external_baselines/josh_runs/harmony4d88_20260906/cases/harmony4d",
            "status.json",
            case_id,
        )
    raise KeyError(method)


def extract(method: str, case: dict[str, Any]) -> dict[str, Any]:
    line = int(case["line"])
    case_id = str(case["case_id"])
    path = runtime_path(method, line, case_id)
    value = load(path)
    actual_case = str(value.get("case_id"))
    if actual_case != case_id:
        raise ValueError(f"case mismatch for {path}: {actual_case} != {case_id}")
    if method == "josh":
        seconds = float(value["inference_seconds"])
        status = value.get("status")
        frames = int(value.get("frames", -1))
        hardware = str(value.get("hardware", ""))
        if status != "audited":
            raise ValueError(f"JOSH record is not audited: {path}")
    else:
        seconds = float(value["wall_time_seconds"])
        status = value.get("status")
        if method in {"human3r", "shot3r", "tram"}:
            frames = int(value.get("frames", -1))
            hardware = str(value.get("hardware", ""))
        else:
            summary = value.get("raw_summary", {})
            frames = int(summary.get("frames", 150))
            hardware = str(value.get("hardware", "NVIDIA L20 (formal-run host audit)"))
        if status != "success":
            raise ValueError(f"unsuccessful record: {path}")
    if frames != 150:
        raise ValueError(f"expected 150 frames for {path}, got {frames}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"invalid runtime for {path}: {seconds}")
    if method in {"human3r", "shot3r", "tram", "josh"} and "NVIDIA L20" not in hardware:
        raise ValueError(f"runtime was not recorded on an NVIDIA L20: {path}: {hardware}")
    return {
        "method": method,
        "display": DISPLAY[method],
        "temporal_access": ACCESS[method],
        "line": line,
        "case_id": case_id,
        "action": case["action"],
        "viewpoint": case["viewpoint"],
        "frames": frames,
        "seconds": seconds,
        "fps": frames / seconds,
        "record": str(path.resolve()),
        "record_sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    selection = load(SELECTION)
    cases = selection["cases"]
    if len(cases) != 5 or len({case["case_id"] for case in cases}) != 5:
        raise ValueError("selection must contain exactly five distinct cases")
    rows = [extract(method, case) for method in METHOD_ORDER for case in cases]
    summary = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        seconds = [float(row["seconds"]) for row in selected]
        total_frames = sum(int(row["frames"]) for row in selected)
        summary.append(
            {
                "method": method,
                "display": DISPLAY[method],
                "temporal_access": ACCESS[method],
                "cases": len(selected),
                "frames": total_frames,
                "total_seconds": sum(seconds),
                "mean_seconds_per_150_frames": statistics.mean(seconds),
                "population_stdev_seconds": statistics.pstdev(seconds),
                "median_seconds_per_150_frames": statistics.median(seconds),
                "micro_average_fps": total_frames / sum(seconds),
            }
        )

    payload = {
        "schema_version": "Shot3R-H4D-CS150-seven-method-runtime-summary-v1",
        "protocol": str((ROOT / "PROTOCOL.md").resolve()),
        "selection": str(SELECTION.resolve()),
        "aggregation": "micro-average FPS = 750 / sum of five wall-clock runtimes",
        "rows": rows,
        "summary": summary,
    }
    (ROOT / "runtime_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(ROOT / "runtime_per_case.csv", rows)
    write_csv(ROOT / "runtime_summary.csv", summary)

    md = [
        "# H4D-CS150 runtime results",
        "",
        "Five 150-frame sequences (750 frames total); wall-clock process latency on NVIDIA L20.",
        "",
        "| Method | Access | s / 150 frames (mean ± std) | FPS ↑ |",
        "|---|---|---:|---:|",
    ]
    for row in summary:
        md.append(
            f"| {row['display']} | {row['temporal_access']} | "
            f"{row['mean_seconds_per_150_frames']:.1f} ± {row['population_stdev_seconds']:.1f} | "
            f"{row['micro_average_fps']:.3f} |"
        )
    (ROOT / "runtime_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        "% Auto-generated by aggregate.py; do not edit values manually.",
        "% 中文图注由正文表格所在文件提供。",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Method & Temporal access & s / 150 frames $\\downarrow$ & FPS $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in summary:
        method = "\\textbf{Shot3R}" if row["method"] == "shot3r" else row["display"]
        seconds = f"{row['mean_seconds_per_150_frames']:.1f} $\\pm$ {row['population_stdev_seconds']:.1f}"
        fps = f"{row['micro_average_fps']:.3f}"
        tex.append(f"{method} & {row['temporal_access']} & {seconds} & {fps} \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (ROOT / "runtime_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
