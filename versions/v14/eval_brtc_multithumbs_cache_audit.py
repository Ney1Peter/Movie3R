#!/usr/bin/env python3
"""Audit which Multi-THuMBS-style metrics saved BRTC-LC artifacts support.

This script is deliberately JSON-only.  It does not load a model, torch, image,
mesh, or GPU.  It validates the frozen reports and writes a strict availability
matrix plus the cut-level fixed-world proxies that can be defended from the
saved artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_EGOHUMANS = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/multithumbs_protocol/"
    "human3r_raw_egohumans_provisional.json"
)
DEFAULT_FRESH_BRTC = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/b0_two_view_person_triangulation/"
    "confirm_three_offset1.json"
)
DEFAULT_POSTHOC_BRTC = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/b0_two_view_person_triangulation/"
    "posthoc_dance_box_layout_consensus.json"
)
DEFAULT_B0_IDENTITY = (
    REPO_ROOT
    / "output/v14/b0_identity_matching_offset1_confirm/"
    "v14_b0_identity_matching.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_multithumbs_cache_audit"
)
DEFAULT_DOC = (
    REPO_ROOT / "versions/v14/docs/V14_BRTC_MULTITHUMBS_CACHE_AUDIT_20260801.md"
)

PAPER_REFERENCE = {
    "dataset": "EgoHumans",
    "w_mpjpe_mm": 279.0,
    "wa_mpjpe_mm": 166.0,
    "mpjpe_mm": 228.3,
    "mpvpe_mm": 262.2,
    "accel_unit_unspecified": 27.3,
    "ate_alignment_and_unit_unspecified": 0.7,
    "ids_aggregation_unspecified": 0.97,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_egohumans", type=Path, default=DEFAULT_RAW_EGOHUMANS)
    parser.add_argument("--fresh_brtc", type=Path, default=DEFAULT_FRESH_BRTC)
    parser.add_argument("--posthoc_brtc", type=Path, default=DEFAULT_POSTHOC_BRTC)
    parser.add_argument("--b0_identity", type=Path, default=DEFAULT_B0_IDENTITY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_case_value(report: dict, *keys: str) -> float:
    values = []
    for case in report["cases"]:
        value: Any = case
        for key in keys:
            value = value[key]
        values.append(float(value))
    return float(sum(values) / len(values))


def proxy_split(report: dict, status: str) -> dict:
    summary = report["summary"]
    baseline = summary["baseline"]
    corrected = summary["corrected"]
    proxy = {
        "status": status,
        "phase": report["phase"],
        "case_count": int(summary["case_count"]),
        "person_count": int(summary["person_count"]),
        "coverage": float(summary["coverage"]),
        "camera_max_abs_change": float(summary["camera_candidate_max_abs_change"]),
        "root_harm_over_5cm_rate": float(summary["root_harm_over_5cm_rate"]),
        "b0": {
            "fixed_world_root_mm": float(baseline["root_error_m"]["mean"] * 1000.0),
            "fixed_world_joint_mm": float(baseline["joint_error_m"]["mean"] * 1000.0),
            "fixed_world_vertex_mm": float(baseline["vertex_error_m"]["mean"] * 1000.0),
            "pairwise_root_distance_mm": float(
                baseline["pairwise_distance_error_m"]["mean"] * 1000.0
            ),
            "pairwise_root_vector_mm": float(
                baseline["pairwise_vector_error_m"]["mean"] * 1000.0
            ),
        },
        "b0_brtc_lc": {
            "fixed_world_root_mm": float(corrected["root_error_m"]["mean"] * 1000.0),
            "fixed_world_joint_mm": float(corrected["joint_error_m"]["mean"] * 1000.0),
            "fixed_world_vertex_mm": float(corrected["vertex_error_m"]["mean"] * 1000.0),
            "pairwise_root_distance_mm": float(
                corrected["pairwise_distance_error_m"]["mean"] * 1000.0
            ),
            "pairwise_root_vector_mm": float(
                corrected["pairwise_vector_error_m"]["mean"] * 1000.0
            ),
        },
        "relative_gain": {
            "root": float(summary["root_relative_gain"]),
            "joint": float(
                1.0
                - corrected["joint_error_m"]["mean"]
                / baseline["joint_error_m"]["mean"]
            ),
            "vertex": float(
                1.0
                - corrected["vertex_error_m"]["mean"]
                / baseline["vertex_error_m"]["mean"]
            ),
            "pairwise_root_vector": float(summary["layout_vector_relative_gain"]),
        },
    }
    return proxy


def metric_availability() -> dict:
    common_reason = (
        "Saved BRTC artifact contains one corrected post boundary frame per independent cut, "
        "not a continuous corrected identity trajectory under the paper protocol."
    )
    return {
        "w_mpjpe": {
            "status": "unavailable_for_strict_BRTC_comparison",
            "reason": common_reason
            + " W-MPJPE additionally needs a declared initial-frame alignment and full track.",
        },
        "wa_mpjpe": {
            "status": "unavailable_for_strict_BRTC_comparison",
            "reason": common_reason
            + " WA-MPJPE additionally fits on the complete evaluated trajectory.",
        },
        "mpjpe": {
            "status": "unavailable_as_paper_column",
            "reason": (
                "Saved joint_error is fixed-world, unaligned error, not per-frame pelvis-centered "
                "MPJPE. BRTC is a rigid translation, so a correctly pelvis-centered MPJPE would "
                "be unchanged, but its B0 value was not stored in this report."
            ),
        },
        "mpvpe": {
            "status": "unavailable_as_paper_column",
            "reason": (
                "Saved vertex_error is fixed-world, unaligned SMPL-X error, not topology-declared, "
                "pelvis-centered MPVPE. Rigid BRTC translation would cancel under pelvis alignment."
            ),
        },
        "accel": {
            "status": "unavailable_for_BRTC",
            "reason": common_reason
            + " The paper also does not publish Accel coordinates, fps, or unit.",
        },
        "ate": {
            "status": "unavailable_as_ATE",
            "reason": (
                "BRTC camera is bit-exact B0, so it cannot change ATE. Saved BRTC reports contain "
                "only per-cut first-post camera error, not a declared aligned camera trajectory ATE."
            ),
        },
        "ids": {
            "status": "unavailable_as_official_IDs",
            "reason": (
                "Fresh `three` retains automatic boundary association correctness, but not native "
                "continuous track IDs or the paper's miss/entry/exit/aggregation protocol."
            ),
        },
    }


def raw_egohumans_summary(report: dict) -> dict:
    metric = report["aggregate"]["metrics"]
    return {
        "status": "full_local_provisional_metrics_available",
        "scope": "Human3R raw only; three self-built 15-frame EgoHumans chains; no B0/BRTC",
        "w_mpjpe_mm": float(metric["w_mpjpe_mm"]),
        "wa_mpjpe_mm": float(metric["wa_mpjpe_mm"]),
        "mpjpe_mm": float(metric["mpjpe_mm"]),
        "mpvpe_mm": float(metric["mpvpe_mm"]),
        "accel_delta2_mm_per_frame2": float(
            metric["accel_second_difference_mm_per_frame2"]
        ),
        "accel_physical_m_per_s2": float(metric["accel_physical_m_per_s2"]),
        "ate_m_sim3": float(metric["ate_m_sim3_translation_rmse"]),
        "identity_switches_mean_per_stream": float(
            metric["identity_switches_mean_per_stream"]
        ),
        "official_comparability": False,
    }


def markdown(report: dict) -> str:
    raw = report["raw_egohumans"]
    fresh = report["brtc_strict_proxies"]["fresh_three_offset1"]
    posthoc = report["brtc_strict_proxies"]["posthoc_dance_box"]
    availability = report["brtc_metric_availability"]
    lines = [
        "# B0+BRTC-LC 的 Multi-THuMBS 指标缓存可用性审计",
        "",
        "> 日期：2026-08-01。全程只读取已有 JSON；未使用 GPU、未重新推理。",
        "",
        "## 1. 最终结论",
        "",
        "必须把两层结果分开：",
        "",
        "1. 旧 EgoHumans raw 连续链保存了逐帧轨迹，可计算完整的本地 provisional 指标；",
        "   但它没有 B0 或 BRTC-LC。",
        "2. 当前最佳 B0+BRTC-LC 保存的是独立 cut 的 first-post fixed-world 结果，能够严格",
        "   报 root/joint/vertex/layout proxy，不能把它们改名成论文 W/WA/MPJPE/MPVPE/Accel/ATE/IDs。",
        "",
        "因此目前仍没有 B0+BRTC-LC 在 EgoHumans 同数据、同 provisional evaluator 下的完整对表。",
        "",
        "## 2. raw EgoHumans：可计算但不是当前方法",
        "",
        "| Scope | W | WA | MPJPE | MPVPE | Accel Δ² | Accel physical | ATE | IDs/stream |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Human3R raw, 3×15 frames | {raw['w_mpjpe_mm']:.1f} | "
            f"{raw['wa_mpjpe_mm']:.1f} | {raw['mpjpe_mm']:.1f} | "
            f"{raw['mpvpe_mm']:.1f} | {raw['accel_delta2_mm_per_frame2']:.2f} | "
            f"{raw['accel_physical_m_per_s2']:.2f} | {raw['ate_m_sim3']:.3f} | "
            f"{raw['identity_switches_mean_per_stream']:.2f} |"
        ),
        "",
        "这组数据属于本地 EgoHumans `001_legoassemble` 自建短链，只能诊断 raw Human3R；",
        "不能作为 BRTC-LC 结果，也不是论文官方 split。",
        "",
        "## 3. B0+BRTC-LC：当前能严格报告的 fixed-world proxy",
        "",
        "| Split | Method | Root | World joint | World vertex | Pair distance | Pair vector |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, row in (("fresh three offset1", fresh), ("post-hoc dance+box", posthoc)):
        for method in ("b0", "b0_brtc_lc"):
            value = row[method]
            lines.append(
                f"| {split} | {method} | {value['fixed_world_root_mm']:.1f} | "
                f"{value['fixed_world_joint_mm']:.1f} | {value['fixed_world_vertex_mm']:.1f} | "
                f"{value['pairwise_root_distance_mm']:.1f} | {value['pairwise_root_vector_mm']:.1f} |"
            )
    lines.extend(
        [
            "",
            f"fresh `three offset1` 是策略冻结后的自动-ID 确认：{fresh['case_count']} cuts、",
            f"{fresh['person_count']} 人、覆盖率 {fresh['coverage']:.1%}、camera 最大改动 "
            f"`{fresh['camera_max_abs_change']:.1f}`。BRTC-LC 的 world joint/vertex 为 "
            f"`{fresh['b0_brtc_lc']['fixed_world_joint_mm']:.1f}/"
            f"{fresh['b0_brtc_lc']['fixed_world_vertex_mm']:.1f} mm`。",
            "",
            f"`dance+box` 已用于发现独立修正的 layout failure，因此共识版只能算 post-hoc "
            f"support；其 world joint/vertex 为 "
            f"`{posthoc['b0_brtc_lc']['fixed_world_joint_mm']:.1f}/"
            f"{posthoc['b0_brtc_lc']['fixed_world_vertex_mm']:.1f} mm`。",
            "",
            "这些值没有 pelvis alignment、trajectory Sim(3) 或论文 aggregation，禁止与",
            "Multi-THuMBS 的 MPJPE/MPVPE 同列比较。",
            "",
            "## 4. BRTC 论文指标可用性",
            "",
            "| 指标 | 状态 | 原因 |",
            "|---|---|---|",
        ]
    )
    labels = {
        "w_mpjpe": "W-MPJPE",
        "wa_mpjpe": "WA-MPJPE",
        "mpjpe": "MPJPE",
        "mpvpe": "MPVPE",
        "accel": "Accel",
        "ate": "ATE",
        "ids": "IDs",
    }
    for key in ("w_mpjpe", "wa_mpjpe", "mpjpe", "mpvpe", "accel", "ate", "ids"):
        row = availability[key]
        lines.append(f"| {labels[key]} | `{row['status']}` | {row['reason']} |")
    paper = report["paper_reference_only"]
    lines.extend(
        [
            "",
            "## 5. 论文参考线：当前不能判断胜负",
            "",
            "Multi-THuMBS EgoHumans 报告：",
            "",
            "```text",
            f"W/WA/MPJPE/MPVPE = {paper['w_mpjpe_mm']}/{paper['wa_mpjpe_mm']}/"
            f"{paper['mpjpe_mm']}/{paper['mpvpe_mm']} mm",
            f"Accel/ATE/IDs = {paper['accel_unit_unspecified']}/"
            f"{paper['ate_alignment_and_unit_unspecified']}/{paper['ids_aggregation_unspecified']}",
            "```",
            "",
            "raw EgoHumans 与 BRTC proxy 各缺一半条件，任何‘已经打过’或‘没有打过’的数值",
            "结论都不成立。当前唯一可靠结论是 BRTC-LC 显著改善 fixed-world root/layout，",
            "但刚性平移不会修复 pelvis-centered 内部 pose/shape。",
            "",
            "## 6. 为什么不能仅靠现有缓存补出 BRTC EgoHumans",
            "",
            "- EgoHumans raw evaluator 的三条 V13 cache 没有当前 frozen B0+BRTC shift；",
            "- B0+DA3 EgoHumans JSON 只保存 boundary 和标量误差，没有可直接复用的 BRTC 人体几何；",
            "- BRTC `three/dance/box` cache 属于另一 MultiHuman capture、相机和 cut 构造；",
            "- 混拼上述产物会把不同 checkpoint/forward/cache 当成同一次预测，结论无效。",
            "",
            "## 7. 最小闭环",
            "",
            "下一次只需在 EgoHumans 已有 chain 上用同一 frozen forward 保存：",
            "",
            "```text",
            "per-frame B0 camera c2w",
            "stable/native identity",
            "B0 and BRTC corrected 24 joints + 6890 vertices",
            "GT visibility/miss/FP association",
            "all pre/post frame indices and timestamps",
            "```",
            "",
            "随后复用 `eval_multithumbs_protocol.py`，即可得到 B0 与 B0+BRTC-LC 的同口径",
            "provisional W/WA/MPJPE/MPVPE/Accel/ATE/IDs。作者协议公开后再做正式对榜。",
            "",
            "## 8. 产物",
            "",
            "```text",
            "versions/v14/eval_brtc_multithumbs_cache_audit.py",
            "versions/v14/docs/V14_BRTC_MULTITHUMBS_CACHE_AUDIT_20260801.md",
            "output/v14/fine_alignment_research/brtc_multithumbs_cache_audit/audit.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> None:
    synthetic = {
        "phase": "confirm",
        "summary": {
            "case_count": 1,
            "person_count": 2,
            "coverage": 0.5,
            "camera_candidate_max_abs_change": 0.0,
            "root_harm_over_5cm_rate": 0.0,
            "root_relative_gain": 0.5,
            "layout_vector_relative_gain": 0.25,
            "baseline": {
                "root_error_m": {"mean": 1.0},
                "joint_error_m": {"mean": 2.0},
                "vertex_error_m": {"mean": 4.0},
                "pairwise_distance_error_m": {"mean": 0.2},
                "pairwise_vector_error_m": {"mean": 0.4},
            },
            "corrected": {
                "root_error_m": {"mean": 0.5},
                "joint_error_m": {"mean": 1.0},
                "vertex_error_m": {"mean": 2.0},
                "pairwise_distance_error_m": {"mean": 0.1},
                "pairwise_vector_error_m": {"mean": 0.3},
            },
        },
    }
    result = proxy_split(synthetic, "test")
    assert result["b0_brtc_lc"]["fixed_world_joint_mm"] == 1000.0
    assert result["relative_gain"]["joint"] == 0.5
    assert metric_availability()["ids"]["status"] != "available"


def main() -> None:
    args = parse_args()
    run_self_test()
    if args.self_test:
        print(">> self-test passed")
        return
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must remain inside the Movie3R /data workspace")
    if not str(args.doc.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Document must remain inside the Movie3R /data workspace")
    raw = read_json(args.raw_egohumans)
    fresh = read_json(args.fresh_brtc)
    posthoc = read_json(args.posthoc_brtc)
    identity = read_json(args.b0_identity)
    if fresh.get("phase") != "confirm" or not fresh.get("pass", False):
        raise ValueError("Expected passing frozen three offset1 confirmation")
    if posthoc.get("phase") != "frozen":
        raise ValueError("Expected dance+box frozen/post-hoc report")
    if float(fresh["summary"]["camera_candidate_max_abs_change"]) != 0.0:
        raise AssertionError("Fresh BRTC report changed camera")
    auto = identity["summary"]["all"]["learned_b0"]["root_torso_joints"]
    if float(auto["assignment_accuracy"]) != 1.0:
        raise AssertionError("Fresh automatic B0 association is not 100%")
    if float(fresh["summary"]["association_accuracy"]) != 1.0:
        raise AssertionError("Fresh BRTC association is not 100%")
    fresh_person_count = int(fresh["summary"]["person_count"])
    automatic_correct_count = int(
        round(float(fresh["summary"]["association_accuracy"]) * fresh_person_count)
    )

    report = {
        "title": "Strict saved-cache availability audit for B0+BRTC-LC vs Multi-THuMBS",
        "execution": {
            "json_only": True,
            "gpu_used": False,
            "model_inference_run": False,
        },
        "inputs": {
            "raw_egohumans": args.raw_egohumans,
            "fresh_brtc": args.fresh_brtc,
            "posthoc_brtc": args.posthoc_brtc,
            "b0_identity": args.b0_identity,
        },
        "paper_reference_only": PAPER_REFERENCE,
        "raw_egohumans": raw_egohumans_summary(raw),
        "brtc_metric_availability": metric_availability(),
        "brtc_strict_proxies": {
            "fresh_three_offset1": proxy_split(
                fresh, "fresh_frozen_policy_automatic_boundary_association"
            ),
            "posthoc_dance_box": proxy_split(
                posthoc, "posthoc_support_not_pristine_confirmation"
            ),
        },
        "fresh_automatic_boundary_association": {
            "accuracy": float(fresh["summary"]["association_accuracy"]),
            "correct_count": automatic_correct_count,
            "person_count": fresh_person_count,
            "identity_report_accuracy_cross_check": float(
                auto["assignment_accuracy"]
            ),
            "official_ids_metric": False,
        },
        "b0_first_post_camera_diagnostic": {
            "fresh_three_translation_error_m_mean": mean_case_value(
                fresh, "camera", "b0_translation_error_m"
            ),
            "fresh_three_rotation_error_deg_mean": mean_case_value(
                fresh, "camera", "b0_rotation_error_deg"
            ),
            "is_ate": False,
            "brtc_changes_camera": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit.json"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    args.doc.write_text(text, encoding="utf-8")
    (args.output_dir / "README.md").write_text(text, encoding="utf-8")
    print(text)
    print(f">> wrote {json_path}")


if __name__ == "__main__":
    main()
