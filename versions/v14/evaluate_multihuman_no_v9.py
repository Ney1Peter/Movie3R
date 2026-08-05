#!/usr/bin/env python3
"""Offline ablation evaluator for the 30-frame three-person case.

The evaluator never feeds GT to a runtime method.  GT camera and SMPL-X OBJ
meshes are opened only here, after all payloads have been produced.  A common
pre-shot camera gauge is used for every method, then post-shot camera and
vertex/identity errors are reported.  The pre-shot prediction is used to
derive the stable row-to-GT identity map; the same map is then applied to the
post shot, which exposes a cross-shot permutation rather than hiding it with
an independent per-frame Hungarian match.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


DEFAULT_CASE = Path(
    "output/v14/joint_two_case_payloads_full/"
    "three_t1100_c1_c2_pre5_post25"
)
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--markdown", type=Path, default=None)
    return p.parse_args()


def transform(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(x) @ np.asarray(g)[:3, :3].T + np.asarray(g)[:3, 3]


def load_obj(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                rows.append((float(fields[1]), float(fields[2]), float(fields[3])))
    out = np.asarray(rows, dtype=np.float64)
    if out.shape != (10475, 3):
        raise ValueError(f"Unexpected OBJ shape at {path}: {out.shape}")
    return out


def gt_camera(data_root: Path, camera: int, frame: int) -> np.ndarray:
    path = data_root / "three/three/person0/parameter" / str(frame) / f"{camera}_extrinsic.npy"
    value = np.asarray(np.load(path), dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3] = value
    return np.linalg.inv(w2c)


def camera_error(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    relative = np.asarray(pred) @ np.linalg.inv(np.asarray(target))
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return (
        float(np.linalg.norm(pred[:3, 3] - target[:3, 3])),
        float(np.degrees(np.arccos(cosine))),
    )


def payload(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as z:
        return (
            np.asarray(z["verts_world"], dtype=np.float64),
            np.asarray(z["smpl_id"], dtype=np.int64).reshape(-1),
            np.asarray(z.get("scores", np.ones(len(z["verts_world"]))), dtype=np.float64),
        )


def vertex_cost(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target, axis=1).mean())


def assignment_cost(pred: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [[vertex_cost(pred[i], targets[j]) for j in range(len(targets))] for i in range(len(pred))],
        dtype=np.float64,
    )


def best_perm(cost: np.ndarray) -> tuple[tuple[int, ...], float]:
    n = int(cost.shape[0])
    candidates = [(perm, float(sum(cost[i, perm[i]] for i in range(n)))) for perm in itertools.permutations(range(n))]
    return min(candidates, key=lambda item: item[1])


def main() -> None:
    a = args()
    case = a.case.resolve()
    data = a.data_root.resolve()
    output = (a.output or case / "NO_V9_MULTI_COMPARISON.json").resolve()
    markdown = (a.markdown or case / "NO_V9_MULTI_COMPARISON.md").resolve()
    pre_index, cut, first_frame = 4, 5, 1100
    gt_pre_camera = gt_camera(data, 1, first_frame)
    gt_post_camera = gt_camera(data, 2, first_frame)
    with np.load(case / "original_human3r/camera/000004.npz") as z:
        pre_prediction = np.asarray(z["pose"], dtype=np.float64)
    gauge = pre_prediction @ np.linalg.inv(gt_pre_camera)
    target_camera = gauge @ gt_post_camera
    methods = [
        "original_human3r",
        "movie3r_b0_brtc_c1",
        "joint_camera_human_gate",
        "no_v9_raw_se3",
        "no_v9_raw_se3_human",
    ]
    if (case / "no_v9_adaptive_joint").is_dir():
        methods.append("no_v9_adaptive_joint")
    if (case / "no_v9_direct_adaptive_joint").is_dir():
        methods.append("no_v9_direct_adaptive_joint")

    # Stable pre-shot row -> GT identity map.  This is only for offline
    # evaluation and is never used by any payload transform.
    gt_pre_vertices = [
        transform(gauge, load_obj(data / "three/three" / f"person{i}/smplx/{first_frame}/smplx.obj"))
        for i in range(3)
    ]
    pre_vertices, pre_ids, _ = payload(case / methods[0], pre_index)
    pre_cost = assignment_cost(pre_vertices, gt_pre_vertices)
    pre_perm, pre_total = best_perm(pre_cost)
    # pre_perm[row] = GT person index.  Native ID -> GT is fixed from pre.
    native_id_to_gt = {int(pre_ids[row]): int(pre_perm[row]) for row in range(3)}

    report_methods: dict[str, dict] = {}
    for method in methods:
        directory = case / method
        camera_rows, body_rows, best_rows = [], [], []
        post_id_rows = []
        for index in range(cut, 30):
            frame = first_frame + (index - cut)
            gt_vertices = [
                transform(gauge, load_obj(data / "three/three" / f"person{i}/smplx/{frame}/smplx.obj"))
                for i in range(3)
            ]
            with np.load(directory / "camera" / f"{index:06d}.npz") as z:
                pred_camera = np.asarray(z["pose"], dtype=np.float64)
            pred_vertices, native_ids, _ = payload(directory, index)
            camera_rows.append(camera_error(pred_camera, target_camera))
            cost = assignment_cost(pred_vertices, gt_vertices)
            perm, total = best_perm(cost)
            best_rows.append({"mpvpe_m": float(total / 3.0), "perm_row_to_gt": list(perm)})
            direct_errors = []
            direct_assignments = {}
            for row, native in enumerate(native_ids.tolist()):
                gt_index = native_id_to_gt.get(int(native))
                if gt_index is None:
                    continue
                direct_assignments[str(int(native))] = int(gt_index)
                direct_errors.append(float(cost[row, gt_index]))
            body_rows.append({
                "direct_mpvpe_m": float(np.mean(direct_errors)) if direct_errors else float("nan"),
                "matched_count": len(direct_errors),
                "direct_native_id_to_gt": direct_assignments,
            })
            post_id_rows.append(native_ids.tolist())
        camera_arr = np.asarray(camera_rows, dtype=np.float64)
        direct = np.asarray([row["direct_mpvpe_m"] for row in body_rows], dtype=np.float64)
        best = np.asarray([row["mpvpe_m"] for row in best_rows], dtype=np.float64)
        # A permutation is present if the best geometry assignment changes
        # relative to the fixed pre-shot row->GT map.
        pre_perm_list = list(pre_perm)
        # Only compare permutations on frames where all three people are
        # present.  A two-person frame has no well-defined three-person
        # permutation and must not be counted as an ID failure.
        changed = [
            row["perm_row_to_gt"] != pre_perm_list
            for row, native in zip(best_rows, post_id_rows)
            if len(native) == 3
        ]
        report_methods[method] = {
            "first_post_camera": {"translation_error_m": float(camera_arr[0, 0]), "rotation_error_deg": float(camera_arr[0, 1])},
            "mean_post25_camera": {"translation_error_m": float(camera_arr[:, 0].mean()), "rotation_error_deg": float(camera_arr[:, 1].mean())},
            "max_post25_camera": {"translation_error_m": float(camera_arr[:, 0].max()), "rotation_error_deg": float(camera_arr[:, 1].max())},
            "first_post_direct_mpvpe_m": float(direct[0]),
            "mean_post25_direct_mpvpe_m": float(np.nanmean(direct)),
            "first_post_best_perm_mpvpe_m": float(best[0]),
            "mean_post25_best_perm_mpvpe_m": float(best.mean()),
            "pre_row_to_gt_perm": pre_perm_list,
            "pre_native_id_to_gt": native_id_to_gt,
            "post_native_ids_first": post_id_rows[0],
            "post_native_ids_unique_orders": sorted({tuple(row) for row in post_id_rows}),
            "geometry_permutation_rate": float(np.mean(changed)) if changed else float("nan"),
            "geometry_permutation_changed_frames": int(sum(changed)),
            "geometry_permutation_evaluable_frames": int(len(changed)),
            "post_frames_with_all_three": int(sum(len(row) == 3 for row in post_id_rows)),
            "post_frames": len(post_id_rows),
            "per_frame": [
                {"local_index": cut + i, "camera": {"translation_error_m": float(camera_rows[i][0]), "rotation_error_deg": float(camera_rows[i][1])}, "body": body_rows[i], "best": best_rows[i], "native_ids": post_id_rows[i]}
                for i in range(len(post_id_rows))
            ],
        }

    report = {
        "status": "evaluated",
        "case": str(case),
        "runtime_contract": "GT-free payload generation; GT only in this offline evaluator",
        "coordinate_contract": "predicted verts_world and cameras are compared in the pre-shot GT gauge",
        "pre_shot_row_to_gt_perm": list(pre_perm),
        "pre_shot_native_id_to_gt": native_id_to_gt,
        "pre_shot_assignment_cost_sum_m": pre_total,
        "methods": report_methods,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# No-V9 ablation: three-person `three_t1100_c1_c2_pre5_post25`", "",
        "GT is evaluator-only. All predictions are compared in one common pre-shot GT gauge.", "",
        f"Pre-shot row→GT permutation: `{list(pre_perm)}`; native-ID→GT map: `{native_id_to_gt}`.", "",
        "| Method | First cam t (m) | First cam R (°) | Mean cam t (m) | Mean cam R (°) | Mean direct MPVPE (m) | Mean best-perm MPVPE (m) | Geometry permutation rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report_methods.items():
        first, mean = row["first_post_camera"], row["mean_post25_camera"]
        lines.append(
            f"| {name} | {first['translation_error_m']:.3f} | {first['rotation_error_deg']:.2f} | "
            f"{mean['translation_error_m']:.3f} | {mean['rotation_error_deg']:.2f} | "
            f"{row['mean_post25_direct_mpvpe_m']:.3f} | {row['mean_post25_best_perm_mpvpe_m']:.3f} | "
            f"{100.0 * row['geometry_permutation_rate']:.1f}% |"
        )
    lines += [
        "", "## Interpretation", "",
        "- `no_v9_raw_se3` uses only the raw Human3R post-camera-to-pre-camera SE(3), with no V9/B0 checkpoint.",
        "- `no_v9_raw_se3_human` additionally commits one boundary Kabsch human residual; it is still a no-V9 control, not the proposed adaptive joint solver.",
        "- Direct MPVPE keeps the pre-shot identity map fixed; best-permutation MPVPE is an identity-agnostic geometry diagnostic.",
        "- A high geometry permutation rate means that per-frame geometry must be matched before a persistent ID can be claimed.",
    ]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
