#!/usr/bin/env python3
"""Evaluate a regenerated full Movie3R singleton payload in GT space.

This evaluator expects the payload contract used by ``demo.py``:
``smpl/verts_world`` is already camera-to-world transformed.  It compares
strict Human3R and the full ``B0 + BRTC-LC + C1-EMA25`` output on the same
30-frame AvatarReX case.  GT is opened only after prediction and is never a
runtime input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = REPO_ROOT / "output/v14/report_comparison_viewers/avatarrex_t1836_c22070935_c22053912_pre5_post25"
DEFAULT_CALIB = Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/calibration_full.json")
DEFAULT_SMPL = Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/smpl_params.npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--smpl-params", type=Path, default=DEFAULT_SMPL)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument("--frame0", type=int, default=1836)
    p.add_argument(
        "--extra-method", action="append", default=[], metavar="NAME=PATH",
        help="Additional saved payload to evaluate; may be repeated.",
    )
    return p.parse_args()


def gt_camera(calibration: dict, name: str) -> np.ndarray:
    value = calibration[str(name)]
    r = np.asarray(value["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(value["T"], dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def pose(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["pose"], dtype=np.float64)


def verts(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        value = np.asarray(z["verts_world"], dtype=np.float64)
    if value.shape[0] != 1:
        raise ValueError(f"Singleton evaluator received {value.shape} at {path}")
    return value[0]


def tf(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def camera_error(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rel = pred @ np.linalg.inv(target)
    angle = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))
    return {
        "translation_error_m": float(np.linalg.norm(pred[:3, 3] - target[:3, 3])),
        "rotation_error_deg": float(angle),
    }


def rigid_mpvpe(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    a = pred - pred.mean(0)
    b = target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    aligned = a @ r.T
    return float(np.linalg.norm(aligned - b, axis=1).mean()), float(
        np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)))
    )


def body_error(pred: np.ndarray, target: np.ndarray, regressor: np.ndarray) -> dict[str, float]:
    pj = regressor @ pred
    tj = regressor @ target
    joint_norm = np.linalg.norm(pj - tj, axis=1)
    centered = np.linalg.norm((pj - pj[0]) - (tj - tj[0]), axis=1)
    aligned, rotation = rigid_mpvpe(pred, target)
    return {
        "root_error_m": float(joint_norm[0]),
        "mean_joint_error_m": float(joint_norm.mean()),
        "p95_joint_error_m": float(np.percentile(joint_norm, 95)),
        "mpvpe_m": float(np.linalg.norm(pred - target, axis=1).mean()),
        "centered_joint_error_m": float(centered.mean()),
        "centered_mpvpe_m": float(np.linalg.norm((pred - pred.mean(0)) - (target - target.mean(0)), axis=1).mean()),
        "rigid_aligned_mpvpe_m": aligned,
        "best_rigid_rotation_deg": rotation,
    }


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def build_gt_vertices(model, params: dict[str, np.ndarray], index: int, gauge: np.ndarray) -> np.ndarray:
    keys = ("global_orient", "body_pose", "jaw_pose", "left_hand_pose", "right_hand_pose", "expression", "transl")
    kwargs = {key: torch.from_numpy(params[key][index:index + 1]).float() for key in keys}
    kwargs["betas"] = torch.from_numpy(params["betas"][0:1]).float()
    with torch.no_grad():
        local = model(**kwargs).vertices[0].detach().cpu().numpy().astype(np.float64)
    return tf(gauge, local)


def main() -> None:
    a = parse_args()
    case = a.case.resolve()
    output = (a.output or case / "CORRECTED_FULL_PIPELINE_GT_EVALUATION.json").resolve()
    markdown = (a.markdown or case / "CORRECTED_FULL_PIPELINE_GT_EVALUATION.md").resolve()
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads(a.calibration.read_text(encoding="utf-8"))
    with np.load(a.smpl_params) as z:
        params = {key: np.asarray(z[key]) for key in z.files}

    model = smplx.create(str(REPO_ROOT / "src/models"), "smplx", gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10).eval()
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    original_dir = case / "original_human3r"
    movie_dir = case / "movie3r_b0_brtc_c1"
    pre_pred = pose(original_dir / "camera" / f"{a.boundary - 1:06d}.npz")
    c_pre = gt_camera(calibration, "22070935")
    c_post = gt_camera(calibration, "22053912")
    gauge = pre_pred @ np.linalg.inv(c_pre)
    post_target = gauge @ c_post

    methods = {"original_human3r": original_dir, "movie3r_b0_brtc_c1": movie_dir}
    for item in a.extra_method:
        if "=" not in item:
            raise ValueError(f"--extra-method expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        methods[name] = Path(path).expanduser().resolve()
    per_method = {}
    for name, directory in methods.items():
        camera_rows, body_rows = [], []
        for local_index in range(a.boundary, 30):
            dataset_frame = a.frame0 + local_index - a.boundary
            target = build_gt_vertices(model, params, dataset_frame, gauge)
            pred_camera = pose(directory / "camera" / f"{local_index:06d}.npz")
            pred_vertices = verts(directory / "smpl" / f"{local_index:06d}.npz")
            camera_rows.append(camera_error(pred_camera, post_target))
            body_rows.append(body_error(pred_vertices, target, regressor))
        per_method[name] = {
            "first_post_camera": camera_rows[0],
            "mean_post25_camera": mean_rows(camera_rows),
            "first_post_body": body_rows[0],
            "mean_post25_body": mean_rows(body_rows),
            "per_frame": [
                {"local_index": a.boundary + i, "dataset_frame": a.frame0 + i, "camera": camera_rows[i], "body": body_rows[i]}
                for i in range(len(camera_rows))
            ],
        }

    # Actual seam in the unified world gauge, plus GT seam for reference.
    def root_at(directory: Path, index: int) -> np.ndarray:
        return regressor @ verts(directory / "smpl" / f"{index:06d}.npz")
    gt_pre_body = build_gt_vertices(model, params, a.frame0, gauge)
    gt_post_body = build_gt_vertices(model, params, a.frame0, gauge)
    gt_root_seam = float(np.linalg.norm((regressor @ gt_post_body)[0] - (regressor @ gt_pre_body)[0]))
    seam = {}
    for name, directory in methods.items():
        seam[name] = {
            "predicted_root_jump_m": float(np.linalg.norm(root_at(directory, a.boundary)[0] - root_at(directory, a.boundary - 1)[0])),
            "gt_root_jump_m": gt_root_seam,
        }

    report = {
        "status": "evaluated",
        "case": str(case),
        "pipeline": "strict original vs clean reset + shadow B0 + BRTC-LC + C1-EMA25",
        "coordinate_contract": "all saved verts_world are C_cam_to_world @ V_cam; GT uses AvatarReX raw calibration gauge",
        "runtime_manifest": manifest,
        "gauge": {"matrix": gauge.tolist(), "post_target_camera": post_target.tolist()},
        "methods": per_method,
        "seam": seam,
        "comparison": {
            "first_post_root_gain_m": float(per_method["original_human3r"]["first_post_body"]["root_error_m"] - per_method["movie3r_b0_brtc_c1"]["first_post_body"]["root_error_m"]),
            "first_post_mpvpe_gain_m": float(per_method["original_human3r"]["first_post_body"]["mpvpe_m"] - per_method["movie3r_b0_brtc_c1"]["first_post_body"]["mpvpe_m"]),
            "mean_post25_root_gain_m": float(per_method["original_human3r"]["mean_post25_body"]["root_error_m"] - per_method["movie3r_b0_brtc_c1"]["mean_post25_body"]["root_error_m"]),
            "mean_post25_mpvpe_gain_m": float(per_method["original_human3r"]["mean_post25_body"]["mpvpe_m"] - per_method["movie3r_b0_brtc_c1"]["mean_post25_body"]["mpvpe_m"]),
        },
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Corrected full-pipeline singleton evaluation", "",
        "Coordinate contract: every saved `verts_world` uses `C_cam_to_world @ V_cam`; GT is evaluator-only.", "",
        "## Post-camera", "",
        "| Method | First translation | First rotation | Mean translation (25) | Mean rotation (25) |", "|---|---:|---:|---:|---:|",
    ]
    for name, value in per_method.items():
        f, m = value["first_post_camera"], value["mean_post25_camera"]
        lines.append(f"| {name} | {f['translation_error_m']:.3f} m | {f['rotation_error_deg']:.2f}° | {m['translation_error_m']:.3f} m | {m['rotation_error_deg']:.2f}° |")
    lines += ["", "## Post-human", "", "| Method | First root | First mean joint | First MPVPE | Mean root (25) | Mean MPVPE (25) |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in per_method.items():
        f, m = value["first_post_body"], value["mean_post25_body"]
        lines.append(f"| {name} | {f['root_error_m']:.3f} m | {f['mean_joint_error_m']:.3f} m | {f['mpvpe_m']:.3f} m | {m['root_error_m']:.3f} m | {m['mpvpe_m']:.3f} m |")
    lines += [
        "", "## Interpretation", "",
        "- B0 is intentionally coarse; its camera error is not treated as the final method error.",
        "- The table evaluates the complete B0+BRTC+C1 output after the corrected world-coordinate export.",
        "- BRTC/C1 currently leave the camera bit-exact (`camera_max_abs_change = 0`), so the final camera table is still the B0 camera table.",
        "- Human translation improves strongly, but first-post centered joint error is "
        f"{per_method['movie3r_b0_brtc_c1']['first_post_body']['centered_joint_error_m']:.3f} m "
        f"(baseline {per_method['original_human3r']['first_post_body']['centered_joint_error_m']:.3f} m); "
        "the remaining issue is global body orientation/relative camera-human geometry.",
        "- Positive gains below mean Movie3R is better than strict Human3R on this case.",
        "", f"Comparison: `{json.dumps(report['comparison'], ensure_ascii=False)}`", "",
    ]
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(markdown), "comparison": report["comparison"], "methods": {name: {"first_post_body": value["first_post_body"], "mean_post25_body": value["mean_post25_body"], "first_post_camera": value["first_post_camera"], "mean_post25_camera": value["mean_post25_camera"]} for name, value in per_method.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
