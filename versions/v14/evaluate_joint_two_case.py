#!/usr/bin/env python3
"""Unified evaluator for the AvatarReX and three-person joint probes.

This file is evaluator-only.  Calibration/SMPL-X supervision is loaded after
the saved GT-free payloads are complete.  It reports camera, root/joint/mesh,
seam, and anonymous ID continuity metrics for the fixed two-case protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--avatar-case", type=Path, required=True)
    p.add_argument("--avatar-method", type=Path, required=True)
    p.add_argument("--multi-case", type=Path, required=True)
    p.add_argument("--multi-method", type=Path, required=True)
    p.add_argument("--multi-cache", type=Path, required=True)
    p.add_argument("--multi-data-root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"))
    p.add_argument("--multi-boundary", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def pose(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64)


def vertices(path: Path, index: int) -> np.ndarray:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as z:
        return np.asarray(z["verts_world"], dtype=np.float64)


def camera_error(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from scipy.spatial.transform import Rotation
    rel = pred @ np.linalg.inv(target)
    return {
        "translation_m": float(np.linalg.norm(pred[:3, 3] - target[:3, 3])),
        "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel[:3, :3]).as_rotvec()))),
    }


def body_error(pred: np.ndarray, target: np.ndarray, reg: np.ndarray) -> dict[str, float]:
    pj, tj = reg @ pred, reg @ target
    joints = np.linalg.norm(pj - tj, axis=1)
    return {
        "root_m": float(joints[0]),
        "joint_m": float(joints.mean()),
        "mpvpe_m": float(np.linalg.norm(pred - target, axis=1).mean()),
        "centered_joint_m": float(np.linalg.norm((pj - pj[0]) - (tj - tj[0]), axis=1).mean()),
    }


def avatar(args: argparse.Namespace) -> dict:
    import smplx
    import torch
    case = args.avatar_case.resolve()
    calibration = json.loads(Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/calibration_full.json").read_text())
    with np.load("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/smpl_params.npz") as z:
        params = {key: np.asarray(z[key]) for key in z.files}
    model = smplx.create(str(Path(__file__).resolve().parents[2] / "src/models"), "smplx", gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10).eval()
    reg = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    def gt_cam(name: str) -> np.ndarray:
        val = calibration[name]; r = np.asarray(val["R"]).reshape(3, 3); t = np.asarray(val["T"])
        out = np.eye(4); out[:3, :3] = r.T; out[:3, 3] = -r.T @ t; return out
    def tf(M: np.ndarray, x: np.ndarray) -> np.ndarray:
        return x @ M[:3, :3].T + M[:3, 3]
    pre = pose(case / "original_human3r", 4); gauge = pre @ np.linalg.inv(gt_cam("22070935")); target_camera = gauge @ gt_cam("22053912")
    def gt_body(index: int) -> np.ndarray:
        kw = {key: torch.from_numpy(params[key][index:index + 1]).float() for key in ("global_orient", "body_pose", "jaw_pose", "left_hand_pose", "right_hand_pose", "expression", "transl")}
        kw["betas"] = torch.from_numpy(params["betas"][0:1]).float()
        with torch.no_grad(): local = model(**kw).vertices[0].detach().cpu().numpy().astype(np.float64)
        return tf(gauge, local)
    rows = []
    for name, directory in (("b0", case / "movie3r_b0_brtc_c1"), ("joint", args.avatar_method.resolve())):
        cameras, bodies = [], []
        for index in range(5, 30):
            pred_camera = pose(directory, index); pred_body = vertices(directory, index)[0]
            cameras.append(camera_error(pred_camera, target_camera)); bodies.append(body_error(pred_body, gt_body(1836 + index - 5), reg))
        rows.append({"method": name, "first_camera": cameras[0], "mean_camera": {k: float(np.mean([r[k] for r in cameras])) for k in cameras[0]}, "first_body": bodies[0], "mean_body": {k: float(np.mean([r[k] for r in bodies])) for k in bodies[0]}, "post25": 25})
    return {"case": "AvatarReX_lbn1_t1836", "methods": rows}


def multi(args: argparse.Namespace) -> dict:
    import torch
    import smplx
    cache = torch.load(args.multi_cache, map_location="cpu", weights_only=False)
    from types import SimpleNamespace
    from versions.v13 import gt_id_consensus as gt
    case = args.multi_case.resolve(); method = args.multi_method.resolve(); boundary = int(args.multi_boundary)
    reg_model = smplx.create(str(Path(__file__).resolve().parents[2] / "src/models"), "smplx", gender="neutral", use_pca=False, flat_hand_mean=True, num_betas=10).eval()
    reg = reg_model.J_regressor.detach().cpu().numpy().astype(np.float64)
    baseline_dir = case / "movie3r_b0_brtc_c1"
    gauge = pose(baseline_dir, boundary - 1) @ np.linalg.inv(np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64))
    target_camera = gauge @ np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    targets = {identity: {key: np.asarray(value, dtype=np.float64) for key, value in human.items() if key in ("root", "joints", "vertices")} for identity, human in cache["gt"]["post_humans"].items()}
    assignment = cache["assignment"][-1]["assignments"]
    identity_by_detection = {int(row["detection_index"]): str(row["identity"]) for row in assignment}
    result = {}
    gt_args = SimpleNamespace(data_root=args.multi_data_root, sequence="three", output_dir=case / "gt_eval_frames", size=512)
    post_dataset_frame = int(cache["case"]["timestamp"])
    post_target_by_frame = {}
    for index in range(boundary, len(list((baseline_dir / "camera").glob("*.npz")))):
        frame = post_dataset_frame + index - boundary
        gt_post = np.linalg.inv(gt.gt_w2c(gt_args, int(cache["case"]["target_camera"]), frame))
        post_target_by_frame[index] = (gauge @ gt_post, gt.gt_human_payload(gt_args, frame, reg))
    for name, directory in (("b0", baseline_dir), ("joint_gate", method)):
        pred_camera = pose(directory, boundary); pred = vertices(directory, boundary)
        body_rows = []
        for detection, identity in identity_by_detection.items():
            if detection >= len(pred) or identity not in targets: continue
            target = targets[identity]
            target_world = {key: value @ gauge[:3, :3].T + gauge[:3, 3] for key, value in target.items()}
            body_rows.append({"identity": identity, "root_m": float(np.linalg.norm(reg[0] @ pred[detection] - target_world["root"])), "mpvpe_m": float(np.linalg.norm(pred[detection] - target_world["vertices"], axis=1).mean()), "joint_m": float(np.linalg.norm(reg @ pred[detection] - target_world["joints"], axis=1).mean())})
        pre_ids = np.asarray(vertices(directory, boundary - 1)).shape[0]
        post_ids = np.asarray(vertices(directory, boundary)).shape[0]
        frame_rows = []
        survival = []
        for index in sorted(post_target_by_frame):
            cam_target, humans_target = post_target_by_frame[index]
            cam_row = camera_error(pose(directory, index), cam_target)
            pred_frame = vertices(directory, index)
            with np.load(directory / "smpl" / f"{index:06d}.npz", allow_pickle=True) as payload:
                ids = np.asarray(payload["smpl_id"], dtype=np.int64).reshape(-1).tolist()
            rows = []
            for row, identity_id in enumerate(ids):
                identity = identity_by_detection.get(int(identity_id))
                if identity is None or identity not in humans_target: continue
                target = humans_target[identity]
                target_world = {key: np.asarray(value) @ gauge[:3, :3].T + gauge[:3, 3] for key, value in target.items() if key in ("root", "joints", "vertices")}
                rows.append({"root_m": float(np.linalg.norm(reg[0] @ pred_frame[row] - target_world["root"])), "mpvpe_m": float(np.linalg.norm(pred_frame[row] - target_world["vertices"], axis=1).mean())})
            frame_rows.append({"index": index, "camera": cam_row, "mean_root_m": float(np.mean([r["root_m"] for r in rows])) if rows else float("nan"), "mean_mpvpe_m": float(np.mean([r["mpvpe_m"] for r in rows])) if rows else float("nan"), "count": len(rows)})
            survival.append(len(rows))
        result[name] = {"camera": camera_error(pred_camera, target_camera), "body": body_rows, "mean_root_m": float(np.mean([r["root_m"] for r in body_rows])), "mean_mpvpe_m": float(np.mean([r["mpvpe_m"] for r in body_rows])), "mean_joint_m": float(np.mean([r["joint_m"] for r in body_rows])), "id_continuity": bool(pre_ids == post_ids == 3), "id_count_pre": int(pre_ids), "id_count_post": int(post_ids), "post25": {"mean_camera_translation_m": float(np.mean([r["camera"]["translation_m"] for r in frame_rows])), "mean_camera_rotation_deg": float(np.mean([r["camera"]["rotation_deg"] for r in frame_rows])), "mean_root_m": float(np.nanmean([r["mean_root_m"] for r in frame_rows])), "mean_mpvpe_m": float(np.nanmean([r["mean_mpvpe_m"] for r in frame_rows])), "track_counts": survival, "all_tracks_survive": bool(all(count == 3 for count in survival)), "frames": frame_rows}}
    result["gate_diagnostics"] = json.loads((method / "joint_camera_human.json").read_text())
    return {"case": "three_t1100_c1_c2_first_post", "methods": result}


def main() -> None:
    a = args(); report = {"avatar": avatar(a), "multi": multi(a), "protocol": "GT only in this evaluator"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
