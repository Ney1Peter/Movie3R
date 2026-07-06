#!/usr/bin/env python3
"""Export a 4-frame dataloader clip to Human3R viewer payloads without mesh baking.

This is a narrow utility for visualization/debugging. The standard
``v8_4_view_pose_benchmark_scene.py`` path calls ``demo.prepare_output()``, which
constructs SMPL-X meshes before writing any payload files. For a few MVHuman
samples that path can be very slow, while the saved-output viewer only needs the
camera, depth/color/conf, and SMPL parameter npz files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import roma
import torch
from smplx.joint_names import JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from dust3r.inference import loss_of_one_batch
from dust3r.model import ARCroco3DStereo
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.smpl_model import SMPLModel
from dust3r.utils.camera import pose_encoding_to_camera
from dust3r.utils.device import todevice
from dust3r.utils.geometry import geotrf, matrix_cumprod
from dust3r.utils.image import unpad_image
from scripts.v8_4_view_pose_benchmark_scene import (
    case_name,
    clone_outputs_with_pose,
    load_manifest,
    make_single_record_dataset,
    write_one_record_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entry", type=int, default=0)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--case_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--test_split", default="Test/v8_4_mixed_aabb_aaaa")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="resize_only_16")
    parser.add_argument("--raw_roots", default="null")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _filter_reset_overlap(outputs: dict) -> dict:
    preds = list(outputs["pred"])
    views = list(outputs["views"])
    reset_mask = torch.cat([view["reset"] for view in views], 0).cpu()
    shifted = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    preds = [pred for pred, mask in zip(preds, shifted) if not bool(mask)]
    views = [view for view, mask in zip(views, shifted) if not bool(mask)]
    reset_mask = reset_mask[~shifted]
    return {"pred": preds, "views": views, "reset_mask": reset_mask}


def _camera_poses_from_preds(preds: list[dict], reset_mask: torch.Tensor) -> list[torch.Tensor]:
    poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in preds]
    if reset_mask.any():
        cat = torch.cat(poses, 0)
        identity = torch.eye(4, device=cat.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), cat, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        cat = torch.einsum("bij,bjk->bik", shifted_bases, cat)
        poses = list(cat.unsqueeze(1).unbind(0))
    return poses


def _smpl_rotvec(pred: dict) -> torch.Tensor:
    rotmat = pred.get("smpl_rotmat", None)
    if rotmat is None:
        return torch.empty(0, 53, 3)
    rotmat = rotmat[0].cpu()
    if rotmat.shape[0] == 0:
        return torch.empty(0, 53, 3)
    return roma.rotmat_to_rotvec(rotmat)


def _array_or_zeros(value: torch.Tensor | None, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return value.detach().cpu().numpy().astype(np.float32)


def _unpad_frame_map(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return a single HxW map matching the point/depth resolution."""
    value = value.detach().cpu()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if tuple(value.shape[-2:]) != (height, width):
        value = unpad_image(value, [height, width])
    if value.ndim == 3:
        value = value[0]
    return value


def _bake_smpl_vertices_world(
    smpl_model: SMPLModel | None,
    rotvec: torch.Tensor,
    shape: torch.Tensor,
    transl: torch.Tensor,
    expression: torch.Tensor | None,
    cam2world: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if smpl_model is None or shape.shape[0] == 0:
        return np.empty((0, 0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    device = smpl_model.device
    num_betas = int(shape.shape[-1])
    body_model = smpl_model.smplx_neutral_11 if num_betas == 11 else smpl_model.smplx_neutral_10
    pose = rotvec.to(device=device, dtype=torch.float32)
    betas = shape.to(device=device, dtype=torch.float32)
    trans = transl.to(device=device, dtype=torch.float32)
    expr = expression.to(device=device, dtype=torch.float32) if expression is not None else body_model.expression.repeat(pose.shape[0], 1)

    out = body_model(
        betas=betas,
        global_orient=body_model.global_orient.repeat(pose.shape[0], 1),
        body_pose=pose[:, 1:22].flatten(1),
        left_hand_pose=pose[:, 22:37].flatten(1),
        right_hand_pose=pose[:, 37:52].flatten(1),
        jaw_pose=pose[:, 52:53].flatten(1),
        leye_pose=body_model.leye_pose.repeat(pose.shape[0], 1),
        reye_pose=body_model.reye_pose.repeat(pose.shape[0], 1),
        expression=expr.flatten(1),
    )
    verts = out.vertices
    joints = out.joints
    root_rot = roma.rotvec_to_rotmat(pose[:, 0])
    pelvis = joints[:, [0]]
    verts = (root_rot.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)
    joints = (root_rot.unsqueeze(1) @ (joints - pelvis).unsqueeze(-1)).squeeze(-1)

    # Human3R's saved SMPL translation is head-centered, matching SMPL_Layer(person_center="head").
    head_idx = JOINT_NAMES.index("head")
    verts = verts - joints[:, [head_idx]] + trans.unsqueeze(1)
    verts_world = geotrf(cam2world.to(device=device, dtype=torch.float32).unsqueeze(0), verts.unsqueeze(0))[0]
    return verts_world.detach().cpu().numpy().astype(np.float32), np.asarray(body_model.faces, dtype=np.int32)


def save_payload_light(outputs: dict, out_dir: Path, pose_key: str, smpl_model: SMPLModel | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("depth", "conf", "color", "camera", "smpl"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    outputs = clone_outputs_with_pose(outputs, pose_key)
    filtered = _filter_reset_overlap(outputs)
    preds = filtered["pred"]
    views = filtered["views"]
    reset_mask = filtered["reset_mask"]

    pts3ds_self = torch.cat([pred["pts3d_in_self_view"].cpu() for pred in preds], 0)
    conf_self = torch.cat([pred["conf_self"].cpu() for pred in preds], 0)
    colors = torch.cat([0.5 * (view["img"].cpu().permute(0, 2, 3, 1) + 1.0) for view in views], 0)
    poses = _camera_poses_from_preds(preds, reset_mask)

    batch, height, width, _ = pts3ds_self.shape
    pp = torch.tensor([width // 2, height // 2], dtype=torch.float32).repeat(batch, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld").detach().cpu()
    cam2world = torch.cat(poses).detach().cpu()
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]

    for frame_id in range(batch):
        pred = preds[frame_id]
        depth = pts3ds_self[frame_id, ..., 2].detach().cpu().numpy().astype(np.float32)
        conf = conf_self[frame_id].detach().cpu().numpy().astype(np.float32)
        color = np.clip(colors[frame_id].detach().cpu().numpy(), 0.0, 1.0)

        smpl_shape_t = pred.get("smpl_shape", torch.empty(1, 0, 10))[0].cpu()
        n_humans = int(smpl_shape_t.shape[0])
        smpl_expr_t = pred.get("smpl_expression", None)
        if smpl_expr_t is not None:
            smpl_expr = smpl_expr_t[0].detach().cpu().numpy().astype(np.float32)
        else:
            smpl_expr = np.zeros((n_humans, 10), dtype=np.float32)
        smpl_scores_t = pred.get("smpl_scores", None)
        if smpl_scores_t is None:
            smpl_scores = np.zeros((height, width), dtype=np.float32)
        else:
            smpl_scores = _unpad_frame_map(smpl_scores_t[..., 0], height, width).numpy().astype(np.float32)
        msk_t = pred.get("msk", None)
        if msk_t is None:
            msk = np.zeros((1, height, width), dtype=np.float32)
        else:
            msk = _unpad_frame_map(msk_t[..., 0], height, width).numpy().astype(np.float32)[None]
        smpl_id_t = pred.get("smpl_id", None)
        if smpl_id_t is None:
            smpl_id = np.arange(n_humans, dtype=np.int64)
        else:
            smpl_id = smpl_id_t[0].detach().cpu().numpy()
        rotvec_t = _smpl_rotvec(pred)
        transl_t = pred.get("smpl_transl", torch.empty(1, 0, 3))[0].detach().cpu()
        expr_t = None if smpl_expr_t is None else smpl_expr_t[0].detach().cpu()
        verts_world, faces = _bake_smpl_vertices_world(
            smpl_model,
            rotvec_t,
            smpl_shape_t,
            transl_t,
            expr_t,
            cam2world[frame_id],
        )

        np.save(out_dir / "depth" / f"{frame_id:06d}.npy", depth)
        np.save(out_dir / "conf" / f"{frame_id:06d}.npy", conf)
        iio.imwrite(out_dir / "color" / f"{frame_id:06d}.png", (color * 255.0).astype(np.uint8))
        np.savez(
            out_dir / "camera" / f"{frame_id:06d}.npz",
            pose=cam2world[frame_id].numpy().astype(np.float32),
            intrinsics=intrinsics[frame_id].numpy().astype(np.float32),
        )
        np.savez(
            out_dir / "smpl" / f"{frame_id:06d}.npz",
            scores=smpl_scores,
            msk=msk,
            shape=smpl_shape_t.numpy().astype(np.float32),
            rotvec=rotvec_t.numpy().astype(np.float32),
            transl=_array_or_zeros(transl_t, (n_humans, 3)),
            expression=smpl_expr,
            smpl_id=smpl_id,
            verts_world=verts_world,
            faces=faces,
        )


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    record = records[args.entry]
    case_dir = args.case_root / case_name(record)
    corrected_dir = case_dir / "corrected"
    raw_dir = case_dir / "raw_human3r"
    needed = [corrected_dir / "camera" / "000000.npz", raw_dir / "camera" / "000000.npz"]
    if all(path.is_file() for path in needed) and not args.overwrite:
        print(json.dumps({"case_dir": str(case_dir), "status": "exists"}, sort_keys=True))
        return

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "viewer_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    one_record_manifest = case_dir / "one_record_manifest.jsonl"
    write_one_record_manifest(one_record_manifest, record)

    add_path_to_dust3r(str(args.model_path))
    print(f"Loading model: {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).float().eval()
    smpl_model = SMPLModel(
        torch.device(args.device),
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    dataset_args = argparse.Namespace(
        data_root=args.data_root,
        test_split=args.test_split,
        resolution=tuple(args.resolution),
        resize_mode=args.resize_mode,
        raw_roots=args.raw_roots,
    )
    dataset = make_single_record_dataset(dataset_args, record, one_record_manifest)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    batch = todevice(next(iter(loader)), args.device)

    print("Running benchmark-matched forward...")
    with torch.no_grad():
        result = loss_of_one_batch(
            batch,
            model,
            criterion=None,
            accelerator=None,
            symmetrize_batch=False,
            inference=False,
            smpl_model=smpl_model,
        )
    outputs = todevice({"views": result["views"], "pred": result["pred"]}, "cpu")
    print(f"Saving corrected payload: {corrected_dir}")
    save_payload_light(outputs, corrected_dir, "camera_pose", smpl_model=smpl_model)
    print(f"Saving raw payload: {raw_dir}")
    save_payload_light(outputs, raw_dir, "v8_raw_camera_pose", smpl_model=smpl_model)
    print(json.dumps({"case_dir": str(case_dir), "corrected_dir": str(corrected_dir), "raw_dir": str(raw_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
