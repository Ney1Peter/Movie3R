#!/usr/bin/env python3
"""Build and launch a full SceneHumanViewer for one V8.4 benchmark clip.

Unlike ``v8_4_view_pose_benchmark_npz.py``, this script does not only show
camera frustums. It reruns image-only inference for the selected 4-frame clip,
saves Human3R-style viewer payloads, then displays:

- scene pointmaps and SMPL meshes from the corrected output, or from
  ``--external_scene_dir`` when comparing against a saved demo.py viewer payload
- GT cameras in red
- raw Human3R cameras in gray, or from ``--external_raw_dir`` when comparing
  against a saved demo.py viewer payload
- corrected cameras in yellow

GT is only used for visualization alignment, never as model input.
When comparing against a saved demo.py payload, pass ``--align_corrected_to_raw0``
so the corrected camera/SMPL output is rigidly mapped into the same world frame
as the gray raw Human3R camera.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import viser.transforms as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from demo import prepare_input, prepare_output
from dust3r.datasets.avatarrex import AvatarReX_AABB, AvatarReX_Pattern, AvatarReX_Video
from dust3r.inference import inference_recurrent_lighter
from dust3r.inference import loss_of_one_batch
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import todevice
from scripts.view_human3r_saved_output import load_cam_dict, load_viewer_payload
from viser_utils import SceneHumanViewer


DEFAULT_RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
    "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entry", type=int, default=0)
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/eval/pose_only_final_test_only_gpu"),
    )
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=Path("output/v8_4_pose_benchmark"),
        help="Directory containing test_aabb.jsonl/test_aaaa.jsonl used by the benchmark.",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--test_split", default="Test/v8_4_mixed_aabb_aaaa")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512), metavar=("W", "H"))
    parser.add_argument(
        "--resize_mode",
        default="human3r_demo",
        help="AvatarReX image preprocessing mode for dataloader input.",
    )
    parser.add_argument("--raw_roots", default=json.dumps(DEFAULT_RAW_ROOTS, sort_keys=True))
    parser.add_argument(
        "--input_mode",
        choices=["dataloader", "demo"],
        default="dataloader",
        help="dataloader matches benchmark/training preprocessing; demo matches demo.py image-folder inference.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("output/v8_4_train_runs/v8_4_mixed_no_zxc_bs10_long_run1/checkpoint-final.pth"),
    )
    parser.add_argument(
        "--case_root",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/scene_viewer_cases"),
    )
    parser.add_argument("--port", type=int, default=8140)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--vis_threshold", type=float, default=2.0)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    parser.add_argument("--mask_morph", type=int, default=10)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--smpl_downsample", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument(
        "--freeze_updates_from_view",
        type=int,
        default=-1,
        help=(
            "Inference-only state freeze probe. If >=0, views from this index "
            "onward do not update recurrent state, pose memory, or V8/V9 history."
        ),
    )
    parser.add_argument(
        "--no_correction_before_view",
        type=int,
        default=-1,
        help=(
            "Inference-only correction probe. If >=0, force V8/V9 pose and "
            "human correction residuals to zero before this view index."
        ),
    )
    parser.add_argument("--reuse_saved", action="store_true", help="Skip inference if saved raw/corrected dirs already exist.")
    parser.add_argument("--build_only", action="store_true", help="Build saved viewer payloads and exit without launching Viser.")
    parser.add_argument("--show_labels", action="store_true", help="Show text labels for camera sets and metrics.")
    parser.add_argument(
        "--external_raw_dir",
        type=Path,
        default=None,
        help="Optional saved original Human3R output used for gray raw cameras and GT alignment.",
    )
    parser.add_argument(
        "--external_scene_dir",
        type=Path,
        default=None,
        help="Optional saved Human3R output used as the displayed scene/SMPL payload.",
    )
    parser.add_argument(
        "--align_corrected_to_raw0",
        action="store_true",
        help="Rigidly align corrected cameras/SMPL/pointmaps to the raw camera frame-0 world.",
    )
    parser.add_argument(
        "--display_corrected_smpl",
        action="store_true",
        help="Use corrected SMPL meshes as the viewer's main human meshes.",
    )
    parser.add_argument(
        "--show_gt_smpl_overlay",
        action="store_true",
        help="Overlay GT SMPL meshes in the same viewer coordinate system as the GT camera overlay.",
    )
    parser.add_argument(
        "--use_pose_dump_external_raw0",
        action="store_true",
        help=(
            "Use eval pose dump relative matrices for raw/GT/corrected cameras, "
            "anchored to the gray raw camera frame 0. This matches the camera-only "
            "V8.4 viewer coordinate system. If --external_raw_dir is omitted, the "
            "raw output generated by this script is used."
        ),
    )
    parser.add_argument(
        "--pose_dump_raw_source",
        choices=["payload", "dump"],
        default="payload",
        help=(
            "When --use_pose_dump_external_raw0 is set, keep gray raw cameras from "
            "the saved Human3R payload by default. Use 'dump' only for old camera-only "
            "debug views that intentionally visualized raw_c2w_rel from the eval npz."
        ),
    )
    return parser.parse_args()


def parse_raw_roots(text: str):
    if text is None or str(text).strip().lower() in {"", "none", "null"}:
        return None
    value = json.loads(text) if text.strip().startswith("{") else text
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return str(value)


def write_one_record_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep only dataset-facing fields plus metadata that the dataset ignores.
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def make_single_record_dataset(args: argparse.Namespace, record: dict, manifest_path: Path):
    subset = str(record.get("benchmark_subset", "test_aabb"))
    group = str(record.get("group", ""))
    pattern_seqs = record.get("seqs", None)
    is_pattern = pattern_seqs is not None and record.get("frames", None) is not None
    first_seq = str(pattern_seqs[0]) if is_pattern and pattern_seqs else str(record.get("seqA", ""))
    is_mvhuman = group.isdigit() or first_seq.split("/", 1)[0].isdigit()
    if is_mvhuman:
        split = "Training/mvhuman" if is_pattern or subset.startswith("train_sanity") else args.test_split
    else:
        split = "Training" if is_pattern or subset.startswith("train_sanity") else args.test_split
    common = dict(
        allow_repeat=True,
        split=split,
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=tuple(args.resolution),
        num_views=int(getattr(args, "num_views", 4)),
        seed=401,
        n_corres=0,
        manifest_path=str(manifest_path),
        load_da3_depth=False,
        raw_calibration_root=parse_raw_roots(args.raw_roots),
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )
    if is_pattern:
        return AvatarReX_Pattern(**common)
    if str(record.get("clip_type", "")).lower() == "aaaa" or subset.endswith("aaaa"):
        return AvatarReX_Video(**common)
    if is_mvhuman:
        common["pair_scope"] = "same_parent"
    return AvatarReX_AABB(**common)


def load_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list manifest: {path}")
    return data


def override_record_metrics_from_eval(record: dict, eval_dir: Path) -> dict:
    """Use metrics from the selected eval directory when available.

    The visual-random manifest is shared across checkpoints, so its metric
    fields may come from a different checkpoint. Match by benchmark subset/index
    and keep visualization-only fields such as rgb_paths from the input record.
    """
    subset = str(record.get("benchmark_subset", ""))
    index = record.get("benchmark_index", None)
    if not subset or index is None:
        return record
    path = eval_dir / f"{subset}.json"
    if not path.is_file():
        return record
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("rows", []):
        if int(row.get("benchmark_index", -1)) == int(index):
            merged = dict(record)
            merged.update(row)
            return merged
    return record


def case_name(record: dict) -> str:
    clip_type = str(record.get("clip_type", "clip")).lower()
    if record.get("pattern_id"):
        return f"{clip_type}_{record['pattern_id']}_{record.get('angle_bucket', 'pattern')}"
    index = int(record.get("benchmark_index", -1))
    group = str(record.get("group", "group"))
    bucket = str(record.get("angle_bucket", "same"))
    return f"{clip_type}_{index:04d}_{group}_{bucket}"


def clone_outputs_with_pose(outputs: dict, pose_key: str) -> dict:
    preds = []
    for pred in outputs["pred"]:
        cloned = dict(pred)
        if pose_key != "camera_pose":
            if pose_key not in pred:
                if pose_key == "v8_raw_camera_pose" and "camera_pose" in pred:
                    print(
                        "Model output has no v8_raw_camera_pose; "
                        "using camera_pose as the raw Human3R pose."
                    )
                    cloned["camera_pose"] = pred["camera_pose"]
                    preds.append(cloned)
                    continue
                raise KeyError(f"Model output does not contain {pose_key}; cannot build raw Human3R viewer payload.")
            cloned["camera_pose"] = pred[pose_key]
        preds.append(cloned)
    return {"views": list(outputs["views"]), "pred": preds}


def build_saved_outputs_demo(args: argparse.Namespace, record: dict, out_dir: Path) -> tuple[Path, Path]:
    raw_dir = out_dir / "raw_human3r"
    corrected_dir = out_dir / "corrected"
    needed = [
        raw_dir / "camera" / "000000.npz",
        corrected_dir / "camera" / "000000.npz",
        corrected_dir / "smpl" / "000000.npz",
    ]
    if args.reuse_saved and all(path.is_file() for path in needed):
        print(f"Reusing saved viewer payloads under {out_dir}")
        return raw_dir, corrected_dir

    img_paths = [str(Path(p)) for p in record.get("rgb_paths", [])]
    if len(img_paths) != 4:
        raise ValueError(f"Expected four rgb_paths in manifest record, got {len(img_paths)}")

    add_path_to_dust3r(str(args.model_path))
    print(f"Loading model: {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    if args.no_correction_before_view >= 0:
        model.v9_force_no_correction_before_view = int(args.no_correction_before_view)
    img_res = getattr(model, "mhmr_img_res", None)

    print("Preparing four image-only input frames:")
    for idx, path in enumerate(img_paths):
        print(f"  {idx}: {path}")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    if args.freeze_updates_from_view >= 0:
        for view_idx, view in enumerate(views):
            if view_idx < args.freeze_updates_from_view:
                continue
            update_mask = torch.zeros_like(view["img_mask"], dtype=torch.bool)
            view["update"] = update_mask
            view["update_state"] = update_mask
            view["update_mem"] = update_mask
            view["update_v8_history"] = update_mask

    print("Running image-only recurrent inference...")
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, args.device, use_ttt3r=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viewer_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saving corrected viewer payload: {corrected_dir}")
    corrected_outputs = clone_outputs_with_pose(outputs, "camera_pose")
    prepare_output(
        corrected_outputs,
        str(corrected_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=img_res,
        subsample=1,
    )

    print(f"Saving raw Human3R viewer payload: {raw_dir}")
    raw_outputs = clone_outputs_with_pose(outputs, "v8_raw_camera_pose")
    prepare_output(
        raw_outputs,
        str(raw_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=img_res,
        subsample=1,
    )
    return raw_dir, corrected_dir


def build_saved_outputs_dataloader(args: argparse.Namespace, record: dict, out_dir: Path) -> tuple[Path, Path]:
    raw_dir = out_dir / "raw_human3r"
    corrected_dir = out_dir / "corrected"
    needed = [
        raw_dir / "camera" / "000000.npz",
        corrected_dir / "camera" / "000000.npz",
        corrected_dir / "smpl" / "000000.npz",
    ]
    if args.reuse_saved and all(path.is_file() for path in needed):
        print(f"Reusing saved viewer payloads under {out_dir}")
        return raw_dir, corrected_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viewer_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    one_record_manifest = out_dir / "one_record_manifest.jsonl"
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

    print("Building the same dataloader sample used by the benchmark:")
    print(f"  subset={record.get('benchmark_subset')} index={record.get('benchmark_index')} clip={record.get('clip_type')}")
    print(f"  resolution={tuple(args.resolution)} split={args.test_split}")
    dataset = make_single_record_dataset(args, record, one_record_manifest)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    batch = todevice(batch, args.device)

    print("Running benchmark-matched image-only forward...")
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
    gate_values = []
    for pred in result["pred"]:
        gate = pred.get("v8_pose_prompt_gate", None)
        if gate is not None:
            gate_values.append(float(gate.detach().float().mean().cpu()))
    if gate_values:
        (out_dir / "viewer_forward_metrics.json").write_text(
            json.dumps({"v82_gate_mean": float(np.mean(gate_values))}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    outputs = todevice({"views": result["views"], "pred": result["pred"]}, "cpu")

    print(f"Saving corrected viewer payload: {corrected_dir}")
    corrected_outputs = clone_outputs_with_pose(outputs, "camera_pose")
    prepare_output(
        corrected_outputs,
        str(corrected_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=getattr(model, "mhmr_img_res", None),
        subsample=1,
    )

    print(f"Saving raw Human3R viewer payload: {raw_dir}")
    raw_outputs = clone_outputs_with_pose(outputs, "v8_raw_camera_pose")
    prepare_output(
        raw_outputs,
        str(raw_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=getattr(model, "mhmr_img_res", None),
        subsample=1,
    )
    return raw_dir, corrected_dir


def build_saved_outputs(args: argparse.Namespace, record: dict, out_dir: Path) -> tuple[Path, Path]:
    if args.input_mode == "demo":
        return build_saved_outputs_demo(args, record, out_dir)
    return build_saved_outputs_dataloader(args, record, out_dir)


def load_gt_aligned_cam_dict(record: dict, eval_dir: Path, raw_dir: Path, num_frames: int) -> dict[str, np.ndarray]:
    if "pose_npz" not in record:
        raise KeyError("Manifest record has no pose_npz; cannot load GT cameras for red overlay.")
    pose_data = np.load(eval_dir / record["pose_npz"])
    gt_abs = pose_data["gt_c2w_abs"].astype(np.float32)
    if gt_abs.shape[0] < num_frames:
        raise ValueError(f"GT pose count {gt_abs.shape[0]} is smaller than viewer frame count {num_frames}")

    raw0 = np.load(raw_dir / "camera" / "000000.npz")["pose"].astype(np.float32)
    gt0_inv = np.linalg.inv(gt_abs[0])
    gt_aligned = np.stack([(raw0 @ gt0_inv @ pose).astype(np.float32) for pose in gt_abs[:num_frames]], axis=0)

    focal, pp, R, t = [], [], [], []
    for i in range(num_frames):
        K = np.load(raw_dir / "camera" / f"{i:06d}.npz")["intrinsics"].astype(np.float32)
        focal.append(float(0.5 * (K[0, 0] + K[1, 1])))
        pp.append(K[:2, 2])
        R.append(gt_aligned[i, :3, :3])
        t.append(gt_aligned[i, :3, 3])
    return {
        "focal": np.asarray(focal, dtype=np.float32),
        "pp": np.asarray(pp, dtype=np.float32),
        "R": np.asarray(R, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
    }


def load_pose_dump_external_raw0_cam_dicts(
    record: dict,
    eval_dir: Path,
    raw_cam_dir: Path,
    num_frames: int,
    raw_source: str = "payload",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if "pose_npz" not in record:
        raise KeyError("Manifest record has no pose_npz; cannot load pose dump camera overlays.")
    pose_data = np.load(eval_dir / record["pose_npz"])
    gt_rel = pose_data["gt_c2w_rel"].astype(np.float32)
    raw_rel = pose_data["raw_c2w_rel"].astype(np.float32)
    corrected_rel = pose_data["corrected_c2w_rel"].astype(np.float32)
    if gt_rel.shape[0] < num_frames or raw_rel.shape[0] < num_frames or corrected_rel.shape[0] < num_frames:
        raise ValueError(
            f"Pose dump frame count is smaller than viewer frame count: "
            f"raw={raw_rel.shape[0]}, gt={gt_rel.shape[0]}, corrected={corrected_rel.shape[0]}, viewer={num_frames}"
        )

    anchor_cam_dict = load_cam_dict(raw_cam_dir, num_frames)
    raw0 = cam_pose(anchor_cam_dict, 0)
    if raw_source == "payload":
        raw_cam_dict = anchor_cam_dict
    elif raw_source == "dump":
        raw_abs = np.einsum("ij,njk->nik", raw0, raw_rel[:num_frames])
        raw_cam_dict = poses_to_cam_dict(raw_abs, raw_cam_dir)
    else:
        raise ValueError(f"Unsupported raw_source: {raw_source}")
    gt_abs = np.einsum("ij,njk->nik", raw0, gt_rel[:num_frames])
    corrected_abs = np.einsum("ij,njk->nik", raw0, corrected_rel[:num_frames])
    gt_cam_dict = poses_to_cam_dict(gt_abs, raw_cam_dir)
    corrected_cam_dict = poses_to_cam_dict(corrected_abs, raw_cam_dir)
    return raw_cam_dict, corrected_cam_dict, gt_cam_dict


def cam_pose(cam_dict: dict[str, np.ndarray], index: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = cam_dict["R"][index]
    pose[:3, 3] = cam_dict["t"][index]
    return pose


def poses_to_cam_dict(poses: np.ndarray, intrinsics_dir: Path) -> dict[str, np.ndarray]:
    focal, pp, R, t = [], [], [], []
    for i, pose in enumerate(poses):
        K = np.load(intrinsics_dir / "camera" / f"{i:06d}.npz")["intrinsics"].astype(np.float32)
        focal.append(float(0.5 * (K[0, 0] + K[1, 1])))
        pp.append(K[:2, 2])
        R.append(pose[:3, :3].astype(np.float32))
        t.append(pose[:3, 3].astype(np.float32))
    return {
        "focal": np.asarray(focal, dtype=np.float32),
        "pp": np.asarray(pp, dtype=np.float32),
        "R": np.asarray(R, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
    }


def transform_cam_dict(cam_dict: dict[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    R, t = [], []
    for i in range(len(cam_dict["R"])):
        pose = transform @ cam_pose(cam_dict, i)
        R.append(pose[:3, :3].astype(np.float32))
        t.append(pose[:3, 3].astype(np.float32))
    out = dict(cam_dict)
    out["R"] = np.asarray(R, dtype=np.float32)
    out["t"] = np.asarray(t, dtype=np.float32)
    return out


def transform_points_array(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3].astype(np.float32)
    trans = transform[:3, 3].astype(np.float32)
    return (points @ rot.T + trans).astype(np.float32)


def transform_pointclouds(pc_list: list[np.ndarray], transform: np.ndarray) -> list[np.ndarray]:
    return [transform_points_array(pc, transform) for pc in pc_list]


def transform_smpl_verts(verts_list: list[np.ndarray], transform: np.ndarray) -> list[np.ndarray]:
    out = []
    for verts in verts_list:
        if verts.size == 0:
            out.append(verts)
        else:
            out.append(transform_points_array(verts, transform))
    return out


def load_gt_smpl_verts_for_record(
    args: argparse.Namespace,
    record: dict,
    gt_cam_dict: dict[str, np.ndarray],
    num_frames: int,
) -> list[np.ndarray]:
    """Load official GT SMPL-X world meshes and align them to the viewer frame.

    The camera overlay may be anchored to a saved Human3R raw frame-0 camera.
    Therefore GT meshes are first generated in the dataset/raw world frame, then
    transformed by the same frame-0 alignment used by the red GT cameras.
    """
    one_record_manifest = args.case_root / case_name(record) / "one_record_manifest_for_gt_smpl.jsonl"
    write_one_record_manifest(one_record_manifest, record)
    dataset = make_single_record_dataset(args, record, one_record_manifest)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    if "raw_camera_pose" not in batch[0]:
        raise KeyError("GT SMPL overlay needs raw_camera_pose in the dataloader batch.")
    original_gt0 = batch[0]["raw_camera_pose"][0].detach().cpu().numpy().astype(np.float32)
    viewer_gt0 = cam_pose(gt_cam_dict, 0)
    world_to_viewer = viewer_gt0 @ np.linalg.inv(original_gt0)

    sample = dataset.samples[0]
    split = "Training" if str(record.get("benchmark_subset", "")).startswith("train_sanity") else args.test_split
    split_path = Path(args.data_root) / split
    clip_type = str(record.get("clip_type", "")).lower()
    if clip_type == "aaaa" or len(sample) == 2:
        seq_name, start_frame = sample
        view_specs = [(seq_name, int(start_frame) + i) for i in range(num_frames)]
    else:
        seq_a, seq_b, start_frame = sample
        start_frame = int(start_frame)
        view_specs = [
            (seq_a, start_frame),
            (seq_a, start_frame + 1),
            (seq_b, start_frame + 2),
            (seq_b, start_frame + 3),
        ][:num_frames]

    group = str(record.get("group", ""))
    is_mvhuman = group.isdigit() or str(record.get("seqA", "")).split("/", 1)[0].isdigit()
    if is_mvhuman and split == "Training":
        split = "Training/mvhuman"
        split_path = Path(args.data_root) / split

    smpl_model = SMPLModel(
        torch.device(args.device),
        model_args={"patch_size": 16, "mhmr_img_res": 896, "bb_patch_size": 14},
    )
    gt_verts = []
    with torch.no_grad():
        for seq_name, frame_idx in view_specs:
            smpl_path = split_path / seq_name / "smpl" / f"{int(frame_idx):08d}.pkl"
            with smpl_path.open("rb") as f:
                annots = pickle.load(f)
            if isinstance(annots, dict):
                annots = [annots]
            frame_verts = []
            for human in annots:
                shape_np = np.asarray(human.get("smplx_shape", np.zeros(11, dtype=np.float32)), dtype=np.float32).reshape(1, -1)
                body = np.asarray(human.get("smplx_body_pose", np.zeros((21, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 21 * 3)
                out = smpl_model.smplx_neutral_11(
                    global_orient=torch.as_tensor(
                        np.asarray(human.get("smplx_root_pose", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(1, 3),
                        device=args.device,
                    ),
                    body_pose=torch.as_tensor(body, device=args.device),
                    jaw_pose=torch.as_tensor(
                        np.asarray(human.get("smplx_jaw_pose", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 3),
                        device=args.device,
                    ),
                    leye_pose=torch.as_tensor(
                        np.asarray(human.get("smplx_leye_pose", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 3),
                        device=args.device,
                    ),
                    reye_pose=torch.as_tensor(
                        np.asarray(human.get("smplx_reye_pose", np.zeros((1, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 3),
                        device=args.device,
                    ),
                    left_hand_pose=torch.as_tensor(
                        np.asarray(human.get("smplx_left_hand_pose", np.zeros((15, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 15 * 3),
                        device=args.device,
                    ),
                    right_hand_pose=torch.as_tensor(
                        np.asarray(human.get("smplx_right_hand_pose", np.zeros((15, 3), dtype=np.float32)), dtype=np.float32).reshape(1, 15 * 3),
                        device=args.device,
                    ),
                    betas=torch.as_tensor(shape_np[:, :11], device=args.device),
                    transl=torch.as_tensor(
                        np.asarray(human.get("smplx_transl", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(1, 3),
                        device=args.device,
                    ),
                    expression=smpl_model.smplx_neutral_11.expression.repeat(1, 1),
                )
                frame_verts.append(out.vertices[0].detach().cpu().numpy().astype(np.float32))
            if frame_verts:
                verts = np.stack(frame_verts, axis=0).astype(np.float32)
                verts = transform_points_array(verts, world_to_viewer)
            else:
                verts = np.empty((0, 0, 3), dtype=np.float32)
            gt_verts.append(verts)
    print("GT SMPL overlay: official SMPL-X world meshes aligned by GT frame-0 camera.")
    return gt_verts


def transform_payload_between_camera_tracks(
    pts3ds: list[np.ndarray],
    verts: list[np.ndarray],
    source_cam_dict: dict[str, np.ndarray],
    target_cam_dict: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    out_pts, out_verts = [], []
    for i, pc in enumerate(pts3ds):
        transform = cam_pose(target_cam_dict, i) @ np.linalg.inv(cam_pose(source_cam_dict, i))
        out_pts.append(transform_points_array(pc, transform))
        if i < len(verts) and verts[i].size > 0:
            out_verts.append(transform_points_array(verts[i], transform))
        else:
            out_verts.append(verts[i])
    return out_pts, out_verts


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3] @ b[:3, :3].T
    angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def update_record_metrics_from_cam_dicts(
    record: dict,
    raw_cam_dict: dict[str, np.ndarray],
    corrected_cam_dict: dict[str, np.ndarray],
    gt_cam_dict: dict[str, np.ndarray],
) -> dict:
    raw_trans, corrected_trans, raw_rot, corrected_rot = [], [], [], []
    for i in range(len(gt_cam_dict["R"])):
        gt_pose = np.eye(4, dtype=np.float32)
        gt_pose[:3, :3] = gt_cam_dict["R"][i]
        gt_pose[:3, 3] = gt_cam_dict["t"][i]
        raw_pose = np.eye(4, dtype=np.float32)
        raw_pose[:3, :3] = raw_cam_dict["R"][i]
        raw_pose[:3, 3] = raw_cam_dict["t"][i]
        corrected_pose = np.eye(4, dtype=np.float32)
        corrected_pose[:3, :3] = corrected_cam_dict["R"][i]
        corrected_pose[:3, 3] = corrected_cam_dict["t"][i]
        raw_trans.append(float(np.linalg.norm(raw_pose[:3, 3] - gt_pose[:3, 3])))
        corrected_trans.append(float(np.linalg.norm(corrected_pose[:3, 3] - gt_pose[:3, 3])))
        raw_rot.append(rotation_error_deg(raw_pose, gt_pose))
        corrected_rot.append(rotation_error_deg(corrected_pose, gt_pose))
    out = dict(record)
    out["v82_raw_trans_err"] = float(np.mean(raw_trans))
    out["v82_trans_err"] = float(np.mean(corrected_trans))
    out["v82_raw_rot_err_deg"] = float(np.mean(raw_rot))
    out["v82_rot_err_deg"] = float(np.mean(corrected_rot))
    return out


def add_camera_set(
    viewer: SceneHumanViewer,
    cam_dict: dict[str, np.ndarray],
    color: tuple[int, int, int],
    prefix: str,
    y_offset: float,
    show_labels: bool,
) -> None:
    for step in range(len(cam_dict["R"])):
        focal = float(cam_dict["focal"][step])
        pp = cam_dict["pp"][step]
        R = cam_dict["R"][step]
        t = cam_dict["t"][step]
        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(float(pp[0]) / max(focal, 1e-6))
        aspect = float(pp[0]) / max(float(pp[1]), 1e-6)
        viewer.server.add_camera_frustum(
            name=f"/frames/{step}/{prefix}_camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=0.14,
            line_width=2.5,
            color=color,
        )
        if show_labels:
            viewer.server.scene.add_label(
                f"/frames/{step}/{prefix}_label",
                f"{prefix} {step}",
                position=t + np.asarray([0.0, y_offset, 0.0], dtype=np.float32),
                font_size_mode="scene",
                font_scene_height=0.055,
                depth_test=False,
            )


def make_title(record: dict) -> str:
    clip_type = str(record.get("clip_type", "clip")).upper()
    if clip_type == "AABB":
        source = f"{record.get('seqA')} -> {record.get('seqB')}"
        angle = f"{float(record.get('view_angle_deg', 0.0)):.1f} deg"
    else:
        source = str(record.get("seq"))
        angle = "same camera"
    return f"{clip_type} #{int(record.get('benchmark_index', -1)):04d} | {record.get('group')} | {record.get('angle_bucket', angle)} | {source}"


def add_title(viewer: SceneHumanViewer, record: dict, gt_cam_dict: dict[str, np.ndarray]) -> None:
    metrics = (
        f"trans {record.get('v82_raw_trans_err', float('nan')):.4f} -> {record.get('v82_trans_err', float('nan')):.4f}; "
        f"rot {record.get('v82_raw_rot_err_deg', float('nan')):.3f} -> {record.get('v82_rot_err_deg', float('nan')):.3f}; "
        f"gate mean {record.get('v82_gate_mean', float('nan')):.3f}"
    )
    anchor = gt_cam_dict["t"].mean(axis=0)
    viewer.server.scene.add_label(
        "/v8_4_pose_legend",
        f"{make_title(record)}\n{metrics}\nGT red | Human3R raw gray | corrected yellow",
        position=anchor + np.asarray([0.0, -0.45, 0.25], dtype=np.float32),
        font_size_mode="scene",
        font_scene_height=0.06,
        depth_test=False,
    )


def load_viewer_forward_metrics(case_dir: Path) -> dict:
    path = case_dir / "viewer_forward_metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.use_pose_dump_external_raw0 and args.external_scene_dir is not None and args.display_corrected_smpl:
        raise ValueError(
            "Invalid viewer combination: --external_scene_dir displays an external/raw scene payload, "
            "while --display_corrected_smpl replaces only the human mesh with the corrected payload. "
            "This mixes raw scene geometry with corrected human geometry and pose-dump camera overlays. "
            "For a self-consistent corrected scene, omit --external_scene_dir. For a camera-only overlay "
            "on the raw scene, omit --display_corrected_smpl."
        )
    records = load_manifest(args.manifest)
    if args.entry < 0 or args.entry >= len(records):
        raise IndexError(f"--entry {args.entry} out of range for {args.manifest} ({len(records)} entries)")
    record = override_record_metrics_from_eval(records[args.entry], args.eval_dir)
    case_dir = args.case_root / case_name(record)
    raw_dir, corrected_dir = build_saved_outputs(args, record, case_dir)

    if args.build_only:
        print(f"Built viewer case at {case_dir}")
        return

    scene_dir = args.external_scene_dir if args.external_scene_dir is not None else corrected_dir
    print(f"Viewer scene payload: {scene_dir}")
    num_frames = len(list((corrected_dir / "camera").glob("*.npz")))
    pts3ds, colors, confs, verts, faces, smpl_ids, msks = load_viewer_payload(scene_dir, num_frames, args.device)
    raw_cam_dir = args.external_raw_dir if args.external_raw_dir is not None else raw_dir
    if args.use_pose_dump_external_raw0:
        if args.pose_dump_raw_source == "payload":
            print(f"Gray raw camera overlay: saved Human3R payload cameras from {raw_cam_dir}")
        else:
            print(f"Gray raw camera overlay: pose dump raw_c2w_rel anchored to {raw_cam_dir}")
    else:
        print(f"Gray raw camera overlay: {raw_cam_dir}")
    print(f"Yellow corrected camera overlay: {corrected_dir}")
    source_scene_cam_dict = load_cam_dict(scene_dir, num_frames)
    source_corrected_cam_dict = load_cam_dict(corrected_dir, num_frames)
    raw_cam_dict = load_cam_dict(raw_cam_dir, num_frames)
    corrected_cam_dict = load_cam_dict(corrected_dir, num_frames)
    corrected_to_viewer = None

    if args.use_pose_dump_external_raw0:
        raw_cam_dict, corrected_cam_dict, gt_cam_dict = load_pose_dump_external_raw0_cam_dicts(
            record, args.eval_dir, raw_cam_dir, num_frames, raw_source=args.pose_dump_raw_source
        )
        if args.pose_dump_raw_source == "payload":
            print("Camera overlays use payload raw cameras plus pose dump GT/corrected relative matrices anchored to raw frame 0.")
        else:
            print("Camera overlays use pose dump raw/GT/corrected relative matrices anchored to gray raw frame 0.")
        if scene_dir == corrected_dir:
            pts3ds, verts = transform_payload_between_camera_tracks(
                pts3ds, verts, source_scene_cam_dict, corrected_cam_dict
            )
            print("Viewer pointcloud/SMPL payload transformed frame-wise to corrected pose-dump cameras.")
    elif args.align_corrected_to_raw0:
        corrected_to_viewer = cam_pose(raw_cam_dict, 0) @ np.linalg.inv(cam_pose(corrected_cam_dict, 0))
        corrected_cam_dict = transform_cam_dict(corrected_cam_dict, corrected_to_viewer)
        print("Aligned corrected output to gray raw camera frame 0.")
        if scene_dir == corrected_dir:
            pts3ds = transform_pointclouds(pts3ds, corrected_to_viewer)

    if args.display_corrected_smpl:
        _, _, _, corrected_verts, corrected_faces, corrected_smpl_ids, _ = load_viewer_payload(
            corrected_dir, num_frames, args.device
        )
        if args.use_pose_dump_external_raw0:
            _, corrected_verts = transform_payload_between_camera_tracks(
                [np.empty((0, 3), dtype=np.float32) for _ in range(num_frames)],
                corrected_verts,
                source_corrected_cam_dict,
                corrected_cam_dict,
            )
        elif corrected_to_viewer is not None:
            corrected_verts = transform_smpl_verts(corrected_verts, corrected_to_viewer)
        verts = corrected_verts
        faces = corrected_faces
        smpl_ids = corrected_smpl_ids
        print("Viewer SMPL meshes: corrected output.")
    else:
        print(f"Viewer SMPL meshes: {scene_dir}.")

    if not args.use_pose_dump_external_raw0:
        gt_cam_dict = load_gt_aligned_cam_dict(record, args.eval_dir, raw_cam_dir, num_frames)
    record = update_record_metrics_from_cam_dicts(record, raw_cam_dict, corrected_cam_dict, gt_cam_dict)
    record.update(load_viewer_forward_metrics(case_dir))
    gt_smpl_verts = None
    if args.show_gt_smpl_overlay:
        gt_smpl_verts = load_gt_smpl_verts_for_record(args, record, gt_cam_dict, num_frames)

    print("Launching full SceneHumanViewer.")
    print("Color legend: GT camera/body=red, Human3R raw=gray, corrected=yellow.")
    print(f"Open http://127.0.0.1:{args.port} after forwarding this port.")
    viewer = SceneHumanViewer(
        pts3ds,
        colors,
        confs,
        corrected_cam_dict,
        verts,
        faces,
        smpl_ids,
        msks,
        gt_smpl_verts=gt_smpl_verts,
        device=args.device,
        port=args.port,
        edge_color_list=[None] * len(pts3ds),
        show_camera=False,
        show_gt_camera=False,
        show_gt_smpl=args.show_gt_smpl_overlay,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    add_camera_set(viewer, gt_cam_dict, color=(255, 40, 40), prefix="GT", y_offset=0.08, show_labels=args.show_labels)
    add_camera_set(viewer, raw_cam_dict, color=(150, 150, 150), prefix="raw", y_offset=0.02, show_labels=args.show_labels)
    add_camera_set(viewer, corrected_cam_dict, color=(255, 220, 0), prefix="corr", y_offset=-0.06, show_labels=args.show_labels)
    if args.show_labels:
        add_title(viewer, record, gt_cam_dict)
    viewer.run()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
