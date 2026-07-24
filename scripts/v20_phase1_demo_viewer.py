#!/usr/bin/env python3
"""Launch one demo.py-style viewer for a V20 Phase 1 case and method."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from scripts.v20_phase1_gt_id_multihuman_consensus import (  # noqa: E402
    IDENTITIES,
    VIDEO_NAMES,
    evaluate_case,
    full_intrinsics,
    gt_w2c,
    load_obj_vertices,
    mesh_path,
    reassign_cache_gt_identities,
    transform_points,
)
from viser_utils import SceneHumanViewer  # noqa: E402


METHOD_KEYS = {
    "confidence": "single_highest_confidence",
    "oracle": "oracle_best_single",
    "multi": "naive_mean",
    "multi_corrected": "layout_select_one_reject",
}
METHOD_TITLES = {
    "confidence": "Single Highest Confidence (deployable)",
    "oracle": "Single Oracle Best (GT upper bound)",
    "multi": "GT-ID Multi Naive Mean (best fusion)",
    "multi_corrected": "GT-ID Multi Layout/Reject",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=ROOT / "output/v20_phase1_gt_id_multihuman_consensus",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument("--case", default="three_t0900_c0_c3_k0")
    parser.add_argument("--method", choices=tuple(METHOD_KEYS), required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def frame_image(args: argparse.Namespace, camera: int, frame: int) -> np.ndarray:
    cached = args.result_dir / "input_frames" / f"cam{camera}" / f"{frame:06d}.jpg"
    if cached.is_file():
        image = cv2.imread(str(cached), cv2.IMREAD_COLOR)
    else:
        video = (
            args.data_root
            / "three_original_video/three_new"
            / VIDEO_NAMES[camera]
        )
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            raise RuntimeError(f"Cannot decode camera {camera}, frame {frame}")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cloud_colors(
    cloud_world: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    camera = transform_points(np.linalg.inv(pose), cloud_world)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-5)
    uv = np.zeros((len(camera), 2), dtype=np.float64)
    uv[valid] = camera[valid, :2] / camera[valid, 2:3]
    uv[valid] = uv[valid] @ intrinsics[:2, :2].T + intrinsics[:2, 2]
    pixels = np.rint(uv).astype(np.int64)
    valid &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image.shape[0])
    )
    colors = np.full((len(camera), 3), 150, dtype=np.uint8)
    colors[valid] = image[pixels[valid, 1], pixels[valid, 0]]
    return colors


def annotate_identity_frame(
    image: np.ndarray, humans: dict, assignment: dict
) -> np.ndarray:
    header_height = 72
    header = np.full((header_height, image.shape[1], 3), 24, dtype=np.uint8)
    canvas = np.concatenate([header, image.copy()], axis=0)
    colors = {
        "person0": (40, 210, 80),
        "person1": (70, 130, 255),
        "person2": (255, 155, 45),
    }
    ordered = sorted(
        humans.items(),
        key=lambda item: 0.5 * (item[1]["bbox"][0] + item[1]["bbox"][2]),
    )
    left_rank = {
        int(human["detection_index"]): rank
        for rank, (_, human) in enumerate(ordered)
    }
    predicted_order = "  ".join(
        f"L{rank}:D{int(human['detection_index'])}->P{identity[-1]}"
        for rank, (identity, human) in enumerate(ordered)
    )
    gt_order = sorted(
        assignment.get("gt_bboxes", {}),
        key=lambda identity: 0.5
        * (
            assignment["gt_bboxes"][identity][0]
            + assignment["gt_bboxes"][identity][2]
        ),
    )
    cv2.putText(
        canvas, f"PRED  {predicted_order}", (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.51, (245, 245, 245), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "GT LEFT ORDER  " + "  ".join(f"P{x[-1]}" for x in gt_order),
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.51,
        (190, 190, 190), 1, cv2.LINE_AA,
    )
    for identity, human in humans.items():
        box = np.rint(human["bbox"]).astype(int)
        box[[1, 3]] += header_height
        color = colors[identity]
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 3)
        text = (
            f"D{int(human['detection_index'])} "
            f"L{left_rank[int(human['detection_index'])]} -> P{identity[-1]}"
        )
        cv2.putText(
            canvas, text, (box[0], max(box[1] + 22, 24)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA,
        )
    for identity, value in assignment.get("gt_bboxes", {}).items():
        box = np.rint(value).astype(int)
        box[[1, 3]] += header_height
        color = colors[identity]
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 1)
        cv2.putText(
            canvas, f"GT P{identity[-1]}",
            (box[0], min(box[3] - 5, canvas.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )
    return canvas


def save_identity_audit(
    args: argparse.Namespace, images: list[np.ndarray], cache: dict
) -> Path:
    selected = [len(images) - 2, len(images) - 1]
    annotated = [
        annotate_identity_frame(
            images[index], cache["humans"][index], cache["assignment"][index]
        )
        for index in selected
    ]
    for label, image in zip(("pre", "post"), annotated):
        cv2.putText(
            image, label.upper(), (12, image.shape[0] - 17), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), 2, cv2.LINE_AA,
        )
    output_dir = args.result_dir / "identity_audit" / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"pre_post_{args.method}.png"
    combined = np.concatenate(annotated, axis=1)
    cv2.imwrite(str(output), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return output


def load_payload(args: argparse.Namespace) -> tuple[dict, dict]:
    report = json.loads(
        (args.result_dir / "v20_phase1_gtid_v2_offsets_0_1_2_4_8.json").read_text(
            encoding="utf-8"
        )
    )
    case = next(
        row for row in report["cases"] if row["case"]["key"] == args.case
    )
    cache = torch.load(
        args.result_dir / "case_cache" / f"{args.case}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return case, cache


def left_order_corrected_cache(cache: dict) -> dict:
    """Diagnostic association using predicted and GT left-to-right order."""
    output = copy.copy(cache)
    output["humans"] = list(cache["humans"])
    post_humans = cache["humans"][-1]
    assignment = cache["assignment"][-1]
    detections = sorted(
        post_humans.values(),
        key=lambda row: 0.5 * (row["bbox"][0] + row["bbox"][2]),
    )
    gt_order = sorted(
        IDENTITIES,
        key=lambda identity: 0.5
        * (
            assignment["gt_bboxes"][identity][0]
            + assignment["gt_bboxes"][identity][2]
        ),
    )
    corrected = {}
    for identity, detection in zip(gt_order, detections):
        row = dict(detection)
        row["identity"] = identity
        corrected[identity] = row
    output["humans"][-1] = corrected
    return output


def identity_order_markdown(cache: dict) -> str:
    rows = []
    for frame_index in (len(cache["humans"]) - 2, len(cache["humans"]) - 1):
        ordered = sorted(
            cache["humans"][frame_index].items(),
            key=lambda item: 0.5 * (item[1]["bbox"][0] + item[1]["bbox"][2]),
        )
        values = " | ".join(
            f"L{rank}: D{int(human['detection_index'])} -> P{identity[-1]}"
            for rank, (identity, human) in enumerate(ordered)
        )
        rows.append(("Pre-cut" if frame_index == len(cache["humans"]) - 2 else "Post-cut", values))
    return "\n".join(f"**{label}:** `{values}`  " for label, values in rows)


def main() -> None:
    args = parse_args()
    case, cache = load_payload(args)
    cache = reassign_cache_gt_identities(
        SimpleNamespace(data_root=args.data_root, size=512), cache
    )
    method_key = METHOD_KEYS[args.method]
    method = evaluate_case(cache)["methods"][method_key]
    boundary = np.asarray(method["boundary"], dtype=np.float64)

    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False,
        person_center="head",
    ).to(torch.device("cpu")).eval()
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    del layer

    spec = cache["case"]
    frames = [int(value) for value in spec["pre_frames"]] + [int(spec["post_frame"])]
    cameras = [int(spec["source_camera"])] * len(spec["pre_frames"]) + [
        int(spec["target_camera"])
    ]
    geometry_args = SimpleNamespace(data_root=args.data_root)
    images = [
        frame_image(args, camera, frame)
        for camera, frame in zip(cameras, frames)
    ]
    identity_audit_path = save_identity_audit(args, images, cache)
    intrinsics = []
    for camera in cameras:
        intrinsic = full_intrinsics(geometry_args, camera).copy()
        intrinsic[:2] *= 512.0 / 2048.0
        intrinsics.append(intrinsic)

    local_colors = [
        cloud_colors(
            np.asarray(cloud, dtype=np.float64),
            np.asarray(pose, dtype=np.float64),
            intrinsic,
            image,
        )
        for cloud, pose, intrinsic, image in zip(
            cache["clouds"], cache["poses"], intrinsics, images
        )
    ]

    poses, clouds, vertices, smpl_ids = [], [], [], []
    predicted_labels = []
    bounds = []
    for index, (pose, cloud, humans) in enumerate(
        zip(cache["poses"], cache["clouds"], cache["humans"])
    ):
        transform = boundary if index == len(frames) - 1 else np.eye(4)
        world_pose = transform @ np.asarray(pose, dtype=np.float64)
        world_cloud = transform_points(transform, np.asarray(cloud, dtype=np.float64))
        poses.append(world_pose)
        clouds.append(world_cloud.astype(np.float32))
        if len(world_cloud):
            bounds.append(world_cloud)
        frame_vertices = []
        frame_ids = []
        frame_labels = []
        ordered = sorted(
            humans.items(),
            key=lambda item: 0.5
            * (item[1]["bbox"][0] + item[1]["bbox"][2]),
        )
        left_rank = {
            int(human["detection_index"]): rank
            for rank, (_, human) in enumerate(ordered)
        }
        for identity_index, identity in enumerate(IDENTITIES):
            if identity not in humans:
                continue
            human = humans[identity]
            value = transform_points(
                transform, np.asarray(human["vertices"])
            )
            frame_vertices.append(value.astype(np.float32))
            frame_ids.append(identity_index)
            bounds.append(value[::100])
            root = transform_points(
                transform, np.asarray(human["root"])[None]
            )[0]
            frame_labels.append(
                (
                    identity,
                    int(human["detection_index"]),
                    left_rank[int(human["detection_index"])],
                    root,
                )
            )
        vertices.append(
            np.stack(frame_vertices)
            if frame_vertices
            else np.empty((0, 0, 3), dtype=np.float32)
        )
        smpl_ids.append(np.asarray(frame_ids, dtype=np.int64))
        predicted_labels.append(frame_labels)

    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    gt_poses, gt_vertices, gt_labels = [], [], []
    for index, (camera, frame) in enumerate(zip(cameras, frames)):
        gt_poses.append(gauge @ np.linalg.inv(gt_w2c(geometry_args, camera, frame)))
        frame_gt = []
        frame_gt_labels = []
        for identity in IDENTITIES:
            value = transform_points(
                gauge, load_obj_vertices(mesh_path(geometry_args, identity, frame))
            )
            frame_gt.append(value.astype(np.float32))
            bounds.append(value[::100])
            position = np.mean(value, axis=0)
            position[1] = np.min(value[:, 1]) - 0.08
            frame_gt_labels.append((identity, position))
        gt_vertices.append(
            np.stack(frame_gt)
            if frame_gt
            else np.empty((0, 0, 3), dtype=np.float32)
        )
        gt_labels.append(frame_gt_labels)

    poses = np.stack(poses)
    gt_poses = np.stack(gt_poses)
    cam_dict = {
        "focal": np.asarray([value[0, 0] for value in intrinsics]),
        "pp": np.asarray([value[:2, 2] for value in intrinsics]),
        "R": poses[:, :3, :3],
        "t": poses[:, :3, 3],
    }
    gt_cam_dict = {
        "focal": cam_dict["focal"],
        "pp": cam_dict["pp"],
        "R": gt_poses[:, :3, :3],
        "t": gt_poses[:, :3, 3],
    }
    color_list = [color[None] for color in local_colors]
    confidence_list = [
        np.full((1, len(cloud)), 2.0, dtype=np.float32) for cloud in clouds
    ]
    masks = [None] * len(clouds)

    viewer = SceneHumanViewer(
        clouds,
        color_list,
        confidence_list,
        cam_dict,
        vertices,
        faces,
        smpl_ids,
        masks,
        gt_cam_dict=gt_cam_dict,
        gt_smpl_verts=gt_vertices,
        device="cpu",
        port=int(args.port),
        edge_color_list=[None] * len(clouds),
        show_camera=True,
        show_gt_camera=True,
        show_gt_smpl=True,
        vis_threshold=1.0,
        msk_threshold=0.1,
        mask_morph=0,
        size=512,
        downsample_factor=1,
        smpl_downsample_factor=1,
        camera_downsample_factor=1,
        initial_timestep=len(frames) - 1,
    )
    title = METHOD_TITLES[args.method]
    viewer.server.gui.add_markdown(
        f"## {title}\n"
        f"**Case:** `{args.case}`  \n"
        f"**Camera T:** {method['camera_translation_error_m']:.3f} m  \n"
        f"**Camera R:** {method['camera_rotation_error_deg']:.2f} deg  \n"
        f"**Composite:** {method['camera_composite']:.3f}  \n"
        "Frame 4 is pre-cut cam0; frame 5 is post-cut cam3. "
        "Red wireframe and gray camera are GT references."
    )
    viewer.server.gui.add_markdown(
        identity_order_markdown(cache)
        + "\n`D` = raw detection index, `L` = image left-to-right rank, "
        "`P` = assigned GT person identity."
    )
    viewer.server.gui.add_markdown(f"**Identity audit:** `{identity_audit_path}`")
    for index, frame_labels in enumerate(predicted_labels):
        for identity, detection, rank, position in frame_labels:
            viewer.server.scene.add_label(
                f"/frames/{index}/pred_label_{identity}",
                text=f"Pred P{identity[-1]} | D{detection} | L{rank}",
                position=np.asarray(position) + np.array([0.0, -0.18, 0.0]),
                font_screen_scale=0.75,
                depth_test=False,
            )
        for identity, position in gt_labels[index]:
            viewer.server.scene.add_label(
                f"/frames/{index}/gt_label_{identity}",
                text=f"GT P{identity[-1]}",
                position=position,
                font_screen_scale=0.70,
                depth_test=False,
            )

    all_bounds = np.concatenate(bounds, axis=0)
    lower, upper = np.percentile(all_bounds, (2.0, 98.0), axis=0)
    center = 0.5 * (lower + upper)
    extent = max(float(np.linalg.norm(upper - lower)), 1.5)

    @viewer.server.on_client_connect
    def _fit(client) -> None:
        def apply() -> None:
            client.camera.position = center + extent * np.array([0.46, -0.30, 0.46])
            client.camera.up_direction = np.array([0.0, -1.0, 0.0])
            client.camera.look_at = center

        threading.Timer(1.0, apply).start()

    print(
        f">> {title}: http://127.0.0.1:{args.port}",
        flush=True,
    )
    viewer.run()


if __name__ == "__main__":
    main()
