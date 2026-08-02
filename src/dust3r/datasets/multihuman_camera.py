#!/usr/bin/env python3
"""Camera-only first-cut supervision from MultiHuman Real-World Capture.

This adapter intentionally supplies *only* synchronized RGB frames and the
official per-frame camera calibration.  The public extracted MultiHuman
release contains GT SMPL-X meshes but no native SMPL-X parameter targets that
can be passed safely into the Human3R training loss.  Pretending otherwise
would quietly mix incompatible supervision.  The all-false ``smpl_mask`` is
therefore a deliberate contract: this dataset trains the shadow camera
proposal, while the normal AvatarReX/THuman/MVHuman batches retain their
existing camera and human supervision.

Each manifest event is a strictly causal three-view pattern::

    (camera A, t-1), (camera A, t), (camera B, t)
      shot labels:       0              0              1

The input RGB is decoded from the original calibrated six-camera videos, not
from the person-specific 512px crops stored alongside the meshes.  This makes
training geometry match the full-frame streaming inference path.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import PIL.Image

from dust3r.datasets.avatarrex import _empty_depthmap, _resize_crop_like_human3r_demo
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.datasets.utils.transforms import ImgNorm


def _read_manifest(path: str | Path, split: str) -> tuple[dict, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get(split)
    if not isinstance(records, list) or not records:
        raise ValueError(f"MultiHuman camera manifest {path} has no nonempty '{split}' list")
    normalized = []
    for index, record in enumerate(records):
        required = ("sequence", "pre_camera", "post_camera", "frame")
        if not isinstance(record, dict) or any(key not in record for key in required):
            raise ValueError(f"invalid camera-only MultiHuman event {index}: {record!r}")
        sequence = str(record["sequence"])
        pre_camera = int(record["pre_camera"])
        post_camera = int(record["post_camera"])
        frame = int(record["frame"])
        if pre_camera == post_camera or min(pre_camera, post_camera) < 0 or max(pre_camera, post_camera) > 5:
            raise ValueError(f"invalid camera pair in event {index}: {record!r}")
        normalized.append(
            {
                "event_id": str(record.get("event_id", f"{sequence}_{pre_camera}_{post_camera}_{frame}")),
                "sequence": sequence,
                "pre_camera": pre_camera,
                "post_camera": post_camera,
                "frame": frame,
            }
        )
    return tuple(normalized)


def _w2c_to_c2w(value: np.ndarray) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3] = np.asarray(value, dtype=np.float64)
    return np.linalg.inv(world_to_camera).astype(np.float32)


class MultiHumanCameraCut(BaseMultiViewDataset):
    """Full-frame, camera-only MultiHuman cut events defined by a JSON manifest."""

    def __init__(
        self,
        *args,
        ROOT: str | Path,
        manifest_path: str | Path,
        manifest_split: str = "train",
        num_views: int = 3,
        resolution: int | tuple[int, int] = 512,
        transform=ImgNorm,
        aug_crop: int = 0,
        allow_repeat: bool = True,
        seed: int | None = None,
        n_corres: int = 0,
        max_humans: int = 1,
        **kwargs,
    ):
        if int(num_views) != 3:
            raise ValueError("MultiHumanCameraCut requires exactly three views: A(t-1), A(t), B(t)")
        self.ROOT = Path(ROOT)
        self.manifest_path = Path(manifest_path)
        self.manifest_split = str(manifest_split)
        self.max_humans = int(max_humans)
        self.samples = _read_manifest(self.manifest_path, self.manifest_split)
        self._video_paths: dict[str, tuple[Path, ...]] = {}
        self._intrinsics: dict[str, dict[int, np.ndarray]] = {}
        self._captures: dict[tuple[str, int], cv2.VideoCapture] = {}
        self._validate_sources()
        # ``scenes`` controls the base Dataset length.  Samples are explicit
        # events rather than random temporal windows, so one scene per event.
        self.scenes = list(range(len(self.samples)))
        self.is_metric = True
        super().__init__(
            *args,
            num_views=3,
            split=self.manifest_split,
            resolution=resolution,
            transform=transform,
            aug_crop=aug_crop,
            allow_repeat=allow_repeat,
            seed=seed,
            n_corres=n_corres,
            **kwargs,
        )

    def _sequence_root(self, sequence: str) -> Path:
        return self.ROOT / sequence

    def _validate_sources(self) -> None:
        for sequence in sorted({sample["sequence"] for sample in self.samples}):
            original_root = self.ROOT / f"{sequence}_original_video"
            video_root = original_root / f"{sequence}_new"
            videos = tuple(sorted(video_root.glob("*.mp4")))
            calibration_path = original_root / "calibration_new.json"
            if len(videos) != 6 or not calibration_path.is_file():
                raise FileNotFoundError(
                    f"expected six original videos and calibration under {original_root}; "
                    f"found videos={len(videos)}, calibration={calibration_path.is_file()}"
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            intrinsics = {}
            for camera in range(6):
                value = calibration.get(str(camera), {}).get("K")
                if value is None:
                    raise KeyError(f"missing K for {sequence} camera {camera} in {calibration_path}")
                intrinsics[camera] = np.asarray(value, dtype=np.float32).reshape(3, 3)
            self._video_paths[sequence] = videos
            self._intrinsics[sequence] = intrinsics

        for sample in self.samples:
            sequence = sample["sequence"]
            frame = int(sample["frame"])
            parameter = self._sequence_root(sequence) / sequence / "person0" / "parameter"
            for timestamp, camera in ((frame - 1, sample["pre_camera"]), (frame, sample["pre_camera"]), (frame, sample["post_camera"])):
                path = parameter / str(timestamp) / f"{camera}_extrinsic.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"missing calibration required by {sample['event_id']}: {path}")

    def get_stats(self):
        return f"{len(self.samples)} full-frame camera-only cut events ({self.manifest_split})"

    def _capture(self, sequence: str, camera: int) -> cv2.VideoCapture:
        key = (sequence, int(camera))
        capture = self._captures.get(key)
        if capture is None or not capture.isOpened():
            capture = cv2.VideoCapture(str(self._video_paths[sequence][int(camera)]))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open MultiHuman video {self._video_paths[sequence][int(camera)]}")
            self._captures[key] = capture
        return capture

    def _read_rgb(self, sequence: str, camera: int, frame: int) -> np.ndarray:
        capture = self._capture(sequence, camera)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f"cannot decode sequence={sequence}, camera={camera}, frame={frame}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (2048, 2048):
            raise ValueError(
                f"expected a 2048x2048 original frame, got {image.shape[:2]} "
                f"for sequence={sequence}, camera={camera}, frame={frame}"
            )
        return image

    def _camera_pose(self, sequence: str, camera: int, frame: int) -> np.ndarray:
        path = (
            self._sequence_root(sequence)
            / sequence
            / "person0"
            / "parameter"
            / str(int(frame))
            / f"{int(camera)}_extrinsic.npy"
        )
        return _w2c_to_c2w(np.load(path))

    def _load_view(
        self,
        sample: dict,
        camera: int,
        frame: int,
        shot_label: int,
        resolution,
        rng,
        view_index: int,
    ) -> dict:
        sequence = sample["sequence"]
        rgb = self._read_rgb(sequence, int(camera), int(frame))
        depth = _empty_depthmap(rgb.shape)
        intrinsics = self._intrinsics[sequence][int(camera)].copy()
        # This is deliberately the same resize/crop transform used by
        # ``prepare_full_square_input(..., square_ok=True)`` at streaming
        # inference: full 2048 square frame -> full 512 square frame.
        image, depth, intrinsics = _resize_crop_like_human3r_demo(
            PIL.Image.fromarray(rgb), depth, None, intrinsics, resolution, square_ok=True
        )
        img_mask, ray_mask = self.get_img_and_ray_masks(
            self.is_metric, view_index, rng, p=[0.85, 0.0, 0.15]
        )
        pose = self._camera_pose(sequence, int(camera), int(frame))
        return {
            "img": image,
            "msk": False,
            "depthmap": depth,
            "camera_pose": pose,
            "raw_camera_pose": pose.copy(),
            "camera_intrinsics": intrinsics.astype(np.float32),
            "dataset": "MultiHumanCameraCut",
            "label": f"{sample['event_id']}_c{camera}_f{frame}",
            "instance": f"{sequence}/camera{camera}/frame{frame}",
            "is_metric": self.is_metric,
            "is_video": False,
            "quantile": np.array(1, dtype=np.float32),
            "img_mask": img_mask,
            "ray_mask": ray_mask,
            "camera_only": True,
            "depth_only": False,
            "single_view": False,
            "reset": False,
            "shot_label": np.array(int(shot_label), dtype=np.int64),
            # The camera-only supervision contract.  SMPLModel sees an empty
            # batch and therefore supplies no human target/loss for this view.
            "smpl_mask": np.zeros((self.max_humans,), dtype=np.bool_),
        }

    def _get_views(self, idx, resolution, rng, num_views):
        if int(num_views) != 3:
            raise AssertionError(f"requested {num_views} views from a three-view cut dataset")
        sample = self.samples[int(idx)]
        frame = int(sample["frame"])
        pre = int(sample["pre_camera"])
        post = int(sample["post_camera"])
        return [
            self._load_view(sample, pre, frame - 1, 0, resolution, rng, 0),
            self._load_view(sample, pre, frame, 0, resolution, rng, 1),
            self._load_view(sample, post, frame, 1, resolution, rng, 2),
        ]

    def __del__(self):
        for capture in getattr(self, "_captures", {}).values():
            try:
                capture.release()
            except Exception:
                pass
