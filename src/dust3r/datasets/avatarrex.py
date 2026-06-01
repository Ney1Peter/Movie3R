#!/usr/bin/env python3
"""
AvatarReX Dataset for Human3R

包含两种采样模式：
1. AvatarReX_Video：正常连续相机运动采样
   - 从同一相机的连续帧中采样（t, t+1, t+2, t+3）
   - is_video=True，学习正常相机运动

2. AvatarReX_AABB：AABB 镜头跳变采样
   - 帧0,1 来自相机A的t/t+1，帧2,3来自相机B的t/t+1
   - is_video=False，学习镜头跳变

目录结构（由 preprocess_avatarrex_fast.py 生成）：
  ROOT/Training/{seq_id}/
    rgb/{frame:08d}.png    ← PNG（扁平结构，无 camera 子目录）
    cam/{frame:08d}.npz    ← pose(4,4) + intrinsics(3,3)
    smpl/{frame:08d}.pkl   ← SMPLX参数
    depth/{frame:08d}.npy  ← 深度图（uint16 mm）
    mask/{frame:08d}.png   ← 前景遮罩
"""

import os
import os.path as osp
import json
import numpy as np
import pickle
from tqdm import tqdm

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import imread_cv2


def _empty_depthmap(image_shape):
    h, w = image_shape[:2]
    return np.zeros((h, w), dtype=np.float32)


def _load_depthmap_meters(depth_path, image_shape):
    """Load DA3 depth as float32 meters, supporting legacy uint16 millimeters."""
    if not osp.exists(depth_path):
        return _empty_depthmap(image_shape)

    depth_raw = np.load(depth_path)
    depthmap = depth_raw.astype(np.float32)
    # Legacy Movie3R-dataset outputs saved DA3 depths as uint16 millimeters.
    if np.issubdtype(depth_raw.dtype, np.integer):
        depthmap = depthmap / 1000.0
    depthmap[~np.isfinite(depthmap)] = 0.0
    depthmap[depthmap > 200.0] = 0.0
    return depthmap


def _avatarrex_has_required_frame_files(split_path, seq_name, frame_idx, require_depth=True):
    frame_str = f"{int(frame_idx):08d}"
    seq_path = _avatarrex_scene_path(split_path, seq_name)
    required_paths = [
        osp.join(seq_path, "rgb", f"{frame_str}.png"),
        osp.join(seq_path, "cam", f"{frame_str}.npz"),
        osp.join(seq_path, "smpl", f"{frame_str}.pkl"),
    ]
    if require_depth:
        required_paths.append(osp.join(seq_path, "depth", f"{frame_str}.npy"))
    return all(osp.isfile(path) for path in required_paths)


def _avatarrex_load_camera_pose(split_path, seq_name, frame_idx):
    frame_str = f"{int(frame_idx):08d}"
    cam_path = osp.join(_avatarrex_scene_path(split_path, seq_name), "cam", f"{frame_str}.npz")
    cam = np.load(cam_path)
    return cam["pose"].astype(np.float32)


def _avatarrex_scene_path(split_path, seq_name):
    return osp.join(split_path, *str(seq_name).split("/"))


def _avatarrex_is_sequence_dir(path):
    return (
        osp.isdir(osp.join(path, "rgb"))
        and osp.isdir(osp.join(path, "cam"))
        and osp.isdir(osp.join(path, "smpl"))
    )


def _avatarrex_discover_sequences(split_path):
    """Return sequence names relative to split_path.

    Supports both layouts:
      split/22010708/{rgb,cam,smpl}
      split/lbn1/22010708/{rgb,cam,smpl}
    """
    direct = []
    grouped = []
    for name in sorted(os.listdir(split_path)):
        path = osp.join(split_path, name)
        if not osp.isdir(path):
            continue
        if _avatarrex_is_sequence_dir(path):
            direct.append(name)
            continue
        for child in sorted(os.listdir(path)):
            child_path = osp.join(path, child)
            if osp.isdir(child_path) and _avatarrex_is_sequence_dir(child_path):
                grouped.append(f"{name}/{child}")
    return direct if direct else grouped


def _avatarrex_frame_ids(split_path, seq_name):
    rgb_dir = osp.join(_avatarrex_scene_path(split_path, seq_name), "rgb")
    frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    return [int(osp.splitext(f)[0]) for f in frames]


def _avatarrex_camera_view_direction(camera_pose):
    # Dataset camera_pose is c2w. The camera z-axis is enough for view-pair angle
    # filtering; using +z or -z gives the same pair angle when used consistently.
    direction = np.asarray(camera_pose[:3, 2], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return direction / norm


def _avatarrex_camera_angle_deg(pose_a, pose_b):
    dir_a = _avatarrex_camera_view_direction(pose_a)
    dir_b = _avatarrex_camera_view_direction(pose_b)
    cos_angle = float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _load_avatarrex_raw_calibration(raw_calibration_root):
    if raw_calibration_root is None:
        return None
    if isinstance(raw_calibration_root, dict):
        grouped = {}
        for group, root in raw_calibration_root.items():
            calibration_path = osp.join(str(root), "calibration_full.json")
            if not osp.isfile(calibration_path):
                raise FileNotFoundError(f"AvatarReX raw calibration not found: {calibration_path}")
            with open(calibration_path, "r", encoding="utf-8") as f:
                grouped[str(group)] = json.load(f)
        return {"__grouped_avatarrex_calibration__": True, "groups": grouped}
    calibration_path = osp.join(str(raw_calibration_root), "calibration_full.json")
    if not osp.isfile(calibration_path):
        raise FileNotFoundError(f"AvatarReX raw calibration not found: {calibration_path}")
    with open(calibration_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _raw_calibration_c2w(calibration, seq_name):
    """Return raw AvatarReX c2w from calibration convention X_cam = R @ X_world + T."""
    if calibration is None:
        return None
    if isinstance(calibration, dict) and calibration.get("__grouped_avatarrex_calibration__"):
        parts = str(seq_name).split("/", 1)
        if len(parts) != 2:
            raise KeyError(
                f"Grouped AvatarReX raw calibration requires seq_name like 'group/seq', got {seq_name}"
            )
        group, seq_key = parts
        groups = calibration["groups"]
        if group not in groups:
            raise KeyError(f"{group} not found in grouped AvatarReX raw calibration")
        calibration = groups[group]
        seq_name = seq_key
    if seq_name not in calibration:
        raise KeyError(f"{seq_name} not found in raw AvatarReX calibration")
    cal = calibration[seq_name]
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_w2c.T
    pose[:3, 3] = -R_w2c.T @ T_w2c
    return pose


def _avatarrex_read_sample_manifest(manifest_path):
    if manifest_path is None:
        return None
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return tuple()
    if text[0] == "[":
        parsed = json.loads(text)
        records.extend(parsed)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    samples = []
    for record in records:
        if isinstance(record, (list, tuple)) and len(record) == 3:
            seq_a, seq_b, start_frame = record
        else:
            seq_a = record.get("seqA", record.get("seq_a"))
            seq_b = record.get("seqB", record.get("seq_b"))
            start_frame = record.get("start_frame", record.get("frame", record.get("t")))
        if seq_a is None or seq_b is None or start_frame is None:
            raise ValueError(f"Invalid AvatarReX AABB manifest record: {record}")
        samples.append((str(seq_a), str(seq_b), int(start_frame)))
    return tuple(samples)


def _empty_anchor_info(top_k=16):
    return dict(
        anchor_valid=np.array(False, dtype=np.bool_),
        anchor_ref_view_idx=np.array(1, dtype=np.int64),
        anchor_cur_view_idx=np.array(2, dtype=np.int64),
        anchor_ref_patch_idx=np.zeros((top_k,), dtype=np.int64),
        anchor_cur_patch_idx=np.zeros((top_k,), dtype=np.int64),
        anchor_ref_pos_norm=np.zeros((top_k, 2), dtype=np.float32),
        anchor_cur_pos_norm=np.zeros((top_k, 2), dtype=np.float32),
        anchor_local_residual_norm=np.zeros((top_k, 2), dtype=np.float32),
        anchor_confidence=np.zeros((top_k,), dtype=np.float32),
        anchor_quality_gate=np.zeros((1,), dtype=np.float32),
        anchor_mask=np.zeros((top_k,), dtype=np.bool_),
    )


def _pad_anchor_array(array, shape, dtype):
    out = np.zeros(shape, dtype=dtype)
    if array is None:
        return out
    arr = np.asarray(array, dtype=dtype)
    n = min(shape[0], arr.shape[0])
    if n > 0:
        out[:n] = arr[:n]
    return out


def _compute_anchor_local_residual(ref_pos_norm, cur_pos_norm, affine_inverse):
    if affine_inverse is None:
        return ref_pos_norm - cur_pos_norm
    cur_h = np.concatenate(
        [cur_pos_norm, np.ones((cur_pos_norm.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    base_ref = cur_h @ affine_inverse.astype(np.float32).T
    return ref_pos_norm - base_ref.astype(np.float32)


class AvatarReX_AABB(BaseMultiViewDataset):
    """
    AvatarReX AABB 镜头跳变数据集。

    数据来源：preprocess_avatarrex.py 转换后的 BEDLAM 格式数据。
    同一 sequence 内的 16 个相机对应同一个人在同一时刻的不同视角，
    因此 (seq, camA, t) 和 (seq, camB, t) 的 SMPL 参数完全相同
    （同一人的同一时刻），满足 AABB 采样的前提。

    AABB 采样逻辑：
      1. 从同一 sequence 内选两个不同相机 camA ≠ camB
      2. 从同一时间轴上选 t 和 t+1 两帧
      3. 组成 4 帧样本：(camA,t), (camA,t+1), (camB,t), (camB,t+1)

    每个 sequence 的样本数 = C(16,2) × 2 × (frames_per_seq - 1)
                          = 240 × 1999 ≈ 480,000 样本/sequence
    """

    def __init__(
        self,
        *args,
        split="Training",
        ROOT=None,
        num_views=4,
        resolution=(512, 288),
        transform=ImgNorm,
        aug_crop=16,
        allow_repeat=False,
        seed=None,
        anchor_cache_root=None,
        anchor_cache_only=False,
        anchor_top_k=16,
        anchor_quality_threshold=0.0,
        fixed_samples=None,
        manifest_path=None,
        min_view_angle_deg=None,
        max_view_angle_deg=None,
        max_samples=None,
        pair_strategy="all",
        load_da3_depth=True,
        raw_calibration_root=None,
        **kwargs,
    ):
        assert ROOT is not None, "AvatarReX_AABB requires ROOT"
        self.ROOT = ROOT
        self.split = split
        self.is_metric = True
        self.max_interval = 1           # AABB 固定间隔1
        self.max_humans = 10
        self.anchor_cache_root = anchor_cache_root
        self.anchor_cache_only = anchor_cache_only
        self.anchor_top_k = anchor_top_k
        self.anchor_quality_threshold = anchor_quality_threshold
        manifest_samples = _avatarrex_read_sample_manifest(manifest_path)
        if fixed_samples is not None and manifest_samples is not None:
            raise ValueError("Use either fixed_samples or manifest_path, not both.")
        self.fixed_samples = self._normalize_fixed_samples(
            fixed_samples if fixed_samples is not None else manifest_samples
        )
        self.manifest_path = manifest_path
        self.min_view_angle_deg = min_view_angle_deg
        self.max_view_angle_deg = max_view_angle_deg
        self.max_samples = None if max_samples is None else int(max_samples)
        self.pair_strategy = str(pair_strategy)
        self.load_da3_depth = bool(load_da3_depth)
        self.raw_calibration_root = raw_calibration_root
        self.raw_calibration = _load_avatarrex_raw_calibration(raw_calibration_root)
        self.sample_view_angles = {}
        self.anchor_cache_index = {}
        self.smpl_key2shape = {
            "smplx_root_pose": (1, 3),
            "smplx_body_pose": (21, 3),
            "smplx_jaw_pose": (1, 3),
            "smplx_leye_pose": (1, 3),
            "smplx_reye_pose": (1, 3),
            "smplx_left_hand_pose": (15, 3),
            "smplx_right_hand_pose": (15, 3),
            "smplx_shape": (11,),
            "smplx_transl": (3,),
            "smplx_gender_id": (),
        }

        super().__init__(
            *args,
            num_views=num_views,
            split=split,
            resolution=resolution,
            transform=transform,
            aug_crop=aug_crop,
            allow_repeat=allow_repeat,
            seed=seed,
            **kwargs,
        )

        self._load_index()

    def _load_index(self):
        """
        构建 AABB 样本索引。

        设计说明：
        每个 avatarrex "序列" 对应 1 台相机。16 台相机 = 16 个序列目录。
        所有 16 个序列共享同一套 SMPL 参数（同一人的同一 motion，
        只是相机视角不同）。
        AABB 采样：跨序列选取两个不同相机 A 和 B，在同一时刻 t，
        A 的 (t,t+1) 帧与 B 的 (t,t+1) 帧组成 4 帧样本。

        self.samples: list of (seqA_name, seqB_name, t)
            → 对应样本 views = [
                (seqA_name, cam=0000, t),
                (seqA_name, cam=0000, t+1),
                (seqB_name, cam=0000, t),
                (seqB_name, cam=0000, t+1),
            ]
        """
        seq_dir = osp.join(self.ROOT, self.split)
        if not osp.exists(seq_dir):
            raise FileNotFoundError(f"AvatarReX data not found at {seq_dir}")

        self.scenes = _avatarrex_discover_sequences(seq_dir)
        if not self.scenes:
            raise FileNotFoundError(f"No AvatarReX sequence directories found under {seq_dir}")

        # 每个序列只有 1 个相机（cam_id=0000）
        self.seq_cams = {s: [0] for s in self.scenes}

        # **========== 原始代码：假设帧号从 0 连续开始 ==========**
        # # 获取帧数（所有序列帧数相同）
        # # 预处理脚本输出为扁平结构：rgb/{frame:08d}.png（无 camera 子目录）
        # sample_seq = self.scenes[0]
        # rgb_dir = osp.join(seq_dir, sample_seq, "rgb")
        # frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        # self.num_frames = len(frames)
        # self.seq_frames = {s: self.num_frames for s in self.scenes}
        # **========== 新代码：使用真实文件名帧号，例如 00000005 起始 ==========**
        self.scene_frame_ids = {s: _avatarrex_frame_ids(seq_dir, s) for s in self.scenes}
        sample_seq = self.scenes[0]
        self.frame_ids = self.scene_frame_ids[sample_seq]
        self.frame_to_pos = {frame_id: pos for pos, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)
        self.seq_frames = {s: len(ids) for s, ids in self.scene_frame_ids.items()}
        # **========== 结束 ==========**

        print(f"  {len(self.scenes)} sequences, sample sequence has {self.num_frames} frames")
        if not self.load_da3_depth:
            print("  AvatarReX_AABB: DA3 depth disabled; depthmap is zero-filled for pose-only training")
        if self.raw_calibration is not None:
            print(f"  AvatarReX_AABB: raw calibration camera pose enabled from {self.raw_calibration_root}")
        print(f"  Building AABB index...")

        # AABB: 两两跨序列组合（允许同序列但这里只有不同序列才有效）
        # seqA, seqB 来自不同的序列目录（不同相机）
        # t ∈ [0, num_frames-4]（需要 t, t+1, t+2, t+3 均有效）
        if self.fixed_samples:
            self.samples = list(self.fixed_samples)
            print(f"  AvatarReX_AABB manifest/fixed samples: {len(self.samples):,}")
        else:
            self.samples = []
            for i, seqA in enumerate(self.scenes):
                for j, seqB in enumerate(self.scenes):
                    if i == j:
                        continue  # 跳过同一相机
                    # **========== 原始代码：样本保存从 0 开始的位置索引 ==========**
                    # for t in range(self.num_frames - 3):
                    #     self.samples.append((seqA, seqB, t))
                    # **========== 新代码：使用每个 seqA 真实帧号，便于和 cache/start_frame 对齐 ==========**
                    for frame_id in self.scene_frame_ids[seqA]:
                        if frame_id + 3 <= self.scene_frame_ids[seqA][-1]:
                            self.samples.append((seqA, seqB, frame_id))
                    # **========== 结束 ==========**

            print(f"  AvatarReX_AABB: {len(self.samples):,} candidate samples "
                  f"from {len(self.scenes)} sequence folders")

        # **========== 原始代码：索引阶段不检查文件完整性，缺帧会在 DataLoader 读图时报错 ==========**
        # self.samples 保持上方构建结果，不做文件存在性过滤。
        # **========== 新代码：跳过 rgb/cam/depth/smpl 不完整的 AABB sample ==========**
        before_file_filter = len(self.samples)
        self.samples = [
            sample for sample in self.samples
            if self._sample_has_required_files(seq_dir, *sample)
        ]
        skipped = before_file_filter - len(self.samples)
        if skipped > 0:
            print(f"  AvatarReX_AABB skipped incomplete samples: {skipped:,}/{before_file_filter:,}")
        # **========== 结束 ==========**

        self._apply_view_angle_filter(seq_dir)

        if self.anchor_cache_root:
            self._load_anchor_cache_index()
            if self.anchor_cache_only:
                before = len(self.samples)
                self.samples = [s for s in self.samples if s in self.anchor_cache_index]
                print(f"  AvatarReX_AABB anchor cache-only: {len(self.samples):,}/{before:,} samples")

        # **========== V6.1 overfit 原始代码备份：不支持指定单个 AABB sample ==========**
        # Overfit 只能依赖 `N @ AvatarReX_AABB(...)` 从已有 samples 中取前 N 个样本，
        # 不能显式指定 seqA/seqB/start_frame，因此训练样本和后续可视化视频不够可控。
        # **========== 结束 ==========**
        # **========== V6.1 overfit 新代码：允许显式指定一个或多个 AABB sample ==========**
        if self.fixed_samples:
            missing = [sample for sample in self.fixed_samples if sample not in self.samples]
            if missing:
                raise ValueError(
                    "AvatarReX_AABB fixed_samples not found after file/cache filtering: "
                    f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
                )
            print(f"  AvatarReX_AABB fixed_samples: {len(self.samples):,} valid samples")
        # **========== 结束 ==========**

        self._apply_max_samples()

    @staticmethod
    def _normalize_fixed_samples(fixed_samples):
        if fixed_samples is None:
            return None
        if isinstance(fixed_samples, tuple) and len(fixed_samples) == 3:
            fixed_samples = [fixed_samples]

        normalized = []
        for sample in fixed_samples:
            if len(sample) != 3:
                raise ValueError(
                    "Each AvatarReX_AABB fixed sample must be "
                    "(seqA_name, seqB_name, start_frame)"
                )
            seq_a, seq_b, start_frame = sample
            normalized.append((str(seq_a), str(seq_b), int(start_frame)))
        return tuple(normalized)

    def _pair_angle_deg(self, split_path, seqA_name, seqB_name):
        key = (seqA_name, seqB_name)
        if key in self.sample_view_angles:
            return self.sample_view_angles[key]
        common_frames = sorted(
            set(self.scene_frame_ids.get(seqA_name, []))
            .intersection(self.scene_frame_ids.get(seqB_name, []))
        )
        if not common_frames:
            raise ValueError(f"No common frame ids for {seqA_name} and {seqB_name}")
        frame_id = common_frames[0]
        pose_a = _avatarrex_load_camera_pose(split_path, seqA_name, frame_id)
        pose_b = _avatarrex_load_camera_pose(split_path, seqB_name, frame_id)
        angle = _avatarrex_camera_angle_deg(pose_a, pose_b)
        self.sample_view_angles[key] = angle
        return angle

    def _apply_view_angle_filter(self, split_path):
        if self.min_view_angle_deg is None and self.max_view_angle_deg is None:
            return

        min_angle = -np.inf if self.min_view_angle_deg is None else float(self.min_view_angle_deg)
        max_angle = np.inf if self.max_view_angle_deg is None else float(self.max_view_angle_deg)
        before = len(self.samples)
        filtered = []
        for sample in self.samples:
            seqA_name, seqB_name, _ = sample
            angle = self._pair_angle_deg(split_path, seqA_name, seqB_name)
            if min_angle <= angle <= max_angle:
                filtered.append(sample)
        self.samples = filtered
        print(
            "  AvatarReX_AABB view-angle filter: "
            f"{len(self.samples):,}/{before:,} samples "
            f"({min_angle:.1f} <= angle <= {max_angle:.1f})"
        )

        if self.pair_strategy == "all":
            return
        if self.pair_strategy == "top_angle":
            self.samples.sort(
                key=lambda s: self._pair_angle_deg(split_path, s[0], s[1]),
                reverse=True,
            )
            return
        if self.pair_strategy == "fixed":
            return
        raise ValueError(
            "AvatarReX_AABB pair_strategy must be one of "
            f"'all', 'top_angle', 'fixed', got {self.pair_strategy!r}"
        )

    def _apply_max_samples(self):
        if self.max_samples is None:
            return
        before = len(self.samples)
        self.samples = self.samples[: self.max_samples]
        print(f"  AvatarReX_AABB max_samples: {len(self.samples):,}/{before:,} samples")

    def get_sample_metadata(self, idx):
        seqA_name, seqB_name, start_frame = self.samples[idx]
        split_path = osp.join(self.ROOT, self.split)
        frames = [int(start_frame) + offset for offset in range(4)]
        return {
            "seqA": seqA_name,
            "seqB": seqB_name,
            "start_frame": int(start_frame),
            "frames": frames,
            "view_angle_deg": self._pair_angle_deg(split_path, seqA_name, seqB_name),
        }

    def _sample_has_required_files(self, split_path, seqA_name, seqB_name, start_frame):
        t = int(start_frame)
        t1 = t + 1
        t2 = t + 2
        t3 = t + 3
        return (
            _avatarrex_has_required_frame_files(split_path, seqA_name, t, require_depth=self.load_da3_depth)
            and _avatarrex_has_required_frame_files(split_path, seqA_name, t1, require_depth=self.load_da3_depth)
            and _avatarrex_has_required_frame_files(split_path, seqB_name, t2, require_depth=self.load_da3_depth)
            and _avatarrex_has_required_frame_files(split_path, seqB_name, t3, require_depth=self.load_da3_depth)
        )

    def _load_anchor_cache_index(self):
        manifest_path = osp.join(self.anchor_cache_root, "manifest.jsonl")
        if not osp.isfile(manifest_path):
            raise FileNotFoundError(f"Anchor cache manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "ok":
                    continue
                if float(record.get("quality_gate", 0.0)) < self.anchor_quality_threshold:
                    continue
                boundary = record["boundary"]
                key = (boundary["ref_seq"], boundary["cur_seq"], int(record["start_frame"]))
                cache_path = record["cache_path"]
                if not osp.isabs(cache_path):
                    cache_path = osp.join(self.anchor_cache_root, cache_path)
                self.anchor_cache_index[key] = cache_path

        print(f"  Loaded anchor cache entries: {len(self.anchor_cache_index):,} from {manifest_path}")

    def _load_anchor_info(self, seqA_name, seqB_name, start_frame):
        key = (seqA_name, seqB_name, int(start_frame))
        cache_path = self.anchor_cache_index.get(key)
        if not cache_path:
            return _empty_anchor_info(self.anchor_top_k)

        data = np.load(cache_path)
        top_k = self.anchor_top_k
        n = min(top_k, int(data["ref_patch_idx"].shape[0]))
        ref_pos = _pad_anchor_array(data["ref_pos_norm"], (top_k, 2), np.float32)
        cur_pos = _pad_anchor_array(data["cur_pos_norm"], (top_k, 2), np.float32)
        residual = np.zeros((top_k, 2), dtype=np.float32)
        if n > 0:
            residual[:n] = _compute_anchor_local_residual(
                data["ref_pos_norm"][:n].astype(np.float32),
                data["cur_pos_norm"][:n].astype(np.float32),
                data["affine_inverse"],
            )

        info = _empty_anchor_info(top_k)
        info.update(
            anchor_valid=np.array(n > 0, dtype=np.bool_),
            anchor_ref_patch_idx=_pad_anchor_array(data["ref_patch_idx"], (top_k,), np.int64),
            anchor_cur_patch_idx=_pad_anchor_array(data["cur_patch_idx"], (top_k,), np.int64),
            anchor_ref_pos_norm=ref_pos,
            anchor_cur_pos_norm=cur_pos,
            anchor_local_residual_norm=residual,
            anchor_confidence=_pad_anchor_array(data["confidence"], (top_k,), np.float32),
            anchor_quality_gate=np.asarray(data["quality_gate"], dtype=np.float32).reshape(1),
            anchor_mask=np.arange(top_k) < n,
        )
        return info

    def __len__(self):
        return len(self.samples)

    def get_image_num(self):
        return sum(self.seq_frames.values())

    def _get_views(self, idx, resolution, rng, num_views):
        assert num_views == 4, "AABB dataset only supports num_views=4"

        seqA_name, seqB_name, t = self.samples[idx]
        # **========== 原始代码：假设真实帧号连续等于 t/t+1/t+2/t+3 ==========**
        # t1 = t + 1
        # t2 = t + 2
        # t3 = t + 3
        # **========== 新代码：使用 manifest/索引中的真实起始帧号 ==========**
        t = int(t)
        t1 = t + 1
        t2 = t + 2
        t3 = t + 3
        # **========== 结束 ==========**
        cam = 0  # 每个序列只有 1 个相机，ID=0

        split_path = osp.join(self.ROOT, self.split)

        # SMPL 来自 motion 序列（所有相机同一时刻的 SMPL 相同）
        # 时间连续：t, t+1, t+2, t+3 对应 motion 的连续帧
        annots_t  = self._load_smpl(split_path, seqA_name, cam, t)
        annots_t1 = self._load_smpl(split_path, seqA_name, cam, t1)
        annots_t2 = self._load_smpl(split_path, seqA_name, cam, t2)
        annots_t3 = self._load_smpl(split_path, seqA_name, cam, t3)

        # shot_label: frame i-1 → frame i 是否发生 shot change
        # view 0: 0 (first frame, 无 previous)
        # view 1: 0 (seqA → seqA, 相机连续)
        # view 2: 1 (seqA → seqB, 相机跳变)
        # view 3: 0 (seqB → seqB, 相机连续)
        shot_labels = [0, 0, 1, 0]

        views = []
        view_specs = [
            (seqA_name, cam, t,  annots_t,  shot_labels[0]),   # view 0: 相机A @ t
            (seqA_name, cam, t1, annots_t1, shot_labels[1]),  # view 1: 相机A @ t+1
            (seqB_name, cam, t2, annots_t2, shot_labels[2]),  # view 2: 相机B @ t+2（跳变后）
            (seqB_name, cam, t3, annots_t3, shot_labels[3]),  # view 3: 相机B @ t+3
        ]

        boundary_anchor_info = self._load_anchor_info(seqA_name, seqB_name, t)
        view_angle_deg = np.array(
            self._pair_angle_deg(split_path, seqA_name, seqB_name), dtype=np.float32
        )
        for v, (seq_name, cam_id, frame_idx, annots, shot_label) in enumerate(view_specs):
            view = self._load_view(
                split_path, seq_name, cam_id, frame_idx, annots,
                resolution, rng, v, shot_label,
            )
            view["aabb_view_angle_deg"] = view_angle_deg
            view.update(boundary_anchor_info if v == 2 else _empty_anchor_info(self.anchor_top_k))
            views.append(view)

        assert len(views) == num_views
        return views

    def _load_smpl(self, split_path, seq_name, cam_id, frame_idx):
        """加载 SMPL 参数。"""
        # fast 脚本输出为扁平结构：smpl/{frame:08d}.pkl（无 camera 子目录）
        smpl_path = osp.join(
            _avatarrex_scene_path(split_path, seq_name), "smpl",
            f"{frame_idx:08d}.pkl"
        )
        annots = []
        if osp.isfile(smpl_path):
            with open(smpl_path, "rb") as f:
                annots = pickle.load(f)
        return annots

    def _load_view(self, split_path, seq_name, cam_id, frame_idx, annots,
                   resolution, rng, v, shot_label=0):
        """加载单个 view 的所有数据。"""
        frame_str = f"{frame_idx:08d}"  # 原始文件格式: 00000000.png

        # fast 脚本输出为扁平结构：rgb/{frame:08d}.png, cam/{frame:08d}.npz（无 camera 子目录）
        seq_path = _avatarrex_scene_path(split_path, seq_name)
        rgb_path = osp.join(seq_path, "rgb", f"{frame_str}.png")
        cam_path = osp.join(seq_path, "cam", f"{frame_str}.npz")
        depth_path = osp.join(seq_path, "depth", f"{frame_str}.npy")

        rgb_image = imread_cv2(rgb_path)

        # Camera params
        cam = np.load(cam_path)
        camera_pose = cam["pose"].astype(np.float32)
        intrinsics = cam["intrinsics"].astype(np.float32)
        raw_camera_pose = _raw_calibration_c2w(self.raw_calibration, seq_name)

        # **========== 原始代码：直接按 float 米单位读取，导致旧 uint16 毫米数据被 >200 阈值清零 ==========**
        # if osp.exists(depth_path):
        #     depthmap = np.load(depth_path).astype(np.float32)
        #     depthmap[~np.isfinite(depthmap)] = 0
        #     depthmap[depthmap > 200.0] = 0.0
        # else:
        #     h, w = rgb_image.shape[:2]
        #     depthmap = np.zeros((h, w), dtype=np.float32)
        # **========== 新代码：兼容旧 uint16 毫米和新 float32 米单位 ==========**
        # V8.1 pose-prompt experiments can disable DA3 pseudo-depth completely:
        # the pose loss does not use GT depth, and Human3R's predicted pointmap
        # is the reconstruction cue we want to inspect.
        if self.load_da3_depth:
            depthmap = _load_depthmap_meters(depth_path, rgb_image.shape)
        else:
            depthmap = _empty_depthmap(rgb_image.shape)
        # **========== 结束 ==========**

        # Mask（可能不存在）
        mask_path = osp.join(seq_path, "mask", f"{frame_str}.png")
        if osp.exists(mask_path):
            mask_image = imread_cv2(mask_path)
        else:
            mask_image = None

        # 图像预处理（crop/resize）
        if mask_image is not None:
            rgb_image, depthmap, mask_image, intrinsics = \
                self._crop_resize_if_necessary_mask(
                    rgb_image, depthmap, mask_image, intrinsics,
                    resolution, rng=rng, info=f"{seq_name}/{cam_id}/{frame_idx}"
                )
        else:
            rgb_image, depthmap, intrinsics = \
                self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics,
                    resolution, rng=rng, info=f"{seq_name}/{cam_id}/{frame_idx}"
                )

        # -------------------------------------------------------------------------
        # smplx_transl 坐标系修复：
        # 预处理脚本保存的 smplx_transl 是 mocap 世界坐标，
        # 但过滤/排序时错误地用 mocap Z (> 0.01) 判断"人在相机前方"。
        # 实际上需要变换到相机坐标系再判断和排序。
        # camera_pose (c2w) = [R | -R @ (T - person_transl)]，已用 person_transl 调整。
        # 逆变换：smpl_cam = R_c2w.T @ (smpl_world - t_c2w)
        # -------------------------------------------------------------------------
        R_c2w = camera_pose[:3, :3]
        t_c2w = camera_pose[:3, 3]

        humans_with_cam_z = []
        for h in annots:
            smpl_world = np.array(h.get("smplx_transl", [0, 0, 100]), dtype=np.float32)
            smpl_cam = R_c2w.T @ (smpl_world - t_c2w)   # 变换到相机坐标系
            h = dict(h)  # 复制，避免修改原始数据
            h["_smplx_transl_cam"] = smpl_cam
            h["_smplx_transl_cam_z"] = smpl_cam[2]
            humans_with_cam_z.append(h)

        # 按相机坐标系的 Z 值排序（人在相机前方 Z > 0）
        if humans_with_cam_z:
            l_dist = [hh["_smplx_transl_cam_z"] for hh in humans_with_cam_z]
            order = sorted(range(len(l_dist)), key=lambda i: l_dist[i])
            humans_with_cam_z = [humans_with_cam_z[i] for i in order]

        # 过滤：人在相机前方即可（相机坐标系 Z > -0.5，留足容差）
        # 注意：原来错误的 mocap Z > 0.01 条件已废弃
        humans = [hh for hh in humans_with_cam_z if hh["_smplx_transl_cam_z"] > -0.5]

        smpl_mask = np.zeros(self.max_humans, dtype=np.bool_)
        if len(humans) > 0:
            smpl_mask[:len(humans)] = True

        smpl_dict = {}
        for k, shape in self.smpl_key2shape.items():
            smpl_dict[k] = np.zeros(
                (self.max_humans, *shape), dtype=np.float32
            )
            if len(humans) > 0:
                for h in range(len(humans)):
                    val = humans[h].get(k, np.zeros(shape))
                    if isinstance(val, np.ndarray):
                        val = val.astype(np.float32)
                        # 预处理脚本保存时将多维数组展平，加载时需reshape回原始形状
                        if len(shape) > 1:
                            val = val.reshape(shape)
                        smpl_dict[k][h] = val
                    else:
                        smpl_dict[k][h] = float(val)
            # smplx_transl 使用变换后的相机坐标系值
            if k == "smplx_transl":
                for h in range(len(humans)):
                    smpl_dict[k][h] = humans[h]["_smplx_transl_cam"]

        # img/ray mask
        img_mask, ray_mask = self.get_img_and_ray_masks(
            self.is_metric, v, rng, p=[0.85, 0.00, 0.15]
        )

        return dict(
            img=rgb_image,
            msk=False if mask_image is None else mask_image,
            depthmap=depthmap,
            camera_pose=camera_pose,
            **({} if raw_camera_pose is None else {"raw_camera_pose": raw_camera_pose}),
            camera_intrinsics=intrinsics,
            dataset="AvatarReX_AABB",
            label=f"{seq_name}_{frame_str}",
            instance=rgb_path,
            is_metric=self.is_metric,
            is_video=False,          # AABB 不是时序连续
            quantile=np.array(1, dtype=np.float32),
            img_mask=img_mask,
            ray_mask=ray_mask,
            camera_only=False,
            depth_only=False,
            single_view=False,
            reset=False,
            shot_label=shot_label,   # 0=连续, 1=相机跳变
            smpl_mask=smpl_mask,
            **smpl_dict,
        )


class AvatarReX_Video(BaseMultiViewDataset):
    """
    AvatarReX 正常连续相机运动数据集。

    采样模式：同一相机内的连续帧 (t, t+1, t+2, t+3)
    - is_video=True，学习正常相机运动
    - 与 AABB 模式互补

    每个 sequence 的样本数 ≈ frames_per_seq - num_views + 1
    """

    def __init__(
        self,
        *args,
        split="Training",
        ROOT=None,
        num_views=4,
        resolution=(512, 288),
        transform=ImgNorm,
        aug_crop=16,
        allow_repeat=False,
        seed=None,
        anchor_top_k=16,
        load_da3_depth=True,
        raw_calibration_root=None,
        **kwargs,
    ):
        assert ROOT is not None, "AvatarReX_Video requires ROOT"
        self.ROOT = ROOT
        self.split = split
        self.is_metric = True
        self.max_interval = 4
        self.max_humans = 10
        self.anchor_top_k = anchor_top_k
        self.load_da3_depth = bool(load_da3_depth)
        self.raw_calibration_root = raw_calibration_root
        self.raw_calibration = _load_avatarrex_raw_calibration(raw_calibration_root)
        self.smpl_key2shape = {
            "smplx_root_pose": (1, 3),
            "smplx_body_pose": (21, 3),
            "smplx_jaw_pose": (1, 3),
            "smplx_leye_pose": (1, 3),
            "smplx_reye_pose": (1, 3),
            "smplx_left_hand_pose": (15, 3),
            "smplx_right_hand_pose": (15, 3),
            "smplx_shape": (11,),
            "smplx_transl": (3,),
            "smplx_gender_id": (),
        }

        super().__init__(
            *args,
            num_views=num_views,
            split=split,
            resolution=resolution,
            transform=transform,
            aug_crop=aug_crop,
            allow_repeat=allow_repeat,
            seed=seed,
            **kwargs,
        )

        self._load_index()

    def _load_index(self):
        """构建 Video 采样索引。"""
        seq_dir = osp.join(self.ROOT, self.split)
        if not osp.exists(seq_dir):
            raise FileNotFoundError(f"AvatarReX data not found at {seq_dir}")

        self.scenes = sorted([
            d for d in os.listdir(seq_dir)
            if osp.isdir(osp.join(seq_dir, d))
        ])

        # 每个序列只有 1 个相机（cam_id=0000）
        self.seq_cams = {s: [0] for s in self.scenes}

        # **========== 原始代码：假设帧号从 0 连续开始 ==========**
        # # 获取帧数
        # sample_seq = self.scenes[0]
        # rgb_dir = osp.join(seq_dir, sample_seq, "rgb")
        # frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        # self.num_frames = len(frames)
        # self.seq_frames = {s: self.num_frames for s in self.scenes}
        # **========== 新代码：使用真实文件名帧号，例如 00000005 起始 ==========**
        sample_seq = self.scenes[0]
        rgb_dir = osp.join(seq_dir, sample_seq, "rgb")
        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        self.frame_ids = [int(osp.splitext(f)[0]) for f in frames]
        self.frame_to_pos = {frame_id: pos for pos, frame_id in enumerate(self.frame_ids)}
        self.num_frames = len(self.frame_ids)
        self.seq_frames = {s: self.num_frames for s in self.scenes}
        # **========== 结束 ==========**

        print(f"  AvatarReX_Video: {len(self.scenes)} sequences, "
              f"{self.num_frames} frames each, frames {self.frame_ids[0]}-{self.frame_ids[-1]}")
        if not self.load_da3_depth:
            print("  AvatarReX_Video: DA3 depth disabled; depthmap is zero-filled for pose-only training")
        if self.raw_calibration is not None:
            print(f"  AvatarReX_Video: raw calibration camera pose enabled from {self.raw_calibration_root}")

        # 构建索引：每个 scene 的每个有效起始位置
        self.samples = []
        for seq_idx, seq_name in enumerate(self.scenes):
            # 可起始位置：[0, num_frames - num_views]
            # **========== 原始代码：样本保存从 0 开始的位置索引 ==========**
            # for t in range(self.num_frames - self.num_views + 1):
            #     self.samples.append((seq_name, t))
            # **========== 新代码：样本保存真实起始帧号 ==========**
            for start_pos in range(self.num_frames - self.num_views + 1):
                self.samples.append((seq_name, self.frame_ids[start_pos]))
            # **========== 结束 ==========**

        # **========== 原始代码：索引阶段不检查文件完整性，缺帧会在 DataLoader 读图时报错 ==========**
        # self.samples 保持上方构建结果，不做文件存在性过滤。
        # **========== 新代码：跳过 rgb/cam/depth/smpl 不完整的 Video sample ==========**
        before_file_filter = len(self.samples)
        self.samples = [
            sample for sample in self.samples
            if self._sample_has_required_files(seq_dir, *sample)
        ]
        skipped = before_file_filter - len(self.samples)
        if skipped > 0:
            print(f"  AvatarReX_Video skipped incomplete samples: {skipped:,}/{before_file_filter:,}")
        # **========== 结束 ==========**

        print(f"  AvatarReX_Video: {len(self.samples):,} samples")

    def _sample_has_required_files(self, split_path, seq_name, start_frame):
        start_pos = self.frame_to_pos.get(int(start_frame))
        if start_pos is None or start_pos + self.num_views > len(self.frame_ids):
            return False

        for v in range(self.num_views):
            frame_idx = self.frame_ids[start_pos + v]
            if not _avatarrex_has_required_frame_files(
                split_path, seq_name, frame_idx, require_depth=self.load_da3_depth
            ):
                return False
        return True

    def __len__(self):
        return len(self.samples)

    def get_image_num(self):
        return sum(self.seq_frames.values())

    def _get_views(self, idx, resolution, rng, num_views):
        seq_name, t = self.samples[idx]
        cam = 0

        split_path = osp.join(self.ROOT, self.split)

        # Video 模式: 所有帧相机连续，shot_label 全为 0
        shot_labels = [0] * num_views

        views = []
        # **========== 原始代码：frame_idx = t + v ==========**
        # for v in range(num_views):
        #     frame_idx = t + v
        #     views.append(self._load_view(
        #         split_path, seq_name, cam, frame_idx, resolution, rng, v,
        #         shot_labels[v],
        #     ))
        # **========== 新代码：从真实帧号列表中取连续 num_views 个可用文件帧 ==========**
        start_pos = self.frame_to_pos[t]
        for v in range(num_views):
            frame_idx = self.frame_ids[start_pos + v]
            view = self._load_view(
                split_path, seq_name, cam, frame_idx, resolution, rng, v,
                shot_labels[v],
            )
            view.update(_empty_anchor_info(self.anchor_top_k))
            views.append(view)
        # **========== 结束 ==========**

        return views

    def _load_view(self, split_path, seq_name, cam_id, frame_idx, resolution, rng, v, shot_label=0):
        """加载单个 view。"""
        frame_str = f"{frame_idx:08d}"

        rgb_path = osp.join(split_path, seq_name, "rgb", f"{frame_str}.png")
        cam_path = osp.join(split_path, seq_name, "cam", f"{frame_str}.npz")
        depth_path = osp.join(split_path, seq_name, "depth", f"{frame_str}.npy")
        smpl_path = osp.join(split_path, seq_name, "smpl", f"{frame_str}.pkl")

        rgb_image = imread_cv2(rgb_path)

        # Camera params
        cam = np.load(cam_path)
        camera_pose = cam["pose"].astype(np.float32)
        intrinsics = cam["intrinsics"].astype(np.float32)
        raw_camera_pose = _raw_calibration_c2w(self.raw_calibration, seq_name)

        # **========== 原始代码：直接按 float 米单位读取，导致旧 uint16 毫米数据被 >200 阈值清零 ==========**
        # if osp.exists(depth_path):
        #     depthmap = np.load(depth_path).astype(np.float32)
        #     depthmap[~np.isfinite(depthmap)] = 0
        #     depthmap[depthmap > 200.0] = 0.0
        # else:
        #     h, w = rgb_image.shape[:2]
        #     depthmap = np.zeros((h, w), dtype=np.float32)
        # **========== 新代码：兼容旧 uint16 毫米和新 float32 米单位 ==========**
        # V8.1 pose-prompt experiments can disable DA3 pseudo-depth completely:
        # the pose loss does not use GT depth, and Human3R's predicted pointmap
        # is the reconstruction cue we want to inspect.
        if self.load_da3_depth:
            depthmap = _load_depthmap_meters(depth_path, rgb_image.shape)
        else:
            depthmap = _empty_depthmap(rgb_image.shape)
        # **========== 结束 ==========**

        # Mask
        mask_path = osp.join(split_path, seq_name, "mask", f"{frame_str}.png")
        if osp.exists(mask_path):
            mask_image = imread_cv2(mask_path)
        else:
            mask_image = None

        # SMPL
        annots = []
        if osp.isfile(smpl_path):
            with open(smpl_path, "rb") as f:
                annots = pickle.load(f)

        # Crop/resize
        if mask_image is not None:
            rgb_image, depthmap, mask_image, intrinsics = \
                self._crop_resize_if_necessary_mask(
                    rgb_image, depthmap, mask_image, intrinsics,
                    resolution, rng=rng, info=f"{seq_name}/{cam_id}/{frame_idx}"
                )
        else:
            rgb_image, depthmap, intrinsics = \
                self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics,
                    resolution, rng=rng, info=f"{seq_name}/{cam_id}/{frame_idx}"
                )

        # SMPL 整理
        # smplx_transl 坐标系修复：变换到相机坐标系后再判断和排序
        R_c2w = camera_pose[:3, :3]
        t_c2w = camera_pose[:3, 3]

        humans_with_cam_z = []
        for h in annots:
            smpl_world = np.array(h.get("smplx_transl", [0, 0, 100]), dtype=np.float32)
            smpl_cam = R_c2w.T @ (smpl_world - t_c2w)
            h = dict(h)
            h["_smplx_transl_cam"] = smpl_cam
            h["_smplx_transl_cam_z"] = smpl_cam[2]
            humans_with_cam_z.append(h)

        if humans_with_cam_z:
            l_dist = [hh["_smplx_transl_cam_z"] for hh in humans_with_cam_z]
            order = sorted(range(len(l_dist)), key=lambda i: l_dist[i])
            humans_with_cam_z = [humans_with_cam_z[i] for i in order]

        # 相机坐标系 Z > -0.5 即可通过
        humans = [hh for hh in humans_with_cam_z if hh["_smplx_transl_cam_z"] > -0.5]

        smpl_mask = np.zeros(self.max_humans, dtype=np.bool_)
        if len(humans) > 0:
            smpl_mask[:len(humans)] = True

        smpl_dict = {}
        for k, shape in self.smpl_key2shape.items():
            smpl_dict[k] = np.zeros(
                (self.max_humans, *shape), dtype=np.float32
            )
            if len(humans) > 0:
                for h in range(len(humans)):
                    val = humans[h].get(k, np.zeros(shape))
                    if isinstance(val, np.ndarray):
                        val = val.astype(np.float32)
                        if len(shape) > 1:
                            val = val.reshape(shape)
                        smpl_dict[k][h] = val
            # smplx_transl 使用变换后的相机坐标系值
            if k == "smplx_transl":
                for h in range(len(humans)):
                    smpl_dict[k][h] = humans[h]["_smplx_transl_cam"]

        # Masks
        img_mask, ray_mask = self.get_img_and_ray_masks(
            self.is_metric, v, rng, p=[0.85, 0.00, 0.15]
        )

        return dict(
            img=rgb_image,
            msk=False if mask_image is None else mask_image,
            depthmap=depthmap,
            camera_pose=camera_pose,
            **({} if raw_camera_pose is None else {"raw_camera_pose": raw_camera_pose}),
            camera_intrinsics=intrinsics,
            dataset="AvatarReX_Video",
            label=f"{seq_name}_{frame_str}",
            instance=rgb_path,
            is_metric=self.is_metric,
            is_video=True,           # 连续视频
            quantile=np.array(1, dtype=np.float32),
            img_mask=img_mask,
            ray_mask=ray_mask,
            camera_only=False,
            depth_only=False,
            single_view=False,
            reset=False,
            shot_label=shot_label,   # 0=连续, Video 模式全为 0
            smpl_mask=smpl_mask,
            **smpl_dict,
        )
