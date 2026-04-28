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
import numpy as np
import pickle
from tqdm import tqdm

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import imread_cv2


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
        **kwargs,
    ):
        assert ROOT is not None, "AvatarReX_AABB requires ROOT"
        self.ROOT = ROOT
        self.split = split
        self.is_metric = True
        self.max_interval = 1           # AABB 固定间隔1
        self.max_humans = 10
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

        self.scenes = sorted([
            d for d in os.listdir(seq_dir)
            if osp.isdir(osp.join(seq_dir, d))
        ])

        # 每个序列只有 1 个相机（cam_id=0000）
        self.seq_cams = {s: [0] for s in self.scenes}

        # 获取帧数（所有序列帧数相同）
        # 预处理脚本输出为扁平结构：rgb/{frame:08d}.png（无 camera 子目录）
        sample_seq = self.scenes[0]
        rgb_dir = osp.join(seq_dir, sample_seq, "rgb")
        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        self.num_frames = len(frames)
        self.seq_frames = {s: self.num_frames for s in self.scenes}

        print(f"  {len(self.scenes)} sequences, {self.num_frames} frames each")
        print(f"  Building AABB index...")

        # AABB: 两两跨序列组合（允许同序列但这里只有不同序列才有效）
        # seqA, seqB 来自不同的序列目录（不同相机）
        # t ∈ [0, num_frames-4]（需要 t, t+1, t+2, t+3 均有效）
        self.samples = []
        for i, seqA in enumerate(self.scenes):
            for j, seqB in enumerate(self.scenes):
                if i == j:
                    continue  # 跳过同一相机
                for t in range(self.num_frames - 3):
                    self.samples.append((seqA, seqB, t))

        print(f"  AvatarReX_AABB: {len(self.samples):,} samples "
              f"({len(self.scenes)} cameras × {len(self.scenes)-1} pairs × {self.num_frames-3} time steps)")

    def __len__(self):
        return len(self.samples)

    def get_image_num(self):
        return sum(self.seq_frames.values())

    def _get_views(self, idx, resolution, rng, num_views):
        assert num_views == 4, "AABB dataset only supports num_views=4"

        seqA_name, seqB_name, t = self.samples[idx]
        t1 = t + 1
        t2 = t + 2
        t3 = t + 3
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

        for v, (seq_name, cam_id, frame_idx, annots, shot_label) in enumerate(view_specs):
            views.append(self._load_view(
                split_path, seq_name, cam_id, frame_idx, annots,
                resolution, rng, v, shot_label,
            ))

        assert len(views) == num_views
        return views

    def _load_smpl(self, split_path, seq_name, cam_id, frame_idx):
        """加载 SMPL 参数。"""
        # fast 脚本输出为扁平结构：smpl/{frame:08d}.pkl（无 camera 子目录）
        smpl_path = osp.join(
            split_path, seq_name, "smpl",
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
        rgb_path = osp.join(split_path, seq_name, "rgb", f"{frame_str}.png")
        cam_path = osp.join(split_path, seq_name, "cam", f"{frame_str}.npz")
        depth_path = osp.join(split_path, seq_name, "depth", f"{frame_str}.npy")

        rgb_image = imread_cv2(rgb_path)

        # Camera params
        cam = np.load(cam_path)
        camera_pose = cam["pose"].astype(np.float32)
        intrinsics = cam["intrinsics"].astype(np.float32)

        # Depth（可能不存在，用全零占位）
        if osp.exists(depth_path):
            depthmap = np.load(depth_path).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0
            depthmap[depthmap > 200.0] = 0.0
        else:
            h, w = rgb_image.shape[:2]
            depthmap = np.zeros((h, w), dtype=np.float32)

        # Mask（可能不存在）
        mask_path = osp.join(split_path, seq_name, "mask", f"{frame_str}.png")
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
        **kwargs,
    ):
        assert ROOT is not None, "AvatarReX_Video requires ROOT"
        self.ROOT = ROOT
        self.split = split
        self.is_metric = True
        self.max_interval = 4
        self.max_humans = 10
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

        # 获取帧数
        sample_seq = self.scenes[0]
        rgb_dir = osp.join(seq_dir, sample_seq, "rgb")
        frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        self.num_frames = len(frames)
        self.seq_frames = {s: self.num_frames for s in self.scenes}

        print(f"  AvatarReX_Video: {len(self.scenes)} sequences, "
              f"{self.num_frames} frames each")

        # 构建索引：每个 scene 的每个有效起始位置
        self.samples = []
        for seq_idx, seq_name in enumerate(self.scenes):
            # 可起始位置：[0, num_frames - num_views]
            for t in range(self.num_frames - self.num_views + 1):
                self.samples.append((seq_name, t))

        print(f"  AvatarReX_Video: {len(self.samples):,} samples")

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
        for v in range(num_views):
            frame_idx = t + v
            views.append(self._load_view(
                split_path, seq_name, cam, frame_idx, resolution, rng, v,
                shot_labels[v],
            ))

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

        # Depth
        if osp.exists(depth_path):
            depthmap = np.load(depth_path).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0
            depthmap[depthmap > 200.0] = 0.0
        else:
            h, w = rgb_image.shape[:2]
            depthmap = np.zeros((h, w), dtype=np.float32)

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
