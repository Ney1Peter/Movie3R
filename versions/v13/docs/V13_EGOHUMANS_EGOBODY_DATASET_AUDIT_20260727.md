# V13 EgoHumans / EgoBody 数据集归属与完整性审计

> 审计日期：2026-07-27  
> 目的：纠正本地目录命名混用，记录 EgoBody 解压结果，并冻结两套数据在 V13 多人实验中的职责。

## 1. 结论

以下两个本地路径不是同一个数据集：

| 本地路径 | 实际数据集 | 当前内容 |
|---|---|---|
| `/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble` | **EgoHumans** | 三人、8 路 exo、3 路 Aria、RGB、SMPL 和多人标注 |
| `/data/wangzheng/iJCV-CODE/data/EgoHuman` | **EgoBody** | 双人 EgoBody release 的标定、Kinect 参数和 val 人体 GT；当前无 RGB |

本地父目录名称与真实数据集名称发生了交叉：

```text
data/EgoBody/001_legoassemble  -> 实际是 EgoHumans
data/EgoHuman                  -> 实际是 EgoBody
```

后续文档和结果统一使用正式数据集名：

```text
EgoHumans 001_legoassemble
EgoBody val recordings
```

现阶段不移动数据目录。已有代码、cache 和结果依赖旧绝对路径；直接移动会破坏实验复现。

## 2. 数据集身份依据

### 2.1 EgoHumans `001_legoassemble`

目录包含：

```text
exo/cam01 ... cam08
ego/aria01 ... aria03
processed_data/poses2d
processed_data/poses3d
processed_data/bboxes
processed_data/smpl
colmap/workplace
```

其 8 路外部相机、3 个 Aria wearer、三人稳定身份以及 `001_legoassemble` capture 命名均对应
EgoHumans。该数据当前约占 `3.9 GB`，exo RGB 图像可用，已经能够运行 Human3R probe。

### 2.2 EgoBody release

`/data/wangzheng/iJCV-CODE/data/EgoHuman` 包含：

```text
calibrations/
kinect_cam_params/
smpl_interactee_val/
smplx_camera_wearer_val/
data_info_release.csv
data_splits.csv
```

`camera wearer + interactee`、HoloLens-to-Kinect 标定、wearer SMPL-X 和 interactee SMPL
是 EgoBody release 的结构。它不是 EgoHumans 的另一个 capture。

## 3. EgoBody 解压与压缩包处理

以下压缩包已通过 `unzip -tq` 完整性测试，成功解压后删除：

```text
calibrations.zip
kinect_cam_params.zip
smpl_interactee_val.zip
smplx_camera_wearer_val.zip
```

`kinect_color.zip` 原文件约为 `211,884,048,384` bytes，但存在以下问题：

- 文件开头存在 ZIP local header；
- 文件末尾仍是压缩图像数据；
- 缺少 ZIP/ZIP64 central-directory end record；
- 本地不存在配套 `.z01/.z02/...` 分卷；
- `unzip` 无法列出或恢复目录。

因此该文件是截断或未完成下载的残包，不是可正常解压的完整 ZIP，已按清理压缩文件的要求删除。
EgoBody RGB 若要用于推理，必须重新下载完整包，不能从当前目录恢复。

清理后：

```text
EgoBody 本地目录占用：约 462 MB
残留 ZIP/TAR/RAR/7z：0
/data 可用空间：约 26 GB（文件系统仍接近满载）
```

## 4. EgoBody GT 完整性审计

### 4.1 Release metadata

`data_info_release.csv` 共列出：

```text
125 recordings
15 scenes
```

`data_splits.csv` 划分为：

```text
train: 65
val:   17
test:  43
```

当前只下载了 val 的 wearer/interactee 人体参数。所有 recording 的 calibration metadata 均在，
但 train/test 人体 GT 不在当前目录。

### 4.2 Val 人体 GT

17 个 val recording 的覆盖为：

| 内容 | 文件数 | 检查结果 |
|---|---:|---|
| Interactee SMPL | 29,140 PKL | 全部可加载，字段存在，数值有限 |
| Camera wearer SMPL-X | 29,140 PKL | 全部可加载，字段存在，数值有限 |
| 合计 | 58,280 PKL | 0 个坏文件 |

每个 recording 内：

- wearer 与 interactee 帧号完全一致；
- 覆盖 metadata 的 `start_frame ... end_frame`；
- 没有逐帧 GT 缺失。

Interactee SMPL 至少包含：

```text
global_orient: (1, 3)
body_pose:     (1, 69)
betas:         (1, 10)
transl:        (1, 3)
```

Camera wearer SMPL-X 至少包含：

```text
global_orient
transl
betas
body_pose: (1, 63)
left_hand_pose / right_hand_pose
jaw_pose / eye poses / expression
gender
```

### 4.3 Camera GT

17 个 val recording 中：

```text
7 recordings:  Kinect 11, 12, 13       (3 cameras)
10 recordings: Kinect 11, 12, 13, 14, 15 (5 cameras)
```

标定提供：

```text
holo_to_kinect12
kinect12_to_world
kinect_11to12_color
kinect_13to12_color
部分 recording 的 kinect_14to12_color / kinect_15to12_color
```

全量检查结果：

```text
626 calibration matrices: 0 bad
10 Kinect intrinsic files: 0 bad
```

所有被检查的外参均为有限、合法的 `4 x 4` 变换；内参维度、焦距和数值均合法。

## 5. 是否适合 V13 多人测试

### 5.1 适合的任务

在重新获得同步 RGB 后，EgoBody 适合作为独立的双人辅助验证集：

- 两人跨 Kinect camera cut；
- 大 camera span 的同步视角切换；
- wearer/interactee 跨 shot identity；
- 近距离双人交互和遮挡；
- camera translation / rotation；
- human root 和 common joints；
- pairwise human distance / relative-vector layout；
- 某一人物漏检时的 single-human fallback。

相机标定和 val 双人体参数足以支持上述 GT evaluation。

### 5.2 当前不能直接做的任务

当前不能直接运行 Human3R 或形成完整 benchmark，原因是：

1. RGB 包损坏后已删除，当前没有 Kinect color images；
2. 只有 val 人体 GT，当前不是完整 train/val/test release；
3. wearer 使用 SMPL-X，interactee 使用 SMPL，二者 vertex topology 不统一；
4. 不应直接比较两个人的完整 vertex error，优先评价 root/common joints/layout；
5. 场景主要为两人，不能验证三人 consensus 和 `1/2/3` 人数消融；
6. 当前目录没有完整 depth、scene mesh、HoloLens RGB/trajectory 等 release 内容。

## 6. 与现有多人数据的职责划分

| 数据集 | 人数/相机 | 当前优势 | V13 推荐职责 |
|---|---|---|---|
| MultiHuman `three` | 3 人，6 路同步 camera | 统一逐帧 SMPL-X，三人 consensus | GT-ID 多人几何主 benchmark |
| MultiHuman `dance/box` | 2 人，6 路同步 camera | 动作、遮挡、独立 sequence | frozen cross-sequence evaluation |
| EgoHumans `001_legoassemble` | 3 人，8 路 exo + 3 路 Aria | RGB、鱼眼、人数变化、重现 | identity/TTL/fallback stress test |
| EgoBody val | 2 人，3/5 路 Kinect + HoloLens | 双人近距离交互、完整相机标定 | 双人跨视角辅助验证；需先补 RGB |

EgoBody 可以增加一套有价值的跨数据验证，但不能替代 MultiHuman 主 benchmark，也不能与
EgoHumans `001_legoassemble` 合并成同一个数据来源。

## 7. 后续数据准备建议

由于 `/data` 仅余约 `26 GB`，不要立即重新下载完整约 198 GB 的 Kinect RGB 包。优先方案为：

1. 确认官方下载端是否支持按 split 或 recording 下载；
2. 只准备一个 val recording 的 3 或 5 路同步 color frames；
3. 验证 frame index、相机标定、人体 GT 和 RGB 严格对应；
4. 先构造 4-frame `AABB` 和少量 multi-cut probe；
5. 通过后再决定是否扩展 EgoBody RGB 覆盖。

在 RGB 补齐前，EgoBody 状态应标记为：

```text
GT audit passed
RGB unavailable
not runnable by Human3R
auxiliary benchmark pending
```
