# V8.5 AIST 多视角数据预处理计划

## 目标

把 `/data/wangzheng/iJCV-CODE/organized_by_motion.zip` 解压并转换成和现有 AvatarReX / THUman 一致的训练格式，作为后续 pose correction token 训练数据。

核心原则：

- 每个视频只保留前 3 秒，避免动作重复并节省空间。
- 视频是 60 FPS，但短时间相邻帧重复较多，因此只在前 3 秒内每 4 帧取 1 帧。
- 每个视角最终输出 45 帧，等价于 15 FPS；RGB、SMPL、camera 都按同一个源帧时间戳对齐。
- RGB、SMPL、camera 必须严格对齐同一段时间。
- 先用一个小序列测试，确认格式和坐标系正确，再批量转换。
- 暂时不生成 depth。

## 输入

原始压缩包：

```text
/data/wangzheng/iJCV-CODE/organized_by_motion.zip
```

压缩包内目前看到的结构类似：

```text
asit/
  gBR_sBM_cAll_d04_mBR0/
    gBR_sBM_cAll_d04_mBR0_ch01/
      camera/
        c01_camera.json
        c02_camera.json
        ...
      gt/
        smpl.pkl
      videos/
        c01.mp4
        c02.mp4
        ...
```

其中：

- `*_chXX` 表示一个 motion / take 的子序列。
- `videos/c01.mp4` 到 `videos/c09.mp4` 表示不同相机视角。
- `camera/cXX_camera.json` 是对应视角的相机参数。
- `gt/smpl.pkl` 是该序列的人体 GT。

## 目标输出格式

最终输出放到统一训练目录：

```text
/data/wangzheng/iJCV-CODE/data/Training/asit/
```

目标结构对齐现有 `Training/lbn1`：

```text
/data/wangzheng/iJCV-CODE/data/Training/asit/
  gBR_sBM_cAll_d04_mBR0_ch01/     # 同一个多视角 scene / take
    c01/
      rgb/
        00000000.png
        ...
        00000179.png
      mask/        # 如果原始数据没有 mask，先生成占位全 1 mask 或暂时不启用
      smpl/
        00000000.pkl
        ...
        00000179.pkl
      cam/
        00000000.npz
        ...
        00000179.npz

    c02/
      rgb/
      mask/
      smpl/
      cam/
```

这样 `gBR_sBM_cAll_d04_mBR0_ch01` 表示同一个多视角场景，`c01-c09` 是这个场景下的不同相机。后续构造 `aaaa` 时从同一个 camera 目录取连续 4 帧；构造 `aabb` 时只允许在同一个 scene 下选两个不同 camera，不能跨 `ch01/ch02` 或跨不同 motion 乱配。

## 执行步骤

### 1. 解压原始数据

输出目录：

```text
/data/wangzheng/iJCV-CODE/data/asit/
```

计划命令：

```bash
mkdir -p /data/wangzheng/iJCV-CODE/data/asit
7z x /data/wangzheng/iJCV-CODE/organized_by_motion.zip -o/data/wangzheng/iJCV-CODE/data/asit
```

解压后检查：

- 是否有 `*_chXX/camera`
- 是否有 `*_chXX/gt/smpl.pkl`
- 是否有 `*_chXX/videos/cXX.mp4`
- 每个 `chXX` 是否相机数量一致，允许少数视角缺失，但转换时要显式跳过

### 2. 写小序列检查脚本

脚本位置：

```text
/data/wangzheng/iJCV-CODE/Movie3R/scripts/v8_5_check_aist_projection.py
/data/wangzheng/iJCV-CODE/Movie3R/scripts/v8_5_check_aist_all_camera_projection.py
/data/wangzheng/iJCV-CODE/Movie3R/scripts/v8_5_check_aist_sync_offsets.py
```

功能：

- 读取一个 `*_chXX` 序列。
- 打印可用 camera 文件和 video 文件。
- 检查每个视频 FPS、总帧数、分辨率。
- 读取 `smpl.pkl`，打印 SMPL 参数 key、shape、帧数。
- 读取一个 camera json，打印内参、外参字段。

目标：

确认 AIST 的 camera / SMPL 表达方式，再决定是否需要坐标系转换。

### 3. 写小序列转换脚本

脚本位置：

```text
/data/wangzheng/iJCV-CODE/Movie3R/scripts/v8_5_preprocess_aist_to_training.py
```

先支持单个序列，例如：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v8_5_preprocess_aist_to_training.py \
  --input_root /data/wangzheng/iJCV-CODE/data/asit \
  --output_root /data/wangzheng/iJCV-CODE/data/Training \
  --sequence gBR_sBM_cAll_d04_mBR0/gBR_sBM_cAll_d04_mBR0_ch01 \
  --seconds 3 \
  --fps 60 \
  --frame_stride 4 \
  --clean_output
```

转换内容：

- 从每个 `videos/cXX.mp4` 的前 3 秒 source frames 中每 4 帧取 1 帧，保存到 `rgb/`。
- 输出文件名重新编号为连续的 `00000000.png ... 00000044.png`，但 `cam/*.npz` 会记录真实 `source_frame_idx` 和 `source_timestamp_sec`。
- 对每一帧保存对应 `smpl/*.pkl`，内容是一个 human list，字段为 native SMPL：
  `smpl_root_pose`、`smpl_body_pose`、`smpl_shape`、`smpl_transl`、`smpl_scale`、`smpl_gender_id`。
- 对每一帧保存对应 `cam/*.npz`。
- 如果没有可用 mask，先生成全 1 mask，保证 dataloader 不报错；后续训练可以选择不使用 mask。
- 不生成 depth。

### 4. 小序列可视化验证

验证对象：

```text
gBR_sBM_cAll_d04_mBR0_ch01
```

至少检查 3 个相机：

```text
c01, c05, c09
```

验证内容：

- RGB 帧是否完整，没有错误裁剪。
- 第 0、60、120、179 帧是否和视频对应。
- SMPL 投影到图像上是否贴合人体。
- camera 坐标是否和现有 AvatarReX / THUman 的训练格式一致。

输出可视化：

```text
output/v8_5_aist_preprocess_check/
  gBR_sBM_cAll_d04_mBR0_ch01/
    c01_projection_check.png
    c05_projection_check.png
    c09_projection_check.png
    camera_smpl_scene_check/
```

通过标准：

- SMPL 投影和人像基本对齐。
- 相机位姿方向合理。
- 同一帧不同视角的人体 GT 在同一个世界坐标下是统一的。
- 读取到 dataloader 后尺寸走 Human3R demo resize，不再出现旧版 `512x288` 强裁剪问题。

### 5. 批量转换

小序列验证通过后，再批量转换全部可用序列：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v8_5_preprocess_aist_to_training.py \
  --input_root /data/wangzheng/iJCV-CODE/data/asit \
  --output_root /data/wangzheng/iJCV-CODE/data/Training \
  --seconds 3 \
  --fps 60 \
  --frame_stride 4 \
  --clean_output \
  --all
```

批量时需要记录 manifest：

```text
/data/wangzheng/iJCV-CODE/data/Training/asit_manifest.json
```

manifest 内容：

- sequence name
- camera id
- extracted frame count
- image size
- smpl frame count
- camera file path
- skipped reason, if any

### 6. 构造训练 clip

转换完成后，再更新现有 V8.4 / V8.5 manifest 构造脚本，使其支持 `asit`：

```text
/data/wangzheng/iJCV-CODE/data/Training/asit
```

构造方式：

- `aaaa`：同一个 `scene/camera` 内连续 4 帧。
- `aabb`：同一个 `scene` 内两个不同 camera，时间连续 4 帧，例如 `c01` 的 t,t+1 和 `c05` 的 t+2,t+3。
- 优先选择视角差大的相机组合，例如 `c01-c05`、`c01-c09`、`c03-c08`。
- 保留一部分 sequence / camera pair 作为显式 test，不进入训练。

### 7. 训练前最终检查

正式训练前必须做两个检查：

1. Dataloader 可视化：
   - 随机选 3 组 `aaaa`
   - 随机选 3 组 `aabb`
   - 显示 4 帧 RGB、SMPL 投影、GT camera

2. Human3R raw 生成检查：
   - 用 demo-aligned resize 重新跑 raw Human3R
   - 确认 raw / GT / corrected 三套相机可视化坐标一致

## 风险点

### camera 坐标系

AIST 的 camera json 可能是 world-to-camera，也可能是 camera-to-world。必须通过 SMPL 投影验证，不能只看字段名。

当前已验证的正确约定：

```text
V_smpl --SMPL model--> mesh
X_world = smpl_scaling * V_smpl + smpl_trans
X_cam   = R(camera.rotation) @ X_world + camera.translation
uv      = K @ X_cam / z
```

也就是说，AIST 原始 `camera/*.json` 是 world-to-camera 外参；转换成训练格式时保存为 c2w：

```text
cam/{frame}.npz:
  pose       = inv([R_w2c | T_w2c])
  intrinsics = K
```

训练 dataloader 中，`asit/*` 原计划标记 `human_params_are_world=True`，由 `SMPLModel.update_smpl_gt()` 把 world-space SMPL mesh 变换到当前相机坐标后再投影 / 生成 GT。

混合训练时仍然可以传 AvatarReX 的 raw calibration 字典，例如只包含 `lbn1/zzr`。`asit/*` 和 `thuman*/*` 不会查 AvatarReX raw calibration，而是使用自身 `cam/*.npz` 中的 `camera_pose` 作为 `raw_camera_pose`，避免因为缺少 `asit` calibration 报错。

2026-06-09 状态更新：

```text
ASIT 预处理结果保留，但暂时不进入 V8.6 pose-only 主线训练。
为 ASIT 增加的 native SMPL 训练分支已先回退，避免影响 AvatarReX + THuman 的 SMPL-X 主线。
```

### SMPL 参数格式

`smpl.pkl` 可能不是逐帧 npz 格式，需要检查 key 和 shape。转换时要按输出帧对应的源帧取人体参数。例如 `--frame_stride 4` 时，输出第 `k` 帧使用原始 SMPL 第 `4k` 帧。

当前已确认 AIST 是 native SMPL，不是 SMPL-X：

```text
smpl_poses:   (720, 72)
smpl_scaling: (1,)
smpl_trans:   (720, 3)
```

因此如果后续重新启用 ASIT，需要再单独引入 native SMPL 兼容分支，并做投影、dataloader、loss、viewer 四重验证。当前 V8.6 主线不启用该分支。

### mask

目前只确认有视频、camera、SMPL，未确认是否有 mask。因为当前 pose correction 训练主要依赖 RGB、camera、SMPL，可以先用全 1 mask 占位；如果后续 Human3R human prompt 对 mask 很敏感，再补做人像 mask。

### 数据重复

同一个 motion 内动作可能重复，所以每个视频先只取前 3 秒，并用 `--frame_stride 4` 降到 15 FPS。后续如果发现前 3 秒仍然重复，可以改成按 motion interval 抽样。

## 第一阶段完成标准

第一阶段只算完成到小序列测试：

- `gBR_sBM_cAll_d04_mBR0_ch01` 已转换到 `Training/asit`。
- 至少 3 个相机视角完成 SMPL 投影检查。
- dataloader 能读出 4 帧 `aaaa` 和 `aabb`。
- Human3R demo resize 下图像完整，没有旧版裁剪问题。
- 明确 AIST camera / SMPL 坐标系是否需要转换。

通过后再进入全量批处理。

## 当前验证结果

### 原始数据统计

- motion 目录：9 个
- 序列：80 个
- camera-video 可用 pair：660 个
- 原始解压目录大小：约 7.6G
- 当前小转换输出：约 374M / 3 个视角序列

### 多视角角度分布

按所有 sequence 内可用 camera pair 统计：

```text
pair 总数: 2513
p50: 90.6 deg
p75: 135.5 deg
p90: 177.1 deg
>=60 deg: 1744
>=90 deg: 1342
```

说明 AIST 可以提供大量大角度 AABB 样本。

### Dataloader 坐标验证

新增训练路径验证脚本：

```text
scripts/v8_5_check_aist_dataloader_coordinates.py
```

验证内容：

```text
converted Training/asit
  -> AvatarReX_AABB / AvatarReX_Video dataloader
  -> Human3R demo resize
  -> SMPLModel.update_smpl_gt()
  -> SMPL joints / vertices 投影回 resized image
```

已验证两个样本：

```text
output/v8_5_aist_dataloader_coordinate_check_small_nested/
  asit_aabb_overlay.png
  asit_aaaa_overlay.png

output/v8_5_aist_dataloader_coordinate_check_large_angle_nested/
  asit_aabb_overlay.png
  asit_aaaa_overlay.png
```

结果：

- 小角度 AABB：c01 -> c09，约 7.8 deg，SMPL 投影正确。
- 大角度 AABB：c02 -> c06，约 179.6 deg，SMPL 投影正确。
- AAAA 连续帧：相机角度 0 deg，SMPL 投影正确。
- 所有验证帧 `human_params_are_world=True`，相机坐标下 SMPL depth 为正，未出现上下颠倒、尺度错误或坐标系错位。

### Manifest 构造 smoke

现有脚本可以直接支持 `asit`：

```bash
.venv/bin/python scripts/v8_4_build_mixed_aabb_aaaa_manifests.py \
  --training_root /data/wangzheng/iJCV-CODE/data/Training \
  --output_dir output/v8_5_asit_manifest_smoke \
  --groups asit \
  --train_aabb 20 \
  --train_aaaa 20 \
  --val_aabb 5 \
  --val_aaaa 5 \
  --test_aabb 5 \
  --test_aaaa 5 \
  --min_aabb_angle 60 \
  --overwrite \
  --no_test_symlinks
```

smoke 结果：

- `aabb` / `aaaa` 都能成功构造。
- train / val / test 的 image frame overlap 为 0。
- AABB 能覆盖 `060_090`、`090_120`、`120_150`、`150_180` 角度 bucket。
- 混合训练时只需要在 `--groups` 中加入 `asit`。
- 对 `asit/<scene>/<camera>`，AABB 采样必须限制在同一个 `<scene>` 内的不同 camera，不允许跨 scene 配对。当前 manifest 脚本已按 parent scene 分组。

### 已发现异常视角

`gBR_sBM_cAll_d04_mBR0_ch01/c05` 视频和 SMPL 时间不同步，已删除该视角对应的 camera/video，转换脚本会自动跳过缺失 camera-video pair。
