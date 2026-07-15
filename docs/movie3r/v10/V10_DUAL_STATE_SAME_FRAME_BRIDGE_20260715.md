# V10 双状态同帧桥接实验

日期：2026-07-15

## 1. 实验目的

验证 camera cut 发生后，原版 Human3R 继续读取旧 recurrent state 得到的边界输出，是否仍然保留了可利用的旧世界坐标信息。

如果该信息存在，就可以在 cut 后第一帧同时运行两个分支：

```text
Continue：只在边界读取一次旧 state，提供旧世界参考
Reset：使用 fresh state，负责重建整个新 shot
```

然后根据两个分支处理同一张 RGB 时的 camera 或 pointmap，计算一次从 Reset 局部坐标系到 Continue 旧世界坐标系的 SE(3)，固定用于整个后续 shot。

本实验不训练网络、不检测 cut、不使用未来帧，也不使用 GT 计算桥接变换。GT 只在所有输出完成后用于评测。

## 2. 数据选择

使用 AvatarReX，而不是 BEDLAM。

原因是本实验要诊断 Human3R 的 RGB encoder、decoder 和 recurrent state。AvatarReX 具有真实 RGB 输入和完整相机 GT，已有 V10 oracle probe 也证明该样本存在明显的 cut 后 state pollution。

实验序列为：

```text
seq A：lbn1/22053926
seq B：lbn1/22010716
start_frame：1192
A 段：10 帧
B 段：11 帧
总长度：21 帧
cut_idx：10
```

这不是 4 帧测试。评测覆盖 cut 后 `offset=0..10`，可以观察一次性 bridge 在后续 shot 中是否持续漂移。

边界两个分支读取的都是 B 段帧 `00001202`。脚本同时检查了文件 SHA256 和解码后的逐像素内容，两者完全一致。

## 3. 实现

脚本：

```text
scripts/v10_dual_state_same_frame_bridge_probe.py
```

基础严格 Human3R 输出：

```text
output/v10_oracle_state_vs_gauge_probe/avatarrex_lbn1_1192_cut10
```

本次输出：

```text
output/v10_dual_state_same_frame_bridge/avatarrex_lbn1_1192_cut10
```

### 3.1 对比组

| Variant | 含义 |
|---|---|
| `A_raw_continue` | 原版 Human3R 整段连续运行，cut 后继续旧 state |
| `R_reset_raw` | cut 后 fresh state 重建，但不接回旧世界 |
| `D0_reset_camera_bridge` | Reset + 边界同帧 camera bridge |
| `D1_reset_pointmap_bridge` | Reset + 边界同帧 pointmap bridge |
| `C_reset_oracle_output` | Reset + GT boundary SE(3)，只作为上限参考 |

### 3.2 Camera Bridge

Human3R 保存的 pose 已确认是 camera-to-world。边界变换为：

```text
T_reset_to_continue = T_continue_c2w @ inverse(T_reset_c2w)
```

对 Reset 后续 shot 的 camera pose 统一左乘：

```text
T_camera_world_new = T_reset_to_continue @ T_camera_world_reset
```

脚本自检显示，变换后的 Reset 边界 camera 与 Continue 边界 camera 的残差为严格的 `0 m / 0 deg`，说明 c2w/w2c convention 和变换方向正确。

### 3.3 Pointmap Bridge

Continue 和 Reset 在边界处理同一张 RGB，因此像素位置天然对应。脚本执行：

1. 分别将 depth、intrinsics 和 c2w 恢复成 world pointmap。
2. 排除两个分支 human mask 的并集，并对人体区域膨胀。
3. 排除低置信度、无效深度、非有限 3D 点。
4. 固定相同像素对应，不做最近邻搜索。
5. 使用鲁棒加权 Kabsch 迭代剔除外点。

该过程不是 ICP，不使用 BA、未来帧或整段优化。

本次从 50,000 个采样对应中保留 13,997 个内点，内点平均残差约 `0.0164 m`。

### 3.4 统一变换输出

D0/D1 对 camera pose 左乘同一个 SE(3)。由于 Human3R 保存的 depth 和 SMPL-X 参数位于 camera frame，因此更新同一个 c2w 后，world pointmap、world root、世界朝向和 world mesh 会自然使用相同变换。

为了审计，脚本还显式缓存了：

```text
pelvis_world
anchor_world
root_orient_world
verts_world
```

## 4. 结果

cut 后 11 帧 camera 指标：

| Variant | Cam T mean ↓ | Cam R mean ↓ | Boundary T ↓ | Boundary R ↓ | Last T ↓ | Last R ↓ | RPE T mean ↓ | RPE R mean ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_raw_continue` | 3.2878 | 134.48 | 3.3013 | 133.81 | 3.2864 | 133.81 | 0.1278 | 2.69 |
| `R_reset_raw` | 3.3024 | 133.29 | 3.3044 | 133.25 | 3.3021 | 133.32 | 0.0042 | 0.05 |
| `D0_reset_camera_bridge` | 3.2995 | 133.85 | 3.3013 | 133.81 | 3.2992 | 133.88 | 0.0042 | 0.06 |
| `D1_reset_pointmap_bridge` | 3.2238 | 133.90 | 3.2257 | 133.86 | 3.2235 | 133.94 | 0.0042 | 0.05 |
| `C_reset_oracle_output` | 0.0042 | 0.05 | 0.0000 | 0.00 | 0.0038 | 0.10 | 0.0042 | 0.04 |

Camera Bridge 本身只预测出约 `0.106 m / 0.71 deg` 的变化；Pointmap Bridge 约为 `0.164 m / 1.02 deg`。但真实 A/B shot 的跨镜头变化约为 `3.3 m / 133 deg`。

因此 Continue 和 Reset 的边界输出都停留在接近单位位姿的局部 gauge。Continue 虽然读取了旧 state，但其最终 camera 输出没有携带正确的跨 shot 世界重定位信息。

## 5. 结论

这次结果是否定但有价值的：

1. `R/D0/D1/C` 的 gauge-free RPE 都很低，证明 fresh Reset state 能提供干净、稳定的新 shot 相对轨迹。
2. `A_raw_continue` 的 RPE 明显更高，继续证明旧 state 会污染 cut 后后续轨迹。
3. D0 在边界能够严格复现 Continue camera，但绝对误差仍约 `3.30 m / 134 deg`，说明错误不在 SE(3) 应用方向，而在 Continue 参考本身不正确。
4. D1 的同帧背景 pointmap 只能带来约 `0.08 m` 的平移改善，无法恢复巨大的跨镜头旋转，不能视为有效桥接。
5. Oracle C 几乎完全正确，说明最终方案仍然可以采用 `fresh local state + segment-level re-anchor`，但 re-anchor 不能只读取原版 Continue 的最终 camera 或 pointmap。

最终判断：

```text
旧 recurrent state 确实会影响 cut 后预测，
但原版 Human3R 的同帧 Continue 最终输出不包含足够可靠的旧世界重定位量。
```

这意味着下一步若继续“双状态”路线，旧 state 更适合作为只读 latent/feature memory 输入一个专门的跨 shot relocalization 模块，而不能直接把 Continue camera 当作世界坐标 teacher。

## 6. 输出文件

```text
dual_state_same_frame_bridge_metrics.json
dual_state_same_frame_bridge_metrics.md
boundary/continue_boundary_payload.npz
boundary/reset_boundary_payload.npz
analysis/per_frame_bridge_metrics.csv
analysis/bridge_camera_error_curves.png
analysis/camera_human_trajectories.png
analysis/boundary_pointcloud_stitch.png
```

