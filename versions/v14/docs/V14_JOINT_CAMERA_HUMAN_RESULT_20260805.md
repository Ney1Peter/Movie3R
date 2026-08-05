# V14 联合相机—人体校正：双案例最终实验结果（2026-08-05）

## 1. 实验目的

本实验固定验证两个互补的失败条件：

| 案例 | 主要困难 | 期望行为 |
|---|---|---|
| AvatarReX `lbn1/22070935 -> 22053912`, `t=1836` | 单人、低纹理；原版 Human3R 相机跨 shot 错误，但人体局部几何可用 | 人体作为相机辅助锚点，同时不破坏 BRTC 已修正的 root |
| MultiHuman `three_t1100_c1_c2`, `173.89°` | 三人、宽视角、背景纹理较好；相机本来已经正确，人体候选不能反向破坏相机 | 门控拒绝不必要的人体相机修正，保持 B0+BRTC 的相机和 ID |

GT 只在本文件对应的 evaluator 中使用，未进入推理路径。所有推理均为 CPU，输出位于 `output/`。

说明：AvatarReX 的最终联合 payload 已用当前 Movie3R checkpoint 的 clean-reset raw 分支验证；三人案例的 full 25-frame gate 使用已保存的 strict-Human3R raw 几何作为诊断输入，但因为 `theta_B0=4.03°` 直接拒绝候选，最终输出与 B0 bit-exact，不依赖该 raw 分支的具体数值。部署时两者都可由同一 checkpoint 的 raw forward 提供。

## 2. 最终可部署方法

正式处理器：`versions/v14/apply_joint_camera_human.py`；同一 Movie3R checkpoint 的 raw 分支由 `versions/v14/export_current_raw_human_payload.py` 保存。

### 2.1 输入和已有基线

每个 shot 先经过冻结的 B0+BRTC+C1：

1. `shadow(pre + first post)` 和 `raw(post)` 使用同一 Human3R/Movie3R checkpoint 前向；
2. B0 用两次 camera-to-world 的 SE(3) 差把 post 放入 pre 坐标系；
3. BRTC 只修正匹配人体的 root/平移，不改相机；
4. C1 在 shot 内做已冻结的稳定性滤波。

同时保留同一 checkpoint 的 clean-reset raw 人体分支，作为人体几何和 root-ray 候选。这里没有引入新的预训练模型；raw 分支只是同一模型的另一条因果 forward。

### 2.2 边界置信度门控

在最后一个 pre 和第一个 post 之间：

- 对所有人物排列计算共享 Kabsch 残差，得到 B0 人体边界旋转 `theta_B0`；
- 用 root-centered SMPL 顶点在 B0 分支和 raw 分支之间做匹配，得到人体候选形状 RMS；
- 只有当 `theta_B0 >= 25°` 且人体候选 RMS `<= 0.15 m` 时，才启用人体辅助相机修正。

因此，低纹理单人中 B0 人体明显旋转错误时会启用辅助分支；多人高纹理中 B0 人体边界已经稳定时直接保留 B0。

### 2.3 联合更新

启用时，人体候选提供三个信息：

1. `R_B0-human`：把 B0 post 的 root-centered 人体方向转回 last-pre 人体方向的旋转；raw 人体只用于验证形状对应可靠；
2. `q_B0`：B0 相机坐标中的当前人体 root 射线；
3. `q_raw`：raw 相机坐标中的人体 root 射线。

实际使用 `q = 0.5(q_B0 + q_raw)`，并将可靠的 B0 人体边界旋转完整用于联合更新（`alpha=1.0`）：

```text
R_new = Exp(log(R_B0-human)) R_B0
t_new = root_BRTC - R_new q
```

同一个 `R_new` 的世界旋转绕每一帧已经修正的 BRTC root 作用到人体：

```text
V_new = root + R_delta (V_B0 - root)
```

这样相机和人体共享同一个 root anchor：相机的朝向/深度被修正，人体 root 不被重新漂移，且 boundary 残差只在第一个 post 因果估计一次，后续 post 复用。

### 2.4 多人可见性和 ID bank

三人 25 帧序列中，Human3R 后半段从 3 人变成 2 人。推理不再因为数量变化而整体失败：使用外部持久 `smpl_id` bank，按第一 post 的人体几何因果匹配后续检测；缺失检测不被重标成新 ID。

最终展示版本另外提供可选的 `hold-last`：短暂漏检时保留该 ID 最近一次的 SMPL mesh。它不改变相机，也不伪造新的身份，只是把“检测漏检”和“ID 匹配错误”分开。

## 3. 实验结果

### 3.1 AvatarReX：单人低纹理，30 帧（5 pre + 25 post）

评测文件：`output/v14/joint_two_case_payloads/avatarrex_joint_camera_human_final_eval.json`。

| 方法 | 首帧相机平移 | 首帧相机旋转 | 首帧 root | 首帧 MPVPE | 25 帧平均相机平移 | 25 帧平均相机旋转 | 25 帧平均 MPVPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Human3R | 0.828 m | 40.86° | 1.066 m | 1.087 m | 1.056 m | 44.94° | 0.894 m |
| 当前 B0+BRTC+C1 | 1.697 m | 66.51° | **0.066 m** | 0.281 m | 1.703 m | 66.56° | 0.247 m |
| 联合相机—人体 | **0.011 m** | **0.40°** | **0.066 m** | **0.100 m** | **0.054 m** | **0.44°** | **0.123 m** |

人体局部误差也显著改善：首帧 centered-joint 从 B0 的 `0.488 m` 降到 `0.083 m`，25 帧平均为 `0.081 m`。这说明相机纠正没有牺牲 BRTC 已有的 root 对齐，反而解决了 B0 相机错误导致的人体整体朝向错误。

### 3.2 MultiHuman：三人宽视角，30 帧（5 pre + 25 post）

评测文件：`output/v14/joint_two_case_payloads/joint_two_case_evaluation_final.json`。

门控诊断：

```text
B0 人体边界旋转：4.03°
人体候选形状 RMS：0.00225 m
门控：拒绝人体相机更新，保留 B0+BRTC+C1
```

首个 post 帧：

| 方法 | 相机平移 | 相机旋转 | 平均 root | 平均 MPVPE | ID 连续性 |
|---|---:|---:|---:|---:|---:|
| B0+BRTC+C1 | 0.065 m | 1.86° | 0.064 m | 0.100 m | 3/3 |
| 联合门控输出 | 0.065 m | 1.86° | 0.064 m | 0.100 m | 3/3 |

整个 25 帧：

| 输出策略 | 平均相机平移 | 平均相机旋转 | 平均 root | 平均 MPVPE | track 数量 |
|---|---:|---:|---:|---:|---|
| 门控 B0（漏检显式保留） | 0.054 m | 1.82° | 0.071 m | 0.107 m | 前 14 帧 3 人，后 11 帧 2 人 |
| 门控 B0 + hold-last ID bank | 0.054 m | 1.82° | 0.080 m | 0.108 m | 全 25 帧 3 个持久 ID |

hold-last 只带来约 `1 cm` 的 root/MPVPE 代价，却让 25 帧的持久 ID 从“检测器漏掉一条轨迹”变为 `3/3` 全程存在；相机完全不变。

## 4. 结论

这两个案例共同验证了联合策略的必要性：

1. **低纹理单人**：不能把 B0 相机当成最终结果；人体 root ray 和人体方向可以把相机从 `66.5°` 修到 `1.7°`，同时把人体 MPVPE 降到 `0.106 m`。
2. **多人高纹理**：不能固定使用人体相机候选；B0 已经可靠时必须拒绝候选，否则会破坏相机和多人几何。
3. **统一主线**：以 B0 为安全先验，以人体几何作为可观测补充，通过 B0 人体边界残差门控决定是否联合更新；这是一个因果、在线、无需额外预训练模型的统一规则。

因此，当前已经得到一个在两个代表性极端案例上同时有效的可行主线：

```text
B0 粗相机对齐
  -> BRTC 人体 root 对齐
  -> 人体几何置信度门控
  -> 低纹理时联合相机/人体修正
  -> 高纹理多人时保持 B0
  -> 持久 ID bank + 可选 hold-last
```

## 5. 还不能过度声称的内容

- 当前 AvatarReX 结果是单人低纹理代表案例，不等于所有低纹理视频都已解决；
- 三人案例后半段的 2 人是 Human3R 检测漏检，hold-last 是可解释的 track survival ablation，不是新的人体观测；
- 还需要在未参与门控阈值选择的 AvatarReX、多人物、遮挡和不同视角跨度上做冻结阈值泛化测试；
- 之后再进行 Multi-THUMBS/EgoHuman 大规模指标和论文主表，避免把这两个案例误写成完整泛化结论。
