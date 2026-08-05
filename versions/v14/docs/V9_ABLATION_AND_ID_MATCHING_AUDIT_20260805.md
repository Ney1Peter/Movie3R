# V9/B0 消融与跨 shot 人物 ID 审计（2026-08-05）

## 结论先行

在目前两个代表性案例上，不能得出“去掉 V9 后最终结果几乎不变”。严格的 no-V9 对照使用原版 Human3R payload，再用原版相机计算显式 raw SE(3)，不调用 V9/B0 checkpoint。结果是：

- 三人、约 174° 跨相机案例中，no-V9 的 post 相机误差约为 **4.27 m / 151.89°**（25 帧平均），当前 B0 为 **0.054 m / 1.82°**；V9/B0 的粗 gauge proposal 是决定性组件。
- 单人 AvatarReX 低纹理案例中，no-V9 raw SE(3) 的相机误差为 **2.107 m / 64.53°**，而当前最终 adaptive joint 为 **0.054 m / 0.44°**。这里暴露的是低纹理下原版相机本身不可靠，V9/B0 单独也不能解决全部问题，仍必须有后续人体—相机联合优化。
- 原版 Human3R 输出有 `smpl_id` 字段，但它是每次检测的 native index，不是可靠的跨 shot persistent identity。41 个可比的三人 controlled cuts 上，原始 direct matching 的 all-correct rate 只有 **41.5%**；加入 learned B0 后，四种 root/torso/joint matcher 均达到 **100%**。

因此，V9/B0 应保留在主线，并作为可消融的“learned coarse gauge + identity preconditioning”模块；论文不能声称原版完全没有 ID 字段，应准确表述为：

> 原版 Human3R provides anonymous per-shot detections, but does not provide reliable persistent cross-shot identity association. Movie3R learns a boundary-conditioned coarse gauge that makes permutation-aware identity association observable and stable.

## 1. 对照定义

所有方法都只读取 RGB 和同一模型输出；GT 只在离线 evaluator 中读取，不进入 runtime。

### no-V9 控制

给定原版 Human3R 的 pre/post camera pose：

\[
G_{raw}=C_{pre}^{raw}(C_{post}^{raw})^{-1}.
\]

post 相机和 post 人体一起乘以 `G_raw`，得到不经过 V9/B0 的显式 raw-SE(3) 基线。为了不把“去掉 V9”和“去掉 ID 关联”混在一起，另有 `no_v9_raw_se3_human`：只在边界用一次人体 Kabsch 求 permutation/residual，但仍不使用 V9 checkpoint。`no_v9_adaptive_joint` 则把该 no-V9 payload 送入当前 causal camera-human gate，检验后续联合修正能否独立挽救错误的粗相机。

### 当前主线

`movie3r_b0_brtc_c1` 是冻结的 `B0 + BRTC-LC + C1`；AvatarReX 另有当前最终 `adaptive joint` 结果。B0 先给跨 shot 的粗 gauge 和身份预条件，BRTC/C1 只在相机冻结后修正人体；低纹理案例再由 adaptive joint 对相机和人体做联合修正。

## 2. 单人低纹理：AvatarReX

案例：`avatarrex_t1836_c22070935_c22053912_pre5_post25`，5 帧 pre + 25 帧 post。

| 方法 | 首个 post 相机 t / R | 25 帧平均相机 t / R | 首帧 MPVPE | 25 帧平均 MPVPE |
|---|---:|---:|---:|---:|
| 原版 Human3R | 0.828 m / 40.86° | 1.056 m / 44.94° | 1.087 m | 0.894 m |
| no-V9 raw SE(3) | 1.884 m / 60.45° | 2.107 m / 64.53° | 0.760 m | 0.598 m |
| no-V9 raw SE(3) + human residual | 0.589 m / 20.01° | 0.797 m / 24.11° | 0.648 m | 0.476 m |
| B0 + BRTC + C1 | 1.697 m / 66.51° | 1.703 m / 66.56° | 0.281 m | 0.247 m |
| 当前最终 adaptive joint | **0.011 m / 0.40°** | **0.054 m / 0.44°** | **0.100 m** | **0.123 m** |
| no-V9 + 当前 adaptive gate | 0.548 m / 20.01° | 0.693 m / 24.11° | 0.715 m | 0.580 m |

另外，将当前 gate 直接接在**原版 Human3R 原始 payload**（不先做 raw-SE(3)）上时，AvatarReX 的 gate 会拒绝该低置信边界并保持原版结果；这说明没有 B0 proposal 时，后处理没有可靠的相机候选可用。

解释：单人低纹理中，原版相机估计没有可靠背景约束；仅做 raw SE(3) 会把错误相机继续传播。人体 residual 能改善人体的刚体误差，但不能把相机恢复到 GT。当前 adaptive joint 必须建立在较好的 B0 proposal 上，才能同时把 camera 和 human 拉回正确 gauge。因此该案例证明的是“V9/B0 是有用的粗初始化，但不是最终解”。

完整 JSON：

- `output/v14/report_comparison_viewers/avatarrex_t1836_c22070935_c22053912_pre5_post25/NO_V9_ADAPTIVE_COMPARISON.json`
- `output/v14/report_comparison_viewers/avatarrex_t1836_c22070935_c22053912_pre5_post25/NO_V9_COMPARISON.json`

## 3. 三人约 174° 跨 shot：MultiHuman `three_t1100_c1_c2`

案例：`three_t1100_c1_c2_pre5_post25`，GT camera span 为 **173.89°**。多人 evaluator 在所有方法中使用同一个 pre-shot GT gauge；人体误差同时报告固定 pre-ID 的 direct MPVPE 和忽略 ID 的 best-permutation MPVPE。

| 方法 | 首个 post 相机 t / R | 25 帧平均相机 t / R | 固定 ID 平均 MPVPE | 几何 permutation（14 个三人帧） |
|---|---:|---:|---:|---:|
| 原版 Human3R | 0.905 m / 25.18° | 0.824 m / 3.59° | 0.464 m | **14/14** |
| B0 + BRTC + C1 | **0.065 m / 1.86°** | **0.054 m / 1.82°** | **0.107 m** | **0/14** |
| no-V9 raw SE(3) | 4.117 m / 173.89° | 4.265 m / 151.89° | 0.353 m | 0/14（显式 Kabsch remap 后） |
| no-V9 raw SE(3) + human residual | 4.018 m / 143.58° | 4.005 m / 121.54° | 0.446 m | 0/14（显式 Kabsch remap 后） |
| no-V9 + 当前 adaptive gate | 3.931 m / 143.58° | 3.883 m / 121.54° | 0.373 m | 0/14（显式 Kabsch remap 后） |

更严格的“原版直接输出 + 当前 gate”对照为 **3.878 m / 121.54°**（25 帧平均），且 14/14 个三人帧仍发生几何 permutation；因此不是 raw-SE(3) 对照的构造单独造成退化。

这里 no-V9 的相机错误接近 180°，说明原版 camera pose 的 shot gauge 不能直接用于跨 shot 对齐；后续人体优化即使能降低部分顶点误差，也无法替代正确的粗相机 proposal。B0 的作用不是“最终把人体完全对齐”，而是先把相机和人体送进正确的局部 gauge，使后面的 identity association 和 BRTC-LC 可观测。

完整 JSON：

- `output/v14/joint_two_case_payloads_full/three_t1100_c1_c2_pre5_post25/NO_V9_MULTI_COMPARISON.json`
- `output/v14/joint_two_case_payloads_full/three_t1100_c1_c2_pre5_post25/NO_V9_MULTI_COMPARISON.md`

## 4. 原版 Human3R 的人物 ID 能力

### 单案例证据

在上述三人案例中，pre 的几何行→GT 身份排列为 `[1, 0, 2]`：

- 原版 post 首帧排列为 `[2, 0, 1]`，发生跨 shot permutation；原生 `smpl_id` 顺序为 `[2, 1, 0]`。
- B0 post 首帧排列恢复为 `[1, 0, 2]`，原生输出顺序为 `[0, 1, 2]`，与 pre 的显示 track 对齐。

这里的“原版失败”不是说它没有任何整数 ID，而是说 native detection index 不保证跨 shot 的持久语义；不同 shot 的 detection order/ID 可以变化。若直接把 native index 当作 persistent person ID，就会把不同的人写到同一条轨迹上。

### 41-cut 控制集证据

文件：`output/v14/b0_identity_matching/v14_b0_identity_matching.json`。

协议是 63 个候选 cuts 中 41 个 pre/post 可见人数完全相同的 controlled cuts，GT identity 只用于 evaluator，matcher 本身不读 GT。

| Boundary | Root matcher all-correct | Torso matcher all-correct | Root+torso+joints all-correct |
|---|---:|---:|---:|
| direct（原始 Human3R，不经 B0） | 41.5% | 41.5% | 43.9% |
| learned B0 后 | **100%** | **100%** | **100%** |

该结果支持把 identity association 写成论文贡献，但应把贡献定义为“boundary-conditioned permutation-aware association / identity-preserving gauge correction”，而不是普通 tracking：它利用 B0 把跨 shot 的 gauge 差异先压缩，再用 root/torso/centered-joints 的匿名 Hungarian 关联保持 persistent IDs。

## 5. 对 ICLR 主线的决定

1. **保留 V9/B0。** 消融不能删掉它；三人 174° 案例会从 0.054 m/1.82° 退化到约 4.27 m/151.89°。
2. **不要把 V9 写成最终精对齐。** 它的准确定位是 learned coarse gauge proposal、低纹理下的先验和 identity preconditioner；最终精修仍由 adaptive camera-human joint 和 BRTC-LC 完成。
3. **加入人物 ID matching 作为独立贡献。** 主张“原版没有可靠 persistent cross-shot association”，并用 direct vs B0 的 41-cut 消融证明，而不是声称原版没有 `smpl_id` 字段。
4. **下一步实验必须拆开三种因素：** `raw Human3R`、`raw + explicit ID association`、`B0 + ID association`、`B0 + BRTC`、`B0 + adaptive joint`。多人要报告 camera error、fixed-ID MPVPE、best-permutation MPVPE、ID switch/track continuity；单人要报告 camera-human joint error。
