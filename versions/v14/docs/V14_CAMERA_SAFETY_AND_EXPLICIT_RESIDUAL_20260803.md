# V14 Camera Safety：已关闭分支与显式边界残差实验

日期：2026-08-03

状态：进行中；本文记录已完成的 No-Go 和当前唯一打开的 camera-tail 实验。

对应 ICLR 蓝图的 Phase 4。所有实验保持：first-post-cut only、无 future frame、shadow state 不提交、clean raw state 是唯一 recurrent commit、person refinement 不修改 camera。

## 1. 固定背景

同一 cross96 checkpoint：

```text
output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth
SHA256 05274f7b4841f6ebc73f2f5bdb419d63d272396724db886b6e10987d7210a144
```

在 frozen180 的已建立结果：B0 是合格的粗 gauge proposal，但仍有 `86/180` camera catastrophic。BRTC-LC/Kabsch 能在相机冻结时强力修复人体，却不应承担或掩盖 camera tail。

新旧 B0 的 camera safety 开发集是 `v14_vsp_pair_disjoint_20260802/dev_all.jsonl`：4 个来源各 24 条、共 96 条，pair 与 cross96 train、frozen10、frozen180、VSP confirm 均不重合。

其 cross96 B0 基线：

| N | Composite mean | Composite P95 | Catastrophic |
|---:|---:|---:|---:|
| 96 | 1.98949 | 5.49841 | 49 |

## 2. 已关闭：两套既有隐式 B0 的全局 SE(3) 混合

实现：`versions/v14/select_dual_b0_fixed_se3_mixture.py`。

方法在运行时只使用 cross96 B0 和已严格转入相同 gauge 的 old B0，对所有 cut 共享唯一常数：

```text
B(alpha) = interpolate_SE3(B_cross96, B_old_adapted, alpha)
alpha in {0, .125, ..., 1}
```

rotation 用 geodesic interpolation，translation 线性插值；不使用 GT、selector、future 或 state change。读 confirm 前固定的 Go 条件：非零 alpha、overall mean 至少改善 1%、P95 不变差、catastrophic 至少下降 10%、每个 source 的 mean 与 catastrophic 均不退化。

| alpha | Composite mean | P95 | Catastrophic | 判定 |
|---:|---:|---:|---:|---|
| 0 (cross96) | 1.98949 | 5.49841 | 49 | baseline |
| .125 | 1.98215 | 5.39729 | 51 | 不足 1% 且 tail 更差 |
| .25 | 2.00954 | 5.30763 | 52 | 不通过 |
| .50 | 2.12604 | 5.20312 | 60 | 不通过 |
| 1.0 (old) | 2.41013 | 5.32035 | 66 | 不通过 |

结论：

```text
NO_GO_DUAL_B0_FIXED_SE3_MIXTURE
```

全局混合略微降低了 P95，却增加 disaster，且 MVHuman100/AvatarReX/THuman 至少一个来源均值退化。因此不能再投入“重加权或混合已有隐式 proposal”；这与此前 `NO_GO_DUAL_B0_CAMERA_SELECTOR`、广覆盖 abstention No-Go 的结论一致。

## 3. 已关闭：在 B0 原训练 pair 上训练直接显式残差头

实现：

```text
cache_explicit_boundary_residual_features.py
train_explicit_boundary_residual_probe.py
```

冻结 cross96 模型只产生 causal feature：5 组 pooled 768-D post-cut latent（correction token、raw/corrected pose token、raw/applied delta）和 8 个 B0/pointmap/gate 几何量。总维度 `3848`。

head 预测 B0 右侧的 local `SE(3)` residual：

```text
Delta* = inverse(B0) @ B_gt                 # training only
B_final = B0 @ Delta_hat                    # runtime
```

右乘令 residual 位于 new-shot local gauge；若历史 world 同时左乘任何刚体，`Delta*` 保持不变。输出 trust region 固定为 translation `<=3m`、rotation `<=180°`。没有外部预训练模型，也没有 shadow state commit。

首轮将 B0 自己训练过的 cross96 train96（384 条）作为 head train，VSP dev96 作为 held-out 开发。三种预注册 head：linear、MLP-64、MLP-128。它们在 train 上出现不可信的近乎完全拟合，而 dev 失败：

| Head | train composite | dev composite | dev gain vs B0 | dev catastrophe |
|---|---:|---:|---:|---:|
| B0 | 0.92729 | 1.98949 | — | 49 |
| MLP-64 | 0.15895 | 2.17736 | **−9.44%** | 53 |
| MLP-128 | 0.19677 | 2.22381 | **−11.78%** | 56 |
| Linear | 6.23858 | 6.67720 | −235.62% | 96 |

MLP-64 只在 THuman 得到约 4.9% 的平均改善；AvatarReX、MVHuman100 和 MVHuman200 均退化。预注册的 source / mean / P95 / catastrophic 条件没有一个模型通过，因此：

```text
NO_GO_EXPLICIT_BOUNDARY_RESIDUAL_LATENT_PROBE_ON_B0_TRAIN_PAIRS
```

确认集未读取。科学含义不是“显式 residual 一定无效”，而是 B0 已见的 cross96 pair 对该 supervision 有严重过拟合：B0 train mean `0.927`，独立 dev mean `1.989`，不能用于判断 tail 修复能力。

## 4. 已关闭：pair-disjoint 训练后的 latent-only 直接 residual

新 manifest：

```text
config/manifests/v14_explicit_boundary_residual_pair_disjoint_20260803/
```

每来源 96 条（总 384），排除：

- cross96 B0 train96 camera pairs；
- frozen10 / frozen180 pairs；
- VSP dev pairs；
- VSP confirm pairs。

审计表明每个来源与上述排除集合的 pair overlap 均为 0；相机 pair 数为 AvatarReX 70、其余三来源 96。这个新 train pool 的 B0 error 已与 dev 更接近，而不是旧 B0-train pool 的容易分布：

| split | B0 composite mean | P95 | catastrophic |
|---|---:|---:|---:|
| original cross96 train96 | 0.92729 | 2.99652 | 77 / 384 |
| new residual train96 | 1.81339 | 5.19614 | 186 / 384 |
| VSP dev96 | 1.98949 | 5.49841 | 49 / 96 |

在此 train pool 上保持**完全相同**的 feature、caps、三种 head 和 dev Go 条件，结果仍没有 winner：

| Head | dev composite | dev gain vs B0 | dev P95 | dev catastrophe |
|---|---:|---:|---:|---:|
| B0 | 1.98949 | — | 5.49841 | 49 |
| MLP-64 | 2.05126 | −3.10% | **4.90909** | 60 |
| MLP-128 | 2.05199 | −3.14% | 5.07152 | 57 |
| Linear | 6.47018 | −225.22% | 7.29276 | 96 |

MLP 对 MVHuman100/200 有很小 mean 改善（`1.4%–10.1%`），却在 AvatarReX/THuman 明显退化，并增加每一个/或多个 source 的 catastrophic。P95 变好不能抵消 tail 增加，故：

```text
NO_GO_EXPLICIT_BOUNDARY_RESIDUAL_LATENT_PROBE
```

VSP confirm 仍未读取。两次训练（B0 已见 pairs 与 new pair-disjoint pairs）一致表明：**把单帧 pooled V9/cross96 latent 直接回归为 residual，能拟合 train，但不能在 source/pair 之间泛化为安全 camera action。** 不应再扩展 MLP 深度、宽度或 threshold。

## 5. 已关闭：由现有 dense token/pointmap 直接构造 Kabsch camera proposal

为了找与 pooled latent 正交的证据，额外检查了 Human3R 已有的 image token 和 pointmap，不引入任何外部模型：

```text
last-pre image token <-> first-post clean raw image token
mutual nearest cosine matching
+ their predicted 3D pointmap coordinates
+ fixed RANSAC-Kabsch
-> B_geom (post raw local -> pre world)
```

实现：`probe_dense_token_kabsch_boundary.py`。RANSAC 固定为 256 个互为最近匹配、512 hypotheses、`0.30m` inlier threshold、至少 8 inliers。

先使用 decoder token。它出现表面上很高的 cosine（约 `0.97–0.99`）和高 inlier ratio，却给出错误刚体：这说明 token 被位置/decoder context 主导，不能把 similarity 当 correspondence confidence。随后以仅诊断的 opt-in flag 导出**encoder 前 decoder-state 混合的 image tokens**，规则不变；同样失败。

四来源 smoke（VSP dev 的每源第一条）结果：

| Source | B0 composite | encoder-token `B_geom` composite |
|---|---:|---:|
| AvatarReX | 1.962 | 3.244 |
| THuman | **0.093** | 2.319 |
| MVHuman100 | 0.831 | 6.550 |
| MVHuman200 | 1.971 | 4.850 |

特别是 THuman 已正确 B0 被 geometry candidate 破坏，说明它不能作为 direct replacement 或轻率 fallback。这个四源 counterexample 已足够拒绝“裸 mutual-token + pointmap Kabsch”；不应消耗 VSP confirm 来扩大这种无效 protocol。

代码中新增的 `return_v14_encoder_tokens` 默认为 `False`，只读导出已有 encoder tokens；默认 Human3R forward、state 与输出不变。

## 6. 研究决策：停止当前 camera-tail 候选家族

已完成的证据支持下列边界：

```text
No-Go: shadow root/orientation direct projection
No-Go: old/new B0 causal selector and broad abstention
No-Go: fixed global two-B0 SE(3) mixture
No-Go: pooled latent -> direct SE(3) MLP, with both easy and pair-disjoint hard train
No-Go: bare encoder/decoder token mutual matching + pointmap Kabsch
```

这不推翻 B0：它仍是已冻结、有效的 **coarse gauge**。但在当前 data/model evidence 下，不能写成“camera tail 已被解决”，也不应继续调上述同类方法。

下一优先级相应转为蓝图最能形成 ICLR 闭环、且不会重新假设 B0 能一步到位的部分：

1. 在真正 `max_humans>1`、variable visibility 条件下执行 `B0 -> BRTC-LC -> bounded Kabsch`，确认 camera 严格不变、root/layout/ID 发生安全改善；
2. 用同一 forward 扩展 EgoHumans / Multi-THuMBS-style evaluator，透明报告 `W/WA/MPJPE/MPVPE/Accel/ATE/ID` 与 causality/runtime；
3. 如果未来重新打开 camera tail，必须先提出一个有理论可观测性、与上述 latent/token matching 正交的证据（例如有独立 2D correspondence 保证的 geometry cycle），并先以 signed residual diagnostic 证明信号，再训练/选择。

VSP confirm 继续保持未读；B0、BRTC、Kabsch 的冻结状态不变。
