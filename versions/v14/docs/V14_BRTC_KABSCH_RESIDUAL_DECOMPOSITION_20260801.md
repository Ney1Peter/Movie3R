# BRTC + individual Kabsch：EgoHumans 残差分解

> 全程只读复用 `current_v14_cpu_geometry.pt`；没有运行 Human3R、DA3、GPU 或新预训练模型。
> GT 只在 evaluator 中用于误差归因，未进入匹配、gate、修正或候选选择。
> 这是 3 条自建 15-frame chain 的 provisional 协议，不是 Multi-THuMBS 未公开的官方 split。

## 1. 指标口径

- `fixed root/joint`：首帧固定 gauge 后直接比较世界坐标；root 是 SMPL-X→SMPL 后的 `joint 0`。
- `W`：每个 GT identity 用最早两个可见帧拟合一个 Sim(3)，随后固定应用到整条轨迹。
- `WA`：每个 GT identity 用整条可见轨迹拟合一个 Sim(3)。
- `pelvis raw`：按 GVHMR 口径用 SMPL joints 1/2 均值去中心后直接 MPJPE。
- `pelvis SO(3)`：每帧允许 oracle 全局旋转，不允许平移和缩放。
- `pelvis PA`：每帧允许 oracle Sim(3)，是 articulation/shape floor。

## 2. 按镜头段的主要误差（mm）

| Method | Bucket | W | WA | Fixed root | Fixed joint | Pelvis raw | SO(3) | PA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| brtc_v1 | segment0_pre | 104.531 | 240.625 | 452.525 | 444.077 | 113.490 | 86.197 | 76.280 |
| brtc_v1 | segment1_post | 373.734 | 164.036 | 325.642 | 339.240 | 110.281 | 68.752 | 59.801 |
| brtc_v1 | segment2_post | 458.549 | 197.575 | 357.958 | 366.037 | 120.114 | 80.869 | 68.947 |
| brtc_v1 | boundary_first_post | 411.305 | 184.453 | 352.249 | 362.745 | 116.178 | 75.360 | 64.609 |
| brtc_v1 | all | 314.059 | 202.461 | 380.654 | 384.729 | 115.025 | 79.169 | 68.786 |
| brtc_kabsch | segment0_pre | 104.531 | 240.931 | 452.525 | 444.077 | 113.490 | 86.197 | 76.280 |
| brtc_kabsch | segment1_post | 374.158 | 160.691 | 326.002 | 335.714 | 87.776 | 68.752 | 59.801 |
| brtc_kabsch | segment2_post | 454.750 | 193.358 | 357.770 | 366.638 | 112.322 | 80.869 | 68.947 |
| brtc_kabsch | boundary_first_post | 409.805 | 179.954 | 352.268 | 361.745 | 101.656 | 75.360 | 64.609 |
| brtc_kabsch | all | 312.769 | 200.029 | 380.688 | 383.933 | 105.618 | 79.169 | 68.786 |

## 3. Kabsch 相对 BRTC v1 的分段变化（mm）

| Bucket | W | WA | Fixed root | Fixed joint | Pelvis raw | SO(3) | PA |
|---|---:|---:|---:|---:|---:|---:|---:|
| segment0_pre | +0.000 | +0.306 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cut0_pre_last | +0.000 | -1.524 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cut0_first_post | +1.126 | -3.889 | +0.245 | -3.527 | -24.052 | -0.000 | -0.000 |
| segment1_post | +0.423 | -3.344 | +0.360 | -3.526 | -22.505 | -0.000 | -0.000 |
| cut1_pre_last | +0.380 | -4.073 | +0.166 | -3.359 | -22.994 | -0.000 | -0.000 |
| cut1_first_post | -3.543 | -4.973 | -0.157 | +0.965 | -7.110 | +0.000 | +0.000 |
| segment2_post | -3.799 | -4.218 | -0.188 | +0.601 | -7.792 | +0.000 | +0.000 |
| post_all | -1.952 | -3.836 | +0.052 | -1.204 | -14.229 | +0.000 | +0.000 |
| boundary_first_post | -1.501 | -4.499 | +0.019 | -1.000 | -14.522 | -0.000 | -0.000 |
| all | -1.290 | -2.432 | +0.034 | -0.796 | -9.408 | +0.000 | +0.000 |

## 4. Root：shared/camera 与 individual 分解

逐帧把每个人的 fixed-root 误差向量写成 `shared mean + individual residual`。
`remove shared` 与 `remove camera` 均是 GT evaluator-only oracle ceiling，不是可部署方法。

| Method | Bucket | Total root | Shared norm | Remove shared | Shared squared fraction | Shared-camera cosine | Remove camera |
|---|---|---:|---:|---:|---:|---:|---:|
| brtc_v1 | segment0_pre | 439.848 | 352.549 | 278.272 | 0.595 | 0.502 | 437.842 |
| brtc_v1 | segment1_post | 280.384 | 252.646 | 178.286 | 0.656 | 0.976 | 303.751 |
| brtc_v1 | segment2_post | 357.958 | 328.806 | 129.441 | 0.752 | 0.431 | 355.746 |
| brtc_v1 | boundary_first_post | 335.375 | 300.400 | 154.602 | 0.695 | 0.667 | 339.699 |
| brtc_v1 | post_all | 326.928 | 298.342 | 148.979 | 0.714 | 0.649 | 334.948 |
| brtc_kabsch | segment0_pre | 439.848 | 352.549 | 278.272 | 0.595 | 0.502 | 437.842 |
| brtc_kabsch | segment1_post | 280.804 | 253.484 | 177.039 | 0.660 | 0.975 | 301.863 |
| brtc_kabsch | segment2_post | 357.770 | 328.750 | 129.106 | 0.753 | 0.432 | 355.452 |
| brtc_kabsch | boundary_first_post | 335.395 | 300.716 | 153.889 | 0.698 | 0.668 | 338.775 |
| brtc_kabsch | post_all | 326.984 | 298.643 | 148.279 | 0.716 | 0.649 | 334.016 |

## 5. 剩余误差归因

Kabsch 后 post-shot fixed root 为 `326.984 mm`；oracle 去掉每帧 shared 分量后为 `148.279 mm`，平均下降 `178.705 mm`。
shared 的平方误差解释率为 `71.6%`，shared-camera cosine 为 `0.649`；直接减去 camera error 后为 `334.016 mm`。

Kabsch 后 post-shot pelvis raw / SO(3) / PA 为 `101.583` / `75.568` / `64.946 mm`。
因此额外 oracle orientation 最多再解释 `26.015 mm`，uniform scale 最多解释 `10.622 mm`，而 articulation/shape floor 仍是 `64.946 mm`。这些差值是上界，不可相加为严格因果占比。

本地 W/WA 为 `312.769` / `200.029 mm`，相对论文 EgoHumans 279/166 mm 仍差 `+33.769` / `+34.029 mm`。

## 6. 按 chain / identity 的完整结果

JSON 中保留了每个 chain、identity、segment/cut bucket 的全部统计，以及每条 identity 的 W/WA Sim(3) scale、rotation、translation。这里仅列 identity-track 总表。

| Method | Chain | Identity | Frames | W | WA | Fixed root | Fixed joint | Raw pose | SO(3) | PA |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| brtc_v1 | 0 | aria01 | 15 | 380.839 | 230.846 | 467.732 | 503.832 | 95.723 | 71.432 | 65.904 |
| brtc_v1 | 0 | aria02 | 15 | 170.097 | 136.781 | 214.366 | 233.055 | 84.535 | 68.910 | 53.592 |
| brtc_v1 | 0 | aria03 | 15 | 260.008 | 170.219 | 369.901 | 355.415 | 137.457 | 72.242 | 56.093 |
| brtc_v1 | 1 | aria01 | 12 | 324.852 | 214.529 | 294.030 | 305.072 | 124.311 | 88.312 | 81.640 |
| brtc_v1 | 1 | aria02 | 14 | 101.658 | 91.947 | 106.772 | 170.202 | 84.551 | 79.477 | 62.546 |
| brtc_v1 | 1 | aria03 | 15 | 402.399 | 245.765 | 331.709 | 296.541 | 129.644 | 71.233 | 64.450 |
| brtc_v1 | 2 | aria01 | 10 | 382.136 | 259.879 | 630.594 | 647.922 | 154.351 | 111.836 | 106.124 |
| brtc_v1 | 2 | aria02 | 15 | 525.838 | 283.375 | 512.756 | 487.417 | 118.526 | 79.108 | 74.043 |
| brtc_v1 | 2 | aria03 | 10 | 277.067 | 203.260 | 628.306 | 588.542 | 121.083 | 84.481 | 69.536 |
| brtc_kabsch | 0 | aria01 | 15 | 374.823 | 229.711 | 468.130 | 505.303 | 85.229 | 71.432 | 65.904 |
| brtc_kabsch | 0 | aria02 | 15 | 171.805 | 137.374 | 214.379 | 230.473 | 82.180 | 68.910 | 53.592 |
| brtc_kabsch | 0 | aria03 | 15 | 250.236 | 147.119 | 370.580 | 346.543 | 103.912 | 72.242 | 56.093 |
| brtc_kabsch | 1 | aria01 | 12 | 329.074 | 218.917 | 293.448 | 307.441 | 126.773 | 88.312 | 81.640 |
| brtc_kabsch | 1 | aria02 | 14 | 99.503 | 90.459 | 107.211 | 169.545 | 82.401 | 79.477 | 62.546 |
| brtc_kabsch | 1 | aria03 | 15 | 403.705 | 247.362 | 330.597 | 295.493 | 103.776 | 71.233 | 64.450 |
| brtc_kabsch | 2 | aria01 | 10 | 382.136 | 259.879 | 630.594 | 647.922 | 154.351 | 111.836 | 106.124 |
| brtc_kabsch | 2 | aria02 | 15 | 526.837 | 283.677 | 513.110 | 490.743 | 114.937 | 79.108 | 74.043 |
| brtc_kabsch | 2 | aria03 | 10 | 277.067 | 203.260 | 628.306 | 588.542 | 121.083 | 84.481 | 69.536 |

## 7. 复现

```bash
.venv/bin/python versions/v14/analyze_brtc_kabsch_residual_decomposition.py --self_test
.venv/bin/python versions/v14/analyze_brtc_kabsch_residual_decomposition.py
```

## 8. 明确结论、失败实验与下一候选

### 8.1 现在能确定什么

1. **相机不是当前主要可直接修正项。** post-shot 多人 root 的确有很大的 shared 分量：去掉它的 oracle ceiling 可把 326.984 mm 降到 148.279 mm，shared 占 71.6% 平方误差；但把已知 GT camera-error vector 直接从人体减掉，误差反而变成 334.016 mm。shared human bias 与 camera drift 不是同一个向量，不能再整体改相机。
2. **第一主矛盾是 shared person-root/gauge bias，第二是 individual root。** shared 去除后仍有 148.279 mm，所以只做一个 scene/camera transform 也不够；必须先估计安全的共同人体平移，再保留 individual refinement。
3. **orientation 值得保留，但不是全部答案。** 当前 Kabsch 已将 post-shot raw pelvis pose 降到 101.583 mm；额外 oracle SO(3) 仍最多解释 26.015 mm。Kabsch 的方向正确，但 bounded causal estimate 尚未吃完上限。
4. **scale 不是下一步。** SO(3)→PA 的 uniform-scale oracle ceiling 只有 10.622 mm，且已有独立 body-scale 候选在 EgoHumans fixed joint/vertex 上失败；当前更大的 PA floor 是 64.946 mm，属于 articulation/shape。
5. **W/WA 的约 34 mm 论文差距不是单一模块能补齐。** chain 2 本地 W/WA 是 414.131/253.901 mm，明显主导总体差距；后续必须同时看 chain/identity tail，而不能只优化 aggregate mean。

### 8.2 已验证但淘汰：two-frame group tangent translation

CPU probe 在 BRTC accepted people 上计算 `pre_root - brtc_post_root` 的 post-camera-ray 切平面分量，对至少两人的分量取坐标中位数，再把同一小平移施加到 accepted people；camera、scale、orientation 均不改，rejected/unmatched 为 exact B0。dev-three-offset0 冻结参数为 `fraction=0.2, cap=0.15 m, median dispersion gate=0.1 m, min people=2`。

| Split | Δroot | Δjoint | Δvertex | Δpair dist | Δpair vec | Decision |
|---|---:|---:|---:|---:|---:|---|
| dev three offset0 | -2.640 mm | -2.780 mm | -2.992 mm | +0.000 mm | -0.000 mm | pass |
| three offset1 | -3.531 mm | -3.854 mm | -3.931 mm | +0.000 mm | +0.000 mm | pass |
| dance | **+2.096 mm** | -3.233 mm | -2.163 mm | +0.000 mm | +0.000 mm | **fail** |
| box | -1.062 mm | -0.510 mm | -0.677 mm | +0.000 mm | +0.000 mm | pass |

结论是 **NO_GO_TWO_FRAME_GROUP_TANGENT**。失败不是 cap 不够小，而是可观测性不足：last-pre/current-post 看到的多人共同位移，既可能是 shared root/gauge bias，也可能是多人真实的同向运动。dance 中后者被错误地当成对齐误差。GT oracle projection 只用于分析：individual tangent 与真实 root correction 的 mean cosine 在 dev/three1/dance/box 仅为 0.244/0.225/0.403/0.441，不能构成安全 runtime gate。

该 cross-split 结果不宣称正式 blind validation：探索阶段先打开过 held-out observability summary；实际 policy selection 代码只读取 dev-three-offset0。完整数值保存在 `output/v14/fine_alignment_research/brtc_kabsch_residual_decomposition/GROUP_TANGENT_FEASIBILITY_RESULTS.json`。

### 8.3 唯一明确的下一候选

**Timestamp-aware velocity-residual group tangent translation**：

```text
v_i       = robust velocity from the last 3-5 causal pre-shot roots
anchor_i  = root_pre_i + delta_t * v_i
d_i       = anchor_i - root_brtc_post_i
ray_i     = normalize(root_brtc_post_i - camera_post_center)
tangent_i = d_i - dot(d_i, ray_i) * ray_i
group     = robust_median(tangent_i over accepted matched people)
shift     = bounded_fraction(group), with dispersion gate and small cap
```

`delta_t` 只来自输入帧时间戳；历史只用 cut 前已经看到的 3-5 帧，不读 future、GT、source label 或新模型。同一 group shift 传播到 post shot，之后再串联已 qualified 的 individual Kabsch orientation。它与失败版本相比只增加一个关键可观测量：pre-shot velocity，用来把 coherent human motion 从 shared alignment residual 中先扣掉。

当前 two-frame BRTC runtime API 只把 last-pre/current-post 交给 refinement，因此**在这个接口内已经没有可证明安全的 shared-vs-motion 判别信息**。下一轮应先扩展 causal state 读取 pre-shot root history，再在新 dev split 冻结；因为当前 dance/box 已被打开，不应再把它们称为 blind held-out。
