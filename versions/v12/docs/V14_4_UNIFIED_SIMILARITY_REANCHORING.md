# V14.4 Unified Projection-Consistent Similarity Re-anchoring

## Executive Conclusion

V14.4 在同一套 180-cut 协议中重新评测了 V11.1、V11.4、V14.3，并实现了真正的
`Shared Shot Scale + Coupled Root + One Boundary`。实验没有支持把 Human Projection
Unified 作为最终主方法：它虽然把 torso 重投影误差从 `19.2 px` 降到 `6.6 px`，但相机、
人体绝对位置和场景同时明显退化。

最高精度、统一 Conditional VGGT rotation tail 下的主要结果为：

| Method | Camera T | Rotation | Human root | Joints | Scene | Reprojection |
|---|---:|---:|---:|---:|---:|---:|
| V11.4 Uniform Similarity | **0.403 m** | 12.09 deg | **0.163 m** | **0.216 m** | **0.532 m** | 19.2 px |
| Unified Human Projection | 0.674 m | 12.09 deg | 0.364 m | 0.381 m | 0.721 m | **6.6 px** |
| Unified DA3 Scale + Root | 0.435 m | 12.09 deg | 0.184 m | 0.248 m | 0.557 m | 16.9 px |

因此当前最终选择仍是：

```text
V11.4 Uniform Similarity + Conditional VGGT rotation tail
```

V14.4 的主要价值是得到一个清楚的必要性结论，而不是得到更好的最终数字：

1. camera、scene、human 共用一个 gauge 的实现原则是正确的，数值卫生检查全部通过；
2. 简单把 V11.4 和 V14.3 顺序叠加会发生 double correction，明显失败；
3. 但当前 Human3R 的 human projection root 与 scene pointmap 不能由一个共享 scalar
   同时校准；
4. DA3 最有价值的部分是 absolute human root depth，不是 pre/post relative shot scale；
5. V14.2 continuity 可以安全放在 alignment 后，但只提供轻量连续性正则，不改变本轮
   alignment 决策。

换成通俗说法：统一缩放整个新镜头这件事本身没有错；问题是 Human3R 当前算出的人体
深度和背景深度并不是只差同一个倍数。强行用人体投影去决定共同位置，会让人体在 RGB
里更贴，但会把本来较准的三维相机、人体和背景一起拉错。

## 1. Experiment Scope

本轮只研究 cut 发生后的第一次流式重对齐，不训练 Human3R，也不修改正常视频路径。
所有可部署版本满足：

- 使用 GT cut index 作为触发信号，不使用 GT 求可部署结果；
- 第一张 post-cut 图像解码前 hard reset；
- Human3R 冻结；
- 只读取已经到达的 pre-cut 历史和第一张 post-cut 图像；
- 不访问完整未来 shot；
- 不做 BA、全局轨迹优化或逐帧修正；
- 每个新 shot 只计算一次 scale、rotation 和 translation；
- 整个 post-cut shot 固定复用同一个 Boundary；
- 正常无 cut 帧完全执行原始 Human3R；
- GT scale 和 Boundary 只出现在显式 Oracle 中。

主实验在 `cuda:5` 上完成。180 个 real cross-camera cuts 的构成为：

| Source | Cuts |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |
| Total | 180 |

所有方法共享同一 Human3R cache、crop/resize、intrinsics、camera-to-world convention、
GT gauge conversion 和人体有效样本。Scene 采用共同有效子集，180 个 cut 中有 147 个
对所有主要方法都有效；foot-scene 配对有 154 个有限样本。任何 scene mean 都只在同一
147-case 子集上计算。

## 2. Unified Geometry

### 2.1 Common pre-shot gauge

先固定 cut 前最后一帧的旧世界 gauge。每个方法都从同一个旧世界开始，避免某个版本
因为使用了不同旧尺度而获得不公平优势。

post-cut shot 只允许一个尺度 `s`、一个旋转 `R`、一个平移 `t`：

```text
scene_world = R * (s * scene_local) + t
camera translation inside shot *= s
body_centered_scaled = s * body_centered_local
human_world = R * (body_centered_scaled + r_calibrated) + t
```

其中 `r_calibrated` 是同一 scaled camera gauge 内的人体 root。它只求一次，并同时用于：

- 根据旧世界人体 anchor 反求 camera translation；
- 放置最终 SMPL-X root、joints 和 vertices。

这修复了 V14.3 之前 camera 使用 calibrated root、最终 human 使用 raw root 的方程矛盾。

### 2.2 Projection-consistent scaling

如果 camera-frame root、root-centered joints、vertices 和 scene points 同时乘 `s`，
透视投影中的 `x/z`、`y/z` 不变。因此 shared scale 本身不应该移动 RGB 上的人体。

本轮实测最大投影不变误差为 `9.35e-6 px`，说明缩放原点和坐标 convention 正确。

### 2.3 Translation solve

固定 `R`、`s` 和 `r_calibrated` 后，只显式计算一次：

```text
t = pre_cut_human_world_anchor - R * r_calibrated
```

最终 human world root 与右侧方程的最大闭环误差为 `3.21e-7 m`。没有额外 foot
translation，也没有第二次 root correction。

## 3. Rotation Fairness

实验分成两层。

第一层是核心几何比较：所有方法统一使用 Fixed Explicit coarse rotation、V16
torso-motion correction 和 `20 deg` bound，不使用 VGGT。该层只比较 scale、root 和
translation。

第二层是最高精度比较：完整方法统一使用同一个 Conditional VGGT tail。触发规则、
输入和预算完全一致，触发率为 `18.89%`。它只替换困难样本的 rotation branch，不读取
source ID，也不修改 scale/root 规则。

## 4. Compared Methods

### 4.1 Retained baselines

- **Fixed Explicit**：hard reset 后的显式粗对齐基线。
- **V11.1 Conditional Wide Raw Scale**：V16 为主，困难 rotation tail 使用 VGGT；不做
  post-shot scale correction。
- **V11.4 Uniform Similarity**：使用 cut-time scale cue，对 camera translation、
  pointmap、human root、body offsets、joints 和 vertices 统一缩放。

### 4.2 V14.3 baselines

- **V18 Camera-Only**：人体投影 root 只用于 camera，最终 human 仍用 raw root。
- **V18 Coupled**：同一个 human-projection root 同时用于 camera 和 human。
- **DA3 Camera-Only / Coupled**：把投影 root 换成 DA3 absolute root depth。

### 4.3 V14.4 candidates

- **Naive Sequential**：先做 V11.4，再把独立 V14.3 correction 加上去，是 double
  correction 负面对照。
- **Unified Human Projection**：先把整个 shot 放进 V11.4 shared-scale gauge，再在这个
  gauge 中重新求一次 human projection root。
- **Unified Relative-DA3 Scale + Human Root**：DA3 只提供 post/pre scale ratio，root
  仍由人体投影求。
- **Unified DA3 Absolute Scale + Human Root**：DA3 提供绝对 shot scale，root 仍由人体
  投影求。
- **Unified V11 Scale + DA3 Root**：V11.4 scale 加 DA3 absolute root depth。
- **Unified DA3 Scale + DA3 Root**：DA3 同时提供 absolute scale 和 root depth，但所有
  量仍共同使用一个 scale。
- **Unified + Continuity**：alignment 结束后才应用 V14.2 shape/scale `alpha=0.25`、
  local pose `alpha=0.15`，world quantities Align-Then-Commit。

### 4.4 Oracles

- **GT Shared Scale Oracle**：GT 只选择一个共同 scalar；
- **GT Human Scale Oracle**：只优化 human/camera 目标；
- **GT Scene Scale Oracle**：只优化 scene；
- **GT Separate Human/Scene Scale Oracle**：允许 human 和 scene 使用不同尺度，仅作诊断；
- **Boundary Oracle**：GT camera Boundary 上界。

Separate-scale Oracle 不可部署，也不满足统一世界要求。它只回答“当前误差是否真能用
一个 scalar 解释”。

## 5. Unified Core Results

下面是统一 V16 rotation、无 VGGT 的核心公平比较：

| Method | Cam T | Rot | Root | Joints | Vertices | Scene | Reproj | Foot-scene | Joint success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 0.712 | 24.20 | 0.234 | 0.290 | 0.285 | **0.483** | 19.2 | 0.169 | 42.8% |
| V16 raw relative scale | 0.518 | 16.04 | **0.163** | **0.223** | **0.215** | 0.526 | 19.2 | 0.169 | 60.6% |
| V11.4 Uniform Similarity | **0.463** | 16.04 | **0.163** | 0.225 | 0.218 | 0.536 | 19.2 | 0.168 | **66.1%** |
| V14.3 V18 Coupled | 0.730 | 16.04 | 0.364 | 0.385 | 0.381 | 0.718 | **6.6** | 0.160 | 25.0% |
| V14.3 DA3 Coupled | 0.491 | 16.04 | 0.184 | 0.256 | 0.249 | 0.578 | 18.1 | 0.231 | 62.8% |
| Naive Sequential | 1.025 | 16.04 | 0.589 | 0.603 | 0.600 | 0.943 | 62.0 | 0.287 | 20.0% |
| Unified Human Projection | 0.712 | 16.04 | 0.364 | 0.388 | 0.384 | 0.726 | **6.6** | **0.160** | 24.4% |
| Unified V11 Scale + DA3 Root | 0.491 | 16.04 | 0.184 | 0.257 | 0.250 | 0.568 | 17.8 | 0.211 | 62.8% |
| Unified DA3 Scale + DA3 Root | 0.491 | 16.04 | 0.184 | 0.256 | 0.250 | 0.564 | 16.9 | 0.187 | 62.8% |

主要观察：

1. V16 rotation 本身把 Fixed camera translation 从 `0.712 -> 0.518 m`，rotation 从
   `24.20 -> 16.04 deg`，是必要组件。
2. V11.4 shared scale 继续把 camera 从 `0.518 -> 0.463 m`，且 root 保持 `0.163 m`、
   投影完全不变，是 effect-first 主体。
3. Coupled Human Projection 把投影降到 `6.6 px`，但 camera/root/scene 同时变差；这不是
   完整三维成功。
4. DA3 root 比 human projection root 稳定得多。它把 Unified root 从 `0.364 -> 0.184 m`，
   但仍没有超过 V11.4 的 `0.163 m`。
5. Naive Sequential 所有主要指标都明显最差，证明不能在 V11.4 后再独立叠加一次 root
   correction。

## 6. Highest-Precision Tail Results

统一加入 Conditional VGGT 后：

| Method | Cam T mean/median/P90/P95 | Rot mean/P95 | Root | Joints | Scene | Reproj | Joint success |
|---|---:|---:|---:|---:|---:|---:|---:|
| V11.4 + VGGT | **0.403/0.334/0.753/0.910** | 12.09/37.75 | **0.163** | **0.216** | **0.532** | 19.2 | **73.3%** |
| Unified Human + VGGT | 0.674/0.676/1.019/1.114 | 12.09/37.75 | 0.364 | 0.381 | 0.721 | **6.6** | 27.8% |
| Unified DA3 + VGGT | 0.435/0.340/0.824/1.064 | 12.09/37.75 | 0.184 | 0.248 | 0.557 | 16.9 | 67.2% |

V11.4 与 Unified Human 的 paired comparison：

- camera 平均退化 `+0.271 m`，73.9% 样本产生超过 `0.05 m` 的 harmful correction，
  Wilcoxon `p=3.81e-19`；
- root 平均退化 `+0.201 m`，harmful rate 76.7%，`p=2.23e-20`；
- joints 平均退化 `+0.165 m`，`p=4.39e-19`；
- scene 平均退化 `+0.189 m`，`p=8.94e-11`；
- torso reprojection 改善 `-12.62 px`，180/180 样本改善。

这说明 Human Projection Unified 学到的是“让人体投影贴图像”的解，但这个解与当前
Human3R 三维 camera/scene gauge 不一致。

V11.4 与 Unified DA3 的 paired comparison 更接近：

- camera `+0.032 m`，`p=0.169`，没有显著差异；
- root `+0.022 m`，`p=0.753`，没有显著差异；
- joints `+0.032 m`，`p=4.47e-6`，显著更差；
- scene `+0.024 m`，`p=0.0219`，显著更差；
- reprojection `-2.33 px`，`p=0.0477`；
- foot-scene `+0.017 m`，`p=1.46e-9`，显著更差。

所以 Unified DA3 是接近 V11.4 的第二候选，但没有稳定额外价值，不满足最终主线的
“camera/scene 不损失、human 明显提高”标准。

## 7. Per-Source Results

下表均使用相同 Conditional VGGT tail，顺序为 `V11.4 / Unified Human / Unified DA3`：

| Source | Camera T | Human root | Joints | Scene | Reprojection |
|---|---:|---:|---:|---:|---:|
| AvatarReX | **0.209** / 0.866 / **0.198** | 0.106 / 0.518 / **0.091** | **0.208** / 0.541 / 0.235 | 0.611 / 1.104 / **0.608** | 19.0 / **7.0** / 22.2 |
| MVHuman100 | **0.547** / 0.696 / 0.577 | 0.207 / 0.380 / **0.201** | **0.230** / 0.380 / 0.235 | **0.184** / 0.433 / 0.262 | 20.3 / **5.5** / 14.6 |
| MVHuman200 | **0.615** / 0.642 / 0.757 | **0.250** / 0.287 / 0.360 | **0.311** / 0.323 / 0.408 | **0.251** / 0.312 / 0.320 | 30.8 / **8.0** / 21.7 |
| THuman | 0.293 / 0.484 / **0.287** | **0.110** / 0.252 / 0.128 | **0.139** / 0.265 / 0.154 | 0.780 / **0.722** / 0.780 | 9.7 / **6.3** / 10.3 |

Unified Human 在 AvatarReX、MVHuman100 和 THuman 的 camera/root 明显退化，只在
THuman scene 和所有 source 的 2D projection 上有优势，不满足三源同向改善。

Unified DA3 在 AvatarReX 最好，THuman 基本持平，MVHuman100 略差，MVHuman200 明显
退化。它没有系统性破坏 THuman，但也没有解决 MVHuman200，不能用 source-independent
规则替代 V11.4。

## 8. Scale Cue Ablation

| Scale/root cue | Camera T | Root | Joints | Scene | Reproj | Height/GT |
|---|---:|---:|---:|---:|---:|---:|
| No relative correction | 0.518 | **0.163** | **0.223** | 0.526 | 19.2 | 0.828 |
| V11.4 shared scale + raw root | **0.463** | **0.163** | 0.225 | 0.536 | 19.2 | 0.831 |
| V11.4 scale + human projection root | 0.712 | 0.364 | 0.388 | 0.726 | **6.6** | 0.831 |
| Relative DA3 scale + human root | 0.716 | 0.364 | 0.388 | 0.726 | **6.6** | 0.825 |
| Human relative scale + human root | 0.748 | 0.364 | 0.390 | 0.726 | **6.6** | 0.829 |
| DA3 absolute scale + human root | 0.721 | 0.364 | 0.391 | 0.734 | **6.6** | 0.859 |
| V11.4 scale + DA3 root | 0.491 | 0.184 | 0.257 | 0.568 | 17.8 | 0.831 |
| DA3 absolute scale + DA3 root | 0.491 | 0.184 | 0.256 | 0.564 | 16.9 | 0.859 |

结论：

- DA3 pre/post relative scale 相比 V11.4 scale 基本没有新增价值；
- Human projection 更适合提供 2D-consistent root，但当前不能作为 world-scale cue；
- DA3 absolute root depth 是 DA3 最稳定、最有用的量；
- DA3 absolute body/scene scale 只把 scene `0.568 -> 0.564 m`、reprojection
  `17.8 -> 16.9 px`，收益很小；
- 因而 DA3 当前应定位为 cut-time absolute root-depth cue，而不是 relative shot-scale
  模块。

## 9. One-Scalar Oracle Diagnosis

| Oracle | Camera T | Root | Joints | Scene | Reproj | Height/GT |
|---|---:|---:|---:|---:|---:|---:|
| GT shared composite scalar | 0.502 | 0.364 | **0.381** | 0.800 | 6.6 | 0.931 |
| GT human scalar | **0.470** | 0.364 | 0.384 | 0.870 | 6.6 | 0.989 |
| GT scene scalar | 1.001 | 0.364 | 0.423 | 0.586 | 6.6 | 0.621 |
| GT separate human/scene scalars | **0.470** | 0.364 | 0.384 | **0.430** | 6.6 | 0.989 |

从 GT shared scalar 切换到 separate human/scene scalars：

- scene 平均改善 `0.370 m`；
- 147 个有效 pair 中 92.5% 改善；
- Wilcoxon `p=1.01e-24`；
- camera 还改善 `0.032 m`；
- joints 只退化 `0.0028 m`。

但 separate scales 会把 foot-scene 平均距离恶化 `0.220 m`，因为人和场景已经不再是
同一个 metric world。它是诊断 Oracle，不是可部署方案。

这组结果说明：在当前 Human Projection coupled formulation 中，即使知道 GT 后再选
一个 scalar，也不能同时得到好的人体和场景；human 与 pointmap 的局部误差不是一个
纯粹的 shot-wide scale factor。可能来源包括 view-dependent depth、spatially varying
pointmap error，或 Human3R 的 human root 与 scene depth 本来就在不同的局部偏差中。

因此触发停止条件 B：停止把“一个 shared scalar + human projection root”继续当成最终
主线。这里停止的是当前 coupled scalar 方案，不是否定 V11.4 对完整 Human3R geometry
进行 uniform scaling 的工程价值。

## 10. Naive Sequential vs Unified

Naive Sequential 相比正确 Unified Human core：

- camera `1.025 -> 0.712 m`；
- root `0.589 -> 0.364 m`；
- joints `0.603 -> 0.388 m`；
- scene `0.943 -> 0.726 m`；
- reprojection `62.0 -> 6.6 px`；
- foot-scene `0.287 -> 0.160 m`。

paired improvement rate 分别为 camera 76.1%、root 77.8%、joints 78.9%、scene
85.7%。主要指标 Wilcoxon 均显著。

所以正确统一方程明显优于简单串联。V14.4 formulation 确实消除了 double scale 和
double root correction，只是当前 human projection cue 本身不够适合三维共同 gauge。
不能因为 Unified 没赢 V11.4，就反过来说顺序拼接也足够。

## 11. Geometry Sanity Checks

| Check | Result |
|---|---:|
| Camera-human equation closure max | `3.21e-7 m` |
| Homogeneous projection invariance max | `9.35e-6 px` |
| Projection-root scale homogeneity max | `4.77e-7 m` |
| Camera/scene/root/body use same `s` | 180/180 true |
| Root calibration count | exactly 1 |
| Extra foot/contact translation | none |
| No-cut camera max diff | `0` |
| No-cut pointmap max diff | `0` |
| No-cut SMPL-X max diff | `0` |

这些检查说明 Unified 失败不是代码把 root 缩了两次、缩放中心错了或 camera/human 方程
没有闭合。它是 cue/gauge 的实际数据问题。

## 12. Conditional VGGT Contribution

Conditional VGGT 将所有完整版本的 rotation mean 从 `16.04 -> 12.09 deg`，P95 统一为
`37.75 deg`。在 Unified Human 中：

- camera `0.712 -> 0.674 m`；
- root 完全不变；
- joints 改善 `0.0067 m`；
- scene 改善 `0.0049 m`；
- reprojection完全不变。

这符合其预期职责：只救 rotation tail，不估计 scale，不替代 coupled root，也不决定
scene gauge。它没有掩盖本轮 scale/root 对比。

## 13. Continuity Memory

在 V16 Unified alignment 后加入 V14.2 continuity：

- camera、root、scene 精确不变；
- joints 平均改善 `0.00118 m`，65.6% 样本改善，`p=1.28e-4`；
- reprojection只变化 `-0.017 px`，无工程意义；
- foot-scene 变化 `+0.00138 m`，很小。

这再次证明 continuity 可以在 alignment 结束后安全叠加，但不参与 scale 或 Boundary
求解。其主要证据仍来自 V14.2：shape jump -22.3%、scale jump -23.2%、local-pose
residual -14.8%，8-cut shape drift `0.582 -> 0.484`。

本轮没有把 continuity 与 Conditional VGGT 重新组合成一个额外 180-cut 方法，因为两者
职责正交：VGGT 只改变统一 Boundary rotation，continuity 不改变 camera/root/scene
anchor。报告不把这种可组合性虚构成新的实测 alignment gain。

## 14. Multi-Cut Diagnostic

当前 V14.4 的 1/2/4/8-cut 结果是把不同 held-out single-cut error transforms 依次组合的
**误差传播诊断**，不是在一条真实连续长视频上重新运行 Human3R。8-cut camera drift：

| Method | 1 cut | 2 cuts | 4 cuts | 8 cuts |
|---|---:|---:|---:|---:|
| V11.4 | 0.421 | 0.637 | 0.967 | **1.459** |
| Unified Human | 0.692 | 1.196 | 2.037 | 3.674 |
| Unified V11 Scale + DA3 Root | 0.459 | 0.678 | 1.005 | 1.532 |
| Naive Sequential | 1.025 | 1.813 | 3.196 | 5.845 |

该诊断支持单 cut 排名：V11.4 最稳，DA3-root Unified 接近，Human Projection Unified
和 Naive 累积更快。但它不能替代真正的 contiguous multi-cut rollout，论文中不能把它
写成真实长视频实验。V14.2 的 memory multi-cut 是独立的 recurrent memory replay，见
`output/v14_2_canonical_human_memory/multicut_replay/`。

## 15. Streaming and Runtime

| Item | Result |
|---|---:|
| GPU | `cuda:5` |
| 180-cut evaluator wall time | 576.83 s |
| Cached-cue evaluator mean | 3.20 s/cut |
| DA3 six-frame latency mean/median/P90/P95 | 0.185/0.173/0.220/0.244 s |
| Conditional VGGT trigger rate | 18.89% |
| Peak tracked PyTorch allocation | 144.5 MB |
| Normal-frame added latency | 0 |

`3.20 s/cut` 主要包含 CPU scene nearest-neighbor metric 和多个 GT Oracle scalar search，
不是部署推理 latency。DA3 的真实缓存测量为一次 5 pre + 1 post 推理；正常帧不运行
DA3/VGGT/Oracle。Scale 和 Boundary 每个 shot 只计算一次。

## 16. Visualization

交互式三维 viewer：

```text
http://127.0.0.1:8106
```

默认样本：

```text
mvhuman200_120_150_200002_410_22327109_22236235
```

viewer 使用 10 帧 cut 前 + 10 帧 cut 后，固定第三方世界视角，并排显示：

- V11.4 Uniform Similarity；
- Unified Human Projection；
- Unified DA3 Root；
- Naive Sequential。

每个版本同时包含 camera frustum、scene pointmap、SMPL-X mesh、joints 和 root trajectory，
不是跟随相机的 RGB overlay。四个 source 的成功、V11.4 更好、Human Projection 更好、
DA3 更好和 naive double-correction 案例索引位于：

```text
output/v14_4_unified_similarity_reanchoring/visualization/case_selection.json
```

默认四个版本均验证了 20 帧、51,104 scene points、有限 camera/vertices 和非零几何范围：

```text
output/v14_4_unified_similarity_reanchoring/visualization/geometry_validation.json
```

当前环境没有成功生成浏览器截图：Python Playwright 不存在，磁盘根分区不足以安装 npx
浏览器依赖，Firefox headless 也无法初始化多 GPU WebGL。HTTP viewer 和实际三维数组已经
完成验证，不影响在本机浏览器直接查看。

## 17. Why Old Reports Look Different

旧报告中的 V11.4、V14.3 使用了不同指标和 gauge，不能把数字直接横向比较。本轮统一后：

| Method | Old report | Unified protocol | Main reason |
|---|---:|---:|---|
| V11.4 camera T | 0.397 | 0.403 | 新值使用共同 180-case gauge 和相同 VGGT tail，差异很小 |
| V11.4 scene | 0.302 | 0.532 | 旧 scene trimmed metric 与本轮共同双向 pointmap metric 不同 |
| V14.3 V18 camera/root | 0.872/0.444 | 0.730/0.364 | 共同 pre-shot gauge、target conversion 和有效定义改变 |
| V14.3 V18 scene | 0.998 | 0.718 | scene metric 和有效子集改变 |
| V14.3 DA3 camera/root | 0.518/0.220 | 0.491/0.184 | 共同 gauge 后重新计算 |
| V14.3 DA3 scene | 1.382 | 0.578 | 旧 scene discontinuity 与本轮共同 pointmap metric 不同 |

V11.4 camera 从 `0.397` 到 `0.403 m` 说明它的主要 camera 结论稳定。Scene 和 V14.3 的
较大数值变化主要来自 gauge、scene metric 和有效子集，不是算法突然提高。无法把全部
差异压成一个“有多少百分比来自指标”的数字；正确做法是从本轮开始只引用统一协议表。

另一个关键变化是指标含义：旧 V11.4 的 `human relative-motion error` 不是本轮的
`absolute human world-root error`。统一协议下 V11.4 的 absolute root 为 `0.163 m`，
不能继续拿旧 relative-motion 数字与 V14.3 absolute root 横比。

## 18. Final Answers

1. **V11.4 和 V14.3 各自强在哪里？** V11.4 真正强在 camera、absolute human 和
   scene/contact 的完整三维效果；V14.3 Human Projection 真正强在 camera-human 方程
   闭环和 RGB 2D torso projection。V14.3 DA3 则强在独立 absolute root depth。
2. **旧数值差异有多少来自指标或 gauge？** V11.4 camera 差异只有 `0.006 m`；scene 和
   V14.3 的大变化主要来自定义、gauge 和有效子集。没有统一百分比，后续应只引用本轮
   统一结果。
3. **一个 shared shot scale 是否足够？** 对当前 Human3R human-projection root 与 scene
   pointmap 不够。Separate-scale Oracle 的 scene 比 shared Oracle 低 `0.370 m`，92.5%
   pair 改善，但会破坏统一世界和 foot contact。
4. **Coupled root 能否在 shared gauge 保留 human accuracy？** 它精确保留 V14.3 自身的
   root 数值和方程闭环，例如 V18 均为 `0.364 m`；但这个 accuracy 本身不如 V11.4 raw
   Human3R root 的 `0.163 m`，所以没有形成新的总体 human gain。
5. **Shared scale 是否修复 V14.3 scene？** Human Projection 没有，scene 从 V14.3
   `0.718` 到 Unified `0.726 m`；DA3 absolute scale 只从 `0.578` 小幅到 `0.564 m`，
   不足以称为完整 scene closure。
6. **Unified 是否优于 Naive Sequential？** 明显优于。Camera/root/scene 分别改善约
   `0.312/0.225/0.217 m`，并消除了 62 px 的 double-correction 投影失败。
7. **DA3 最适合提供什么？** 当前最适合提供 cut-time absolute human root depth。
   Relative shot scale 几乎无新增价值，absolute scene/body scale 只有很小收益。
8. **Conditional VGGT 是否仍只负责 rotation tail？** 是。Rotation `16.04 -> 12.09 deg`，
   root 与 reprojection 不变，只间接小幅改善 camera/joints。
9. **Continuity 能否安全叠加？** 能，但必须在 alignment 后使用并 Align-Then-Commit。
   Camera/root/scene 保持完全不变，joints 只有约 `1.2 mm` 小幅改善。
10. **最终选哪个？** 选择 **V11.4 effect-first + Conditional VGGT**。V14.3 coupled
    equation 保留为必要性/一致性消融；V14.4 Unified 是重要负结果和诊断，不升级为主线。

## 19. Route Decision

本轮对应题目中的情况 B、D、F、G 的组合：

- **B**：GT shared scalar 都不能统一 human 和 scene，停止当前 single-scalar coupled
  projection 主线；
- **D**：Unified 与 V11.4 相比没有 human gain，因此主方法回到 V11.4 effect-first；
- **F**：Naive 明显更差，证明 Unified formulation 不是简单包装；
- **G**：Unified 确实优于 Naive，但当前 cue 仍不足以超过 V11.4。

最终部署骨架：

```text
GT/automatic cut trigger
-> pre-decode Human3R hard reset
-> Fixed Explicit + V16 torso rotation, 20 deg bound
-> Conditional VGGT only for the difficult rotation tail
-> V11.4 one cut-time uniform shot scale
-> one fixed Boundary for camera, pointmap and complete SMPL-X
-> optional V14.2 continuity after alignment
-> Align-Then-Commit for world human memory
```

论文中可以将 V14.4 作为严谨的结构验证：projection consistency 是必要条件，但不是
充分条件；当前 Human3R 的人和场景误差不能用一个简单共同 scalar 完整解释。不能把它
描述成已经优于 V11.4 的最终方法。

## 20. Artifacts

代码：

```text
versions/v12/experiments/v14_4_unified_similarity_reanchoring_probe.py
versions/v12/experiments/v14_4_interactive_unified_viewer.py
```

机器可读结果和自动总表：

```text
output/v14_4_unified_similarity_reanchoring/full180_final/
  v14_4_unified_similarity_reanchoring.json
  v14_4_unified_similarity_reanchoring.md
```

可视化：

```text
output/v14_4_unified_similarity_reanchoring/visualization/
```

复现实验：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v12/experiments/v14_4_unified_similarity_reanchoring_probe.py \
  --device cuda:5 \
  --output_dir output/v14_4_unified_similarity_reanchoring/full180_final
```

启动三维 viewer：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v12/experiments/v14_4_interactive_unified_viewer.py \
  --device cuda:5 --port 8106
```
