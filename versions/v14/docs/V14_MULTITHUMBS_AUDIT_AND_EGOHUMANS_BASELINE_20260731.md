# V14 Multi-THuMBS 审计、EgoHumans 对标基线与下一步精对齐主线

> 日期：2026-07-31
>
> 论文：`/data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf`
>
> 论文版本：arXiv:2607.01626v1
>
> 当前状态：已完成论文、源码包、项目页、数据重合、评测协议和两组本地实验审计。

## 1. 最终结论

Multi-THuMBS 与当前问题高度相关，并直接支持我们已经通过 GT 可视化和 180-cut
实验得到的判断：

```text
把相机/shot 坐标系对齐，并不能消除 Human3R 的 camera-local 人体 root、深度、尺度和
global orientation 误差。一个共享 SE(3) Boundary 不会改变人体与相机的相对结构。
```

论文的解决方式不是只继续修相机，而是在 shared scene 中对每个人独立修改 root
translation 和 global orientation。它使用人体 mask 内的 VGGT pointmap、2D joints、
silhouette 和 depth loss 做迭代优化，然后再传播到完整 shot。

这给 V14 的主线提供了明确方向：

```text
B0 固定相机粗对齐
→ 可选 DA3 相机细修，但必须独立做 domain/gate 审计
→ identity association
→ 相机彻底冻结
→ 每个人独立做 camera-local scene registration
→ 只输出 per-person root/depth/orientation residual，不再反向改相机
```

本轮同时得到一个重要负结果：在本地重合的 EgoHumans `001_legoassemble` 上，当前
frozen DA3 gate 6/6 全部接受，但总体让相机平移和人体误差变差。因此当前 DA3 gate
不能直接泛化到原始鱼眼 EgoHumans；后续人体精对齐也不能把“DA3 相机提案被接受”当成
“人体几何可信”的证据。

## 2. Multi-THuMBS 到底做了什么

### 2.1 输入与输出

输入是包含多个 shot 的多人单目视频。论文在主文中展开的是两个相邻 shot：

```text
S1 = frames 1 ... tb-1
S2 = frames tb ... T
```

它假设 cut 两侧时间间隔很小，属于同一物理场景，近似同步多视角。跨完全不同场景的
电影 cut 不属于其方法的有效范围。

输出包括：

- 一个共享世界坐标中的多人 SMPL mesh 和轨迹；
- 跨 shot 稳定的人物 identity；
- 对齐后的完整相机轨迹；
- 经过全序列平滑的多人运动。

### 2.2 完整 pipeline

```text
RGB video
→ PySceneDetect 检测 cut
→ 4DHumans 逐 shot 重建/跟踪 SMPL
→ ViTPose 预测 2D joints
→ Grounded SAM 预测逐人 mask
→ VGGT 只处理 cut 两侧两张图，建立 shared pointmap/camera
→ DROID-SLAM 处理每个完整 shot 的相机轨迹
→ 边界处逐人优化 root/global orientation，同时优化 camera
→ 将边界变换传播到完整 shot
→ geometry + UV appearance + pose 做 Hungarian Re-ID
→ 全序列 trajectory smoothing + cross-camera reprojection
→ global camera + identity-consistent human trajectories
```

这不是一次前向、因果流式模型。论文报告 150 帧、1920×1080 视频在单张 RTX 3090 上
约需 10 分钟。

### 2.3 边界处最关键的逐人精对齐

每个人的 SMPL 参数写作：

```text
global orientation Φ
root translation Γ
body pose θ
shape β
```

论文在边界两帧优化每个人的 `Γ/Φ` 和 camera pose，总目标为：

```text
L = λ2D L2D + λsil Lsil + λdepth Ldepth
```

三级过程为：

1. 用 2D root pixel 对应的 VGGT 3D point 初始化每个人的 root；
2. 只优化人体 root 和 global orientation，500 iterations，`λ2D=10`；
3. 联合优化人体 root/orientation 和 camera，1500 iterations，
   `λ2D=50, λsil=0.01, λdepth=1`。

三类约束分别是：

- `L2D`：预测 3D joints 投影后接近 ViTPose 2D joints；
- `Lsil`：mesh 投影不能跑出该人的 segmentation mask；
- `Ldepth`：投影到该人 mask 内的 mesh vertex 要接近对应 scene pointmap。

论文还先用人体/人体点云的最大质心半径比例来缩放 VGGT shared point cloud。这说明它也
不直接信任 scene prior 的 metric scale，而是显式处理人体和 scene gauge。

### 2.4 Re-ID

边界两侧人物 `i/j` 的代价为：

```text
Uij = 1.0 Dij + 0.2 Aij + 0.35 Pij
```

其中：

- `Dij`：shared world 中两个人体 root 的 3D 距离；
- `Aij`：4DHumans UV texture 的非遮挡颜色差；
- `Pij`：共同可见 SMPL joints 的 axis-angle pose 差。

之后使用 Hungarian，一旦 `Dij > 1 m` 就拒绝匹配，以支持进入/退出。

### 2.5 全序列后处理

论文最后还对完整视频做：

```text
10 Lsmooth + 1 Lcross
```

`Lsmooth` 平滑全局 joint trajectory，并正则 shape 和 VPoser latent；`Lcross` 把 cut
前人体放到 cut 后相机投影，并反向再做一次。这一步使用完整 shot 和未来帧，是其效果的
重要组成，不可以与我们的单 boundary causal 方法混为一谈。

## 3. 论文计算了哪些指标

### 3.1 证据层级：论文事实与本地推断必须分开

论文 PDF 第 11 页 `Metrics` 段明确写出的只有：

- 用 `MPJPE`、`MPVPE` 评价 pose；
- 用 `Accel` 评价 temporal smoothness；
- 用 `ATE` 评价 camera localization；
- 用 identity switches (`IDs`) 评价 identity consistency；
- `W-MPJPE` 是 initial-frame alignment，用于 trajectory consistency；
- `WA-MPJPE` 是 trajectory-level alignment，论文称其用于 shape accuracy；
- 无 GT 的真实编辑视频再报告 cross-shot `PCK*`、`Jitter`、`FS`。

主文没有给出任何一个评测指标的完整计算公式。下面必须区分：

| 内容 | 证据状态 |
|---|---|
| 指标名称、用途、升降方向、Table 1--4 数字 | 论文明确公开 |
| W/WA 的 Sim(3)、first two frames、100-frame chunk | WHAM/GVHMR/Human3R 公共惯例与本地 evaluator 实现，**不是论文确认协议** |
| MPJPE/MPVPE 的 pelvis alignment、24 joints、6890 vertices | 相关公共 evaluator 惯例，**论文主文未确认** |
| Accel 的二阶差分、是否乘 `fps^2` 及单位 | **论文主文未确认**；本地同时报告两种诊断量，不把任一种冒充官方口径 |
| ATE 的 SE(3)/Sim(3)、尺度、单位、逐 clip 聚合 | **未知** |
| IDs 的 matching、漏检/误检、进入退出与平均方式 | **未知** |
| PCK* 阈值/normalizer、Jitter/FS 公式 | **未知** |

论文表头也没有显式写单位。本文后续沿常见 convention 把人体距离值记为 mm、ATE 记为
m，但正式复现仍必须等待作者协议确认。

### 3.2 有 GT 数据上的本地 provisional 公式

以下是我们为了先跑诊断而采用的公开 GVHMR/Human3R-style 实现，不应写成
Multi-THuMBS 官方公式。

#### W-MPJPE

本地实现对每条 GT identity track 的前两帧全部 joints 拟合一个 Sim(3)，再把它应用到
完整轨迹：

```text
fit s,R,t on first 2 frames
W-MPJPE = mean || GT - (s R Pred + t) ||
```

`eval/global_human/utils.py` 对长序列还按 100 帧 chunk。论文第 11 页只说
`initial-frame alignment`，没有确认 two-frame、Sim(3) 或 chunk 规则。

#### WA-MPJPE

本地实现用完整人物轨迹拟合一个 Sim(3)：

```text
fit s,R,t on all trajectory frames
WA-MPJPE = mean || GT - (s R Pred + t) ||
```

它通常比 W-MPJPE 更宽容，因为完整 GT trajectory 都参与 gauge 拟合；但论文没有公开
其 transform 类型、track 切分或可见帧规则。

#### MPJPE / MPVPE

本地实现按公共 convention，在每帧 camera coordinate 中分别减去预测和 GT pelvis，再对
24 joints 或映射后的 6890 vertices 求平均误差。论文只命名指标，没有确认 pelvis index、
joint/vertex topology、是否 root-align、miss/FP 是否计罚。

按本地定义，共享 Boundary 世界平移会被 pelvis alignment 抵消，所以这两项更偏向局部
pose/shape，不能单独评价跨 shot 世界位置。

#### Accel

本地 provisional evaluator 同时保存 joint 的离散二阶差分和按采样率换算的物理量：

```text
delta2(t) = X(t-1) - 2 X(t) + X(t+1)
Accel-delta2   = mean || delta2_pred - delta2_gt || * 1000  [mm/frame^2]
Accel-physical = mean || delta2_pred - delta2_gt || * fps^2 [m/s^2]
```

论文未公开坐标系、pelvis centering、fps、单位、invalid/missing-frame 处理和 boundary
timestamp 处理。因此论文表中的 `27.3` 不能与其中任一列正式对表；两列都只是诊断。

#### ATE

本地 provisional evaluator 使用 Sim(3)-aligned camera-center translation RMSE。论文只说
`camera localization with ATE`，没有公开用 SE(3) 还是 Sim(3)、是否校正尺度、单位、每个
clip 的 alignment 范围和最终聚合方式。

#### IDs

本地 provisional evaluator 在 evaluator-side GT association 后，沿稳定 GT identity 统计
native predicted track ID 的变化。论文只说 identity switches；Table 2 有小数，但未说明
是每 clip 平均、每 boundary 平均还是全数据归一化，也未说明 miss/FP/重新进入如何处理。

### 3.3 无 3D GT 的 AVA、Friends、The Big Bang Theory

论文第 11 页仅把三者命名为 cross-shot motion-quality metrics，Table 4 caption 进一步说
`PCK*` 是引用 Multishot [26] 的 cross-shot PCK。主文没有给出：

- PCK 的投影方向、joint set、阈值、bbox/head/image normalizer 和 visibility；
- Jitter 的差分阶数、fps、单位、trajectory gauge 和聚合；
- FS 的 foot vertices、contact detector、阈值、单位和聚合。

因此 Table 4 三项目前都无法严格复现。把 `PCK*` 描述成“把一侧人体投影到另一侧”以及
把 Jitter 描述成三阶差分，只能是相关方法/代码推断，不能当成本论文公开公式。

## 4. Multi-THuMBS 的逐表公开结果和胜负线

实验相关页码索引：

| PDF 页 | 公开内容 |
|---:|---|
| 10 | propagation 声明、Re-ID Eq. 7--8、Hungarian 和 1 m rejection threshold |
| 11 | post-processing Eq. 9--11、运行时间、数据集构造概述、指标名称/用途 |
| 12 | Table 1 人体、Table 2 IDs/ATE、Table 3 消融、Table 4 编辑视频；baseline 列表 |
| 13 | camera/Re-ID/编辑视频定性与定量结论描述 |
| 14 | Table 3 消融解释、方法 limitation |

PDF 第 1--17 页与论文印刷页码一致；实验表均集中在第 12 页。

### 4.1 Table 1（PDF 第 12 页）：人体指标完整对比

Table 1 比较 `Multishot`、`GVHMR`、`PromptHMR`、为 multi-shot/multi-person 场景修改的
`HSfM†`，以及 Multi-THuMBS (`Ours`)。

#### EgoHumans

| Method | W-MPJPE ↓ | WA-MPJPE ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ |
|---|---:|---:|---:|---:|---:|
| Multishot | 474.1 | 287.4 | 347.1 | 408.2 | 63.15 |
| GVHMR | 404.8 | 204.7 | 287.4 | 371.4 | 59.0 |
| PromptHMR | 1778.2 | 440.9 | 285.3 | 364.1 | 74.3 |
| HSfM† | 544.2 | 187.7 | 263.1 | 294.3 | 48.7 |
| Multi-THuMBS | **279.0** | **166.0** | **228.3** | **262.2** | **27.3** |

五项最佳均是 Multi-THuMBS。因此同协议下要打过它，五项都需严格低于最后一行。

#### EgoBody

| Method | W-MPJPE ↓ | WA-MPJPE ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ |
|---|---:|---:|---:|---:|---:|
| Multishot | 185.1 | 144.1 | 147.1 | 166.5 | 17.9 |
| GVHMR | 174.0 | 133.3 | 108.0 | 147.1 | 14.7 |
| PromptHMR | 1228.1 | 395.3 | 99.0 | 133.9 | 17.4 |
| HSfM† | 113.1 | 96.3 | 113.3 | 123.2 | 10.9 |
| Multi-THuMBS | **99.2** | **72.8** | **72.0** | **94.9** | **6.0** |

五项最佳也均是 Multi-THuMBS。

#### Harmony4D

| Method | W-MPJPE ↓ | WA-MPJPE ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ |
|---|---:|---:|---:|---:|---:|
| Multishot | 248.0 | 231.6 | 511.2 | 609.5 | 37.1 |
| GVHMR | 244.9 | 166.7 | 244.7 | 334.1 | 29.6 |
| PromptHMR | 1746.3 | 399.8 | 675.7 | 746.0 | 66.8 |
| HSfM† | 372.0 | 178.4 | 225.6 | **257.6** | 28.3 |
| Multi-THuMBS | **221.0** | **116.9** | **215.9** | 278.3 | **17.4** |

Harmony4D 的 MPVPE 最佳是 HSfM† `257.6`，不是 Multi-THuMBS `278.3`。如果目标是既打过
Multi-THuMBS 又成为该表 SOTA，Harmony4D MPVPE 的真正胜负线是 `<257.6`。

### 4.2 Table 2（PDF 第 12 页）：Re-ID 和 camera

#### Identity switches

| Method | EgoHumans IDs ↓ | EgoBody IDs ↓ | Harmony4D IDs ↓ |
|---|---:|---:|---:|
| PromptHMR | 10.40 | 1.29 | 8.00 |
| HSfM† | 3.87 | 0.20 | 1.58 |
| KPR | 2.54 | 0.05 | 1.19 |
| Pose2ID | 4.62 | 0.72 | 1.32 |
| Multi-THuMBS distance-only | 1.66 | **0.00** | 0.54 |
| Multi-THuMBS full | **0.97** | **0.00** | **0.46** |

distance-only 是只用 Eq. 7 shared-world root distance 的 Re-ID；full 再加入 UV appearance
和 pose。EgoBody 上两者同为 0。

#### Camera ATE

| Method | EgoHumans ATE ↓ | EgoBody ATE ↓ | Harmony4D ATE ↓ |
|---|---:|---:|---:|
| VGGT | 1.4 | 1.3 | 1.4 |
| PromptHMR | 2.3 | 1.6 | 2.3 |
| HSfM† | 2.8 | 2.9 | 3.2 |
| Multi-THuMBS | **0.7** | **0.1** | **0.7** |

三项最佳均是 Multi-THuMBS，但 ATE 的 alignment/scale/unit 协议没有公开。

### 4.3 Table 3（PDF 第 12 页，讨论在第 14 页）：EgoHumans 消融

| Variant | W-MPJPE ↓ | MPJPE ↓ | MPVPE ↓ | Accel ↓ | ATE ↓ |
|---|---:|---:|---:|---:|---:|
| w/o camera optimization | 389.7 | 230.1 | 264.7 | 33.7 | 1.40 |
| w/o post-processing | 311.2 | 241.6 | 298.3 | 47.9 | 0.77 |
| w/o hierarchical stage | 491.9 | 368.1 | 359.4 | 33.9 | 2.75 |
| w/o boundary alignment | 882.7 | 422.4 | 392.1 | 34.9 | 1.40 |
| Full | **278.8** | **228.3** | **262.1** | **27.3** | **0.77** |

这张表直接说明 boundary human-scene alignment 是最大贡献项；移除它后 W-MPJPE 从
`278.8` 变为 `882.7`。post-processing 对 Accel 很重要（`27.3 -> 47.9`）。camera
optimization 同时影响 ATE 和人体世界轨迹。Table 3 full 与 Table 1/2 的 W、MPVPE、ATE
有小幅数字差异（`278.8 vs 279.0`、`262.1 vs 262.2`、`0.77 vs 0.7`），主文未解释是
rounding、run 还是聚合差异，不能擅自合并。

### 4.4 Table 4（PDF 第 12 页）：编辑视频

| Method | PCK* ↑ | Jitter ↓ | FS ↓ |
|---|---:|---:|---:|
| PromptHMR | 62.7 | 162.44 | 23.11 |
| Multi-THuMBS | **90.7** | **31.5** | **10.7** |

Table 4 只对比 PromptHMR，数据是 AVA、Friends、The Big Bang Theory 的汇总；论文没有
逐数据集结果、clip 数、每项样本权重或公式，故这些数字只能作为公开参考线。

## 5. 官方协议为什么暂时不能严格复现

论文第 10 页说 propagation algorithm 和 Re-ID hyperparameter ablation 在 supplementary，
第 11 页又明确说 dataset construction、implementation settings、evaluation protocols
都在 supplementary。但截至 2026-07-31：

- 17 页 PDF 没有 supplementary；
- PDF 没有 embedded attachment；
- arXiv source 的 26 个文件也没有 supplementary；
- 项目页没有 Code 或 Supplementary 下载；
- 没有公开 sequence、camera pair、cut frame、clip 数和 manifest；
- 没有公开 person matching、visibility、miss/FP 和指标聚合细节；
- 没有公开 PCK*/Jitter/FS 的关键参数。

论文第 11 页唯一公开的运行代价是：150 帧、1920×1080 视频，在单张 RTX 3090 上完整
optimization 约 10 分钟；没有给各 stage wall time、峰值显存、batch 或评测预处理耗时。

所以当前只能做：

```text
protocol-matched-as-far-as-public-information
```

不能声称复现了 Multi-THuMBS 官方表格，也不能因为某个自建 split 的数字更小就宣称打败。

## 6. 本地数据重合审计

本地唯一可立即运行的重合数据是：

```text
/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble
```

目录名是历史错误，它正式属于 **EgoHumans**，包含：

- 601 个同步时间戳；
- 8 路 exo fisheye RGB；
- 3 个稳定 identity：`aria01/aria02/aria03`；
- SMPL 6890 vertices、45 joints、pose、shape、translation；
- 2D joints、bbox 和相机标定；
- Aria world 到 COLMAP world 的 Sim(3)。

归属证据不是只看目录名：EgoHumans 官方配置中存在
`egohumans/configs/legoassemble/001_legoassemble.yaml`，其 sequence 名和
`aria01/aria02/aria03` 与本地一致；官方 EgoHumans 数据结构也正是
`colmap + ego/aria* + exo/cam* + processed_data`，与本地同构。当前 `/data` 下没有更多
可运行的 EgoBody 或 Harmony4D RGB+GT overlap。

本地已有三条 15-frame chain、共 6 cuts：

```text
cam01 296-300 → cam06 300-304 → cam07 304-308
cam02 176-180 → cam05 180-184 → cam08 184-188
cam03 416-420 → cam04 420-424 → cam01 424-428
```

每个 cut 两侧重复同一 timestamp，适合纯跨相机 boundary 诊断，但不是论文官方 split，
也不适合把 Accel 当成正式自然视频指标。

## 7. 实验 A：旧 Human3R cache 的 provisional 论文式指标

入口：

```text
versions/v14/eval_multithumbs_protocol.py
```

它修正了现有通用 evaluator 在多人情况下把不同人物按帧拼成一条 trajectory 的问题：
现在先按稳定 GT identity 分轨，再分别计算 W/WA/Accel。GT identity 只用于 evaluator
association，不进入预测。

输入是三条旧 V13 Human3R raw cache，不含 B0、DA3 或当前精修结果。

| Scope | W-MPJPE | WA-MPJPE | MPJPE | MPVPE | Accel delta2 | Accel physical | ATE | IDs/stream |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Human3R raw, 3×15 frames | 1088.3 mm | 405.1 mm | 109.3 mm | 130.0 mm | 58.33 mm/frame² | 52.49 m/s² | 1.848 | 4.00 |
| Multi-THuMBS EgoHumans reference | 279.0 | 166.0 | 228.3 | 262.2 | 27.3（单位未知） | 不可得 | 0.7 | 0.97 |

这些数字不能按列直接宣布胜负：

- 本地只有一个 capture 的三个手工短链，不是论文 split；
- W/WA/ATE/IDs 显著落后论文参考线；
- 本地 pelvis-aligned MPJPE/MPVPE 虽然数值更低，但只对成功匹配的人求平均，且短链、
  topology 和可见性规则均不同；
- 其中两条链存在漏检，pose metric 不惩罚 miss，会让 MPJPE/MPVPE 偏乐观；
- 两种 Accel 诊断都把重复 timestamp 的 cut 两侧当作普通相邻帧，只能衡量 cut jump，且
  不能与论文 `27.3` 正式对表；
- ATE 对几乎停在局部原点的 reset camera 做 Sim(3) 时 scale 达数百，只应作为失败诊断。

因此当前最准确结论是：Human3R 的单帧局部 pose/shape 不差，但 raw camera/trajectory 和
native identity 无法处理 cut；这正是论文 W/WA/ATE/IDs 所测的部分。

## 8. 实验 B：当前 V14 B0 和 B0+DA3 的 EgoHumans 实测

入口：

```text
versions/v14/probe_b0_da3_egohumans.py
```

实际执行了正确的当前路径：

```text
V14 shadow/raw
→ frozen B0
→ DA3 forward + reverse
→ frozen DA3 gate
→ Boundary 冻结
→ 再加载 GT 做 evaluation
```

6 个 cuts 全部成功，0 failures。DA3 gate `6/6` 全接受，0 fallback。

| Method | Camera T | Camera R | Root | World joint-24 | World vertex-6890 | Pair root vector |
|---|---:|---:|---:|---:|---:|---:|
| B0 | **0.3968 m** | 4.143° | **0.3462 m** | **0.3478 m** | **0.3500 m** | **0.3358 m** |
| B0+DA3 | 0.4046 m | **4.111°** | 0.3721 m | 0.3602 m | 0.3608 m | 0.3379 m |

cut 级改善计数：

```text
camera translation: 4 / 6
camera rotation:    3 / 6
human joint:        2 / 6
```

人体覆盖是 `80/90 person-frame`；`cam03→cam04` 中 Human3R 每帧只检测到一人。

这里的 `347.8 mm` 是固定 pre-shot gauge 下、未做 trajectory Sim(3) 和 pelvis alignment 的
world joint error，不是论文 MPJPE。因此本表是当前 Boundary 的绝对空间诊断，不能直接与
Table 1 的 MPJPE 对表。

### 8.1 实验结论

1. 原始鱼眼 EgoHumans 是 DA3 的明显 domain shift。
2. 当前 forward/reverse agreement gate 不足以判断 DA3 是否真的更接近 GT。
3. gate 6/6 接受却使总体 camera T 和 human error 变差，必须增加 domain/geometry gate。
4. 相机 rotation 略改善并未带来人体改善。
5. 即使相机完全正确，共享 Boundary 仍不能消除 camera-local 人体结构误差。

## 9. 下一步主线：Person-Conditioned Boundary Scene Registration

### 9.1 设计原则

下一步不再让人体修改 shared camera Boundary。相机和人体必须分成两个 gauge：

```text
shared camera gauge: 一个 shot 一个 Boundary
person-local gauge:  每个人独立的 root/depth/orientation residual
```

整个过程仍发生在同一个 boundary event 内，因此对外仍是一次流式 transaction，而不是
依赖完整未来 shot 的离线 pipeline。

### 9.2 输入

只使用 cut 时已经存在的因果信息：

- cut 前人体 identity memory、root trajectory、shape/height prior；
- last-pre RGB 和 first-post RGB；
- B0 camera，或通过严格 gate 的 DA3 camera；
- first-post Human3R 的 SMPL-X、2D projection、bbox/mesh mask；
- DA3 boundary pointmap、confidence 和 forward/reverse consistency；
- 多人 pairwise layout。

不使用：

- GT camera/GT body/GT identity 生成 candidate；
- 第二张及之后的 post frame；
- DROID-SLAM 完整 shot；
- 全序列 smoothing。

### 9.3 求解变量

第一阶段只允许每个人：

```text
ray depth Δz_i
small transverse residual Δx_i, Δy_i
optional global yaw/orientation residual ΔR_i
```

camera Boundary 保持 bit-exact 不变。body pose 和 shape 先固定，只在证据充分时加入很小的
orientation residual。

### 9.4 约束

建议的目标为：

```text
L_person(i) =
    λdepth  robust_pointmap_depth
  + λ2D     joint_reprojection
  + λsil    silhouette_distance
  + λsize   persistent_body_size
  + λtrack  pre_track_soft_prior
  + λreg    residual_regularization

L_multi = λpair pairwise_layout
```

关键点：

- `pointmap depth` 必须只使用该人的 mask/mesh 可见区域和高置信点；
- `2D/silhouette` 防止 3D correction 破坏当前图像对齐；
- 旧 shot root 只能是 soft prior，180-cut 实验已证明不能直接作为硬 anchor；
- `pairwise layout` 约束多人相对结构，但不能强迫所有人共享一个 correction；
- correction 需要独立 cap，禁止覆盖 camera。

### 9.5 接受 gate

人体精修必须拥有与 DA3 camera gate 分离的新 gate：

- person mask/mesh coverage；
- pointmap confidence；
- forward/reverse person depth consistency；
- optimize 前后 pointmap、2D、silhouette 三类 loss 是否同时下降；
- correction magnitude/cap；
- 多人 pairwise layout 是否恶化；
- 与 persistent body size 是否冲突。

无法通过时回退到 B0 人体，不影响 camera。

### 9.6 实验顺序

1. 在 `three` 上做 GT-only evaluation 的 observability probe，检查 DA3 person pointmap
   是否确实包含正确 depth signal；GT 不进入 candidate。
2. 固定 B0 camera，比较仅 `Δz`、`Δz+Δxy`、`Δz+ΔR`。
3. 加入 2D/silhouette，验证 image projection 不恶化。
4. 加入 pairwise layout 和独立 gate。
5. `three` 只用于开发；`dance/box` frozen，EgoHumans 作为 fisheye cross-domain。
6. 如果确定性优化有信号但 gate 不够稳，再训练 camera-conditioned human residual head；
   监督只作用于 person-local residual，禁止梯度或求解结果覆盖 camera。

## 10. 必须达到的准入标准

### 10.1 当前内部 benchmark

- B0/DA3 camera 输出 bit-exact 不变；
- Root、world joint、world vertex 均显著下降；
- `three/dance/box` 都改善，不能只在 `three` 调参后让 `box` 退化；
- >5 cm harm rate 有明确下降并受 gate 控制；
- 单帧 post-cut、无未来帧；
- 自动 identity 协议下仍成立。

### 10.2 Multi-THuMBS 正式胜负线

拿到或自行冻结完整 benchmark 后，至少需要：

```text
EgoHumans:
W-MPJPE < 279.0 mm
WA-MPJPE < 166.0 mm
MPJPE    < 228.3 mm
MPVPE    < 262.2 mm
Accel    < 27.3  # 仅在作者口径确认后
ATE      < 0.7
IDs      < 0.97
```

同时必须报告我们的因果优势和运行时间，不能通过使用完整未来 shot/全序列优化换取不公平
提升。若作者协议继续不公开，应发布 Movie3R 自己的 sequence/camera/cut manifest、公式和
evaluator，并让 Multi-THuMBS 官方代码发布后直接接入同一 harness。

## 11. 可复现入口与产物

论文式 provisional evaluator：

```bash
.venv/bin/python versions/v14/eval_multithumbs_protocol.py --self_test
.venv/bin/python versions/v14/eval_multithumbs_protocol.py --device cpu
```

当前 B0+DA3 EgoHumans：

```bash
.venv/bin/python versions/v14/probe_b0_da3_egohumans.py --device cuda:0
```

文件：

```text
versions/v14/eval_multithumbs_protocol.py
versions/v14/probe_b0_da3_egohumans.py
output/v14/fine_alignment_research/multithumbs_protocol/README.md
output/v14/fine_alignment_research/multithumbs_protocol/human3r_raw_egohumans_provisional.json
output/v14/fine_alignment_research/b0_da3_egohumans/v14_b0_da3_egohumans.md
output/v14/fine_alignment_research/b0_da3_egohumans/v14_b0_da3_egohumans.json
output/v14/fine_alignment_research/b0_da3_egohumans/run_all.log
```

所有本轮新输出均位于 `/data` 下。没有向系统根目录写入实验 cache、日志或临时模型。

## 12. 2026-07-31 更新：BRTC-LC 已通过内部 root/layout 门槛

第 9 节的逐人优化设计仍适合后续 global orientation 和内部 pose，但 root/depth 的第一条
可落地主线已经由更简单的显式几何方法给出：

```text
frozen B0
-> automatic association
-> last-pre / first-post five-joint ray triangulation
-> ray-gap/parallax/MAD gate
-> group shift + pre-layout-selected individual residual
-> rigid post-person translation; camera unchanged
```

该方法称为 BRTC-LC。新的 `three offset1` 自动-ID 确认结果为：

| 指标 | B0 | BRTC-LC | Gain |
|---|---:|---:|---:|
| Root | 0.3779 m | 0.2314 m | 38.8% |
| Fixed-world joint | 0.4117 m | 0.2745 m | 33.3% |
| Fixed-world vertex | 0.3891 m | 0.2525 m | 35.1% |
| Pairwise root vector | 0.3297 m | 0.2588 m | 21.5% |

覆盖 `88.0%`，root harm >5 cm 为 `7.2%`，自动匹配 `100%`，camera 数值改动为 `0`。
部署 callable 与 probe 的 shift parity 为 `1.10e-15 m`，4 个 runtime 单元测试通过。

这与 Multi-THuMBS Table 3 的结论一致：boundary 处必须显式做 human alignment。不过这些
不是论文的 W-MPJPE/WA-MPJPE/MPJPE/MPVPE，也没有使用官方 split，因此只证明 Movie3R
内部 root/layout 主线成立，**仍不能宣称正式击败 Multi-THuMBS**。

详细方法和完整失败记录：

```text
versions/v14/docs/V14_B0_TWO_VIEW_TRIANGULATION_FINAL_20260731.md
```
