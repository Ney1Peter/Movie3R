# Movie3R Camera-Cut Alignment Method Comparison

## 文档目的

本文档面向不了解实验历史的研究人员或 AI，详细比较以下方法：

1. 昨天的旧 **V47**：现在重命名为 **V11.1 Conditional Wide Rotation**；
2. 昨天的旧 **V46**：现在重命名为 **V11.2 Contact-Preserving Alignment**；
3. 昨天在 V46/V47 之后完成的人体大小统一修正版：
   **V11.4 Uniform Similarity - Conditional Wide**；
4. 今天的 **V14.3 Projection-Consistent Human-Camera Re-anchoring**。

目标不是只比较某一个 camera translation 数字，而是判断：

- 哪一种方法的几何关系最自洽；
- 哪一种方法的实际三维结果最可靠；
- 哪一种方法最有研究创新性；
- 哪一种方法最适合作为后续统一方案的基础；
- 哪些模块可以组合，哪些模块不能直接叠加。

本文档中的“昨天”和“今天”分别指 2026-07-21 和 2026-07-22 完成的实验。

---

## 1. 版本编号映射

工作区已经将早期不断增长的 V30、V40、V50 编号重新整理为紧凑版本。缓存目录仍然
保留旧编号，以免复制或破坏已有实验结果，但活跃代码使用新编号。

| 旧编号/旧称呼 | 当前编号/当前名称 | 当前定位 |
|---|---|---|
| Fixed Explicit | V10.1 Fixed Explicit | 所有显式方法的基础基线 |
| 旧 V47 | V11.1 Conditional Wide Rotation | 保留的 raw-scale 刚体方法 |
| 旧 V46 | V11.2 Contact-Preserving Alignment | 接触修正诊断，不是最终部署方法 |
| V48 | V11.3 Component Ablation | 组件必要性消融 |
| V53 | V11.4 Uniform Similarity | 昨天最终的人体/背景统一尺度修正版 |
| V52 viewer | V12.1/V12.2 | 10+10 帧缓存和交互查看器 |
| 当前 coupled 方法 | V14.3 Projection-Consistent Re-anchoring | 今天的人-相机联合 root 修正 |

最容易混淆的一点是：

> 今天的 V14.3 不是昨天 V47/V11.4 的改名版本，也没有自动包含 V11.4。
> 它是一条新的、解决“camera 使用校准 root，但 human 仍使用 raw root”问题的分支。

版本规则见：

- `docs/movie3r/BOUNDARY_VERSIONING.md`
- `versions/v12/docs/ACTIVE_BOUNDARY_ALIGNMENT.md`

---

## 2. 所有方法共享的问题设定

### 2.1 Human3R 在 camera cut 上的问题

Human3R 是流式人体-场景重建模型，每帧输出：

- camera pose；
- scene pointmap/depth；
- SMPL-X pose、shape、root translation；
- 人体 joints 和 vertices；
- recurrent state。

在普通连续视频中，这些输出共享同一条历史状态和局部坐标系。但 camera cut 发生后，
旧镜头 state 会污染新镜头，因此已经确认必须：

```text
camera cut
-> 在第一张 post-cut 图像解码前 hard reset Human3R
-> 新 shot 在 fresh local state 中重新开始
-> 再计算一个固定的 shot-level Boundary transform
-> 将新 shot 放回旧 world gauge
```

所有保留方法都不允许逐帧重算 Boundary。一次 cut 只求一次变换，后续整个 post-cut
shot 固定复用。

### 2.2 三类容易混淆的“对齐”

必须区分三件事：

1. **朝向对齐**：新旧相机坐标轴的旋转是否正确；
2. **整体尺度/场景对齐**：camera、pointmap、human 是否处于同一个 shot scale；
3. **人-相机 root 一致性**：计算 camera translation 时使用的人体 root，是否也是最终
   SMPL-X 真正使用的 root。

旧 V47 主要解决第 1 项；V11.4 主要解决第 2 项；V14.3 主要解决第 3 项。

### 2.3 流式约束

保留方案共同满足：

- 暂时使用 GT cut index，只把它当作触发信号；
- Human3R 主体冻结；
- post-cut 使用 fresh state；
- 只读 cut 前已经到达的历史和第一张 post-cut 图像；
- 不访问完整未来 shot；
- 不做 BA 或全局轨迹优化；
- 不训练完整 SE(3) 回归器；
- 不使用 raw token 回归 Boundary；
- 每个 post-cut shot 只有一个固定 Boundary；
- GT camera、GT depth、GT scene mesh 只用于评测，不进入可部署推理。

---

## 3. 共同组件和预训练模型

## 3.1 Human3R

**使用时间**：正常视频每帧都运行；camera cut 后先 reset，再对新 shot 正常前馈。

**提供内容**：

- raw camera pose；
- raw pointmap/depth；
- SMPL-X body；
- raw camera-frame human root；
- torso pose/motion history。

**作用边界**：Human3R 提供局部重建，但它的 shot-local gauge 不保证跨数据源都是正确
米制尺度。尤其 MVHuman 上，raw root depth 和 shot scale bias 较大。

## 3.2 V10.1 Fixed Explicit

**使用时间**：每次 cut 发生时计算一次。

**作用**：以人体姿态/root 和非人体区域 pointmap 为几何依据，构造一个确定性的粗
Boundary SE(3)。它不是神经网络回归器，而是显式几何基线。

**特点**：

- 稳定、可解释；
- AvatarReX 和 THuman 的原始结果经常已经不错；
- 大视角困难样本的 rotation tail 较大；
- 180 cuts 上 camera translation `1.715 m`，rotation `24.20 deg`。

## 3.3 V16 torso-motion rotation

**使用时间**：cut 时读取 pre-cut torso motion 和第一张 post-cut 人体姿态，只运行一次。

**作用**：利用人体躯干的相对方向和运动，修正 Fixed Explicit 的 rotation。

**特点**：

- 不直接回归 translation 或 scale；
- training-free；
- 当前 V14.3 使用统一 `20 deg` residual bound；
- 在四个数据源上总体保持同方向 rotation 改善；
- 单独将 rotation mean 从 `24.20 deg` 降至约 `16 deg`。

## 3.4 VGGT

**模型性质**：冻结的外部预训练多视图几何模型。

**使用时间**：只在 camera cut 时，对最后一张 pre-cut RGB 和第一张 post-cut RGB
执行 `1+1` 推理；普通帧不运行。

**作用**：只作为大视角 rotation tail rescue，不作为最终 translation/scale 回归器。

**重要限制**：纯 VGGT 不能无条件替代 Fixed/torso。无条件使用时 rotation mean
`37.13 deg`、P95 `166.91 deg`，会严重破坏 AvatarReX 等本来简单的样本。

因此旧 V47 使用固定、source-independent 的困难触发规则，只有诊断表明 torso 解不可靠
时才采用 VGGT rotation。触发规则使用：

- Fixed 后剩余 torso correction；
- VGGT correction 相对 torso 的大小；
- torso/VGGT 方向是否一致；
- VGGT forward/reverse rotation spread；
- 当前图像纹理。

它不读取 source ID，也不直接使用 GT view angle。

## 3.5 Depth Anything 3 / DA3Metric-Large

**模型性质**：冻结的外部预训练 metric-depth 模型。

**本地权重**：

```text
/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large
```

**使用时间**：只在 cut 时运行一次。当前缓存协议输入 cut 前 5 帧和 cut 后第一帧；后续
shot 不再运行 DA3。

**作用**：提供 Human3R 本身缺失的 metric depth/shot-scale cue，尤其改善 MVHuman。

**坐标换算**：Human3R crop 可直接作为 RGB 输入，但 DA3 canonical depth 需要结合处理后
intrinsics 换算为米：

```text
depth_meter = raw_depth * mean(fx_processed, fy_processed) / 300
```

随后按相同 OpenCV camera convention 反投影。

**限制**：DA3 depth 可以让 camera/human 更接近 GT metric gauge，但如果只改 camera 或
root，不同步修改完整 pointmap 和人体尺寸，就会破坏人体投影或脚地接触。

## 3.6 Frozen 2D keypoint detector

**模型**：Torchvision Keypoint R-CNN ResNet50-FPN default pretrained weights。

**使用时间**：V18/V14.3 在 cut 处读取 pre-cut 历史和第一张 post-cut RGB；正式求解主要
使用当前 post-cut torso joints。

**作用**：把 SMPL-X 米制身体投影与图像中的 torso joints 对应起来，用于估计人体
camera-frame root。

实验已确认 detector 不是主要瓶颈：predicted pose + GT shape + detector 2D 的 depth
error 约 `0.063 m`；真正瓶颈是 predicted shape/world body scale。

---

## 4. 昨天版本一：旧 V47 / 当前 V11.1

## 4.1 当前名称

```text
V11.1 Conditional Wide Rotation
```

在缓存和内部字段中也可能看到：

- `V47`；
- `V32 raw-scale`；
- `raw_scale_v32`；
- `Conditional Wide Rotation`。

这些名称指向同一条保留主线。

## 4.2 完整流程

```text
GT cut trigger
-> pre-decode Hard Reset Human3R
-> V10.1 Fixed Explicit coarse transform
-> V16 torso-motion rotation
-> 固定困难规则判断 torso 是否仍然不可靠
-> 只有困难样本运行 frozen VGGT 1+1 rotation
-> 在 Human3R 原始 shot gauge 中用 human root 显式重算 translation
-> 得到一个固定 shot-level SE(3)
-> 同一个 SE(3) 作用于 camera、pointmap 和完整 SMPL-X
```

## 4.3 它如何处理尺度

旧 V47 **不做 metric shot-scale correction**。它保留 Human3R 原始 shot gauge：

- camera translation 不单独缩放；
- pointmap 不单独缩放；
- human root 不单独缩放；
- SMPL-X body offsets/人体尺寸不缩放。

这意味着它不能解决 Human3R 原始尺度本身不准的问题，但有一个重要优点：Human3R
已经重建出的“人站在地面上”“人与背景相对位置合理”等关系不会被破坏。

## 4.4 VGGT 是否每次都运行

不是。180-cut 消融中：

- Conditional VGGT 只修改 `34/180` 个样本；
- 被修改样本 rotation 从 `36.04 deg` 降到 `17.09 deg`；
- `30` 个样本改善超过 `5 deg`；
- 只有 `1` 个样本恶化超过 `5 deg`；
- AvatarReX 和 THuman 基本不触发，收益主要来自 MVHuman tail。

在 1079-cut 扩展审计中，触发并不由相机跨度本身决定。AvatarReX/THuman 同样包含
60-180 度 view changes，但它们的 Fixed+torso 已经准确，因此不触发 VGGT。真正的困难
定义是“显式人体/torso 解仍然存在冲突”，不是“属于 MVHuman”或“角度大”。

## 4.5 结果

| Method | Camera T mean/P95 | Rotation mean/P95 | Scene | Foot/ground distortion | Reprojection shift |
|---|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715/4.123 m | 24.20/73.61 deg | 0.347 m | 0 | 0 px |
| Torso raw scale | 1.606/3.960 m | 16.04/53.56 deg | 0.409 m | 0 | 0 px |
| **Conditional Wide raw scale** | **1.568/3.798 m** | **12.09/37.75 deg** | 0.413 m | **0** | **0 px** |

## 4.6 优点

- 几何关系最保守；
- 是真正统一的 shot-level rigid SE(3)；
- 不会改变人体在 RGB 中的投影；
- 不会改变人体和地面的接触；
- 对大 rotation tail 有经过多个 holdout 的稳定改善；
- source-blind，未使用 source ID；
- cut-only、流式、普通帧无额外成本。

## 4.7 缺点

- 不修正 Human3R shot scale；
- camera translation mean 仍有 `1.568 m`；
- MVHuman 的 metric depth bias 仍然存在；
- 使用外部 VGGT，尽管只在少量困难 cut 上运行；
- 创新主要在安全触发和模块组合，而不是新的基础模型。

## 4.8 当前定位

它是目前最安全的 **raw-gauge rotation/rigid alignment backbone**，但不是完整 metric
translation solution。

---

## 5. 昨天版本二：旧 V46 / 当前 V11.2

## 5.1 当前名称

```text
V11.2 Contact-Preserving Alignment
```

旧 V46 报告内部包含三个不同概念，不能混为一个结果：

1. `raw_scale_v32`：实际上就是旧 V47/V11.1 raw-scale 方法；
2. `current_v45`：DA3 独立 metric scaling 的 camera-metric 版本；
3. `contact_v32`：在 metric scaling 后增加脚地接触修正的旧 V46 主实验。

## 5.2 为什么设计 V46

旧 V45 使用 DA3 得到独立的 root scale 和 scene scale，camera translation 从
`1.568 m` 大幅降到约 `0.434 m`。但它出现明显视觉问题：

- 人、背景和相机的整体尺度不一致；
- SMPL-X body dimensions 没有同步缩放；
- 人脚可能陷入地面；
- RGB 中人体投影发生变化。

V46 尝试在保留 camera metric gain 的同时恢复脚地接触。

## 5.3 完整流程

```text
Hard Reset Human3R
-> 旧 V47/V32 conditional rotation
-> DA3-derived root scale 和 scene scale
-> 分别缩放 camera/root 与 pointmap
-> 计算 pelvis-to-feet 方向
-> 测量缩放后 human foot 与 scene contact 的差
-> 沿 pelvis-to-feet 方向平移 human root
-> 重新显式求 camera translation
```

接触修正公式只调整 root translation，不改变局部关节角度。它能让三维脚地距离 proxy
重新变为 0，但没有保持原始图像投影。

## 5.4 结果

| Variant | Camera T mean/P95 | Rotation mean/P95 | Scene | Foot distortion | Human reprojection shift |
|---|---:|---:|---:|---:|---:|
| raw-scale V47 | 1.568/3.798 m | 12.09/37.75 deg | 0.413 m | 0 | 0 px |
| V45 independent metric scale | **0.434/1.040 m** | 12.09/37.75 deg | **0.288 m** | **0.515 m** | 29.9 px |
| V46 contact correction | 0.465/1.003 m | 12.09/37.75 deg | 0.305 m | **0** | **112.1 px** |

V46 平均需要移动人体 root `0.515 m`。某些 MVHuman viewer 案例超过 `0.8 m`，最终造成
上百像素的人体投影移动。

## 5.5 为什么三维 viewer 有时看起来不错

固定第三视角中，脚与地面可能重新接触，相机和 pointcloud 也可能看起来更靠近，因此
V46 在部分案例中主观效果不错。但它没有同时满足：

- RGB 中人体仍贴合真实人；
- camera、pointmap、root 和完整人体尺寸来自同一个 similarity；
- 局部人体几何关系不被额外 root correction 改写。

因此第三视角“脚落地”不能代替 image-space reprojection 和完整几何审计。

## 5.6 当前定位

V11.2/旧 V46 是一个重要的 **失败诊断**：它证明“只在最后补一个脚地平移”不能修复
前面已经不一致的尺度。它不应作为最终部署版本。

---

## 6. 昨天最终大小修正版：V11.4 Uniform Similarity

## 6.1 为什么必须单独列出

用户在观察旧 V45/V46/V47 后指出：某些 metric 版本虽然对齐了，但整个人、背景和相机
一起变小，人体高度或脚地关系也不合理。为此新增 V11.4，它才是昨天在“整体合理性和
局部合理性都要正确”之后保留的最终尺度版本。

当前 viewer 名称：

```text
Uniform Similarity - Conditional Wide
```

长序列入口：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/v12_2_long_sequence_viewer.py \
  --device cuda:0 --port 8096
```

## 6.2 核心原则

如果一个 shot 需要尺度 `s`，就必须对所有三维量统一使用 `s`：

```text
camera translation      *= s
pointmap camera points  *= s
SMPL-X root             *= s
SMPL-X body offsets     *= s
complete joints/mesh    *= s
```

然后再使用旧 V47 的 conditional rotation 和显式 translation 求一个固定 Boundary。

这相当于在 Boundary 前加入一个完整的 shot-level similarity，而不是分别给 camera、root、
scene 设置不一致的尺度。

## 6.3 使用的模型和时机

- Human3R：正常流式重建，cut 后 hard reset；
- V16 torso：每次 cut 的默认 rotation；
- frozen VGGT：只在困难 trigger 上对 1 pre + 1 post RGB 运行；
- DA3-derived scene/root scale：只在 cut 时计算一次；
- 最终 scale 和 Boundary：整个 post-cut shot 固定复用。

没有 learned gate、learned SE(3)、token adapter、BA 或完整未来 shot。

## 6.4 为什么它保持投影和脚地关系

透视投影中，如果同一 camera frame 内 root、body offsets 和 scene points 都乘以同一个
尺度，`x/z`、`y/z` 不变，因此 2D reprojection 不变。人与地面的三维距离也按同一比例
变化，不会产生“人体缩了但地面没缩”或“root 缩了但腿长没缩”的结构冲突。

## 6.5 结果

保留的 `v47_uniform_scene` 结果：

| Metric | Result |
|---|---:|
| Camera translation mean/median | **0.397/0.323 m** |
| Camera translation P90/P95 | **0.750/0.928 m** |
| Rotation mean/P95 | **12.09/37.75 deg** |
| Human relative-motion error | 0.012 m |
| Scene trimmed mean | 0.302 m |
| Human reprojection shift | **0 px** |
| Foot/ground distortion | **0 m** |
| Scale absolute error vs GT diagnostic | 0.164 |

## 6.6 优点

- 在昨天的方法中，camera、human 和 scene 的内部几何最完整；
- camera translation 指标明显优于 raw-scale V47；
- 保留 V47 的 rotation tail rescue；
- 不产生 V46 的脚地和投影冲突；
- 完整人体尺寸被统一处理；
- 流式、cut-only、固定一次 scale 和 Boundary。

## 6.7 缺点和风险

- 核心 metric scale cue 仍来自冻结 DA3，而不是 Movie3R 自己学习的尺度；
- 是 Sim(3)+SE(3) 组合，不是纯 SE(3)；
- scale accuracy 仍有误差，不能因为投影不变就认为绝对尺度一定正确；
- `human relative-motion error` 与 V14.3 的 `absolute world-root error` 不是同一个指标；
- 目前尚未与 V14.3 calibrated human root 在同一统一方程中重新评测。

## 6.8 当前定位

如果现在只从“完整三维几何是否看起来自洽、人体是否仍站在场景中”判断，V11.4 是昨天
最合理、最适合作为最终系统骨架的版本。

---

## 7. 今天版本：V14.3 Human-Camera Coupled Re-anchoring

## 7.1 它解决的具体 bug

V18 原 camera-only 流程是：

1. 用人体投影估计 calibrated camera-frame root `r_calibrated`；
2. 用它计算 camera translation；
3. 最终显示 SMPL-X 时却继续使用 Human3R raw root `r_raw`。

相机求解满足：

```text
t_camera = human_world - R_camera * r_calibrated
```

但最终人体变成：

```text
human_world_output = R_camera * r_raw + t_camera
```

因此 camera 即使正确，最终人体仍会偏移：

```text
human_world_output - human_world_target
= R_camera * (r_raw - r_calibrated)
```

这就是“camera 对了、人没有对”的直接数学原因。

## 7.2 V14.3 的 coupled 修正

V14.3 强制相机和最终人体使用同一个 `r_calibrated`：

```text
human_world_target = R_camera * r_calibrated + t_camera

final joints  = R_camera * joints_centered  + human_world_target
final vertices = R_camera * vertices_centered + human_world_target
```

也就是说：

- calibrated root 用于反求 camera translation；
- 同一个 calibrated root 也用于放置完整 SMPL-X；
- joints 和 vertices 整体跟随 root；
- 不再只移动 camera 而把 human 留在 raw depth。

数值闭环最大误差仅 `2.73e-7 m`。

## 7.3 V14.3 Human Projection 主分支

完整流程：

```text
GT cut trigger
-> pre-decode Hard Reset Human3R
-> V16 torso-motion rotation, unified 20 deg bound
-> frozen Keypoint R-CNN 提取当前 post-cut torso 2D joints
-> 当前 Human3R SMPL-X pose/shape 提供物理人体
-> 使用 intrinsics 做 torso reprojection fitting
-> 估计 post-cut calibrated camera-frame human root
-> 用 pre-cut human history 预测当前 human world root
-> 显式求 camera translation
-> 同一个 calibrated root 放置完整 SMPL-X
-> one fixed Boundary transform + one fixed post-shot root correction
-> 后续整个 shot 固定复用
```

V14.3 主方法不使用 DA3；DA3 是并行 metric-depth 诊断分支。

## 7.4 它如何处理人体尺寸

V14.3 Human Projection 使用当前 post-cut predicted SMPL-X 物理尺寸估计深度，但不会像
V11.4 一样统一缩放完整 pointmap 和人体尺寸。它主要做的是：

- 估计一个更合理的 camera-frame root；
- 平移完整人体到这个 root；
- 保留当前 body offsets 和局部形状。

优点是不会像 DA3 full 那样无条件放大身体；缺点是 Human3R pointmap 的 metric scale
没有同步校正。

## 7.5 180-cut 结果

| Method | Camera T | T P90 | Human root | Joints | Vertices | Torso reprojection |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 1.715 | 3.718 | 0.926 | 0.946 | 0.945 | 19.2 px |
| V18 camera-only | 0.872 | 2.079 | 0.676 | 0.700 | 0.691 | 19.2 px |
| **V18 coupled full-root** | **0.872** | **2.079** | **0.444** | **0.472** | **0.468** | **6.6 px** |
| DA3 camera-only | 0.518 | 1.180 | 1.005 | 1.028 | 1.019 | 19.2 px |
| DA3 coupled full-root | **0.518** | **1.180** | **0.220** | **0.295** | **0.287** | 19.6 px |

关键结论：

- V18 coupled 完全保留 camera-only 的 camera gain；
- human root 从 `0.676 m` 降至 `0.444 m`；
- `68.3%` 样本的人体改善超过 5 cm；
- Boundary Oracle + raw human 的 root 仍有 `0.949 m` error，证明 coupled 是必要步骤。

## 7.6 人体投影完整性

| Method | Mesh bbox IoU | Width ratio | Height ratio |
|---|---:|---:|---:|
| Fixed Explicit | 0.620 | 0.830 | 0.843 |
| **V18 coupled** | **0.872** | **1.005** | **1.018** |
| DA3 alpha=0.75 | 0.729 | 1.142 | 1.153 |
| DA3 full | 0.631 | 1.455 | 1.445 |

V18 coupled 的人体高度和宽度接近真实检测框，是当前 V14.3 中 geometry-safe 的
camera-human 方案。DA3 full 虽然 camera/human metric error 更低，却会系统性放大人体，
MVHuman100 height ratio 可达 `2.30`。

## 7.7 场景安全边界

| Method | Scene discontinuity | Foot-scene distance |
|---|---:|---:|
| Fixed Explicit | **0.587 m** | **0.268 m** |
| V18 coupled | 0.998 m | 0.273 m |
| DA3 coupled | 1.382 m | 0.499 m |
| DA3 alpha=0.75 | 1.272 m | 0.375 m |

V14.3 已经让 camera 和 human 使用同一个 root，但没有修复 raw Human3R pointmap 的
metric scale。因此它还不是完整的 camera-human-scene metric closure。

这也是 V14.3 与 V11.4 最关键的区别：

- V11.4 优先保持整个人体和场景 similarity；
- V14.3 优先保证 camera 和 human root 数学闭环。

## 7.8 V14.2 continuity 的作用

V14.3 可以在 coupled alignment 后叠加轻量人体 memory：

- shape/scale blend alpha `0.25`；
- root-centered local pose alpha `0.15`；
- world root/global orientation 必须 Align-Then-Commit。

180-cut 平均结果：

| Metric | Hard Reset | Continuity memory | Change |
|---|---:|---:|---:|
| Shape jump | 0.718 | 0.558 | -22.3% |
| Body-scale jump | 0.00751 | 0.00577 | -23.2% |
| Local-pose residual | 5.37 deg | 4.58 deg | -14.8% |

但平均 joint/mesh 视觉变化只有 `1.15/1.53 px`。因此 continuity 是轻量稳定器，不是
camera/human 对齐的主要来源，也不应被描述成显著视觉贡献。

## 7.9 当前 8105 viewer 实际显示什么

当前 viewer 使用：

```text
V18 Human Projection Coupled
+ V14.2 Continuity
```

它没有使用 V11.4 uniform similarity，也没有使用 conditional VGGT。绿色实心人体与
黄色线框人体的 camera、root 和 scene 完全相同，只比较 continuity memory。因此两者
看起来接近是正常结果。

当前选定 THuman 案例中，coupled 相对 Fixed：

- camera translation `0.540 -> 0.227 m`；
- human root `0.136 -> 0.067 m`；
- world joints `0.187 -> 0.054 m`；
- torso reprojection `9.86 -> 6.06 px`；
- mesh bbox IoU `0.912 -> 0.951`；
- foot-scene distance `0.192 -> 0.175 m`。

这些大改善来自 coupled alignment；绿色和黄色之间的微小差异才来自 continuity。

---

## 8. 三种路线的直接比较

| 维度 | 旧 V47 / V11.1 | 旧 V46 / V11.2 | V11.4 Uniform Similarity | V14.3 Coupled |
|---|---|---|---|---|
| 主要目标 | 修 rotation tail | 修 metric 后脚地接触 | 统一整个 shot scale | 修 camera-human root 不一致 |
| Human3R | frozen + reset | frozen + reset | frozen + reset | frozen + reset |
| Torso motion | 是 | 是 | 是 | 是，20 deg bound |
| VGGT | 困难 cut 才运行 | 使用 V47 rotation | 困难 cut 才运行 | 当前主实验不使用 |
| DA3 | 最终 raw-scale 不使用 | 使用独立 root/scene scale | 使用 cut-time scale cue | 主分支不使用；并行诊断使用 |
| 2D detector | 不需要 | 不需要 | 不需要 | frozen Keypoint R-CNN |
| Scale 类型 | 原始 Human3R gauge | root/scene 独立 scale | 统一 shot-level similarity | human projection root calibration |
| Camera 与 human root 同源 | raw root，同源但不一定 metric | 接触修正后重求 | 同一 uniform scale | **同一 calibrated root** |
| 完整 body offsets 同步 | 不改变 | 否 | **是** | 保持当前 body，不做统一 scene scale |
| Pointmap 同步 | 同一 SE(3) | 独立 scene scale | **同一 scale + SE(3)** | 同一 Boundary，但 local metric scale 未修 |
| Human reprojection shift | 0 px | 112.1 px | 0 px | V18 torso 6.6 px absolute error |
| Foot/ground distortion | 0 | proxy 0，但投影坏 | **0** | 平均接近 Fixed，但 scene closure 未完成 |
| Camera T mean | 1.568 m | 0.465 m | **0.397 m** | 0.872 m |
| Human absolute root | 未按 V14.3 指标报告 | 未按 V14.3 指标报告 | 未按 V14.3 指标报告 | **0.444 m** |
| 部署判断 | 安全但不 metric | 拒绝 | 昨天整体几何最佳 | camera-human 成立，scene 未闭环 |

注意：V11.4 报告的 `human relative-motion error` 与 V14.3 的 `absolute world-root error`
定义不同，不能直接把 `0.012 m` 和 `0.444 m` 当作同一指标比较。Camera translation
虽然都在同一 180-cut 协议上报告，也应结合各自的 old-shot metric gauge 审计理解，不能
只凭单一 mean 宣布方法全面胜出。

---

## 9. 哪个方法更合理

## 9.1 只看现有完整三维输出

**V11.4 Uniform Similarity 更合理。**

原因：

- camera、pointmap、human root 和完整人体尺寸使用同一 scale；
- 保留 RGB reprojection；
- 保留脚地接触；
- 保留 conditional VGGT 的 rotation tail gain；
- 20 帧长序列中整体几何关系最稳定。

## 9.2 只看 camera-human 数学一致性

**V14.3 Coupled 更合理。**

原因：

- 明确消除了 camera 使用 calibrated root、human 使用 raw root 的矛盾；
- 公式闭环可直接验证；
- 同时报告 camera、root、joints、vertices 和 reprojection；
- Boundary Oracle + raw body 的反例提供了强因果证据。

## 9.3 旧 V46 是否合理

不适合作为最终方法。它证明接触约束重要，但“最后平移 root 把脚拉回地面”会产生
`112.1 px` reprojection shift。它更适合作为失败消融和问题发现过程。

---

## 10. 哪个方法更有创新性

创新性需要区分“新基础模型”和“新的系统/几何 formulation”。这些方法都大量使用冻结
预训练模型，因此不能宣称发明了 Human3R、VGGT、DA3 或 Keypoint R-CNN 的能力。

## 10.1 V11.1 / 旧 V47

创新性主要来自：

- 把 camera-cut difficulty 定义为 residual geometry conflict，而不是 source 或角度；
- source-independent conditional VGGT tail rescue；
- 通过多 holdout 证明“外部模型只应在困难 tail 使用”。

这是有价值的系统设计，但核心 rotation 信息仍来自预训练 VGGT 和人体 torso。

## 10.2 V11.2 / 旧 V46

把 foot-ground contact 显式加入 camera/root re-solving 有一定物理动机，但当前 formulation
破坏 reprojection，难以成为最终创新点。它更像说明“错误的尺度不能靠局部接触补丁
修复”的负面证据。

## 10.3 V11.4

创新性主要来自：

- camera-cut 后的完整 human-camera-scene similarity consistency；
- 明确要求 camera translation、pointmap、root 和 body offsets 共用 shot scale；
- 用投影不变性和 contact preservation 约束外部 metric cue。

相比简单“用 DA3 改尺度”，这个 formulation 更严谨，也更容易成为论文方法的一部分。
但 scale cue 本身仍来自 DA3，必须把贡献描述为 **coherent metric integration**，不能描述
为新的 metric-depth 网络。

## 10.4 V14.3

创新性主要来自：

- 把人体当作 camera translation 的动态物理锚点；
- 同一个 calibrated human root 同时约束 camera translation 和最终 SMPL-X placement；
- 把 camera metric accuracy 与 final human reconstruction consistency 联合评价；
- 证明 camera-only GT 指标改善并不等于完整重建改善。

这一因果链比单纯后处理 camera pose 更清晰，可能是当前最有论文表达潜力的部分。但它
还缺 scene metric closure，因此单独作为完整方法仍不够。

---

## 11. 哪个更适合后续使用

## 11.1 立即部署或继续做可视化

优先使用 **V11.4 Uniform Similarity - Conditional Wide** 作为完整几何骨架，因为它最
不容易出现整体缩小、人体悬空、脚陷地或人景比例失真。

## 11.2 继续提高 camera-human 精度

保留 **V14.3 coupled equation**，但不能直接覆盖到 V11.4 后面。否则 V11.4 已经应用的
shot scale 与 V14.3 root depth correction 可能重复修改尺度。

## 11.3 最合理的下一阶段统一方法

建议形成一个统一的 shot-level Sim(3) + coupled root solve：

```text
Hard Reset Human3R
-> Fixed Explicit
-> V16 torso rotation
-> conditional VGGT only for difficult rotation tail
-> DA3/human projection jointly提供 metric constraints
-> 求一个统一 shot scale s
-> camera、pointmap、human root、body offsets 全部使用 s
-> calibrated human root 同时进入 camera translation 和 final SMPL-X placement
-> 一个固定 post-shot Boundary
-> V14.2 continuity only after alignment
-> Align-Then-Commit
```

统一求解必须满足：

```text
human_world = R * human_camera_calibrated + t

scene_world = R * (s * scene_local) + t

body_world = R * (s * body_centered + human_camera_calibrated) + t
```

具体 convention 需要根据 Human3R camera-to-world 定义写成同一左乘形式，但核心约束是：

- `s` 只能有一套；
- camera/human 不能使用不同 root；
- pointmap/body 不能独立缩放；
- 不能在末尾再做一个破坏 reprojection 的 foot correction。

## 11.4 必须重新报告的指标

统一版本不能只比较 camera translation。至少同时报告：

- camera translation/rotation mean、median、P90、P95；
- final human world-root、joints、vertices；
- mesh/joint reprojection；
- scene discontinuity；
- foot-ground distance；
- body height/width ratio；
- camera-human joint success；
- 四个数据源分别结果；
- 1/2/4/8-cut drift；
- 正常无-cut exact no-op。

只有同时优于或保持 V11.4 的 scene/contact integrity，并保留 V14.3 的 human-root gain，
才能称为最终统一方法。

---

## 12. 给评审 AI 的建议判断框架

请不要只问“哪个 camera error 最小”，而应依次判断：

1. 方法是否保持 camera、pointmap、SMPL-X 在同一个明确 gauge 中？
2. 是否只有一个统一 shot scale？
3. camera translation 和 final human 是否使用同一个 calibrated root？
4. 是否保持 RGB reprojection？
5. 是否保持人体与场景接触？
6. 是否在无 GT、无 source ID、无完整未来 shot 时成立？
7. 外部预训练模型是在 cut 时提供物理 cue，还是替代了全部核心能力？
8. 困难分支是否有 source-independent trigger 和独立 holdout 验证？
9. 结果是否同时覆盖 camera、human 和 scene，而不是只优化其中之一？
10. 新贡献是新的模型、几何 formulation，还是对预训练模型的工程拼装？

基于现有证据，一个谨慎的暂时结论是：

- **完整几何合理性**：V11.4 最强；
- **rotation tail**：V11.1/旧 V47 最可靠；
- **camera-human 因果一致性**：V14.3 最清晰；
- **旧 V46**：保留为接触约束失败消融；
- **最终研究方向**：统一 V11.4 similarity 与 V14.3 coupled root，而不是二选一或简单叠加。

---

## 13. 代码、报告和结果位置

### 活跃代码

```text
scripts/v11_1_boundary_method_comparison_viewer.py
scripts/v11_2_contact_preserving_probe.py
scripts/v11_3_component_ablation.py
scripts/v11_4_uniform_similarity_probe.py
scripts/v12_2_long_sequence_viewer.py
versions/v12/experiments/v14_3_projection_consistent_reanchoring_probe.py
versions/v12/experiments/v14_3_human_continuity_visualization.py
versions/v12/experiments/v14_3_interactive_continuity_viewer.py
```

### 核心报告

```text
docs/movie3r/v11/V11_RETAINED_GEOMETRY_INTEGRITY.md
versions/v12/docs/V14_3_PROJECTION_CONSISTENT_REANCHORING.md
versions/v12/docs/ACTIVE_BOUNDARY_ALIGNMENT.md
docs/movie3r/BOUNDARY_VERSIONING.md
```

### 机器可读结果

```text
output/v46_contact_preserving_metric_bridge/v46_contact_preserving_metric_bridge_probe.json
output/v48_component_necessity_ablation/v48_component_necessity_ablation.json
output/v53_uniform_similarity_integrity/v53_uniform_similarity_integrity_probe.json
output/v14_3_projection_consistent_reanchoring/quantitative/v14_3_projection_consistent_reanchoring.json
```

### 长序列缓存和 viewer

```text
output/v52_long_sequence_visualization/cache/
scripts/v12_2_long_sequence_viewer.py
versions/v12/experiments/v14_3_interactive_continuity_viewer.py
```

### 历史归档

```text
archive/20260721/
```

旧文件名继续存在仅用于追溯，不表示它仍是当前活跃版本。
