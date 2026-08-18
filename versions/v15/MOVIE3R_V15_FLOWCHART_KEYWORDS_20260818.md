# Movie3R-v15 论文流程图关键词与绘图说明

日期：2026-08-18  
对应版本：Movie3R-v15-final  
状态：当前冻结主线，用于批量实验与 ICLR 论文方法图  
推荐论文题目：

> **Movie3R: Causal Camera–Human Gauge Transactions for Streaming 3D Reconstruction across Shot Cuts**

中文定位：

> **面向跨镜头流式三维重建的因果、自适应、可拒绝相机—人体坐标事务框架。**

本文档不是新的方法设计，而是把当前冻结方法转换成可供绘图 AI、设计师或 PowerPoint 使用的视觉关键词。主图应当“看起来简单”，但必须准确表达状态所有权、坐标系桥接、人物匹配、人体精修、相机—人体联合提交和安全回退。

### 给绘图 AI 的推荐投喂顺序

不要一次把整份长文档全部丢给绘图模型。推荐：

1. 首先提供 **第 10.1 节 ICLR 主图包装版**；
2. 接着提供 **第 10.2 节视觉优先级**；
3. 最后附上 **第 12 节 Negative Prompt**；
4. 只有当模型把方法关系画错时，再补充第 5 节对应模块或第 10.3 节技术约束；
5. 第 14–19 节用于人工审图和论文写作，不需要全部输入绘图模型。

这样可以让图在保持技术正确的同时，优先呈现论文核心 insight，而不是变成拥挤的软件架构图。

---

## 1. 一句话流程

> 输入连续 RGB 流；因果 detector 发现 shot cut 后，同一个 Human3R 派生模型并行运行 clean reset 与只读 shadow 分支，V9/B0 提出粗跨镜头坐标桥；随后在 B0 坐标中完成匿名人物匹配、相机冻结的人体 root/depth 精修和 shot 内静态稳定；若人体几何证明相机仍不可信，则共同修正 post-shot 相机与人体，否则精确回退到可信 baseline；最终只提交 clean state，并把一次修正因果传播到后续帧。

最简关键词链：

~~~text
RGB stream
→ causal shot detector
→ clean reset + read-only shadow
→ V9/B0 coarse gauge proposal
→ permutation-aware identity association
→ BRTC-LC camera-frozen human refinement
→ C1-EMA25 within-shot stabilization
→ adaptive camera-human geometry gate
→ accept joint update OR exact fallback
→ atomic commit + causal propagation
→ camera + scene + SMPL-X humans + persistent IDs
~~~

---

## 2. 图必须讲清楚的核心问题

### 2.1 Shot cut 造成两类断裂

镜头切换前后记为：

~~~text
Pre-shot:  ... I[b-2], I[b-1]
                         │
                         ╳ abrupt camera cut
                         │
Post-shot:              I[b], I[b+1], ...
~~~

直接延续旧 recurrent state：

~~~text
Carry old state
→ 旧场景状态污染新镜头
→ state contamination
~~~

直接清空旧 state：

~~~text
Hard reset
→ 新镜头重建干净
→ 但落在独立的 local gauge
→ world-gauge discontinuity
~~~

即使相机坐标大致接上，人体仍可能存在：

- 跨 shot 人物 ID 置换；
- root/depth 偏差；
- body orientation 偏差；
- 静止人物在 shot 内漂移；
- 低纹理背景下相机旋转和平移错误。

因此，主图中心不是“一个更大的神经网络”，而是：

> **Proposal → Verification → Commit / Abstain**

中文：

> **先提出坐标修正，再用人体几何验证，最后只提交可信修正；证据不足时拒绝更新。**

### 2.2 最重要的论文视觉概念

图中必须突出三个 ownership：

1. **State ownership**：只有 clean reset branch 的 recurrent state 可以继续写入未来。
2. **Gauge ownership**：shadow branch 只能提出 B0 坐标桥，不能成为长期状态。
3. **Geometry ownership**：相机、场景和人体只有在对应 gate 通过后才允许改变。

推荐在图中央放一个醒目的横向标签：

~~~text
CAUSAL GAUGE TRANSACTION
Proposal          Verify              Commit / Abstain
~~~

---

## 3. 推荐的主图整体版式

### 3.1 画布

- ICLR 双栏横跨页宽的 Figure*；
- 横向比例约 2.5:1 到 3:1；
- 推荐最终尺寸约 7.0 in × 2.6–3.0 in；
- 优先输出 SVG/PDF 矢量图；
- 白色背景；
- 最终论文缩放后字体至少 7.5–8 pt；
- 主流程从左到右；
- 最多 8 个视觉大模块；
- 模块内部细节使用 1–3 行小字或图标，不要再拆成十几个同等大小的方框。

### 3.2 三层信息带

推荐用三条水平带组织：

~~~text
上层：时间与状态
RGB timeline → cut proposal → pre-state snapshot / clean new state

中层：主方法
dual branches → B0 → ID → BRTC → adaptive gate → commit

下层：安全性与输出
exact fallback / C1 stabilization → causal propagation → camera + scene + humans
~~~

更具体的布局：

~~~text
┌──────────────┐   ┌─────────────┐   ┌──────────────────────────────┐
│ RGB timeline │ → │ Causal GRU  │ → │ Cut: dual same-model paths  │
│ pre | cut |  │   │ cut proposal│   │ Clean reset / Read-only V9  │
└──────────────┘   └─────────────┘   └──────────────────────────────┘
                                                │
                                                ▼
┌──────────────┐   ┌─────────────┐   ┌──────────────────────────────┐
│ Final output │ ← │ Transaction │ ← │ B0 → ID → BRTC → C1 → Gate │
│ C/S/H/ID/diag│   │ commit      │   │ accept joint / exact fallback│
└──────────────┘   └─────────────┘   └──────────────────────────────┘
~~~

### 3.3 视觉主次

视觉最大的模块：

1. shot cut 前后的双分支；
2. B0 粗 gauge；
3. adaptive gate 的 accept/fallback 分叉；
4. 最终相机—场景—人体统一输出。

视觉较小但必须出现：

- anonymous identity association；
- BRTC-LC；
- C1-EMA25；
- shadow state discarded；
- pre-shot unchanged；
- no future frames；
- exact fallback。

---

## 4. 配色、形状和箭头语义

### 4.1 推荐配色

颜色控制在 5 种以内。

| 语义 | 推荐颜色 | Hex 示例 | 用法 |
|---|---|---:|---|
| RGB / 输入 / 普通流 | 深灰蓝 | #334155 | 时间轴、输入帧 |
| 可学习模块 | 靛蓝或紫色 | #4F46E5 / #7C3AED | GRU detector、V9 shadow |
| Clean state / 可提交状态 | 青蓝或蓝绿 | #0891B2 / #0F766E | clean reset、state owner |
| 显式几何验证 | 橙色 | #EA580C / #F59E0B | ID、BRTC、Kabsch、root rays |
| 接受与最终提交 | 绿色 | #16A34A | accepted joint update、commit |
| 拒绝与 fallback | 中性灰 | #64748B | exact baseline fallback |
| 禁止/丢弃 | 红色 | #DC2626 | discarded shadow state、forbidden future |

### 4.2 形状

- 圆角矩形：普通模块；
- 双层圆角矩形：同一个模型的两个运行角色；
- 菱形：gate / accept or reject；
- 圆柱或小 memory icon：recurrent state；
- 相机视锥：camera pose；
- 彩色人体轮廓或简化 SMPL-X mesh：human；
- 稀疏点云小图标：scene / pointmap；
- 两条射线交会：BRTC triangulation；
- 环形旋转箭头：Kabsch shared rotation；
- 绿色锁或 check：committed；
- 红色叉号或垃圾桶：shadow state discarded。

### 4.3 箭头

- 粗实线：被提交并进入未来的主数据流；
- 细实线：显式几何数据；
- 紫色虚线：只读 proposal，不拥有未来 state；
- 绿色箭头：通过 gate 的更新；
- 灰色旁路箭头：exact fallback；
- 红色截断箭头：禁止提交的 shadow state；
- 向右时间箭头：因果 post-shot propagation；
- 不要画任何从未来 post 帧返回 boundary 的箭头。

推荐在图例中写：

~~~text
solid = committed stream
dashed = read-only proposal
green = accepted update
gray = exact fallback
red × = never committed
~~~

---

## 5. 逐模块绘图关键词

下面每个模块都给出：框中文字、输入、处理、输出、图标和必须避免的误解。

### M0. 输入：连续 RGB 流

框标题：

> **Streaming RGB**

副标题：

> Pre-shot → Cut → Post-shot

画面：

- 3–5 个小 RGB 缩略图；
- shot 前图像使用同一浅蓝底；
- shot 后图像使用另一浅灰或浅橙底；
- 中间用闪电、剪刀或竖直断线标出 camera cut；
- 标出最后 pre 帧 I[b−1] 与第一 post 帧 I[b]。

关键词：

~~~text
monocular RGB stream
edited video
abrupt camera cut
last pre frame
first post frame
causal input
online
~~~

输入：

- 当前 RGB 帧；
- 过去已看见的 RGB / bounded recurrent state。

边界处禁止输入：

- future post frames；
- GT camera；
- GT body；
- GT identity。

### M1. 因果 Shot Detector

框标题：

> **Causal Cut Proposal**

副标题：

> RGB pair features + 3-step GRU

框内小字：

~~~text
RGB / gray difference
color histogram
optical flow
ORB matches
→ p(cut)
~~~

输入：

- 相邻帧 I[t−1], I[t]；
- 最近 3 个 pair-feature 历史。

处理：

- 廉价图像差异统计；
- causal GRU；
- 输出 cut probability；
- 不读取 3D GT、不读取未来帧。

输出：

- p_cut；
- boundary event proposal e_b。

训练属性：

- 可学习的小型模块；
- 当前冻结审计结果：F1 0.982、FPR 0.031、Brier 0.015；
- batch 实验允许用已知 cut timestamp，但要与 detector 结果分开报告。

视觉：

- 小 GRU 单元；
- 上方写 Learned；
- 输出进入几何事务前写 Proposal only；
- detector 不能直接改变 3D 世界。

必须避免：

- 不要画成大型 CNN / ViT；
- 不要画成 detector 一触发就强制改相机；
- 不要画 GT cut 进入默认推理。

### M2. Boundary 时的双分支：同一模型、不同所有权

这是主图最重要的结构之一。

共同输入：

- 第一帧 post RGB I[b]；
- 同一个 Human3R-derived checkpoint。

#### M2-A. Clean Reset Branch

框标题：

> **Clean Reset**

副标题：

> New-shot state owner

输入：

- I[b]；
- empty/reset state；
- correction OFF。

输出：

- raw post camera C_raw；
- raw post scene / pointmap；
- raw SMPL-X humans；
- clean recurrent state S_raw。

语义：

- 新 shot 唯一合法的长期 state；
- 后续 post 帧从该 clean state 继续；
- 不允许旧 shot 污染它。

视觉：

- 青蓝色实线框；
- memory icon 上加绿色锁；
- 从它到 commit 使用粗实线。

框内最短文字：

~~~text
Human3R, reset state
camera + scene + people
✓ future state owner
~~~

#### M2-B. Read-only V9 Shadow Branch

框标题：

> **Read-only Shadow**

副标题：

> Learned coarse proposal, never committed

输入：

- I[b]；
- pre-shot read-only state snapshot；
- cut event；
- V9 correction ON。

V9 内部可用小字：

~~~text
semantic relation token
alignment relation token
momentum relation token
decoder attention
gated pose/human residual
LoRA-adapted heads
~~~

输出：

- shadow camera C_shadow；
- coarse shadow body / pose proposal；
- 临时 shadow state。

核心定位：

- V9 是 learned coarse gauge proposal；
- V9 是 identity preconditioner；
- V9 不是最终精对齐器；
- shadow state 在 proposal 后丢弃。

视觉：

- 紫色虚线框；
- 从 pre-state 到 shadow 使用虚线；
- shadow 输出到 B0 使用虚线；
- shadow state 旁画红色 ×，文字 Never committed；
- 不要让 shadow 分支的 state 连到未来。

框内最短文字：

~~~text
V9 + read-only old state
coarse camera/body proposal
✗ shadow state discarded
~~~

#### 双分支上方总标签

~~~text
Same Human3R-derived checkpoint
Two roles, different ownership
~~~

### M3. B0：Learned Coarse Gauge Bridge

框标题：

> **B0 Coarse Gauge**

副标题：

> Reset-to-world bridge

核心公式：

~~~text
T_B0 = C_shadow · inverse(C_raw)
~~~

输入：

- clean-reset camera C_raw；
- shadow camera C_shadow；
- raw scene；
- raw post humans。

处理：

- 计算一次 4×4 SE(3) 坐标桥；
- 同一个 T_B0 作用于 post camera、background/pointmap 和所有 post humans；
- 把新 shot 从独立 local gauge 搬到 pre-shot world gauge 的大致 basin；
- 对整个 post shot 缓存和复用。

输出：

- B0 camera；
- B0 scene；
- B0 humans；
- 可比较的 pre/post 粗世界坐标。

视觉：

- 两个相机视锥：raw 灰色、shadow 紫色；
- 中间一个 4×4 transform 矩阵或旋转+平移箭头；
- 输出相机、点云、多人一起被搬动；
- 文字 Same transform for camera + scene + all people。

框内最短文字：

~~~text
C_shadow · C_raw⁻¹
camera + scene + all humans
coarse, shot-level, cached
~~~

必须避免：

- 不要把 B0 画成最终精对齐；
- 不要画成只移动人体；
- 不要画成只靠原版 Human3R 相机 SE(3)；
- 不要画成逐帧重新估计。

### M4. 匿名跨 Shot 人物身份匹配

框标题：

> **Anonymous ID Association**

副标题：

> Permutation-aware, confidence-gated

输入：

- last pre-shot people；
- first post-shot B0 people。

匹配特征：

~~~text
root position
torso direction and scale
root-centered joints
centered mesh/body shape
~~~

处理：

- 枚举小规模 permutation 或 Hungarian one-to-one assignment；
- 最优 assignment；
- second-best assignment；
- permutation margin；
- unmatched / count mismatch。

输出：

- persistent cross-shot person IDs；
- permutation π*；
- association confidence / margin；
- unmatched tracks。

视觉：

- 左右两组彩色人形；
- pre 的人标 A/B/C，post 原始顺序乱序；
- 交叉连线经过 matcher 后恢复同色；
- 第二佳匹配用淡灰虚线；
- margin 太小时进入 abstain。

框内最短文字：

~~~text
root + torso + centered joints
one-to-one permutation
persistent ID + margin
~~~

必须准确表述：

> Human3R 有帧内 native detection index，但没有可靠的 persistent cross-shot identity。

不要写：

> Human3R 完全没有 ID。

当前受控证据可作为可选小徽标：

~~~text
41 multi-person cuts
direct matching: 41.5–46.3%
after B0: 100%
~~~

### M5. BRTC-LC：相机冻结的人体精对齐

全称：

> **Camera-Frozen Boundary Ray Triangulation with Layout Consensus**

框标题：

> **BRTC-LC**

副标题：

> Camera frozen; refine human root/depth/layout

输入：

- matched pre/post people；
- pre camera；
- B0 post camera；
- 五个核心 SMPL-X joints。

五个关节：

~~~text
pelvis
left hip
right hip
left shoulder
right shoulder
~~~

几何步骤：

1. 从 pre camera center 到 pre joint 形成 ray；
2. 从 B0 post camera center 到 post joint 形成 ray；
3. 求两条射线最近点和 midpoint；
4. 得到 joint-wise root/depth candidates；
5. 用 ray gap、parallax sine、MAD gate 去掉不可观测关节；
6. 有效关节取 robust median；
7. 通过多人 pairwise layout consensus 选择 group/individual residual；
8. 将同一个 bounded translation 写入该人的 root、joints、vertices。

明确不改变：

- camera；
- background；
- body articulation pose；
- shape；
- global orientation。

输出：

- refined human root/depth；
- refined pairwise layout；
- rejected track 保持 exact B0。

视觉：

- 两台相机视锥；
- 同一个人体的五个彩色关节点；
- 两组橙色射线相交；
- midpoint 或 closest-point segment；
- 旁边小锁写 Camera locked；
- 人体沿深度方向平移；
- 不画相机移动箭头。

框内最短文字：

~~~text
5 core joints + two-view rays
gap / parallax / MAD gates
root-depth shift only
🔒 camera unchanged
~~~

当前确认集可选小徽标：

~~~text
42 cuts / 125 people
root: 0.378 → 0.231 m
joint: 0.412 → 0.275 m
vertex: 0.389 → 0.253 m
camera change: 0
~~~

### M6. C1-EMA25：Shot 内静止人体稳定

框标题：

> **C1-EMA25**

副标题：

> Causal within-shot static stabilization

输入：

- 已匹配 track；
- 当前和历史 camera-local root/body steps；
- B0+BRTC output。

处理：

- causal exponential moving average；
- alpha = 0.25；
- warmup = 2；
- static/moving hysteresis；
- root enter 0.01 m，exit 0.025 m；
- body enter 0.015 m，exit 0.035 m；
- moving hold = 3 frames；
- correction cap = 0.15 m。

只对以下 track 稳定：

- persistent ID 已建立；
- 可见；
- 历史足够；
- 被判断为静止；
- gate 可信。

以下情况 exact fallback：

- 真实运动；
- 新出现/消失；
- 未匹配；
- 短历史；
- 可见性变化；
- 几何拒绝。

更新：

- 对 root、joints、vertices 加同一个 bounded translation；
- camera 永远不变；
- 不使用未来帧。

视觉：

- 三个时间帧的人体轮廓；
- 静止人体有抖动虚影，经过 EMA 后稳定；
- 运动人体旁写 preserve motion；
- camera 上画锁；
- 可画一个小 hysteresis switch。

框内最短文字：

~~~text
static? → causal EMA α=.25
stabilize root/joints/mesh
moving/uncertain → fallback
🔒 camera unchanged
~~~

时间语义提示：

> C1 是 post-shot 因果传播中的 shot 内模块。图中可放在 BRTC 后，但应画一条横跨后续 post 帧的时间带，避免让人误以为 boundary 决策偷看未来帧。

### M7. Adaptive Shared Camera–Human Gate

框标题：

> **Adaptive Geometry Gate**

副标题：

> Is the residual observable and shared?

这是菱形决策模块。

输入：

- B0/BRTC 人体；
- pre/post anonymous association；
- clean-reset/raw reference root rays；
- B0 root rays；
- shared body residual。

候选估计：

- 对所有匹配人物及其 SMPL-X vertices 做 shared Kabsch；
- 同时搜索多人 permutation；
- 得到 post-to-pre shared body rotation ΔR；
- 计算 vertex RMS；
- 计算 body-scale-normalized RMS；
- 计算 best-vs-second permutation margin。

冻结 gate：

~~~text
shared rotation ≥ 20°
vertex RMS ≤ 0.20 m
normalized RMS ≤ 0.20
permutation margin ≥ 0.01 m
~~~

为什么 rotation 要足够大：

- 小 residual 的多人高纹理场景通常 B0 camera 已经可信；
- 此时强行联合旋转可能破坏正确背景；
- 因而小 residual 直接 exact fallback，而不是“必须优化”。

视觉：

- 橙色菱形；
- 菱形内写 Trustworthy shared residual?；
- 旁边四个小 gate 指标；
- 左下或下方灰色 Reject；
- 右侧绿色 Accept。

框内最短文字：

~~~text
shared Kabsch + permutation
rotation / RMS / scale / margin
observable?
~~~

### M8-A. Accept：联合修正相机、人体和场景

框标题：

> **Shared Camera–Human Update**

副标题：

> Accept and commit one causal boundary transform

人体更新：

- 使用 shared body rotation ΔR；
- 每个人体绕当前 BRTC refined root 旋转；
- 保留已经可信的 root location；
- root、mesh 和 joints 保持一致。

相机更新：

- camera orientation 使用同一个 ΔR；
- camera translation 不单独依赖低纹理背景；
- 使用 B0 path 与 clean raw reference 的 person root rays；
- 对匹配人物求平均 camera center；
- camera 与 human 相对位置保持一致。

场景更新：

- background / pointmap 跟随实际 camera-world transform；
- 不把人体单独贴过去而留下错误相机；
- 相机、场景、人体在一个 shared gauge transaction 内一致改变。

输出：

- corrected post camera；
- corrected post scene/pointmap；
- corrected post humans；
- accepted boundary transform；
- 后续 post 帧复用。

视觉：

- 一个绿色 shared transform 环包住 camera + point cloud + people；
- camera 和 human 同方向旋转；
- 人体根部用小锚点固定；
- 标注 Shared gauge update；
- 绿色 check。

框内最短文字：

~~~text
body rotates around BRTC root
camera rotation + root-ray translation
scene follows camera-world transform
✓ shared consistent update
~~~

### M8-B. Reject：Exact Baseline Fallback

框标题：

> **Exact Fallback**

副标题：

> Keep B0 + BRTC-LC + C1 unchanged

触发条件：

- residual 太小；
- body fit 不一致；
- scale-normalized RMS 过大；
- permutation margin 不足；
- 人数变化；
- unmatched；
- 缺少 raw reference；
- 几何不可观测。

语义：

- 不执行危险的 camera-human 世界变换；
- 输出必须与直接父 baseline 精确相同；
- abstention 是设计能力，不是失败。

视觉：

- 灰色旁路箭头；
- 盾牌图标；
- 标注 No harmful update；
- 直接汇入 transaction commit。

框内最短文字：

~~~text
ambiguous / unobservable
→ exact parent output
→ no harmful update
~~~

### M9. Atomic Transaction Commit

框标题：

> **Atomic Commit**

副标题：

> Clean state + verified geometry only

提交内容：

- clean reset recurrent state；
- cached B0 / accepted boundary transform；
- persistent person IDs；
- accepted BRTC residuals；
- C1 per-track causal state；
- adaptive gate decision；
- diagnostic reason。

永不提交：

- shadow recurrent state；
- 未通过验证的几何；
- GT；
- future information。

必须满足：

~~~text
pre-cut outputs unchanged
post-cut only
first-frame causal
fixed bounded state
exact fallback
~~~

视觉：

- 绿色 commit box；
- 事务/数据库 commit 图标；
- clean state 实线进入；
- verified geometry 绿色进入；
- shadow state 红叉停在外面。

框内最短文字：

~~~text
clean state
+ verified gauge/person residuals
→ one atomic transaction
~~~

### M10. Causal Post-shot Propagation

框标题：

> **Causal Propagation**

副标题：

> Reuse one committed correction

处理：

- 后续帧正常运行 clean Human3R stream；
- 复用已提交的 shot-level transform；
- 复用 persistent IDs 和 C1 track state；
- 不逐帧重新做跨 shot 全局优化；
- 下一次 cut 再开启新 transaction。

视觉：

- I[b], I[b+1], I[b+2] 向右时间箭头；
- 相机轨迹连续；
- 同色人物 ID 连续；
- point cloud 处于同一 world；
- 每次 cut 只有一个 transaction marker。

关键词：

~~~text
online
streaming
causal
fixed memory
one update per cut
no history rewrite
no future smoothing
~~~

### M11. 最终输出

框标题：

> **Unified World Output**

副标题：

> Camera + Scene + SMPL-X Humans + Persistent IDs

输出：

- world camera trajectory；
- camera frustums；
- dense scene / pointmap / confidence；
- multiple SMPL-X human meshes；
- roots and joints；
- persistent cross-shot person IDs；
- boundary diagnostics；
- gate accept/reject reason；
- demo.py-compatible 3D payload。

视觉：

- 一个统一三维世界；
- 蓝/黄相机视锥；
- 背景点云；
- 1–3 个不同颜色的人体 mesh；
- 跨 shot 相同人物保持同色；
- 连续轨迹线；
- diagnostics 小 JSON 图标。

框内最短文字：

~~~text
globally consistent camera
dense scene / pointmap
multi-person SMPL-X + persistent IDs
auditable diagnostics
~~~

---

## 6. 两种典型场景必须作为小插图

主流程图旁边建议放两个很小的 behavior inset，说明“自适应”为什么必要。

### Case A：多人、高纹理、相机基本正确

标题：

> **Textured Multi-person: Protect the Camera**

现象：

- 背景特征丰富；
- Human3R/B0 camera 已较准；
- 但人物 detection order / identity / root depth 可能错；
- shared body rotation residual 较小。

系统行为：

~~~text
B0 coarse gauge
→ anonymous ID association
→ BRTC human root/depth refinement
→ C1 static stabilization
→ adaptive gate rejects large joint update
→ exact baseline fallback
~~~

视觉：

- 背景有建筑/纹理；
- 三个人物颜色被重新匹配；
- 相机画锁；
- gate 指向灰色 fallback；
- 标签：camera preserved, humans refined。

关键词：

~~~text
multi-person
rich texture
camera reliable
ID permutation
human-only refinement
safe abstention
zero camera change
~~~

### Case B：单人、低纹理、相机错误

标题：

> **Low-texture Single-person: Use the Human as Anchor**

现象：

- 背景纯色或纹理弱；
- 原版/B0 camera rotation/translation 可能错；
- 人体结构仍相对一致；
- shared body rotation residual 大且低 RMS。

系统行为：

~~~text
body Kabsch residual accepted
+ B0/raw root rays
→ shared camera-human correction
→ body rotates around refined root
→ camera and scene move consistently
~~~

视觉：

- 简单低纹理背景；
- 一个彩色人体作为 anchor；
- 错误 camera 用红色虚线；
- 修正 camera 用绿色实线；
- camera、scene、human 被同一个绿色环形箭头连接。

关键词：

~~~text
single person
low texture
camera gauge failure
human anchor
root rays
shared camera-human update
relative geometry preserved
~~~

当前受控案例的可选结果徽标：

~~~text
AvatarReX low texture
camera: 1.703 m / 66.56° → 0.054 m / 0.44°
MPVPE: 0.247 m → 0.123 m
~~~

注意：这些数字是受控案例，不要在方法主图中画成大规模 benchmark 结论。更适合 teaser、qualitative 或 ablation inset。

---

## 7. Learned 与 Explicit 模块的标注

绘图时建议在模块上方加小标签，帮助审稿人快速理解系统不是“全手工后处理”，也不是“黑盒网络全包”。

### Learned

1. **Human3R/CUT3R backbone**
   - streaming recurrent 3D reconstruction；
   - camera + pointmap + SMPL-X humans。
2. **V9 cut-conditioned shadow adapter**
   - relation tokens；
   - coarse gauge proposal；
   - identity preconditioner。
3. **Causal GRU shot detector**
   - RGB pair statistics；
   - p_cut。

### Explicit / Geometry-based

1. B0 SE(3) bridge；
2. anonymous one-to-one ID association；
3. BRTC-LC ray triangulation；
4. layout consensus；
5. C1 EMA + hysteresis；
6. shared Kabsch；
7. root-ray camera translation；
8. gate / abstain / exact fallback；
9. atomic commit。

推荐小图例：

~~~text
Purple = learned proposal
Orange = explicit geometric verification
Green = accepted transaction
Gray = exact fallback
~~~

---

## 8. 主图建议保留的公式

主图最多保留 2–3 个公式，不要塞满数学。

### 粗坐标桥

~~~text
T_B0 = C_shadow C_raw⁻¹
~~~

含义：

> 同一个变换作用于 post camera、scene 和所有 humans。

### 匿名匹配

~~~text
π* = arg min_π Σ_i d(H_pre^i, T_B0 H_post^{π(i)})
~~~

含义：

> 在粗统一坐标中寻找一对一人物排列，并保留 margin。

### Camera root-ray translation

可只画概念而不画完整公式：

~~~text
camera translation
← average(B0 root ray, clean-raw root ray)
~~~

如果论文方法图空间足够，再写：

~~~text
c' = mean_i (r_i − R' q_i)
~~~

---

## 9. ICLR 主图的概念包装与最简框中文字

### 9.1 包装原则

主图不要像工程流程清单，也不要把 GRU、ORB、EMA、Hungarian、阈值和文件名放在同一视觉层级。ICLR 主图应先传递方法论，再用小字保证技术准确：

~~~text
第一视觉层：论文核心概念
State–Gauge Decoupling
Identity-Conditioned Verification
Observability-Gated Co-Commit

第二视觉层：算法机制
clean reset / read-only shadow
permutation / rays / layout
body consensus / root-ray constraints

第三视觉层：实现名称
V9 / B0 / BRTC-LC / C1-EMA25 / GRU
~~~

推荐将“普通工程名”包装成下列论文图标题。右栏技术名可作为小号副标题，不能完全删掉。

| 工程实现 | 主图高级标题 | 小号技术副标题 |
|---|---|---|
| shot detector | **Causal Boundary Proposal** | RGB-only causal GRU |
| clean reset + shadow | **State–Gauge Decoupling** | committable reset / non-committing shadow |
| V9/B0 | **Learned Read-only Gauge Proposal** | cut-conditioned adapter + reset-to-world SE(3) |
| ID matcher | **Identity-Conditioned Association** | permutation-aware root/torso/body matching |
| BRTC-LC | **Camera-Frozen Structural Verification** | five-joint rays + layout consensus |
| C1-EMA25 | **Motion-Preserving Causal Stabilization** | static-only bounded EMA |
| adaptive joint | **Observability-Gated Camera–Human Co-Commit** | shared body consensus + dual root rays |
| exact fallback | **Precision-First Abstention** | exact parent retention |
| post-shot reuse | **Transactional Gauge Propagation** | one cached correction per shot |

不建议作为主框标题的词：

~~~text
post-processing
heuristic correction
simple EMA
manual threshold
if-else branch
SE(3) fix
human alignment module
~~~

更合适的论文表达：

~~~text
typed geometric verification
observability-aware refinement
confidence-gated transaction
precision-first abstention
camera-human co-commit
causal gauge propagation
~~~

### 9.2 最终主图只保留三个宏观阶段

~~~text
I. Boundary-Aware State–Gauge Decoupling
   clean reset trajectory
   read-only shadow proposal
   learned coarse world bridge

II. Identity-Conditioned Structural Verification
    permutation-aware association
    camera-frozen ray/layout refinement
    motion-preserving causal stabilization

III. Observability-Gated Camera–Human Co-Commit
     shared body consensus + dual root-ray evidence
     accept a consistent camera–scene–human update
     or abstain to the exact trusted parent

→ Atomic Commit & Causal Gauge Propagation
→ Unified camera + scene + persistent humans
~~~

### 9.3 最简主图框中文字

如果最终图空间很小，只保留以下 7 个主框。V9、B0、BRTC-LC、C1 放在副标题，不作为七个并列的大模块。

~~~text
1. Streaming RGB across Shots
   last pre | boundary | first post

2. Causal Boundary Proposal
   RGB-only, first-frame causal

3. State–Gauge Decoupling
   clean state owner | read-only shadow prior

4. Learned Coarse World Bridge
   one gauge for camera + scene + anonymous people

5. Identity-Conditioned Structural Verification
   association + camera-frozen ray/layout refinement

6. Observability-Gated Camera–Human Co-Commit
   shared update OR precision-first abstention

7. Atomic Commit & Causal Gauge Propagation
   unified camera + scene + persistent humans
~~~

---

## 10. 可直接复制给绘图 AI 的中文提示词

### 10.1 ICLR 主图包装版（首选）

> 请设计一张真正具有 ICLR 主方法图气质的横向矢量图，而不是普通工程流程图。画面比例约 2.7:1，白色背景，采用克制、现代、学术化的视觉语言。标题为 “Movie3R: Causal Camera–Human Gauge Transactions across Shot Boundaries”。整张图围绕一个中心思想展开：**camera cut 不是普通的 recurrent update，而是一次需要显式区分 state ownership 与 world-gauge ownership 的因果事务。**
>
> 左侧先用非常简洁的单目 RGB 时间带表示 pre-shot、abrupt boundary 和 post-shot，并用一个小型模块 “Causal Boundary Proposal” 给出事件先验；只在小字注明 RGB-only causal model，不要展开 GRU、ORB、光流和具体统计。随后将主体组织为三个编号清晰、浅色背景分区的宏观阶段。
>
> **Stage I — Boundary-Aware State–Gauge Decoupling。** 在第一帧 post-shot，同一个 Human3R-derived model 形成两条语义完全不同的轨迹：青蓝色 **Clean Reconstruction Trajectory** 从空状态启动，恢复 post camera、dense scene 与 anonymous SMPL-X humans，并以粗实线标记为唯一的 future-state owner；紫色虚线 **Read-only Shadow Prior** 读取 pre-shot context，通过 cut-conditioned V9 adapter 提取跨镜头 camera/body proposal，但其 recurrent state 在分支末端被明确截断并标记 “non-committing”。不要把它画成两个独立的大模型，而要强调 “same model, asymmetric ownership”。两路 camera 在 **Learned Coarse World Bridge** 汇合，以小号公式 T₀ = C_shadow C_reset⁻¹ 表示 B0；用一个共享 gauge 环同时包围 post camera、pointmap 和所有 anonymous people，表达它们被一致带入 pre-shot world 的可比较 basin，而不是已经完成最终对齐。
>
> **Stage II — Identity-Conditioned Structural Verification。** 该阶段使用橙色的 typed geometry，而不是黑盒网络。先画 **Permutation-Aware Association**：pre/post 人体以固定颜色编码，通过 root、torso 与 centered body structure 恢复 persistent cross-shot identity，并保留 assignment confidence。接着画 **Camera-Frozen Ray–Layout Verification**：两台锁定的 camera frustums、五个核心人体关节的交会射线、可靠性筛选与 multi-person layout consensus；视觉上只让人体 root/depth 发生有界修正，相机保持锁定。shot 内稳定不要画成一个笨重的独立大框，只在该阶段底部增加一条细长时间 ribbon，标注 **Motion-Preserving Causal Stabilization**，说明只抑制静止 track 的 residual drift，真实运动与不确定轨迹保持原样。V9/B0、BRTC-LC、C1-EMA25 可作为小号技术标签出现，但不要抢占概念标题。
>
> **Stage III — Observability-Gated Camera–Human Co-Commit。** 这是整张图视觉和论文创新的中心。用一个较大的橙色菱形或验证面板表示 **Is a shared gauge residual observable?**，其证据只概括为三组：cross-person body consensus、dual root-ray constraints、association confidence；在小字中注明 shared Kabsch、normalized fit 与 margin，不要把四个数值阈值写进主图。绿色 **Accept** 分支不是普通后处理，而是一次 camera–scene–human co-commit：人体围绕已经验证的 root 旋转，相机方向与 root-ray translation 同步更新，scene pointmap 跟随同一个 world-gauge change；用一个统一的绿色变换环把 camera、scene 和 humans 连接起来，突出 relative geometry preserved。灰色 **Abstain** 分支写成 **Precision-First Abstention**，直接保留 exact trusted parent，表达不确定时不提交危险更新是方法能力而非失败。
>
> 两个分支最终汇入绿色 **Atomic Transaction Commit**：只接收 clean recurrent state 与 verified geometry；shadow-owned state 用红色截断符号留在 commit 边界之外。右侧用向前延伸的 post-shot 时间轴表示 **Causal Gauge Propagation**：一次 boundary correction 被缓存并复用，pre-shot 完全不被修改，后续不存在 future-frame 回流或 history rewriting。最终渲染一个统一三维世界，包含连续 camera frustums、稀疏或稠密 pointmap、多个固定颜色的 SMPL-X humans、persistent IDs 与一个很小的 diagnostics 图标。
>
> 视觉层级必须清晰：三个宏观阶段是第一视觉层；clean/shadow、association/rays、accept/abstain 是第二层；GRU、V9、B0、BRTC-LC、C1-EMA25、Kabsch 只作为第三层小号技术注释。采用 Human3R/CUT3R 式的在线 camera–scene–human 表达，以及 Multi-THuMBS 式的跨 shot 同色人物与视角变化，但整体更简洁、更强调 transaction、ownership 和 verification。使用粗实线表示 committed trajectory，紫色虚线表示 read-only prior，橙色表示 explicit verification，绿色表示 accepted co-commit，灰色表示 abstention，红色截断表示 never committed。不要使用渐变、发光、复杂神经网络堆叠或软件工程图标。

### 10.2 重要信息与次要信息的视觉优先级

绘图 AI 必须遵守以下优先级：

~~~text
必须最大、最清晰：
1. clean state owner vs read-only shadow
2. proposal → verification → commit / abstain
3. observability-gated camera-human co-commit
4. pre unchanged + causal propagation

中等大小：
1. B0 shared camera-scene-human gauge
2. cross-shot persistent identity
3. camera-frozen ray/layout verification
4. exact fallback

只用小字：
1. GRU / RGB pair features
2. V9 / relation tokens / LoRA
3. five joints
4. Kabsch / Hungarian
5. EMA alpha and thresholds
6. implementation acronyms and numerical gates

不要放进主图：
1. 文件名、checkpoint 路径、JSON payload
2. 训练 loss
3. 具体 benchmark 数字
4. 所有阈值的完整列表
5. 工程运行命令
~~~

### 10.3 技术约束补充（当绘图 AI 容易画错时附加）

> Detector 只提出候选 boundary，不能直接改变 3D state。Clean reset 是 future state 的唯一 owner；V9 shadow 读取旧 state，但 shadow state 永不提交。B0 的同一个 SE(3) 必须同时作用于 post camera、scene 和所有 humans。人物关联发生在 B0 之后，原版 Human3R 的 ID 只能画成 frame-local detection index。BRTC-LC 冻结相机，只修正匹配人体的 root/depth/layout。C1 只稳定静止且可信的 track，不能抹去真实运动。Adaptive accept 时必须一致更新 camera、scene 和 humans，并保持 camera-human relative geometry；reject 时必须 exact 保留 B0+BRTC+C1。所有 pre-shot 输出不变，boundary 决策不读取未来 post 帧，不进行离线 bundle adjustment，也不引入外部 SLAM、Re-ID 或额外大型预训练模型。

### 10.4 极简提示词

> ICLR-style scientific vector figure organized as a three-stage causal transaction: **Boundary-Aware State–Gauge Decoupling** with a committable clean trajectory and a non-committing read-only shadow prior; **Identity-Conditioned Structural Verification** with permutation-aware people, camera-frozen ray/layout refinement, and motion-preserving causal stabilization; **Observability-Gated Camera–Human Co-Commit** that either accepts one consistent camera–scene–human gauge update or performs precision-first exact abstention. Finish with atomic clean-state commit and causal gauge propagation to a unified world containing camera, scene, SMPL-X humans, and persistent identities. Minimal white background, modern ICLR aesthetics, three light conceptual panels, solid committed path, dashed shadow prior, orange verification, green co-commit, gray abstention, no future frames or offline optimization.

---

## 11. English prompt for figure-generation models

### 11.1 Polished ICLR prompt（recommended）

> Design a publication-ready ICLR method figure rather than an engineering flowchart. Use a wide 2.7:1 vector canvas, white background, restrained academic colors, and a left-to-right transactional narrative. Title: “Movie3R: Causal Camera–Human Gauge Transactions across Shot Boundaries.” The intellectual center of the figure is that an abrupt camera cut is not an ordinary recurrent update: it requires explicit separation of **state ownership**, **world-gauge ownership**, and **geometry commit rights**.
>
> Begin with a compact monocular RGB timeline showing the last pre-shot frame, an abrupt boundary, and the first post-shot frame. A small “Causal Boundary Proposal” block produces only an event prior; describe it as RGB-only and first-frame causal, without exposing low-level ORB, flow, or GRU details in the main visual.
>
> Organize the core method into three large, lightly tinted conceptual panels. **I. Boundary-Aware State–Gauge Decoupling:** show two asymmetric roles of the same Human3R-derived model. A cyan “Clean Reconstruction Trajectory” starts from an empty state, reconstructs the post-shot camera, dense scene, and anonymous SMPL-X people, and is the sole owner of all future recurrent state. A purple dashed “Read-only Shadow Prior” accesses a frozen snapshot of the pre-shot context through the cut-conditioned V9 adapter, proposes a cross-shot camera/body prior, and terminates before the commit boundary. Emphasize “same model, asymmetric ownership,” not two independent networks. Merge both camera hypotheses in a “Learned Coarse World Bridge,” with the small equation T₀ = C_shadow C_reset⁻¹. Visualize one shared gauge envelope moving the post camera, pointmap, and all anonymous people into a comparable pre-shot world basin; do not imply final alignment.
>
> **II. Identity-Conditioned Structural Verification:** use orange typed geometry. First recover persistent cross-shot identity through permutation-aware association of color-coded people, using root, torso, and centered body structure with an explicit confidence margin. Then visualize “Camera-Frozen Ray–Layout Verification” using locked camera frustums, five core body-joint rays, robust observability filtering, and multi-person layout consensus. Only human root/depth is refined. Represent within-shot stabilization as a slim temporal ribbon labeled “Motion-Preserving Causal Stabilization,” suppressing residual drift only for confident static tracks while preserving genuine motion. Keep V9/B0, BRTC-LC, C1-EMA25, Hungarian, and Kabsch as small technical subtitles rather than primary headlines.
>
> **III. Observability-Gated Camera–Human Co-Commit:** make this the visual focal point. A verification panel asks, “Is a shared gauge residual observable?” and summarizes three evidence families: cross-person body consensus, dual human root-ray constraints, and association confidence. The green Accept branch performs one consistent camera–scene–human co-commit: bodies rotate about verified roots, camera orientation and root-ray translation are updated coherently, and the scene pointmap follows the same world-gauge change. Enclose camera, scene, and people in one green transformation motif to communicate preserved relative geometry. The gray Reject branch is “Precision-First Abstention,” retaining the exact trusted parent rather than applying an uncertain correction.
>
> Merge both branches in an “Atomic Transaction Commit” that accepts only the clean recurrent state and geometrically verified updates; leave the shadow-owned state visibly outside the commit boundary with a red termination mark. Extend a causal post-shot timeline to the right: cache one correction per boundary, preserve all pre-shot outputs, never rewrite history, and never use future-frame feedback. End with a unified 3D world containing continuous camera frustums, a dense or sparse scene pointmap, consistently color-coded SMPL-X people, persistent cross-shot identities, and a subtle diagnostics icon.
>
> Visual hierarchy: the three conceptual panels are primary; clean-vs-shadow ownership, structural verification, and accept-vs-abstain are secondary; GRU, V9, B0, BRTC-LC, C1-EMA25, thresholds, and solver names are tertiary annotations. Borrow Human3R/CUT3R’s unified online camera–scene–human visual language and Multi-THuMBS’s cross-shot identity coloring, but make the composition more minimal and transaction-centric. Use thick solid lines for committed trajectories, purple dashed lines for read-only priors, orange for typed verification, green for accepted co-commit, gray for abstention, and a red termination symbol for never-committed state. Avoid gradients, glossy effects, software icons, dense neural-network internals, future-frame loops, ground truth, offline bundle adjustment, and external SLAM/Re-ID systems.

### 11.2 Keywords only

~~~text
scientific vector diagram
ICLR / CVPR paper figure
white background
three-stage conceptual hierarchy
camera-cut transaction
state–gauge decoupling
asymmetric state ownership
clean reconstruction trajectory
non-committing shadow prior
learned coarse world bridge
identity-conditioned structural verification
permutation-aware persistent identity
camera-frozen ray–layout consensus
motion-preserving causal stabilization
observability-gated camera–human co-commit
cross-person body consensus
dual human root-ray evidence
precision-first abstention
atomic transaction commit
causal gauge propagation
unified camera–scene–human world
pre-shot invariance
no future-frame feedback
fixed-memory online inference
typed geometric verification
auditable diagnostics
~~~

---

## 12. Negative prompt：禁止画错的内容

将下面内容作为绘图 AI 的 negative prompt：

~~~text
Do not show ground truth entering inference.
Do not show future post-shot frames feeding back to the boundary.
Do not show full-sequence smoothing or offline bundle adjustment.
Do not show DROID-SLAM, VGGT, Grounded-SAM, ViTPose, 4DHumans, or an external Re-ID model inside Movie3R.
Do not show V9 as the final alignment result.
Do not show the shadow recurrent state being propagated.
Do not show BRTC moving the camera.
Do not show C1 smoothing genuinely moving people.
Do not show the pre-shot reconstruction being modified.
Do not show camera-only correction in the accepted low-texture branch.
Do not show human-only translation while leaving a wrong camera unchanged in the accepted joint branch.
Do not imply that Human3R has no ID field; describe it as frame-local, non-persistent detection identity.
Do not claim severe occlusion, new-person entry/exit, or arbitrary long-term Re-ID is solved.
Do not use photorealistic 3D art, glossy gradients, excessive shadows, neon colors, or more than five main colors.
Do not include training-loss arrows in the inference flowchart.
~~~

---

## 13. Mermaid 结构骨架

下面的 Mermaid 不是最终论文图，但可以交给支持 Mermaid 的 AI 作为结构约束。

~~~mermaid
flowchart LR
    RGB["Streaming RGB<br/>pre | CUT | post"] --> DET{"Causal cut proposal<br/>RGB pair features + GRU"}
    DET -- "no cut" --> NORMAL["Normal Human3R stream<br/>no-cut invariance"]
    DET -- "candidate cut" --> RAW["Clean reset<br/>raw camera + scene + humans<br/>future state owner"]
    PRE["Read-only pre-shot state"] -.-> SHADOW["V9 shadow proposal<br/>coarse camera/body"]
    RGB --> SHADOW
    SHADOW -. "shadow state discarded" .-> DISCARD["× never committed"]

    RAW --> B0["B0 coarse gauge<br/>T_B0 = C_shadow C_raw^-1<br/>camera + scene + all humans"]
    SHADOW -.-> B0
    B0 --> ID["Anonymous ID association<br/>root + torso + centered joints<br/>permutation + margin"]
    ID --> BRTC["BRTC-LC<br/>5-joint ray triangulation<br/>camera frozen"]
    BRTC --> C1["C1-EMA25<br/>causal static-human stabilization"]
    C1 --> GATE{"Adaptive geometry gate<br/>Kabsch + RMS + margin<br/>B0/raw root rays"}

    GATE -- "accept" --> JOINT["Shared camera-human update<br/>body about refined root<br/>scene follows camera"]
    GATE -- "reject / abstain" --> FALLBACK["Exact B0+BRTC+C1 fallback"]
    JOINT --> COMMIT["Atomic commit<br/>clean state + verified geometry"]
    FALLBACK --> COMMIT
    RAW --> COMMIT
    COMMIT --> PROP["Causal post-shot propagation<br/>one cached correction"]
    NORMAL --> OUT["Unified world output<br/>camera + scene + SMPL-X + IDs"]
    PROP --> OUT

    classDef learned fill:#EDE9FE,stroke:#7C3AED,color:#2E1065;
    classDef clean fill:#CFFAFE,stroke:#0891B2,color:#164E63;
    classDef geom fill:#FFEDD5,stroke:#EA580C,color:#7C2D12;
    classDef accept fill:#DCFCE7,stroke:#16A34A,color:#14532D;
    classDef fallback fill:#E2E8F0,stroke:#64748B,color:#334155;
    classDef danger fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D;

    class DET,SHADOW learned;
    class RAW,NORMAL clean;
    class B0,ID,BRTC,C1,GATE geom;
    class JOINT,COMMIT,PROP,OUT accept;
    class FALLBACK fallback;
    class DISCARD danger;
~~~

---

## 14. 参考论文与建议借鉴的视觉语言

### 14.1 Human3R

论文：

> Human3R: Everyone Everywhere All at Once, ICLR 2026

本地 PDF：

> /data/wangzheng/iJCV-CODE/paper/Human3R-ori.pdf

Movie3R 与它的关系：

- Human3R 是统一 streaming backbone；
- 同时输出 camera、scene 和 multiple SMPL-X humans；
- Movie3R 不替换这一基础能力，而是解决 camera cut 后的 state/gauge/identity transaction。

建议参考：

- Figure 1 的“RGB stream → unified human-scene-camera output”视觉语言；
- camera frustums + point cloud + colored human meshes；
- 强调 online、continuous、unified，而非多阶段拼装。

不要照搬：

- 不要让图看起来只是 Human3R 后接一个普通 post-processing box；
- Movie3R 的双分支 ownership 与 proposal/verify/commit 必须更突出。

### 14.2 CUT3R

论文：

> Continuous 3D Perception Model with Persistent State

本地 PDF：

> /data/wangzheng/iJCV-CODE/paper/CUT3R.pdf

Movie3R 与它的关系：

- Human3R 的 persistent recurrent world state 来自 CUT3R；
- camera cut 破坏的正是 persistent state 的连续性假设。

建议参考：

- Figure 1 的连续输入、persistent state 与逐帧输出；
- 用 memory/state icon 和横向时间轴表达 streaming。

Movie3R 要新增的视觉对比：

~~~text
ordinary frame: state is read and written normally
cut frame: old state is read-only; clean reset owns future state
~~~

### 14.3 Multi-THuMBS

论文：

> Multi-THuMBS: Multi-person Tracking of 3D Human Meshes Beyond Video Shots

本地 PDF：

> /data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf

它是最直接的外部任务参考：

- multi-person；
- multi-shot；
- global camera/human alignment；
- cross-shot Re-ID；
- world trajectory continuity。

建议参考：

- Figure 1 的 Shot 1 / Shot 2 明确分区；
- 相同人物跨 shot 使用固定颜色；
- camera viewpoint、global alignment 和 propagation 的可视表达；
- before/after 错位对比。

Movie3R 必须视觉区分：

- Multi-THuMBS 是多模块、优化、Re-ID、传播和 smoothing 系统；
- Movie3R 是 first-frame causal transaction；
- Movie3R 不新增 VGGT、Grounded-SAM、ViTPose、4DHumans、DROID-SLAM 等外部预训练模块；
- Movie3R 不画未来帧 smoothing 回流；
- 可在 related-work 对比图中写：

~~~text
Multi-THuMBS: multi-stage + future/global optimization
Movie3R: same-model proposal + causal geometry verification + immediate commit
~~~

### 14.4 HumanMM

论文：

> HumanMM: Global Human Motion Recovery from Multi-shot Videos

本地 PDF：

> /data/wangzheng/iJCV-CODE/paper/HumanMM.pdf

建议参考：

- Figure 1 的 multi-shot 时间带；
- 多个 camera view 切换后仍恢复同一 global human motion；
- shot detector、camera motion 和 global trajectory 的视觉关联。

Movie3R 的区别：

- HumanMM 更偏单人 global motion；
- Movie3R 同时维护 camera、dense scene、multiple humans 与 identity；
- Movie3R 强调 first-post-frame causality 和 exact abstention。

### 14.5 UniCon3R

论文：

> UniCon3R: Unified Contact-aware 4D Human-Scene Reconstruction from Monocular Video

本地 PDF：

> /data/wangzheng/iJCV-CODE/paper/UniCon3R.pdf

与 V9 的方法启发：

- 人—场景关系 cue 不只是 auxiliary output；
- relation cue 可以进入 latent correction / decoder refinement；
- V9 的 relation tokens 是隐式 proposal carrier。

建议参考：

- “base unified stream + relation cue feedback” 的小回路视觉；
- 将 relation tokens 画成轻量提示，不要画成另一个大模型。

Movie3R 的区别：

- 不以 contact 为主要问题；
- relation token 只提出跨 cut coarse gauge；
- 最终 commit 必须经过显式几何 gate。

### 14.6 DUSt3R / TTT3R / ReCal3R / TTSA3R

本地材料：

- /data/wangzheng/iJCV-CODE/paper/TTT3R.pdf
- /data/wangzheng/iJCV-CODE/paper/ReCal3R.pdf
- /data/wangzheng/iJCV-CODE/paper/TTSA3R.pdf

相关性：

- DUSt3R/CUT3R：点图、相机和场景 gauge 基础；
- TTT3R/ReCal3R/TTSA3R：如何更新长期 3D recurrent state。

论文定位中要强调：

> 这些方法研究“怎样更新 state”，Movie3R 研究 camera cut 时“谁有权拥有并提交 state”。

图形上可参考其 state/update 箭头，但 Movie3R 必须画出 read-only shadow 与 clean state owner 的权限差异。

### 14.7 Kabsch 与 Hungarian/Munkres

基础算法：

- Kabsch: shared rigid body rotation；
- Hungarian/Munkres: one-to-one assignment。

图中只需使用小标签：

~~~text
Permutation-aware matching
Shared Kabsch residual
~~~

不要展开 SVD 或完整 assignment matrix，避免方法图过密。

---

## 15. 推荐的论文图风格

### 15.1 风格关键词

~~~text
ICLR scientific figure
CVPR method overview
flat vector graphics
minimal white background
consistent line weight
rounded rectangles
small camera frustums
simple point cloud
color-coded human silhouettes
clear state ownership
proposal-verification-commit
precision-first safety gate
high information density but low visual clutter
~~~

### 15.2 字体

- Arial；
- Helvetica；
- Source Sans 3；
- LaTeX 正文中可保持与论文相近的 sans-serif；
- 模块标题 semibold；
- 公式使用 LaTeX serif；
- 不使用手写、卡通或装饰字体。

### 15.3 线条

- 主线 1.5–2.0 pt；
- 次线 0.8–1.2 pt；
- dashed shadow line 的 dash pattern 保持一致；
- 箭头统一；
- 不用发光、投影、玻璃拟态。

### 15.4 人体与相机

- 人体使用简化 mesh 或关节骨架；
- 同一人物跨 shot 保持同色；
- 不同人物用蓝、橙、绿；
- 原始错配可以淡化或交叉；
- camera frustum 颜色与时段一致；
- raw/shadow/corrected camera 分别用灰、紫、绿。

### 15.5 场景

- 使用简化稀疏点云或浅灰网格；
- 不需要画完整室内 3D 渲染；
- 低纹理 inset 用简单墙面；
- 高纹理 inset 用少量建筑/物体轮廓；
- 背景只用于说明 camera observability，不要喧宾夺主。

---

## 16. 方法图 Caption 草稿

### 中文

> **Movie3R 总体框架。** 因果 RGB detector 只提出候选 camera cut。在第一帧 post-shot，同一个 Human3R 派生模型分别运行可提交的 clean reset 分支和只读 V9 shadow 分支；后者的状态永不进入未来，仅通过 shadow/raw camera 差值提出 B0 粗坐标桥。B0 将 post camera、scene 和所有 anonymous humans 搬入 pre-shot world 的可比较 basin。Movie3R 随后进行 permutation-aware 跨 shot 身份关联、相机冻结的五关节射线/布局人体精修和因果 shot 内静态稳定。只有当 shared body Kabsch、root rays 与置信度门控共同证明残差可观测时，系统才联合更新 post camera、scene 和 humans；否则精确保留 B0+BRTC+C1。最终事务只提交 clean recurrent state 和经过验证的几何，保持 pre-shot 输出不变，并将一次修正因果传播到后续帧。

### English

> **Overview of Movie3R.** A causal RGB detector only proposes a candidate camera cut. At the first post-shot frame, the same Human3R-derived model runs a committable clean-reset branch and a read-only V9 shadow branch. The shadow state is never propagated; the shadow/raw camera difference only proposes a coarse B0 reset-to-world gauge applied consistently to the post camera, scene, and all anonymous people. Movie3R then performs permutation-aware cross-shot association, camera-frozen five-joint ray/layout refinement, and causal within-shot stabilization. A shared camera–human update is committed only when body Kabsch residuals, root rays, and confidence gates make the correction observable; otherwise the system retains B0+BRTC+C1 exactly. The transaction commits only the clean recurrent state and verified geometry, leaves all pre-cut outputs unchanged, and causally reuses one correction throughout the post shot.

---

## 17. 可以放在图角落的四条性质

建议用四个小 icon，而不是长句：

1. **First-frame causal**
   - last pre + first post；
   - no future frames。
2. **State-pure**
   - clean state committed；
   - shadow state discarded。
3. **Camera–human consistent**
   - accepted updates share one gauge transaction。
4. **Safe abstention**
   - uncertain geometry → exact fallback。

系统属性补充：

~~~text
no additional large pretrained model
fixed persistent memory
one extra same-model shadow forward per detected cut
CPU explicit geometry
auditable diagnostics
~~~

---

## 18. 不应在主图中夸大的内容

当前 v15 冻结的是可批量测试的方法主线，不等于 ICLR 大规模结果已经闭环。

主图和 caption 不应声称：

- 已完整解决 severe occlusion；
- 已完整解决 new-person entry/exit；
- 已完成任意长时跨镜头 Re-ID；
- 已官方复现 Multi-THuMBS 全协议；
- adaptive joint 是已训练神经模块；
- 所有数据集上都已超过 SOTA；
- V9 单独完成最终相机精对齐。

正确表述：

~~~text
causal multi-shot human-scene reconstruction
identity-preserving gauge correction
adaptive low-texture camera-human correction
geometry-verified commit
precision-first abstention
~~~

---

## 19. 方法模块与当前实现对应

| 图中模块 | 当前实现/冻结配置 |
|---|---|
| v15 release contract | versions/v15/FINAL_RUNTIME_SPEC.json |
| v15 method summary | versions/v15/ICLR_METHOD_SUMMARY.md |
| causal detector | versions/v14/train_causal_detector.py / causal_image_detector.py |
| V9/B0 export path | versions/v14/export_report_multihuman_comparison.py |
| ID association | v14 exporter + B0 identity audit |
| BRTC-LC | versions/v14/b0_person_triangulation.py |
| C1-EMA25 | versions/v14/eval_streaming_within_shot_stability.py |
| adaptive geometry | src/dust3r/adaptive_joint.py |
| adaptive payload gate | versions/v14/adaptive_post_human_boundary.py |
| runtime wrapper | versions/v15/run_case.py / run_batch.py |
| standard visualization | demo.py-compatible payload |

主要方法依据：

- versions/v15/README.md
- versions/v15/ICLR_METHOD_SUMMARY.md
- versions/v15/FINAL_RUNTIME_SPEC.json
- versions/v14/docs/ADAPTIVE_JOINT_BOUNDARY_FREEZE_20260805.md
- versions/v14/docs/ADAPTIVE_V9_LONG_TASK_FINAL_20260805.md
- versions/v14/docs/V9_ABLATION_AND_ID_MATCHING_AUDIT_20260805.md
- versions/v14/docs/V14_WITHIN_SHOT_C1_STATIC_GATE_V1_20260804.md
- versions/v14/docs/MOVIE3R_ICLR_PAPER_BLUEPRINT_20260802.md

---

## 20. 最终推荐：主图只讲一个故事

主图不要平均强调所有模块。最佳叙事顺序是：

~~~text
Camera cut breaks both recurrent state and world gauge
                    ↓
Clean state owns the future; old state may only propose
                    ↓
V9/B0 brings the new shot into a comparable coarse gauge
                    ↓
Identity and human geometry verify typed residuals
                    ↓
Camera and human are jointly updated only when observable
                    ↓
Otherwise abstain exactly
                    ↓
Commit once and continue online
~~~

最核心的视觉句子：

> **Read old state to propose; commit only clean state and verified geometry.**

中文：

> **旧状态只读地提出建议，未来只继承干净状态和通过验证的几何。**

这句话应当成为整张 Movie3R 方法图的设计中心。
