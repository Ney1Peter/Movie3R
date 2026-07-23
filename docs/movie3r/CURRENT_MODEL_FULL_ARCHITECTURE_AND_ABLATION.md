# Movie3R 当前完整模型、架构与组件消融说明

> 文档日期：2026-07-23
>
> 用途：提供给不了解实验历史的研究人员或 AI，分析当前方法是否合理、创新性在哪里、
> 哪些模块真正有效，以及后续应该保留、简化还是替换哪些组件。
>
> 当前方法定位：**面向 short shot / short horizon 的 camera-cut 后流式重对齐**。
> 它不是无限长度 world mapping，也没有证明 4-8 次以上 camera cuts 后仍保持稳定。

---

## 1. 一句话定义当前方法

当检测到 camera cut 时，在第一张新镜头图像解码前清空 Human3R 的旧 scene/camera
recurrent state；随后从 fresh local reconstruction 出发，用人体 torso 历史修正旋转，
用 DA3 与 2D keypoint 的联合物理尺度 cue 求一个共享 shot scale，再通过上一镜头的
人体 world anchor 显式求一次 translation，最终让新 shot 的 camera、pointmap 和完整
SMPL-X 共同使用一个固定的 shot-level similarity re-anchoring。

默认路径不使用 VGGT，也不启用 V14.2 continuity memory。

---

## 2. 当前方法解决什么问题

### 2.1 Human3R 在普通连续视频中的假设

Human3R 是一个冻结的流式人体-场景重建模型。它在相邻普通帧之间维护 recurrent state，
并默认认为：

- 相机观察连续；
- scene/camera local gauge 连续；
- 人体身份和运动连续；
- 当前帧可以继续读取和更新上一帧 state。

普通视频中这个假设合理，但 camera cut 会瞬间切换视角、相机和局部场景观测。

### 2.2 Camera cut 的两个错误来源

当前实验确认了两个不同问题。

第一，**state contamination**：

- post-cut frame 如果继续读取旧 Human3R state，旧 camera/scene token 会污染新镜头；
- camera 和 pointmap 会错误继承旧镜头结构；
- 污染不只影响第一帧，还会继续写入后续 trajectory。

第二，**world gauge discontinuity**：

- 即使 hard reset 后得到稳定的新 shot-local reconstruction；
- 新 shot 仍处于自己的 local coordinate system；
- 它的 rotation、translation 和 metric scale 未必与旧 world 一致；
- 必须额外估计一个明确的 shot-level transform 才能接回旧世界。

因此当前方法把问题拆成：

```text
state reset
+ shot-local reconstruction
+ explicit shot-level re-anchoring
```

而不是让一个神经网络直接从 raw token 回归任意 SE(3)。

---

## 3. 当前默认架构总览

```text
Streaming RGB + intrinsics + cut trigger
                  |
                  v
        Frozen Human3R recurrent model
                  |
        +---------+--------------------+
        |                              |
   normal frame                    camera cut
        |                              |
 original Human3R path       save external world anchor
        |                    reset state before decode
        |                              |
        |                    fresh post-cut Human3R output
        |                              |
        |                    Fixed Explicit coarse anchor
        |                              |
        |                    V16 torso-motion rotation
        |                         bound = 20 deg
        |                              |
        |           +------------------+------------------+
        |           |                                     |
        |   Keypoint R-CNN                         DA3Metric-Large
        |   2D torso/root pixels                 metric human/background depth
        |           |                                     |
        |           +---------------+---------------------+
        |                           |
        |                 V11.4 fused shot scale s
        |                           |
        |              explicit anchor translation solve
        |                           |
        |               one fixed shot Boundary B
        |                           |
        +---------------------------+
                                    |
                   camera + pointmap + SMPL-X + joints + vertices
                   all transformed into the same predicted world gauge
                                    |
                       optional V14.2 continuity after alignment
```

当前默认实际只有两个主要 alignment correction block：

1. V16 rotation；
2. V11.4 shared scale。

Fixed Explicit 是 coarse initialization/fallback，Keypoint R-CNN 和 DA3 是 V11.4 内部
尺度 cue，最后 translation 由明确方程求解。它们不应该被描述成六个平级网络串联。

---

## 4. 输入定义

### 4.1 推理时输入

每一帧的基础输入为：

- RGB image `I_t`；
- 与 crop/resize 后图像一致的 camera intrinsics `K_t`；
- Human3R recurrent state，仅普通连续帧保留；
- camera cut trigger `c_t`；
- 已经到达的 pre-cut 历史，不读取未来 shot。

当前实验预处理主要使用：

- Human3R resolution：`512 x 288`；
- resize mode：`human3r_demo`；
- DA3 process resolution：`504`；
- 当前主线 `max_humans=1`。

### 4.2 Cut trigger 的真实状态

当前 180-cut、holdout 和 recurrent audit 主要使用 GT cut index 作为**触发信号**。

GT cut index 只告诉系统“这里发生了 cut”，不提供：

- GT camera；
- GT rotation；
- GT translation；
- GT scale；
- GT human；
- source ID。

但自动 cut detector 目前不是已完整验证的主模块。因此论文和后续分析必须区分：

```text
alignment after a known cut        已验证
fully automatic cut detection      尚未作为主结论验证
```

### 4.3 明确禁止进入部署 candidate 的输入

- GT camera pose；
- GT depth 或 GT scene mesh；
- GT SMPL-X；
- source ID；
- camera-pair ID；
- sequence/file name；
- test-set source-specific threshold；
- post-cut future frames；
- Oracle selector 输出。

---

## 5. 输出定义

对每个 frame，最终输出包括：

### 5.1 Camera

- camera-to-world pose `C_t^W`；
- rotation；
- translation/camera center；
- 当前 shot 使用的固定 Boundary transform。

### 5.2 Scene

- Human3R pointmap/depth；
- 经过 shared scale 和 Boundary 后的 world-space pointmap；
- viewer 中使用稀疏采样点云，但底层 reconstruction 不因显示采样改变。

### 5.3 Human

- SMPL-X pose；
- shape/beta；
- expression；
- camera-frame root translation；
- world root；
- joints；
- vertices；
- global orientation。

### 5.4 Shot-level diagnostics

- cut index；
- shared scale `s`；
- Boundary rotation `R_B`；
- Boundary translation `t_B`；
- DA3 root/background scale；
- keypoint confidence；
- V16 correction magnitude；
- fallback/validity状态。

---

## 6. Frozen Human3R 主干

### 6.1 模型

当前使用：

```text
class: ARCroco3DStereo
checkpoint: src/human3r_896L.pth
checkpoint size: approximately 4.67 GB
```

Human3R 可以抽象为一个带 recurrent state 的多头重建模型：

```text
RGB encoding
-> recurrent encoder/decoder state
-> camera head
-> pointmap/depth head
-> human/SMPL-X head
```

### 6.2 当前方法没有修改什么

- 不训练 Human3R encoder；
- 不训练 Human3R decoder；
- 不修改原 camera head；
- 不修改 pointmap head；
- 不修改 SMPL-X head；
- 不增加 raw token SE(3) regressor；
- 不让 decoder “只看人体、不看相机”；
- 不在普通帧插入新的 attention routing。

当前对 Human3R 内部唯一关键操作是：

```text
camera cut -> reset state before decoding first post-cut frame
```

之后的 V16、V11.4 和 translation 都是显式 cut-time geometry post-processing。

### 6.3 Normal frame 路径

无 cut 时完整运行原始 Human3R：

```text
camera max diff   = 0
pointmap max diff = 0
SMPL-X max diff   = 0
```

因此新逻辑不应该影响正常连续视频。

---

## 7. Pre-decode Hard Reset

### 7.1 时序

正确时序是：

```text
detect cut
-> preserve only allowed external history/anchor
-> clear Human3R recurrent state
-> decode first post-cut RGB from fresh state
-> compute one Boundary
-> transform and emit final post-cut output
```

“pre-decode reset”指 reset 必须发生在 post-cut 第一帧进入 recurrent decoder 之前，
不是指 Boundary 必须在看见 post-cut 图像前求出。

### 7.2 Reset 后保留和删除的信息

删除：

- old scene recurrent state；
- old camera recurrent state；
- old shot-local decoder history。

允许在 Human3R 外部保留：

- 上一帧 predicted world root；
- pre-cut torso motion history；
- 上一 shot 已确定的 scale/gauge；
- optional canonical human memory。

这保证 scene/camera 不读取旧 local state，同时允许显式几何模块使用物理上可连续的人体
信息。

---

## 8. Fixed Explicit Coarse Alignment

### 8.1 模块性质

Fixed Explicit 不是训练模型，而是确定性的几何初始化。当前固定候选名称为：

```text
human_mean_pointmap_history_standard
```

### 8.2 输入

- pre-cut Human3R human root poses；
- first post-cut fresh human root pose；
- pre-cut non-human/background pointmap history；
- first post-cut background pointmap。

### 8.3 计算过程

1. 对 pre-cut human root rotations 求 rotation average；
2. 对 pre-cut root translations 求 median；
3. 用该历史人体目标与 post-cut current root 得到 coarse rigid transform；
4. 使用第一张 post-cut background cloud 对 pre-cut history background cloud 做局部稳健
   pointmap refinement；
5. standard refinement 使用固定迭代与距离范围，不读取 GT。

### 8.4 当前职责

- 提供 coarse Boundary rotation；
- 在没有有效 post-cut human 时作为 fallback；
- 给 V16 一个确定、可解释的初始化。

Fixed 的初始 translation 不是最终一定保留的 translation。V16/当前主路径会在 rotation
和 scale 确定后，根据 human world anchor 重新显式求 translation。

### 8.5 证据边界

Fixed 是当前实现的必要起点，但尚未完成：

```text
alternative coarse initializer + same V16/V11.4
```

的 clean replacement ablation。因此不能声称 Fixed 内部每个 pointmap refinement 步骤都
已经被独立证明不可替代。

---

## 9. V16 Torso-Motion Rotation

### 9.1 核心思想

Camera cut 使图像观察不连续，但人体物理运动时间连续。V16 使用 pre-cut torso orientation
history 预测 cut 后时刻的 torso heading，再用 first post-cut fresh torso 反推出 Boundary
rotation residual。

### 9.2 输入

- Fixed Explicit coarse rotation；
- pre-cut Human3R/SMPL-X 3D torso frames；
- first post-cut Human3R/SMPL-X torso frame；
- 固定 `20 deg` correction bound。

V16 不使用 Keypoint R-CNN，也不使用 DA3。

### 9.3 计算

1. 从 pre-cut torso frames 计算相邻 rotation delta；
2. 使用 robust rotvec center/mean 估计 torso angular motion；
3. 向 first post-cut physical time 外推 torso frame；
4. 比较 Fixed 映射后的 current torso 与预测 torso；
5. 求 bounded heading/yaw residual；
6. 在 Fixed rotation 上应用一次 correction；
7. correction 最大为 `20 deg`。

若 post-cut human 无效，则退回 Fixed rotation。

### 9.4 作用边界

- 只负责 rotation；
- 不预测 scale；
- 不直接回归 translation；
- 不修改 Human3R token/state；
- 不逐帧更新 rotation；
- 一个新 shot 只求一次。

### 9.5 消融结果

| 指标 | Fixed | V16 | Delta | Paired p |
|---|---:|---:|---:|---:|
| Camera translation | 0.712 m | 0.518 m | -0.194 m | `2.20e-14` |
| Camera rotation | 24.20 deg | 16.04 deg | -8.17 deg | `3.10e-15` |
| Human joints | 0.290 m | 0.223 m | -0.068 m | `2.90e-10` |
| Scene | 0.483 m | 0.526 m | +0.043 m | `7.96e-11` |

Camera/rotation 分别有 `75.6%/77.2%` 样本改善。V16 是当前后处理模块中独立增益最明确
的一项，但它同样存在 scene trade-off。

---

## 10. Keypoint R-CNN

### 10.1 模型

```text
Torchvision Keypoint R-CNN
backbone: ResNet50-FPN
weights: torchvision default pretrained weights
```

### 10.2 输入输出

输入：cut-time RGB。

输出：

- person detection score；
- person bounding box；
- 17 个 COCO 2D keypoints；
- 每个 keypoint confidence。

当前实现选取 detection score 足够高的目标人物；主线只支持一人。

### 10.3 当前职责

Keypoint R-CNN 不直接预测 Boundary，也不参与 V16。它只用于：

- 定位 pelvis/root pixel；
- 定位 torso joints；
- 告诉 DA3 在人体哪些像素读取 metric depth；
- 构造 human root scale；
- 约束 background scale 是否可信。

### 10.4 已证明和未证明的部分

已证明：human keypoint cue 与 DA3/background gate 联合时，对 V11.4 尺度规则有帮助。

未证明：

- Torchvision Keypoint R-CNN 这一具体架构不可替代；
- keypoint-only 可以独立改善 alignment；
- 它比更轻 detector 更好；
- 它比 Human3R 自身 projected joints 更好。

因此论文中应将它描述为 frozen 2D detector/cue，而不是核心创新模块。

---

## 11. Depth Anything 3

### 11.1 模型

```text
model: DA3Metric-Large
checkpoint:
/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large
checkpoint size: approximately 1.34 GB
```

### 11.2 使用时机

DA3 只在 shot/cut reference frame 上运行，不逐帧运行。

当前真实 recurrent 实现的意图是：

- 每个 shot 开始时估计一次 scale；
- 发生下一个 cut 时，旧 shot scale 已存储；
- first post-cut RGB 估计新 shot scale；
- 后续整个 shot 固定复用。

180-cut 历史 cache 使用独立 pre/post reference frames 做同一尺度估计。两个协议都不读取
完整未来 shot，但 reference frame 选择并不完全相同，后续论文实现应统一说明。

### 11.3 Metric depth 换算

DA3 canonical depth 结合处理后 intrinsics 转为米制深度。当前历史实现使用的核心换算为：

```text
depth_meter = raw_depth * mean(fx_processed, fy_processed) / 300
```

之后使用与 Human3R 一致的 OpenCV camera convention 反投影。

### 11.4 两类尺度 cue

Human cue：

- 在 Keypoint R-CNN torso/root pixel 读取 DA3 metric depth；
- 结合 Human3R root/body offsets，得到 torso-consistent camera-frame root；
- 与 Human3R raw root depth 的比值形成 `s_human`。

Background cue：

- 排除人体 mask；
- 选择 Human3R confidence 足够高的 background pixels；
- 计算 `DA3_depth / Human3R_depth` 的稳健 median ratio；
- 得到 `s_background`。

### 11.5 当前结论

DA3 的单独 background cue 和单独 human-root cue 均未达到显著 camera gain。真正保留的是
DA3 与 Keypoint gate 组成的 V11.4 fused scale estimator。

因此不能声称“DA3 本身就是完整 alignment 方法”。

---

## 12. V11.4 Uniform Shot Similarity

### 12.1 为什么需要 shared scale

早期方法曾只缩放 camera translation 或 human root，导致：

- camera 变好但人体仍在错误位置；
- root 被缩放但 body dimensions 不变；
- 脚地关系破坏；
- 2D projection 移动；
- camera、human 和 pointmap 进入不同 gauge。

V11.4 的关键不是单独预测了一个 scale，而是规定所有 shot-local 三维量共同使用一个
scalar。

### 12.2 当前 fused scale 规则

设：

```text
s_h  = DA3 + keypoint torso/root metric scale
s_bg = DA3/Human3R background median depth ratio
q    = s_bg / s_h
```

基础安全范围：

```text
s_h clipped to [0.35, 3.0]
s_bg_bounded = s_h * clip(q, 0.85, 1.15)
```

最终规则：

```text
if q < 0.95:
    s = s_bg_bounded
else:
    s = s_h
```

如果 background 有效像素不足，则 fallback 到 human-root scale。

该规则在历史实验中称为：

```text
median_ratio_q15_gate_lt95
```

### 12.3 Shared scale 必须作用的量

同一个 `s` 同时作用于：

- post-shot camera relative translation；
- pointmap；
- SMPL-X camera-frame root；
- root-centered body offsets；
- joints；
- vertices。

禁止：

- camera scale 和 human scale 分开；
- scene scale 和 body scale 分开；
- 只缩放 root，不缩放身体；
- 末尾再添加独立 foot/contact translation；
- 同一个 shot 每帧重估 scale。

### 12.4 Projection invariance

对 perspective projection：

```text
pi(K * (s X)) = pi(K * X)
```

因此 root、body offsets、joints 和 vertices 围绕 camera origin 同比缩放时，理论上不应
改变 2D mesh projection。

实测 scale 分支 torso reprojection 均保持约 `19.2 px`，几何检查中 projection
invariance max error 为 `9.35e-6 px`。

### 12.5 当前作用

- 主要改善 camera translation 和 tail；
- 保持 Human3R 原始 2D projection；
- 不改善 final world root anchor；
- 不显著改善 joints；
- scene consistency 略有退化。

它应被称为 camera-oriented effect-first scale block，而不是 camera-human-scene 全面统一。

---

## 13. 显式 Translation 与固定 Boundary

### 13.1 坐标符号

定义：

- `C_0^L`：first post-cut Human3R local camera-to-world pose；
- `s`：V11.4 post-shot scale；
- `R_B`：Fixed + V16 得到的 Boundary rotation；
- `r_0^C`：first post-cut raw camera-frame human root；
- `a_pre^W`：pre-cut 已预测的 human world anchor；
- `C_0^W`：first post-cut target world camera pose；
- `B`：scaled post-shot local gauge 到旧 predicted world gauge 的 Boundary。

### 13.2 Camera rotation

```text
R_C^W = R_B * R_C^L
```

### 13.3 Shared-scale human root

```text
r_scaled^C = s * r_0^C
```

当前默认 V11.4 使用 raw Human3R root 的统一缩放版本，不采用 V14.3 projection-coupled
root 替换。

### 13.4 Translation equation

当前 recurrent 实现使用上一帧/上一 shot 的 predicted world root 作为 anchor。部分
single-cut evaluator 使用冻结的 last-root motion rule；两者都只读取过去。

核心方程是：

```text
c_C^W = a_pre^W - R_C^W * r_scaled^C
```

于是：

```text
C_0^W = [R_C^W, c_C^W]
```

将 local camera translation 同比缩放，记为 `ScalePose(C_0^L, s)`，则：

```text
B = C_0^W * inverse(ScalePose(C_0^L, s))
```

### 13.5 为什么 final root 受 anchor 支配

由 translation equation：

```text
r_world = R_C^W * r_scaled^C + c_C^W
        = a_pre^W
```

因此只要 camera translation 和 final human placement 使用同一个 root，first post-cut final
world root 会代数上闭合到 `a_pre^W`。

这带来一个重要评价结论：

- root cue 会改变 camera translation；
- 但 final root error 主要评价 pre-cut anchor/motion model；
- 不能把 Fixed 到 V11.4 的全部 root gain 归因于 V11.4 scale；
- common-anchor audit 后，不同 coupled root 方法的 final root 都约为 `0.163 m`。

### 13.6 整个 shot 固定复用

Boundary 和 scale 在 shot 内固定：

```text
C_i^W       = B * ScalePose(C_i^L, s)
X_scene_i^W = B * (s * X_scene_i^L)
```

人体：

```text
r_i_scaled = s * r_i
o_i_scaled = s * o_i
J_i^W      = transform(C_i^W, r_i_scaled + o_i_scaled)
V_i^W      = transform(C_i^W, r_i_scaled + v_i_scaled)
```

不逐帧重估 Boundary，不运行 BA，不做全局 trajectory optimization。

---

## 14. Optional V14.2 Human Continuity Memory

### 14.1 默认状态

默认关闭。它不是 alignment module。

### 14.2 Memory 内容

- canonical beta/shape；
- body-scale scalar；
- root-centered local pose；
- torso motion history；
- aligned world root/global orientation，仅 Align-Then-Commit 后保存。

### 14.3 融合强度

```text
shape/scale alpha = 0.25
local pose alpha  = 0.15
```

### 14.4 正确时序

```text
fresh current human reconstruction
-> complete Boundary alignment
-> optionally blend canonical shape/scale/local pose
-> transform world quantities
-> Align-Then-Commit
```

历史 local pose 不进入 camera scale/root solve，canonical memory 不预测 SE(3)。

### 14.5 效果

| Continuity metric | Hard Reset | Memory | Improvement |
|---|---:|---:|---:|
| Shape jump | 0.718 | 0.558 | -22.3% |
| Body-scale jump | 0.00751 | 0.00577 | -23.2% |
| Local-pose residual | 5.37 deg | 4.58 deg | -14.8% |
| 8-cut shape drift | 0.582 | 0.484 | lower |

在统一 alignment 中：

- camera 不变；
- root 不变；
- scene 不变；
- joints 只改善约 `0.00118 m`；
- 肉眼变化整体较小。

所以它可作为轻量 continuity regularizer，但不能作为主要 alignment contribution。

---

## 15. Optional Conditional VGGT

### 15.1 当前状态

默认关闭。真实 recurrent 脚本只有显式传入：

```text
--enable_vggt
```

才加载和运行 VGGT。

### 15.2 如果开启

- 输入 last pre-cut RGB + first post-cut RGB，通常 `1+1`；
- 只产生困难 wide-baseline rotation candidate；
- frozen trigger 决定是否替代 torso rotation；
- 不预测 scale；
- 不预测 translation；
- 不修改 scene gauge。

### 15.3 为什么默认关闭

- 增加约 5 GB checkpoint 和常驻显存；
- 增加 cut latency；
- 存在少量 harmful trigger；
- 当前目标是尽量减少外部模型；
- V16 已提供最主要的稳定 rotation gain。

### 15.4 已知收益

| Dataset/protocol | No VGGT | Conditional VGGT |
|---|---:|---:|
| 180 cuts rotation | 16.04 deg | 12.09 deg |
| Untouched 60 rotation | 17.62 deg | 14.08 deg |
| Post-freeze 419 rotation | 14.91 deg | 13.16 deg |

它是有效的可选 tail rescue，不是当前默认主方法。

当前端口 `8107` 的 viewer 使用的是历史 Conditional-VGGT recurrent cache，界面已明确
标注，不能把该可视化直接当作新默认无 VGGT 结果。

---

## 16. 当前没有进入主路径的历史方法

### 16.1 V11.1，旧 V47

```text
Fixed + V16 + Conditional VGGT rotation, raw scale
```

保留为 wide-rotation 对比。VGGT 关闭后不再是默认主方法。

### 16.2 V11.2，旧 V46

Contact-Preserving Alignment。曾通过额外 contact/root correction 改善视觉接触，但会
修改局部 human-scene geometry，因此保留为诊断，不进入最终路径。

### 16.3 V14.3 Projection-Consistent Coupled Root

该路线解决：

```text
camera uses calibrated root
human uses raw root
```

的不一致。它让 camera 和 final human 使用同一个 calibrated camera-frame root，方程
closure 接近零，2D torso projection 很好。

但统一 evaluator 中 Human Projection Coupled 结果为：

```text
camera 0.730 m
root   0.364 m
scene  0.718 m
reproj 6.6 px
```

也就是说它主要优化了 2D reprojection，却没有匹配当前 Human3R camera/scene world gauge。
因此不作为主方法。

### 16.4 V14.4 Unified Shared Scale + Coupled Root

尝试把 V11.4 shared scale 与 V14.3 coupled root 放入同一方程。正确 Unified 明显优于
Naive Sequential，但仍没有超过 V11.4 raw-root主线。

Naive Sequential 的失败说明不能：

```text
先运行 V11.4
再独立运行一次 V14.3 root correction
```

否则会发生 double scale/root correction。

### 16.5 Raw token / learned SE(3)

已有实验表明 raw image token、state token 和 raw human token 不适合直接预测 Boundary
SE(3)。当前不训练 selector、fusion 或 SE(3) head。

---

## 17. 训练与参数更新

### 17.1 当前 alignment 是否训练

否。当前 alignment 是 training-free/frozen geometry pipeline。

| 部件 | 是否训练 | 当前状态 |
|---|---|---|
| Human3R | 否 | frozen |
| SMPL-X layer/heads | 否 | frozen |
| Keypoint R-CNN | 否 | pretrained frozen |
| DA3Metric-Large | 否 | pretrained frozen |
| Fixed Explicit | 无参数训练 | fixed geometry |
| V16 | 无参数训练 | fixed robust rules + 20 deg bound |
| V11.4 | 无参数训练 | fixed `q15/gate_lt95` rule |
| V14.2 memory | 无训练 | fixed EMA/blending |
| VGGT | 否 | pretrained frozen, default off |

### 17.2 当前可学习部分

没有新增可学习 alignment module。所有阈值来自历史开发协议并冻结。

这意味着当前创新主要是：

- shot-aware state transition；
- 模态连续性职责分离；
- 显式、共享、投影一致的几何约束；
- cut-time cue orchestration；
- one-boundary streaming protocol。

它不是一个新的大型网络 backbone。

---

## 18. 180-Cut 组件必要性消融，VGGT Off

### 18.1 数据

总计 180 个 real cross-camera cuts：

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |

Scene 共同有效子集为 `147/180`。

所有方法共享：

- 相同 pre-shot gauge；
- 相同 V16 rotation；
- VGGT off；
- 相同 raw Human3R root placement；
- 相同 translation equation；
- 相同 evaluator 和有效样本。

### 18.2 比较方法

1. Fixed Explicit；
2. V16 raw scale；
3. DA3 background only；
4. DA3 + Keypoint root scale；
5. Keypoint physical projection only；
6. V11.4 fused scale。

### 18.3 统一结果

| Method | Camera T mean/P90/P95 | Rotation | Root | Joints | Vertices | Scene | Camera success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 0.712/1.509/1.703 | 24.20 | 0.234 | 0.290 | 0.285 | **0.483** | 41.1% |
| V16 raw scale | 0.518/0.934/1.314 | **16.04** | **0.163** | **0.223** | **0.215** | 0.526 | 55.0% |
| DA3 background only | 0.480/0.903/1.110 | 16.04 | 0.163 | 0.225 | 0.219 | 0.545 | 56.1% |
| DA3 + Keypoint root | 0.492/1.027/1.148 | 16.04 | 0.163 | 0.225 | 0.218 | 0.542 | 58.3% |
| Keypoint projection only | 0.504/0.965/1.088 | 16.04 | 0.163 | 0.225 | 0.218 | 0.539 | 57.2% |
| **V11.4 fused scale** | **0.463/0.918/1.088** | 16.04 | 0.163 | 0.225 | 0.218 | 0.536 | **60.6%** |

### 18.4 Single cue 配对统计

相对 V16 raw scale：

| Cue | Camera delta | Improved/Harmed | Camera p | Scene delta | Scene p |
|---|---:|---:|---:|---:|---:|
| DA3 background only | -0.039 m | 52.2% / 38.9% | `0.0684` | +0.018 m | `1.57e-5` |
| DA3 + Keypoint root | -0.027 m | 48.9% / 43.3% | `0.1169` | +0.016 m | `0.0229` |
| Keypoint projection only | -0.014 m | 57.2% / 42.8% | `0.1285` | +0.013 m | `0.0167` |

三个单独 cue 的 camera 检验都没有达到 `p < 0.05`，且 scene 均显著变差。

### 18.5 Fused V11.4 配对统计

相对 V16：

```text
camera 0.518 -> 0.463 m
delta  -0.055 m
p       0.00107
P95     1.314 -> 1.088 m
success 55.0% -> 60.6%
```

同时：

```text
root   unchanged at 0.163 m
joints 0.223 -> 0.225 m, p=0.406, not significant
scene  0.526 -> 0.536 m, p=0.0380, slightly but significantly worse
```

V11.4 相比 DA3+Keypoint root：

```text
camera delta = -0.028 m, p=6.71e-7
scene delta  = -0.0067 m, p=0.0073
```

V11.4 相比 keypoint-only：

```text
camera delta = -0.041 m, p=0.00279
```

结论：有用的是联合门控规则，不是三个单独模块各自都成立。

### 18.6 分数据源

| Source | V16 Camera | V11.4 Camera | Scene V16 -> V11.4 |
|---|---:|---:|---:|
| AvatarReX | 0.226 | 0.209 | 0.605 -> 0.611 |
| MVHuman100 | 0.680 | 0.658 | 0.167 -> 0.180 |
| MVHuman200 | 0.792 | 0.770 | 0.230 -> 0.269 |
| THuman | 0.443 | 0.293 | 0.788 -> 0.780 |

Camera 四源同方向改善，但主要均值收益来自 THuman。Scene 在三个 source 退化，只在
THuman 略改善。

---

## 19. 当前主结果与 Holdout

### 19.1 180-cut，无 VGGT默认路径

| Metric | Fixed | Current V11.4 |
|---|---:|---:|
| Camera translation | 0.712 m | 0.463 m |
| Camera rotation | 24.20 deg | 16.04 deg |
| Human root | 0.234 m | 0.163 m |
| Human joints | 0.290 m | 0.225 m |
| Human vertices | 0.285 m | 0.218 m |
| Scene discontinuity | 0.483 m | 0.536 m |
| Camera success | 41.1% | 60.6% |

Fixed 到 V11.4 的 human root/joint gain 主要来自 V16、anchor 和共同 formulation，不能
全部归因于 V11.4 scale。

### 19.2 Untouched 60-cut capture-disjoint holdout，无 VGGT

| Metric | Fixed | Current V11.4 |
|---|---:|---:|
| Camera translation | 0.663 m | 0.508 m |
| Camera rotation | 23.05 deg | 17.62 deg |
| Human root | 0.234 m | 0.195 m |
| Human joints | 0.291 m | 0.245 m |
| Human vertices | 0.285 m | 0.240 m |
| Scene discontinuity | 0.475 m | 0.547 m |
| Camera success | 41.7% | 60.0% |

Holdout 复现了：

```text
camera/human improvement + scene trade-off
```

而不是 camera、human、scene 全部改善。

### 19.3 如果显式开启 Conditional VGGT

180-cut camera/rotation：

```text
0.403 m / 12.09 deg
```

Untouched 60 camera/rotation：

```text
0.450 m / 14.08 deg
```

这些是可选最高精度结果，不是当前默认无 VGGT 结果。

---

## 20. Scene Trade-off

Scene 是当前方法最重要的负面结果。

原因不是 pointmap 在 Human3R decoder 内被 human memory 污染。Hard Reset 的 fresh
pointmap 本身不变；变化发生在 V16 rotation 和 V11.4 shared world scale 对 pointmap 的
世界放置。

统一审计发现：

- 180-set Fixed scene `0.483 m`；
- V16 scene `0.526 m`；
- V11.4 no-VGGT scene `0.536 m`；
- untouched holdout `0.475 -> 0.547 m`；
- 独立 scene metric 同方向；
- MVHuman200 scene trade-off 最明显。

诊断 Oracle 表明 human-optimal scale 与 scene-optimal scale 可能分离。当前 Human3R local
geometry 可能包含：

- view-dependent depth error；
- spatially varying depth bias；
- human/scene scale mismatch；
- 非单一 scalar 可以解释的 local geometry error。

因此准确表述必须是：

> 当前方法优先改善 camera-human placement，但 scene consistency 存在轻微到中等、统计
> 显著的 trade-off。

不能宣称完整 camera-human-scene closure 已经解决。

---

## 21. Multi-Cut 与适用长度

真实 recurrent rollout 使用前一次 predicted world 作为下一次 anchor，不在每个 cut
恢复 GT gauge。

| Cuts | Camera drift | Rotation drift | Human root drift |
|---:|---:|---:|---:|
| 1 | 0.229 m | 7.81 deg | 0.093 m |
| 2 | 0.326 m | 23.97 deg | 0.094 m |
| 4 | 0.698 m | 37.99 deg | 0.134 m |
| 8 | 0.946 m | 59.03 deg | 0.193 m |

因此：

- 1 cut：主要目标场景；
- 2 cuts：仍属于 short-horizon 可用范围；
- 4 cuts：明显累计漂移；
- 8 cuts：rotation 已不稳定，只能作为压力测试。

当前没有：

- loop closure；
- BA；
- global trajectory optimization；
- map reuse；
- long-term gauge correction；
- drift-aware reinitialization。

准确英文定位：

> Causal, fixed-boundary re-anchoring for short shots and sparse camera cuts,
> not an unlimited-horizon mapping system.

---

## 22. 几何、泄漏与独立评测审计

### 22.1 已通过

- synthetic Sim(3) 恢复到约 `1.8e-15 m`；
- c2w/w2c convention 检查通过；
- camera-origin scaling 检查通过；
- independent evaluator 与主 evaluator 一致；
- camera translation 最大差约 `5.70e-7 m`；
- joints/vertices 最大差低于 `7e-7 m`；
- reprojection 最大差约 `3.62e-6 px`；
- no-cut camera/pointmap/SMPL-X exact no-op；
- candidate 层未发现 GT/source/path leakage；
- scale/Boundary repeated runs deterministic。

### 22.2 Gauge 数值变化解释

旧报告 Fixed camera 曾为 `1.715 m`，统一协议变为 `0.712 m`。这不是算法突然提升，也不
是 GT scale leakage，而是旧评测和统一评测使用了不同 pre-shot metric gauge/指标定义。

所有当前横向结论应使用统一 V14.4/V14.6 evaluator，不能直接引用异构旧数字比较。

### 22.3 尚未完全通过

- 旧 180-case 的所有 DA3/final Boundary cache 未逐数组从 raw RGB 全量重放；
- 发现旧 V10 candidate 对 manifest-order RNG 的 cache-key 风险；
- metadata leakage 主要在缓存 candidate 层测试，不是完整 raw-RGB dynamic taint proof；
- absolute root 排名受 pre-cut anchor confound；
- 真实 8-cut 不满足长期稳定标准。

因此当前方法可以冻结为 short-horizon camera-human-priority 方法，不能冻结为完整长期
camera-human-scene mapping 结论。

---

## 23. 运行速度与显存

V14.5 runtime audit 使用 NVIDIA L20：

| Item | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| No-VGGT cut cue | 0.609 s | 0.407 s | 1.084 s | 1.106 s |
| Triggered VGGT cut cue | 3.615 s | 2.319 s | 6.878 s | 7.171 s |
| Normal Human3R | 3.570 FPS | 3.571 FPS | 3.579 FPS | 3.580 FPS |

No-VGGT cut 加上正常 Human3R 当前帧约：

```text
0.609 + 0.280 ~= 0.889 seconds
```

历史四模型同时常驻：

```text
peak allocated = 12.21 GiB
peak reserved  = 12.94 GiB
```

VGGT 默认关闭后实际常驻需求应更低，但尚未重新做一轮只加载 Human3R + DA3 + Keypoint
的正式 peak-memory benchmark。

当前满足 causal streaming 和显存约束，不满足 25/30 FPS 实时视频要求。

---

## 24. 当前模块保留决策

| Module | Default | Evidence | Final role |
|---|---|---|---|
| Human3R | On | 基础模型 | frozen per-frame reconstruction |
| Pre-decode reset | On at cut | 强因果证据 | 清除旧 scene/camera state |
| Fixed Explicit | On | 稳定 coarse/fallback；替代消融未完成 | coarse anchor |
| V16 torso | On | 最强独立 paired gain | rotation correction |
| DA3 | On inside V11.4 | 单独不显著，联合有效 | metric depth cue |
| Keypoint R-CNN | On inside V11.4 | 特定模型必要性未证明 | DA3 human-depth sampling/gate |
| V11.4 shared scale | On | camera显著改善，scene略退化 | effect-first shot scale |
| Explicit translation | On | 统一方程/closure | old anchor re-anchoring |
| V14.2 continuity | Off | continuity有效，alignment无增益 | optional smoothing |
| VGGT | Off | tail有效但昂贵、有风险 | optional rotation rescue |
| V14.3 coupled root | Off | projection好，world metrics差 | diagnostic only |
| BA/global optimizer | Absent | 不符合当前严格流式设计 | future long-horizon work |

---

## 25. 推荐的两个实际配置

### 25.1 Effect-first 当前默认

```text
Human3R
+ pre-decode hard reset
+ Fixed Explicit
+ V16 torso rotation
+ DA3 + Keypoint fused V11.4 shared scale
+ explicit translation
+ one fixed shot Boundary
```

适合：

- 追求当前最好 camera translation；
- single cut / 1-2 cuts；
- 可接受 cut-time 外部 depth/detector；
- 接受 scene 约厘米级到数厘米级 trade-off。

### 25.2 Minimal / simplicity-first

```text
Human3R
+ pre-decode hard reset
+ Fixed Explicit
+ V16 torso rotation
+ raw scale
+ explicit translation
```

指标变化：

```text
camera 0.463 -> 0.518 m
scene  0.536 -> 0.526 m
```

优点：

- 不需要 DA3；
- 不需要 Keypoint R-CNN；
- 架构更简洁；
- cut latency 和显存更低；
- 研究贡献更集中在 state reset + torso geometry。

如果论文更重视创新纯度、方法简洁性和部署成本，这个版本值得作为正式强基线或候选
主线，而不是默认把所有外部模型都保留。

---

## 26. 创新性应该如何表述

### 26.1 可以主张

- camera cut 前后 scene/camera 与 human 具有不同状态连续性；
- cut 后必须 pre-decode reset scene/camera state；
- human physical motion 可以跨 shot 辅助 bounded rotation；
- 新 shot 应只求一次 shared similarity/Boundary；
- camera、pointmap、root、body offsets、joints、vertices 必须共享同一个 scale/gauge；
- world memory 必须 Align-Then-Commit；
- projection consistency 不等于完整 world consistency；
- short-shot streaming re-anchoring 的完整因果和几何审计。

### 26.2 不应主张

- 发明 Human3R、DA3、Keypoint R-CNN 或 VGGT；
- DA3 或 Keypoint 单独构成新的 alignment module；
- camera、human、scene 三者全部提高；
- V11.4 改善了 absolute human root；
- 自动 cut detection 已解决；
- 8-cut/无限长度稳定；
- 新训练的 modality-selective decoder 已部署；
- raw human token 可以回归 SE(3)；
- 2D reprojection 更好就代表 3D world alignment 更好。

### 26.3 当前最有价值的论文问题

提供给其他 AI 分析时，可以重点要求其判断：

1. `Hard Reset + V16 + V11.4` 是统一方法，还是多个冻结模型的工程组合？
2. V11.4 约 `5.5 cm` camera gain 是否值得 DA3 + Keypoint 的复杂度？
3. Scene trade-off 是否会削弱“统一 similarity”的核心论点？
4. 是否应把 minimal `Hard Reset + Fixed + V16` 作为更有创新性的主线？
5. Fixed Explicit 是否需要替代 initializer clean ablation？
6. Keypoint R-CNN 能否由 Human3R 自身 joints 或轻量 detector 替代？
7. 当前 single scalar 不足时，是否应该停止追求 camera-human-scene 统一，而明确定位为
   camera-human-priority？
8. 长期方法是否应该增加 loop closure/BA，而不是继续堆叠 cut-time cue？

---

## 27. 版本编号映射

| 历史名称 | 当前名称 | 当前定位 |
|---|---|---|
| Fixed Explicit | V10.1 | coarse baseline/fallback |
| 旧 V47 | V11.1 Conditional Wide Rotation | VGGT tail 对比，非默认 |
| 旧 V46 | V11.2 Contact-Preserving | diagnostic |
| V48 | V11.3 Component Ablation | historical ablation |
| V53 | V11.4 Uniform Similarity | 当前 effect-first scale block |
| V14.1 | Shot-Aware State Routing | reset/memory职责验证 |
| V14.2 | Canonical Human Memory | optional continuity |
| V14.3 | Projection-Consistent Coupled Root | diagnostic |
| V14.4 | Unified Similarity Re-anchoring | shared-scale/coupled-root统一比较 |
| V14.5 | Final Geometry/Leakage/Streaming Audit | 最终严格审计 |
| V14.6 | Alignment Component Necessity Audit | 最新无 VGGT 组件消融 |

---

## 28. 关键代码与结果路径

### 28.1 当前模块

```text
scripts/v10_1_fixed_explicit_candidate_probe.py
scripts/v11_4_uniform_similarity_probe.py
scripts/v14_4_unified_similarity_reanchoring_probe.py
scripts/v14_5_true_recurrent_multicut_audit.py
scripts/v14_5_multicut_interactive_viewer.py
scripts/boundary_shot_scale_support.py
scripts/boundary_metric_depth_support.py
```

V16 与部分历史 DA3 implementation 已归档：

```text
archive/20260721/scripts/v16_human_torso_candidates.py
archive/20260721/scripts/v21_absolute_shot_background_scale_probe.py
```

### 28.2 主要报告

```text
docs/movie3r/V11_4_SHORT_SHOT_METHOD_FREEZE.md
docs/movie3r/V14_2_CANONICAL_HUMAN_MEMORY.md
docs/movie3r/V14_4_UNIFIED_SIMILARITY_REANCHORING.md
docs/movie3r/V14_5_FINAL_GEOMETRY_STREAMING_AUDIT.md
docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md
```

### 28.3 最新组件消融 JSON

```text
output/v14_6_alignment_component_necessity/full180_no_vggt/
  v14_4_unified_similarity_reanchoring.json
```

### 28.4 复现实验命令

```bash
TMPDIR=output/v14_5_final_audit/tmp \
.venv/bin/python scripts/v14_4_unified_similarity_reanchoring_probe.py \
  --device cuda:6 \
  --output_dir output/v14_6_alignment_component_necessity/full180_no_vggt
```

真实 recurrent 默认 VGGT off：

```bash
TMPDIR=output/v14_5_final_audit/tmp \
.venv/bin/python scripts/v14_5_true_recurrent_multicut_audit.py \
  --device cuda:5
```

只有需要 VGGT tail rescue 时显式添加：

```text
--enable_vggt
```

### 28.5 Viewer

```text
http://127.0.0.1:8107
```

Viewer 默认稀疏显示：

```text
point_stride = 32
confidence > 1.5
```

该采样只影响浏览器流畅度，不影响底层 Human3R pointmap 或评价指标。

---

## 29. 最终总结

当前方法不是“所有模块都有效”的大流水线。严格证据支持的结构是：

```text
核心状态操作：pre-decode Hard Reset
核心 rotation：Fixed initialization + V16 torso correction
有效但较窄的 scale：V11.4 fused DA3/Keypoint shared scale
核心 world placement：one explicit translation + one fixed Boundary
可选 continuity：V14.2, default off
可选 tail rescue：VGGT, default off
```

它最明确的收益是 short-shot camera-human re-anchoring。它的主要未解决问题是 scene
trade-off、anchor confound、外部模型复杂度、自动 cut detection 和 long-horizon drift。

如果目标是当前最好效果，保留 V11.4 fused scale；如果目标是更简洁、更有方法创新性、
更容易部署的论文主线，则 `Hard Reset + Fixed + V16` 是必须认真比较的简化候选。
