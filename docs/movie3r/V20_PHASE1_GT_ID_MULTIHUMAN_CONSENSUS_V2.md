# V20 Phase 1 v2: Strict GT-ID Multi-Human Consensus Re-evaluation

| 项目 | 当前定义 |
|---|---|
| 版本名称 | V20 Phase 1 v2 Strict GT-ID Multi-Human Geometry |
| 实验性质 | 多人 shared-Boundary 几何可行性验证，不是最终部署版 |
| 重建骨干 | Frozen Human3R `human3r_896L.pth` |
| 输入 | 5 帧 pre-shot 全图 + 1 帧 post-shot 全图，2048 缩放至 512，不裁剪 |
| Cut 处理 | post 第一帧 decode 前 fresh hard reset |
| 身份 | GT SMPL-X mesh-projection Oracle，仅用于 WHO |
| 单人候选 | Fixed Explicit + V16 torso 20 deg + explicit root translation |
| 当前最佳多人融合 | SO(3) mean of `R_i` + arithmetic mean of raw `t_i` |
| Shot scale | `s=1`，不使用 V11.4/DA3 |
| 最终输出 | camera、pointmap、全部 SMPL-X 共用一个 `B=[R,t]` |
| 可部署性 | 身份 Oracle 不可部署；geometry candidate 本身不读取 GT root/camera Boundary |

## 1. Executive conclusion

The original V20 Phase 1 report is invalid as a strict GT-ID experiment. Its per-frame identity assignment used only projected bounding-box IoU and center distance. Overlapping people caused frequent `person0/person1` swaps across the cut, and those swaps were then incorrectly interpreted as 150-180 degree human-geometry failures.

V2 replaces the identity oracle with corresponding-vertex SMPL-X projection matching and reruns all 315 cached cuts. The revised conclusions are:

1. Correct identity association removes most catastrophic multi-human failures.
2. Multi-human geometry is useful relative to every deployable single-human selector tested.
3. The best current fusion is the simple mean of all valid human candidates.
4. The current Huber/layout/reject logic is worse than the simple mean and is not ready for use.
5. Multi-human mean does not beat the GT-evaluated Oracle Best Single on camera composite, so the original strict Phase 1 gate remains FAIL.
6. The result supports continued work on consensus and anchor reliability, but does not yet justify freezing the current robust consensus or starting a learned token adapter.

In short:

> Correct IDs show that multiple humans provide useful redundant geometry. The old negative result was largely contaminated by ID swaps. However, the current robust selector is still the bottleneck, and multi-human fusion has not exceeded the unavailable per-case best-single oracle.

## 2. Experiment scope

Dataset:

- MultiHuman Real-World-Capture `three` sequence;
- 3 stable GT identities: `person0`, `person1`, `person2`;
- 6 calibrated synchronized cameras;
- timestamps: 500, 700, 900, 1000, 1100, 1300, 1500;
- camera pairs: `0->1`, `1->2`, `2->3`, `3->4`, `4->5`, `5->0`, `0->3`, `1->4`, `2->5`;
- temporal offsets: 0, 1, 2, 4, 8 frames;
- total: 315 cuts.

Frozen Lite pipeline:

```text
full 2048x2048 RGB, resized to 512x512 without cropping
-> Human3R multi-human inference
-> pre-decode hard reset at the cut
-> Fixed Explicit coarse alignment
-> V16 torso residual, 20 degree bound
-> explicit root-based translation
-> one shared shot Boundary
```

Disabled:

- DA3;
- Keypoint R-CNN;
- V11.4 shared scale;
- VGGT;
- continuity memory;
- scene refinement;
- token Re-ID.

The existing 2.1 GB Human3R cache was reused. Human3R inference was not rerun, so the only changed variable is identity assignment and the geometry evaluation derived from it.

## 3. 当前版本的完整方法架构

### 3.1 这一版到底是什么

这一版是 V20 Phase 1 的 **Lite、GT-ID Oracle、多人几何可行性验证版**。它不是完整可部署的多人 Movie3R，因为跨镜头身份由 GT 提供；但除“谁是谁”之外，Boundary 的 rotation、translation 和最终 camera/human 位置全部由预测量显式计算。

完整数据流为：

```text
cut 前 source camera 的连续 5 帧 RGB
                    |
                    v
          Frozen Human3R recurrent inference
                    |
        camera / pointmap / all SMPL-X humans
                    |
      保存每个 GT identity 的 root / torso history
                    |
                    | camera cut trigger
                    v
      在 post-cut 第一帧 decode 前 hard reset
                    |
                    v
        target camera 第一帧 fresh Human3R inference
                    |
        camera / pointmap / all SMPL-X detections
                    |
                    v
  GT SMPL-X projection Oracle association（仅回答 WHO）
                    |
                    v
       每个 matched human 独立生成 (R_i, t_i)
       - root-motion anchor
       - Fixed Explicit pointmap refinement
       - V16 torso yaw residual
       - explicit translation equation
                    |
                    v
      Multi-human shared-Boundary consensus
                    |
                    v
           ONE B = [R, t] for the whole shot
                    |
       +------------+-------------+
       |            |             |
       v            v             v
   camera pose   pointmap      all SMPL-X humans
```

核心职责分离是：

```text
GT identity Oracle answers WHO（仅本阶段）
predicted human geometry answers WHERE
all matched humans vote for ONE shared Boundary
```

### 3.2 使用的模型与权重

主模型是冻结的 Human3R：

```text
checkpoint: src/human3r_896L.pth
inference:  forward_recurrent_lighter(...)
training:   none
fine-tune:  none
TTT:        disabled
```

本实验没有训练任何新增网络，也没有修改 Human3R 权重。模型结构在本实验中的职责可以概括为：

```text
RGB image
   |
   v
Human3R image encoder + recurrent scene/camera decoder
   |                                      |
   |                                      v
   |                              multi-human head
   |                                      |
   v                                      v
camera pose + dense pointmap      human detections + SMPL-X
```

Human3R 为每一帧输出：

| 输出 | 当前实验中的作用 |
|---|---|
| camera pose / `camera_matrix` | 定义 Human3R 当前 shot-local c2w gauge，并在最后应用 Boundary |
| `pts3d_in_self_view` | 生成稀疏背景 point cloud，供 Fixed Explicit coarse refinement 使用 |
| `conf_self` | 在每个图像网格中选择高置信度背景点 |
| head detections / head score | 决定当前检测到多少人，并构造人物质量分数 |
| `smpl_rotmat` | 得到 root orientation、body pose 和 torso frame |
| `smpl_shape` | 经过 SMPL-X layer 生成当前人物 mesh；不用于 Boundary scale |
| `smpl_transl` | 得到 camera-frame human placement 和 root |
| `smpl_expression` | 生成完整 SMPL-X 输出；不参与 Boundary 求解 |
| SMPL-X joints | 构造 root、torso frame 和人体评价指标 |
| SMPL-X 10475 vertices | 可视化、GT-ID projection Oracle 和 vertex error |

Human3R 的原始 detection index `D0/D1/D2` 只是当前帧输出顺序，不是跨帧或跨 shot 的可靠人物 ID。V2 不再假设数组顺序等于身份。

`return_token_debug=True` 在代码中用于取得 head score 等调试量，但本阶段 **没有使用 refined human token、head token 或 Multi-HMR token 做 Re-ID，也没有让 token 预测 SE(3)**。

### 3.3 当前启用与关闭的模块

| 模块 | 状态 | 作用 |
|---|---:|---|
| Frozen Human3R | 开启 | 一次前馈输出 camera、pointmap 和全部人物 SMPL-X |
| Pre-decode hard reset | 开启 | camera cut 后清除旧 shot 的 scene/camera recurrent state |
| GT mesh-projection identity Oracle | 开启，仅实验 | 严格确定 pre/post 的人物对应关系 |
| Fixed Explicit | 开启 | 以人体初值启动，用稀疏背景 pointmap 做小范围 coarse refinement |
| V16 torso-motion rotation | 开启 | 用 torso heading 对 coarse rotation 做最多 20 度的 yaw 修正 |
| Explicit root translation | 开启 | 用 `t_i = a_i - R_i r_i` 得到每个人的 translation candidate |
| Multi-human consensus | 开启 | 将所有可靠人物候选合成一个 shot Boundary |
| DA3 | 关闭 | 不提供 metric depth 或 shot scale |
| Keypoint R-CNN | 关闭 | 不提供外部 2D keypoint cue |
| V11.4 shared scale | 关闭 | 本实验固定 `s=1`，不研究尺度修正 |
| VGGT | 关闭 | 不提供 rotation tail rescue |
| V14.2 continuity | 关闭 | 不平滑 shape、scale 或 local pose |
| Human token Re-ID | 关闭 | 不研究可部署身份匹配 |
| learned adapter | 关闭 | 不训练身份或几何网络 |
| BA / global optimization | 关闭 | 保持一次 cut、固定预算的流式设置 |

因此，本实验隔离的问题是：

> 当 Human3R 预测、scale、外部模型和 rotation branch 全部固定时，把一个人体 anchor 改成多个已正确匹配的人体 anchor，是否能产生更稳定的同一个 Boundary？

## 4. 数据集、输入与 cut 构造

### 4.1 数据目录与内容

数据根目录：

```text
/data/wangzheng/iJCV-CODE/data/MultiHuman/
  Real-World-Capture/extracted/
```

本实验使用动态 `three` 序列：

```text
three_original_video/
  calibration_new.json       # 6 个相机的内参
  three_new/                  # 6 路同步 2048x2048 RGB 视频

three/three/
  person0/
  person1/
  person2/
    parameter/<frame>/        # 每个相机的逐帧 extrinsic
    smplx/<frame>/smplx.obj   # 10475 顶点的逐帧 GT SMPL-X
```

该序列适合本阶段的原因：

- 三个人在同一个空间内持续运动和交互；
- 六个同步相机可以构造只改变视角的 `k=0` camera cut；
- 也可以构造带人体运动的 `k=1/2/4/8` temporal cut；
- `person0/person1/person2` 文件夹提供稳定 GT identity；
- 每个人每一帧都有与 Human3R 同拓扑的标准 SMPL-X mesh；
- 有明显遮挡、人体接近和 bbox 重叠，能暴露身份和 consensus 问题。

局限性：

- 没有 GT dense scene depth、pointmap 或 scene mesh；
- 只能严格评价 camera 和 human，不能将当前 pointmap residual 宣称为 GT scene accuracy；
- 目前只使用一个三人 capture，属于方法调试集，不是最终论文 benchmark；
- GT identity 和 GT mesh 使本阶段是 Oracle geometry study，不是部署测试。

### 4.2 图像预处理

每张原图为 `2048 x 2048`。输入规则是：

```text
完整原图
-> 等比例缩放到 512 x 512
-> 不裁剪
-> 不做人物 crop
-> 不单独放大某个人
```

因此 Human3R 同时看到：

- 全部人物；
- 人物间相对布局；
- 地面和背景；
- 完整相机视野。

需要区分两套 intrinsics 用途：

- Human3R inference 使用模型内部 `get_camera_parameters(mhmr_img_res)` 生成的 `K_mhmr`，不读取 GT calibration 生成 camera/pointmap/SMPL-X candidate；
- dataset `calibration_new.json` 中的真实 K 按 `2048 -> 512` 缩放，只用于 GT mesh projection identity Oracle、GT evaluation 和 viewer。

二者遵循相同的完整画面 resize convention，但真实 GT K 不进入 Boundary candidate generation。

### 4.3 单个实验样本的输入

对时间点 `t` 和 camera pair `A -> B`：

```text
pre-shot:  camera A frames [t-4, t-3, t-2, t-1, t]
post-shot: camera B frame  [t+k]
```

其中：

- `k=0`：同步 camera cut，只改变相机视角；
- `k=1/2/4/8`：相机改变，同时允许人物继续运动；
- 每个样本共输入 6 帧；
- 前 5 帧建立 Human3R 预测历史；
- 第 6 帧是 hard reset 后的 fresh post-cut frame。

315 cuts 的组成：

```text
7 timestamps
x 9 camera pairs
x 5 temporal offsets
= 315 cuts
```

## 5. 严格流式 reset 与状态生命周期

### 5.1 Cut 前

前 5 帧在 source camera 中按顺序运行 Human3R recurrent inference。保存的是预测历史：

- camera pose；
- background point cloud；
- 每个已知 identity 的 world root；
- torso frame；
- root orientation；
- detection score、bbox 和 completeness。

这些历史都处于 Human3R 自己的 pre-shot predicted world gauge，而不是 GT world。

### 5.2 Cut 时

代码使用：

```text
shot_routing = {
    enabled: true,
    mode: fresh,
    cut_indices: [5],
    consume_view_reset_at_cut: true
}
```

也就是说，第 6 帧在进入 decoder 前触发 fresh reset。post-cut camera、pointmap 和人体 reconstruction 不能读取旧 shot 的 scene/camera recurrent state。

这一步避免 Human3R 把完全不同相机的画面当成连续运动帧，但也导致 post-cut 输出落入一个新的 shot-local gauge，所以必须重新求 Boundary。

### 5.3 本阶段的身份状态

当前没有部署级 human tracklet memory：

- scene/camera recurrent state：cut 时 reset；
- Human3R 原生 detection index：不跨 cut 信任；
- human token memory：不使用；
- GT identity table：作为 evaluator 外部 Oracle，跨 cut 提供正确 ID；
- long-term Align-Then-Commit：Phase 1 尚未集成。

因此它验证的是“正确 ID 已知时，多人几何是否有价值”，不是“系统是否已经自动找对人”。未来可部署版本必须用 token/appearance/geometry matching 替代 GT identity Oracle。

## 6. Human3R 输出到几何量的转换

### 6.1 每个人的预测量

Human3R 的 SMPL-X 参数经过冻结的 SMPL-X layer 后，得到 camera-frame joints 和 vertices。再通过 Human3R 预测 camera pose 变换到当前 predicted world：

```text
x_world = C_pred * x_camera
```

对每个人保存：

- `root`：SMPL-X root joint 的 predicted-world 位置；
- `root_rotation`：SMPL-X global/root orientation 乘 predicted camera rotation；
- `torso`：由肩、髋等关节构造的正交 torso frame，再乘 predicted camera rotation；
- `joints`、`vertices`：完整 predicted-world 人体；
- `bbox`：predicted mesh 的图像投影框；
- `score`：Human3R head detection score；
- `completeness`：mesh vertices 落在图像内且深度有效的比例。

需要注意：Human3R 的局部 mesh 可以较准确，但 camera-frame root depth 仍可能有几十厘米误差。Phase 1 Lite 不用 DA3、V11.4 或 coupled root 修正这部分误差，它只研究多个 raw human anchors 的冗余是否有帮助。

### 6.2 稀疏背景 point cloud

Human3R dense pointmap 不直接全部送入 coarse refinement。为了速度和减少人体污染：

1. 只保留有限、深度在 `0.05-50 m` 的点；
2. 删除所有人体 bbox，并在 bbox 外扩 8% margin；
3. 将图像划分为 `24 x 24` 网格；
4. 每个网格最多选择一个最高 `conf_self` 的点；
5. 总数最多保留 1024 点；
6. 使用 predicted camera pose 将点转到各自 shot-local world。

cut 前 5 帧的稀疏背景点合并为 target cloud，post-cut 第一帧作为 source cloud。这个 point cloud 只参与 Fixed Explicit coarse refinement，并不会单独改变人体，也不会在最终阶段对背景做 BA。

## 7. 每个人如何产生一个 Boundary candidate

对 cut 前后均有预测且 GT-ID 匹配成功的人物 `i`，生成一个候选：

```text
B_i = [R_i, t_i]
```

### 7.1 Root motion anchor

从 cut 前最多 5 帧 root 计算逐帧速度。实现先用 median 找稳健中心，再剔除偏离过大的速度，得到：

```text
v_i = robust_velocity(root_i history)
```

外推到 post frame：

```text
a_i = root_i(last pre frame) + delta_frame * v_i
```

`a_i` 是“人物 i 在旧 predicted world 中，此刻应该在哪里”的 anchor。`k=0` 时 `delta_frame=0`，主要排除人体运动影响；`k>0` 时进行短时线性外推。

### 7.2 Torso motion prediction

对相邻历史 torso frames 计算 SO(3) relative rotation 的 rotvec velocity，取 median angular velocity，再外推：

```text
T_i_target = Exp(delta_frame * median_angular_velocity) * T_i_last
```

这为 V16 提供 post 时刻的目标 torso orientation，而不是假设人体完全静止。

### 7.3 Fixed Explicit initial transform

最近 3 个历史 root orientations 做 SO(3) mean，得到目标 root orientation：

```text
Q_i_target = SO3Mean(last three pre root rotations)
```

post-cut fresh root orientation 为 `Q_i_post`，初始 rotation：

```text
R_i_initial = Q_i_target * inverse(Q_i_post)
```

初始 translation：

```text
t_i_initial = a_i - R_i_initial * r_i_post
```

该人体初值启动一次 background pointmap local refinement：

- 8 iterations；
- correspondence bound 从约 `0.60 m` 收紧到 `0.12 m`；
- source 或 target 少于 32 点时跳过；
- 不访问未来帧；
- 不做 BA；
- 不改变 scale。

因为不同人物给出的 initial transform 不同，Fixed pointmap refinement 可能收敛到不同的 coarse branch。这也是随后需要多人融合的原因之一。

### 7.4 V16 torso yaw refinement

Fixed coarse rotation 得到后，用当前 post torso heading 和预测 target torso heading 计算 residual。只围绕 target torso up axis 修正 heading：

```text
theta_i = signed_heading_residual(
    R_i_fixed * post_torso_heading,
    target_torso_heading
)

theta_i_bounded = clip(theta_i, -20 deg, +20 deg)
R_i = Rot(target_up, theta_i_bounded) * R_i_fixed
```

V16 只负责 bounded torso-motion rotation correction，不估计 scale，也不直接修改人体 pose。

### 7.5 Explicit translation

rotation 固定后重新显式求 translation：

```text
t_i = a_i - R_i * r_i_post
```

它确保在人物 `i` 的候选 Boundary 下：

```text
R_i * r_i_post + t_i = a_i
```

每个人只产生候选，不允许拥有独立的最终 world transform。

### 7.6 人物质量分数

用于 confidence-weighted/robust 消融的质量分数为：

```text
q_i = sqrt(mean(pre_head_score) * post_head_score)
      * post_completeness
      * exp(-min(motion_dispersion / 0.10, 4))
```

它综合：

- cut 前检测置信度；
- cut 后检测置信度；
- 当前 mesh 可见完整度；
- 历史 root motion 是否稳定。

当前最佳 `naive_mean` 不使用该质量权重，这说明现有 quality calibration 还不够可靠。

## 8. 多人如何合成 ONE shared Boundary

### 8.1 当前效果最好的方法：Naive Mean

对所有 matched humans 的 rotation candidates 做 SO(3) mean：

```text
R = SO3Mean(R_1, R_2, ..., R_N)
```

translation 使用各人物已经显式求出的 raw candidates 的算术平均：

```text
t = Mean(t_1, t_2, ..., t_N)
```

最终：

```text
B = [R, t]
```

这里不是对欧拉角逐分量平均，rotation mean 在 SO(3) 上完成。translation 平均的是每个人自己的 `R_i` 下得到的 `t_i`，对应代码中的 `mean_raw_t`。

该方法简单但当前最稳定，因为正确 ID 后，三个人的大多数候选位于同一个 rotation/translation 邻域，平均可以抵消单个人的 root、torso 和 pointmap 噪声。

### 8.2 其他被比较的方法

| 方法 | Rotation | Translation | V2 结论 |
|---|---|---|---|
| `shared_rotation_mean` | SO(3) mean | 在共同 R 下重新计算所有 `a_i-Rr_i` 后平均 | 不如 raw candidate mean |
| `confidence_weighted` | 质量加权 SO(3) mean | 质量加权 mean | quality 不能稳定识别好人物 |
| rotation geomedian + median | SO(3) geometric median | coordinate median | 没有超过 naive mean |
| rotation geomedian + geomedian | SO(3) geometric median | geometric median | 没有超过 naive mean |
| trimmed mean | robust SO(3) center | coordinate trimmed mean | 三人时容易丢掉有效信息 |
| Huber | 10 deg Huber SO(3) center | 0.25 m Huber mean | 当前为 0.734 composite，弱于 0.657 |
| layout select | 枚举某个人的 `R_i` | translation geometric median | selector 命中率不足 |
| layout + one reject | layout select 后最多删除一人 | 重新求一次 | 当前 0.805，明显弱于 naive mean |

### 8.3 Layout residual 的定义

给定共同 `R,t`，人物 `i` 的三类内部 residual 是：

```text
translation residual:
||a_i - (R r_i + t)||

torso rotation residual:
angle(R T_i_post, T_i_target)

pairwise layout residual:
mean_j ||(a_i-a_j) - R(r_i-r_j)||
```

归一化分数：

```text
score_i = translation / 0.25
        + rotation_deg / 10
        + layout / 0.25
```

原 reject 规则仅在至少 3 人时触发，且最大分数需要满足：

```text
score_max > max(1.0, 1.5 * median_score)
```

最多删除一个人并重新求解一次，满足固定预算。但 V2 表明该规则只在 39.4% 的触发样本中删除了 GT-evaluated worst single，因此目前只能保留为负面消融。

### 8.4 人数不足时

- 至少 2 个 matched humans：可以计算多人 consensus；
- 只有 1 个 matched human：退化为该人物的单人 Boundary；
- 没有 matched human：本 Phase 1 样本无法由 human geometry 求 Boundary；
- 315 cuts 中有 308 个至少两人可用；
- 212 个 cuts 中三个人全部可用。

## 9. Boundary 的统一应用与输出

最终所有人共享：

```text
B = [R, t]
scale s = 1
```

对 post-cut camera：

```text
C_post_world = B * C_post_local
```

对 post-cut pointmap 中任意点：

```text
X_scene_world = R * X_scene_local + t
```

对每一个人的 root、joints 和 vertices：

```text
r_i_world = R * r_i_local + t
J_i_world = R * J_i_local + t
V_i_world = R * V_i_local + t
```

重要约束：

- 所有人使用同一个 `B`；
- camera、pointmap 和所有 SMPL-X 使用同一个 `B`；
- 不允许 per-person Boundary；
- 不额外修改某个人的 root；
- 不增加 foot translation；
- 不使用独立 human scale 或 scene scale；
- 整个 post shot 理论上固定复用一次求得的 Boundary。

当前 Phase 1 的数值 evaluator严格评价 camera 和 humans。pointmap 会在 viewer 中应用同一 Boundary，但由于 MultiHuman 没有 GT scene，本文不报告 scene accuracy，也不声称解决 Human3R 原有的人-地接触或悬空问题。

## 10. GT 使用边界：什么进入方法，什么只用于实验

| 变量 | Identity association | Boundary candidate | Evaluation |
|---|---:|---:|---:|
| RGB | Human3R prediction | 是 | 可视化 |
| Human3R camera/pointmap | 否 | 是 | 是 |
| Human3R root/torso/SMPL-X | 被分配 ID | 是 | 是 |
| head score/completeness | 否 | 仅 weighted 消融 | 诊断 |
| GT identity | 是，Oracle | 只提供对应关系 | 是 |
| GT SMPL-X projection | 是，Oracle cost | 否 | 是 |
| GT camera | 只为 GT mesh 投影 | 否 | 是 |
| GT root/joints/vertices | 否 | 否 | 是 |
| GT Boundary | 否 | 否 | 不直接使用 |
| source/camera ID learned cue | 否 | 否 | 仅定义实验 pair |

必须准确理解这里的 `GT-ID`：

- 它允许知道 post detection 对应哪个 pre person；
- 它不允许把 GT root 放进 translation equation；
- 它不允许把 GT torso 放进 V16；
- 它不允许用 GT camera 选择实际 Boundary；
- `Oracle Best Single` 例外地读取 evaluator 结果，只作为不可部署上界。

## 11. 为什么旧 GT-ID 实验不严格

The old assignment was:

```text
GT SMPL-X projected bbox
+ Human3R predicted mesh bbox
-> IoU and bbox-center cost
-> per-frame Hungarian assignment
```

This is GT-assisted association, but it is not a reliable GT-ID oracle. A bbox does not describe which overlapping body owns which pose. Human3R root/depth error also changes predicted bbox size and center.

The diagnostic case `three_t0900_c0_c3_k0` exposed the problem:

```text
pre predicted left order:  person1, person0, person2
pre GT left order:         person1, person0, person2
post GT left order:        person1, person0, person2
post old assignment:       person0, person1, person2
```

The old post-cut assignment swapped the first two people.

With the old assignment:

```text
candidate translation dispersion: 3.458 m
candidate rotation dispersion:    116.86 deg
camera error:                      4.947 m / 165.34 deg
```

After correcting `person0/person1`:

```text
candidate translation dispersion: 0.458 m
candidate rotation dispersion:    11.71 deg
camera error:                      0.695 m / 7.67 deg
```

Therefore, this failure was primarily an identity error, not evidence that two humans independently predicted the same wrong 180-degree Boundary.

## 12. 严格 GT-ID Oracle 实现与审计

For every Human3R detection and every GT identity:

1. Transform the predicted SMPL-X vertices back into the predicted camera frame.
2. Transform the GT SMPL-X vertices into the calibrated GT camera frame.
3. Project both meshes into the same 512x512 image.
4. Uniformly sample one of every 20 corresponding SMPL-X vertices and use their median 2D pixel distance.
5. Add only a small bbox-IoU tie-break term.
6. Solve one-to-one assignment using Hungarian matching.

The identity cost is:

```text
median corresponding-vertex projection distance / image diagonal
+ 0.05 * (1 - bbox IoU)
```

This uses GT SMPL-X and GT camera only to answer `WHO`. The selected identity labels connect pre-cut history to post-cut detections. GT geometry is not inserted into root, torso, rotation, translation, scale, pointmap, or Boundary solving.

Repeated-key determinism audit:

- 1,052 repeated `(camera, frame, detection index)` keys;
- identity contradictions across repeated runs: 0.

Assignment quality:

- assigned mesh projection distance median: 21.34 px;
- P90: 31.87 px;
- P95: 41.39 px;
- global assignment margin median: 0.104;
- old-to-new changed cases: 136 / 315;
- changed detection identities over all six-frame case windows: 667;
- changed post-cut identities: 147 in 74 cases.

This large correction count explains why the old aggregate was severely distorted.

## 13. Evaluation gauge 与指标定义

### 13.1 为什么需要 common gauge

Human3R 的 pre-shot prediction 和 dataset GT 不天然处于同一个 world coordinate。Evaluator 只用最后一张 pre-cut camera 建立合法的公共评价 gauge：

```text
G = C_pred_pre * inverse(C_gt_pre)
```

它把 dataset GT world 映射到 Human3R 的 pre-shot predicted world：

```text
C_gt_post_in_pred_gauge = G * C_gt_post
X_gt_human_in_pred_gauge = G * X_gt_human
```

方法输出则是：

```text
C_final = B * C_pred_post
X_final_human = B * X_pred_post_human
```

最终比较：

```text
C_final  vs  G * C_gt_post
X_final  vs  G * X_gt_post
```

`G` 只在 evaluator 中使用，不进入 `R_i`、`t_i`、quality、consensus 或最终 Boundary candidate。这样既消除合法 world gauge 差异，也不会把 GT Boundary 泄漏给方法。

### 13.2 Camera 指标

- translation error：预测和 GT camera center/translation 的欧氏距离；
- rotation error：两个 c2w rotation 的 SO(3) geodesic angle；
- composite：

```text
camera_composite = translation_error_m
                 + 0.02 * rotation_error_deg
```

Composite 只用于汇总、配对比较和 gate，不是训练 loss。

Catastrophic failure 定义为：

```text
translation_error > 2.0 m
OR
rotation_error > 45 deg
```

### 13.3 Human 指标

对 post-cut 检测到的每个 GT identity：

- world-root error；
- mean world-joint error；
- mean world-vertex error；
- per-person error；
- pairwise human distance error；
- pairwise human relative-vector error。

所有人物都先应用方法输出的同一个 `B`。Evaluator 不允许针对每个人单独做 Procrustes、translation alignment 或 scale alignment。

### 13.4 公平统计

- 单人方法可以在所有 315 cuts 上评价；
- 多人方法只在至少 2 个 shared identities 的 308 cuts 上评价；
- 单人与多人 paired test 使用同一 308-case support；
- 1/2/3 人数消融使用三人均可用的同一 212-case support；
- 报告 mean、median、P90、P95、improvement rate、harmful rate 和 Wilcoxon paired p-value；
- `Oracle Best Single` 在每个 cut 后读取 GT evaluator 选择最低 composite，只是不可部署理论上界。

## 14. 主要汇总结果

Camera composite is:

```text
translation error in meters + 0.02 * rotation error in degrees
```

| Method | N | Camera T mean/P90 | Camera R mean/P90 | Composite mean/P90 | Human root | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Single first | 315 | 0.663 / 1.011 | 11.51 / 19.24 | 0.893 / 1.370 | 0.402 | 1.3% |
| Single largest | 315 | 0.640 / 0.958 | 12.24 / 19.82 | 0.884 / 1.339 | 0.406 | 0.0% |
| Single highest confidence | 315 | 0.616 / 0.932 | 9.90 / 19.60 | 0.814 / 1.314 | 0.383 | 0.0% |
| Oracle Best Single | 315 | 0.493 / 0.764 | 7.01 / 11.66 | 0.633 / 0.935 | 0.364 | 0.0% |
| Multi naive mean | 308 | 0.517 / 0.793 | 7.01 / 11.65 | 0.657 / 0.977 | 0.361 | 0.0% |
| Multi confidence weighted | 308 | 0.569 / 0.872 | 7.39 / 12.01 | 0.717 / 1.077 | 0.386 | 0.0% |
| Multi Huber | 308 | 0.580 / 0.869 | 7.71 / 13.00 | 0.734 / 1.094 | 0.383 | 0.0% |
| Multi layout select + reject | 308 | 0.615 / 0.860 | 9.53 / 16.21 | 0.805 / 1.119 | 0.381 | 0.3% |

The most important observation is that naive multi-human mean is close to Oracle Best Single and clearly better than all deployable single-human selection rules.

## 15. 公平配对比较

All comparisons below use the same 308 cuts where at least two matched humans are available.

### 15.1 Highest-confidence single vs multi mean

| Metric | Single | Multi mean | Improvement rate | Wilcoxon p |
|---|---:|---:|---:|---:|
| Camera translation | 0.613 | 0.517 | 69.8% | 4.75e-14 |
| Camera rotation | 9.90 deg | 7.01 deg | 68.5% | 7.20e-13 |
| Camera composite | 0.811 | 0.657 | 74.0% | 1.20e-16 |
| Human root | 0.383 | 0.361 | 53.9% | 1.11e-3 |
| Human joints | 0.402 | 0.380 | 54.9% | 5.97e-4 |
| Human vertices | 0.392 | 0.372 | 54.5% | 4.24e-3 |

This is the strongest evidence that multi-human geometry is practically useful. It improves both camera and average human absolute accuracy over the best tested deployable single-anchor heuristic.

### 15.2 First single vs multi mean

- composite: 0.892 -> 0.657;
- improvement rate: 77.6%;
- harmful rate: 22.4%;
- Wilcoxon p: 4.85e-29;
- human root: 0.403 -> 0.361.

### 15.3 Oracle Best Single vs multi mean

Oracle Best Single chooses the lowest-error person after reading GT camera metrics. It is not deployable.

- composite: 0.627 -> 0.657;
- multi improvement rate: 45.1%;
- multi harmful rate: 54.9%;
- Wilcoxon p: 6.29e-3;
- human root: 0.363 vs 0.361, no significant difference;
- human joints: 0.382 vs 0.380, no significant difference;
- human vertices: 0.371 vs 0.372, no significant difference.

Multi mean nearly reaches the unavailable best-single camera upper bound, but does not exceed it. Therefore the pre-registered strict gate is not passed.

## 16. 人数消融

The following table uses the same 212 cases where all three humans are available. All subsets of each size are evaluated.

| Humans | Evaluations | T mean/P90 | R mean/P90 | Composite mean/P90 |
|---:|---:|---:|---:|---:|
| 1 | 636 | 0.594 / 0.861 | 10.80 / 18.75 | 0.810 / 1.236 |
| 2 | 636 | 0.560 / 0.848 | 8.81 / 16.15 | 0.737 / 1.098 |
| 3 | 212 | 0.549 / 0.813 | 7.49 / 12.52 | 0.699 / 1.028 |

This is a clear monotonic trend:

- more people reduce translation error;
- more people reduce rotation error more strongly;
- P90 improves together with the mean;
- catastrophic rate is zero for all subset sizes after strict identity correction.

Thus the multi-human benefit comes from additional redundant constraints, especially rotation averaging and tail suppression.

## 17. 为什么当前 robust consensus 更差

The old robust strategy assumes that a high layout/translation residual identifies the bad human. After identity correction:

- one-person rejection triggered in 99 cases;
- the rejected person was actually the GT-evaluated worst single only 39.4% of the time;
- layout selector chose the Oracle Best identity only 38.6% of the time;
- highest-confidence selection achieved 42.2%, which is still weak but better.

Consequently:

```text
naive mean composite:             0.657
Huber composite:                  0.734
layout select + reject composite: 0.805
```

Current residuals mix together:

- true reconstruction quality;
- human motion;
- raw root-depth error;
- torso orientation error;
- pairwise layout changes.

They are not calibrated well enough to decide which person should be rejected. The current system often removes useful evidence.

## 18. Temporal offset

| Offset | Oracle Best Single mean/P90 | Multi mean mean/P90 | Layout/reject mean/P90 |
|---:|---:|---:|---:|
| 0 | 0.615 / 0.910 | 0.631 / 0.954 | 0.803 / 1.114 |
| 1 | 0.616 / 0.907 | 0.654 / 0.989 | 0.827 / 1.116 |
| 2 | 0.626 / 0.872 | 0.644 / 0.969 | 0.820 / 1.061 |
| 4 | 0.634 / 0.899 | 0.653 / 0.977 | 0.788 / 1.185 |
| 8 | 0.675 / 0.988 | 0.706 / 1.054 | 0.788 / 1.118 |

Multi mean remains stable through 8-frame temporal cuts. Error increases mildly at offset 8, but there is no recurrence of the old catastrophic 180-degree failure distribution.

## 19. Viewer 案例解释

For `three_t0900_c0_c3_k0` after strict GT-ID correction:

| Method | Camera T | Camera R | Composite | Human root |
|---|---:|---:|---:|---:|
| Single person1 | 0.453 | 5.56 deg | 0.564 | 0.396 |
| Single person2 | 0.244 | 12.89 deg | 0.502 | 0.725 |
| Multi mean | 0.398 | 5.33 deg | 0.504 | 0.574 |
| Multi layout/reject | 0.695 | 7.67 deg | 0.848 | 0.572 |

The corrected multi viewer looks much better than the old multi viewer because the 165-degree identity-induced failure is gone. It does not prove that this multi method is better than every single-person Boundary:

- `person2` is best by camera composite but poor for absolute human placement;
- `person1` is better for average human placement;
- multi mean gives the best rotation and a balanced camera result;
- layout/reject is not the best multi solution for this case.

The old viewer comparison was therefore insufficiently fair.

## 20. 路线决策

### What is now established

1. The old Phase 1 FAIL report was confounded by identity swaps.
2. With strict GT-ID, multi-human averaging significantly improves over deployable single-human selection.
3. Two people help; three people help further.
4. The main gain is rotation stability and tail reduction, with a smaller translation and human-accuracy gain.
5. Multi-human geometry is worth retaining as a research direction.

### What is not established

1. Current multi fusion does not beat GT-evaluated Oracle Best Single.
2. Current confidence, Huber, layout, and rejection rules do not identify the best anchor reliably.
3. A deployable cross-shot identity matcher has not been tested.
4. Entry/exit, dustbin, occlusion TTL, and token memory remain untested.
5. This one sequence is still a debugging dataset, not a final benchmark.

### Frozen decision

Under the original strict rule `GT-ID multi > Oracle Best Single`, Phase 1 remains FAIL. However, the previous claim that multi-human geometry itself is ineffective is withdrawn.

The next engineering target should be a safer consensus rule that never performs worse than naive mean and only rejects a person when independent evidence is strong. Token Re-ID should not be trained until that geometry rule is fixed and revalidated. Raw Human3R token retrieval can still be probed separately without integrating it into Boundary solving.

## 21. 复现方式与产物

### 21.1 代码入口与函数职责

主要实现：

```text
scripts/v20_phase1_gt_id_multihuman_consensus.py
```

关键函数：

| 函数 | 职责 |
|---|---|
| `prepare_full_square_input` | 读取完整 RGB，缩放到 512，构造 Human3R view 输入 |
| `run_fresh_stream` | 调用 frozen recurrent inference，并在 index 5 做 pre-decode fresh reset |
| `layer_humans` | 将 Human3R SMPL-X 参数变成 roots、torso、joints、vertices、bbox 和 score |
| `sampled_background_cloud` | 删除人体区域并提取最多 1024 个稀疏高置信度 pointmap 点 |
| `assign_gt_identities` | 用 GT/predicted SMPL-X 对应顶点投影 + Hungarian 产生严格 Oracle ID |
| `reassign_cache_gt_identities` | 不重新推理 Human3R，直接修正旧 cache 中的 identity labels |
| `human_candidates` | 根据每个人的历史和 fresh post reconstruction 生成 `R_i,t_i` |
| `fixed_refine` | 运行 8-step local pointmap coarse refinement |
| `yaw_residual` | 执行 V16 bounded torso heading correction |
| `solve_consensus` | 实现 SO(3) mean、weighted、median、Huber 等多人融合 |
| `layout_candidate_selection` | 计算 translation/torso/layout residual 并选择候选 rotation |
| `evaluate_solution` | 在 common pre-shot gauge 中独立评价 camera 和全部人物 |
| `aggregate` | 生成 paired statistics、人数消融和 gate 结论 |

Viewer：

```text
scripts/v20_phase1_demo_viewer.py
```

Viewer 使用与 `demo.py` 相同的 `SceneHumanViewer`，显示：

- 稀疏背景 point cloud；
- predicted/GT camera；
- 所有 predicted/GT SMPL-X；
- `D` detection index、`L` 左右顺序和 `P` GT identity；
- 逐帧 0-5 timestep；
- single deployable、multi mean 和 Oracle Best Single 对照。

### 21.2 Cache 内容

每个 cut 的 cache 位于：

```text
output/v20_phase1_gt_id_multihuman_consensus/case_cache/
```

单个 cache 保存：

| 字段 | 内容 |
|---|---|
| `case` | timestamp、source/target camera、offset、pre/post frame indices |
| `poses` | 六帧 Human3R predicted c2w camera matrices |
| `humans` | 六帧全部人物的 root、torso、rotation、joints、vertices、bbox、score |
| `clouds` | 六帧稀疏背景 point clouds |
| `assignment` | 磁盘旧 cache 中是 legacy bbox assignment；V2 读取后立即在内存中替换为 mesh-projection cost、Hungarian result 和 margin |
| `gt.pre_c2w/post_c2w` | evaluator 使用的 GT camera |
| `gt.post_humans` | evaluator 使用的 GT root/joints/vertices |
| `inference_contract` | reset、输入尺寸及所有关闭模块的登记 |

现有 cache 总大小约 2.1 GB。V2 复用 raw Human3R 输出，只在内存中重新执行 GT-ID assignment，所以新旧结果差异不会来自重新推理的随机性。完整 V2 assignment 和 identity audit 保存在最终 JSON，而不是反向覆盖旧 cache。

### 21.3 结果文件

Machine-readable result:

```text
output/v20_phase1_gt_id_multihuman_consensus/
  v20_phase1_gtid_v2_offsets_0_1_2_4_8.json
```

Generated summary:

```text
output/v20_phase1_gt_id_multihuman_consensus/
  v20_phase1_gtid_v2_offsets_0_1_2_4_8.md
```

本文档是对模型、方法、实验和路线结论的完整解释；output 目录下的 JSON 是逐 cut 可复核的原始数值。

### 21.4 重评命令

Command:

```bash
.venv/bin/python scripts/v20_phase1_gt_id_multihuman_consensus.py \
  --evaluation_only \
  --timestamps 500 700 900 1000 1100 1300 1500 \
  --camera_pairs 0-1 1-2 2-3 3-4 4-5 5-0 0-3 1-4 2-5 \
  --offsets 0 1 2 4 8
```
