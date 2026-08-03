# V14 P1：冻结 B0 后的足部—局部场景可观测性诊断

日期：2026-08-03  
状态：**预注册；尚无结果。**

实现审计更新（2026-08-03）：首轮临时 cache 使用了 `smpl_v2d[joint_id]`，即把
joint index 错当作 vertex index，足部 patch 不在实际 joint UV 上。该 cache 和它的
`NO_GO` 输出只记录为实现无效审计，不能用于结论；正式 cache schema `2` 改用
`smpl_j2d[joint_id]` 并在完全相同的冻结 36-event split 上重跑。

## 问题与范围

P0 / P0.1 已经否定了“仅改变 real-multi camera-only 训练比例，就能让单一隐式
`B0` 同时解决跨域 camera tail 和真实多人的 root/layout”的假设。P1 不重新训练或
改写相机，而只检验一个更窄、可证伪的问题：

```text
冻结 P0 B0 camera
  + 冻结 BRTC-LC 的已接受人
  + 同一次 Human3R forward 输出的、足部附近非人体 pointmap/confidence
  -> 能否在 GT 不参与 action 的前提下，预测一个与真实 root residual 同向的
     小型 per-person translation？
```

这不是旧的 pelvis depth/mask 规则重跑。旧规则已在
`V14_INTERNAL_ROOT_DEPTH_FEASIBILITY_20260730.md` 判定 No-Go：同一 forward 的
pointmap 可以平滑却共享错误 gauge，且不应把人或场景反过来用作 camera Boundary。
本实验只把局部 surface 当作**person placement consistency**证据；camera、B0、
raw recurrent state、scene pointmap 都没有任何更新权限。

## 固定基线与数据隔离

固定 checkpoint：

```text
output/v14_cut_first_cross_source/
v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth
SHA256: de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265
```

固定 runtime：P0 read-only shadow 产生 `B0 = C_shadow @ inv(C_raw)`，clean
raw-reset 仍是唯一提交状态；再执行原样、camera-free 的 BRTC-LC。P1 的 action
只允许叠加在 BRTC 已接受的 post person 上；未匹配、BRTC 拒绝、脚/平面不可观测、
左右脚冲突时必须逐数值精确回退到 BRTC（或 B0）。

开发集：`v14_multihuman_camera_supervision_20260803.json` 中原先已打开的 36 个
`three` pair/timestamp-disjoint event。它只用于判断信号是否存在和冻结一个 policy，
绝不作为最终 confirmation。P1 的第一个阶段不读取 EgoHumans/dance/box。

运行输入先于 GT 严格构造：模型前向、B0、anonymous association、BRTC 和全部
foot-plane proposal 都写入 `runtime` cache；之后才读取 calibration/SMPL-X GT 并写入
`evaluator`。信号 probe 先从 runtime 创建 action/feature，随后才访问 evaluator。

## 新缓存（只保留紧凑 runtime evidence）

每个 pre/post event 的 raw post（以及 shadow pre）均从当前 Human3R forward 提取：

```text
camera-local pts3d_in_self_view, conf_self, emitted human mask
+ 初始实现的 SMPL-X 左/右 foot joint（10/11）projected UV、camera-local 3D、深度、in-frame flag
-> 每个 anchor 一个 33×33 patch
   - raw XYZ / raw confidence
   - finite/depth validity
   - union-human-mask dilation 3px 后的 non-human flag
   - 4..16 px foot-centered annulus、精确 patch UV
-> transform at runtime only with frozen B0 / camera
```

不保存全图 pointmap，不使用完整 human bbox 排除，不用 GT identity、相机或网格
构造 patch。mask 是 Human3R 输出的 union mask，故这是保守的 local scene support；
它不能声称是独立深度 teacher。

## 预测量与 action 之前的固定算法

对 BRTC accepted 的匿名匹配 `(pre i, post j)`：

1. 从每个脚 anchor 的 patch 仅保留有效、非 human、annulus 内点；按 confidence
   rank deterministic 加权；要求至少 24 点、至少 3 个 UV quadrants、3D extent ≥5 cm。
2. 用加权 PCA 拟合局部 plane，要求 weighted median point-to-plane residual ≤2 cm；
   normal 翻向 foot anchor。相同 person 的 pre 与 post plane normal 必须相差 ≤25°。
3. `d_pre` 与 `d_post` 分别是足 anchor 对 pre/post plane 的 signed distance；不把
   `d_post` 逼为 0，而是保持 last-pre 的 signed offset。仅接受两脚均可用、
   `|d_pre|, |d_post| ≤20 cm`、两脚 translation proposal 的差 ≤2 cm，且 proposed
   local residual 能降低 ≥10%。
4. per-person action 是 `clip(0.5 * median(proposal_left, proposal_right), ||·||≤30mm)`。
   该 action 平移该 person 的 root/joints/vertices；不旋转、不改 shape/pose、
   不改 camera、也不改 pointmap。

这些数值是第一轮 development diagnostic 的固定 gate，不会为 held-out 调整。

## 第一阶段的可证伪 gate

先不报告“效果提升”，只判别是否存在有用可观测性。对每个可用人，令

```text
r* = GT world root - frozen B0+BRTC world root
p  = prediction-only signed foot-support proposal（action 之前、未截断）
```

同时报告：coverage、每个 fallback reason、foot/plane quality、proposal magnitude、
`dot(p, r*)` 的同向率、cosine/axis correlation，以及 30 mm 小动作对 root/joint/
vertex 与 pair layout 的反事实结果。相机 SHA256 与 B0/BRTC camera max delta 必须为零。

P1 仅在同时满足下列条件时才进入一个小型 root-only sidecar 或确定性 action 的选择：

```text
valid matched person count >= 24
coverage >= 20% of BRTC-accepted persons
proposal-vs-residual direction: bootstrap 95% lower CI of mean cosine > 0
and sign/directional agreement >= 60%
30-mm bounded action improves mean root error by >= 5 mm
and no camera change / no fallback mutation
```

任一条件失败则记录：

```text
NO_GO_HUMAN3R_FOOT_SCENE_SIGNAL
```

并停止把该同次 forward 的足部 pointmap 当作可部署精对齐证据；不在 held-out 数据上
调阈值救结果。若通过，才 freeze cache/policy SHA 后在 `three` offset1、dance、box
和 untouched EgoHumans chain 上进行独立 confirmation。

## 成功也不能越界的结论

即使 P1 通过，也只能说明 **camera-invariant、bounded person root residual** 在有
observable foot-scene evidence 时值得继续。它不证明 Human3R 绝对尺度已经正确，
不允许更新 B0，也不等价于 Multi-THuMBS official benchmark 或独立 contact GT。

## 已完成结果（schema 2；2026-08-03）

结论：

```text
NO_GO_HUMAN3R_FOOT_SCENE_SIGNAL
```

这是一条关于**通用 first-post root refinement**的 No-Go，不是“脚部几何永远无用”的
结论。36-event development 的绝大多数 Human3R person crop 没有同时看见两只脚，故
没有达到预注册的最小可观测 coverage；在这种条件下训练一个 head 只会把未观测 root
residual 变成先验猜测。

固定对象：

```text
checkpoint: v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth
SHA256:     de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265
split:      v14_multihuman_camera_supervision_20260803.json / 36 `three` dev events
cache:      output/v14/fine_alignment_research/p1_foot_scene_observability_v2/
report:     .../P1_FOOT_SCENE_SIGNAL_REPORT.json
```

每一个 event 均先完成 prediction-only raw/shadow `B0`、anonymous match、frozen
BRTC 和 joint-UV foot patch；GT calibration/mesh 只随后写入 evaluator payload。36 个
B0 camera 的 SHA256 都在 runtime 与 evaluator 访问前复核；camera max change 是
`0.0`，未使用 future post frame、DA3 或其他外部模型。

| 项目 | 数值 |
|---|---:|
| development events | 36 / 36 成功 cache |
| matched person rows | 96 |
| BRTC accepted 且有 evaluator target | 92 |
| 通过完整双足 gate 的非零 candidate | 0 |
| direction / counterfactual 可评价样本 | 0 |
| BRTC reject exact fallback | 4 |
| 双足不可观测 exact fallback | 90 |
| left/right proposal 冲突 exact fallback | 2 |

失败原因来自 prediction-only gate，而不是 GT：在 BRTC accepted rows 的足部检查中，
post/pre foot `not_in_frame` 分别出现 `113/112` 次；只有 27 个 individual foot
pre/post plane pair 同时完成 plane fit，且其中仍有 9 个违反 contact-range 或 normal
agreement。因两个足 anchor 是预注册的最小反错配条件，不能看完结果后把“双足”放宽为
单足或去掉 contact gate 来制造 coverage。

因此 P1 的五项可证伪 Go 条件（`>=24` valid、`>=20%` coverage、cosine CI 下界为正、
`>=60%` direction agreement、root mean 至少 `5 mm` gain）全部不可满足；只有 camera
bit-exact 条件通过。没有运行 offset1/dance/box/EgoHumans confirmation，也没有训练
任何 root head。

### 无效 schema 1 的记录

`output/v14/fine_alignment_research/p1_foot_scene_observability/` 保留为可复查的
**invalid implementation artifact**：它曾将 `smpl_v2d[joint_id]` 当作 joint UV，实际
选中的是 vertex `joint_id`，造成错误 foot patch。它的初始 No-Go 数字不参与上述表格，
不得引用。修正后的 schema 2 使用 `smpl_j2d[joint_id]`，已通过单 case pixel/pointmap
对照后完整重跑。对应 smoke cache 也只用于 implementation audit。

### P1 给出的下一步约束

不要把 Human3R same-forward foot pointmap 加入主线，也不要在已读 development 上调
patch、单脚、confidence、contact 或 cap 阈值。若未来系统允许“可见脚的后验 optional
correction”，它必须在全新 split 上作为低 coverage 的独立 module 重启协议；它不能是
当前 one-shot Movie3R 的统一精对齐答案。
