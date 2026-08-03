# V14 P3：双视图关节射线约束的人体刚体精对齐诊断

日期：2026-08-03  
状态：**预注册；尚无结果。**

## 问题

冻结 `B0` 后，已有 BRTC-LC 将每个 pre/post 同名关节的两条预测射线三角化，但它只把
所得 evidence 压缩成一个沿 post root ray 的平移；随后 Kabsch 只用 pre/post 预测 torso
相对结构旋转身体。两者之间尚有一个未直接检验的、物理上明确的组合：

```text
same-time pre/post predicted joint rays
  -> per-joint closest-point 3D midpoint target
  -> a rigid SE(3) fit from post predicted torso/core skeleton to these targets
  -> bounded person root + orientation correction at frozen B0 camera
```

P3 不使用外部 2D keypoint/ReID/depth 网络，也不让人体反写 B0。它只是检验：已有的多视角
ray evidence 是否足以同时约束 root 与 global orientation，而不是分别由 BRTC 和 Kabsch
处理。若这个量在严格的 GT-after-action 诊断中都不能超过 BRTC，则不能把“联合刚体拟合”
包装成新方法。

## 固定输入与隔离

直接复用 P1 schema-2 的正式 runtime cache：

```text
output/v14/fine_alignment_research/p1_foot_scene_observability_v2/
```

它固定为相同的 `three` 36-event development split、same P0 checkpoint
`de2430ed...828265`，且先完成以下 prediction-only transaction：

```text
pre/post RGB -> shadow/raw forward -> B0 -> geometry Hungarian -> frozen BRTC
```

P3 action 只读取其中的 `pre_camera_c2w`、`b0_camera_c2w`、`pre_people`、
`b0_post_people`、geometry Hungarian pair 与 frozen BRTC output。每一 pair 的 ray target、
SE(3) candidate 和所有 action 在本 probe 打开 `evaluator` target / P2 identity labels 之前
构造。P2 cache 仅在 action 已完整后用来做 evaluator-only 匹配正确性分层；它不能改变
runtime match 或 gate。

`B0` camera 数组的 SHA256 在 action 前后复核。P3 禁止改变 camera、scene pointmap、
recurrent state、shape、articulation 或未匹配/BRTC-rejected person；无有效 rigid candidate
时 exact fallback 到对应 B0/BRTC geometry。

## 固定 ray target 与诊断候选

对 geometry Hungarian 的 pair `(pre i, post j)`，用固定 joint set：

```text
CORE5 = pelvis(0), hips(1,2), shoulders(16,17)
```

每个 joint 从 `C_pre -> J_pre`、`C_post -> J_post` 得到两条 ray，保留正深度且非退化的
closest-point midpoint。只有全部五个 joint 均有效时才生成 rigid candidate；这比 BRTC
更严格，避免把不完整 skeleton 假装成 6-DOF 证据。

一次性预注册两个候选，**不在本 split 上选择 alpha 或 gate**：

1. `ray_rigid_se3_full`：对 raw B0 post `CORE5` 到五个 ray midpoints 做完整刚体 Kabsch，
   将该 SE(3) 作用于此人的 root/joints/vertices。这是 “joint ray rigid registration” 的
   直接、最大动作诊断。
2. `brtc_ray_target_so3_q25`：保持 frozen BRTC root 和 translation；只将 raw B0 core
   skeleton 到 ray-target skeleton 的 Kabsch rotation 的固定 `0.25` fraction 绕 BRTC root
   作用于 joints/vertices。它只检验 ray target 是否比 existing pre/post relative Kabsch
   提供更好的 orientation target。

`q25` 的 `0.25` 和 full action 都在读取任何 P3 result 前固定。二者都不是 P3 通过前的
部署 policy；尤其 full SE(3) 的预测拟合变好不能代替 GT gain。

## 可证伪 Go / No-Go

这是可观测性筛查，而非在 development 上调一个新 runtime。P3 仅当下列全部满足时，才有
资格另外建立 pair-disjoint policy-selection / confirmation：

```text
correct geometry-matched, five-ray, BRTC-accepted persons >= 24
ray_rigid_se3_full mean root error <= frozen BRTC mean - 5 mm
brtc_ray_target_so3_q25 mean joint and vertex error <= BRTC - 5 mm each
neither candidate increases its compared root/joint/vertex >5 cm harm rate
camera bit-exact; runtime action occurs before evaluator access
```

比较同时报告 all-geometry-pair 和 evaluator-correct pair strata；Go 以正确关联 stratum
判断，但全量 stratum 不能隐藏。任何条件失败则记录：

```text
NO_GO_RAY_RIGID_PERSON_OBSERVABILITY
```

不能在已读 split 后再把五关节改为三关节、扫描 fraction/cap、利用 GT association，或只保留
看起来好的一条 person。若 No-Go，结论限于：当前 Human3R 关节 ray 只支持 BRTC 已有的
depth/root evidence，不能可靠提供额外可提交的 joint-rigid target；这不否定未来拥有独立
2D correspondence detector 的方法。

## 已完成结果（2026-08-03）

结论：

```text
NO_GO_RAY_RIGID_PERSON_OBSERVABILITY
```

P3 只重放 P1 schema-2 和 P2 的同 checkpoint、同 36-event `three` runtime cache；无 GPU
forward、无新模型、无 future frame。每个 event 的五关节 ray midpoint、full SE(3) 与 q25
rotation 都在读取 P1 target/P2 identity evaluator 前创建。所有 `36` 个 B0 camera SHA256
保持一致，camera max change 为 `0.0`。

| Stratum / method | N | Root (m) | Joint (m) | Vertex (m) |
|---|---:|---:|---:|---:|
| all geometry pairs: B0 | 96 | .3285 | .3559 | .3469 |
| all geometry pairs: frozen BRTC | 96 | **.1724** | **.1983** | **.1788** |
| all geometry pairs: ray-rigid full SE(3) | 96 | .1850 | .2184 | .2049 |
| all geometry pairs: BRTC + ray-target SO(3) q25 | 96 | .1724 | .1987 | .1811 |
| correct geometry match + five rays + BRTC accepted: B0 | 92 | .3268 | .3551 | .3457 |
| correct geometry match + five rays + BRTC | 92 | **.1638** | **.1906** | **.1703** |
| correct geometry match + five rays + full SE(3) | 92 | .1770 | .2116 | .1975 |
| correct geometry match + five rays + q25 SO(3) | 92 | .1638 | .1911 | .1728 |

可观测 coverage 并不是问题：`92` 个正确匹配且 BRTC accepted person 都有完整五 ray target，
只有 `4` 个 BRTC rejected rows exact fallback。问题是 target 本身不能提供 BRTC 之后额外的
结构真值。full SE(3) 相对 BRTC 使正确匹配 stratum 的 root/joint/vertex 分别恶化
`13.15/20.95/27.21 mm`，且三项 `>5 cm` harm 分别为 `10.87/9.78/14.13%`。q25 保持 root
不变，但 joint/vertex 仍恶化 `0.43/2.44 mm`，且两项各有 `2.17%` 的 `>5 cm` harm。

预注册的八项 gate 中，只有 sample count、camera bit-exact 和 runtime-before-evaluator
通过；full root gain、q25 joint/vertex gain 和两条 harm 条件均失败。不能在已读 split 上
扫描 core joint subset、rotation fraction、cap 或 post-hoc reliability score 来救该候选。

P3 给出的可复用结论是：以同一 Human3R predicted 3D joints 生成的 cross-view rays 可以
稳健提供 BRTC 已利用的 ray-depth cue，但 ray midpoint 的横向/刚体结构误差与 Human3R 的
camera-relative pose bias耦合，不能作为更高维 SE(3) person target。后续若要处理 residual
orientation/articulation，必须引入与该预测 joint geometry 正交的可观测信息；不要重扫
BRTC/Kabsch 同一 ray family。
