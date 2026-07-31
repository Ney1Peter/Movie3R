# V14 BRTC 后的 person-local global-orientation Kabsch（2026-08-01）

## 1. 结论

这个实验给出了目前第一个具有较高落地可行性的精对齐候选：

```text
frozen B0 camera
-> frozen BRTC-LC v1 root/layout translation
-> accepted person 的 torso4 显式对应
-> bounded person-local Kabsch SO(3)
-> 绕已经校正的 Human3R root 旋转 joints/vertices
-> camera、native root、rejected/unmatched person 不变
```

它没有引入 DA3、图像 backbone、ReID 或其他预训练模型，只使用上一 shot 最后一帧和
当前 shot 第一帧，严格在线、因果。冻结参数后，它在 `three offset1`、`dance`、`box`
上都降低 joint/vertex 均值，同时保持 native root、camera 和 pair-root layout 不变；在
EgoHumans CPU cache 上还同时改善 W、WA、pelvis MPJPE/MPVPE、fixed joint/vertex 和
root/joint Accel。

需要保留两层决策：

- 按预先声明的“所有 evaluator 浮点均值都不得增加”零容差门槛，它是 **strict NO-GO**；
  唯一回退是 SMPL-X→SMPL 映射后重新回归的 pelvis-root `+0.034 mm`。
- native Human3R root 实际为 bit-exact，且 mapped pelvis 与 native root 本身中位相差
  `18.193 mm`。若对这个转换 proxy 使用显式 `0.1 mm` 容差，则它是
  **QUALIFIED_GLOBAL_ORIENTATION_KABSCH_CANDIDATE**。

因此当前不直接替换 frozen BRTC-LC v1，而是把它作为“BRTC 后的人体朝向精对齐模块”
保留并继续扩大数据验证。它已经明显比前面的 scalar group damping、angular gate 和
纯几何 identity dustbin 更接近可部署答案。

## 2. 为什么需要这个模块

BRTC-LC v1 只给每个人施加刚体平移。它能明显修正 root/depth 和多人布局，但不能改变：

- 人体在世界系中的 global orientation；
- pelvis 对齐后的局部 pose/shape；
- 因 Human3R 不同 shot 独立预测而产生的人体整体旋转漂移。

同一个人在相邻 shot 边界处，躯干的真实朝向通常连续。相机已经由 B0 放入同一世界系
后，可以把 last-pre 与 current-post 的 root-centred torso joints 当成显式 3D 对应，
估计一个只属于这个人的小旋转。该旋转不需要再次修改相机或 root。

## 3. 输入、模块与输出

### 3.1 输入

每个 shot boundary 只读取当前时刻可获得的内容：

```text
pre_camera:    上一 shot 最后一帧的 frozen-B0 camera-to-world
post_camera:   当前 shot 第一帧的 frozen-B0 camera-to-world
pre_people:    last-pre Human3R root/joints/vertices
post_people:   current-post Human3R root/joints/vertices
matches:       当前匿名 root+torso+joint Hungarian 结果
BRTC evidence: frozen ray triangulation accept/reject 与 translation
```

不读取 future post frame、GT、数据集名字、camera ID、人物身份字符串或 RGB feature。

### 3.2 Frozen BRTC translation

首先原样执行 frozen BRTC-LC v1：

```text
post person
-> core-joint camera-ray triangulation
-> observable evidence gate
-> shared group + layout-selected individual residual
-> accepted person 的 corrected root/joints/vertices
```

BRTC rejected 或 unmatched 的人直接保持 exact B0，后续 Kabsch 不得触碰。

### 3.3 Torso4 显式对应

对每个 BRTC accepted match 取关节：

```text
TORSO4 = (left/right hip, left/right shoulder) = (1, 2, 16, 17)
```

分别减去各自 Human3R root：

```text
X_pre  = joints_pre[TORSO4]  - root_pre
X_post = joints_post[TORSO4] - root_post
```

这样 translation 已经被消掉，只估计人体局部朝向差。

### 3.4 Kabsch 与有界旋转

求解：

```text
R_raw = argmin_R || X_post R^T - X_pre ||,  R in SO(3)
```

SVD 后显式修正 reflection，保证 `det(R)=+1`。冻结策略不应用完整旋转，而是：

```text
applied angle = min(0.5 * raw angle, 25 degrees)
```

只有候选旋转确实降低 predicted torso correspondence residual 时才应用。最终：

```text
root_out     = BRTC root                       # bit-exact
joints_out   = (BRTC joints   - root) R^T + root
vertices_out = (BRTC vertices - root) R^T + root
camera_out   = frozen B0 camera                # bit-exact
```

同一旋转在 post shot 内绕每一帧自己的 translated root 传播；第二个 cut 使用已经因果继承
的 orientation state。没有平滑未来帧或回看完整序列。

## 4. Development、冻结与验证协议

### 4.1 Development

只在 `three offset0` 的 41 cuts / 122 people 上扫描：

```text
max angle:       2, 5, 10, 15, 25 degrees
rotation fraction: 0.5, 0.75, 1.0
minimum observable relative improvement: 0, .05, .10, .20
```

候选必须同时满足 root、joint、vertex、pair distance、pair vector、coverage、root harm 和
camera non-regression，再按 joint+vertex 最小选择。60 个组合通过，最终冻结：

```json
{
  "max_angle_deg": 25.0,
  "rotation_fraction": 0.5,
  "min_observable_relative_improvement": 0.0
}
```

Policy canonical SHA256：

```text
59e42e235134f5cf3a1e2962d30e06de5cc386c1033f36372db4ace35ff5a423
```

### 4.2 Development 结果

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Applied |
|---|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .225088 | .270404 | .248604 | .102222 | .260536 | 0% |
| BRTC + Kabsch | .225088 | **.267442** | **.246160** | .102222 | .260536 | 88.5% |

### 4.3 冻结后的 MultiHuman 结果

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Applied |
|---|---|---:|---:|---:|---:|---:|---:|
| three offset1 | BRTC v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 0% |
| three offset1 | + Kabsch | .231437 | **.271315** | **.250248** | .098351 | .258779 | 88.0% |
| dance | BRTC v1 | .125131 | .177804 | .152914 | .044141 | .078318 | 0% |
| dance | + Kabsch | .125131 | **.168764** | **.148234** | .044141 | .078318 | 99.2% |
| box | BRTC v1 | .372345 | .421610 | .434528 | .063069 | .427334 | 0% |
| box | + Kabsch | .372345 | **.418583** | **.429938** | .063069 | .427334 | 98.7% |

这些是 candidate-specific freeze-then-read 结果，但 `dance/box` 在整个 V14 研究中已经被
其他分支使用过，因此不能称为整个项目从未看过的 pristine test。后续仍需新增序列确认。

## 5. EgoHumans 与 Multi-THuMBS 风格指标

同一份 `001_legoassemble`、3×15 帧、6 cuts CPU geometry cache：

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | Fixed root | Fixed joint | Fixed vertex | Pair dist | Pair vec | Root Accel | Joint Accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 314.059 | 202.461 | 109.266 | 129.960 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 |
| + Kabsch | **312.769** | **200.029** | **101.526** | **119.928** | 380.688 | **383.933** | **383.791** | **176.559** | **333.091** | **115.698** | **123.167** |

相对 BRTC v1：

- W/WA 改善 `1.290/2.432 mm`；
- pelvis MPJPE/MPVPE 改善 `7.741/10.032 mm`；
- fixed joint/vertex 改善 `0.796/1.447 mm`；
- root/joint Accel 改善 `0.315/2.104 mm/frame²`；
- mapped-pelvis fixed-root 退化 `0.034 mm`。

Multi-THuMBS 论文 EgoHumans 参考 W/WA 为 `279.0/166.0 mm`。当前 Kabsch 仍分别高
`33.769/34.029 mm`；相比 BRTC v1 原先的差距 `35.059/36.461 mm`，只关闭了其中一小部分。
本地 pelvis MPJPE/MPVPE 数字看似低于论文 `228.3/262.2 mm`，但本地 split、匹配、漏检
处理和官方 evaluator 未公开，不能据此宣称超过论文。论文 Accel `27.3` 的完整定义也未
公开，不能与本地 `mm/frame²` 直接排名。

## 6. 安全性与尾部风险

Ego runtime 审计：

```text
BRTC accepted                 = 11/14 matched people
Kabsch applied                = 11/11 accepted people
post-shot propagated frames   = 55/80 (68.8%)
rejected/unmatched B0 delta   = 0
native root vs BRTC delta     = 0
camera vs B0 delta            = 0
SO(3) orthogonality/det error = about 1e-16
joint/vertex harm >5cm        = 0%
```

MultiHuman 全 offset 统计中，`dance offset8` 有 2 个人的 joint mean 相对 BRTC 增加超过
5 cm；online immediate `offset0` 没有这种 >5 cm 反例。该现象符合预期：pre/post 间隔
变大后，真实人体动作会混入“shot coordinate orientation drift”，单帧 Kabsch 无法区分。
因此 runtime 应只在边界当前帧使用，遇到长时间缺帧应 fallback，而不能把 `k8` 当普通
相邻 shot 处理。

## 7. 当前判断与下一步

这个候选已经回答了一个重要问题：BRTC translation 后，使用显式 person-local SO(3)
确实能继续降低人体误差，而且不需要再动相机或 native root。它不是重新做一次粗对齐，
而是补上 BRTC 从结构上无法表达的 global orientation 自由度。

正式晋级前还需要：

1. 在未参与当前研究的新 MultiHuman/EgoHumans shot 序列上复验；
2. 给真实 frame gap 增加可观测 hard fallback，避免 `offset8` 动作混淆；
3. 统一 native Human3R root 与 Multi-THuMBS mapped-SMPL pelvis 的评估语义；
4. 将 deployable runtime 与冻结 probe 做全 case parity，并保留 causal orientation state；
5. 与正在测试的 shared/group SO(3) 和 body-scale consistency 比较，最终最多保留三种。

## 8. 产物

```text
versions/v14/b0_person_triangulation_orientation_kabsch.py
versions/v14/probe_brtc_global_orientation_kabsch.py
versions/v14/eval_brtc_global_orientation_kabsch_egohumans.py
versions/v14/tests/test_b0_person_triangulation_orientation_kabsch.py
versions/v14/docs/V14_BRTC_GLOBAL_ORIENTATION_KABSCH_EGOHUMANS_20260801.md

output/v14/fine_alignment_research/brtc_global_orientation_kabsch/
  DEV_SCAN.json
  FROZEN_POLICY_BEFORE_VALIDATION.json
  VALIDATION_RESULTS.json
  egohumans/report.json
```
