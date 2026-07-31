# V14 BRTC Shared/Group SO(3) Kabsch 实验报告

> 日期：2026-08-01
>
> 约束：参数直接继承已冻结 individual Kabsch；不在已打开的 held-out 调参；frozen B0、
> frozen BRTC-LC v1；无新模型、无 GT 决策、无未来帧；camera/root translation 不更新；
> rejected/unmatched exact B0；CPU cache only；不修改 V9。

## 1. 最终结论

本实验测试：多人是否存在一个共同的 shot-level global body rotation，可以减少 individual
Kabsch 把每个人真实动作误当成跨 shot 朝向误差的问题。

继承策略：

```text
max_angle_deg = 25
rotation_fraction = 0.5
min_observable_relative_improvement = 0
```

结果分成两部分：

1. `three offset0` 机制检查和盲测 `offset1/dance/box` 全部通过。root 与 pair layout exact，
   joint/vertex 三组均改善，证明 accepted-set 共同 SO(3) 是可观测且有跨序列空间价值的。
2. EgoHumans 连续链中，W/WA、pelvis MPJPE/MPVPE、两项 Accel 均改善，但 fixed-world
   joint/vertex 分别轻微退化 `+0.177/+0.006 mm`。按预先要求的所有均值零容忍
   non-regression，最终仍不能部署。

```text
GO_SHARED_KABSCH_TO_EGO
NO_GO_SHARED_ORIENTATION_KABSCH_EGOHUMANS
```

所以 shared Kabsch 是明确的正向研究证据，但没有超过已经测试过的 individual Kabsch，也
没有成为最终默认精对齐方法。

## 2. 原理与架构

### 2.1 Frozen translation branch

每个 boundary 首先保留 frozen BRTC-LC v1 的 translation 结果：

```text
pre_camera, post_camera
pre_people, post_people
anonymous matches
        ↓
frozen BRTC ray evidence + accept/reject + group/residual translation
        ↓
BRTC-corrected roots/joints/vertices
```

本候选不改变任何 root translation，也不改变 BRTC accept/reject。

### 2.2 Accepted-set shared torso4

只汇总 BRTC accepted people。对每个 accepted match，使用 SMPL-X torso4：

```text
left hip, right hip, left shoulder, right shoulder
joint indices = [1, 2, 16, 17]
```

分别减去自己的 root：

```text
P_pre_i  = pre_torso4_i  - pre_root_i
P_post_i = post_torso4_i - post_root_i
```

把所有 accepted 人的 4 点拼成一个集合：

```text
P_pre  = concat(P_pre_1, ..., P_pre_N)
P_post = concat(P_post_1, ..., P_post_N)
```

用 Kabsch 求一个共同 rotation：

```text
R_raw = argmin_R mean ||R * P_post - P_pre||, R ∈ SO(3)
```

将 rotvec 乘 frozen fraction `0.5`，再截断到最大 `25°`，得到 `R_shared`。

### 2.3 Observable gate

只使用 predicted torso residual：

```text
before = mean ||P_post - P_pre||
after  = mean ||R_shared * P_post - P_pre||
relative_improvement = (before - after) / before
```

当：

```text
applied_angle > 0
after < before
relative_improvement >= 0
```

才动作。没有 GT、图像模型、future post frame 或 dataset identity。

### 2.4 输出动作

同一个 `R_shared` 分别绕每个 accepted 人自己的 corrected root 旋转：

```text
joints'_i   = R_shared * (joints_i   - root_i) + root_i
vertices'_i = R_shared * (vertices_i - root_i) + root_i
root'_i     = root_i
```

- accepted 人共享同一 rotation；
- rejected 人 exact B0；
- unmatched 人 exact B0；
- native Human3R root exact frozen BRTC；
- camera exact B0；
- native-root pair distance/vector exact。

## 3. 参数继承与防污染

没有扫描 shared Kabsch 新参数。直接读取此前在 individual Kabsch development 冻结、并在
held-out 打开之前写入的 policy：

```text
output/v14/fine_alignment_research/brtc_global_orientation_kabsch/
FROZEN_POLICY_BEFORE_VALIDATION.json
```

SHA256：

```text
59e42e235134f5cf3a1e2962d30e06de5cc386c1033f36372db4ace35ff5a423
```

代码会同时检查 checksum 与 `25° × 0.5 × improvement>=0` 精确值，防止事后换参数。

## 4. Three offset0 机制检查

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Applied boundaries | Applied people |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .225088 | .270404 | .248604 | .102222 | .260536 | 6.6% | 0 | 0 |
| shared Kabsch | .225088 | **.269937** | **.248233** | .102222 | .260536 | 6.6% | 39/41 | 102 |

root/layout/harm/camera/rejected fallback 均通过；joint/vertex 改善 `0.467/0.371 mm`。这一步
只检查机制，没有选择或修改参数。

## 5. Blind held-out

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Applied boundaries | Applied people |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| offset1 | BRTC v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 7.2% | 0 | 0 |
| offset1 | shared Kabsch | .231437 | **.273645** | **.252027** | .098351 | .258779 | 7.2% | 42/42 | 110 |
| dance | BRTC v1 | .125131 | .177804 | .152914 | .044141 | .078318 | 3.3% | 0 | 0 |
| dance | shared Kabsch | .125131 | **.175848** | **.150222** | .044141 | .078318 | 3.3% | 59/61 | 117 |
| box | BRTC v1 | .372345 | .421610 | .434528 | .063069 | .427334 | 6.4% | 0 | 0 |
| box | shared Kabsch | .372345 | **.419361** | **.430708** | .063069 | .427334 | 6.4% | 78/78 | 154 |

改善量：

| Split | Joint | Vertex |
|---|---:|---:|
| offset1 | -0.848 mm | -0.424 mm |
| dance | -1.956 mm | -2.692 mm |
| box | -2.249 mm | -3.820 mm |

三组 root 和两项 layout 均 bit-exact；harm 不增加；rejected/unmatched fallback 为零。因此
盲测状态为：

```text
GO_SHARED_KABSCH_TO_EGO
```

## 6. EgoHumans causal CPU replay

### 6.1 状态分离

Ego replay 把 translation 与 orientation 状态明确分开：

- frozen BRTC translation branch 始终读取自己的 v1 reference history，保证 orientation
  不会反向污染下一 cut 的 root/ray translation；
- shared Kabsch estimator 读取实际已旋转的 last-pre orientation state；
- rotation 传播到该 post shot 中相同 native track 的所有帧；
- 第二 cut 确实读取第一 cut 的旋转结果。

这样保持一次流式/因果，同时保证 stored roots bit-exact v1。

### 6.2 主要结果

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | Fixed root | Fixed joint | Fixed vertex | Pair dist | Pair vec | Root Accel | Joint Accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 314.059 | 202.461 | 109.266 | 129.960 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 |
| shared Kabsch | **312.885** | **200.587** | **105.551** | **125.282** | 380.717 | 384.906 | 385.244 | **176.922** | **333.633** | **115.914** | **124.117** |

相对 v1：

```text
W / WA                 -1.174 / -1.874 mm
pelvis MPJPE / MPVPE   -3.715 / -4.678 mm
fixed root             +0.063 mm
fixed joint            +0.177 mm
fixed vertex           +0.006 mm
pair distance/vector   -0.103 / -0.237 mm
root/joint Accel       -0.100 / -1.153 mm/frame²
```

好的一面是 W/WA、local pose、layout proxy 和两项 Accel 同时改善。但 fixed-world joint 和
vertex 没有满足严格 non-regression，因此不能把 aggregate 中其它收益用于掩盖失败项。

### 6.3 Boundary runtime

| Chain/cut | Matched | Accepted | Shared applied | Applied angle | Observable improvement |
|---|---:|---:|---|---:|---:|
| 0/0 | 3 | 3 | True | 11.688° | 15.5% |
| 0/1 | 3 | 2 | True | 8.797° | 11.6% |
| 1/0 | 3 | 3 | True | 7.680° | 7.0% |
| 1/1 | 3 | 2 | True | 4.846° | 31.1% |
| 2/0 | 1 | 0 | False | - | - |
| 2/1 | 1 | 1 | True | 11.274° | 45.9% |

总计 BRTC accepted `11/14`，5/6 个 boundary 动作，11/11 个 accepted boundary people
应用；post-shot person-frame 传播率 `68.75%`。第二 cut 的 6/7 tracks 读取到非零 inherited
orientation。

### 6.4 安全与 harm

```text
rejected/unmatched exact B0 max change = 0
native root max delta versus v1        = 0
camera max delta                       = 0
SO(3) orthogonality max error          = 3.33e-16
SO(3) determinant max error            = 4.44e-16
```

在 post-shot 80 个 person-frames 上，相对 v1：

| Error | Mean delta | Improve rate | Harm >1cm | Harm >5cm | Max harm |
|---|---:|---:|---:|---:|---:|
| Fixed root | +0.096 mm | 32.5% | 0.0% | 0.0% | 2.129 mm |
| Fixed joint | +0.268 mm | 33.8% | 16.2% | 0.0% | 23.174 mm |
| Fixed vertex | +0.009 mm | 28.7% | 13.8% | 0.0% | 28.880 mm |

没有 `>5 cm` catastrophic harm，但 mean joint/vertex 仍是正向退化，不能判为通过。

### 6.5 Native root 与 mapped pelvis caveat

Runtime 保证的是 native Human3R `person['root']` exact。Ego 的 `fixed root/pair/root Accel`
则由 SMPL-X→SMPL mapped vertices 回归 pelvis；绕 native root 旋转 vertices 后，该 mapped
pelvis 会轻微移动。所以 `fixed root +0.063 mm` 和 pair proxy 的变化不表示存储 root 被改。

不过这一 caveat 不能解释掉 `fixed joint +0.177 mm`；joint 是本方法必须真正面对的失败项。

## 7. 与 individual Kabsch 的对比

同一 frozen 参数下，individual Kabsch 已得到：

```text
Ego W/WA             = 312.769 / 200.029 mm
fixed joint/vertex   = 383.933 / 383.791 mm
```

shared Kabsch 为：

```text
Ego W/WA             = 312.885 / 200.587 mm
fixed joint/vertex   = 384.906 / 385.244 mm
```

shared 策略在 held-out 三组保持稳定小收益，但 Ego 没有超过 individual。多人共用 rotation
确实更保守，却也会把不同人的真实 orientation residual 平均掉；当前数据不支持“共享 SO(3)
比 individual SO(3) 更适合做默认动作”。

## 8. 最终决策

通过项：

- 机制和 blind held-out 三组全部通过；
- root/camera/rejected/unmatched runtime invariants 全部通过；
- W、WA、pelvis MPJPE、MPVPE、pair proxy、root/joint Accel 改善；
- `>5 cm` joint/vertex harm 为零。

失败项：

- Ego fixed-world joint `+0.177 mm`；
- Ego fixed-world vertex `+0.006 mm`。

最终严格状态：

```text
NO_GO_SHARED_ORIENTATION_KABSCH_EGOHUMANS
```

不再用 Ego 反调 angle/fraction/gate。该实验应作为“shared rigid orientation 有跨数据空间
价值，但不足以替代 per-person orientation 与后续时序/identity 证据”的归档结果。

## 9. 产物与复现

独立单测覆盖：共同 rotation/25°×0.5 bound、root exact、rejected exact B0、unmatched exact
B0、camera 不变。连同 frozen BRTC、strict FAGD、angular-safe 相关回归测试：

```text
16 passed
```

```text
versions/v14/b0_person_triangulation_shared_orientation_kabsch.py
versions/v14/tests/test_b0_person_triangulation_shared_orientation_kabsch.py
versions/v14/probe_brtc_shared_orientation_kabsch.py
versions/v14/eval_brtc_shared_orientation_kabsch_egohumans.py

output/v14/fine_alignment_research/brtc_shared_orientation_kabsch/
  DEV_MECHANISM.json
  DEV_MECHANISM.md
  HELDOUT_RESULTS.json
  HELDOUT_RESULTS.md
  egohumans/report.json
  egohumans/README.md
```

复现：

```bash
.venv/bin/python versions/v14/probe_brtc_shared_orientation_kabsch.py --phase dev
.venv/bin/python versions/v14/probe_brtc_shared_orientation_kabsch.py --phase validate
.venv/bin/python versions/v14/eval_brtc_shared_orientation_kabsch_egohumans.py
```
