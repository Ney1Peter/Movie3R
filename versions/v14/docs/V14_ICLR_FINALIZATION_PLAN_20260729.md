# Movie3R V14：面向 ICLR 投稿的收敛与执行计划

日期：2026-07-29

文档性质：基于当前代码、已有 V9/V12/V13/V14 实验和真实失败结果制定的执行计划。
目标不是继续扩大系统，而是用最少的方法变量验证一个清晰、可证伪的 scientific
insight，并形成可投稿的完整实验闭环。

---

## 0. 总体判断

用户给出的 `Movie3R V14 ICLR Finalization Plan` 方向正确，建议采用，但不能原样执行。
它正确抓住了当前最重要的四件事：

1. 论文主线应收敛到 state-gauge decoupling；
2. 当前第一风险是多人 refinement 破坏较强 `B0`；
3. 必须证明 `B0-before-WHO`、multi-cut 和 no-cut parity；
4. 不能再增加复杂 matcher、top-K、learned fusion 或大型 memory。

需要修正的地方：

1. **当前 `B0` checkpoint 不是最终模型。** 它由正式 V9 权重初始化，只在一个
   AvatarReX event 上做 V14.1 微调。现有 cross-dataset probe 很有价值，但 ICLR 最终
   claim 必须补 broad training 和真正 held-out evaluation。
2. **`B_final = DeltaB * B0` 不能直接用于 rotation-only 消融。** 当前 Boundary 使用
   camera-to-world 左乘约定，左 residual 的 rotation 会同时旋转 `B0` translation。
   必须采用右不变 residual或显式 `(R,t)` 分量锁定，保证消融含义准确。
3. **`Final >= B0` 不能靠单个 mean 或主观补偿判断。** 必须预先冻结 camera
   non-inferiority 约束，再在安全集合中比较 human/layout；不能看到结果后临时说
   “human 变好足以补偿 camera 变差”。
4. **`B0-before-WHO` 的 claim 需要收窄。** 现有结果证明 `B0` 使 root/torso geometry
   重新可比较，不证明所有 identity features 都没有用。DINO/native token 应作为正交
   control，而不是被错误解释成 coordinate gauge 的受害者。
5. **主 benchmark 必须包含 MultiHuman。** AvatarReX、THuman、MVHuman 首先是训练和
   controlled-geometry 数据；`three/dance/box` 与 EgoHumans 才是当前多人结论的关键
   development/frozen evaluation。
6. **当前 probe runner 语义因果，但不是最终集成 runtime。** 它会重放固定 pre-cut
   frames 并分别运行模型。multi-cut 和 runtime claim 前需要显式 state snapshot、
   Boundary composition 和单一提交路径。

结论：该计划经过上述修正后，适合作为下一阶段唯一主线。

---

## 1. 投稿问题与唯一核心假设

### 1.1 问题定义

Camera cut 对 streaming human-scene reconstruction 产生两个正交失败：

```text
State contamination:
old recurrent state continues into a new shot

World gauge discontinuity:
a clean hard-reset shot has an independent local coordinate system
```

因此：

```text
Camera cut = state transition + gauge transition
```

### 1.2 核心假设

> State continuity and world continuity should be decoupled. A streaming model
> should commit a clean reset trajectory, while using old state only in a
> non-committing transaction to estimate an explicit world-gauge bridge.

中文：

> 状态连续性和世界坐标连续性是两个不同问题。新 shot 只提交一条干净的 reset
> trajectory；旧状态只在一次不提交的 shadow transaction 中用于估计显式 world bridge。

### 1.3 核心 slogan

```text
Decouple state continuity from world continuity.
```

### 1.4 论文不再主张

- 一个通用多人 tracking 系统；
- 一个新的 appearance Re-ID 网络；
- 一个更复杂的 camera alignment optimizer；
- 一个 learned multi-human fusion network；
- 一个完整未来视频优化方法。

---

## 2. 最终主方法的范围

### 2.1 Main-paper core

主文只保留四个概念模块：

```text
Module A: Causal Shadow Transaction
    old state read-only
    first-post-cut correction once
    discard all shadow state

Module B: Explicit Coarse Boundary B0
    B0 = C_shadow @ inverse(C_raw)

Module C: B0-guided WHO
    compare cross-shot humans only after coarse gauge normalization

Module D: ONE Shared Boundary
    camera, pointmap and every human use the same fixed shot transform
```

### 2.2 System components，不能包装成核心创新

- externally supplied cut event 或标准 causal shot detector；
- Hungarian assignment；
- simple mutual/margin abstention；
- 人数不足 fallback；
- visualization/export；
- engineering state snapshot；
- standard SO(3) mean。

### 2.3 从主方法删除

以下内容停止开发，不进入当前投稿主线：

- Top-K WHO-WHERE joint search；
- learned identity adapter；
- learned fusion weight；
- appearance feature fusion 作为主方法；
- large persistent identity memory；
- multi-round assignment/refinement；
- DA3、VGGT、scale refinement；
- global BA 或 future-frame optimization。

现有证据支持这一决定：

```text
Phase 4 / three:
accepted precision = 100%
multi activation coverage = 7.62%
overall precision-first composite = 3.882

Phase 5 / three:
Top-6 GT assignment recall = 92.22%
identity-top1 composite/catastrophic = 1.598 / 0.267
selected joint scorer composite/catastrophic = 1.855 / 0.289
```

复杂 identity state/search 没有形成端到端收益，继续投入会削弱论文主线。

### 2.4 多人 refinement 的条件性地位

多人 refinement 暂时保留为一个严格 go/no-go 模块：

```text
if B0-centered multi refinement passes camera safety and improves humans:
    include as main-method refinement
else:
    final Boundary = B0
    move GT-ID multi-human consensus to analysis/appendix
```

不能因为已经投入多人实验，就强行把退化模块留在最终方法里。

---

## 3. 当前证据基线

所有后续计划必须从以下事实出发。

### 3.1 已成立

1. V14.1 shadow correction 可以从第一张 post-cut frame 产生有用 `B0`；
2. shadow state 可丢弃，raw hard-reset state 可作为后续唯一 trajectory；
3. strict GT-ID 下 uniform multi-human consensus 优于可部署 single-human anchor；
4. `B0` 显著改善匿名 root+torso matching：

| Sequence | Eligible | Direct all-correct | `B0` all-correct |
|---|---:|---:|---:|
| `three` | 41 | 46.3% | 100.0% |
| `dance` | 61 | 65.6% | 100.0% |
| `box` | 78 | 65.4% | 98.7% |

5. 24-frame 2/3-person probe 已跑通因果 segment propagation。

### 3.2 当前最大失败

| Case | `B0` camera error | Current final multi error |
|---|---:|---:|
| `dance_t0600_c1_c4_k1` | 0.427 m / 2.24 deg | 0.568 m / 4.93 deg |
| `box_t0470_c1_c4_k8` | 0.390 m / 2.81 deg | 0.466 m / 5.40 deg |
| `three_t0900_c3_c4_k0` | 0.113 m / 3.85 deg | 0.408 m / 2.26 deg |

当前 uniform multi 能改善部分 rotation/layout，但旧 root-anchor translation 会继承
Human3R root-depth bias，并无条件覆盖较准 `B0`。

### 3.3 尚未覆盖

```text
automatic cut detector
broadly trained V14.1 checkpoint
real multi-cut state composition
no-cut exact parity benchmark
variable visibility/dustbin end-to-end path
formal runtime/memory report
full frozen evaluation
```

变量可见性排除数量必须一直公开：

```text
three: 22/63
dance: 29/90
box: 12/90
```

---

## 4. 预注册式方法冻结规则

### 4.1 一次只改一个模块

后续实验分三条冻结轴：

```text
Axis 1: shadow/B0 model
Axis 2: WHO matcher
Axis 3: Boundary refinement
```

任何正式 ablation 一次只能改变一个轴。禁止在同一个结果中同时更换 checkpoint、
identity cost 和 fusion rule。

### 4.2 数据冻结

```text
Development:
    V14.1 single/ten-event training diagnostics
    MultiHuman three for thresholds and residual bounds

Frozen evaluation:
    MultiHuman dance
    MultiHuman box
    EgoHumans 001_legoassemble
    held-out AvatarReX/THuman/MVHuman captures
```

`three` 已被反复用于 Phase 1-5，不得作为唯一 final result。

### 4.3 GT 使用边界

| 信息 | Training target | Deployment inference | Evaluation |
|---|---:|---:|---:|
| RGB | 是 | 是 | 是 |
| cut index | 主实验 oracle trigger | 计划外部 detector | 是 |
| GT camera | 是 | 否 | 是 |
| GT identity | 否，除受控 identity probe | 否 | 是 |
| GT SMPL/SMPL-X | loss 可用 | 否 | 是 |
| GT bbox/keypoints | 否 | 否 | 是，仅 audit |
| future frames | loss 可选但不能作输入 | 否 | 否 |

### 4.4 选择规则

所有 threshold、residual bound 和 method selection 只允许读取 development set。冻结后
一次性运行 `dance/box/EgoHumans/held-out captures`，不回调参数。

---

## 5. Phase 0：冻结可复现基线

目的：在修改方法前把当前 `B0`、WHO 和 multi Boundary 结果变成一个可复现 baseline。

### 5.1 冻结 checkpoint provenance

当前 checkpoint：

```text
/dev/shm/movie3r_v14_1/
v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth
```

必须先复制到非易失 `checkpoints/v14/`，同时保存：

- checkpoint SHA-256；
- resolved model config；
- optimizer/training epoch；
- initialization chain；
- train manifest；
- RGB preprocessing；
- FP32/AMP setting。

在完成复制前，不能把 `/dev/shm` 文件当正式基线。

### 5.2 冻结 case manifest

把当前：

```text
three 41 eligible cuts
dance 61 eligible cuts
box 78 eligible cuts
three/dance/box long 24-frame probes
```

写成 versioned JSONL manifest。每个 case 固定：

```text
sequence
timestamp
source/target camera
temporal offset k
pre/post frame count
visibility status
GT used only for evaluation fields
```

### 5.3 冻结报告 schema

每个 method 必须输出统一 JSON：

```text
camera T/R/composite
median/P90/P95/catastrophic
human root/joint/vertex/layout
identity assignment/all-correct/margin
coverage/exclusion reason
runtime/memory
checkpoint/config hash
```

### 5.4 当前代码入口

```text
versions/v14/run_v14_autoid_visual_ladder.py
versions/v14/probe_b0_identity_matching.py
versions/v14/run_v14_2_single_sequence.py
versions/v14/run_v14_2_multihuman_sequence.py
src/dust3r/v14_outputs.py
```

### 5.5 Phase 0 退出条件

- 同一 manifest 重跑数值一致；
- no GT candidate generation audit 通过；
- checkpoint/config hash 被写入报告；
- 所有 excluded cuts 仍出现在 coverage table，而不是被静默丢弃。

---

## 6. Phase 1：B0-Centered Refinement Go/No-Go

这是最高优先级方法实验。

### 6.1 技术修正：采用右不变 residual

当前 Boundary 采用：

\[
C^W=B C^{local}.
\]

如果直接使用左 residual：

\[
B^*=\Delta B B_0,
\]

那么即使 \(\Delta B=[\Delta R,0]\)，也有：

\[
t^*=\Delta R t_0,
\]

rotation-only 会偷偷修改 `B0` translation，不适合作为干净消融。

本阶段改用右不变 residual：

\[
E_i=B_0^{-1}B_i.
\]

若：

\[
B_0=[R_0,t_0],\qquad B_i=[R_i,t_i],
\]

则：

\[
\Delta R_i=R_0^\top R_i,
\]

\[
\Delta u_i=R_0^\top(t_i-t_0).
\]

等权 residual consensus：

\[
\Delta R=\operatorname{SO3Mean}(\Delta R_1,\ldots,\Delta R_N),
\]

\[
\Delta u=\frac1N\sum_i\Delta u_i.
\]

最终：

\[
B^*=B_0[\Delta R,\Delta u]
=[R_0\Delta R,\ t_0+R_0\Delta u].
\]

这样：

- rotation-only 设置 \(\Delta u=0\)，Boundary translation 保持 \(t_0\)；
- translation-only 设置 \(\Delta R=I\)，Boundary rotation 保持 \(R_0\)；
- full residual 同时允许两者变化。

注意：即使 Boundary translation 保持，rotation 仍可能改变非原点 camera/point 的 world
position，因此所有 camera translation 指标仍必须实际测量。

### 6.2 Bound 规则

Rotation 使用 axis-angle clip：

\[
\omega=\log(\Delta R),\qquad
\omega'=\omega\min(1,\alpha/\|\omega\|).
\]

Translation 使用 norm clip：

\[
u'=\Delta u\min(1,\beta/\|\Delta u\|).
\]

只在 `three` development 上比较一个小网格：

```text
alpha in {5, 10, 20} degrees
beta  in {0.05, 0.10, 0.25} meters
```

`20 degree` 与现有 V16 bound 对齐。不得为每个 sequence、camera pair 或人物人数选择
不同 bound。

### 6.3 必须比较的方法

```text
A0. Hard Reset only
A1. B0 only
A2. Current full Phase-2 uniform multi Boundary
A3. B0 + rotation-only residual
A4. B0 + translation-only residual
A5. B0 + bounded full residual
A6. B0 + unbounded full residual
A7. GT-ID versions of A2-A6
```

Automatic-ID 与 GT-ID 结果必须分开。先用 GT-ID 隔离 WHERE，再用 frozen automatic ID
测试端到端；否则 ID error 会混淆 residual 是否正确。

### 6.4 代码实现

新增独立模块，不继续把公式写进实验脚本：

```text
versions/v14/b0_residual_consensus.py
```

建议接口：

```python
def residual_candidates(b0, full_candidates): ...
def bounded_uniform_residual(b0, candidates, mode, rot_bound_deg, trans_bound_m): ...
```

复用：

```text
versions/v13/gt_id_consensus.py::so3_mean
src/dust3r/v14_outputs.py::apply_boundary_to_prediction
```

新增测试：

```text
rotation-only keeps Boundary t0 exactly
translation-only keeps Boundary R0 exactly
zero residual returns B0 exactly
unbounded residual reconstructs uniform full candidate under consistent inputs
all humans/camera/pointmap receive the same B_final
candidate order permutation does not change result
```

### 6.5 预先冻结的选择标准

不能用一个任意加权 camera-human 总分选择。采用 safety-constrained lexicographic rule：

第一层，camera non-inferiority 相对 `B0 only`：

```text
mean/median translation degradation <= max(2%, 0.02 m)
mean/median rotation degradation    <= max(2%, 0.5 deg)
P90/P95 composite degradation       <= 2%
new catastrophic failures           = 0
```

第二层，在通过第一层的方法中选择：

```text
lowest human root error
then lowest pairwise relative-vector/layout error
then lowest camera composite
```

同时使用 paired bootstrap 95% CI 报告差异。最终论文不能只写“human improvement
compensates camera loss”，除非预先定义并固定了统一 utility；本计划不采用该做法。

### 6.6 Go/No-Go

```text
GO:
bounded residual passes camera safety
and human root/layout improves on development and frozen evaluation

NO-GO:
no residual variant passes
or improvement only appears with GT/oracle per-case selection
or dance/box tails regress
```

NO-GO 后最终主方法直接使用 `B0`。多人 consensus 保留为 GT-ID oracle 分析，不再阻塞
论文。

---

## 7. Phase 2：把 V14.1 从单事件探针变成正式 B0 模型

Phase 1 先用当前 checkpoint 快速决定 refinement 是否值得保留；随后必须训练正式
shadow/B0 checkpoint，再冻结并重跑 Phase 1 选择。

### 7.1 保持架构不变

冻结当前 V9-parity event-only architecture：

```text
semantic + alignment + momentum correct tokens
full decoder interaction
pose latent correction
human latent correction
event-only pose/human head LoRA
shadow state never committed
```

不在大训练时重新尝试 token architecture、gate 或 matcher。

### 7.2 数据扩展顺序

```text
Stage 1: corrected 10-event pilot
Stage 2: broader AvatarReX + THuman + MVHuman training
Stage 3: held-out subject/capture/camera-pair validation
```

明确排除此前指定的 `/data/wangzheng/iJCV-CODE/data/Training/asit`，除非其数据契约
另行审计通过。

### 7.3 数据划分

至少满足：

```text
subject-disjoint
capture-disjoint
camera-pair-disjoint validation
```

训练 manifest、validation manifest 和 final test manifest 必须在训练前落盘。不能从
`dance/box/EgoHumans` 反向选择 checkpoint。

### 7.4 Loss 冻结

保留当前已验证的：

```text
camera translation/rotation
human translation
latent residual regularization
self pointmap preservation
shared pointmap transform consistency
human parameter preservation
```

不新增 learned identity loss 或 multi-human fusion loss。训练目标仍然是 identity-free
coarse `B0`。

### 7.5 Checkpoint 选择

validation score 使用：

```text
camera Boundary translation
camera Boundary rotation
shared pointmap consistency
human local-parameter preservation
```

必须同时看 P90，不按单事件 minimum training loss 选模型。

### 7.6 Phase 2 退出条件

- corrected 10-event 大多数事件同向改善；
- broad validation 不依赖单 subject/camera pair；
- `B0` 在 held-out capture 上优于 Hard Reset；
- no-cut event-off path 保持 exact parity；
- checkpoint、config、manifest、hash 全部归档。

如果 broad model 明显不如当前单事件 checkpoint 的 cross-data probe，需要先排查训练
contract，不允许回到单样本权重直接投稿。

---

## 8. Phase 3A：证明 State-Gauge Decomposition

这是论文最关键的问题验证，不是普通 ablation。

### 8.1 四个受控路径

| Variant | Recurrent state | Gauge handling | 目的 |
|---|---|---|---|
| Continuous | old state continues | none | 展示 state contamination |
| Hard Reset | fresh state | none | clean state but broken world |
| Corrected Commit / V9 | corrected old-state trajectory | implicit | 对照持续 latent correction |
| V14 Shadow | fresh raw state committed | explicit fixed `B0` | state/gauge decoupling |

多人 refinement 先关闭，避免混淆核心 claim。

### 8.2 如何分别测两个问题

不能直接用同一个 world camera error 同时声称两种 failure。

**State contamination 指标：**

- 对 continuous 和 reset local reconstruction 做 GT/oracle rigid gauge normalization 后，
  比较 local pointmap shape、camera relative motion、human local pose/shape；
- 或使用 gauge-invariant pairwise distances、relative camera increments；
- 比较 contamination 是否传播到 post-cut 后续 1/2/4/8 帧。

**Gauge discontinuity 指标：**

- 在不做 oracle alignment 的前提下，测 Hard Reset 的 absolute camera/root/world layout；
- 比较 `B0` 是否恢复 persistent world；
- 报告 first-post frame 和整段固定传播。

### 8.3 必须包含的控制

```text
same-camera pseudo reset
small-view cut
wide-view cut >=120 deg
k = 0, 1, 2, 4, 8
```

same-camera pseudo reset 用于确认“reset 本身”不会被错误解释成跨镜方法收益。

### 8.4 成功条件

1. old-state continuous 在 cut 后出现可测的 local contamination；
2. fresh reset 显著改善 local purity，但 absolute world gauge 断裂；
3. V14 同时接近 reset 的 local purity和 shadow/B0 的 world continuity；
4. 结果在多个 sequence 和 camera span 分组中一致。

如果无法分别测出两个失败，论文的核心 scientific claim 不成立，应该先重做问题定义
实验，而不是继续优化最终数字。

---

## 9. Phase 3B：Shadow 设计消融

### 9.1 变体

```text
S0. No shadow: Hard Reset only
S1. Shadow output, corrected state committed
S2. Shadow output, state discarded, no explicit propagation
S3. Shadow discarded + fixed explicit B0
S4. Every-frame V9 correction
S5. First-frame-only V14 shadow correction
```

### 9.2 公平性

- 使用相同 V9/V14 backbone initialization；
- 相同输入和 preprocessing；
- 相同 cut indices；
- 相同后续 segment length；
- correction 参数量和 LoRA 开关明确报告；
- 不在 S3/S5 中加入 automatic ID 或 multi-human refinement。

### 9.3 要回答的问题

1. shadow state 为什么不能 commit？
2. 第一帧 correction 是否足以产生整段 bridge？
3. 显式固定 `B0` 是否优于传播 latent/corrected state？
4. every-frame correction 是否增加 drift、成本或正常帧偏差？

---

## 10. Phase 3C：证明 `B0-before-WHO`

### 10.1 正确 claim

目标不是证明 DINO/token 无用，而是：

> Geometry-based identity association is ill-posed across independent shot
> gauges. An identity-free coarse `B0` restores a common coordinate system and
> makes root/torso/layout informative again.

### 10.2 必须比较

```text
W0. Detection-index control
W1. Human3R native/refined token matching
W2. Frozen DINO appearance crop matching
W3. Direct raw root+torso geometry
W4. B0-aligned root+torso geometry
W5. B0-aligned root+torso + frozen DINO control (appendix)
W6. GT identity oracle
```

W1/W2 可以复用 V13 Phase 3/4 已有 feature cache，不训练新网络。W5 只用于判断
appearance 与 geometry 是否互补，不作为新主模块。

### 10.3 当前可直接复用的证据

```text
output/v14/b0_identity_matching/
output/v14/b0_identity_matching_extended/dance/
output/v14/b0_identity_matching_extended/box/
output/v13/phase3_identity/
output/v13/phase4_identity/
```

### 10.4 指标

主指标：

```text
cut-level all-matches-correct
assignment accuracy
best-vs-second margin
ID switches at cut
```

如果使用简单 abstention gate，再报告：

```text
accepted precision
wrong-accept rate
coverage
risk-coverage
```

当前 forced Hungarian 没有“accepted/rejected”语义，不能把 assignment accuracy 误写成
accepted precision。

### 10.5 分组

```text
2 people / 3 people
k = 0/1/2/4/8
camera span <60 / 60-120 / >=120 deg
same visibility / variable visibility
```

same-visibility 是核心可比性 probe；variable visibility 必须单独报告 coverage，不能继续
从总表中消失。

### 10.6 简单 matcher 的上限

主方法只允许：

```text
B0 root+torso cost
Hungarian
optional mutual + one global margin threshold + dustbin
```

不再实现 top-K、learned appearance fusion 或 persistent neural identity state。若简单
matcher 在 variable visibility 上不足，多人 automatic extension 降为受控实验，不得拖累
state-gauge core paper。

---

## 11. Phase 4：真实 Multi-Cut Streaming Runtime

### 11.1 当前代码缺口

现有 runner 通过固定 pre-frame replay 实现因果 probe，但未形成一个持续运行的：

```text
single raw recurrent state
+ state snapshot
+ temporary shadow transaction
+ composed world Boundary
```

因此 multi-cut 前需要做一次收敛型 runtime 重构，不增加新算法。

### 11.2 推荐接口

新增：

```text
versions/v14/runtime.py
versions/v14/run_v14_multicut_stream.py
```

核心对象：

```python
class V14StreamingState:
    raw_recurrent_state
    current_world_boundary
    shot_index
    optional_simple_track_map
```

cut transaction：

```python
snapshot = detach_clone(raw_recurrent_state)
shadow_output = forward_event_from_state(snapshot, commit=False)
raw_output, raw_recurrent_state = forward_event_from_fresh_state(commit=True)
delta_b0 = camera(shadow_output) @ inverse(camera(raw_output))
world_b0 = previous_world_boundary @ delta_b0
```

需要明确 shadow camera 是上一 local gauge 还是已经 world-aligned，防止
`previous_world_boundary` 重复相乘。

### 11.3 Boundary composition test

若 \(G_{j-1}\) 将上一 shot local 映射到 world，\(\Delta B_j\) 将当前 local 映射到上一
local，则：

\[
G_j=G_{j-1}\Delta B_j.
\]

必须先用合成矩阵单元测试，再跑真实数据。

### 11.4 必测 stream

```text
A -> B
A -> B -> C
A -> B -> A
wide -> narrow -> wide view
2/3 people when available
```

identity-free `B0` 先完成 multi-cut；automatic WHO/multi refinement 后加，不能两者同时
调试。

### 11.5 指标

```text
per-cut Boundary error
cumulative camera drift after cut 1/2/3
return-to-A loop error
human world-root drift
pairwise layout drift
ID consistency, only when WHO enabled
```

### 11.6 因果审计

- 截断每个 cut 后 future frames，当前输出 hash 不变；
- shadow commit count 恒为 0；
- raw state 每帧只提交一次；
- 历史输出不回写；
- memory 与 cut 数量线性无关。

---

## 12. Phase 5：No-Cut Parity 与 False-Trigger Safety

### 12.1 正确标准

当前 event-only routing 的目标应是数值 parity，不只是 metric 差异小于 1-2%。

当所有 `shot_label=0`：

```text
correct tokens off
latent correction off
event-only LoRA off
Boundary solver off
```

V14 raw output 应与相同 base checkpoint 的 Human3R path 在 FP32 数值容差内一致。

### 12.2 单元测试

新增 `tests/test_v14_no_cut_parity.py`：

```text
camera pose max abs diff <= 1e-5 or measured deterministic tolerance
pointmap max/mean diff <= measured deterministic tolerance
SMPL parameters and smpl_transl parity
raw recurrent state parity
no shadow allocation
```

容差先在确定性 FP32 重跑中测定，随后冻结；不能随结果扩大。

### 12.3 Dataset-level no-cut benchmark

从已有连续序列选择：

```text
H36M
AIST/AIST++（只作为 no-cut evaluation，不进入当前 V14 shadow training）
AvatarReX held-out continuous views
MultiHuman fixed-camera continuous clips
```

报告 Human3R 与 V14 的 camera relative motion、pointmap、human root/joints/vertices。

### 12.4 False trigger

主方法可以使用 oracle cut 隔离 scientific claim，但 appendix 必须接一个标准 causal shot
detector，报告：

- detector precision/recall；
- false trigger 后恢复；
- missed cut 后污染；
- end-to-end result。

detector 不训练 Boundary，不作为核心贡献。

---

## 13. Baseline 计划

### 13.1 Streaming reconstruction baselines

```text
R0. Original Human3R continuous across cut
R1. Original Human3R Hard Reset
R2. Formal V9 corrected-state/continuous correction
R3. V14 raw-reset + shadow B0
R4. V14 final, only if bounded refinement passes
```

### 13.2 Identity/anchor baselines

```text
I0. Detection index
I1. Highest-confidence single human
I2. Human3R native/refined token
I3. Frozen DINO appearance
I4. Direct root+torso geometry
I5. B0-aligned root+torso geometry
I6. GT-ID uniform multi-human oracle
```

### 13.3 Identity-free alignment baselines

至少实现两个明确 baseline，不能只写模糊的“feature matching SE3”：

**P0 Point-cloud registration**

```text
pre-cut accumulated Human3R point cloud
first post-cut raw point cloud
FPFH/RANSAC or another fixed global initializer
fixed-budget ICP refinement
```

使用成熟库，固定预算，不能读取 GT initial pose。

**P1 Frozen feature correspondence SE(3)**

```text
frozen DINO patch correspondence
+ Human3R predicted 3D points/depth
+ RANSAC + Umeyama/SE3 solve
```

若预测 3D 对应不足，必须报告 invalid rate，不能只保留成功 cases。

### 13.4 Oracle 只作为上界

```text
GT camera Boundary
GT-ID uniform multi-human
Oracle best candidate
```

必须标为 Oracle，不得与可部署方法混在主 ranking 中。

### 13.5 外部文献 baseline

正式写作前需单独完成一次 related-work/baseline audit，至少覆盖：

- online monocular scene reconstruction across cuts；
- causal camera relocalization；
- human-aware camera/world alignment；
- online multi-human tracking。

如果没有同设定方法，应清楚解释 protocol mismatch，而不是省略外部 baseline。

---

## 14. 数据与 benchmark 角色

### 14.1 V14 shadow/B0 training

```text
AvatarReX
THuman
MVHuman100/200
```

作用：训练/验证 first-post-cut coarse gauge。必须 subject/capture/camera-pair disjoint。

### 14.2 MultiHuman development

```text
three
```

作用：选择 residual bound、simple matcher margin 和报告 schema。不能作为唯一 final
泛化结论。

### 14.3 MultiHuman frozen evaluation

```text
dance
box
```

作用：2-person、motion、occlusion、wide view、`k=0/1/2/4/8`。规则冻结后一次性运行。

### 14.4 Cross-dataset robustness

```text
EgoHumans 001_legoassemble
```

作用：fisheye、3->1->3、visibility change、cross-data robustness。SMPL 与 Human3R
SMPL-X 不直接比较完整 vertices，只报告 common joints、root、layout、camera。

### 14.5 No-cut parity

```text
H36M
AIST/AIST++
held-out continuous AvatarReX/MultiHuman clips
```

这些数据不与 multi-shot 主表强行合并。

---

## 15. 指标与统计协议

### 15.1 Camera

```text
translation error
rotation geodesic error
existing frozen composite definition
mean / median / P90 / P95
catastrophic rate
```

catastrophic threshold 必须在看 frozen evaluation 前写入配置。

### 15.2 Human

```text
world root
joints
SMPL-X vertices where compatible
pairwise relative distance
pairwise relative vector
```

### 15.3 Identity

```text
assignment accuracy
cut-level all-matches-correct
ID switches
best-vs-second margin
accepted precision / wrong accept / coverage, only for methods with abstention
```

### 15.4 State/gauge

```text
gauge-normalized local reconstruction error
absolute Boundary error
segment propagation drift
multi-cut cumulative drift
shadow/raw state equality audit
```

### 15.5 Streaming

```text
normal-frame FPS
cut latency: raw, shadow, WHO, refinement separately
peak GPU memory
persistent state memory
number of evaluated hypotheses = 1 in final core
```

### 15.6 统计

- 每个 sequence 单独报告；
- paired cut-level bootstrap 95% CI；
- 同时报告 valid/invalid/excluded count；
- 主表使用所有 eligible cases，不挑 viewer cases；
- viewer 包含 best、median、P90 和 failure；
- development 与 frozen evaluation 使用不同表头明确标记。

---

## 16. 论文核心 Ablation Tables

### 16.1 State-Gauge decomposition

| Method | Committed state | Gauge | Local purity | World camera | Human world |
|---|---|---|---:|---:|---:|
| Continuous | old | implicit/broken | TBD | TBD | TBD |
| Hard Reset | fresh | none | TBD | TBD | TBD |
| Corrected Commit | corrected old | implicit | TBD | TBD | TBD |
| V14 Shadow+B0 | fresh | explicit fixed | TBD | TBD | TBD |

### 16.2 `B0` necessity for WHO

| Matcher | Gauge normalization | All-correct | Margin | Wrong accept | Coverage |
|---|---|---:|---:|---:|---:|
| Native token | none | TBD | TBD | TBD | TBD |
| DINO | gauge independent | TBD | TBD | TBD | TBD |
| Root+torso | none | TBD | TBD | TBD | TBD |
| Root+torso | `B0` | TBD | TBD | TBD | TBD |

### 16.3 Shadow design

| Variant | One-shot | Shadow commit | Explicit `B0` | Camera | Human | FPS |
|---|---:|---:|---:|---:|---:|---:|
| No shadow | - | - | - | TBD | TBD | TBD |
| V9 every-frame | no | yes | no | TBD | TBD | TBD |
| One-shot commit | yes | yes | no | TBD | TBD | TBD |
| One-shot discard | yes | no | no | TBD | TBD | TBD |
| V14 | yes | no | yes | TBD | TBD | TBD |

### 16.4 `B0`-centered refinement

| Method | Keeps `R0` | Keeps `t0` | Camera T/R | P90/P95 | Human root/layout |
|---|---:|---:|---:|---:|---:|
| `B0` | yes | yes | TBD | TBD | TBD |
| Full old multi | no | no | TBD | TBD | TBD |
| Rotation residual | no | yes | TBD | TBD | TBD |
| Translation residual | yes | no | TBD | TBD | TBD |
| Bounded full | no | no | TBD | TBD | TBD |

---

## 17. 代码收敛计划

### 17.1 保留

```text
src/dust3r/model.py
src/dust3r/v14_outputs.py
versions/v14/run_v14_autoid_visual_ladder.py
versions/v14/probe_b0_identity_matching.py
versions/v13/gt_id_consensus.py
```

### 17.2 新增最小模块

```text
versions/v14/b0_residual_consensus.py
versions/v14/runtime.py
versions/v14/run_v14_multicut_stream.py
versions/v14/eval_v14_final.py
tests/test_v14_b0_residual.py
tests/test_v14_no_cut_parity.py
tests/test_v14_multicut_composition.py
tests/test_v14_shadow_noncommit.py
```

当前已有测试可继续作为基础：

```text
tests/test_v14_1_event_routing.py
    shot_label survives deployable preprocessing
    event-off batch does not mutate source labels
    shadow geometry loss is event-only

tests/test_v14_segment_boundary.py
    B0 left-multiplication convention
    camera and world pointmap share one Boundary
    local SMPL parameters remain unchanged

tests/test_v14_b0_identity_matching.py
    coarse rotation can recover anonymous assignment
    prompt history restarts when human count changes
```

这些测试没有覆盖 no-cut exact parity、shadow state non-commit 和 multi-cut composition，
因此新增测试不是重复工作。

### 17.3 不新增

```text
identity_adapter.py
topk_search.py
learned_fusion.py
global_optimizer.py
large_memory_router.py
```

### 17.4 重构原则

- 方法公式从 runner 移到小型可测模块；
- `src/dust3r/model.py` 只保留模型内必要 routing；
- raw/shadow state ownership 在 runtime 层显式表达；
- evaluation 只读 frozen artifacts，不在运行时调 threshold；
- 所有 viewer 输出必须同时有 machine-readable JSON。

---

## 18. 严格执行顺序

### Stage 0：Reproducibility freeze

交付：

```text
non-volatile checkpoint
hash/config/manifest
unified report schema
current baseline rerun
```

失败则停止后续实验，先修复 contract。

### Stage 1：Quick B0-centered refinement decision

使用当前 frozen checkpoint 和 `three` development：

```text
implement right residual
run GT-ID WHERE isolation
run current automatic-ID path
apply pre-registered safety gate
```

交付：保留一个 bounded variant，或决定最终只用 `B0`。

### Stage 2：Formal V14.1 B0 training

```text
corrected ten-event pilot
broad AvatarReX/THuman/MVHuman training
held-out checkpoint selection
archive final model
```

交付：唯一 final B0 checkpoint。

### Stage 3：Re-freeze residual after formal checkpoint

只在 `three` 重新选择一次 residual bound，然后永久冻结。随后运行 `dance/box`，不得
根据结果回调。

### Stage 4：Core ablations

```text
state vs gauge problem validation
shadow commit/discard/one-shot/every-frame
B0-before-WHO
```

交付：论文三张核心 ablation tables。

### Stage 5：Integrated multi-cut runtime

先只做 identity-free `B0`，再接 frozen simple WHO 和通过 Gate 的 refinement。

交付：`A-B`、`A-B-C`、`A-B-A` 结果与 causal audit。

### Stage 6：No-cut parity and runtime

交付：exact parity unit test、no-cut dataset table、FPS/latency/memory。

### Stage 7：Final frozen benchmark

一次性运行：

```text
dance
box
EgoHumans 001_legoassemble
held-out AvatarReX/THuman/MVHuman
```

输出全部 baselines、ablations、coverage 和 failures。

### Stage 8：Paper writing

只有 Stage 7 完成后锁定标题、claim 和最终方法图。不能先写强 claim 再挑实验支持。

---

## 19. 投稿 Go/No-Go 标准

### 19.1 Scientific insight gate

必须分别证明：

```text
old state causes contamination
hard reset causes world-gauge discontinuity
V14 recovers world continuity without committing contaminated state
```

若无法分离两种 failure，不适合按当前主线投稿。

### 19.2 B0 gate

正式 broad-trained `B0` 必须在多个 held-out camera pairs/sequences 上优于 Hard Reset，
并在整段传播中不只改善第一帧。

### 19.3 Refinement gate

多人 refinement 只有满足 Phase 1 camera safety + human/layout gain 才进入主方法。否则
删除，不影响 state-gauge core。

### 19.4 WHO gate

`B0`-aligned geometry 必须在 wide-view 和 frozen sequences 上明显优于 direct geometry。
automatic multi-human 如果不能安全覆盖 variable visibility，则作为受控 extension，不可
称完整 tracking system。

### 19.5 Multi-cut gate

`A-B-C` 和 `A-B-A` 必须保持正确 Boundary composition、无 history rewrite、无未来帧，
累计 drift 明显优于 reset-only/continuous baselines。

### 19.6 No-cut/runtime gate

- no-cut path 数值 parity；
- normal-frame overhead 接近零或明确量化；
- cut latency 和 peak memory 可接受；
- shadow state 不提交。

### 19.7 Empirical completeness gate

- 至少一个 development 和两个 frozen MultiHuman sequences；
- cross-data EgoHumans；
- held-out training-domain captures；
- 外部/非历史 baseline；
- P90/P95/catastrophic/coverage；
- 失败案例和统计区间。

满足所有 gate 后，V14 才具备 ICLR main-track 方法论文的基本完整性。这仍不保证接收，
但能避免当前最明显的审稿拒绝理由。

---

## 20. 论文叙事与图表

### 20.1 标题方向

避免标题突出 multi-human tracking。建议围绕：

```text
Causal State-Gauge Decoupling for Streaming Human-Scene Reconstruction Across Camera Cuts
```

### 20.2 Abstract 主线

```text
Problem observation:
camera cuts create orthogonal state and gauge failures

Method:
clean raw reset + non-committing shadow transaction + explicit B0

Key ordering:
coarse WHERE before geometry-based WHO

Output contract:
one fixed shared Boundary for camera, scene and humans

Evidence:
problem isolation, frozen cross-sequence results, multi-cut streaming, parity/runtime
```

### 20.3 主文图

**Figure 1：Failure decomposition**

```text
continuous old state -> contaminated local reconstruction
hard reset -> clean but disconnected world
V14 -> clean state + recovered world
```

**Figure 2：Method**

```text
one raw committed trajectory
one temporary shadow transaction
B0 extraction
optional safe WHO/refinement
one fixed shared Boundary
```

**Figure 3：B0-before-WHO**

展示同一个 identity cost matrix 在 raw gauge 与 `B0` gauge 下的 permutation 变化。

**Figure 4：Multi-cut**

展示 `A-B-C` 世界轨迹和 `A-B-A` return error。

### 20.4 主表

1. Frozen end-to-end reconstruction；
2. State-gauge decomposition；
3. Shadow design；
4. `B0`-before-WHO；
5. Multi-cut + runtime；
6. `B0` refinement go/no-go（若保留）。

---

## 21. 最终简化流程

如果多人 refinement 通过：

```text
Camera cut
-> fresh raw Human3R state
-> one non-committing V14 shadow correction
-> explicit B0
-> B0-guided simple WHO
-> bounded uniform human residual
-> ONE fixed shared Boundary
-> stream the rest of the shot
```

如果多人 refinement 不通过：

```text
Camera cut
-> fresh raw Human3R state
-> one non-committing V14 shadow correction
-> explicit B0
-> ONE fixed shared Boundary
-> stream the rest of the shot
```

第二条仍然是完整、清晰且更容易 defend 的 state-gauge decoupling 方法。多人 GT-ID
结果和 `B0-before-WHO` 可以作为能力分析与未来 extension，而不是强行成为最终 solver。

---

## 22. 最终停止原则

下一阶段不以“实现更多模块”为进度，而以关闭以下问题为进度：

1. `B0` 是否是正式 broad model 而非单事件偶然结果？
2. final Boundary 是否至少不破坏 `B0`？
3. state contamination 与 gauge discontinuity 是否被独立证明？
4. shadow non-commit 和 one-shot 是否必要？
5. `B0-before-WHO` 是否在冻结数据上成立？
6. multi-cut persistent world 是否真实运行？
7. no-cut parity 和 runtime 是否满足 streaming claim？
8. baseline、tails、coverage 和 failures 是否完整？

在这些问题关闭前，禁止开启新的 identity adapter、top-K、fusion network 或 metric-depth
方向。

本阶段唯一目标：

> 用最小方法证明 camera cut 中 state continuity 与 world continuity 必须解耦，并证明
> 一次不提交的 latent shadow transaction 可以在严格流式系统中恢复显式、可长期传播的
> world gauge。
