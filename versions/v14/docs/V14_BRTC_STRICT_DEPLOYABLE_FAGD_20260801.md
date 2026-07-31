# V14 BRTC strict FAGD-0.9 可部署性终验

> 日期：2026-08-01
>
> 范围：只重放已有 CPU geometry cache；不重跑 Human3R，不使用 DA3/GPU，不读取未来帧；
> 不修改 completeness runtime、现有 evaluator、frozen BRTC-LC v1、B0 camera 或 V9。

## 1. 最终结论

本轮把此前作为空间候选的 Full-Accept Group-Only Damping（FAGD-0.9）实现成了独立、
严格回退的 runtime。要求只有在下面条件完全成立时才缩放共享 group translation：

```text
accepted_count == matched_count
    == max(len(pre_people), len(post_people)) > 0
```

否则直接返回 frozen BRTC-LC v1 的原始 corrected geometry，保持 bit-exact。

实验得到两个同时成立的结论：

1. 在人物集合明确不变的 `offset1 / dance / box` 上，strict FAGD 完整保留旧 FAGD 的
   root、joint、vertex 收益，并保持当前 boundary 的 pair distance/vector 与 BRTC v1
   bit-exact。这证明“共享 group action 略过激、individual residual 有价值”是稳定规律。
2. 该 strict gate 仍不能成为通用 variable-visibility runtime。人数发生变化的 49 个 case
   确实全部 exact-v1 回退，但 `2→2` 或 `1→1` 的等人数人物替换无法由 count/match/accept
   观察出来；14 个此类 case 中 13 个仍会触发 FAGD。

EgoHumans 同-forward 重放中，strict FAGD 改善 W、fixed-world root/joint/vertex 和两项
layout，但 WA 有 `0.0002 mm` 的数值级退化，world-root Accel 明显从 `116.014` 退化至
`118.040 mm/frame²`。因此最终判定是：

```text
NO_GO_STRICT_FAGD_GENERAL_VARIABLE_SAFE
NO_GO_STRICT_FAGD_DEPLOYABLE
```

它仍是有效的**受条件空间操作**，但当前不能替代 frozen BRTC-LC v1 成为默认精对齐主线。
只有上游能够可靠保证 visible identity set 没有等基数替换时，才可条件启用。

## 2. Runtime 原理与数据流

### 2.1 输入

每个 shot boundary 只读取当前在线可见信息：

```text
pre_camera, post_camera
pre_people, post_people
anonymous one-to-one matches
```

每个人至少包含：

```text
root:     [3]
joints:   [J, 3]
vertices: [V, 3]
```

GT identity 和 GT geometry 不进入 runtime，只由 evaluator 在 candidate 输出之后计算误差。

### 2.2 先完整运行 frozen BRTC-LC v1

独立 wrapper 首先原样调用：

```text
versions/v14/b0_person_triangulation.py::refine_matched_people
```

Frozen BRTC 对每个 matched person 用相机中心与人体 root/torso/joints 构造两视图 ray
evidence，得到 individual shift，并根据 gap/parallax 等 frozen 条件接受或拒绝。多人情况下
再计算：

```text
group = median(accepted individual shifts)
residual_i = selected_residual_lambda * (individual_i - group)
frozen_final_i = group + residual_i
```

被拒绝的人和 unmatched person 保持原 BRTC 行为。

### 2.3 Strict FAGD gate

令：

```text
Npre  = len(pre_people)
Npost = len(post_people)
M     = matched_count
A     = accepted_count
```

仅当：

```text
A == M == max(Npre, Npost) > 0
```

才执行：

```text
strict_final_i = 0.9 * group + residual_i
```

这里仅把所有人的公共平移缩到 `0.9`；frozen individual residual 和已选择的 residual
lambda 完全不变。因为同一个 `0.9 * group` 加到所有人身上，当前 boundary 内的两两相对
向量和距离在数学上保持不变。

若条件不成立，则不对数组做减加、重建或类型转换，直接返回第一次 frozen v1 调用产生的
corrected geometry 对象：

```text
strict_final = exact frozen BRTC-LC v1
```

### 2.4 输出

输出仍为：

```text
corrected_post_people, runtime_debug
```

只允许 matched post person 的 `root/joints/vertices` 发生共同平移。相机不更新，unmatched
person 不更新。debug 额外记录人数、匹配数、接受数、strict gate、exact-v1 fallback、
base/scaled group shift。

实现文件：

```text
versions/v14/b0_person_triangulation_strict_fagd.py
```

## 3. Same-visibility 确认实验

`alpha=0.9` 复用此前只在 `three offset0` development 冻结的策略；没有在以下确认集重新
选择参数。

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Strict cuts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| offset1 | BRTC v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 7.2% | - |
| offset1 | strict FAGD | **.226928** | **.271807** | **.248941** | .098351 | .258779 | 7.2% | 30/42 |
| dance | BRTC v1 | .125131 | .177804 | .152914 | .044141 | .078318 | 3.3% | - |
| dance | strict FAGD | **.113956** | **.168753** | **.142597** | .044141 | .078318 | **1.6%** | 60/61 |
| box | BRTC v1 | .372345 | .421610 | .434528 | .063069 | .427334 | 6.4% | - |
| box | strict FAGD | **.352626** | **.402936** | **.414447** | .063069 | .427334 | **5.8%** | 76/78 |

相对 BRTC v1：

- offset1 的 root/joint/vertex 改善 `4.51/2.69/3.51 mm`；
- dance 改善 `11.18/9.05/10.32 mm`；
- box 改善 `19.72/18.67/20.08 mm`；
- 三个 split 的 pair distance/vector 均 bit-exact；
- 三个 split 的 `>5 cm` harm 均未增加。

因此，same-visibility 下的空间规律确认通过。

## 4. Variable-visibility 反例实验

测试已有的 `three / dance / box` variable-visibility cases：

| Split | Cases | 人数变化 | 等人数人物替换 | Strict cuts | 全集 exact v1 | 人数变化 exact v1 | Root v1 → strict | Pair vector v1 → strict |
|---|---:|---:|---:|---:|---|---|---:|---:|
| three | 22 | 19 | 3 | 2 | False | **True** | .199588 → .199271 | .129182 → .129182 |
| dance | 29 | 18 | 11 | 11 | False | **True** | .256754 → .246437 | .444556 → .444556 |
| box | 12 | 12 | 0 | 0 | **True** | **True** | .201444 → .201444 | N/A |

人数变化时，`matched_count` 不可能等于 `max(Npre,Npost)`，所以 49/49 个 case 都正确回退
到 exact v1。这部分目标已经完成。

失败发生在等人数人物替换。例如：

```text
pre visible set  = {A, B}
post visible set = {A, C}
Npre = Npost = matched_count = accepted_count = 2
```

单靠 count/match/accepted，runtime 看起来像完整 `2→2`，但实际 visible identity set 已经
变化。three 有 3 个、dance 有 11 个此类反例，其中 13/14 个通过 strict gate 并执行 FAGD。

这是信息不可观测问题，不是再调整 `alpha` 能解决的问题。若不增加可跨 shot 验证 identity
continuity 的显式信息，任何只看人数和 BRTC ray accept 的 gate 都无法同时做到：

```text
same visibility -> 保留 FAGD 收益
any visibility replacement -> exact v1
```

## 5. EgoHumans 同-forward CPU cache

使用已有 cache：

```text
output/v14/fine_alignment_research/brtc_multithumbs_egohumans/
current_v14_cpu_geometry.pt
```

协议为本地构造的 3 条 `001_legoassemble` 15-frame chain，共 45 帧、6 个 cut；未重新运行
人体网络。coverage 为 `121/135 = 89.6%`。

| Method | W | WA | Root | Joint | Vertex | Pair dist | Pair vec | Root Accel | Joint Accel | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 350.614 | 235.207 | 420.163 | 416.226 | 414.913 | 188.485 | 388.351 | 160.517 | 160.997 | 0.0% |
| BRTC v1 | 314.059 | **202.461008** | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | **116.014** | **125.270** | 23.8% |
| strict FAGD | **313.985** | 202.461206 | **379.179** | **383.234** | **383.543** | **176.024** | **333.759** | 118.040 | 126.342 | 23.8% |

相对 BRTC v1，strict FAGD：

- W 改善 `0.074 mm`；
- fixed-world root/joint/vertex 改善 `1.475/1.495/1.696 mm`；
- pair distance/vector 改善 `1.001/0.111 mm`；
- WA 退化 `0.000199 mm`，属于数值级变化，但不满足预注册的“所有空间项严格改善”；
- world-root Accel 退化 `2.026 mm/frame²`；
- world-joint Accel 退化 `1.072 mm/frame²`；
- harm 保持 `23.8%`，camera max delta 为 `0`。

6 个边界的严格门控如下：

| Chain/cut | Npre→Npost | Matched | Accepted | Strict action |
|---|---:|---:|---:|---|
| 0/0 | 3→3 | 3 | 3 | FAGD-0.9 |
| 0/1 | 3→3 | 3 | 2 | exact v1 |
| 1/0 | 3→3 | 3 | 3 | FAGD-0.9 |
| 1/1 | 3→3 | 3 | 2 | exact v1 |
| 2/0 | 3→1 | 1 | 0 | exact v1 |
| 2/1 | 1→3 | 1 | 1 | exact v1 |

因此 strict runtime 只在 `2/6` 个边界动作。它与旧 FAGD callback 不 bit-exact 是预期行为：
旧 gate `accepted_count == matched_count > 0` 会在最后一个 `1→3` 边界动作，而 strict gate
因 `1 != max(1,3)` 必须回退。二者最大人体 geometry 差为 `8.974 mm`；相机差仍为零。

### Accel caveat

本地 Accel 只能当方向性诊断，不能当 Multi-THuMBS 官方数字：

- 三条链是自建短链，每个 cross-camera cut 两侧重复同一个 dataset timestamp；
- 当前按 stream index 当相邻帧计算离散二阶差分；
- 论文没有公开 Accel 的坐标系、fps、公式和聚合；
- pelvis-centered Accel 会消掉 BRTC/FAGD 的刚性人体平移，不能评价这一步精对齐；
- world-root/world-joint Accel 对精对齐敏感，但不是论文确认口径；
- FAGD 是 piecewise rigid post-shot translation，不是 temporal stabilizer，空间误差变好而 cut
  附近二阶差分变差并不矛盾。

## 6. 与 Multi-THuMBS 的关系

论文在 EgoHumans 上公开的参考线为：

| Method | W | WA | MPJPE | MPVPE | Accel | ATE | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multi-THuMBS | 279.0 | 166.0 | 228.3 | 262.2 | 27.3 | 0.7 | 0.97 |

当前本地 strict FAGD 的 W/WA 为 `313.985/202.461 mm`，按数值距离参考线仍差
`+34.985/+36.461 mm`。

但不能用本地其它更小的 pelvis MPJPE/MPVPE/ATE 宣称已经超过论文。当前公开的 17 页主文
只命名了 W-MPJPE、WA-MPJPE、MPJPE、MPVPE、Accel、ATE、IDs，没有公布完整公式、单位、
坐标系、visibility/miss/FP 处理、sequence manifest 和聚合方式；正文引用的 supplementary
目前也不在本地 PDF/arXiv source 中。

Movie3R 与论文只确认到 dataset-level 重合：本地 `001_legoassemble` 属于 EgoHumans，但无法
确认它是不是论文 official split。当前 3×15-frame、89.6% coverage、自建 camera cuts 和
per-chain Sim(3) 只能称：

```text
same-source EgoHumans provisional benchmark
```

可信结论仅限同一 frozen forward、同一检测和同一 evaluator 下的内部相对改善。正式“打过
Multi-THuMBS”必须获得作者的 benchmark manifest/evaluator 后重跑。

## 7. 安全性与测试

新增单测覆盖：

1. 完整一对一且全接受时，只缩 group，individual residual 不变；
2. `3→2` 人数变化时 exact v1；
3. 人数相同但匹配不完整时 exact v1；
4. 任一 evidence 被拒绝时 exact v1；
5. camera bit-exact。

测试结果：

```text
8 passed
```

其中包括 frozen BRTC v1 原有 4 项测试和 strict FAGD 新增 4 项测试。

## 8. 决策与后续主线

### 8.1 当前可以冻结的知识

- B0 camera 和 BRTC-LC v1 保持 frozen；
- group-only `0.9` 在真正 same-visibility、全接受边界上是稳定空间改进；
- individual residual 应保留，不应与 group 一起整体缩放；
- count-change strict fallback 已达到 exact-v1；
- 仅靠 cardinality 无法识别 equal-count identity replacement；
- 单次跨 shot 刚性平移不能同时解决 temporal smoothness。

### 8.2 不能部署 strict FAGD 的两个独立原因

1. **集合安全性失败**：等人数 identity replacement 会误触发；
2. **时序非退化失败**：Ego world-root Accel `+2.026 mm/frame²`。

任一项都足以给出 NO-GO，因此不应继续仅扫描 `alpha` 来包装成最终方法。

### 8.3 下一步应该补的显式信息

下一条实验主线应从“人数 gate”升级为“identity-set continuity gate”，并保持 FAGD 的
group/residual 分解：

```text
if full one-to-one + all accepted + every match passes identity continuity:
    final_i = 0.9 * group + frozen individual residual_i
else:
    exact BRTC-LC v1
```

identity continuity 必须是在线、无 GT 的显式证据，例如：

- frozen appearance embedding 的 mutual-nearest / margin consistency；
- torso/pose descriptor 与 camera-compensated world-root distance的联合 Hungarian cost；
- pre/post assignment 的 cycle consistency 或拒绝阈值；
- 上游若能提供可靠 persistent track ID，可直接比较 visible identity set。

该 gate 必须专门在本轮 14 个 equal-count replacement 反例和独立 same-visibility 集上验证，
目标是 replacement 全部 exact-v1，同时不切掉现有空间收益。通过集合安全后，再新增独立的
causal temporal stabilizer 或 boundary-aware velocity/acceleration regularizer；不要让一个
piecewise constant spatial shift 同时承担时序平滑职责。

## 9. 可复现产物

```text
versions/v14/b0_person_triangulation_strict_fagd.py
versions/v14/tests/test_b0_person_triangulation_strict_fagd.py
versions/v14/probe_brtc_strict_deployable_fagd.py
versions/v14/eval_brtc_strict_fagd_egohumans.py

output/v14/fine_alignment_research/brtc_strict_deployable_fagd/
  multihuman_report.json
  multihuman_report.md
  egohumans/report.json
  egohumans/README.md
```

复现命令：

```bash
.venv/bin/python versions/v14/probe_brtc_strict_deployable_fagd.py
.venv/bin/python versions/v14/eval_brtc_strict_fagd_egohumans.py
.venv/bin/python -m pytest -q \
  versions/v14/tests/test_b0_person_triangulation.py \
  versions/v14/tests/test_b0_person_triangulation_strict_fagd.py
```
