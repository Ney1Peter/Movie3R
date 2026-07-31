# V14 BRTC 可观测关联 dustbin 实验（2026-08-01）

## 1. 结论先行

本候选的最终结论是：**NO-GO，不能作为通用部署 gate。**

开发集上确实找到一个很干净的规律：在 MultiHuman `three offset-0` 中，正确匹配边的躯干朝向差最大为 `50.75°`，identity replacement 错边最小为 `75.79°`。因此冻结了下面的规则：

```text
先使用 B0 后的 root + torso + centered joints 做 Hungarian；
若一条 Hungarian 边的原始 torso angle <= 60°，保留；
否则把 post person 送入 dustbin，保持 exact B0；
保留的边再进入冻结 BRTC-LC v1 / strict FAGD-0.9。
```

它在开发集上拒绝了全部 4 条 replacement 错边，同时保留全部 162 条正确边。但冻结后在 held-out 上出现两类反例：

1. `dance` 有 2 条正确边的 torso angle 为 `72.16°/73.35°`，被误拒；
2. `box` 有一次完整的两人身份互换，两条错误边的 torso angle 只有 `7.42°/40.67°`，全部漏过。

更关键的是，`box` 的一条错误边在所有当前几何置信特征中都位于正确边分布内部，而且 Hungarian assignment margin 很大，属于“**几何上非常自信，但身份是错的**”。这说明只依靠当前 B0 人体几何，不能可靠解决通用 association / identity replacement；后续必须引入独立于几何的显式身份信息，例如外观、局部纹理或可靠的跨镜头 track memory。

## 2. 为什么做这个实验

Strict FAGD 的原始可观测条件是：

```text
accepted_count == matched_count ==
max(len(pre_people), len(post_people)) > 0
```

它能够识别 `2→3`、`3→2` 这类人数变化，但仅看人数无法识别：

```text
pre:  A, B
post: A, C
count: 2 → 2
```

如果 Hungarian 把 `B→C` 当作有效匹配，strict FAGD 仍可能被触发。这次实验的目标就是增加一个完全可观测、无需 GT、无需新模型的 association dustbin。

## 3. 输入、模块和输出

### 3.1 推理输入

每个 shot boundary 只使用：

- 最后一帧 pre-cut 的 Human3R 人体预测；
- 当前第一帧 post-cut 的 Human3R 人体预测；
- 已冻结 B0 给出的边界刚体变换；
- 已冻结 BRTC-LC v1 和 strict FAGD-0.9。

不使用：

- GT identity；
- GT 相机或 GT 人体；
- future frame；
- 图像外观 encoder 或 ReID 模型；
- DA3 或其他新增预训练模型；
- GPU。

### 3.2 匿名 Hungarian

先用 B0 把 post 人体变换到 pre 世界系，然后计算矩形 cost matrix：

```text
root component:
    || B0(post_root) - pre_root ||

torso component:
    rotation_angle(pre_torso, B0_R * post_torso)

centered-joint component:
    mean_j || (pre_joint_j - pre_root)
              - B0_R * (post_joint_j - post_root) ||

Hungarian cost:
    root / median(root_matrix)
  + torso / median(torso_matrix)
  + joints / median(joints_matrix)
```

身份字符串不参与 cost 或 Hungarian。匹配完成后才用 evaluator ID 标记该边是否正确。

### 3.3 审计的可观测特征

每条 Hungarian 边记录：

- 原始 root distance（米）；
- 原始 torso angle（度）；
- 原始 centered-joint distance（米）；
- median-normalized combined cost；
- row best-vs-alternative margin；
- column best-vs-alternative margin；
- 整个 assignment 的 best-vs-second margin。

需要特别强调：`1×1` cost matrix 中，每个经 matrix median 归一化的 component 恒为 `1`，combined cost 恒为 `3`。因此实际 gate 只能依赖原始绝对量，不能依赖归一化 cost。

### 3.4 Dustbin 后的几何处理

```text
retained edge:
    送入 frozen BRTC-LC v1；
    只有仍满足 full one-to-one + all accepted 时才允许 FAGD-0.9。

rejected/unmatched post person:
    root、joints、vertices 全部保持 exact B0。

camera:
    不作任何修改，bit-exact B0。
```

## 4. 严格开发/冻结/确认协议

### 4.1 Development

只使用：

- `three offset-0` same visibility：41 cuts；
- `three offset-0` variable visibility：22 cuts；
- 合计 63 cuts、166 条 Hungarian 边。

扫描对象被限制为“一个原始绝对特征 + 一个最大阈值”，选择顺序为：

1. 最少 wrong accepts；
2. 最多 retained correct edges；
3. 最大 multiplicative gap；
4. 不使用多特征小模型，避免在 4 条错边上过拟合。

### 4.2 Freeze

读取任何 held-out 之前写入：

`output/v14/fine_alignment_research/brtc_association_dustbin_gate/FROZEN_POLICY_BEFORE_HELDOUT.json`

冻结 policy SHA256：

```text
e3b4b28227a294bc8aa63231bd79feb656df4795cd9b433e21e6adcf3a34d80d
```

注意这是 policy 内容的 canonical SHA；整个 JSON 文件 SHA 为：

```text
7b7ed67d76e5c353b10a750e7d618a7a046132b794039721e28ad37989db1b57
```

### 4.3 Held-out confirmation

冻结后才读取：

- `three offset-1` same visibility：42 cuts；
- `dance` same visibility：61 cuts；
- `dance` variable visibility：29 cuts；
- `box` same visibility：78 cuts；
- `box` variable visibility：12 cuts。

## 5. 开发结果

### 5.1 特征分布

| Feature | Correct min / median / max | Wrong min / median / max | 是否完全可分 |
|---|---:|---:|---|
| Root (m) | 0.070 / 0.414 / 0.775 | 0.290 / 0.353 / 0.564 | 否 |
| Torso (deg) | 1.325 / 7.867 / 50.751 | 75.790 / 84.199 / 130.211 | 是 |
| Centered joints (m) | 0.031 / 0.078 / 0.400 | 0.307 / 0.322 / 0.339 | 否 |
| Normalized combined | 0.454 / 1.225 / 2.505 | 2.698 / 2.970 / 3.274 | 是，但不适用于 1×1 |

因此冻结：

```text
torso_deg <= 60.0
```

### 5.2 Association

| Split | Before precision | Before wrong | After precision | After wrong | Correct coverage |
|---|---:|---:|---:|---:|---:|
| All dev | 97.6% | 4 | 100.0% | 0 | 100.0% |
| Same visibility | 100.0% | 0 | 100.0% | 0 | 100.0% |
| Variable visibility | 90.9% | 4 | 100.0% | 0 | 100.0% |
| True equal-count replacement | 33.3% | 4 | 100.0% | 0 | 100.0% of already-correct Hungarian edges |

`three` 中确实存在 3 个真实 equal-count replacement cuts，共 6 条 Hungarian 边：原始只有 2 条正确、4 条错误；dustbin 保留 2 条正确并拒掉 4 条错误。

### 5.3 开发几何结果

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.301915 | 0.331795 | 0.308963 | 0.079904 | 0.128548 | 0.0% |
| BRTC v1 | 0.162665 | 0.196029 | 0.173515 | 0.058523 | 0.103956 | 4.0% |
| Strict FAGD | 0.160689 | 0.194358 | 0.171147 | 0.058523 | 0.103956 | 3.4% |
| Dustbin + strict FAGD | **0.160618** | 0.194633 | 0.171349 | **0.057275** | 0.104135 | 3.4% |

开发集表现足以冻结，但没有形成 held-out 结论。

## 6. Held-out 结果

### 6.1 Association

| Split | Cases | 真 equal-count replacement | Before precision | After precision | Correct coverage | Wrong accepts |
|---|---:|---:|---:|---:|---:|---:|
| Three offset-1 same | 42 | 0 | 100.0% | 100.0% | 100.0% | 0 |
| Dance same | 61 | 0 | 100.0% | 100.0% | **98.4%** | 0 |
| Dance variable | 29 | 0 | 100.0% | 100.0% | 100.0% | 0 |
| Box same | 78 | 0 | 98.7% | **98.7%** | 100.0% | **2** |
| Box variable | 12 | 0 | 100.0% | 100.0% | 100.0% | 0 |

Held-out 最低正确边 coverage 为 `98.36%`，但仍漏过 2 条错边。

### 6.2 几何

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---|---:|---:|---:|---:|---:|---:|
| Three k1 same | Strict FAGD | 0.153268 | 0.181089 | 0.160828 | 0.047913 | 0.093026 | 3.2% |
| Three k1 same | Dustbin + strict FAGD | 0.153268 | 0.181089 | 0.160828 | 0.047913 | 0.093026 | 3.2% |
| Dance same | Strict FAGD | **0.113956** | **0.168753** | **0.142597** | **0.044141** | **0.078318** | 1.6% |
| Dance same | Dustbin + strict FAGD | 0.120652 | 0.174056 | 0.148361 | 0.055129 | 0.096597 | 1.6% |
| Dance variable | Strict FAGD | 0.271029 | 0.292691 | 0.286258 | 0.241880 | 0.444556 | 0.0% |
| Dance variable | Dustbin + strict FAGD | 0.271029 | 0.292691 | 0.286258 | 0.241880 | 0.444556 | 0.0% |
| Box same | Strict FAGD | 0.356145 | 0.405990 | 0.417160 | 0.058429 | 0.431058 | 5.8% |
| Box same | Dustbin + strict FAGD | 0.356145 | 0.405990 | 0.417160 | 0.058429 | 0.431058 | 5.8% |
| Box variable | Strict FAGD | 0.201444 | 0.462170 | 0.453719 | N/A | N/A | 0.0% |
| Box variable | Dustbin + strict FAGD | 0.201444 | 0.462170 | 0.453719 | N/A | N/A | 0.0% |

`dance same` 的两个 false reject 使五个空间/layout 指标全部退化。因此即使只从最终几何看，这个冻结 gate 也不能部署。

## 7. 两类关键反例

### 7.1 False reject：正确人发生较大躯干变化

```text
dance_t0500_c0_c1_k8:
    person0 -> person0, torso=73.350°, correct, rejected

dance_t0500_c0_c3_k8:
    person0 -> person0, torso=72.158°, correct, rejected
```

这说明 torso orientation 不是稳定 identity cue。动作变化、Human3R 局部旋转预测误差或跨视角身体朝向变化，都可能让同一个人的 torso angle 超过 `60°`。

### 7.2 False accept：几何上高置信的两人完整互换

```text
box_t0630_c0_c3_k8:
    person0 -> person1:
        root=0.447m, torso=40.666°, joints=0.158m,
        normalized cost=1.591, global assignment margin=7.025

    person1 -> person0:
        root=0.465m, torso=7.419°, joints=0.122m,
        normalized cost=1.102, global assignment margin=7.025
```

第二条错误边尤其重要：它的各个 feature 在 held-out 正确边中所处百分位分别约为：

| Feature | 错边值 | 在正确边中的 percentile |
|---|---:|---:|
| Root | 0.465 m | 50.5% |
| Torso | 7.419° | 37.8% |
| Centered joints | 0.122 m | 74.7% |
| Normalized combined | 1.102 | 39.6% |
| Row margin | 3.660 | 49.2% |
| Column margin | 3.854 | 52.0% |
| Global assignment margin | 7.025 | 43.4% |

也就是说，它在当前全部几何观测下就是一条“典型且高置信”的边。继续调 root/torso/joints 阈值无法把它可靠分离；加入 row/column/global margin 也无效。

## 8. 数据标签审计中的一个重要修正

旧 strict FAGD 报告把 variable list 中“人数相同”的 case 统称为 `equal-count replacement`：

- `three`: 3；
- `dance`: 11；
- `box`: 0。

本次重新检查 `strict_cache` 重分配后的 evaluator identity set，发现：

- `three` 的 3 个确实是 pre/post identity set 不同；
- `dance` 的 11 个虽然人数是 `1→1`，但 pre/post 都是同一个 `person0`，不是 replacement；
- held-out 五个 split 的真实 equal-count replacement 数量均为 0。

因此本次 held-out 并没有独立覆盖真正的 `1→1 replacement`。它覆盖到了另一个同样重要的失败：`box` same-visibility 中两个人都存在，但 Hungarian 完整交换身份。

后续报告应把两个概念分开：

```text
equal observable count
    !=
pre/post evaluator identity set replacement
```

## 9. 最终决策与下一步

### 9.1 本候选的最终决策

```text
NO_GO_ASSOCIATION_DUSTBIN_AS_FROZEN_GENERAL_GATE
```

通过的工程性质：

- rejected/unmatched post person 在 root/joints/vertices 上 bit-exact B0；
- camera max absolute change 为 `0`；
- 不用 future、不用 GT 推理、不用新预训练模型；
- dustbin 后 strict FAGD 不会在不完整匹配上误触发。

失败的算法性质：

- held-out wrong accepts 不为 0；
- held-out correct coverage 不是 100%；
- `dance same` 五个主指标全部比 ungated strict FAGD 差；
- 真正的 `1→1 replacement` 没有独立 held-out 数据。

### 9.2 下一条 association 主线

当前结果给出了一个比较明确的答案：**不能继续只从 Human3R 预测的 root、torso、centered joints 和 Hungarian margin 中寻找通用身份置信度。** 人体几何本身会错，而且能够“自洽地错”；这种情况下几何 cost 的 confidence 没有辨识力。

下一步需要显式、独立的身份观测，优先级建议：

1. 从当前帧已有的人体 crop 提取轻量外观 descriptor，和几何 cost 联合做 dustbin Hungarian；
2. 使用 clothing color / local patch token 等跨镜头相对稳定信息，避免只靠 body orientation；
3. 对 shot 边界构建专门的真实 replacement benchmark，至少包含 `1→1`、`2→2 one-replaced`、`2→2 full-swap`；
4. 先报告 association precision/recall，再接入 BRTC/FAGD，不能只看最终平均几何误差；
5. 保留本次已经验证正确的 dustbin 工程语义：低置信 post person exact B0、camera 不变、不完整匹配禁止 FAGD。

这条路线需要新增“显式外观/身份信息”，不是继续调几何阈值。

## 10. 产物与复现

代码：

```text
versions/v14/probe_brtc_association_dustbin_gate.py
```

代码 SHA256：

```text
7e9bdb626d1b92624637b75f8d45de4bf76f18501da8082f5fd5a2e9bc9b2982
```

输出：

```text
output/v14/fine_alignment_research/brtc_association_dustbin_gate/
├── development_report.json
├── development_report.md
├── FROZEN_POLICY_BEFORE_HELDOUT.json
├── confirmation_report.json
└── confirmation_report.md
```

复现：

```bash
.venv/bin/python versions/v14/probe_brtc_association_dustbin_gate.py --phase dev
.venv/bin/python versions/v14/probe_brtc_association_dustbin_gate.py --phase freeze
.venv/bin/python versions/v14/probe_brtc_association_dustbin_gate.py --phase confirm
```

基础测试：

```bash
.venv/bin/python -m py_compile versions/v14/probe_brtc_association_dustbin_gate.py
.venv/bin/python versions/v14/probe_brtc_association_dustbin_gate.py --phase dev --self_test
```
