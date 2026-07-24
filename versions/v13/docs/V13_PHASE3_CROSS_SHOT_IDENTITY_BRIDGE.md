# V13 Phase 3：Cross-Shot Identity Bridge 最终实验报告

## 1. 最终结论

本阶段完成了从 Human3R 原生人物表示到自动跨 shot 身份关联，再到冻结的多人 shared
Boundary 的完整实验链路。结论是：

1. **V13 Phase 2 的 GT-ID 多人几何结论仍然成立。** 当 WHO 正确时，Uniform
   Multi-Human Consensus 能稳定优于可部署单人 anchor。
2. **Human3R 原生 tracker 不能直接跨 camera cut 使用。** 原版 refined token `H'`
   加 raw L2、dustbin 和 Sinkhorn 在 `three` 上只有 `0.4003` IDF1。
3. `three` 开发集上最强的无训练 cue 是 root-centered local pose，而不是原生 token。
   冻结规则达到 `0.9313` IDF1，但它是 short-horizon motion compatibility，不是稳定身份表示。
4. `box` 上没有发生错误 accepted match，自动 ID 完整复现了 GT-ID 收益；但 `dance`
   只发生 2 次 ID swap，就产生 `5.56%` catastrophic failure，并使自动方法差于单人。
5. `three` 中 281 个没有错误 accepted ID 的 cut 上，自动多人和 GT-ID 基本一致；34 个
   包含错误 ID 的 cut 上，camera composite 上升到 `2.463`，其中 `29.4%` catastrophic。
6. 当前 dustbin 能减少部分错误 coverage，但不能可靠区分困难真匹配和错误匹配。
7. 当前保守 geometry verification 不能判断冲突双方谁正确。在 `three` 上将 composite
   从 `0.850` 进一步恶化到 `1.073`，不能保留。
8. EgoHumans 困难的 `3 -> 1 -> 3` 检测链出现一次错误身份关联，说明人数下降和强遮挡下
   仍不安全。

因此，本阶段按预注册路线判定为 **FAIL**：

```text
保留：GT-ID Uniform Multi-Human Geometry Oracle
不保留：当前 native/pose Automatic Identity Bridge
V13 默认：token_reid = false
```

下一步如果继续 WHO，应单独研究轻量 shot-invariant ID adapter 或冻结 appearance cue。
它们只能回答 WHO，不能预测 Boundary、SE(3)、scale 或 fusion weight。

---

## 2. 冻结范围

本阶段父版本为：

```text
commit: 20d5391
tag: movie3r-v13-multihuman-geometry-validated
```

几何路径没有改变：

```text
Frozen Human3R
-> pre-decode Hard Reset
-> Fixed Explicit
-> per-human V16 torso residual
-> 20 degree bound
-> s = 1
-> one (R_i, t_i) per accepted identity
-> equal-weight SO(3) mean of R_i
-> arithmetic mean of raw t_i
-> ONE shared Boundary
```

同一个 Boundary 作用于 camera、pointmap 和所有 SMPL-X。Fallback 固定为：

```text
N >= 2: Uniform Multi-Human Consensus
N == 1: single-human Fixed Explicit + V16
N == 0: identity-free Fixed Explicit fallback
```

关闭模块：DA3、Keypoint R-CNN、V11.4 scale、VGGT、continuity、scene refinement 和
learned fusion。Phase 3 唯一变化是用 automatic identity association 替换 GT identity。

---

## 3. 实现架构

### 3.1 WHO 模块

实现文件：

- `versions/v13/identity_bridge.py`
- `versions/v13/experiments/phase3_cross_shot_identity.py`
- `versions/v13/experiments/phase3_egohumans_identity.py`

输入是 cut 前 5 帧和 cut 后 fresh decode 第一帧的人物检测及人物表示。输出只有：

```text
pre external track ID <-> post detection index
match cost/confidence
accepted/dustbin
```

WHO 模块没有 Boundary solver，也不输出 rotation、translation、scale 或人物权重。

### 3.2 Identity bank

每个 external identity 保存最近 5 个有效 observation。实现支持：

- last、five-frame mean 和 five-frame medoid prototype；
- active/inactive track；
- 固定 TTL `8`；
- tentative matching；
- unmatched/new-person 新 ID；
- Align-Then-Commit；
- rejected match 不更新旧 prototype。

生命周期为：

```text
Match tentative
-> solve shared Boundary
-> optional verification
-> Align
-> Commit identity observation
```

Boundary 确定前，长期 identity bank 不发生写入。

Identity bridge 是 Human3R decode 后的外部模块，不向 recurrent decoder 回写任何值。普通
no-cut Human3R camera、pointmap 和 SMPL-X 输出因此按构造保持 exact no-op；external bank
只观察已经产生的人物表示。

### 3.3 Matching

实验比较：

- raw L2、normalized L2、cosine；
- Hungarian；
- Human3R 风格 Sinkhorn/Optimal Transport；
- dustbin；
- last、mean、medoid prototype。

开发集最终冻结规则为：

```json
{
  "feature": "local_pose",
  "prototype": "last",
  "distance": "cosine",
  "matcher": "hungarian",
  "max_cost": 0.03011337919113551,
  "track_ttl": 8,
  "prototype_window": 5
}
```

该规则只在 MultiHuman `three` 上选择。`dance`、`box` 和 EgoHumans 没有重新调参。

---

## 4. Human3R 原生 tracker 审计

原生 tracking 主路径位于 `src/dust3r/model_human3r.py:1874`：

```text
previous refined H'
-> torch.cdist p=2
-> log optimal transport, alpha=-10, iterations=20
-> mutual nearest check
-> probability > 0.2
-> preserve unmatched previous token
-> allocate new smpl_id to unmatched current detection
```

关键事实：

- memory 保存上一时刻 refined token 和 unmatched token，不是学习到的长期 identity embedding；
- 没有显式 TTL；
- unmatched token 会继续留在 `last_smpl_tk/last_smpl_id`；
- `fresh` routing 在 cut 前清空 `last_smpl_tk`、`last_smpl_id` 和 `max_smpl_id`；
- `tracklet` routing 可保留 native tracklet，但 scene/camera recurrent state 仍 fresh；
- 本阶段最终使用独立 external identity bank，不依赖 post-cut 原生 `smpl_id` 命名空间。

Hook 位于 `src/dust3r/model.py:5063` 和 `src/dust3r/model.py:5646`。实际提取维度：

| 表示 | 维度 |
|---|---:|
| refined human token `H'` | 768 |
| CUT3R head token | 1024 |
| Multi-HMR head token | 1024 |
| fused human prompt | 768 |
| predicted beta | 10 |
| root-centered local pose | 468 |

---

## 5. 数据与协议

### 5.1 MultiHuman `three`，development

```text
7 timestamps
x 9 camera pairs
x offsets 0/1/2/4/8
= 315 cuts
```

每个 cut 使用 5 个 pre-cut 全画面 RGB 和 1 个 post-cut 全画面 RGB。2048x2048 图像只缩放
到 512x512，不做人物 crop。

### 5.2 MultiHuman `dance`，frozen evaluation

```text
6 timestamps x 3 camera pairs x offsets 0/4 = 36 cuts
```

两人全画面序列。规则来自 `three`，不允许调参。

### 5.3 MultiHuman `box`，frozen evaluation

```text
6 timestamps x 3 camera pairs x offsets 0/4 = 36 cuts
```

第二个两人独立序列。规则来自 `three`，不允许调参。

### 5.4 EgoHumans `001_legoassemble`，cross-data stress test

完整鱼眼画面从 3840x2160 等比例缩放到 512x288，不裁剪。测试三条真实 multi-cut 链：

```text
cam01 296-300 -> cam06 300-304 -> cam07 304-308
cam02 176-180 -> cam05 180-184 -> cam08 184-188
cam03 416-420 -> cam04 420-424 -> cam01 424-428
```

第三条链中 Human3R 检测人数为 `3 -> 1 -> 3`，用于单人 fallback 和重新出现压力测试。
EgoHumans 只评价 WHO、memory、dustbin 和 TTL，不把旧 geometry smoke test 作为完整 V13
Boundary 结果。

---

## 6. GT 与 metadata 使用边界

Feature cache 只保存 RGB 推理得到的：

- detection index；
- native `smpl_id`；
- Human3R token/prompt；
- beta；
- local pose；
- detection score 和位置。

自动 matcher 不读取 GT identity、GT camera、GT SMPL-X、source ID、camera ID、camera-pair
ID、sequence 名或文件路径。旧 geometry cache 原本以 GT identity 为字典 key，并在人物
payload 内留下未被读取的 `identity` 字段；Phase 3 最终实现会先按 `detection_index` 重建
纯 detection multiset，并显式删除该字段，再进入自动几何。

GT 只用于：

- feature probe 打分；
- assignment evaluator；
- GT-ID Uniform Consensus 上界；
- camera/human 最终指标。

测试已验证随机修改 source、camera、camera-pair 和 path metadata 不改变 cost 或 assignment。
删除 payload 中未读取的 GT identity 字段后，`three/dance/box` 的全部关键指标保持不变。
另在真实 `three` cache 上检查了 18 个自动几何 detection payload，GT identity 字段数量为 0。

---

## 7. Stage 0：无 cut tracking

| Sequence | Detection recall | Adjacent assignment | Native IDF1 | Switch | Fragmentation |
|---|---:|---:|---:|---:|---:|
| three | 0.9222 | 1.0000 | 1.0000 | 0 | 0 |
| dance | 0.8833 | 1.0000 | 1.0000 | 0 | 0 |
| box | 1.0000 | 0.9792 | 0.9667 | 2 | 2 |

结论：普通连续帧下原生 tracking 总体可用，但 `box` 已存在少量 within-shot switch。Camera
cut 后的明显下降不是由 hook 完全失效造成的。

---

## 8. Stage 1：跨 shot feature probe

### 8.1 `three` 特征排名

每个 feature 取其最好的 prototype、distance 和 matcher：

| Feature | Best IDF1 | 解释 |
|---|---:|---|
| local pose | 0.9338 | 短时动作兼容，非长期身份 |
| SMPL beta | 0.8766 | 当前最强的纯 shape cue |
| beta + pose | 0.8718 | 没有超过 local pose |
| refined H' + beta + pose | 0.7191 | 高维 token 反而降低稳定性 |
| CUT3R token + beta | 0.7057 | 不足以部署 |
| refined H' + beta | 0.6675 | 不足以部署 |
| Multi-HMR token | 0.6571 | 最强 native 单 token，但仍低 |
| refined H' | 0.4878 | 原生 tracking feature 跨 cut 失效 |
| fused prompt | 0.4234 | 失效 |
| CUT3R head token | 0.4007 | 失效 |

### 8.2 Human3R 原生 L2 + Sinkhorn

| Sequence | Assignment accuracy | Recall@1 | IDF1 | ID switches |
|---|---:|---:|---:|---:|
| three | 0.4294 | 0.3749 | 0.4003 | 416 |
| dance | 0.6557 | 0.6557 | 0.6557 | 21 |
| box | 0.7385 | 0.7164 | 0.7273 | 17 |

原生 tracker 规则不能直接跨 wide-view camera cut 使用。

### 8.3 Prototype 结论

在 `three` 的 local pose 上：

| Prototype | Distance | Matcher | IDF1 |
|---|---|---|---:|
| last | cosine | Hungarian | 0.9338 |
| medoid | cosine | Hungarian | 0.9267 |
| mean | normalized L2 | Hungarian | 0.9231 |

Last-frame 最好，进一步证明该 cue 主要依赖短时动作连续性。Sinkhorn 没有超过 Hungarian。

### 8.4 冻结规则的 identity 结果

| Sequence | Accuracy | Recall@1 | IDF1 | Switch | Dustbin P/R |
|---|---:|---:|---:|---:|---:|
| three | 0.9370 | 0.9257 | 0.9313 | 52 | 0.704 / 0.838 |
| dance | 0.9661 | 0.9344 | 0.9500 | 2 | 0.600 / 0.818 |
| box | 1.0000 | 0.9701 | 0.9848 | 0 | 0.556 / 1.000 |

这里的 IDF1 是 camera-boundary assignment 聚合指标，不应冒充完整长视频 HOTA/IDF1。

---

## 9. Stage 2：自动 ID 接入冻结多人 Boundary

### 9.1 Camera 主结果

| Sequence | Method | T m | Rot deg | Composite | P90 | P95 | Catastrophic |
|---|---|---:|---:|---:|---:|---:|---:|
| three | Single highest confidence | 0.616 | 9.90 | 0.814 | 1.314 | 1.437 | 0.0% |
| three | GT-ID Uniform | 0.522 | 7.07 | 0.664 | 0.989 | 1.126 | 0.0% |
| three | Automatic ID Uniform | 0.629 | 11.08 | 0.850 | 1.234 | 1.560 | 3.17% |
| dance | Single highest confidence | 0.625 | 8.87 | 0.802 | 1.299 | 1.528 | 0.0% |
| dance | GT-ID Uniform | 0.610 | 7.38 | 0.758 | 1.317 | 1.371 | 0.0% |
| dance | Automatic ID Uniform | 0.714 | 8.53 | 0.885 | 1.450 | 2.058 | 5.56% |
| box | Single highest confidence | 0.521 | 9.98 | 0.720 | 1.029 | 1.137 | 0.0% |
| box | GT-ID Uniform | 0.455 | 7.94 | 0.614 | 0.857 | 0.910 | 0.0% |
| box | Automatic ID Uniform | 0.453 | 7.92 | 0.612 | 0.838 | 0.909 | 0.0% |

收益保留率：

| Sequence | Retention |
|---|---:|
| three | -0.241 |
| dance | -1.882 |
| box | 1.024 |

只有 `box` 达到 70% 保留率标准。`three` 和 `dance` 的自动方法都比单人更差，整体 gate
失败。

### 9.2 Human 主结果

| Sequence | Method | Root m | Joint m | Vertex m |
|---|---|---:|---:|---:|
| three | Single | 0.383 | 0.402 | 0.392 |
| three | GT-ID | 0.362 | 0.380 | 0.372 |
| three | Automatic | 0.460 | 0.480 | 0.471 |
| dance | Single | 0.526 | 0.533 | 0.529 |
| dance | GT-ID | 0.519 | 0.528 | 0.523 |
| dance | Automatic | 0.588 | 0.594 | 0.591 |
| box | Single | 0.555 | 0.610 | 0.614 |
| box | GT-ID | 0.555 | 0.606 | 0.611 |
| box | Automatic | 0.553 | 0.605 | 0.609 |

ID swap 同时破坏 camera 和所有人的 world placement，不是只影响人物颜色或 track label。

### 9.3 错误尾部分析

`three`：

```text
281/315 无错误 accepted ID:
automatic composite = 0.655
catastrophic = 0%

34/315 至少一个错误 accepted ID:
automatic composite = 2.463
catastrophic = 29.4%
```

`dance`：

```text
34/36 无错误 accepted ID:
automatic composite = 0.770
catastrophic = 0%

2/36 有错误 accepted ID:
automatic composite = 2.840
catastrophic = 100%
```

这说明 WHERE 公式在 WHO 正确时工作正常。失败原因是 shared Boundary 对少量 ID swap
极其敏感，而不是 Uniform Consensus 本身失效。

### 9.4 因果控制

在 `three` 上：

| Control | Composite | Catastrophic |
|---|---:|---:|
| wrong-person | 5.064 | 87.9% |
| shuffled memory | 3.437 | 57.1% |
| zero memory / Fixed fallback | 4.793 | 79.4% |

Correct memory 显著优于 wrong、shuffle 和 zero control，证明 feature memory 确实携带有用的
短时对应信息。但“有用”不等于“足够安全地部署 shared Boundary”。

---

## 10. Dustbin、fallback 与 geometry verification

### 10.1 Dustbin

| Sequence | With dustbin | Without dustbin | Catastrophic 两者 |
|---|---:|---:|---:|
| three | 0.850 | 0.818 | 3.17% |
| dance | 0.885 | 0.859 | 5.56% |
| box | 0.612 | 0.612 | 0.0% |

当前阈值会拒绝部分正确匹配，却没有消除错误 accepted match。它实现了进入/离开语义和
fallback 路径，但安全性尚不合格。

### 10.2 Conservative geometry verification

固定 gross conflict 条件：

```text
max translation candidate disagreement > 3 m
or
max rotation candidate disagreement > 100 deg
```

`three` 触发 26 次，将全部 match 送入 dustbin 并回退 Fixed，结果：

```text
identity-only composite: 0.850, catastrophic 3.17%
geometry-verified:        1.073, catastrophic 7.62%
```

`dance` 的两次真实 swap 没有触发该规则。结论是 candidate disagreement 不能指出哪个身份
错误，Phase 2 关于“residual 不适合 hard reject”的结论再次成立。该 verification 不保留。

---

## 11. EgoHumans multi-cut memory 结果

| Stream | Detection pattern | IDF1 | Switch | 说明 |
|---|---|---:|---:|---|
| cam01 -> cam06 -> cam07 | 3 -> 3 -> 3 | 0.909 | 0 | 一次真匹配被拒，产生 duplicate track |
| cam02 -> cam05 -> cam08 | 2/3 -> 3 -> 3 | 0.909 | 0 | 恢复人数后可匹配，但仍有一次真拒配 |
| cam03 -> cam04 -> cam01 | 3 -> 1 -> 3 | 0.667 | 1 | 单人困难视角匹配到错误旧身份 |

第一条流第一次 cut 接受 2/3，第二次接受 3/3；被拒人物对应的新 external ID 作为 inactive
duplicate 留在 bank 中。第三条流在 `cam04` 只检测到一人时发生错误 accepted match，说明
当前 local-pose threshold 不能保证 single-human fallback 的身份正确性。

TTL、inactive memory、new track 分配和重新出现代码路径均运行，但 identity fragmentation
没有解决。该测试没有真正的 GT 新人物进入，只有检测漏失和恢复，因此不能宣称完整
entry/new-person benchmark 已通过。

---

## 12. 对本阶段 12 个问题的回答

1. **Human3R 原生 tracker 在普通帧和 cut 下如何？**
   普通连续帧总体可靠，`three/dance` IDF1 为 1.0，`box` 为 0.967；跨 cut 的原生规则在
   `three` 下降到 0.400。

2. **哪一种表示最稳定？**
   当前短窗口里 local pose 最强，beta 第二。四种原生 token/prompt 都明显更差；没有一个
   native token 可称为稳定 shot-invariant identity embedding。

3. **Local pose 是 identity 吗？**
   不是。Last-frame 明显优于 mean/medoid，说明它主要是 short-horizon motion compatibility。

4. **原生 L2 + Sinkhorn 能直接跨 shot 吗？**
   不能。`three` IDF1 只有 0.400，且发生 416 个错误 accepted assignment。

5. **Last、mean、medoid 哪个最稳？**
   对当前最强 local-pose cue，last 最好；这不是长期 identity memory 的证据。

6. **Dustbin 能正确处理进入、离开和漏检吗？**
   数据结构和 fallback 工作，但精度不够。它会拒绝真匹配，也没有消除 catastrophic swap。

7. **自动 ID 保留多少 GT-ID 收益？**
   `box` 保留 102%；`three` 和 `dance` 分别为 -24% 和 -188%。总体不通过 70% gate。

8. **少量错误 ID 会导致灾难吗？**
   会。`dance` 两次 swap 对应两次 catastrophic；`three` 错误子集 catastrophic 为 29.4%。

9. **保守 geometry verification 有效吗？**
   无效且有害。它无法从两个冲突 candidate 中判断谁正确，并会触发更差的 Fixed fallback。

10. **规则能泛化到 dance、box 和 EgoHumans 吗？**
    不稳定。`box` 成功，`dance` 和 EgoHumans 困难链失败，不能形成跨数据泛化结论。

11. **人数下降时能否 multi -> single -> Fixed？**
    代码路径可以执行，但 single 分支仍可能继承错误身份。机械 fallback 已实现，语义安全性
    未通过。

12. **是否值得作为最终贡献？**
    `Uniform Multi-Human Geometric Consensus + One Causal Shared Boundary` 仍值得保留为
    GT-ID 几何结论；当前 `Native Cross-Shot Identity Bridge` 不应作为最终可部署贡献。

---

## 13. 路线决策

本阶段符合预注册的情况 C/D/E：

- raw native token 不可靠；
- local pose 只适合 short-horizon compatibility；
- identity 平均分较高，但错误尾部破坏 shared Boundary；
- geometry verification 和当前 dustbin 无法安全修复。

正式决策：

```text
V13 Phase 2 geometry milestone remains valid.
V13 Phase 3 native deployable bridge is a negative result.
Do not enable automatic multi-human alignment by default.
Do not modify Uniform Consensus to hide identity failures.
Do not let identity representation predict SE(3).
```

下一项合法研究任务是单独建立更稳定的 WHO cue：

1. 轻量 shot-invariant identity adapter，Human3R 全冻结；或
2. 冻结轻量 appearance embedding 作为参考；
3. capture/subject/camera-pair disjoint split；
4. 先证明错误 accepted match 足够低，再重新接入同一个冻结几何模块。

---

## 14. 复现命令与产物

开发集：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase3_cross_shot_identity.py \
  --sequence three --role development --mode all --device cuda:4
```

冻结 holdout：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase3_cross_shot_identity.py \
  --sequence dance --role evaluation --mode all --device cuda:4

PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase3_cross_shot_identity.py \
  --sequence box --role evaluation --mode all --device cuda:4
```

EgoHumans frozen identity audit：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/phase3_egohumans_identity.py
```

主产物：

- `output/v13/phase3_identity/three/v13_phase3_identity_bridge.json`
- `output/v13/phase3_identity/dance/v13_phase3_identity_bridge.json`
- `output/v13/phase3_identity/box/v13_phase3_identity_bridge.json`
- `output/v13/phase3_identity/*/selected_distance_matrices.png`
- `output/v13/phase3_identity/*/native_sinkhorn_soft_matrices.png`
- `output/v13/phase3_identity/egohumans_*/v13_phase3_egohumans_identity.json`

验证状态：

- Python static compilation：通过；
- identity bridge direct tests：10/10 通过；
- `git diff --check`：通过；
- `.venv` 没有安装 pytest，因此测试函数由同一 Python 环境直接执行，不能声称 pytest
  runner 已运行。
