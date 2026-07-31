# V14 BRTC variable-visibility 与 soft completeness 实验

> 日期：2026-08-01。目标是检验 EgoHumans 上发现的 association-completeness
> weighting 是否能在独立的人数变化边界上复现，并寻找不引入新模型的严格在线软阻尼。

## 1. 协议

此前 B0 identity 实验为隔离身份变量，排除了可见人数变化的 cut：

- MultiHuman `three`：22 cuts；
- `dance`：29 cuts；
- `box`：12 cuts。

本轮为这些 cut 恢复 frozen B0 boundary，然后使用矩形 Hungarian 做匿名
root+torso+joints 关联。person refinement 只读 last-pre/当前 post，camera 永久冻结；
rejected/unmatched person exact B0。GT identity 与几何只在预测完成后用于 association audit
和 root/joint/vertex/layout/harm 指标。

开发/冻结/确认顺序：

```text
EgoHumans 观察到线性 completeness 规律
-> 独立 three variable-visibility 检验线性规则：失败
-> 把 three variable 作为 soft scale development，扫描后冻结 0.9
-> 冻结文件先落盘
-> 才生成并读取 dance/box variable confirmation
```

## 2. 线性 completeness：独立确认失败

公式：

```text
scale = matched_count / max(pre_person_count, post_person_count)
```

在 22 个 `three` cuts 中有 `8` 个 `3→2`、`11` 个 `2→3`、`3` 个检测集合改变但人数
仍为 `2→2`。主要结果：

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.2893 | 0.3301 | 0.2973 | 0.0713 | 0.1233 | - |
| BRTC-LC v1 | 0.1996 | 0.2434 | 0.2108 | 0.0720 | 0.1292 | 4.5% |
| Linear completeness | 0.2094 | 0.2534 | 0.2189 | 0.0666 | 0.1173 | 2.3% |

结论：安全性和 layout 改善，但单人空间误差退化；`NO-GO`。EgoHumans 的改善不是伪造，
但线性缩到 `2/3` 或 `1/2` 对多数正确 depth action 过强。

## 3. Frozen incomplete-only soft scale=0.9

只对不完整矩形关联使用一个开发集学习的常数，完整一一对应 exact v1：

```text
scale = 1.0  if matched_count == max(pre_count, post_count)
        0.9  otherwise
```

`three` development 扫描 `0.67, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00`。
要求 root/joint/vertex/pair-distance/pair-vector/harm 全部不差于 v1，再依次最小化 harm、root。
`0.85/0.90/0.95/1.0` 合格，规则冻结选择 `0.90`。冻结时间早于 dance/box boundary 与
confirmation 结果。

### 3.1 Development

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 0.1996 | 0.2434 | 0.2108 | 0.0720 | 0.1292 | 4.5% |
| soft 0.9 | **0.1973** | **0.2426** | **0.2087** | **0.0696** | **0.1246** | **2.3%** |

### 3.2 Frozen confirmation

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---|---|---:|---:|---:|---:|---:|---:|
| dance | v1 | 0.2568 | 0.2747 | 0.2714 | 0.2419 | 0.4446 | 0.0% |
| dance | soft 0.9 | **0.2539** | **0.2738** | **0.2688** | **0.2339** | **0.4195** | 0.0% |
| box | v1 | 0.2014 | **0.4622** | **0.4537** | n/a | n/a | 0.0% |
| box | soft 0.9 | **0.2012** | 0.4678 | 0.4562 | n/a | n/a | 0.0% |

`dance` 全指标复现，但 `box` 的 12 个 `2→1` 边界上 joint/vertex 分别退化约
`5.6/2.5 mm`。因此 frozen soft-0.9 仍为 `NO-GO`，不能替换 v1。

## 4. 发现的规律

1. 人数不完整是 BRTC 过激 action 的可靠风险特征，阻尼会稳定降低 layout error/harm；
2. 但“人数变化”不等于“depth proposal 错误”，多数 box 单人 action 实际正确，统一阻尼会
   损失 joint/vertex；
3. association completeness 应作为 fallback gate，而不能单独决定连续 action scale；
4. 更稳的下一规则必须把 completeness 与 evidence acceptance、action reliability 分开：只在
   完整一一对应且所有 evidence 可靠时改 group component，人数变化时 exact v1；
5. `2→2` 但人物集合变化无法仅靠 count ratio 发现，最终仍需要 dustbin/track lifecycle。

## 5. 产物

```text
versions/v14/b0_person_triangulation_completeness_weighted.py
versions/v14/tests/test_b0_person_triangulation_completeness_weighted.py
versions/v14/probe_brtc_variable_visibility.py
versions/v14/probe_brtc_soft_completeness.py
output/v14/fine_alignment_research/brtc_variable_visibility/
output/v14/fine_alignment_research/brtc_soft_completeness/
```
