# V14 BRTC：Joint Ray-Space Layout Least Squares 实验报告

> 日期：2026-08-01
>
> 状态：**NO-GO，不替换 frozen BRTC-LC v1**
>
> 约束：CPU geometry cache only；camera frozen；无额外预训练模型；不读取未来帧；
> rejected/unmatched exact no-op。

## 1. 候选方法

这个候选保留 frozen BRTC-LC v1 的五关节射线三角化、可靠性 gate 和 action cap，只替换
最终的 group-median + 离散 layout consensus。

对一个 cut 中每个 accepted person `i`，只优化沿其 first-post pelvis ray `r_i` 的标量
`a_i`：

```text
corrected_root_i = post_root_i + a_i * r_i
```

联合目标为：

```text
mean_{matched pairs i,j}
    || (post_i + a_i r_i) - (post_j + a_j r_j)
       - (pre_i - pre_j) ||²

+ prior_weight * mean_{accepted i} ||a_i - a_i^BRTC||²
```

其中 `a_i^BRTC` 是 frozen BRTC 经过原 gate 和 `±2 m` cap 后、进入 group/layout
consensus 之前的 individual signed action。所有 pair 和 prior 分别按数量归一化，使同一个
prior weight 不随人数机械变化。

约束与 fallback：

- accepted person 是最小二乘变量；
- rejected matched person 的 action 固定为零，但可作为 pair-layout anchor；
- unmatched post person 完全不进入求解，保持 exact B0；
- 最终 action 仍 clip 到 frozen BRTC 的 `±2 m`；
- 少于两个 matched person 时没有 pair 约束，accepted person 精确返回原 individual BRTC
  action；
- camera 只读，求解器不返回任何 camera update。

## 2. 开发、冻结和确认协议

唯一扫描参数是 ridge `prior_weight`：

```text
0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
1, 3, 10, 30, 100, 300, 1000
```

顺序严格为：

1. 只加载 `three offset0` 的 41 cuts / 122 人；
2. 扫 prior，按 root、joint、vertex、harm 的预设顺序选择；
3. 将策略写入
   `FROZEN_RAY_LAYOUT_LS_POLICY_BEFORE_CONFIRM.json`；
4. 记录 SHA256 与 mtime；
5. 再加载 `three offset1`、dance 和 box；
6. 确认结果不允许回调 prior。

冻结结果：

```text
selected prior = 3
eligible under BRTC layout safety = 0 / 14
dev_go = false
policy sha256 = 8da165866550de1486fc3e59de0492834425b6690d8e16c03cd77d778cab5810
policy mtime ns = 1785527630452676030
confirm report mtime ns = 1785527662912483291
```

冻结文件明确早于确认报告。尽管开发阶段已是 NO-GO，仍按预先约定用冻结 `prior=3` 跑完
所有确认集，以判断失败是否可复现。

## 3. Three offset0 prior sweep

| Prior | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.543098 | 0.575305 | 0.553914 | 0.111754 | 0.298316 | 46.7% |
| 0.001 | 0.535576 | 0.567947 | 0.546490 | 0.111023 | 0.293343 | 46.7% |
| 0.003 | 0.519034 | 0.551779 | 0.530182 | 0.104674 | 0.285853 | 45.1% |
| 0.01 | 0.464601 | 0.498783 | 0.476720 | 0.102967 | 0.283985 | 41.0% |
| 0.03 | 0.393926 | 0.431466 | 0.408489 | 0.101205 | 0.281539 | 33.6% |
| 0.1 | 0.317327 | 0.361260 | 0.337002 | 0.102511 | 0.276878 | 18.9% |
| 0.3 | 0.270323 | 0.317653 | 0.293804 | 0.107056 | 0.270262 | 12.3% |
| 1 | 0.238660 | 0.286356 | 0.263019 | 0.113334 | **0.262450** | 9.8% |
| **3** | **0.225906** | 0.272390 | **0.250356** | 0.121173 | 0.264384 | **5.7%** |
| 10 | 0.227439 | **0.271933** | 0.250657 | 0.131883 | 0.277457 | **5.7%** |
| 30 | 0.230635 | 0.274081 | 0.253001 | 0.138232 | 0.286726 | **5.7%** |
| 100 | 0.232301 | 0.275320 | 0.254319 | 0.141185 | 0.291151 | **5.7%** |
| 300 | 0.232846 | 0.275736 | 0.254759 | 0.142107 | 0.292557 | **5.7%** |
| 1000 | 0.233044 | 0.275888 | 0.254920 | 0.142439 | 0.293065 | **5.7%** |

开发集 frozen BRTC-LC v1 是：

```text
root/joint/vertex = 0.225088 / 0.270404 / 0.248604 m
pair distance/vector = 0.102222 / 0.260536 m
harm >1cm / >5cm = 15.6% / 6.6%
coverage = 88.5%
```

冻结 `prior=3` 是：

```text
root/joint/vertex = 0.225906 / 0.272390 / 0.250356 m
pair distance/vector = 0.121173 / 0.264384 m
harm >1cm / >5cm = 12.3% / 5.7%
coverage = 88.5%
```

它降低了 harm，但五个主误差均没有同时打过 BRTC，尤其 pair distance 恶化约 `18.5%`。
因此没有任何 prior 进入预设 layout-safe eligible 集合。

## 4. 冻结后的完整确认结果

### 4.1 Three offset1：42 cuts / 125 人，自动匿名关联

| Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | - | 0.377940 | 0.411723 | 0.389058 | 0.134098 | 0.329668 | 0.0% | 0.0% |
| BRTC-LC v1 | 88.0% | **0.231437** | **0.274493** | **0.252451** | **0.098351** | **0.258779** | 16.8% | 7.2% |
| Ray-layout LS | 88.0% | 0.234951 | 0.279498 | 0.257161 | 0.118628 | 0.265710 | **13.6%** | **6.4%** |

相对 BRTC，候选：

- root/joint/vertex 分别恶化约 `1.5% / 1.8% / 1.9%`；
- pair distance 恶化约 `20.6%`；
- pair vector 恶化约 `2.7%`；
- harm 有改善，但不足以抵消全部几何指标退化。

### 4.2 Dance：61 cuts / 122 人，post-hoc support

| Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | - | 0.382677 | 0.388187 | 0.378871 | 0.081752 | 0.104185 | 0.0% | 0.0% |
| BRTC-LC v1 | 99.2% | **0.125131** | **0.177804** | **0.152914** | **0.044141** | **0.078318** | 14.8% | 3.3% |
| Ray-layout LS | 99.2% | 0.133227 | 0.187889 | 0.161513 | 0.081975 | 0.112174 | **13.1%** | **0.0%** |

### 4.3 Box：78 cuts / 156 人，post-hoc support

| Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | - | 0.555707 | 0.586514 | 0.591622 | **0.059750** | 0.468762 | 0.0% | 0.0% |
| BRTC-LC v1 | 98.7% | **0.372345** | **0.421610** | **0.434528** | 0.063069 | **0.427334** | 11.5% | **6.4%** |
| Ray-layout LS | 98.7% | 0.376725 | 0.428748 | 0.440285 | 0.150315 | 0.465173 | **9.6%** | 7.7% |

### 4.4 Dance + box combined

| Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | - | 0.479773 | 0.499479 | 0.498256 | 0.069405 | 0.308768 | 0.0% | 0.0% |
| BRTC-LC v1 | 98.9% | **0.263855** | **0.314616** | **0.310942** | **0.054762** | **0.274169** | 12.9% | 5.0% |
| Ray-layout LS | 98.9% | 0.269866 | 0.323047 | 0.317946 | 0.120324 | 0.310260 | **11.2%** | **4.3%** |

combined 相对 BRTC：

- root/joint/vertex 恶化约 `2.3% / 2.7% / 2.3%`；
- pair distance 从 `0.0548` 增至 `0.1203 m`，恶化约 `119.7%`；
- pair vector 恶化约 `13.2%`；
- harm 略降，但布局退化过大。

## 5. 为什么最小二乘的 observable objective 降了，GT layout 却更差

开发集 `prior=3` 的求解器确实把自身使用的预测布局目标从均值 `0.3760` 降到了
`0.3212`，condition number 均值只有 `1.33`，0 次 action clipping。因此失败不是数值病态、
矩阵奇异或 cap 截断。

真正原因是：

```text
last-pre Human3R predicted root pair-vector
不是
真实且跨 shot 稳定的多人布局 GT
```

最小二乘把 pre root pair-vector 当作连续、可精确追踪的目标，所以会强制每个人沿不同射线
移动，直到更贴合这份本身有偏差的 pre 结构。它减少的是 observable self-consistency，不是
GT layout error。

frozen BRTC 的 group median + 离散 `{0, .25, .5, .75, 1}` residual lambda 更粗糙，但也更
鲁棒：共同平移不改变 pair layout，离散 lambda 限制了个体射线 action 对预测 pre-layout 的
过拟合。这个实验说明不能仅凭“预测 layout loss 下降”接受一个精对齐候选。

## 6. 最终决定

```text
NO_GO_JOINT_RAY_LAYOUT_LS
```

不替换 frozen BRTC-LC v1，也不在 offset1、dance 或 box 上重新调 prior。可以保留的研究结论
是：

1. joint scalar ray solve 在数值上稳定，且能减少自身 observable layout objective；
2. pre predicted root layout 不能作为连续最小二乘的硬目标；
3. 任何后续 layout refinement 必须给 pre layout 建模置信度、使用 robust bounded influence，
   或只允许不会改变 pair layout 的 shared group translation；
4. harm 降低不能单独作为 promotion 依据，必须同时检查 pair distance/vector。

## 7. 产物与复现

运行：

```bash
cd /data/wangzheng/iJCV-CODE/Movie3R

.venv/bin/python -m pytest -q \
  versions/v14/tests/test_b0_person_triangulation_ray_layout_least_squares.py

.venv/bin/python versions/v14/probe_b0_brtc_ray_layout_least_squares.py --phase dev
.venv/bin/python versions/v14/probe_b0_brtc_ray_layout_least_squares.py --phase confirm
```

代码与报告：

```text
versions/v14/b0_person_triangulation_ray_layout_least_squares.py
versions/v14/tests/test_b0_person_triangulation_ray_layout_least_squares.py
versions/v14/probe_b0_brtc_ray_layout_least_squares.py
versions/v14/docs/V14_BRTC_JOINT_RAY_LAYOUT_LEAST_SQUARES_20260801.md

output/v14/fine_alignment_research/b0_brtc_ray_layout_ls/
  FROZEN_RAY_LAYOUT_LS_POLICY_BEFORE_CONFIRM.json
  dev_report.json
  dev_report.md
  confirm_report.json
  confirm_report.md
```
