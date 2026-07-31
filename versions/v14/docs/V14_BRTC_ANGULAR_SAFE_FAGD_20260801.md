# V14 Angular/Reprojection-Safe Group Damping 实验报告

> 日期：2026-08-01
>
> 约束：frozen B0 + frozen BRTC-LC v1；无新模型、无内参、无 GT 决策、无未来帧；
> camera 不更新；只使用已有 CPU cache；不修改 V9、completeness runtime 或已有 evaluator。

## 1. 结论

本实验验证“人体修正不应让 fixed post camera 下的图像 ray 方向变化过大”能否成为共享
group translation 的在线置信度。`three offset0` 冻结出的策略是：

```text
statistic = all-joints median angular displacement
budget    = 4.0 degrees
alpha     = largest value in {0.50, 0.55, ..., 1.00} satisfying budget
```

Development、offset1 和 box 均改善，但冻结后 dance 的 root/joint 退化；EgoHumans 又因两个
eligible boundary 的 BRTC ray angle 已低于预算而全部选择 `alpha=1`，与 BRTC v1 bit-exact，
没有任何新收益。因此最终结论是：

```text
NO_GO_ANGULAR_SAFE_FAGD_HELDOUT
NO_GO_ANGULAR_SAFE_FAGD_DEPLOYABLE
```

核心失败规律是：**图像 ray angular displacement 衡量 2D/reprojection 扰动大小，却不能判断
world-depth group correction 的方向或是否过激。** 同样的 4° budget 在 offset1/box 中倾向
筛出有益阻尼，在 dance 中却对 52 个人平均造成 `+9.936 mm` 的 root 退化。

## 2. 方法

### 2.1 输入与 frozen BRTC 分解

每个 boundary 输入：

```text
pre_camera, post_camera
pre_people, post_people
anonymous matches
```

每个人包含 world `root/joints/vertices`。首先完整运行 frozen BRTC-LC v1：

```text
frozen_final_i = group + selected_lambda * (individual_i - group)
```

记 frozen individual residual：

```text
residual_i = selected_lambda * (individual_i - group)
```

### 2.2 严格动作门

只有满足：

```text
accepted_count == matched_count
    == max(len(pre_people), len(post_people)) > 0
```

才允许选择新的 group alpha。任一 rejected、incomplete match、unmatched person 或人数变化
都会直接返回第一次 frozen BRTC 调用产生的 corrected geometry，不重建数组，确保 exact v1。

### 2.3 无内参 angular statistic

固定 post camera center 为 `C`，对每个 matched person 的 post joints `p_ij`，候选 alpha 的
最终 shift 为：

```text
s_i(alpha) = alpha * group + residual_i
```

应用修正前后的 world ray：

```text
r_before = p_ij - C
r_after  = p_ij + s_i(alpha) - C
```

角度：

```text
theta_ij(alpha) = acos(
    dot(r_before, r_after) / (norm(r_before) * norm(r_after))
)
```

它只需要 camera center 和 3D joints，不需要焦距、主点或图像。Development 扫描：

- joints：SMPL-X core 11 joints 或全部 127 joints；
- aggregation：median 或 p90；
- budget：`0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15°`；
- alpha：`0.50–1.00`，步长 `0.05`。

每个 boundary 选择满足 budget 的最大 alpha；若 `alpha=0.5` 仍不能满足，只选择角度最小的
alpha 并记录 `budget_satisfied=False`。alpha 为 1 或 strict gate 不成立时，输出 exact v1。

### 2.4 不变性

候选只改变所有 accepted 人共享的 `alpha * group`，individual residual 不变，所以当前
boundary 的两两 root vector/distance 数学上与 BRTC v1 相同。camera 始终 bit-exact。

## 3. Development 与冻结

只使用 `three offset0` 的 41 个 cases；held-out 尚未加载。48 个 statistic/budget 组合中有
7 个同时满足：五项误差、coverage、`>1 cm/>5 cm` harm 不差于 v1，root/joint/vertex 严格
改善，pair layout invariant，且至少一个 boundary 非平凡动作。

| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | .225088 | .270404 | .248604 | .102222 | .260536 | 15.6% | 6.6% |
| frozen angular-safe | **.222789** | **.268228** | **.246239** | .102222 | .260536 | 15.6% | 6.6% |

冻结策略为 `all_median / 4.0°`。30/41 个 case 满足 strict BRTC gate，实际只对 6/41 动作；
全体 mean alpha 为 `0.9707`，动作 alpha 分布为：

```text
0.70:1, 0.75:1, 0.80:2, 0.85:1, 0.90:1, 1.00:35
```

策略 checksum：

```text
441ce38142d47ff1325ca338c96a3e7250ae1386ac336169af468ce3258ae1dc
```

## 4. 冻结后的 same-visibility 结果

| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Damped | Mean alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| offset1 | BRTC v1 | .231437 | .274493 | .252451 | .098351 | .258779 | 7.2% | 0 | 1.000 |
| offset1 | angular-safe | **.228636** | **.272004** | **.249786** | .098351 | .258779 | 7.2% | 6 | .980 |
| dance | BRTC v1 | **.125131** | **.177804** | .152914 | .044141 | .078318 | 3.3% | 0 | 1.000 |
| dance | angular-safe | .129366 | .177947 | **.152882** | .044141 | .078318 | 3.3% | 26 | .904 |
| box | BRTC v1 | .372345 | .421610 | .434528 | .063069 | .427334 | 6.4% | 0 | 1.000 |
| box | angular-safe | **.361822** | **.406068** | **.419101** | .063069 | .427334 | 6.4% | 28 | .919 |

offset1 的 root/joint/vertex 改善 `2.801/2.489/2.665 mm`；box 改善
`10.523/15.542/15.427 mm`。但 dance：

```text
root:   +4.235 mm worse
joint:  +0.143 mm worse
vertex: -0.032 mm better
```

dance 的 26 个 damped boundaries 共 52 人，相对 v1 的 root 变化为：

```text
mean delta       = +9.936 mm
improve rate     = 44.2%
worst boundary   = +80.266 mm mean delta
largest person   = +116.221 mm delta
```

最伤的若干边界反而选择了强阻尼 `alpha=0.60–0.70`。selected angular score 都接近同一个
`4°` 上限，但 error 方向可以相反，说明 angular magnitude 不是 world correction sign/
calibration 的可靠 proxy。

## 5. Variable visibility 与回退

| Split | Cases | 人数变化 | 等人数替换 | Damped | 全集 exact v1 | 人数变化 exact v1 |
|---|---:|---:|---:|---:|---|---|
| three | 22 | 19 | 3 | 0 | True | **True** |
| dance | 29 | 18 | 11 | 7 | False | **True** |
| box | 12 | 12 | 0 | 0 | True | **True** |

所有人数变化 `49/49` exact v1，达到本实验要求。three 的 3 个等人数 replacement 也因
angular budget 选择 alpha=1 而偶然 exact；dance 的 11 个等人数 replacement 中有 7 个仍
动作。这再次说明 count + angle 不能证明 visible identity set 连续。

## 6. EgoHumans CPU cache

使用同一份 3×15-frame、6-cut、89.6% coverage 的 `001_legoassemble` cache：

| Method | W | WA | Root | Joint | Vertex | Pair dist | Pair vec | Root Accel | Joint Accel | Harm >5cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BRTC v1 | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 | 23.8% |
| angular-safe | 314.059 | 202.461 | 380.654 | 384.729 | 385.238 | 177.025 | 333.870 | 116.014 | 125.270 | 23.8% |

6 个边界中只有 2 个满足 strict full/all-accepted gate；这两个边界在 `alpha=1` 时 angular
median 已不超过 4°，因此 6/6 均选择 alpha=1，candidate geometry 与 BRTC v1 bit-exact：

```text
damped boundary = 0/6
geometry parity = true
camera max delta = 0
root/joint Accel = exact v1
```

Accel 没有退化只是因为本数据上完全没有动作，不构成时序改进证据。且该 Accel 仍是本地
重复 timestamp 的短链诊断，不是 Multi-THuMBS 官方协议。

## 7. accepted/rejected/unmatched 安全审计

独立 runtime 和单测确认：

- complete one-to-one 且 all accepted：只调整 shared group，residual 不变；
- 人数变化：exact frozen v1；
- 人数相同但 matching 不完整：matched 与 unmatched 全部 exact frozen v1；
- 任一 evidence rejected：全部 exact frozen v1；
- budget 选择为 alpha=1：exact frozen v1；
- camera 输入 bit-exact；
- pair layout 在实际动作 split 上与 v1 bit-exact。

相关 BRTC/strict/angular 测试：

```text
13 passed
```

## 8. 最终判断与可复用规律

该候选满足在线、无 GT、无模型、无内参、人数变化安全和 layout invariant，但没有跨动作
类型泛化，因此按预先规则归档为 NO-GO，不继续看 held-out 调 budget。

可复用结论：

1. Ray angular displacement 可以约束“投影方向改了多少”，不能判断“世界深度应该往哪改”。
2. dance 与 box 对相同 angular budget 呈相反 error response；单一图像角阈值不是通用
   group-depth calibration。
3. 若继续使用角信息，它只能作为 hard sanity cap，不能单独决定 correction alpha。
4. 后续更值得测试的是多人共享的显式刚体结构，例如 accepted 人共同估计的 shared/group
   SO(3) Kabsch，而不是继续扫描 angular budget。

## 9. 产物

```text
versions/v14/b0_person_triangulation_angular_safe_fagd.py
versions/v14/tests/test_b0_person_triangulation_angular_safe_fagd.py
versions/v14/probe_brtc_angular_safe_group_damping.py
versions/v14/eval_brtc_angular_safe_fagd_egohumans.py

output/v14/fine_alignment_research/brtc_angular_safe_fagd/
  DEV_SCAN.json
  DEV_RESULTS.md
  FROZEN_POLICY_BEFORE_HELDOUT.json
  HELDOUT_RESULTS.json
  HELDOUT_RESULTS.md
  egohumans/report.json
  egohumans/README.md
```

复现命令：

```bash
.venv/bin/python versions/v14/probe_brtc_angular_safe_group_damping.py --phase dev
.venv/bin/python versions/v14/probe_brtc_angular_safe_group_damping.py --phase freeze
.venv/bin/python versions/v14/probe_brtc_angular_safe_group_damping.py --phase validate
.venv/bin/python versions/v14/eval_brtc_angular_safe_fagd_egohumans.py
```

