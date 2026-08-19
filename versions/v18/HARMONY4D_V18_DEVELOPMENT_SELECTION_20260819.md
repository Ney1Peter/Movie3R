# Movie3R-v18：Harmony4D 开发集选择与冻结记录

日期：2026-08-19  
状态：开发选择完成；尚未查看独立 train holdout 与官方 test 指标

## 1. 实验范围

本轮只使用 Harmony4D train split 中预注册的三个动作：

- `02_grappling`
- `07_ballroom`
- `12_mma`

每个动作使用固定结构规则选出的 capture 和四档 camera pair，并分别测试 60、90、120、150 帧。共完成 48 个 prediction case；每组参数在同一 Human3R 预测缓存上评测，避免因重复推理产生不公平差异。超参空间、指标权重和晋级门槛均在运行前写入 `HARMONY4D_LENGTH_HPARAM_DEVELOPMENT_PLAN_20260819.md`。

## 2. 150 帧主协议结果

开发集的 150 帧报告包含 9 个可评测 case 和 3 个方法无关的 evaluator-unavailable case。23 组预注册候选中，约束下的最优候选为 `blend_075`：

```text
boundary_kind                 translation
boundary_blend                0.75
root_alpha / root_beta        0.50 / 0.02
gate_max_boundary_residual    0.25 m
gate_max_translation          1.60 m
gate_min_matches              2
```

| 指标 | v17 reference | v18 dev selected | 相对变化 |
|---|---:|---:|---:|
| W-MPJPE ↓ | 629.216 mm | 628.048 mm | -0.19% |
| WA-MPJPE ↓ | 322.276 mm | 321.243 mm | -0.32% |
| MPJPE ↓ | 92.967 mm | 92.967 mm | ≈0.00% |
| MPVPE ↓ | 110.976 mm | 110.976 mm | ≈0.00% |
| Accel ↓ | 100.971 mm/frame² | 100.966 mm/frame² | -0.005% |
| ATE-Sim3 ↓ | 0.01977 m | 0.01983 m | +0.29% |
| ATE-SE3 ↓ | 0.32632 m | 0.32336 m | -0.91% |
| Seam-root ↓ | 1.14111 m | 1.14155 m | +0.04% |
| IDF1 ↑ | 0.52839 | 0.52839 | 0.00% |
| Coverage ↑ | 0.86630 | 0.86630 | 0.00% |

加权几何分数为 v17 的 `0.999222`，即综合改善约 0.08%。候选满足全部安全约束，但提升很小，当前只能称为“进入独立验证的候选”，不能据此命名为最终 v18。

## 3. 长度消融结论

| 总帧数 | pre/post | W ↓ | WA ↓ | Accel ↓ | ATE-Sim3 ↓ | IDF1 ↑ | 是否可替换主协议 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 60 | 30/30 | 597.269 | 270.092 | 118.932 | 0.00960 | 0.53533 | 否 |
| 90 | 45/45 | 566.646 | 278.433 | 116.750 | 0.01288 | 0.52074 | 否 |
| 120 | 60/60 | 600.380 | 298.165 | 104.743 | 0.01624 | 0.54556 | 否 |
| 150 | 75/75 | 628.048 | 321.243 | 100.966 | 0.01983 | 0.52839 | 正文默认 |

短窗口降低了部分全局位置误差，但 60/90 帧的加速度误差明显上升，90 帧还降低了 IDF1；120 帧较均衡，却没有达到预注册的 5% 综合改善门槛。因此正文继续采用固定 150 帧协议。长度结果只作为在线上下文长度—质量权衡的附录消融，不能按 test case 挑选长度。

## 4. 冻结与下一步判定

holdout 候选文件同时包含：

1. `v16_0_m15_geometry`：审计 fallback 的 parent；
2. `v17_reference`：实际晋级基线，`boundary_blend=1.0`；
3. `v18_dev_selected`：待验证候选，`boundary_blend=0.75`。

冻结工件：

```text
output/v18_harmony4d/dev/selection.json
SHA256 1d7d059753631729854bb7ce4ef8987228173185cd199e3deae7d4dacc1c6510

versions/v18/harmony4d/frozen_dev_candidate.json
SHA256 f02c63aaaba3ed61e87d64f0d1b09285043399a07c14605ef824ae463f3f1578
```

下一步只在预注册且尚未参与选择的 `04_sword_part1`、`08_ballroom2`、`13_mma2` 上运行 150 帧验证。只有候选在至少两个核心短板指标上稳定优于 v17，同时保留 ATE、IDF1、MPJPE/MPVPE 和安全 fallback，才升级为 v18；否则最终方法继续冻结为 v17。

