# V10 BEDLAM Motion / Global-State Integrator 验证记录

日期：2026-07-13

## 1. 验证目标

这次不是验证 detector，也不是验证完整 Human3R 系统，而是先验证 V10 最核心的问题：

> 已知哪里发生 shot boundary 的情况下，模型能不能在严格流式条件下，把新的 local shot 接回历史 global state？

严格流式约束如下：

- 当前 boundary 的对齐只使用历史 global 输出和当前 shot 的第一帧 local 输出；
- 不看未来帧；
- 不先跑完整个 segment 再回头修；
- boundary 之后预测一个 segment-to-global transform，并缓存给本 segment 后续帧使用。

## 2. 数据构造

使用 BEDLAM `seq_000021` 过滤后的 29 帧：

```text
0000, 0005, ..., 0140
```

每一帧有 4 个人，NPZ 中的人体顺序和 body mask 顺序已经检查过，可以一一对应。

为了避免模型记住某一个固定世界坐标，每个 synthetic episode 会先随机生成一个全局坐标系：

```text
clean BEDLAM GT -> random global gauge -> target global trajectory
```

然后再把这个 target trajectory 切成多个 segment，并对后面的 segment 加随机 SE3 扰动：

```text
A segment: 保持 target global
B/C segment: 加随机旋转和平移，模拟 local reset 后的新坐标系
```

这样 current-only 模型无法只靠当前帧猜出全局位置，必须依赖历史 global state。

## 3. 对比方法

| 方法 | 作用 |
|---|---|
| raw_perturbed | 不修正，作为下限 |
| fixed_explicit_se3 | 用历史速度预测的人体 root 位置做显式 SE3 粗对齐 |
| current_only_mlp | 只看当前 local frame，不看历史 |
| history_current_integrator | 看历史 global state + 当前 local frame，直接预测 SE3 |
| explicit_se3_residual_integrator | 先显式 SE3 粗对齐，再学习一个小 residual |
| oracle_se3_upper | 用 GT 当前帧求最优 SE3，上限参考 |

## 4. 当前结果

实验脚本：

```text
scripts/v10_bedlam_motion_integrator_probe.py
```

脚本支持两种 trajectory source：

```text
--trajectory_source bedlam_gt
--trajectory_source human3r_saved
```

其中 `human3r_saved` 会读取 `demo.py --save` 输出中的 `camera/*.npz` 和 `smpl/*.npz`，把 Human3R 连续输出作为 pseudo-global target，用于第二阶段验证。

结果目录：

```text
output/v10_bedlam_motion_integrator_probe/gt_synthetic_streaming_globalgauge
```

主要指标如下，越低越好：

| Variant | Root Trans | Root Rot | Cam Trans | Cam Rot | Boundary Jump | Velocity |
|---|---:|---:|---:|---:|---:|---:|
| raw_perturbed | 2.8730 | 37.71 | 5.1981 | 37.71 | 5.0944 | 0.3671 |
| fixed_explicit_se3 | 0.0063 | 3.47 | 0.5706 | 3.47 | 0.0062 | 0.0008 |
| current_only_mlp | 2.8132 | 40.70 | 5.4823 | 40.70 | 4.1863 | 0.3025 |
| history_current_integrator | 0.8696 | 19.52 | 1.5467 | 19.52 | 1.1079 | 0.0808 |
| explicit_se3_residual_integrator | 0.0237 | 0.40 | 0.0542 | 0.40 | 0.0228 | 0.0017 |
| oracle_se3_upper | 0.0000 | 0.08 | 0.0000 | 0.08 | 0.0000 | 0.0000 |

## 5. 结论

这个 probe 支持三个判断：

1. `current_only_mlp` 在随机全局坐标下基本失效，说明只看当前帧不够。
2. `history_current_integrator` 明显优于 current-only，说明历史 global state 是有用的。
3. `explicit_se3_residual_integrator` 最稳定，接近 oracle，说明更合理的结构不是让网络暴力预测完整 SE3，而是：

```text
显式几何粗对齐 + 因果 global-state residual 细修
```

这和当前 V10 方向一致：强约束问题先用可靠几何 proposal 解决大部分，再让可学习模块利用历史 state 做小范围修正。

## 6. Human3R 输出域验证：history-current direct + residual

2026-07-14 已经完成 Human3R 输出域的第二阶段验证。

这一步不再直接使用 BEDLAM GT 轨迹，而是先用原版 Human3R 连续跑 BEDLAM seq21 的 29 帧输出，把它当成 pseudo-global target。然后人为把后面的 segment 加随机 SE3 扰动，模拟 local reset 之后每段落在不同局部坐标系里，再测试 streaming integrator 能否把这些局部段接回连续 Human3R 输出。

结果目录：

```text
output/v10_bedlam_seq21_original_human3r/v10_human3r_domain_history_residual_v1
```

可视化 payload：

```text
output/v10_bedlam_seq21_original_human3r/v10_human3r_domain_history_residual_v1_payload
```

这次新增并重点验证了一个更合理的结构：

```text
history_current_integrator 先直接预测粗 segment-to-global SE3
        ↓
把 local output 变到 coarse global
        ↓
history_direct_residual_integrator 再预测一个小 residual SE3
        ↓
最终 segment transform 被缓存，本 segment 后续帧复用
```

它和 `explicit_se3_residual_integrator` 的区别很关键：

- `explicit_se3_residual_integrator` 的粗对齐来自手工显式 SE3；
- `history_direct_residual_integrator` 的粗对齐来自可学习的 history-current direct head；
- residual head 不是从零预测完整变换，而是在 direct head 已经粗对齐的结果上做小修正；
- 整个过程仍然是严格流式的：boundary 当前帧只看历史 global state 和当前 local 输出，不看未来帧。

最终组合方式：

```text
R_final = R_residual @ R_direct
t_final = R_residual @ t_direct + t_residual
```

residual head 的输入包含：

- 原始 history-current feature；
- direct head 预测的 coarse SE3；
- coarse 对齐后的 root translation / root rotation；
- coarse 对齐后的 camera translation / camera rotation。

也就是说，它不是盲目再预测一次，而是在“已经粗接上的状态”上判断哪里还需要细修。

### Human3R 输出域指标

越低越好：

| Variant | Root Trans | Root Rot | Cam Trans | Cam Rot | Boundary Jump | Velocity | Non-boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_perturbed | 5.3002 | 38.46 | 2.8371 | 38.46 | 9.5535 | 0.8565 | 0.1875 |
| fixed_explicit_se3 | 0.9353 | 68.19 | 7.3069 | 68.19 | 2.0954 | 0.5811 | 0.4646 |
| current_only_mlp | 4.8574 | 37.77 | 2.8997 | 37.77 | 7.4762 | 0.7087 | 0.1881 |
| history_current_integrator | 0.8264 | 10.26 | 0.7319 | 10.26 | 1.0808 | 0.1171 | 0.0430 |
| history_direct_residual_integrator | 0.4919 | 9.52 | 0.3998 | 9.52 | 0.6663 | 0.0835 | 0.0387 |
| explicit_se3_residual_integrator | 3.5561 | 72.13 | 6.0702 | 72.13 | 5.3065 | 0.8208 | 0.4757 |
| oracle_se3_upper | 0.0000 | 0.08 | 0.0000 | 0.08 | 0.0000 | 0.0000 | 0.0000 |

### 新结论

这个结果非常重要：

1. `history_current_integrator` 已经明显优于 current-only，说明历史 state 确实是必要信息。
2. `history_direct_residual_integrator` 又明显优于单纯 direct，说明“先粗接，再小修”比一次性预测完整 SE3 更稳。
3. `fixed_explicit_se3` 和 `explicit_se3_residual_integrator` 在 Human3R 输出域反而失败，说明直接用显式 SE3 粗对齐不一定可靠，尤其 Human3R 输出里人体、相机、多人顺序和局部坐标噪声会让手工几何 proposal 不稳定。
4. 当前更有潜力的路线不是“手工 SE3 + residual”，而是：

```text
可学习 history-current 粗对齐 + 可学习 residual 细修
```

这更符合 V10 的核心动机：不是简单后处理，也不是全局 BA，而是在严格流式状态下学习如何把新 shot 接回历史 global state。

## 7. 下一步

后续重点应该沿着 `history_direct_residual_integrator` 继续推进：

1. 从 synthetic perturbation 进入真实 local-reset Human3R segment 接回 global 的验证。
2. 把 detector 暂时设为 oracle，先专注验证 alignment / integrator 本身。
3. 保留 `history_current_integrator` 和 `history_direct_residual_integrator` 作为主要对照。
4. 不再优先推进 `explicit_se3_residual_integrator`，除非后面能找到更稳定的显式粗对齐 proposal。
5. 继续保持严格流式：当前 boundary 只使用历史 state 和当前帧 local 输出，不能先看完整 segment，也不能回头改历史帧。

这个版本目前是 V10 中最值得继续扩展的分支。
