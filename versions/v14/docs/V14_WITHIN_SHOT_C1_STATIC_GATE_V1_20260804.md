# V14 C1：流式 Shot 内静止人体稳定器（EMA-25）冻结报告

日期：2026-08-04  
状态：**内部冻结候选，允许进入统一端到端大规模指标；尚不能写成论文级普适 claim。**

任务与原始验收标准见
[`V14_STREAMING_WITHIN_SHOT_STABILITY_TASK_20260804.md`](V14_STREAMING_WITHIN_SHOT_STABILITY_TASK_20260804.md)。

## 1. 问题和最终 C1 方法

`B0 + BRTC-LC` 已在 boundary 做完相机粗对齐和首帧逐人显式 root 修正。BRTC 的 translation 在一个 post shot 内固定，因此它不会制造逐帧抖动；剩余的漂移来自 Human3R/P0 的 shot 内人体与相机预测噪声。

冻结方法在每个匿名人轨迹上运行，输入只包含当前及历史的 **预测** `camera, root, joints`：

```text
B0+BRTC root/joints/vertices + frozen B0 camera
  -> 将 root 与 joints 转到 predicted camera-local coordinates
  -> 计算当前 root step、去 root 后的 body-joint deformation
  -> static/moving hysteresis gate
  -> 仅 static: causal EMA(local_root, alpha=0.25)
  -> residual_local -> current camera rotation -> residual_world
  -> 对此人的 root/joints/vertices 加同一个 residual_world
  -> moving / unknown / short history: exact B0+BRTC fallback
```

固定 policy 位于
[`WITHIN_SHOT_STATIC_GATE_EMA25_V1_20260804.json`](../frozen/WITHIN_SHOT_STATIC_GATE_EMA25_V1_20260804.json)：

- warmup `2` 帧；
- static 进入：最近 root step 中位数 `<= 1cm/frame` 且 body deformation `<= 1.5cm/frame`；
- moving 退出：当前 root step `>2.5cm/frame` 或 body deformation `>3.5cm/frame`，并持有 `3` 帧；
- static 时 `EMA alpha=0.25`，单帧 residual cap `15cm`。

camera 不是优化变量也不经过代码写入：所有实验的 camera max absolute change 都是 `0.0`。

## 2. 严格性与实现产物

- 长序列 runtime-first cache builder：
  [`cache_streaming_within_shot_sequence.py`](../cache_streaming_within_shot_sequence.py)。先完成 RGB→P0→B0→BRTC，随后才读取 GT；
- 因果 policy、前缀一致性 self-test、GT-only evaluator：
  [`eval_streaming_within_shot_stability.py`](../eval_streaming_within_shot_stability.py)；
- P0 forward 为 CPU-only；没有 DA3、SLAM、ReID 或额外预训练模型；
- 每个 post stream 连续 25 帧，BRTC 只在 first post frame 提交一次 translation；
- evaluator GT 只定义 static/moving 标签并计算 root/joint/vertex/layout，不参与 runtime gate。

`three t0900 c3→c4` 也被完整尝试，但 post index 20 时 Human3R native detections 从 `[0,1,2]` 变为 `[0,2]`。它不满足 25 帧 track-continuity protocol，故明确记为 **visibility/track failure**，不用于下面的成功统计。

## 3. 固定 C1 策略的结果

所有数字是相对同一缓存的 B0+BRTC baseline。`static` 与 `moving` 仅由完成预测后 GT 25 帧净 root displacement 标注：分别 `<=5cm` 与 `>=10cm`；中间为 ambiguous，不计入两类约束。

| 25-frame stream | static / moving / ambiguous | static camera-local path | static root | moving net / path retain | all root / joint / vertex | layout vector | camera |
|---|---:|---:|---:|---:|---:|---:|---:|
| `three t1100 c4→c5`（开发） | 1 / 1 / 1 | **−38.4%** | +0.7% | 100% / 100% | −0.5% / −0.4% / −0.5% | −3.4% | 0 |
| `three t1200 c4→c5`（时间不重叠确认） | 2 / 1 / 0 | **−41.5%** | +0.4% | 100% / 100% | +0.2% / +0.2% / +0.3% | −0.8% | 0 |
| `box t590 c0→c1`（跨场景慢速运动审计） | 1 / 1 / 0 | −14.0% | −3.9% | 100% / **93.2%** | −1.3% / −0.6% / −1.2% | +4.6% | 0 |

对应原始 JSON：

```text
output/v14/within_shot_stability/c1_static_gate/dev_t1100/C1_FIXED_POLICY_SCAN.json
output/v14/within_shot_stability/c1_static_gate/confirm_t1200/C1_FIXED_POLICY_SCAN.json
output/v14/within_shot_stability/c1_static_gate/confirm_box_t590/C1_FIXED_POLICY_SCAN.json
```

## 4. 验收判定

| 冻结要求 | 结果 | 判定 |
|---|---:|---|
| B0 camera 不变 | 全部 `0.0` | 通过 |
| 两条独立连续 trajectory 的 static drift 至少降 30% | 38.4%、41.5% | 通过 |
| static GT root 不恶化超过 3% | 最差 +0.7% | 通过 |
| moving 位移保留至少 90% | 最差累计路径 93.2% | 通过 |
| all root/joint/vertex 不恶化超过 3% | 最差 +0.3% | 通过 |
| layout vector 不恶化超过 5% | 最差 +4.6% | 通过，但 margin 很小 |
| >5cm root harm 增量不超过 2pp | 三条均 0 | 通过 |

结论：**C1-EMA25 是当前可冻结的 shot 内稳定主线。** 它不是用滤波替代真正的人体运动，而是在可观测到“相机局部 root 与身体形变均低速”的人上做保守 residual；快运动人可被完全拒绝，慢运动的 worst observed trajectory 也保留了 93.2% path。

## 5. 必须如实保留的限制

1. `box` 的 moving person 有 9/25 帧被 gate 为 static，虽然净位移保持 100%、path 仍有 93.2%，但说明 slow motion 与静止仍有混淆；大规模指标必须单列 moving-path retention、gate precision/recall 和 worst-case track。
2. `three` 的开发和 1200 确认属于同一 capture sequence、同一 camera pair但时间不重叠；它们证明长序列规律可重现，不足以取代多数据集正式 test。
3. 这轮 policy grid 在 `three t1100` 被比较后选择 EMA-25；`t1200` 与 `box` 没有据其结果回调 threshold，但历史实验数据并非全新盲测。因此它是可用的内部冻结，而非最终论文 holdout。
4. track 中断不能靠硬 carry 掩盖。`t900` 的失败证明，后续大评测必须将 visibility、track continuity 和 fallback coverage 分开报告。

## 6. 下一阶段：进入统一大规模指标

现在不应继续盲调 alpha。下一步将 frozen `P0+B0+BRTC-LC+C1-EMA25` 以同一 forward 跑 MultiHuman/EgoHumans/Multi-THuMBS-overlap，统一保存每帧：camera、scene、anonymous track、BRTC translation、C1 gate/residual、root/joint/vertex。报告与原版 Human3R、P0/B0、B0+BRTC 的并列结果：

- spatial：root / joint / vertex / pairwise layout；
- temporal：static drift、moving net/path retention、root/joint Accel；
- system：camera safety、track coverage/ID、runtime/memory、cut 失败/abstention；
- Multi-THuMBS：只在确认其公开协议可复现后再用其官方口径对比，不把当前内部 delta2 指标冒充官方 Accel。

若大规模结果违反本页任一 safety gate，C1 退回 exact B0+BRTC，并将失败分布用于 C2（多人相对证据）或 C3（轻量 learned gate）；在此之前不改 B0、也不让人体修正相机。
