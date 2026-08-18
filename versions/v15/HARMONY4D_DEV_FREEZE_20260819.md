# Harmony4D dev 冻结记录（测试前）

日期：2026-08-19  
状态：**方法与阈值已冻结；尚未运行任何 Harmony4D test forward / test metric**

## 1. 冻结依据

本轮只使用官方 train 包：

- `train/01_hugging.zip`：1 capture × 4 个角度层；
- `train/03_grappling2.zip`：1 capture × 4 个角度层；
- `train/15_mma4.zip`：2 captures × 4 个角度层。

共 16 个 H4D-CS150 dev cases，每个 case 为 75 pre + 75 post，全部由 GT
相机/可见性预先选择，未按 Movie3R 结果挑样本。

冻结 manifest：

| Manifest | Cases | 文件 SHA256 |
|---|---:|---|
| `h4d_cs150_dev_train01.jsonl` | 4 | `6e813ccf4bd71a41a3302d58ae75d44fcb255d0d6ad4a23db685d149f8fd168d` |
| `h4d_cs150_dev_train03.jsonl` | 4 | `e03a04e04e506fda448468212d62b759c70ccce0adfeacf64f42d22c22d32367` |
| `h4d_cs150_dev_train15.jsonl` | 8 | `9547bcb9e573f1a7d250b1e7c14d76da5cca17dc061c0abd754d7e4d9936b48c` |

运行结果：16/16 GPU inference 成功，16/16 evaluator 成功，无隐藏失败 case。

## 2. 坐标适配冻结

Harmony4D 不同包的官方坐标元数据不一致：

- hugging：官方 `COLMAP + aria01 similarity` 可直接使用，22 个相机最大
  median reprojection 为 0.389 px；
- MMA：官方 Aria transform 与发布 SMPL world 不一致，自动回退到
  `SMPL45 → poses2d45 static PnP`，两个 capture 的 20 个相机最大 median
  分别为 1.648 px、1.229 px；
- grappling2：官方包不含 Aria transform，使用同一 PnP adapter，20 个相机
  最大 median 为 1.377 px。

所有相机 P95 均低于冻结的 15 px gate，overlay 与 bbox/pose2d 对齐。PnP 只存在于
GT evaluator adapter，RGB runtime 不读取任何 GT。

## 3. Frozen v15 的真实 dev 结果

以下为 16-case clip macro；它们是 Movie3R 自有协议结果，不冒充 Multi-THuMBS
官方复现：

| 方法 | W-MPJPE ↓ | WA-MPJPE ↓ | MPJPE ↓ | ATE Sim(3) ↓ | IDs/clip ↓ | IDF1 ↑ | Coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 656.8 | 279.8 | 104.1 | 0.121 | 4.94 | 0.486 | 0.903 |
| B0 only | 652.3 | 301.4 | 104.2 | **0.020** | 5.44 | 0.412 | **0.907** |
| Frozen B0 + ID | 653.3 | 310.3 | 107.2 | **0.020** | 7.12 | 0.526 | 0.897 |
| Frozen full v15 | **599.3** | **298.8** | 123.3 | **0.020** | 5.69 | 0.493 | 0.801 |
| OS-BRTC candidate | 652.7 | 310.1 | 107.2 | **0.020** | 7.12 | 0.526 | 0.897 |

结论：

1. B0 显著恢复 camera gauge，ATE 从 reset 的 1.49 m 降到约 0.02 m；
2. frozen full v15 的无条件 BRTC 平均 W 有改善，但会把 coverage 从约 90% 降到
   80%，并使 MPJPE/部分 layout 明显恶化，不能作为安全主结果；
3. OS-BRTC 成功拒绝 15/16 个高风险更新，只接受 hugging-medium；它消除了灾难
   coverage harm，但只靠它不会稳定超过 B0；
4. 因此 test 中必须同时保留 frozen v15 与 safe candidate，不能只报告安全分支。

## 4. Detector 冻结

两种流式 RGB-only detector 在 16 个 dev clips 上均为：

- TP = 16，FP = 0，FN = 0；
- precision / recall / F1 = 1.0；
- first-positive 恰为预注册 boundary：16/16；
- future frames = 0。

默认论文部署行使用 causal GRU；static logistic 作为消融。test 开始后不再修改
detector checkpoint、特征、阈值、first-positive policy。

## 5. BRTC 安全门控冻结

OS-BRTC 只读 runtime geometry，必须同时满足：

1. pre/post detection count 相同且全匹配；
2. 所有 matched people 通过原 BRTC evidence gate；
3. group shift norm ≤ 0.15 m；
4. max median ray gap ≤ 0.10 m；
5. observable layout objective relative gain ≥ 10%。

拒绝后 exact fallback 到 B0 + persistent ID，再运行同一个 causal C1/adaptive
安全分支；不删除 case，不使用 GT 判定，不改变相机。

## 6. ID 策略实验与冻结

### 6.1 失败策略（保留记录）

“每帧相邻人体 Hungarian + 0.50 m dustbin”在 close interaction 中会因重建抖动
不断开新 track：

- frozen B0 native IDs total：87；
- 逐帧 dustbin candidate IDs total：117；
- mean IDF1：0.412 → 0.484，但 IDs 明显恶化。

该策略不晋升。

### 6.2 晋升策略：Boundary-Permutation ID

最终 ID 策略：

1. B0 先把第一张 post frame 带到 pre world；
2. 只在 boundary 做一次 anonymous root/torso/centered-pose permutation；
3. 将得到的 persistent ID 绑定到 Human3R 的 causal native slot；
4. shot 内不再反复对第一帧重匹配；新 slot 才创建新 ID；
5. 所有 detection 均保留，不因 first-frame reference 缺失而被丢弃。

在完全相同 M3 geometry/assignment 下：

| ID 策略 | IDs total ↓ | IDs/clip ↓ | mean IDF1 ↑ | association acc. ↑ | Coverage |
|---|---:|---:|---:|---:|---:|
| B0 native slot | 87 | 5.44 | 0.412 | 0.666 | 0.907 |
| Adjacent unbounded | 85 | 5.31 | 0.505 | 0.758 | 0.907 |
| **Boundary-Permutation ID（runner exact）** | **70** | **4.38** | **0.585** | **0.886** | **0.907** |

相对 B0 native，IDs total 下降 19.5%，IDF1 提升 0.173，coverage bit-exact。
该模块符合 online、causal、无额外预训练模型的主线，因此晋升。这里报告的是
runner 实际沿用的 frozen `anonymous_match` boundary pairs；独立穷举 probe 得到 69、
0.590、0.897，但有一个 MMA-extreme case 的 pair 与 runtime 不同，因此它只作为
诊断上界，不作为冻结部署结果。

## 7. Test 前最终方法定义

新增方法行不覆盖 frozen v15：

- M13：`B0 + Boundary-Permutation ID`；
- M14：`M13 + OS-BRTC + C1 + adaptive joint`，oracle boundary；
- M15：`M14 + causal GRU detector`，**默认部署/论文主候选**；
- M16：`M14 + static logistic detector`，detector 消融。

测试阶段固定：

- checkpoint 与 SHA 不变；
- OS-BRTC 五项阈值不变；
- Boundary-Permutation ID 不增加 appearance/Re-ID backbone；
- detector 不变；
- 75 + 75、camera angle strata、capture selection seed 不变；
- test 只用于一次冻结评估，不用于调参。

## 8. 进入 test 的判定

允许进入 test，原因：

1. adapter/reprojection 已闭环；
2. M0–M12 完整 dev 消融已完成；
3. detector 在 dev 无误报/漏报；
4. 不安全 BRTC 已被可解释 gate 隔离；
5. ID 新策略在 16 cases 上同时改善 IDs、IDF1、association 且不损失 coverage；
6. 所有失败策略有记录，没有看 test 后选择方法。

下一步严格先完成 7 个 test 包的 schema audit 与全局 manifest SHA256 冻结，之后
才能启动第一次 test GPU forward。
