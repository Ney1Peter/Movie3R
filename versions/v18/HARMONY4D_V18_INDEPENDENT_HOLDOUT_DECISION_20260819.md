# Movie3R-v18：Harmony4D 独立 holdout 决策

日期：2026-08-19  
最终决策：**不晋级 v18，Harmony4D 全量 test 继续使用冻结的 Movie3R-v17 MultiCue-Safe。**

## 1. 独立验证协议

候选在开发集上冻结以后，才运行三个未参与调参的 Harmony4D train 动作：

- `04_sword_part1`
- `08_ballroom2`
- `13_mma2`

每个动作固定 150 帧（75 pre + 75 post）和四档 camera-pair，共 12 个预注册 case。9 个 case 可评测；`08_ballroom2` 的 3 个 case 因没有共同初始人体匹配而由统一 evaluator 判为 unavailable，对 Human3R、v17、v18 完全相同。

候选与 v17 仅有一个差异：边界共同平移由 `boundary_blend=1.0` 改为 `0.75`。所有模型预测缓存、身份匹配和评测输入保持一致。

## 2. 聚合结果

| 指标 | v17 reference | v18 candidate | 相对变化 |
|---|---:|---:|---:|
| W-MPJPE ↓ | 549.337 mm | 548.412 mm | -0.17% |
| WA-MPJPE ↓ | 223.743 mm | 229.471 mm | +2.56% |
| MPJPE ↓ | 95.221 mm | 95.221 mm | ≈0.00% |
| MPVPE ↓ | 114.500 mm | 114.500 mm | ≈0.00% |
| Accel ↓ | 111.328 mm/frame² | 114.374 mm/frame² | +2.74% |
| ATE-Sim3 ↓ | 0.01873 m | 0.01905 m | +1.67% |
| ATE-SE3 ↓ | 0.30300 m | 0.33240 m | +9.70% |
| Seam-root ↓ | 0.78470 m | 0.91695 m | +16.85% |
| IDF1 ↑ | 0.61374 | 0.61374 | 0.00% |
| Coverage ↑ | 0.88481 | 0.88481 | 0.00% |

v18 的多指标几何分数为 v17 的 `1.037842`，总体恶化约 3.78%。ATE-SE3 与 Seam-root 超过预注册的 5% 安全上限，因此安全门槛也没有通过。

## 3. 为什么开发集的小收益没有泛化

| 动作 | 可评测 case | W 变化 | WA 变化 | Accel 变化 | 结论 |
|---|---:|---:|---:|---:|---|
| sword_part1 | 4 | +2.91% | +14.08% | +7.50% | 明显恶化 |
| ballroom2 | 1 | 0.00% | 0.00% | 0.00% | gate fallback，与 v17 相同 |
| mma2 | 4 | -1.76% | -4.29% | -0.09% | 仅此动作获益 |

`blend=0.75` 的作用是减弱边界平移。在 MMA 上它缓解了过冲，但在 sword 上造成纠正不足。其改善只覆盖一个动作，W、WA、Accel 均没有达到“聚合改善不低于 0.1%，并覆盖严格多数动作”的稳定性要求。这说明固定减小 blend 不是可泛化的新方法，只是动作相关的折中。

## 4. 冻结结论

1. 150 帧继续作为论文正文统一协议；60/90/120 帧只进入附录长度消融。
2. 不把开发集上约 0.08% 的微小收益命名为 v18，也不在 holdout 上继续调参。
3. Harmony4D 全 test-capture 使用原 v17：`boundary_blend=1.0`、prediction-only gate、不确定时 exact parent fallback。
4. 后续若研究自适应 blend，必须在新的开发数据上学习或制定规则，并使用新的独立数据验证，不能复用本 holdout。

冻结方法：

```text
versions/v17/harmony4d/frozen_multicue_candidate.json
SHA256 ae2bc503ca5abdb0735abea231b6b953c0463d2e33f96741469191ad71adbafe
```

结果工件：

```text
output/v18_harmony4d/holdout/decision.json
SHA256 bae2e877faa672730cad8449edaed46d3e43e809ba752b173d307f600dcfb9b1

04_sword_part1/cs150.json
SHA256 18e871bb7cca20ddd86bb8cb47fa3f4901e43f368f4ad641ad6fc85f1c839777

08_ballroom2/cs150.json
SHA256 7a7bc7192b3a1d4b8ebda72ecc6bb5aa39be3169d3fadcae35da0e9c246c6a18

13_mma2/cs150.json
SHA256 a0c0533a3bea78e475336ebe624d24d606835843638c5c9bd5f37bce410728b3
```

