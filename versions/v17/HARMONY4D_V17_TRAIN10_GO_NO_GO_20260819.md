# Movie3R-v17：Harmony4D train10 首次未见验证与 Go/No-Go

日期：2026-08-19  
候选：`Movie3R-v17 MultiCue-Safe`  
协议：4 个预注册 camera strata，75 pre + 75 post，4/4 可评测，0 跳过。

## 1. 结论

**GO。** v17 通过预注册标准；方法、阈值和候选 JSON 不再修改，允许继续读取此前保留的 `train/11_karate3` 作为独立确认集。

## 2. Train10 主结果

| 指标 | v17 parent | v17 MultiCue-Safe | 变化 |
|---|---:|---:|---:|
| W-MPJPE (mm) | 483.8 | **448.9** | **-7.2%** |
| WA-MPJPE (mm) | 264.6 | **248.2** | **-6.2%** |
| Accel (mm/frame²) | 86.4 | **73.3** | **-15.2%** |
| RTE-H3R (%) | 12.99 | **10.52** | **-19.0%** |
| ATE-Sim3 (m) | 0.0293 | **0.0253** | **-13.7%** |
| ATE-SE3 (m) | 0.2518 | **0.1424** | **-43.5%** |
| Seam-root (m) | 1.140 | **0.730** | **-36.0%** |
| MPJPE (mm) | **96.65** | 97.16 | +0.53% |
| MPVPE (mm) | **109.84** | 110.45 | +0.55% |
| IDF1 | **0.79097** | 0.79022 | -0.00075 |
| Coverage | 0.95333 | **0.95417** | +0.00084 |
| IDs | 3.0 | 3.0 | 不变 |

## 3. Gate 行为

| Stratum | Gate | 预测侧原因 | translation | residual | W parent → v17 |
|---|---|---|---:|---:|---:|
| extreme | fallback | translation > 1.6 m | 1.629 m | 0.217 m | 737.9 → 737.9 |
| large | fallback | residual > 0.25 m | 1.118 m | 0.287 m | 355.1 → 355.1 |
| medium | accept | 三项均通过 | 1.164 m | 0.122 m | **443.2 → 324.5 (-26.8%)** |
| small | accept | 三项均通过 | 0.462 m | 0.230 m | **399.0 → 378.1 (-5.2%)** |

2 accept / 2 exact fallback。两个接受例都改善 W；没有 accepted case 出现恶化，更没有超过 20% 的灾难性恶化。

## 4. 预注册判定逐项核对

1. accepted catastrophic W：通过；
2. W、WA、Accel 至少两项改善且其余不明显恶化：三项全部改善，通过；
3. MPJPE/MPVPE 恶化不超过 5%：均约 0.5%，通过；
4. Coverage 不下降且 IDF1 下降不超过 0.01：通过；
5. 不是全 fallback：2/4 接受，通过。

## 5. 解释边界

- 这是候选冻结后的真正未见序列，不是已见官方 test regression。
- `probe` 内部旧的 exploration `passing=[]` 要求 W 至少改善 10%，比本次预注册 go/no-go 更严格；它不推翻上述预注册判定，也不会被改写。
- 单序列只有 4 个 clip，统计把握度有限；因此必须继续用完全相同的方法在 train11 做确认。
- train11 在本报告冻结前没有解压、推理或评测。
