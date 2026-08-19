# Movie3R-v17 Harmony4D 新留出集预注册

日期：2026-08-19  
状态：在解压、推理或读取 `train/10_karate2`、`train/11_karate3` 的任何新方法指标前冻结。

## 1. 为什么需要新留出集

官方 7-sequence test 已用于发现 v16 的强接触错误接受，因此只能作为 seen-test regression diagnostic，不能再作为 v17 的无偏最终验证。v17 的阈值只由此前已有的冻结开发报告确定：

- 至少 2 个预测人体匹配；
- boundary torso residual 不超过 0.25 m；
- boundary common translation 不超过 1.6 m；
- 全部信号只来自当前/历史预测，不读 GT，不读未来帧。

## 2. 数据顺序与不可偷看规则

### 第一阶段：v17 首次未见验证

```text
Harmony4D train/10_karate2
capture: 冻结 SHA-256 结构顺序中的第一个 coordinate-valid capture
camera pairs: small / medium / large / extreme 各一个
clip: 75 pre + 75 post = 150 frames
```

### 第二阶段：保留确认集

```text
Harmony4D train/11_karate3
```

在 `train10` 结果和 go/no-go 决策写入报告前，不解压、不推理、不评测 `train11`。

- 若 v17 在 train10 通过：方法和阈值保持不变，在 train11 做独立确认。
- 若 v17 在 train10 失败：train10 可转为诊断/开发；任何 v18 修改后，只能在仍未读的 train11 上做最终验证。
- 只有 coordinate/GT/evaluator 方法无关不可用时才可透明跳过；绝不因方法数值差而删除 case。

## 3. 固定比较方法

1. Strict Human3R；
2. Movie3R-v15；
3. B0 + boundary ID（v17 parent）；
4. Movie3R-v17 MultiCue-Safe。

## 4. 固定指标与聚合

主指标：W-MPJPE、WA-MPJPE、Accel、RTE-H3R、ATE-Sim3、ATE-SE3。  
保护指标：MPJPE、PA-MPJPE、MPVPE、IDF1、IDs、Coverage。  
Movie3R 专有指标：Boundary-root、Post-root、Seam-root、Seam-CHRGE、Pair-vector。

主聚合为 4 个 camera-stratum clip macro；train10 与 train11 都完成后另报 sequence macro 和分层 bootstrap。150 帧为正文主协议，短序列只允许作为附录长度消融，不能替换主表。

## 5. Train10 go/no-go

相对 v17 parent：

1. 不允许任何被 gate 接受的 case 出现 W-MPJPE 超过 20% 的灾难性恶化；
2. W-MPJPE、WA-MPJPE、Accel 至少两项改善，另一项不得恶化超过 2%；
3. MPJPE/MPVPE 恶化不超过 5%；
4. Coverage 不下降，IDF1 绝对下降不超过 0.01；
5. 若所有 case 均 fallback，只能判为“安全但无新增有效性”，不能宣称通过。

## 6. 运行和磁盘规则

- 长任务放在 tmux；
- 仅在 `/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v17_holdout` 放临时数据；
- `Harmony4D.zip` 永久保留；
- 每个序列完成且结果通过完整性检查后删除对应 staging；
- 使用空闲 GPU 并记录具体 device；
- 所有失败、跳过和重试写入 ledger。
