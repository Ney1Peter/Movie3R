# V14 流式 Shot 内人体稳定：任务、验收与执行协议

日期：2026-08-04  
状态：执行中；本文件是本阶段唯一的实验约束与完成判据。

## 1. 要解决的实际问题

已冻结的 `B0 + BRTC-LC` 已能在 camera cut 时把新 shot 接回旧 world，并在第一帧用显式多人射线几何修正逐人 root。它不改 camera，且该边界修正对同一 post shot 是常量。

但在连续 post 帧中，Human3R 原生预测仍会让真实静止的人出现不应有的 root/mesh 漂移。严格原版 Human3R 对照已证实，这不是 BRTC-LC、ID switch 或展示代码引入的伪影。若直接对所有人做平滑，会把真实行走者的运动错误抹掉。

本阶段问题因此是：

> 在每个 shot 内，能否只利用当前及过去 RGB/Human3R 输出，以因果、逐人、可拒绝的方式，降低“应当静止人”的相机补偿后 root 漂移，同时保留真实运动者的位移，且完全不修改 B0 camera？

它不是再做一个跨 shot Boundary，也不是用未来帧离线 smooth。

## 2. 不可违反的部署约束

1. **严格流式**：时刻 `t` 只能读到该人已到达的轨迹（`<=t`）；禁止 future frame、双向滤波、离线轨迹拟合。
2. **无需新增预训练模型**：仅使用冻结 P0/Human3R 的输出及其 RGB 输入中已有、可解释的几何/运动信号；先做无训练方案。
3. **camera safety**：输入、输出 camera 必须 bit-exact 相同；scene pointmap 不被本阶段改写。
4. **逐人而非全局**：一个人的判定或修正不能强制覆盖其他人；不可靠、遮挡、短轨迹或 ID 不稳定者必须 exact fallback 到当前 B0+BRTC 输出。
5. **真实运动保护**：不允许靠把所有轨迹锁死来获得更小 drift。
6. **边界兼容**：第一 post frame 的 BRTC-LC 固定 translation 继续保留；shot 内模块只能产生该 translation 之后的 causal residual。
7. **GT 隔离**：GT 只用于完成预测后的评测和静止/运动标签审计，绝不进入 runtime gate、阈值或关联。

## 3. 统一的运行时对象

对每个匿名 person track 保存一个固定大小的状态：

```text
S_i(t) = {
  recent camera-compensated root / pelvis,
  recent root velocity and robust residual,
  recent body-frame pose/shape/mesh consistency features,
  confidence and track-age,
  previous committed root residual
}
```

每帧处理：

```text
current RGB -> frozen P0/Human3R -> B0 world camera + raw person
          -> fixed BRTC-LC boundary translation
          -> anonymous within-shot track update
          -> camera-compensated per-person motion evidence
          -> static / moving / unknown causal gate
          -> bounded root residual filter (only static, only if confident)
          -> root/joints/vertices receive same translation
          -> output current frame and commit S_i(t)
```

所谓 *camera-compensated*，是先在 predicted camera-local coordinates（或等价的相机运动消去坐标）中分析 root。这样不会把相机平移抖动误判成人在移动。所有修正仍在 world 中对 root、joints、vertices施加同一平移。

## 4. 先验证的候选阶梯

### C0：诊断基线（不改变输出）

- 分开报告 world drift、predicted-camera-compensated root drift、camera drift；
- 用 GT 做后验的 static / moving label 审计，检查 runtime 证据能否区分两类人；
- 与 strict-original Human3R 和 current P0 进行同一 RGB 长序列比较。

### C1：无训练的因果静止门控 + 保守 root filter（首要候选）

- runtime 特征只用 track 内的 camera-local root velocity/加速度、SMPL pose/shape change、mesh/root consistency、2D bbox 或 joints 尺寸变化、track age/confidence；
- 高置信 static：对 root residual 使用 capped EMA / robust median / alpha-beta filter；
- moving 或 unknown：exact identity，即保留 B0+BRTC；
- gate 采用 hysteresis 和最短历史，避免每帧抖动切换。

### C2：多人相对静止一致性（仅作为 C1 的可选辅助）

当多个高置信人同时静止时，利用相对 root 向量稳定性压制共同的预测噪声；不能假设所有人同动，也不能修改 camera。

### C3：轻量可训练 residual / gate（仅当 C1/C2 失败）

只在 C1/C2 的可解释信号已显示存在稳定规律、但固定阈值不能泛化时开展。网络只预测 bounded root residual 或 static probability；训练、验证和最终测试按 sequence/camera group 严格分离，仍满足 C0--C2 的 fallback 和 camera safety。

## 5. 数据与实验协议

### 5.1 首批数据

- 小跨度三人真实连续 post-25/30 帧：专门覆盖“两个静止、一人行走”的可视化问题；
- 可获取 GT 的 MultiHuman `three/dance/box` 连续轨迹：用于逐人 root/joint/vertex 误差、运动保留和泛化；
- 至少一个未用于选参的 sequence/camera group 做冻结确认。

开发和确认必须以 sequence/camera/time group 划分，禁止同一连续轨迹的帧落到不同阶段。若公开数据不提供连续 GT，先将该限制明确写入报告，不用自造的短缓存替代正式结论。

### 5.2 固定输出与核心指标

每种候选均输出每帧：camera、world root/joints/vertices、track id、gate、残差、runtime feature 摘要。至少报告：

| 目标 | 指标 |
|---|---|
| 静止人稳定 | camera-compensated cumulative drift、mean frame displacement、root acceleration / jerk、GT stationary root error |
| 运动保护 | moving track 的 displacement retention（预测累计位移 / B0+BRTC 累计位移）、moving root/joint/vertex error 与 acceleration |
| 空间精度 | root、joint、vertex、多人 layout vector，与 B0 和 BRTC-LC 并列 |
| 安全性 | camera max absolute change = 0、unknown/moving fallback rate、错误静止门控率、ID continuity |
| 线上性 | 每帧只用的最大历史长度、每人状态大小、CPU/GPU runtime 增量 |

`GT static/moving` 仅用于评估；runtime gate 的 precision/recall/F1 后验报告不能反向选用 GT 阈值。

## 6. 目标与完成判据

候选在开发组选择后冻结，在独立确认组上必须同时满足：

1. camera 最大改变量严格为 `0`；
2. 静止人 camera-compensated cumulative drift 相对 B0+BRTC 至少降低 **30%**，且至少在两个不同轨迹/场景成立；
3. 静止人 GT root 误差不恶化超过 **3%**；
4. 行走人位移保留率至少 **90%**，且 moving GT root/joint/vertex 平均误差不恶化超过 **5%**；
5. 全部人总体 root/joint/vertex 平均误差相对 B0+BRTC 不恶化超过 **3%**，多人 layout vector 不恶化超过 **5%**；
6. 任何人超过 `5 cm` 的额外 root harm 比率增加不超过 **2 个百分点**；
7. 连续至少 25 帧的真实运行，以及单元测试均通过；展示中 gate、残差和人运动行为可审计。

若没有候选通过，记录失败原因后继续下一候选；不得因“视觉更稳”而放宽 1--6。若连续 C1/C2 均失败，再进入 C3。

## 7. 交付物

- 可复现的长序列评测 cache/manifest、诊断报告和原版对照；
- 一个冻结的部署 policy JSON 和对应实现；
- 开发扫描、冻结时间戳、独立确认结果；
- 消融：无稳定、无 motion gate、无 hysteresis、不同 filter、（可选）多人辅助；
- 25+ 帧 demo.py 风格 payload（但不常驻启动 viewer）；
- 一份如实记录成功和失败候选的最终 handoff，作为进入大规模指标前的冻结证据。

## 8. 执行顺序与停机条件

1. 审计数据是否能支撑真实连续 GT 评测，并实现 C0；
2. 只在开发组扫描 C1；冻结规则与 policy；
3. 在未参与选参的组确认；
4. 若失败，记录并尝试 C2；仍失败再做 C3；
5. 一旦某候选满足第 6 节全部门槛，冻结并转入统一 end-to-end 与大规模 benchmark；
6. 最多冻结三个合格方案；只要尚无合格方案，继续实验而非停止在主观观感。

---

## 9. 执行结果（2026-08-04）：C1 完成，转入大规模指标

执行了 C0 长序列诊断、预先固定的五个 C1 无训练 policy，以及 25-frame CPU-only
runtime-first cache。最终选择并冻结 `C1-EMA25`：

```text
camera-local static/moving hysteresis
  + EMA(alpha=0.25) only on high-confidence static tracks
  + identical root/joints/vertices translation
  + exact B0+BRTC fallback elsewhere
```

它在 `three t1100` 与时间不重叠的 `three t1200` 上分别降低静止人 camera-local
path `38.4% / 41.5%`；在跨场景 `box t590` 上，慢速 moving person 的 net/path
retention 为 `100% / 93.2%`。三条 stream 的 camera max change 都为 `0`，worst
all-root 增幅为 `+0.3%`、worst layout-vector 增幅为 `+4.6%`，均通过第 6 节门槛。

完整证据、失败的 `three t900` track-drop 案例及可部署 policy 见
[`V14_WITHIN_SHOT_C1_STATIC_GATE_V1_20260804.md`](V14_WITHIN_SHOT_C1_STATIC_GATE_V1_20260804.md)。
由于 `three` 两条轨迹来自同一 capture sequence 且 box 的 slow-motion 仍有 gate
混淆，这一冻结是内部主线，而不是论文最终泛化 claim。下一阶段不继续调 alpha，直接按该
frozen policy 做统一 end-to-end 大规模 spatial/temporal/system 指标。
