# Movie3R-v18：Harmony4D 长度与多目标超参开发计划

日期：2026-08-19  
目标：在不牺牲 v17 相机、身份和跨-shot连续性优势的前提下，改善 W/WA、Accel 等短板；所有选择只使用 Harmony4D train split。

## 1. 为什么允许研究序列长度

Movie3R 是在线流式模型，长窗口同时带来两种相反影响：更多历史信息有利于身份和状态传播，但循环状态、相机与人体根节点误差也会随时间积累。Harmony4D 为约 30 FPS，因此对称窗口具有明确部署含义：

| 总帧数 | pre/post | 约时长 | 解释 |
|---:|---:|---:|---|
| 60 | 30/30 | 2 s | 低延迟边界恢复 |
| 90 | 45/45 | 3 s | 实时片段的平衡点 |
| 120 | 60/60 | 4 s | 中等长时一致性 |
| 150 | 75/75 | 5 s | 与现有 Harmony4D/Multi-THuMBS 量级最接近的默认协议 |

长度消融必须满足：同一 capture、同一同步边界、同一四档 camera pair、所有方法使用完全相同长度。150帧保留为正文默认主协议；较短窗口只有在独立train holdout上形成稳定的质量—延迟Pareto优势时，才作为额外低延迟配置报告，不能用来替换表现不佳的test case。

## 2. 数据隔离

本轮开发不使用官方 test split，也不继续读取 train09/10/11 的既有留出结果调参。

开发动作：

```text
train/02_grappling.zip
train/07_ballroom.zip
train/12_mma.zip
```

独立确认动作（候选冻结前不读取指标）：

```text
train/04_sword_part1.zip
train/08_ballroom2.zip
train/13_mma2.zip
```

每个动作按既有结构哈希规则选择首个坐标有效且至少150帧的capture，避免用模型结果挑简单样本。

## 3. 搜索空间及依据

保持联合平移、至少2个跨-shot人体匹配、相机—人体相对几何不变这些核心设计不变，只搜索可解释的安全性与时序参数：

1. `gate_max_boundary_residual_m ∈ {0.15, 0.20, 0.25}`：控制人体锚点的一致性；
2. `gate_max_translation_m ∈ {1.2, 1.4, 1.6}`：限制一次边界更新的可信范围；
3. `boundary_blend ∈ {0.75, 0.90, 1.00}`：避免边界共同平移过冲；
4. `root_alpha ∈ {0.35, 0.50, 0.65, 0.80}`：平衡时序响应与抖动；
5. `root_beta ∈ {0.00, 0.02, 0.05}`：控制因果速度外推。

采用正交有限网格而不是全组合：分别研究 gate、blend 和 root filter，每次只改变一组因素。这样既降低过拟合风险，也能形成清晰消融解释。所有候选均为 prediction-only，不使用GT做推理门控。

## 4. 多目标选择标准

主候选在150帧开发集上选择，必须相对 v17 同时满足：

- ATE-Sim3 不恶化超过5%；
- IDF1 下降不超过0.005；
- MPJPE、MPVPE 各不恶化超过2%；
- W-MPJPE 不恶化超过1%；
- Accel 不恶化超过5%；
- accepted case 不出现超过 parent 50%的灾难性 W 退化；
- fallback 保持 bit-exact parent。

可行候选按 W、WA、Accel、ATE-Sim3、Seam-root 的归一化加权几何平均排序，W/WA权重最高。若没有候选可靠优于 v17，则正式结论为“保留 v17”，不为了制造新版本强行换参数。

## 5. 冻结与最终test

开发集选出候选后，先写入不可变JSON和选择报告，再运行三个独立train holdout动作。只有holdout确认相机/ID优势保留且至少两个核心短板指标改善，才命名为v18并进入Harmony4D全test-capture评测；否则使用v17执行全量test。

最终正文仍报告CS150。短窗口结果放在长度消融或效率图中，强调在线系统的质量—上下文—延迟权衡。

