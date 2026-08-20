# EgoHumans v19 组合候选补充预注册

日期：2026-08-20  
数据权限：只读取 basketball、badminton 两个 development capture；未读取其余五类 development、任何 holdout 或 test 指标。

## 1. 触发依据

前两类共同显示，人体锚定的共享 translation 边界校正能够同时降低 W/WA、Accel、ATE 与 seam，且按构造不改变 camera-local pelvis-relative mesh/body。原有限网格中的 causal root filter 单独降低 Accel，但尚未与最优边界项组合。

因此只检验一个明确的正交假设：

> shared boundary translation 负责跨 shot gauge；causal alpha-beta root filter 负责 shot 内 root 抖动。

## 2. 冻结网格

配置文件：`versions/v19/egohumans/development_combined_candidates.json`

- translation blend：主值 1.0；保留一个 0.75 安全折中；
- camera alpha：1.0 或固定相机协议下的 0.0；
- root filter：既有网格中的 `(alpha,beta)=(0.9,0.02)`，以及一个更强的 `(0.8,0.05)`；
- 共四个新候选，不再扩展；
- 无候选通过 development 安全门槛时全部否决；
- 最多两个新候选与冻结 v17 fallback 进入独立 holdout。

## 3. 不变规则

- runtime 只读 RGB、模型预测、当前及历史状态；
- GT/相机标定/identity 只进入 evaluator；
- 100 帧、50+50、四跨度和原 split 不变；
- MPJPE/MPVPE 恶化超过 2%、coverage 下降超过 1pp、IDF1 下降超过 0.01、任一 case W 恶化超过 20%即否决；
- holdout/test 不再新增参数或候选。
