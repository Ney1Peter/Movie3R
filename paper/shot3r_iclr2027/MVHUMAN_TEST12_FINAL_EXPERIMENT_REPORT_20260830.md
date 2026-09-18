# MVHuman Test1/Test2 最终实验报告

日期：2026-08-30

## 1. 数据与准入结论

`Mvhuman-test1.zip` 与 `Mvhuman-test2.zip` 完整可用，共包含 10 个 capture/motion、4,265 个同步时间点和 131,600 张多相机 RGB 图像。数据具有相机内外参、人体 mask、2D 标注、逐帧 SMPL-X、3D 关键点和 mesh，可支持 Camera--Human 定量评价。

新 capture 为 `100021--100025`、`200021--200025`；correction-module 训练 capture 为 `100001--100005`、`200001--200005`。二者在 capture、event 和 frame/member 层面无重叠，但共享 MVHuman 数据域与两套相机 rig。因此论文表述限定为：

> ten held-out MVHuman captures/motions that are disjoint from the correction-module training manifests while retaining the original camera rigs.

不能声称 unseen subjects、unseen rigs、unseen dataset 或 official test split。数据没有场景深度/mesh GT，因此不用于 Scene--Human 定量评价。

## 2. 冻结协议

- 50 cases：10 captures × 5 个 camera-rotation strata；
- 每例 150 帧：75 个切换前帧 + 75 个切换后帧；
- 所有方法读取完全相同的 RGB streams；
- 五档由相机旋转的 SO(3) geodesic 定义，不把最宽档误写为 optical-axis change 大于 150 度；
- 背景人体 mask 外的平均 normalized gradient 为 0.00263，平均 Laplacian variance 为 0.000614；该数据称为 controlled weak-texture studio setting，而不是官方 low-texture benchmark；
- GT boundary、相机标定和人体 GT 只供 evaluator 使用。

冻结协议位于：

`Movie3R/output/bridge3r_mvhuman_v1/protocol_freeze/`

## 3. 内部方法结果

下表为 10-capture macro，所有方法使用相同 50-case 分母。

| 方法 | PA-MPJPE (mm) ↓ | Anchor-MPJPE (mm) ↓ | Seam-root (mm) ↓ | Camera rotation (deg) ↓ | Camera translation (m) ↓ | Coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 27.54 | 217.53 | 442.26 | 94.97 | 1.733 | 99.8% |
| BRIDGE3R | 27.60 | 261.00 | 434.47 | 87.71 | 1.897 | 99.8% |

结果不能解释为 BRIDGE3R 在该协议上整体领先：它改善了 seam-root 和相对 camera rotation，但 Anchor-MPJPE 与 camera translation 变差，局部 PA-MPJPE 基本不变。

## 4. 自动 detector 审计

Detector 在所有 50 个标注 cut 帧均产生过 positive response，但部署状态机使用每个未处理 boundary episode 的首个 positive：

- 10/50 例首次触发恰好位于真实 cut；
- 40/50 例在 frame 2--70 提前首次触发；
- 提前触发后，单 transition state machine 不会在 frame 75 再执行一次 bridge。

因此 framewise recall=1.0 不能代表正确的边界触发；first-trigger accuracy 只有 20%。正式结果不以 GT boundary 替换 detector，不排除提前触发案例，也不把该实验作为自动 hard-cut handling 的正面证据。

## 5. 公开方法

### GVHMR

预注册 12-case pilot 完成 10 例；2 例在官方 SimpleVO two-view solver 产生原生重建前失败。固定 completion 为 83.3%，未通过扩展门槛，因此不运行 50-case 全量，也不报告 10 例条件几何为排行榜。

正式 gate：

`Movie3R/output/bridge3r_mvhuman_v1/gvhmr_official/gvhmr_pilot_gate_failed_sealed.json`

### PromptHMR

官方 offline full-video PromptHMR 完成 50/50，并通过原生结果、SPEC camera、转换、指标、异常记录、代码/权重哈希和固定分母审计。

| 方法 | PA-MPJPE (mm) ↓ | Anchor-MPJPE (mm) ↓ | Seam-root (mm) ↓ | Seam-orient. (deg) ↓ | Camera rotation (deg) ↓ | Coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|
| PromptHMR (official, offline) | 28.46 | 137.60 | 20.34 | 53.64 | 103.87 | 100.0% |

PromptHMR 的 Anchor-MPJPE 和 seam-root 明显低于两个 recurrent routes，但相对 camera rotation 更差。它具有 full-video future access；结合 BRIDGE3R detector 的提前触发，三者不能被包装成统一因果排行榜。

正式审计：

`Movie3R/output/bridge3r_mvhuman_v1/prompthmr_formal50/audit/formal_ledger.json`

## 6. 投稿定位

该实验进入同一论文 PDF 的 Supplement，标题为 “Held-Out MVHuman Weak-Texture Stress Audit”。它承担三项作用：

1. 证明新数据的格式、低纹理背景和不同相机旋转跨度可以被严格复现；
2. 暴露 detector 的 first-trigger failure，防止用 framewise recall 掩盖部署失败；
3. 给出 PromptHMR 完整分母和 GVHMR availability gate，说明公开方法的时序和输出边界。

该实验不进入正文正向主表，不支撑 “BRIDGE3R 在 MVHuman 上最好” 或 “自动 detector 在弱纹理 cut 上可靠” 的论断。正文主结论继续由 EgoBody、EgoHumans、Harmony4D 的相同输入配对实验与大视角分析支撑。
