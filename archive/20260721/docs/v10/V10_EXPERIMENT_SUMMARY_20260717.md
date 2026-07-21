# V10 实验结果简明汇总

日期：2026-07-17

用途：记录当前 V10 非归档实验、关键结果与单句结论，便于后续整理 PPT 和阶段总结。

## 1. V9 Token 消融

Correct token 有一定收益，但 semantic、alignment、momentum、human token 等组合的提升上限有限且会破坏原本稳定的连续帧，因此主线应从逐帧 correction 转向跨 shot 状态管理。

## 2. BEDLAM GT 合成扰动实验

在 BEDLAM GT 合成 segment SE(3) 扰动上，显式 SE3 + residual 达到 `root 0.0237 / cam 0.0542 / boundary 0.0228`、接近 oracle，而 current-only 基本失败，说明对齐必须利用历史状态且干净 GT 域中的显式几何非常有效。

## 3. Human3R 输出域 Integrator 实验

在 Human3R saved-output 合成 local reset 上，`history_direct_residual_integrator` 最新达到 `root 0.4731 / rot 9.42 deg / cam 0.3956 / boundary 0.6459 / velocity 0.0814 / non-boundary 0.0380`，明显优于 current-only 和单层 direct，说明 learned coarse + residual 是当前最有效的 integrator 结构。

## 4. 显式 SE3 跨域实验

显式 SE3 方法在 Human3R 输出域出现约 `68-72 deg` 旋转误差和严重 camera 偏差，说明它对 Human3R 局部坐标噪声、多人顺序和人体几何歧义过于敏感，不适合作为当前主线。

## 5. Bidirectional 与 Reverse Guidance 实验

Bidir teacher、reverse SE3 predictor 和 reverse feature guidance 均未带来稳定收益，部分 camera/non-boundary 指标略有改善但 root/boundary 变差，因此暂不作为推理阶段主线。

## 6. Oracle State vs Gauge 实验

旧 state 继续运行即使施加 boundary oracle SE3 仍有 `0.1278 m / 2.69 deg` 后续漂移，而 reset 后仅为 `0.0042 m / 0.05 deg`，说明 cut 后误差不仅是最终 gauge 错误，还包含 recurrent-state pollution。

## 7. Best-shot SE3/Sim3 离线对齐实验

旧 state 输出经过整段最优 SE3/Sim3 对齐后仍为 `0.0773 m / 1.13 deg`，明显差于 reset 的 `0.0026 m / 0.01 deg`，进一步证明旧 state 改坏了 cut 后的相对轨迹形状而非只引入固定坐标变换。

## 8. 双状态同帧桥接实验

使用 Continue camera 或边界同帧 pointmap 将 Reset shot 接回旧世界后仍约有 `3.2-3.3 m / 134 deg` 误差，说明旧 recurrent state 的最终 camera/pointmap 输出不包含可靠的旧世界重定位信息，旧 state 最多只能作为 latent memory 使用。

## 9. 隐式 Token 跨镜头配准实验

Frozen token 的 implicit-only 结果为 `2.8162 m / 108.76 deg`，物理正确匹配率均值仅 `0.67%` 且 confidence 与正确性几乎无相关性，说明 raw token 会被重复地板、支架和相同 patch 位置严重误导。

## 10. 显式与 Hybrid 跨镜头实验

显式人体 body frame + pointmap refinement 将误差改善到 `1.0237 m / 11.27 deg` 但严格成功率仍为 `0%`，Safe Hybrid 又全部回退为 Explicit-only，说明显式方法能明显修正方向但仍缺少可靠的全局平移约束，而当前 frozen token 没有提供额外增益。

## 11. 显式候选 Oracle Selection 实验

在 AvatarReX、THuman、MVHuman 的 180 个 GT AABB case 上，Joint Oracle 将当前固定 Explicit 从 `1.7047 m / 23.84 deg` 改善到 `1.3891 m / 16.63 deg`，证明人体单帧、历史均值、不同 pointmap refinement 和等待 B2 的候选具有互补性；但最强单一固定候选已经达到 `1.5950 m / 12.93 deg`，Oracle 相对它只改善 `0.2060 m` 且平均旋转变差，成功率几乎不变，因此当前应先提高候选生成质量，而不是直接训练 Selector。

## 12. Boundary Gauge Partial-Oracle 实验

在相同 180 个 case 上，GT Rotation 只能将平移从 `1.5950 m` 改善到 `1.4964 m`，GT Gravity 只解释约 `20.6%` 的旋转收益，而 GT Torso Heading 可将旋转降到 `5.96 deg`；GT Human Root 能将平移降到 `1.1027 m`，但即使 GT Human + Gravity 已把 pelvis 和旋转都对准，camera translation 仍有 `0.9476 m`，远离 Full Boundary Oracle 的 `0.0149 m`。这证明 Human3R 局部人体位置和正确 camera/scene gauge 本身存在约 `0.95 m` 不一致，人体不能作为唯一硬平移锚点，下一步应优先开发场景重定位或世界坐标记忆。

## 当前总体结论

当前证据支持在 camera cut 时 reset/fork Human3R local state，再通过严格流式的 segment re-anchor 接回 global state。核心缺口不是简单的候选选择、旋转或地面法向，而是 cut 后 fresh local reconstruction 如何重新获得旧世界的场景级 translation/gauge。下一步应优先开发 compact scene relocalization / world-coordinate memory，人体 motion/root 作为平移软约束，torso heading 作为旋转约束，gravity 只作为稳定倾斜的辅助；在候选上界提高前暂不训练 Selector。
