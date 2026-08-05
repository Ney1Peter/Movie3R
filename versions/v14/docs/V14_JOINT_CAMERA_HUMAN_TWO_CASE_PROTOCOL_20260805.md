# V14 联合相机—人体校正：双代表案例实验协议

## 目标

验证一个流式的联合校正方法是否能同时处理两类互补失败：

1. AvatarReX 单人低纹理：原版/当前模型的人体局部几何可用，但跨 shot 相机边界明显错误。
2. MultiHuman `three_t1100_c1_c2` 三人高跨度：背景足够时相机基本正确，但 post 人物检测顺序/native ID 不稳定，需要正确匹配人体。

最终要求不是只改善人体或只改善相机，而是两个案例都满足：

```text
post 相机进入 pre 坐标系 + post 人体进入同一坐标系 + 人物身份/几何连续
```

## 固定案例

### Case A：单人低纹理

```text
sequence: AvatarReX lbn1
pre camera: 22070935
post camera: 22053912
cut timestamp: 1836
layout: 5 pre + 25 post
payload: output/v14/report_comparison_viewers/avatarrex_t1836_c22070935_c22053912_pre5_post25
```

已知现象：原版相机第一帧 post 约 `0.828 m / 40.86°`，当前 B0 约 `1.697 m / 66.51°`；BRTC 可显著修正人体 root，但不改变相机，说明必须加入人体辅助相机校正。

### Case B：三人高跨度

```text
sequence: three_t1100_c1_c2
pre/post timestamp: 1100
camera span: 173.891°
input: cam1/001100.jpg + cam2/001100.jpg
matching cache: output/v20_phase1_gt_id_multihuman_consensus/case_cache/three_t1100_c1_c2_k0.pt
```

已知现象：不加 B0 时 root 匹配 `1/3`、torso/joint 匹配 `0/3`；B0 后几何匹配 `3/3`。该案例用于验证联合方法不能破坏相机正确的多人 baseline。

## 方法候选

对每个 post shot 同时构造三个 Boundary 候选：

```text
B_scene = 原版 Human3R shadow/raw camera 的 SE3
B_v9    = 当前 V9/B0 学习粗边界
B_human = 由 root-centered SMPL joints/vertices 的鲁棒匹配估计的 SE3
```

其中 `B_human` 不读取 GT：

- 单人：直接用人体对应点 Kabsch/Procrustes；
- 多人：枚举/软匹配人物后，用共享 SE3 的鲁棒残差选择对应关系；
- 最终用候选一致性和人体匹配残差产生 confidence。

初版联合策略：

1. 高置信 `B_scene/B_v9` 时保持相机候选，使用现有 BRTC/C1 精修人体。
2. 相机候选与 `B_human` 差异大且人体残差低时，使用人体约束修正 Boundary 的旋转/中心。
3. 中间情况在 SE3 Lie algebra 中做有界融合，并保留 B0 作为安全先验。
4. Boundary 确定后，冻结相机并运行当前已验证的人体 residual/refinement；若相机置信度不足，则进行一次相机—人体交替更新。

## 评测

GT 只用于离线评价，不进入运行时：

- camera translation/rotation error；
- first-post seam root jump；
- root、mean joint、MPVPE、root-centered joint error；
- 多人 anonymous assignment accuracy、all-correct rate；
- post 25 帧 within-shot drift。

每个案例至少比较：

```text
original Human3R
current B0 + BRTC + C1
human-only Boundary
joint Boundary
```

## 成功条件

Case A：相机旋转不再出现大角度错误，人体 root/MPVPE 不劣于当前 B0+BRTC，且 25 帧不发散。

Case B：相机误差不劣于 B0，人物匹配保持 `3/3`，人体 seam 和 25 帧稳定性不劣于当前 baseline。

若任一案例失败，记录失败模式并继续修改候选融合/置信度策略；不能只报告单项指标改善。

## 当前执行状态（2026-08-05）

已完成一次 GT-free 联合处理器和双案例完整评测，详见
`V14_JOINT_CAMERA_HUMAN_RESULT_20260805.md`。

最终配置为 `rotation-source=b0_boundary`、`rotation-alpha=1.0`、
`b0-rotation-gate=25°`、`human-rms-gate=0.15 m`；raw 人体分支由同一
Movie3R checkpoint 的 clean-reset forward 提供。

- AvatarReX 30 帧：联合输出首帧相机 `0.011 m / 0.40°`，MPVPE `0.100 m`；25 帧平均相机 `0.054 m / 0.44°`，MPVPE `0.123 m`。
- three_t1100_c1_c2 30 帧：B0 人体边界旋转 `4.03°`，门控拒绝人体相机更新；25 帧平均相机 `0.054 m / 1.82°`，平均 root `0.071 m`，平均 MPVPE `0.107 m`。
- 三人后 11 帧出现 2 人检测，已实现持久 ID bank；可选 hold-last 使 3 个 ID 全 25 帧保持，平均 root/MPVPE 为 `0.080/0.108 m`。

因此，协议中的“两个案例相机和人体均不被破坏”已满足；大规模泛化、遮挡和更多低纹理序列仍需在冻结阈值后继续验证。
