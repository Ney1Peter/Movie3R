# V14 单人两帧人体锚点实验（2026-08-05）

## 实验目的

针对 AvatarReX 单人低纹理案例，只取 shot 边界的两帧：第 4 帧 pre 和第 5 帧 post，测试“人体关键点直接作为锚点，同时更新相机和人体”的最小版本。

该实验不修改冻结的 B0 主线，也不读取 GT 做修正。GT 只在实验结束后用于离线评价。

## 输入

- B0+BRTC+C1：`output/v14/report_comparison_viewers/avatarrex_t1836_c22070935_c22053912_pre5_post25/movie3r_b0_brtc_c1`
- 同一 checkpoint 的 raw 人体分支：`output/v14/joint_two_case_payloads/avatarrex_raw_current`
- 边界：pre=`4`，post=`5`

## 方法

1. 从 B0 的 pre/post SMPL-X mesh 通过 SMPL-X joint regressor 得到 3D 人体关键点。
2. 对 post 和 pre 的人体关键点做加权 Kabsch，求人体锚点旋转 `R_human`。
3. 用 B0 和 raw 两条 root ray 的平均值估计相机平移。
4. 相机和人体同步更新：

```text
R_camera_new = R_human · R_camera_B0
t_camera_new = root_BRTC − R_camera_new · mean(q_B0, q_raw)
V_human_new  = root_BRTC + R_human · (V_human_B0 − root_BRTC)
```

5. 生成只包含两帧的标准 demo payload，便于后续直接可视化。

实现：`versions/v14/run_human_anchor_two_frame.py`。

## 三种关键点选择

| 版本 | 关键点 | 两帧人体 RMS | 相机平移误差 | 相机旋转误差 | MPVPE | centered-joint |
|---|---|---:|---:|---:|---:|---:|
| B0+BRTC+C1 | 当前主线 | - | 1.697 m | 66.51° | 0.281 m | 0.488 m |
| stable_feet | pelvis、髋、躯干、脚、头、肩 | 0.033 m | 0.133 m | 4.11° | 0.104 m | 0.090 m |
| body22 | 前 22 个身体关键点 | 0.039 m | **0.055 m** | **1.66°** | 0.102 m | **0.085 m** |
| torso | 骨盆、躯干、头、髋、肩 | **0.009 m** | 0.148 m | 4.76° | **0.082 m** | 0.088 m |
| 当前联合修正版（全 mesh 边界） | 10,475 个 mesh 顶点 | 0.045 m | **0.011 m** | **0.40°** | 0.100 m | 0.083 m |

## 结论

两帧人体锚点确实可以把单人 B0 的 `66.5°` 相机错误大幅纠正，说明“人体作为相机锚点”方向成立。`body22` 关键点版本已经达到 `0.055 m / 1.66°` 的相机误差，人体误差也明显下降。

但在这个案例上，直接用关键点做一次 Kabsch 还没有超过当前全 mesh 边界版本：当前联合版本为 `0.011 m / 0.40°`。不同关键点子集的结果不一致：torso 的人体 MPVPE 更低，但相机旋转更差，说明关键点选择本身需要置信度或联合目标来决定，不能靠固定子集。

## 下一步

这次实验支持继续做“人体锚点 + 联合优化”，但下一版不应只做一次刚体 Kabsch。应加入：

- 关键点/姿态残差的鲁棒权重；
- 相机旋转、平移和人体 root 的联合优化；
- 对真实人体运动的运动先验，避免把人体姿态变化误判成相机变化；
- 用 post 图像中的人体投影/轮廓作为额外约束；
- 在 2 帧确认后扩展到 post shot 内逐帧更新。

