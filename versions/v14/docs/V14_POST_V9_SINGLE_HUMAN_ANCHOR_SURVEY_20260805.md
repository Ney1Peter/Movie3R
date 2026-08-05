# V9 之后单人“人体锚点重叠”方法检索与两帧复现

## 1. 检索范围

已核对 `versions/v9`、`versions/v12`、`versions/v13` 和 `versions/v14` 中与单人人体锚点、躯干方向、关键部位重叠和 camera-human consistency 相关的文档与实现。

## 2. 各版本真正做过什么

### V9：Human-Anchor Factorized Correction

V9 的核心是把人体锚点作为模型内部的 relation state：先修正 human latent，再让 camera correction 读取 corrected human anchor。V9 还提出 pelvis、hip、feet、torso 等 body-part cue 和 UniCon3R 风格的显式几何 token。

但 V9 主要是训练时的 latent residual 设计，不是推理时直接拿人体关键点做刚体重叠；它不会给出一个明确的 TORSO4/Kabsch 几何规则。

### V12：V16 torso-motion + V11.4 similarity

V12 的正式单人主线是：Fixed Explicit 粗对齐后，用 pre-cut torso 历史预测 post-cut torso 方向，再做一次有界 rotation residual；随后用统一 shot-level similarity 处理 camera、pointmap 和人体。

V16 的关键是“人体运动时间连续”，不是直接把 post 人体强行拉回 pre。它使用 pelvis、髋、肩、头等关节构造 torso frame，估计 pre-cut angular motion，再预测第一张 post 的 torso frame。

### V13：person-local TORSO4 Kabsch

V13/V14 中找到最接近当前问题的显式方法：

```text
冻结 B0 相机
-> BRTC 修正人体 root/布局
-> 取左右髋 + 左右肩 TORSO4
-> 各自减 root
-> Kabsch 求人体局部 SO(3)
-> 绕已修正 root 旋转 joints/vertices
-> camera 和 root 保持不变
```

冻结参数是 `rotation_fraction=0.5`、`max_angle=25°`，并要求 torso residual 确实下降。该参数是在多人 BRTC 已经把 root 和布局修好的条件下冻结的，目的是做小幅人体朝向 refinement，不是解决 70° 级别的低纹理相机错误。

### V14：camera-human consistency 审计

V14 的审计明确指出：人体可以作为人体朝向和 root 的锚点，但不能无条件把人体平移直接变成相机平移；否则可能人体看起来更贴，camera/scene 却被拉到错误世界位置。

因此 V14 保留了 person-local orientation candidate，拒绝默认使用未经验证的人体 translation/camera refinement。

## 3. AvatarReX 两帧复现

输入仍为当前 `B0+BRTC+C1`，pre=`4`、post=`5`，相机保持不动，只修改 post 人体。

| 方法 | 旋转 | MPVPE | centered-joint | 相机 |
|---|---:|---:|---:|---|
| B0+BRTC | - | 0.281 m | 0.488 m | 1.697 m / 66.51° |
| V14 TORSO4 frozen：25° | 25° | 0.202 m | 0.321 m | 完全不变 |
| TORSO4 full | 69.29° | 0.084 m | 0.086 m | 完全不变 |
| V16 torso-motion 20° | 20° | 0.216 m | 0.354 m | 完全不变 |
| V16 torso-motion full | 67.90° | **0.080 m** | 0.087 m | 完全不变 |

解释：

- V14 的 TORSO4 方向是正确线索，但冻结的 `25°` 上限只够做小 residual；AvatarReX 这个低纹理案例需要约 `68–70°` 的大旋转，所以 frozen 版本会明显欠修。
- 使用完整 TORSO4 或 V16 torso-motion rotation 后，人体 mesh 才真正接近 GT。
- V16 full 在这个两帧案例的 MPVPE 最低，但它是针对当前极端案例的诊断结果，尚未证明可以无界部署。
- 相机始终保留 B0，因此 camera 误差仍是 `1.697 m / 66.51°`；这个实验只证明人体朝向可以单独恢复。

## 4. 当前最合理的解释

之前“整个人没有重叠”的原因不是简单的 root 平移，而是：

1. B0 post 人体整体朝向发生了约 67° 的错误；
2. 直接对所有 mesh 做普通 Kabsch 会混入人体局部姿态变化和 root 平移；
3. V14 的 TORSO4 方法只做小幅朝向 residual，面对 AvatarReX 这种极端低纹理案例会欠修；
4. 人体完整平移会破坏 BRTC root，所以不能直接使用 Kabsch translation。

当前证据支持的单人候选应是：

```text
人体 root 固定为 BRTC anchor
-> 用 torso4 / torso frame 确定人体方向
-> 使用历史 torso motion 判断真实运动
-> 仅在方向残差、关键部位重叠和运动先验一致时施加大旋转
-> 人体平移只允许 root-preserving residual
-> camera 暂不被人体直接改写
```

## 5. 输出和脚本

```text
V14 TORSO4 full:
output/v14/human_only_two_frame/avatarrex_torso4_full

V16 torso-motion full:
output/v14/human_only_two_frame/avatarrex_v16_torso_motion_full
```

实验脚本：

```text
versions/v14/run_human_only_two_frame.py
versions/v14/run_v16_torso_motion_two_frame.py
```

可视化：

```text
8137: TORSO4 full
8138: V16 torso-motion full
```

