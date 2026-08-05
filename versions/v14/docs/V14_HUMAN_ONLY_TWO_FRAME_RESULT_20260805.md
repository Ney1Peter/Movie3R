# V14 单人两帧人体-only 对齐实验（2026-08-05）

## 实验假设

AvatarReX 单人低纹理场景中，先不动 B0 的相机、背景、深度和置信度，只把 post 人体以 pre 人体为锚点进行对齐，观察人体本身是否可以恢复。

## 设置

- 输入：`movie3r_b0_brtc_c1`
- pre：第 4 帧；post：第 5 帧
- 相机：post 相机 bit-exact 复制 B0
- 背景/深度/置信度：bit-exact 复制 B0
- 人体：使用 SMPL-X 关键点估计 post 到 pre 的旋转
- `root_rotation`：人体围绕 BRTC root 旋转，不改 root 平移
- `full_rigid`：同时应用关键点 Kabsch 的旋转和平移

## 结果

| 方法 | 相机平移误差 | 相机旋转误差 | root 误差 | MPVPE | centered-joint |
|---|---:|---:|---:|---:|---:|
| B0+BRTC | 1.697 m | 66.51° | 0.066 m | 0.281 m | 0.488 m |
| torso + root_rotation | 1.697 m | 66.51° | **0.066 m** | **0.082 m** | 0.088 m |
| body22 + root_rotation | 1.697 m | 66.51° | 0.066 m | 0.102 m | **0.085 m** |
| torso + full_rigid | 1.697 m | 66.51° | 0.646 m | 0.652 m | 0.088 m |

## 结论

1. 只对人体做旋转、相机完全不动，人体局部几何确实可以明显改善；`torso + root_rotation` 的 MPVPE 从 `0.281 m` 降到 `0.082 m`。
2. 直接把人体做完整刚体平移会破坏 root，说明人体平移不能简单使用 pre/post Kabsch translation；BRTC root 应该继续作为位置锚点。
3. 该方案不会改善相机误差，因为相机被明确保持为 B0：相机仍为 `1.697 m / 66.51°`。
4. 因此“人体-only”可以作为人体修正分支，但如果最终 demo 要求相机、背景和人体三者在同一世界坐标中都正确，仍需单独处理相机；不能用人体的 Kabsch 平移直接替换相机。

输出：

```text
output/v14/human_only_two_frame/avatarrex_torso_root_rotation
output/v14/human_only_two_frame/avatarrex_body22_root_rotation
output/v14/human_only_two_frame/avatarrex_torso_full_rigid
```

实现：`versions/v14/run_human_only_two_frame.py`。

