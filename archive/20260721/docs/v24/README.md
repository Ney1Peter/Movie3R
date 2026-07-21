# V24 当前入口

V24 将 V22 的显式米制 Boundary Bridge 与 V15 已缓存的冻结 VGGT 1+1 rotation 组合，但 VGGT 只在 V16 torso residual 至少为 `10 deg` 时运行，并且只能通过固定、source-independent 的物理安全规则修改 rotation。

当前选中方法：

```text
Hard Reset
-> V22 DA3 human/background metric scale
-> V16 torso-motion + safe gravity rotation
-> conditional frozen VGGT 1+1 rotation
-> explicit human-root translation re-solving
-> one fixed shot-level SE(3) and scale state
```

180-cut 结果：

| Method | Camera T mean/P95 | Rotation mean/P95 | Scene mean/P95 | Catastrophic |
|---|---:|---:|---:|---:|
| V22 | 0.490 / 1.218 m | 15.67 / 52.21 deg | 0.288 / 0.683 m | 7.2% |
| V24 selected | 0.434 / 1.040 m | 12.09 / 37.75 deg | 0.288 / 0.683 m | 2.2% |

V24 实际修正 `34/180` 个 cut，救回 9 个灾难样本，新增灾难样本为 0。Fixed rotation `30-60 deg` 组的灾难率从 `8.8%` 降到 0，`>=60 deg` 组从 `45%` 降到 `15%`。低纹理组从 `14.3%` 降到 `3.6%`，高纹理组保持不变。

完整报告：[V24_SAFE_CONDITIONAL_WIDE_ROTATION_BRIDGE_20260721.md](V24_SAFE_CONDITIONAL_WIDE_ROTATION_BRIDGE_20260721.md)
