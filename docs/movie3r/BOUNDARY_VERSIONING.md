# Boundary Alignment Versioning

## Active Scheme

后续只增加主任务版本，不再为每个失败消融创建新的整数版本。

- `V10.x`：Fixed Explicit 与基础显式候选。
- `V11.x`：可保留的 rotation、contact、scale 方法及消融。
- `V12.x`：缓存、可视化和工程验证。
- `V13.x`：真实视频和流式系统验证。

同一任务内的小实验使用 `.1/.2/.3`。失败实验留在该主版本的子编号或归档
目录，不再继续增加 V30、V40、V50。

共享工具使用 `boundary_*.py`，不分配版本号。

## Legacy Mapping

| 旧编号 | 当前编号/名称 |
|---|---|
| V10 Fixed Explicit candidate | V10.1 Fixed Explicit |
| V47 Torso + Conditional VGGT | V11.1 Conditional Wide Rotation |
| V46 Global + Local Contact | V11.2 Contact-Preserving Alignment |
| V48 component ablation | V11.3 Component Ablation |
| V53 uniform similarity | V11.4 Uniform Similarity |
| V52 long-sequence cache/viewer | V12.1/V12.2 |
| V55 real-video explicit alignment | V13.1 |

V15-V45 的其他失败、Oracle 或诊断实验保持原文件名进入归档，仅用于历史
追溯，不再作为活跃版本。
