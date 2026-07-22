# Movie3R 文档入口

当前 camera-cut 对齐代码采用 `V10.x-V14.x` 的紧凑编号。共享实现使用
`boundary_*.py` 语义化名称，不再占用实验版本号。旧的 V15-V55 编号只在
历史报告和缓存字段中保留。

## 当前入口

| 文档 | 内容 |
|---|---|
| [Boundary Alignment 主线](ACTIVE_BOUNDARY_ALIGNMENT.md) | 保留版本、方法差异、运行命令、输出位置 |
| [版本编号规则](BOUNDARY_VERSIONING.md) | 新旧编号映射和后续编号规则 |
| [V11 几何完整性审计](v11/V11_RETAINED_GEOMETRY_INTEGRITY.md) | Torso、条件宽基线、接触修正和统一尺度结论 |
| [V14.3 人-相机联合对齐](V14_3_PROJECTION_CONSISTENT_REANCHORING.md) | Coupled root、DA3 metric cue、scene 安全边界和连续性可视化 |
| [训练代码入口](train_code.md) | 训练代码说明入口 |

## 历史归档

- 2026-07-21 近期实验：[../../archive/20260721/README.md](../../archive/20260721/README.md)
- V8 历史：[archive_v8/README.md](archive_v8/README.md)
- V7 历史：[archive_v7/README.md](archive_v7/README.md)
- V2-V6 历史：[archive_v2_v6/README.md](archive_v2_v6/README.md)

归档内容没有删除；如需恢复某个实验，应先将对应脚本和报告移回原位置，
再按其原始文档运行。
