# Movie3R 文档入口

当前 camera-cut 对齐代码采用 `V10.x-V14.x` 的紧凑编号。共享实现使用
`boundary_*.py` 语义化名称，不再占用实验版本号。旧的 V15-V55 编号只在
历史报告和缓存字段中保留。

## 当前入口

| 文档 | 内容 |
|---|---|
| [正式版本目录](../../versions/README.md) | V9 训练版、V14.7 单人版和 V20 多人版的独立入口、tag 与 checkpoint hash |
| [当前默认单人版](../../LATEST_MODEL.md) | V14.7 单人默认路径、冻结 tag、结果、适用范围和复现入口 |
| [V14.7 当前独立方法](V14_7_SHOT_AWARE_UNIFORM_SIMILARITY_REANCHORING.md) | 当前方法的正式名称、统一流程、版本关系和冻结边界 |
| [完整模型与消融](CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md) | 输入输出、完整架构、所有模块作用、最新消融和限制 |
| [V14.6 组件必要性审计](V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md) | VGGT off 下 Fixed、V16、DA3、Keypoint 和 V11.4 公平消融 |
| [Boundary Alignment 主线](ACTIVE_BOUNDARY_ALIGNMENT.md) | 保留版本、方法差异、运行命令、输出位置 |
| [版本编号规则](BOUNDARY_VERSIONING.md) | 新旧编号映射和后续编号规则 |
| [V11 几何完整性审计](v11/V11_RETAINED_GEOMETRY_INTEGRITY.md) | Torso、条件宽基线、接触修正和统一尺度结论 |
| [V14.3 人-相机联合对齐](V14_3_PROJECTION_CONSISTENT_REANCHORING.md) | Coupled root、DA3 metric cue、scene 安全边界和连续性可视化 |
| [V11/V14 方法详细比较](V11_V14_ALIGNMENT_METHOD_COMPARISON.md) | 旧 V46/V47、V11.4 与 V14.3 的模型、流程、结果、创新性和后续路线 |
| [V20 多人当前有效报告](V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md) | strict GT-ID 三人 shared-Boundary 结果、限制与复现 |
| [V20 数据集说明](V20_EGOBODY_MULTIHUMAN_DATASET_GUIDE.md) | EgoBody 与 MultiHuman 的数据、标定、人体标注和适用任务 |
| [训练代码入口](train_code.md) | 训练代码说明入口 |

## 历史归档

- 2026-07-21 近期实验：[../../archive/20260721/README.md](../../archive/20260721/README.md)
- V8 历史：[archive_v8/README.md](archive_v8/README.md)
- V7 历史：[archive_v7/README.md](archive_v7/README.md)
- V2-V6 历史：[archive_v2_v6/README.md](archive_v2_v6/README.md)

归档内容没有删除；如需恢复某个实验，应先将对应脚本和报告移回原位置，
再按其原始文档运行。
