# OnlineHMR 扩展实验执行计划（2026-09-05）

## 1. 目标与论文定位

在已经完成的 EgoBody、EgoHumans 和 Harmony4D 单切镜正式实验之外，以相同的 RGB-only 外部方法协议补齐 OnlineHMR 在已有 BRIDGE3R 实验中的有效交集。新增结果用于 ICLR 正文或补充材料中的强在线参考，并回答三类问题：单人场景是否成立、切镜次数增加时是否稳定、低纹理和大视角条件下是否仍能维持跨镜头坐标一致性。

OnlineHMR 在论文中统一称为 **same-input semi-online external reference**。它使用当前帧及内部缓存，但官方实现与 BRIDGE3R 的严格因果执行、输出拓扑和失败支持并不完全等价，因此不宣称为严格同构的因果基线。

## 2. 固定实验范围

| 数据/协议 | 序列数 | 帧数 | 切镜数 | 主要用途 |
|---|---:|---:|---:|---|
| AIST++ CS150 | 100 | 150 | 1 | 单人、官方 test source、大视角切镜 |
| AIST++ MC150-3 | 100 | 150 | 2 | 三镜头重复切镜 |
| AIST++ MC150-4 | 100 | 150 | 3 | 四镜头重复切镜 |
| MVHuman MVH150 | 50 | 150 | 1 | 低纹理与五档视角跨度 |
| Harmony4D multi-cut | 4 | 150 | 2 | 多人三镜头辅助证据 |

本轮共新增 354 条；连同三个主要多人数据集的 307 条正式结果，OnlineHMR 总覆盖 661 条预先固定序列。

H36M 目前只有历史可视化而没有与本文一致的冻结 test 协议，因此不纳入本轮。内部 reset、token、粗/精对齐、检测器和 $\lambda$ 等消融属于 BRIDGE3R 机制，不存在 OnlineHMR 的对应开关，也不补跑。

## 3. 输入与公平性协议

- 每种方法接收完全相同、顺序一致的 150 帧 RGB。
- OnlineHMR 不读取 GT 相机、身份、人数、切镜位置、mask、深度或评估标签。
- 不允许按镜头分别运行后再人工或 GT 辅助对齐。
- AIST++ 与 MVHuman 使用既有正式视频按固定 ffmpeg 解码配置产生 JPEG；Harmony4D 直接复制冻结 manifest 指定的原始 JPEG。
- 使用原有数据集 evaluator、关节映射、first-shot Sim(3) 锚定和匹配规则；不得根据 OnlineHMR 结果调阈值或筛样本。
- 推理失败、空输出、无匹配和无有效几何均保留在正式分母。Coverage 和 completion 使用全量分母；几何均值报告有限支持数，不做缺失插补。

## 4. 指标

### AIST++ CS150 与 MVHuman

PA-MPJPE、first-shot Anchor-MPJPE、root error、orientation proxy、cut seam root/orientation、相对相机旋转/平移、Coverage 和 completion。MVHuman 额外按 small、medium、large、very-large 和 extreme 五档视角汇总。

### AIST++ multi-cut

报告 PA-MPJPE、first-shot Anchor-MPJPE、所有边界的平均 seam 和相机误差、post-first-cut 相机误差、Coverage、completion，并分别汇总第 1、2、3 次切镜的边界指标。

### Harmony4D multi-cut

报告 W-MPJPE、WA-MPJPE、ATE-Sim3、IDF1、Coverage 和两个边界的 seam/camera error。四条均为预注册辅助样本，不能替代主数据集结论。

## 5. 执行顺序与资源

1. 冻结扩展 runtime/evaluator manifest，并验证输入数量和路径。
2. 完成 Harmony4D 四条全量实验，作为多人 multi-cut 与完整表面转换的技术闸门。
3. 完成 AIST++ CS150 100 条。
4. 完成 MVHuman MVH150 50 条。
5. 完成 AIST++ MC150-3 和 MC150-4，各 100 条。
6. 对每个协议执行固定分母聚合、分视角/边界分析和已有方法配对比较。
7. 新建论文版本，在补充材料中加入完整 OnlineHMR 扩展表；正文只在空间允许且结论清晰时加入最关键的一行或一句。

最多使用 5 张空闲 GPU，每张卡一个官方 OnlineHMR worker。因 `/data` 剩余空间有限，采用逐条流式闭环：暂存输入、推理、转换、评估、删除可复现的 native/点云和暂存 JPEG，只保留 runtime、日志、紧凑 prediction、adapter metadata 和 evaluation。

## 6. 完成标准

- 354 条均存在推理 runtime 和评估记录，失败也具有明确的固定分母条目。
- 每个协议均生成 case CSV、aggregate JSON、支持数、失败统计、Coverage/completion 和 95% bootstrap CI。
- multi-cut 结果包含逐次边界分析；MVHuman 包含五档视角分析。
- 新版论文用词明确区分严格因果内部基线、semi-online OnlineHMR 与 offline 参考。
- LaTeX 可本地无错误编译，生成单一 PDF 和可直接导入 Overleaf 的 ZIP。
- 代码、计划、聚合证据和论文版本分别提交，保留可恢复的实验状态。
