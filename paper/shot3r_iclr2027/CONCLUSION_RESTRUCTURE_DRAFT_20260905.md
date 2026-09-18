# Shot3R Discussion 与 Conclusion 重构草案（2026-09-05）

> 当前状态：阶段性确认，暂时保留。本稿尚未写入 v031/v032 的 LaTeX 论文源文件，正式英文改写仍需后续确认。

## 1. 章节结构调整

删除主文中独立的 `Discussion`，使实验之后直接进入 `Conclusion`：

```text
4. Experiments
   4.1 Experimental Setup
   4.2 Multi-Shot Multi-Person Reconstruction
   4.3 Analysis and Ablation Studies

5. Conclusion
```

不删除 Discussion 中有证据支持的内容，而是按照各自功能重新安置：

1. WA-MPJPE、相机 ATE、IDF1 与局部人体误差之间的关系，放入 4.2 Overall Results；
2. 大幅视角变化下的额外收益，放入 4.2 Large Viewpoint Changes；
3. 同输入、相同评测器及外部方法的协议差别，放入 4.1 Baselines；
4. 可训练参数比例，放入 4.1 Implementation Details；
5. 镜头检测的首次触发问题和受控重建开销，放入 4.3 Shot-Transition Detection and Efficiency；
6. Harmony4D 的 W-MPJPE 回退在 4.2 如实保留；AIST++、重复切换和弱纹理压力测试的完整失败结果放入补充材料；
7. 检测、人物关联、重复切换和具体指标回退等可诊断问题只在对应实验或补充材料中如实呈现，不在 Conclusion 中逐项罗列；Conclusion 只将复杂剪辑、人物反复进出画面和快速画面变化概括为仍待进一步研究的复杂条件。

以上内容已经基本包含在 `EXPERIMENTS_RESTRUCTURE_DRAFT_20260905.md` 的 4.1--4.3 候选稿中，不再通过独立 Discussion 重复实验发现。

## 2. Conclusion 当前保留的中文版本

### 第一段：工作总结

本文提出 Shot3R，一个从包含镜头切换的单目视频中进行流式多人体四维重建的在线框架。其核心思想是将镜头间对齐与递归状态传播解耦：历史状态仅用于建立新旧镜头之间的空间关系，而独立重置的状态负责新镜头及后续帧的递归推理；在此基础上，仅依赖预测结果的人物关联与相机—人体共享变换进一步维持跨镜头的坐标和身份一致性。多个多人体数据集上的实验表明，Shot3R 总体改善了世界坐标系下的人体重建、相机轨迹和人物身份连续性，并在大幅视角变化下表现出更明显的优势，同时保留了按时间顺序逐帧处理且无需逐视频全局优化的流式推理方式。

### 第二段：局限与未来工作

如何在复杂剪辑、人物反复进出画面以及快速画面变化下持续保持跨镜头的空间与身份一致性，仍是流式多人体四维重建中值得进一步研究的问题。

## 3. 写作边界

1. Conclusion 不重复逐数据集数值、置信区间或表格结论，只总结由实验共同支持的主要发现。
2. 不使用 `boundary gauge`、`read-only gauge`、`clean-reset recurrence`、`same-scene cuts`、`universal editing` 等旧术语。
3. 不把 Shot3R 表述为 Human3R 的插件、后处理或局部补丁，也不在 Conclusion 中强调其基础框架来源。
4. 不笼统声称在所有数据集、所有指标上全面优于现有方法；使用“总体改善”以保留 Harmony4D W-MPJPE 等已知例外。
5. 不把当前受控重建计时写成完整端到端系统速度。
6. 第二段不逐项列举检测器、人物关联、共享平移或单项指标的局部失败，只将复杂剪辑、人物反复进出画面和快速画面变化概括为更高层次的研究问题。
