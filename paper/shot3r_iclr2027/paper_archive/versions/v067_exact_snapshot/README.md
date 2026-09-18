# Shot3R ICLR 2027 v067：最终轻量修订版

本版本基于 v066，只执行作者逐项确认的最后一轮轻量修改，不改变标题、方法、模型权重、实验数值、主表结构、Figure 1、正文定性图或 AI Use Statement：

- 删除 Section 4.1 中重复的数据集作用说明；
- 统一 Abstract、Introduction、Contribution、Method 和 Conclusion 中递归历史冲突作用的表述；
- Abstract 删除 FPS 与相对 Human3R 开销，以极端视角改善和逐帧流式推理收尾；
- Related Work 第一段按技术路线重组，保留全部引用；
- Conclusion 将跨任务概括收紧为由当前实验支持的观察；
- 分别写清镜头检测器的 learning rate 和 weight decay；
- Supplement Figure 9 使用同一矢量图的两个等比例裁切面板，放大七方法定性结果并消除页面上方留白。

英文完整稿共 34 页：科学正文为第 1--9 页，Conclusion 完整位于第 9 页；声明第 10 页，参考文献第 11 页开始，补充材料第 16 页开始。正文显示 65 条参考文献。中文阅读版共 13 页，与英文正文的修改同步。

交付文件：

- `Shot3R_ICLR2027_v067_20260911.pdf`
- `Shot3R_ICLR2027_v067_20260911_Overleaf.zip`
- `Shot3R_ICLR2027_v067_中文版_20260911.pdf`
- `v067_最终轻量修改执行报告.md`

Overleaf ZIP 包含 63 个匿名编译依赖，已在全新目录中独立完成 pdfLaTeX + BibTeX 编译。包内不含中文稿、PPTX、内部审阅记录、构建日志、机器路径或已退出论文展示的实验材料。
