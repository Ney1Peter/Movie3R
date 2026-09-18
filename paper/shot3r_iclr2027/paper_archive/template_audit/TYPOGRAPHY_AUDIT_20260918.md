# v072 字体与补充表格核查

核查对象：`versions/v072_20260916_registration_controls/manuscript/`。

## 已修正

- `Shot3R` 和 `Human3R` 不再由宏强制切换为无衬线体，而是继承 ICLR
  正文的 Times 兼容 Roman 字体。
- pdfLaTeX 成品中的普通方法名使用 `NimbusRomNo9L-Regu`，加粗方法名使用
  `NimbusRomNo9L-Medi`。标题继续使用 ICLR 官方样式指定的无衬线粗体。
- 本地 XeTeX 校验路径显式绑定 Nimbus Roman 的 Regular、Bold、Italic 和
  BoldItalic，避免粗体解析失败后静默回退到普通体。
- 所有实际编入补充材料的表格正文统一为 9 pt；表注统一保留为 8 pt。
- 实际补充材料依赖中不再包含 `\scriptsize` 或整表 `\resizebox`。宽表通过
  自动换行、缩短表头和上下分表排版。

## 验证结果

- 官方 pdfLaTeX + BibTeX 全新目录构建成功。
- 无 overfull box、未定义引用、未定义文献或致命错误。
- 所有字体均已嵌入，无 Type 3 字体。
- Overleaf ZIP 已重建为 77 个实际编译依赖文件，并在全新目录复编通过。
- 当前页数为 41；本轮按要求不处理页数。
