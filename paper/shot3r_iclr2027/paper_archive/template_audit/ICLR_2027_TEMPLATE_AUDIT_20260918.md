# Shot3R ICLR 2027 模板核查

核查对象：`versions/v072_20260916_registration_controls/manuscript/main.tex` 及其英文完整 PDF。

## 结论

当前稿件使用了正确的 ICLR 2027 官方模板，页面、正文主字体、匿名模式和强制声明结构正确。当前唯一会直接阻止初投稿的格式问题是正文超页：官方上限为 9 页，当前 Conclusion 位于第 11 页。

| 项目 | 官方要求 | 当前状态 | 结论 |
|---|---|---|---|
| 模板 | 使用 ICLR 2027 官方 LaTeX 样式，不得修改参数 | `.sty`、`.bst`、`fancyhdr.sty`、`natbib.sty`、`math_commands.tex` 与官网 ZIP 的 SHA256 完全一致 | 通过 |
| 页面 | US Letter | 612 × 792 pt | 通过 |
| 正文区域 | 5.5 × 9 inch | 由未修改的官方样式设置 | 通过 |
| 正文字号 | 10 pt，11 pt 行距 | 由未修改的官方样式设置 | 通过 |
| 正文主字体 | Times New Roman preferred | pdfLaTeX 的 `times` 包生成嵌入式 URW Nimbus Roman，属于标准 Times 兼容字体；`Shot3R`/`Human3R` 宏继承正文 Roman 字体 | 通过 |
| 方法名粗体 | 使用正文对应的粗体字形 | 加粗的 `Shot3R` 使用嵌入式 `NimbusRomNo9L-Medi`，不再使用 Helvetica/Nimbus Sans | 通过 |
| 补充表格 | 字号清晰且一致 | 所有实际编入补充材料的表格正文统一为 9 pt，表注为 8 pt；不再使用整表缩放 | 通过 |
| 字体嵌入 | PDF 应可稳定显示和打印 | 全部字体已嵌入；已消除补充材料两张图中的 Type 3 字体 | 通过 |
| 匿名 | 双盲，正文和补充材料不得泄露作者身份 | `\iclrfinalcopy` 保持注释，页眉与作者块为匿名；PDF Author 元数据为空 | 通过 |
| 摘要 | 单段 | 当前为单段，约 177 个英文词；官方未规定摘要词数上限 | 通过 |
| 初投稿正文 | 最多 9 页 | Experiments 延续到第 10 页，Conclusion 在第 11 页 | **不通过** |
| AI Use Statement | 必须有，正文后、参考文献前，不计页数 | 第 12 页，已明确披露文献、写作、翻译、实验讨论、结果解释、代码和制图辅助 | 通过 |
| Ethics Statement | 建议有，正文后、参考文献前，不计页数 | 第 12 页，单段 | 通过 |
| Reproducibility Statement | 建议为一段，正文后、参考文献前，不计页数 | 第 12 页，单段并指向协议、实现、方法细节和完整结果附录 | 通过 |
| 参考文献 | 不计页数，可不限页 | 第 13–17 页 | 通过 |
| 附录 | 位于参考文献之后，可不限页 | 从第 18 页开始 | 通过 |
| 编译 | LaTeX；推荐 pdfLaTeX | 全新目录中以 pdfLaTeX + BibTeX 编译成功 | 通过 |
| 引用和版面错误 | 不应有未解析引用或越界 | 无未定义引用、未定义文献、overfull box 或致命错误 | 通过 |

## 官方段落与顺序

官方模板明确要求 Abstract 只能有一个段落。当前摘要满足这一点。作者指南要求 AI Use Statement；Ethics Statement 和 Reproducibility Statement 为推荐项。当前三个声明均位于 Conclusion 之后、References 之前，并共同占一页。参考文献之后才开始 Supplementary Material，顺序正确。

首页 teaser 放在匿名作者块和 Abstract 之间。模板和作者指南没有禁止这一位置，因此不构成格式违规；它仍计入正文页数。

## 字体核查与修正

正文通过 `\usepackage{iclr2027_conference,times}` 使用官方模板建议的 Times 系列。pdfLaTeX 中实际字体名为 `NimbusRomNo9L`，这是 URW 的 Times 兼容字体，不是字号或字体替换错误。数学公式使用 Computer Modern/AMS 数学字体，符合 LaTeX 的正常行为。

原 PDF 在补充材料第 31–32 页的两张 Matplotlib 图中包含 Type 3 DejaVu Sans。两图已用 `pdf.fonttype = 42` 和 `ps.fonttype = 42` 重新导出，最新版 PDF 和 Overleaf ZIP 已同步；当前完整 PDF 不再包含 Type 3 字体。图内的 Liberation Sans/DejaVu Sans 属于嵌入式矢量图字体。官方措辞是 Times New Roman “preferred throughout”，没有规定图内标签必须全部改成 Times。

此前方法名宏使用 `\textsf{Shot3R}` 和 `\textsf{Human3R}`，使表格中的加粗方法名变成 Helvetica/Nimbus Sans。现已改为继承上下文字体，因此正文使用 Times 兼容 Roman，`\textbf{\method{}}` 使用对应的 Nimbus Roman Bold。补充材料表格统一调用 9 pt 的 `\supptablesize`；原有的 `\scriptsize` 和整表 `\resizebox` 已从实际编入的补充材料依赖中清除，宽表改为换行或上下分表。

## 必须完成的后续工作

页数压缩按当前安排暂缓。后续处理时需要通过内容编辑把完整 Conclusion 收回第 9 页，不能修改官方样式、字号、行距或页边距，也不应使用负间距规避页数检查。

在正文压缩完成后，应重新运行 `check_iclr2027.sh`，并人工确认第 9 页包含完整 Conclusion、第 10 页开始为不计页数的声明或 References。
