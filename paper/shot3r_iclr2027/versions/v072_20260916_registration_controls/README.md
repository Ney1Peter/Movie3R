# Shot3R ICLR 2027 v072：传统配准对照整合版

日期：2026-09-16。当前状态：活动工作稿，尚未达到 ICLR 2027 初投稿页数要求。

本版本基于 v071，加入传统几何配准对照及对应补充实验，并同步英文正文、中文阅读版、参考文献、图表和 Overleaf 包。详细修改见 `v072_传统配准实验整合说明.md`。

## 交付文件

- 英文完整稿：`Shot3R_ICLR2027_v072_20260916.pdf`
- 中文阅读稿：`Shot3R_ICLR2027_v072_中文版_20260916.pdf`
- Overleaf 包：`Shot3R_ICLR2027_v072_20260916_Overleaf.zip`
- 英文源文件：`manuscript/`

## 模板和构建状态

- 使用 ICLR 2027 官方 `iclr2027_conference.sty` 和 `.bst`，与官网文件 SHA256 一致；
- pdfLaTeX + BibTeX 为 Overleaf 正式构建路径；本地 Tectonic/XeTeX 验证通过，US Letter，正文 10 pt/11 pt 行距，Times 兼容的 URW Nimbus Roman；
- `Shot3R`/`Human3R` 不再强制使用无衬线体，加粗方法名使用 Nimbus Roman Bold；
- 补充材料表格正文统一为 9 pt，表注为 8 pt，并已移除补充材料中的整表缩放；
- 所有字体已嵌入，没有 Type 3 字体；
- 无未定义引用、未定义文献、overfull box 或致命错误；
- 英文完整稿当前为 41 页；页数压缩暂未处理。

## 历史归档

v067 完整快照、v070 相关记录、去重 PPT 素材和 ICLR 2027 模板核查集中保存在 `../../paper_archive/`。原先误留在本目录中的 v070 README 已按原文保存为 `../../paper_archive/versions/v070_related_materials/V070_RELEASE_NOTES.md`。
