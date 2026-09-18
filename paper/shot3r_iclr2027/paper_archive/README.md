# Shot3R 论文版本与素材档案

本档案现已纳入 Movie3R 主仓库，规范路径为
`Movie3R/paper/shot3r_iclr2027/paper_archive/`。后续论文版本、修改记录和
PPT 素材均以主仓库内的路径为准。

本目录集中保存需要长期保留的论文版本、修改记录、可编辑 PPT 素材和 ICLR 2027 模板核查材料。归档范围按当前约定限定为 v067 和 v070 相关内容；v072 仍是工作区中的当前活动稿，不作为历史版本重复归档。

## 目录

- `versions/v067_exact_snapshot/`：v067 的完整逐文件副本，包括英文和中文 PDF、Overleaf ZIP、源码、图表和原始说明。
- `versions/v070_related_materials/`：目前能够确认的 v070 发布说明和修改报告。当前磁盘、Git 工作树以及用户目录中没有找到完整 v070 源码包或 PDF，因此这里没有把 v071/v072 误标为 v070。
- `ppt_assets/`：四份去重后的可编辑 PPTX。v067 与 v072 中对应文件的 SHA256 完全一致。
- `change_records/`：全局版本规则、论文修改决策、语言修改记录和 v067 执行报告。
- `template_audit/`：ICLR 2027 官方模板原件、官方来源、当前稿件核查报告、检查脚本和原始检查输出。
- `manifests/SHA256SUMS`：本档案中所有文件的 SHA256 校验值。

## 当前结论

v067 已完整归档。v070 只有可确认的相关材料，缺少当时说明中列出的完整 PDF、正文预览 PDF 和 Overleaf ZIP，详情见 `versions/v070_related_materials/RECOVERY_STATUS.md`。

当前 v072 使用的 ICLR 2027 样式文件与官网压缩包逐字一致，US Letter、正文 10 pt/11 pt 行距、匿名模式、声明与参考文献/附录顺序均正确。正文目前延伸到第 11 页，不满足初投稿最多 9 页的硬限制。详细结果见 `template_audit/ICLR_2027_TEMPLATE_AUDIT_20260918.md`。

2026-09-18 已完成字体与补充表格修正：方法名继承 Times 兼容 Roman 字体，加粗 `Shot3R` 使用对应 Roman Bold；实际补充材料表格正文统一为 9 pt、表注为 8 pt，并移除整表缩放。全新 Overleaf 包复编通过。核查记录见 `template_audit/TYPOGRAPHY_AUDIT_20260918.md`。

## 保留与清理规则

1. 本档案只接收 v067 完整快照、v070 相关材料和跨版本通用 PPT/记录。
2. v070 完整包若以后找回，应放入 `versions/v070_exact_snapshot/`，同时保留原始文件名和校验值。
3. 不用 v072 文件拼装或反向推测一个“完整 v070”，避免版本证据失真。
4. 删除工作区中的其他版本前，先运行 `sha256sum -c manifests/SHA256SUMS` 验证档案。
5. 当前活动稿在形成新的可验证交付版本前继续保留在 `versions/v072_20260916_registration_controls/`。
