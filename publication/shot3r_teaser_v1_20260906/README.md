# Shot3R teaser 第一版素材与初稿

论文正文、方法、原始实验结果均未修改。本目录是独立的可视化交付目录。

**可编辑 PPT 已补充：**[全景版](editable/Shot3R_teaser_editable.pptx)、[时间展开版](editable/Shot3R_teaser_temporal_editable.pptx)、[三镜头概念排版模板](editable/Shot3R_concept_template.pptx)。[打开与编辑方法](editable/使用说明.md)，[三镜头概念图的中英文生成提示词](editable/生成提示词_三镜头三人_中英文.md)。后续概念图方向不要求与现有实验逐帧一致；下文仍是原真实结果版的素材溯源说明。

## 建议先看

- [全景版预览](figures/Shot3R_teaser_v1_panorama_preview.png)：更接近 Human3R Figure 1 的“场景全景 + 单目时间带”，突出三人、相机、统一世界坐标和大角度切镜。
- [时间展开版预览](figures/Shot3R_teaser_v1_temporal_preview.png)：逐帧 RGB 与固定视角人体重建对应，时间连续性和姿态变化更容易阅读。
- [连续 RGB—重建视频](assets/egohumans/continuous/Shot3R_continuous_real_predictions.mp4)：f015–f074 共 60 个连续帧，按源片段 20 FPS 播放，时长 3 秒；不是推理速度演示。
- [候选 RGB 总览](selection/candidate_rgb_contact_sheet.jpg)。
- [真实相邻切镜帧核查](selection/exact_boundary_f049_f050.jpg)。

两个正式版式均提供可编辑 SVG、单页 PDF 和 3600 像素宽 PNG。文字、箭头、时间标记和图例为矢量对象；场景和人体图为独立可替换的真实渲染素材。

打包下载：[图片与排版素材包](delivery/Shot3R_teaser_visual_assets.zip)；[含完整预测、网格和连续帧的完整包](delivery/Shot3R_teaser_full_bundle.zip)。前者适合直接绘图，后者适合重新渲染。轻量包的说明中可能提到仅完整包包含的网格与预测文件。

## 样例选择

| 样例 | 人数与切镜 | 完整片段 IDF1 | 建议 |
| --- | --- | ---: | --- |
| EgoHumans / legoassemble | 三人，一次 176.75° 切镜 | 0.95105 | 当前主图推荐；多人交互、室内背景和身份关联较清楚 |
| Harmony4D / case_01 | 两名标注主体，另有背景人物；两次 179.43° / 177.93° 切镜 | 0.55944 | 已导出真实三镜头备选，不宜强调身份稳定 |
| Harmony4D / case_02 | 两名标注主体，另有背景人物；两次 179.49° / 177.70° 切镜 | 0.59940 | 已导出原帧与网格备用，世界位置误差偏大 |
| Harmony4D / case_03 | 两名主体、多人背景；两次大角度切镜 | 0.25492 | 只保留候选总览，不选入主图 |
| Harmony4D / case_04 | 两名主体、多人背景；两次大角度切镜 | 0.32333 | 只保留候选总览，不选入主图 |

EgoHumans 的 W-MPJPE 为 336.985 mm、WA-MPJPE 为 182.106 mm；这些是该片段指标，不是数据集均值。精确来源在 `assets/egohumans/provenance.json`。Harmony4D 候选来自既有多跳变补充协议，源目录属于 train 侧序列，不冒充主测试集样例。

**目前尚未同时满足“两个以上切镜 + 很强的身份稳定性 + 好看场景”这三个条件。** 因此主图只有一次真实切镜。没有把独立片段拼成不存在的三镜头运行，也没有人工改色掩盖 ID 错配。三镜头诊断版单独命名为 `NOT_recommended`，不要直接放进论文当作最佳结果。

各镜头内相机位置基本固定，所以相机视锥会局部重合，不能像 Human3R 的移动长镜头那样自然铺开成长轨迹。没有为了构图拉开相机或平移人体。如果最终一定需要“两次切镜 + 长距离展开”的主图，需要另选真实移动相机的多镜头连续序列，再用当前固定方法导出结果；本次未增加推理实验。

## 素材怎么取

所有路径均相对本目录：

| 文件 / 目录 | 用途 |
| --- | --- |
| `figures/Shot3R_teaser_v1_panorama.*` | 全景版主图，SVG / PDF / PNG |
| `figures/Shot3R_teaser_v1_temporal.*` | 时间展开版主图 |
| `assets/egohumans/rgb/` | f015、f032、f042、f049、f050、f061、f074 原尺寸 RGB |
| `assets/egohumans/renders/` | 对应逐帧透明网格 PNG，三个全景显示视角和固定视角参数 |
| `assets/egohumans/continuous/rgb/` | f015–f074 连续 60 张原尺寸 RGB |
| `assets/egohumans/continuous/renders/` | 连续 60 张透明网格渲染图 |
| `assets/egohumans/meshes/fXXX/` | 每个人的原生世界坐标 PLY、当前帧全部人体 GLB |
| `assets/egohumans/selected_predictions_native_world.npz` | 所选帧的网格、关节、相机 c2w、有效标记与原始 ID |
| `assets/egohumans/full_predictions_native_world.npz` | 全部 100 帧原生预测，供后续重新选帧或交互查看 |
| `assets/egohumans/scene_points_native_world.ply` | 方法自身深度回放生成的未作展示剖切的点云，已按置信度和人体掩码筛选 |
| `assets/egohumans/scene_payload/` | 既有回放的 RGB、深度、置信度、相机和人体包，用于重新渲染 |
| `assets/case_01/`、`assets/case_02/` | 两个真实三镜头候选的原帧、网格、相机与透明渲染 |
| `CAPTIONS_ZH_EN.md` | 简短中英文图注、术语翻译、展示口径 |
| `validation_report.json` | 导出数组与源结果相等、RGB 哈希、ID、透明通道、PDF、视频检查 |

`full_predictions_native_world.npz` 保持原始坐标，未做 GT 对齐、逐镜头重定位或缩放。渲染仅统一了竖直轴显示方向、固定虚拟视角与光照。颜色始终按 `persistent_ids` 查同一颜色表；没有用 GT 身份替换预测身份。

原 RGB 是既有暂存图像的逐字节副本。EgoHumans 暂存 JPEG 在前一次导出中曾经解码重编码，因此不声称它与数据集压缩包成员逐字节相同。源数据已有人脸匿名化，本次予以保留。

## 图的证据范围

- 本图是指标与观感共同挑选的展示样例，不宣称具有数据集平均代表性。
- 主图采用 f015、f032、f042、f050、f061、f074 六个时刻。f042 和 f050 不是边界相邻帧，图中标明具体时刻。真正相邻的 f049、f050 也已导出；f049 只有两个有效人体预测，不能声称全程无漏检。
- 所有显示时刻的人体有效预测均保留。Harmony4D 背景人物误检和 ID 变化也未删除或改色。
- 场景点云来自该方法自身已有冻结权重回放；人体和相机来自已有正式缓存。没有 GT 网格、外部方法场景或生成式补图。
- 全景采用墙体剖切、置信度筛选、降采样和淡化配色，是显示处理，不是方法新增的场景优化。参数见 `renders/world_a.json`，未作展示剖切的点云另存。
- 不标注尚未测得的推理 FPS、端到端时延或加速倍数。视频按源片段帧率播放。

## 复现

在 `/data/wangzheng/iJCV-CODE` 运行，使用项目已有环境，无需重新推理或训练：

```bash
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/build_teaser.py audit
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/build_teaser.py export
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/build_teaser.py render
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/build_teaser.py world --cases egohumans
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/export_continuous_preview.py
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/compose_teaser.py
Movie3R/.venv/bin/python Movie3R/publication/shot3r_teaser_v1_20260906/validate_assets.py
```

脚本只写本目录下的新素材，不写既有实验目录。再次运行会重新生成本目录的交付文件；如要手工修改 SVG，建议另存为新版本，避免被排版脚本覆盖。
