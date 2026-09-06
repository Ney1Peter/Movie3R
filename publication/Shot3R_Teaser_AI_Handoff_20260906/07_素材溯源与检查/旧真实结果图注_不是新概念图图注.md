# Teaser 图注与中英文用语

## 全景版图注

**English**

Shot3R reconstructs multiple people in a shared 3D world from a monocular stream with abrupt shot changes. This EgoHumans example spans a 177° viewpoint change. Colors denote predicted persistent identities, camera frusta show predicted views, and faint meshes indicate earlier poses.

**中文**

Shot3R 从包含突发镜头切换的单目视频流中，在统一三维世界坐标系下重建多个人体。图中 EgoHumans 样例包含约 177° 的视角变化。颜色表示预测的持续人物身份，相机视锥表示预测视角，浅色网格表示较早时刻的姿态。

## 时间展开版图注

**English**

Streaming multi-person reconstruction across a 177° shot change. Sampled monocular inputs are paired with their predicted human meshes, rendered from one fixed world viewpoint. Colors follow the model's persistent identity assignments without manual relabeling.

**中文**

跨越约 177° 镜头切换的流式多人体重建。抽样的单目输入帧与相应人体网格逐一对应，所有网格均从同一固定世界视角渲染。人物颜色遵循模型预测的持续身份关联，未作人工重标注。

## 常用图中文字

| 英文 | 中文 |
| --- | --- |
| Streaming multi-person 4D reconstruction | 流式多人体四维重建 |
| A shared 3D world | 统一的三维世界坐标系 |
| Same world, same viewing angle | 同一世界坐标系、同一渲染视角 |
| Across a 177° viewpoint change | 跨越约 177° 的视角变化 |
| Monocular input stream | 单目视频输入流 |
| Predicted identities | 预测人物身份 |
| Persistent person IDs | 持续关联的人物 ID |
| Predicted cameras | 预测相机 |
| Shot cut | 镜头切换 |
| Before / After the cut | 切镜前 / 切镜后 |
| Shot 1 / Shot 2 / Shot 3 | 镜头 1 / 镜头 2 / 镜头 3 |
| Faint meshes: earlier poses | 浅色网格：较早时刻的姿态 |
| Sampled frames; time increases left to right | 抽样帧；时间从左向右推进 |
| Full-clip IDF1: 0.951 | 完整测试片段的 IDF1：0.951 |
| Source-rate playback (20 FPS), not measured inference speed | 按源片段 20 FPS 播放，并非测得的推理速度 |
| Diagnostic preview, not the recommended teaser | 诊断预览，不推荐用作主 teaser |

完整文字映射见 `figure_labels_zh_en.json`。SVG 与排版脚本保留中文注释，不显示在英文图中。

## 使用时必须保留的语义

1. 输入是依时间取自不同镜头的单目序列，不能描述成同时多视角重建。
2. 177° 来自评估用标定信息，仅作图中角度注释，不是提供给方法的输入。
3. 0.951 是该单个 100 帧测试片段的 IDF1，不是整个 EgoHumans 数据集均值。
4. 全景图是截至 f074 的静态汇总：背景来自 f049、f050 的既有回放；浅色人体来自 f042，实色人体来自 f074。它不是同一时刻存在六个人。
5. 两侧小图 f042 与 f050 是抽样对照，不是直接相邻的边界帧。真正的边界 f049、f050 已另存，f049 有漏检，不能写“每一帧身份和检测都完美”。
6. 场景图为便于观察人体，采用前侧墙体剖切显示。未裁切点云和原始预测另存，不能把显示用剖切称为方法对场景进行了补全或修复。
