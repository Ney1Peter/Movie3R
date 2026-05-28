# Movie3R V8

新对话请先读：

```text
docs/movie3r/v8/START_HERE.md
```

## 阶段定位

V8 是新的调研入口。V7 的后处理式 floor / human / scene correction 与 implicit token adapter 路线已经归档，不再作为当前主线继续扩展。

V8 的核心背景是：Human3R 在纹理丰富数据上通常稳定，但在低纹理、弱背景特征、简单场景的 shot boundary 后第一帧容易出现相机/world gauge 偏移。我们要针对这个具体失败模式重新设计实验。

## 当前起点

V8 需要重新定义 shot-change 场景下的目标、约束和最小实验，不默认沿用 V7 的 offline teacher、post-processing correction、stable window 或 pseudo-label 生成流程。

第一阶段建议只做三件事：

1. 建立低纹理 boundary failure set 和高纹理 stable control set。
2. 明确不依赖后处理 teacher 的评价指标和可视化协议。
3. 再决定是否需要模型结构、训练目标或数据构造上的修改。

## 初始约束

- 不以 offline 后处理 correction 作为主方案。
- 不默认依赖 post-shot stable window、BA、pose graph 或显式 floor/SMPL anchor。
- 先明确新的失败模式、可用输入和训练目标，再新增模型结构。
- V7 归档内容只作为诊断经验和负例参考。

## 相关归档

```text
docs/movie3r/archive_v7/
scripts/archive_v7/
```

## 历史速览

V2-V6 尝试 ShotToken / background AnchorToken / pose adapter，但低纹理场景缺少可靠背景 anchor，方向不稳健。

V7 尝试 offline floor / human / scene correction teacher 和 implicit token adapter。它能作为诊断工具，但依赖 SMPL、floor/background 点云、参考帧和 post-shot 信息，真实视频上不够可靠，因此归档。
