# Movie3R V8

## 阶段定位

V8 是新的调研入口。V7 的后处理式 floor / human / scene correction 与 implicit token adapter 路线已经归档，不再作为当前主线继续扩展。

## 当前起点

V8 需要重新定义 shot-change 场景下的目标、约束和最小实验，不默认沿用 V7 的 offline teacher、post-processing correction、stable window 或 pseudo-label 生成流程。

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
