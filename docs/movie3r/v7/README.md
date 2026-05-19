# Movie3R V7

## 阶段定位

V7 当前不是一个已经确定的模型方案，而是新的调研阶段。

V2-V6 的主要工作围绕 ShotToken、background feature anchor、AnchorToken 和 pose-only camera adapter 展开。近期测试表明，这些方向没有完全命中当前最重要的失败场景：低纹理、弱背景特征、简单场景中的 shot boundary 偏移。

## 当前目标

V7 当前只做三件事：

1. 收集 Human3R 在低纹理 shot change 场景中的失败案例。
2. 区分低纹理失败和 RICH / AvatarReX 高纹理稳定场景之间的差异。
3. 在充分调研前，不急于确定新的模型结构或训练路线。

## 当前约束

- 暂不继续把 V2-V6 作为主线扩展。
- 暂不默认背景特征匹配一定可靠。
- 暂不修改模型、训练、推理代码来定义 V7。
- 当前先整理文档、数据、现象和失败案例。

## 相关文档

```text
docs/movie3r/current_research_context.md
docs/movie3r/archive_v2_v6/README.md
tasklist/TODO.md
```
