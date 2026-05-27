# Movie3R 文档入口

Movie3R 当前准备进入 **V8 调研阶段**。V7 的后处理式 correction / implicit token adapter 路线已归档。

## 当前判断

近期测试显示，Human3R 在 RICH / AvatarReX 等纹理丰富数据上通常表现稳定；明显偏移更多出现在低纹理、弱背景特征、简单场景中的 shot boundary，尤其是镜头变化后的第一帧。

因此，项目当前重点不是继续扩展 V2-V6 的 ShotToken / background AnchorToken 路线，而是重新调研低纹理场景下 Human3R 的失败模式。

## 当前文档

| 文档 | 内容 |
|---|---|
| [当前调研情况](current_research_context.md) | 低纹理 shot change 失败场景和方向边界 |
| [V8 入口](v8/README.md) | V8 调研阶段说明 |
| [V7 历史归档](archive_v7/README.md) | V7 后处理式 correction / implicit token adapter 记录 |
| [V2-V6 历史归档](archive_v2_v6/README.md) | 旧 ShotToken / AnchorToken / V6 记录 |
| [训练代码入口](train_code.md) | 训练代码说明入口 |

## 历史分水岭

V2-V6 文档和报告已归档到：

```text
docs/movie3r/archive_v2_v6/
```

V7 文档和运行脚本已归档到：

```text
docs/movie3r/archive_v7/
scripts/archive_v7/
```

归档内容保留用于复盘，不再代表当前主线。

## 当前代码状态

模型、训练、推理代码仍保留历史 V2-V7 实验路径以便复盘和兼容旧 checkpoint。V8 尚未确定新的模型或训练实现。
