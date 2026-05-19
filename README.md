# Movie3R

Movie3R 是基于 Human3R 的多镜头人体重建研究项目。当前项目已经完成 V2-V6 旧方向归档，进入 **V7 调研阶段**。

## 当前状态

近期测试显示，Human3R 的明显偏移主要出现在低纹理、弱背景特征、简单场景中的 shot boundary；在 RICH / AvatarReX 等纹理更丰富的数据上，原版 Human3R 往往已经较稳定。

因此，项目当前不再把 V2-V6 的 ShotToken / background AnchorToken 作为主线继续推进，而是先重新调研低纹理场景下的失败模式。

详细说明见：

```text
docs/movie3r/current_research_context.md
docs/movie3r/v7/README.md
```

V2-V6 历史文档已归档到：

```text
docs/movie3r/archive_v2_v6/
```

## 快速开始

### 环境安装

详见：

```text
docs/env_setup_h800_cuda124.md
```

### 推理

```bash
PYTHONPATH=src:. ./.venv/bin/python demo.py \
  --model_path src/human3r_896L.pth \
  --seq_path examples/video.mp4 \
  --output_dir output/demo
```

### 训练

```bash
cd src
./train.sh [num_gpus] [epochs] [batch_size]
```

当前训练代码仍保留历史 V2-V6 实验路径；V7 训练/模型方案尚未确定。

## 项目结构

```text
Movie3R/
├── src/                  # 模型、训练、推理代码
├── config/               # 训练配置
├── scripts/              # 数据处理、诊断、扫描脚本
├── docs/                 # 文档
│   └── movie3r/
│       ├── current_research_context.md
│       ├── v7/
│       └── archive_v2_v6/
├── tasklist/             # 当前 TODO 和历史记录
└── examples/             # 本地示例数据，通常不进入 git
```

## 文档入口

| 文档 | 内容 |
|---|---|
| `docs/movie3r/current_research_context.md` | 当前低纹理 shot change 调研结论 |
| `docs/movie3r/v7/README.md` | V7 调研阶段入口 |
| `docs/movie3r/archive_v2_v6/README.md` | V2-V6 历史归档说明 |
| `docs/train_code_explanation.md` | 训练代码流程解析 |
| `docs/inference.md` | 原版 Human3R 推理说明 |
| `tasklist/TODO.md` | 当前 V7 阶段 TODO |

## License

本项目基于 Human3R / CUT3R 相关代码扩展，遵循其原始许可证约束。
