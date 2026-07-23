# Movie3R

Movie3R 是基于 Human3R 的多镜头人体重建研究项目。

## 最新冻结版本

当前默认方法是面向 short shot 和稀疏 camera cuts 的流式重对齐：

```text
pre-decode Human3R hard reset
-> Fixed Explicit
-> V16 bounded torso-motion rotation
-> V11.4 fused DA3/Keypoint shared shot scale
-> one explicit translation
-> one fixed shot-level Boundary
```

Conditional VGGT 和 V14.2 continuity 默认关闭。该版本改善 short-horizon
camera-human placement，但存在 scene trade-off，且不适用于无限长度多 cut mapping。

最新版本、结果、入口和冻结范围见：

```text
LATEST_MODEL.md
docs/movie3r/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md
docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md
```

V2-V8 及其他失败/诊断实验仍保留在历史目录，不作为当前默认方法：

```text
archive/20260721/
docs/movie3r/archive_v2_v6/
docs/movie3r/archive_v7/
docs/movie3r/archive_v8/
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

当前训练代码仍保留历史 V2-V7 实验路径；V8 训练/模型方案尚未确定。

## 项目结构

```text
Movie3R/
├── src/                  # 模型、训练、推理代码
├── config/               # 训练配置
├── scripts/              # 数据处理、诊断、扫描脚本
├── docs/                 # 文档
│   └── movie3r/
│       ├── current_research_context.md
│       ├── v8/
│       ├── archive_v7/
│       └── archive_v2_v6/
├── tasklist/             # 当前 TODO 和历史记录
└── examples/             # 本地示例数据，通常不进入 git
```

## 文档入口

| 文档 | 内容 |
|---|---|
| `docs/movie3r/v8/START_HERE.md` | 新对话优先阅读：背景动机、V2-V7 速览、失败原因和 V8 起点 |
| `docs/movie3r/current_research_context.md` | 当前低纹理 shot change 调研结论 |
| `docs/movie3r/v8/README.md` | V8 调研阶段入口 |
| `docs/movie3r/archive_v7/README.md` | V7 历史归档说明 |
| `docs/movie3r/archive_v2_v6/README.md` | V2-V6 历史归档说明 |
| `docs/train_code_explanation.md` | 训练代码流程解析 |
| `docs/inference.md` | 原版 Human3R 推理说明 |
| `tasklist/TODO.md` | 当前阶段 TODO |

## License

本项目基于 Human3R / CUT3R 相关代码扩展，遵循其原始许可证约束。
