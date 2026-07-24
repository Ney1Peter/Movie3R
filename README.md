# Movie3R

Movie3R 是基于 Human3R 的多镜头人体重建研究项目。

## 正式版本

仓库现在明确维护三条独立版本线：

| 版本 | 定位 | 状态 |
|---|---|---|
| Movie3R-Learned V9.0 | 4-frame AABB 学习式 correction 与 LoRA 权重 | 冻结训练版 |
| Movie3R-Single V14.7 | 单人 short-shot 显式 similarity re-anchoring | 当前默认单人版 |
| Movie3R-Multi V20.0 | GT-ID 多人 shared-Boundary 几何 | 独立研究版 |

统一版本入口、tag、checkpoint hash 和运行命令见：

```text
versions/README.md
```

## 当前默认单人版

当前默认方法独立编号为 **V14.7 Shot-Aware Uniform Similarity Re-anchoring**，
面向 short shot 和稀疏 camera cuts 的流式重对齐：

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

单人版结果、入口和冻结范围见：

```text
LATEST_MODEL.md
versions/v14.7-single/README.md
docs/movie3r/V14_7_SHOT_AWARE_UNIFORM_SIMILARITY_REANCHORING.md
docs/movie3r/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md
docs/movie3r/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md
```

V9 已训练权重保存在 `checkpoints/v9_mixed_60h_pose_human_lora_bs10/`；V20 当前是
严格 GT-ID Oracle 研究版，不应描述成已经完成可部署跨镜头 Re-ID。

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

训练代码仍保留历史实验路径。需要复现正式 V9 训练时，应使用
`movie3r-v9-trained` tag 和 `versions/v9-learned/` 中的冻结配置，不要直接使用当前
master 猜测训练状态。

## 项目结构

```text
Movie3R/
├── versions/             # 三条正式版本线、manifest、checkpoint hash 和复现入口
├── src/                  # 模型、训练、推理代码
├── config/               # 训练配置
├── scripts/              # 数据处理、诊断、扫描脚本
├── docs/                 # 文档
│   └── movie3r/
│       ├── v9/
│       ├── V14_*.md
│       ├── V20_*.md
│       ├── archive_v7/
│       └── archive_v2_v6/
├── tasklist/             # 当前 TODO 和历史记录
└── examples/             # 本地示例数据，通常不进入 git
```

## 文档入口

| 文档 | 内容 |
|---|---|
| `versions/README.md` | V9、单人 V14.7、多人 V20.0 的正式版本目录 |
| `LATEST_MODEL.md` | 当前默认单人版冻结说明 |
| `docs/movie3r/README.md` | 完整研究文档入口 |
| `docs/movie3r/v9/README.md` | V9 历史训练文档入口 |
| `docs/movie3r/V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md` | 当前有效多人几何结论 |
| `docs/movie3r/archive_v7/README.md` | V7 历史归档说明 |
| `docs/movie3r/archive_v2_v6/README.md` | V2-V6 历史归档说明 |
| `docs/train_code_explanation.md` | 训练代码流程解析 |
| `docs/inference.md` | 原版 Human3R 推理说明 |
| `tasklist/TODO.md` | 当前阶段 TODO |

## License

本项目基于 Human3R / CUT3R 相关代码扩展，遵循其原始许可证约束。
