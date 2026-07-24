# Movie3R-Learned V9.0

## 定位

V9.0 是此前已经完成训练的学习式版本，不是当前 V14.7 显式几何主线。它在原版
Human3R 上增加 relation correct token、pose-head LoRA、human-head LoRA 和 human
latent correction，训练目标是 4 帧 `AABB` camera-cut pattern。

```text
4-frame RGB AABB stream
-> Human3R encoder / recurrent decoder
-> relation_v8_2 correction prompt
-> corrected pose token + corrected human latent
-> LoRA-adapted camera and SMPL-X heads
-> camera + pointmap + SMPL-X
```

它不包含 V14.7 的 pre-decode hard reset、V16、DA3/Keypoint shared scale 或显式
Boundary，也不包含 V20 多人 consensus。

## 冻结身份

| 项目 | 值 |
|---|---|
| 训练代码 commit | `6eb64cb2158fb443d53cd4f1713af1899fe5a026` |
| Git tag | `movie3r-v9-trained` |
| 训练开始 | 2026-06-12 18:02 CST |
| 训练配置 | `config/train_v9_mixed_avatarrex_thuman_60h_pose_human_lora_bs10.yaml` |
| 完整 resolved config | `versions/v9-learned/resolved_config.yaml` |
| 原始 Human3R 初始化 | `src/human3r_896L.pth` |
| 推荐推理权重 | `checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth` |

训练代码 commit 选择 `6eb64cb`，因为正式训练在该 commit 提交约 3 分钟后启动，
且该 commit 首次加入了这次运行使用的精确配置。

## 权重

稳定目录中保留三份 checkpoint：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `checkpoint-best.pth` | 4,985,106,570 B | `4376d623a7fb658da33eac03912697ba9a21aa919a25e6a5761eedbe745e25a8` |
| `checkpoint-last.pth` | 4,985,106,570 B | `a1f2173f45db82c914dc646afdae28f1a511279920666217a749f7ba7e07a02d` |
| `checkpoint-final.pth` | 4,831,184,406 B | `3fb2799420f7fd3caa63a47c9cde73090a6f93383520363484eb5158e446fceb` |

resolved Hydra config 的 SHA-256 为
`93c5a8acca8263eeac89a52b944c2ef09630a5fb39762d05d1185639c0f2150f`；训练日志
SHA-256 为 `7ce3c82c88e12af52d7d15e2891fc7bf9ffc4622d5a6a3674f2091e20c7f79c6`。

## 推理

当前兼容 runner：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_human3r_save_output.py \
  --model_path checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth \
  --seq_path /path/to/aabb_images \
  --output_dir output/v9_inference/example \
  --device cuda:0 \
  --size 512
```

不要添加 `--strict_original_human3r`，否则会主动关闭 V9 correction 分支。当前主干比
V9 训练 commit 多了一些后续未启用模块，加载时可能打印这些后续模块的 missing keys；
这不表示 V9 主体权重丢失。要求位级历史复现时，应在 `movie3r-v9-trained` worktree 中
运行该 tag 自带的 `demo.py`。

## 输入输出

输入是完整 RGB 画面按 Human3R 规则缩放后的连续流。V9 的主要训练分布是四帧
`A,A,B,B`。输出包括 camera pose、depth/pointmap、confidence、mask 和 SMPL-X。

V9 可以用于研究 learned relation correction，但不应与 V14.7 的显式 hard-reset
Boundary 方法混称为同一个版本。25 帧、多次 cut 的运行属于分布外定性压力测试，不能
替代它的 4-frame AABB 结论。

## 详细文档

- `docs/movie3r/v9/METHOD_OVERVIEW.md`
- `docs/movie3r/v9/MODEL_ARCHITECTURE_DETAILS.md`
- `docs/movie3r/v9/EXPERIMENT_RECAP_20260630.md`
- `docs/movie3r/v9/GUARDRAILS.md`
