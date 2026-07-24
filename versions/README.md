# Movie3R Version Catalog

本目录是 Movie3R 的正式版本入口。实验编号仍保留在历史报告中，但对外使用时只从
下面三条独立版本线选择，不再把训练模型、单人几何方法和多人研究原型混成一个
“最新版本”。

| 正式版本 | 历史编号 | 状态 | 主要用途 | Git tag |
|---|---|---|---|---|
| [Movie3R-Learned V9.0](v9/README.md) | V9 | 冻结训练版 | 学习式 AABB correction 与 LoRA 权重复现 | `movie3r-v9-trained` |
| [Movie3R-Single V12.0](v12/README.md) | V10.1、V11.4、V14.x | 当前单人主版 | short-shot camera-human 显式流式重对齐 | `movie3r-v12-single` |
| [Movie3R-Multi V13.0](v13/README.md) | V20 Phase 1 v2 | GT-ID 研究版 | 多人 shared-Boundary 几何可行性 | `movie3r-v13-multi` |

## 选择规则

- 需要复现此前训练得到的神经网络 correction：使用 V9.0。
- 需要当前效果最好、可严格流式运行的单人 camera-cut 对齐：使用 V12.0。
- 需要研究多人能否提供冗余 Boundary 约束：使用 V13.0。
- V13.0 当前身份关联使用 GT-ID Oracle，只能用于研究与调试；可部署多人 Re-ID 尚未完成。
- V12.0 适合 short shot 和稀疏 cut，不是无限长度 world mapping。

目录结构固定为：

```text
versions/
├── v9/                  # 已训练学习式版本
├── v12/                 # 单人正式版、viewer、历史消融
└── v13/                 # 多人正式研究版、viewer、数据探针
```

每个版本目录均包含：

1. 独立方法说明；
2. 运行入口、代码 commit 与冻结 tag；
3. checkpoint 及 SHA-256；
4. 主入口、输入输出和复现命令；
5. 已知限制和不能宣称的结论；
6. 机器可读 `manifest.json`。

## Git 使用

需要完全隔离代码时，使用 worktree，不要在当前工作区反复 checkout：

```bash
git worktree add ../Movie3R-v9 movie3r-v9-trained
git worktree add ../Movie3R-single movie3r-v12-single
git worktree add ../Movie3R-multi movie3r-v13-multi
```

模型权重不提交进 Git。V9 三份大权重保存在
`checkpoints/v9_mixed_60h_pose_human_lora_bs10/`，并通过 manifest 中的 SHA-256
验证；它们与旧 archive 文件是同 inode 的硬链接，不重复占用磁盘。

## 为什么正式编号是 V9、V12、V13

早期的 `V10/V11/V14/V20` 是按实验问题增长的研究编号，不是连续的软件发布号。
现在把可独立使用的结果重新冻结为三条正式版本线：训练版 V9、单人版 V12、多人版
V13。旧编号只留在各版本的 `docs/` 和 `experiments/` 中，用于追溯方法来源，不再
出现在仓库根目录作为并列版本。
