# Movie3R 新服务器迁移说明

更新时间：2026-05-14

## 1. Git 同步范围

本仓库应该通过 git 同步代码、配置、文档、脚本，以及已经归档到 V2-V6 历史目录的 AnchorToken 可视化证据：

```text
docs/movie3r/archive_v2_v6/anchor_token_report_v1/README.md
docs/movie3r/archive_v2_v6/anchor_token_report_v1/01_aabb_step1/**
```

当前 V2-V6 报告文件已经从 `output/` 移入文档归档目录。后续新的 `output/` 产物默认仍按本地实验输出处理，不应自动进入 git。

删除旧服务器项目前，先确认：

```bash
git status -sb
git branch -vv
git log --oneline --decorate -5
```

如果 `git status -sb` 仍然有 `M` 或 `??`，说明还有本地改动没有 commit。确认后再执行 commit/push。

注意：不要把 GitHub token 写进 remote URL 或文档。新服务器建议用 SSH remote，或者用凭据管理器保存 token。

## 2. 不通过 git 同步的本地资源

下面这些资源被 `.gitignore` 忽略，需要按需手动复制、重新下载或重新生成：

| 路径 | 当前大小 | 说明 |
|------|----------|------|
| `/workspace/code/Movie3R/src/human3r_896L.pth` | 4.4G | Human3R/Movie3R 主模型权重，必须手动放回同一路径 |
| `/workspace/code/Movie3R/src/models/` | 3.2G | SMPL/SMPL-X 模型、regressor、mean params，许可证资源，不建议进 git |
| `/workspace/code/Movie3R/src/checkpoints/` | 16M | 本地 checkpoint 目录，按需复制 |
| `/workspace/code/Movie3R/data/` | 7.2M | 本地 demo 视频，按需复制 |
| `/workspace/code/Movie3R/.venv/` | 7.7G | Python 虚拟环境，不迁移，重新创建 |
| `/workspace/data/RICH/RICH_4Human3R/Training/` | 211G | RICH 训练/验证数据，报告脚本依赖 |
| `/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1/` | 923K | 当前推荐的 offline anchor cache |
| `/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_v1/` | 145K | 旧版 guitar anchor cache，按需保留 |
| `/workspace/data/Avatarrex/` | 1.4T | AvatarReX 数据集，训练需要时再迁移 |

`src/croco/models/curope/curope.cpython-*.so`、`build/`、`__pycache__/` 属于编译产物，不迁移。新服务器上重新编译即可。

## 3. 新服务器推荐目录布局

保持下面路径最省事，因为当前配置和报告脚本里有这些默认路径：

```text
/workspace/code/Movie3R
/workspace/data/RICH/RICH_4Human3R/Training
/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1
/workspace/data/Avatarrex
```

如果新服务器路径不同，需要同步修改配置、脚本参数或创建软链接。

## 4. 环境重建

推荐 Python 3.10，PyTorch 2.4.0，CUDA 12.4 wheel。基础流程：

```bash
cd /workspace/code
git clone <repo-url> Movie3R
cd Movie3R
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements_Movie3R.txt
```

之后复制手动资源：

```text
src/human3r_896L.pth
src/models/
src/checkpoints/                # 可选
data/                           # 可选 demo 视频
/workspace/data/RICH/RICH_4Human3R/Training/
/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1/
/workspace/data/Avatarrex/      # 训练 AvatarReX 时需要
```

编译 curope：

```bash
cd /workspace/code/Movie3R/src/croco/models/curope
python setup.py build_ext --inplace
```

激活环境：

```bash
cd /workspace/code/Movie3R
source env.sh
```

## 5. 迁移后快速检查

```bash
cd /workspace/code/Movie3R
git status -sb
test -f src/human3r_896L.pth
test -d src/models/smpl
test -d src/models/smplx
test -d /workspace/data/RICH/RICH_4Human3R/Training
test -d /workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

确认历史 AnchorToken 报告已经随 git 同步：

```bash
git ls-files docs/movie3r/archive_v2_v6/anchor_token_report_v1/01_aabb_step1
```

如果该命令没有输出，说明旧服务器上还没有把历史报告文件 commit/push。

## 6. 当前 Movie3R 上下文

迁移后优先阅读：

```text
docs/movie3r/README.md
docs/movie3r/current_research_context.md
docs/movie3r/v8/README.md
docs/movie3r/archive_v7/README.md
docs/movie3r/archive_v2_v6/README.md
```

当前项目已经从 V2-V7 的 ShotToken / background AnchorToken / 后处理式 correction 方向切换到 V8 调研准备阶段。历史 AnchorToken 上下文保存在：

```text
docs/movie3r/archive_v2_v6/ANCHOR_TOKEN_V6_CONTEXT.md
docs/movie3r/archive_v2_v6/anchor_token_report_v1/README.md
```
