# Movie3R 新服务器迁移说明

更新时间：2026-05-14

## 1. Git 同步范围

本仓库应该通过 git 同步代码、配置、文档、脚本，以及汇报用的 Step1 可视化证据：

```text
output/anchor_token_report_v1/README.md
output/anchor_token_report_v1/01_aabb_step1/**
```

当前 `.gitignore` 已经保留原始 `output/` 忽略规则作为注释，并改为只放行上面这些报告文件。`02_correction_proxy/` 到 `05_topk_quality_gate/` 仍然保持忽略，不随 git 同步。

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

确认 Step1 报告已经随 git 同步：

```bash
git ls-files output/anchor_token_report_v1/01_aabb_step1
```

如果该命令没有输出，说明旧服务器上还没有把 Step1 报告文件 commit/push。

## 6. 当前 AnchorToken 相关上下文

迁移后优先阅读：

```text
ANCHOR_TOKEN_V6_CONTEXT.md
output/anchor_token_report_v1/README.md
docs/movie3r/README.md
docs/movie3r/training.md
docs/movie3r/model.md
```

当前实验结论是：已经证明 AnchorToken 在接入主模型前具备有效的 boundary correction evidence；下一步应该把 offline anchor cache 接入 dataset/loader，先做 pose/camera path 的受控小模型实验。仍然不要改 encoder，也不要把 anchor token 插入完整 decoder token sequence。
