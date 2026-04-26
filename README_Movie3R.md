# Movie3R 环境配置

## 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10 |
| 显卡 | ≥ 46GB 显存（如 H800 80GB、L20 48GB） |
| CUDA | PyTorch 2.4.0 使用 cu124 wheel，自带 CUDA 12.4 runtime |

## 1. 创建虚拟环境

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 创建 Python 3.10 环境
uv venv .venv --python 3.10
source .venv/bin/activate
```

## 2. 安装 PyTorch

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# 验证
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')"
```

## 3. 安装依赖

```bash
pip install -r requirements_Movie3R.txt
```

## 4. 编译 RoPE（必须）

```bash
cd src/croco/models/curope
python setup.py build
cd ../../../..

# 验证
python -c "from croco.models.curope import curope2d; print('curope OK')"
```

> 编译需要系统有 `nvcc`。H800 服务器通常自带 nvcc 11.8。

## 5. 启动训练

```bash
cd src

# 4卡训练示例
./train.sh 4 30 2   # 4卡，每卡 batch_size=2，等效 batch=8
```

训练脚本会自动选择空闲 GPU。参数：卡数、epochs、batch_size。

## 6. 验证环境

```bash
python -c "import torch; print(f'torch {torch.__version__}')"
python -c "from croco.models.curope import curope2d; print('curope OK')"
python -c "import smplx; print('smplx OK')"
```

## 注意事项

1. **Dinov2 backbone**：训练时设置 `TORCH_HOME=$HOME/.cache/torch` 使用离线缓存
2. **num_workers=0**：避免 /dev/shm 共享内存不足问题
3. **SMPL 坐标系**：avatarrex.py 已正确处理 mocap→相机坐标变换
