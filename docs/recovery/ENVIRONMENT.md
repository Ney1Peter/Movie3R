# Environment reconstruction

## Frozen server

| Item | Value |
|---|---|
| OS | Ubuntu 20.04.2 LTS |
| Python | 3.10.19 |
| PyTorch | 2.4.0+cu124 |
| torchvision | 0.19.0 |
| CUDA runtime in PyTorch | 12.4 |
| cuDNN | 9.1.0 |
| NVIDIA driver | 550.127.08 |
| GPU | 8 x NVIDIA L20, 46,068 MiB each |

The numerical protocols do not require eight GPUs for inference; the count is
recorded so that training and sharded wall-clock measurements can be
interpreted correctly.

## Clean installation

```bash
git clone https://github.com/Ney1Peter/Movie3R.git
cd Movie3R
git checkout shot3r-recovery-20260916

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements_Movie3R.txt
```

Compile the CUDA RoPE extension on the new machine:

```bash
cd src/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..
```

Then restore the external weights and licensed body-model assets listed in
`WEIGHTS.md`. Do not copy `.venv`, `__pycache__`, compiled `.so` files, or pip
caches from the compromised server.

## Initial verification

```bash
python scripts/verify_shot3r_recovery.py
python -m pytest tests experiments/registration_baselines/tests
```

If package resolution changes in the future, use the pinned versions in
`requirements_Movie3R.txt` first and record any necessary compatibility change
in a new recovery note rather than silently editing the frozen environment.

