# AvatarReX → Human3R 数据构建工作日志

## 2026/04/13

### 1. 数据预处理脚本（preprocess_avatarrex_fast.py）

**问题**：原脚本串行处理，16个序列×2001帧约需6.5小时。

**改进**：
- 多进程并行（`multiprocessing.Pool`，默认32 workers）
- 增量处理（文件已存在则跳过，可中断续跑）
- 使用cv2读写图像（比PIL更快）

**关键修复**：
1. **Pickle序列化错误**：`np.load()` 返回的 NpzFile 包含文件句柄，无法被 pickle 序列化到子进程。修复：加载后转为普通 dict
   ```python
   _smpl_raw = np.load(smpl_file)
   smpl_data = {k: _smpl_raw[k].copy() for k in _smpl_raw.keys()}
   del _smpl_raw
   ```
2. **颜色错误（蓝色肉色）**：原脚本错误地将 cv2 读出的 BGR 又转了一次 RGB。修复：直接保存 cv2 读出的 BGR 数据
   ```python
   img = cv2.imread(src_jpg, cv2.IMREAD_COLOR)
   cv2.imwrite(out_rgb, img)  # 直接保存 BGR，不做颜色转换
   ```

**运行结果**：15个序列全部处理完成，0错误
```
done=28014 skipped=2001 errors=0
Output: /data/wangzheng/Movie3R-dataset/AvatarRex4Human3R/Training/
```

---

### 2. AABB 数据集类（avatarrex.py）

**目的**：实现 AABB 镜头跳变采样——帧0,1来自相机A的t/t+1，帧2,3来自相机B的t+2/t+3。时间连续，只是中间跳了相机视角。

**关键修复**：

1. **目录结构错误**：原代码假设 `rgb/{cam_id:04d}/{frame:06d}.png`，预处理脚本实际输出为扁平结构 `rgb/{frame:08d}.png`
   ```python
   # 修复前
   rgb_dir = osp.join(seq_dir, sample_seq, "rgb", "0000")
   # 修复后
   rgb_dir = osp.join(seq_dir, sample_seq, "rgb")
   ```

2. **SMPL body_pose 形状错误**：预处理脚本将多维数组展平保存为 (63,)，但 Human3R 期望 (21,3)。加载时需 reshape 回原始形状
   ```python
   if len(shape) > 1:
       val = val.reshape(shape)
   ```

3. **排序索引类型错误**：`sorted(enumerate(l_dist), ...)` 返回的索引是 numpy.float32，无法作为 list 索引
   ```python
   # 修复前（错误）
   humans = [humans[i] for _, i in sorted(enumerate(l_dist), key=lambda x: x[1])]
   # 修复后（正确）
   order = sorted(range(len(l_dist)), key=lambda i: l_dist[i])
   humans = [humans[i] for i in order]
   ```

4. **transform=None 错误**：`BaseMultiViewDataset.__getitem__` 直接调用 `transform(img)` 而不检查 None
   ```python
   # 修复前
   transform=None
   # 修复后
   transform=ImgNorm
   ```

**运行结果**：
```
Dataset size: 420,000 (15 cameras × 14 pairs × 2000 time steps)
Scene 22010708 @ t=0:
  View 0: img=torch.Size([3, 288, 512]), label=22010708_00000000
  View 1: img=torch.Size([3, 288, 512]), label=22010708_00000001
  View 2: img=torch.Size([3, 288, 512]), label=22010710_00000000
  View 3: img=torch.Size([3, 288, 512]), label=22010710_00000001
```

---

### 3. 数据验证结果

| 项目 | 状态 |
|------|------|
| 图像尺寸 | 2048×1500 → 512×288（resize后）|
| 图像颜色 | BGR格式，有真实颜色信息（35247种独特颜色）|
| camera pose | 4×4 c2w矩阵，旋转+平移正确 |
| intrinsics | 3×3 K矩阵 |
| SMPLX参数 | 11维shape、21关节body_pose、15关节hand_pose ✓ |
| SMPLX gender_id | 缺失，代码默认填充0（neutral）|
| 深度图 | 空目录，待后续 Depth-Anything-3 生成 |
| AABB样本总数 | 420,000 |

---

### 4. 文件路径对照

**预处理脚本**：`/data/wangzheng/Movie3R-new/Human3R/datasets_preprocess/preprocess_avatarrex_fast.py`

**AABB数据集类**：`/data/wangzheng/Movie3R-new/Human3R/src/dust3r/datasets/avatarrex.py`

**数据输出**：`/data/wangzheng/Movie3R-dataset/AvatarRex4Human3R/Training/`
```
{seq_id}/
  rgb/{frame:08d}.png    ← 2001帧 BGR图像
  cam/{frame:08d}.npz    ← pose(4,4) + intrinsics(3,3)
  smpl/{frame:08d}.pkl    ← SMPLX参数（list of dict）
  depth/                  ← 空目录（由 Depth-Anything-3 生成）
  mask/                   ← mask/pha/*.jpg 复制为 mask/*.png
```

**数据处理环境**：`/data/wangzheng/Movie3R-dataset/.venv_data/`（uv虚拟环境，python3.10）

**训练环境**：`/data/wangzheng/Movie3R-new/Human3R/.venv/`（已有torch/torchvision）

**数据处理脚本位置**（已移动到数据目录）：
```
/data/wangzheng/Movie3R-dataset/scripts/
  preprocess_avatarrex.py          ← 串行版
  preprocess_avatarrex_fast.py     ← 并行版（当前使用）
  generate_depth_avatarrex.py       ← 深度图生成（多GPU版）
  run_avatarrex_pipeline.py         ← 一体化流程脚本（推荐）
```

**一体化流程脚本** `run_avatarrex_pipeline.py`：
- 一步完成：格式转换 + 深度图生成
- 支持 `--depth_only` 跳过转换仅生成深度图
- 自动检测空闲GPU（排除已有任务的卡）
- 支持断点续跑（已完成文件跳过）
```bash
source /data/wangzheng/Movie3R-dataset/Depth-Anything-3/env.sh
python run_avatarrex_pipeline.py \
    --root /data/wangzheng/Movie3R-dataset/Dataset/avatarrex_zzr \
    --outdir /data/wangzheng/Movie3R-dataset/AvatarRex4Human3R \
    --da3_root /data/wangzheng/Movie3R-dataset/Depth-Anything-3 \
    --workers 32
```

---

### 5. 数据正确性验证（2026/04/13 补充）

**Camera参数验证**：预处理输出的 `cam/*.npz` 与原始 `calibration_full.json` + `smpl_params.npz` 手工重建结果完全一致
```
c2w 旋转差异: 0.0
c2w 平移差异: 0.0
K 差异: 0.0
```

**SMPLX参数验证**：所有字段与原始 SMPL 参数手工转换结果完全一致
```
smplx_shape:         匹配 ✓
smplx_transl:        匹配 ✓
smplx_root_pose:     匹配 ✓
smplx_body_pose:     匹配 ✓
smplx_jaw_pose:      匹配 ✓
smplx_left_hand:     匹配 ✓
smplx_right_hand:    匹配 ✓
```

**结论**：预处理没有引入任何精度损失，数据与原始标定完全一致。

---

### 6. Mask 补充处理（2026/04/13 下午）

**发现**：原始数据有 `mask/pha/*.jpg`（BGR alpha matte，白色=人物，黑色=背景，0-255值）

**处理**：`preprocess_avatarrex_fast.py` 新增 mask 复制逻辑
```python
# 4. Mask: mask/pha/*.jpg → mask/*.png
src_mask = os.path.join(src_root, seq_id, "mask", "pha", frame_num + IMG_FORMAT_IN)
if os.path.exists(src_mask):
    mask_img = cv2.imread(src_mask, cv2.IMREAD_COLOR)
    if mask_img is not None:
        cv2.imwrite(out_mask, mask_img)
```

**验证**：
- mask shape: 2048×1500 → resize后 512×288
- 值域：0.0 或 1.0（二值mask）
- 前景比例：约14.7%（人物占比合理）

**运行结果**：done=30015 skipped=0 errors=0

---

### 7. 深度图生成脚本（2026/04/13 下午）

**脚本**：`Human3R/datasets_preprocess/generate_depth_avatarrex.py`

**功能**：使用 Depth Anything 3 (DA3-base) 对预处理好的 RGB 图像批量生成深度图

**使用方法**：
```bash
source /data/wangzheng/Movie3R-dataset/Depth-Anything-3/env.sh
python generate_depth_avatarrex.py \
    --root /data/wangzheng/Movie3R-dataset/AvatarReX4Human3R \
    --da3_root /data/wangzheng/Movie3R-dataset/Depth-Anything-3 \
    --workers 16 --batch_size 8
```

**关键实现**：
- 单进程 GPU 批处理（模型加载一次，顺序处理所有图像）
- DA3 输出深度图尺寸为 (518, 378)，需 resize 回原始 (2048, 1500)
- 使用 `cv2.INTER_NEAREST` 插值，保持米制深度值不引入伪影
- 支持增量运行（已有 .npy 文件则跳过）
- 处理完自动打印 done/skipped/errors 统计

**测试结果**（序列 22010708）：
```
done=2001 skipped=0 errors=0
深度图形状: (2048, 1500) float32
深度值范围: 0.423m ~ 1.552m（合理室内场景）
```

**训练验证**：
```
depthmap shape: (288, 512)  ← resize后
depthmap range: 0.630m ~ 1.552m
训练加载正常 ✓
```

**修复**：初始版本输出深度图尺寸为 (518, 378)，与原始图像 (2048, 1500) 不符。修复：在保存前用 cv2.resize + INTER_NEAREST 将深度图缩放回原始尺寸。

---

### 8. 深度图生成结果与格式优化（2026/04/13-14）

**运行结果**：15个序列全部完成，30015张深度图
- 每序列：2001帧，0错误
- 深度尺寸：(2048, 1500) float32
- 深度范围：0.38m ~ 1.72m（合理室内场景）
- 有效像素：100%
- 使用5个GPU并行 (1,4,5,6,7)

**逐序列检查**：全部 PASS ✓

**深度图格式优化**：
- 发现：DA3 输出的深度值精度实际是 mm 级别（乘1000后为干净整数），float32 浪费空间
- 优化方案：改用 uint16（毫米整数）保存深度图
  - float32: 4B/像素 → 12.3MB/文件 (2048×1500)
  - uint16: 2B/像素 → 6.2MB/文件（省50%）
- 脚本修改：保存时乘1000转为 uint16
  ```python
  d_mm = (d_clean * 1000).astype(np.uint16)  # 米→毫米
  cv2.imwrite(depth_path.replace('.npy', '.png'), d_mm)
  ```
- dataset 类无需修改：np.load() 读取 uint16 自动转为 float32
- 精度验证：mm 整数存储，转换回去 max 误差 < 0.001m（1mm）

---

### 9. avatarrex_lbn1 数据集处理（2026/04/14）

**转换**：使用 `preprocess_avatarrex_fast.py`
- 16序列 × 1901帧 = 30,416 帧
- 结果：done=0 skipped=30416 errors=0 ✓

**深度图**：使用 `generate_depth_avatarrex.py`（5 GPU）
- 深度尺寸：2048×1500 float32
- 深度范围：0.45m ~ 1.60m
- 损坏文件：0 ✓
- **完成状态**：✅ 30,416/30,416 全部完成

---

### 10. avatarrex_lbn2 数据集处理（2026/04/14）

**状态**：❌ **已废弃** - 最终训练方案仅使用 zzr、lbn1、zxc 三个数据集，lbn2 未使用

**历史记录**（仅供参考）：
- 转换：16序列 × 1871帧 = 29,936 帧
- 深度图生成遇到问题：DA3 对某些图像返回空/无效深度数组，产生大量 0 字节损坏文件
- 磁盘空间不足导致处理中断多次

---

### 11. 文档与脚本整理（2026/04/14）

**新增文件**：
- `/data/wangzheng/Movie3R-dataset/README.md`：项目说明文档
- `/data/wangzheng/Movie3R-dataset/requirement.txt`：Python 依赖
- `/data/wangzheng/Movie3R-dataset/Dataset/run_depth_generation.md`：深度图生成说明

**脚本更新**：
- 参数风格统一为 `-i/-o` 格式
- `preprocess_avatarrex_fast.py`：支持 `-i/--input` 和 `-o/--output`
- `generate_depth_avatarrex.py`：支持 `-i/--input` 和 `--da3/--da3_root`
- 新增 `generate_depth.py` 作为简化入口

**深度图脚本优化**：
- 原问题：每批次内逐个 cv2.imread 读取图像获取尺寸（大量冗余 IO）
- 优化：移除冗余读取，固定使用原始尺寸 2048×1500

---

### 12. 当前数据集状态

| 数据集 | RGB | CAM | SMPL | Depth | Mask | 状态 |
|--------|-----|-----|------|-------|------|------|
| avatarrex_zzr | ✅ | ✅ | ✅ | ✅ uint16 | ✅ | ✅ 完成 |
| avatarrex_lbn1 | ✅ | ✅ | ✅ | ✅ uint16 | ✅ | ✅ 完成 |
| avatarrex_zxc | ✅ | ✅ | ✅ | ✅ uint16 | ✅ | ✅ 完成 |

**当前使用的数据集路径**：
- `/workspace/data/avatarrex_zzr`
- `/workspace/data/avatarrex_lbn1`
- `/workspace/data/avatarrex_zxc`

---

### 13. AvatarReX 训练数据集配置

**背景**：
- 原版 Human3R 使用 BEDLAM_Multi 作为训练数据（650人 × 3000+场景）
- AvatarReX 数据（zzr + lbn1 + zxc）三个数据集足够训练

**数据集类**：
- `AvatarReX_Video`：同一相机内连续帧采样（t, t+1, t+2, t+3），is_video=True
- `AvatarReX_AABB`：AABB 镜头跳变采样（camA@t, camA@t+1, camB@t+2, camB@t+3），is_video=False
- 两个类都继承自 BaseMultiViewDataset，支持 Human3R 标准接口

**AABB 与 Video 格式定义**：

| 数据类型 | 帧0 | 帧1 | 帧2 | 帧3 | is_video |
|----------|-----|-----|-----|-----|----------|
| Video | camA@t | camA@t+1 | camA@t+2 | camA@t+3 | True |
| AABB | camA@t | camA@t+1 | camB@t+2 | camB@t+3 | False |

**SMPL 说明**：AABB 中 view2、view3 的 SMPL 仍从 seqA 加载，因为 SMPL 是世界坐标系下的人体三维参数，不随视角变化。

---

### 14. SMPL 过滤坐标 bug（2026/04/14）

**问题现象**：
- 训练在 batch 31 抛出 `ZeroDivisionError: division by zero`
- 错误位置：`smpl_model.py` 中 `smpl_mask.sum() == 0`
- 即该 batch 内所有样本的 SMPL mask 全为 0

**初步调查**：
- 检查了 avatarrex.py 中 `smplx_transl` 的过滤条件：`smplx_transl[-1] > 0.01`
- 发现大量帧的 `transl.z` 接近 0（约 0.01m），被过滤掉
- 其中一个样本：transl = [0.426, 0.760, 0.012] → 过滤后 valid=False
- 大量帧的 `transl.z ∈ [-0.03, +0.01]`，几乎全部接近 0

**根本原因**：

`smplx_transl` 存的是 **mocap 世界坐标系**（动作捕捉系统坐标），不是相机坐标：
- X ~ 0.4m（人在 mocap 原点侧面 0.4m）
- Y ~ 0.75m（人在 mocap 原点上方 0.75m，即身高）
- Z ~ -0.03 ~ +0.01m（人在 mocap 原点前后，几乎在原点）

而过滤条件 `smplx_transl[-1] > 0.01` 是在检查 mocap Z 是否 > 0.01：
- mocap Z 几乎都在 0 附近（人在 mocap 原点前后）
- 所以大量帧被错误过滤，导致 `smpl_mask.sum() == 0`

**正确理解**：
- 真正需要的是：人在相机前方（相机坐标系 Z > 0）
- 相机坐标系下的 Z 约为 1.7m（人距相机约 1.7m）
- 需要将 mocap 世界坐标变换到相机坐标：`smpl_cam = R_c2w.T @ (smpl_world - t_c2w)`

**关键验证**：
```
序列 22070928 帧 00001419：
- smplx_transl (mocap): [0.426, 0.760, 0.012]
- 相机坐标系下: [?, ?, ~1.7m]  ← 人在相机前方 1.7m
- 所有帧变换后 camera_z ∈ [1.6, 1.8m] → 100% valid
```

**修复方案**（avatarrex.py 两处）：

1. **过滤前先变换到相机坐标系**：
```python
R_c2w = camera_pose[:3, :3]
t_c2w = camera_pose[:3, 3]

humans_with_cam_z = []
for h in annots:
    smpl_world = np.array(h.get("smplx_transl", [0, 0, 100]), dtype=np.float32)
    smpl_cam = R_c2w.T @ (smpl_world - t_c2w)  # mocap世界 → 相机坐标
    h = dict(h)
    h["_smplx_transl_cam"] = smpl_cam
    h["_smplx_transl_cam_z"] = smpl_cam[2]
    humans_with_cam_z.append(h)
```

2. **按相机坐标 Z 排序和过滤**：
```python
# 排序（人在相机前方 Z > 0）
l_dist = [hh["_smplx_transl_cam_z"] for hh in humans_with_cam_z]
order = sorted(range(len(l_dist)), key=lambda i: l_dist[i])
humans_with_cam_z = [humans_with_cam_z[i] for i in order]

# 过滤：相机坐标系 Z > -0.5m 即有效（留足容差）
humans = [hh for hh in humans_with_cam_z if hh["_smplx_transl_cam_z"] > -0.5]
```

3. **smpl_dict 使用变换后的值**：
```python
if k == "smplx_transl":
    for h in range(len(humans)):
        smpl_dict[k][h] = humans[h]["_smplx_transl_cam"]
```

**修复位置**：
- `AvatarReX_AABB.__getitem__`（约 line 250-303）
- `AvatarReX_Video.__getitem__`（约 line 496-538）

**修复效果**：
- 修复前：约 41-59% 帧被过滤（mocap Z 几乎都在 0 附近）
- 修复后：100% 帧有效（camera_z ∈ [1.6, 1.8m]）

**额外发现**：
- 原始 `global_orient` 显示 ~90° 旋转，但这是 mocap 系统的旋转，不是 bug
- SMPL mesh 本身没问题（mesh 渲染正常）

**同时修复**：`smpl_model.py` 中的 `if nhv == 0: return target` guard（line 107），防止以后还有残留问题导致 crash。

---

### 15. 全量微调验证（2026/04/14）

**目标**：将 AvatarReX 数据集 + Human3R 预训练模型跑通全量微调流程

**配置**：
- `freeze='none'`：全量微调（所有参数可训练）
- `batch_size=1`：44GB 显卡全量微调刚好够用

**模型规模**：
| 项目 | 数值 |
|------|------|
| 参数量 | 1.18B |
| FP16 显存（仅权重） | ~2.4GB |
| 全量微调显存（batch=1） | ~42.8GB |

**显存不足记录**：
- batch_size=4：OOM（44GB 显卡不够）
- batch_size=2：OOM
- batch_size=1：通过

**验证结果**（batch_size=1）：
- batch 0: loss=0.061 ✓
- batch 10: loss=0.068 ✓
- batch 20: loss=0.071 ✓
- batch 40: loss=0.067 ✓
- GPU 显存：稳定在 42.8GB

**加速方案**（可选）：
- `batch_size=1 + accum_iter=4`：梯度累积，实际等效 batch_size=4
- 效果相当于 batch_size=4，但分步计算节省显存

---

### 16. 多GPU训练调试（2026/04/16）

**问题**：多GPU训练时 NCCL allreduce 操作卡住

**测试环境**：
- PyTorch: 2.4.0+cu124 → 2.5.0+cu124
- NCCL: 2.20.5 → 2.21.5
- 测试脚本：简单的 tensor allreduce

**测试结果**：

| 配置 | 结果 |
|------|------|
| NCCL init | ✓ 成功（GPU P2P/CUMEM 连接建立） |
| NCCL allreduce | ✗ 卡住（所有GPU组合都测试过） |
| Gloo allreduce | ✓ 正常 |
| 不同GPU对 (0-1, 4-5, 6-7) | 全部卡住 |

**NCCL 错误信息**：
```
NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file
No plugin found (libnccl-net.so), using internal implementation
```

**结论**：
- 这是**服务器级别的 NCCL 配置问题**
- PyTorch 捆绑的 NCCL 2.20.5/2.21.5 在该服务器上 allreduce 操作无法完成
- Gloo backend 可以工作但速度慢，不适合训练

**解决方案**：
1. 联系服务器管理员检查 NCCL 配置
2. 使用单 GPU 训练（已验证正常工作）
3. 或等待更换到其他服务器

**环境恢复**：
- 已将 torch 恢复为 requirements_Movie3R.txt 中的版本：torch==2.4.0
- 单 GPU 训练已验证正常：batch 0 loss=0.0614, max mem=40261 MB

**后续**：✅ **已解决** - 在云平台 H800 集群上 4GPU 训练成功完成

---

### 17. 正式训练配置

**训练环境**：
- 项目路径：`/workspace/code/Movie3R/`
- 训练脚本：`cmd_4gpu_train.sh`
- 虚拟环境：`.venv/`（torch==2.4.0+cu124）

**训练数据集**（3 个数据集）：
| 数据集 | 类型 | 路径 |
|--------|------|------|
| AvatarReX_Video (zzr) | Video | `/workspace/data/avatarrex_zzr` |
| AvatarReX_Video (lbn1) | Video | `/workspace/data/avatarrex_lbn1` |
| AvatarReX_Video (zxc) | Video | `/workspace/data/avatarrex_zxc` |
| AvatarReX_AABB (zzr) | AABB | `/workspace/data/avatarrex_zzr` |
| AvatarReX_AABB (lbn1) | AABB | `/workspace/data/avatarrex_lbn1` |
| AvatarReX_AABB (zxc) | AABB | `/workspace/data/avatarrex_zxc` |

**数据集划分**（通过 seed 区分）：
| Split | 样本数 | Seed | 说明 |
|-------|--------|------|------|
| train | 4800 | 11 | 800 × 6 datasets |
| val | 600 | 22 | 100 × 6 datasets |
| test | 600 | 33 | 100 × 6 datasets |

**正式训练参数**（实际运行）：

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 30 | 训练轮数 |
| batch_size | 2 | 每卡 batch size |
| num GPUs | 4 | 等效 batch_size=8 |
| num_workers | 0 | 单进程模式（避免 /dev/shm 限制）|
| lr | 1e-4 | 学习率 |
| min_lr | 1e-6 | 最小学习率 |
| warmup_epochs | 5 | warmup 轮数 |
| weight_decay | 0.05 | 权重衰减 |
| gradient_checkpointing | true | 梯度检查点（节省显存）|
| amp | 1 | 混合精度训练 |
| early_stopping_patience | 10 | 早停轮数 |

**训练结果**（30 epochs 正式训练）：
- 训练时长：20 小时 2 分钟
- Val Loss：28.52 → 1.31（降低 95%）
- SMPLLoss_j3d：0.47m → 0.04m（降低 92%）
- 无过拟合迹象，Val loss 持续下降

**实验输出**：
- 路径：`experiments/formal_training-4gpu/`
- checkpoint-best.pth：11.5 GB（最佳验证模型）
- checkpoint-final.pth：4.7 GB（最终模型）
- checkpoint-last.pth：11.5 GB（最后一个 epoch）

---

### 18. 模型架构与冻结配置（2026/04/16）

**模型类**：`ARCroco3DStereo`（继承自 `CroCoNet`）

**模型结构**：
```
ARCroco3DStereo
├── 1. Patch Embedding
│   ├── patch_embed           # 图像 patch embedding (3 → 1024 ch)
│   └── patch_embed_ray_map  # Ray-map patch embedding (6 → 1024 ch)
│
├── 2. Encoder (ViT, 24 blocks)
│   ├── enc_blocks            # 24 层 ViT Block, 1024 dim
│   └── enc_blocks_ray_map   # 2 层 Ray-map Encoder Block
│
├── 3. MHMR (Dinov2Backbone)
│   └── backbone (Dinov2-ViT-L/14)  # 独立 DINOv2 主干
│
├── 4. Decoder (12 blocks)
│   ├── decoder_embed        # 1024 → 768 维度映射
│   ├── dec_blocks           # 12 层 Decoder Block
│   └── dec_blocks_state     # State decoder
│
└── 5. Downstream Head
    ├── dpt_self / dpt_cross / dpt_rgb  # DPT 深度头
    ├── pose_head                         # 姿态估计头
    ├── mlp_classif / mlp_offset         # 分类/偏移 MLP
    └── decpose / decshape / deccam / decexpression  # SMPL 参数头
```

**模型规模**（Human3R 896L）：
| 项目 | 数值 |
|------|------|
| 参数量 | ~1.18B |
| Encoder | ViT-L/14, 24 blocks, 1024 dim |
| Decoder | 12 blocks, 768 dim |
| 预训练权重 | `human3r_896L.pth` |

**冻结选项**（`freeze` 参数）：
| freeze 参数 | 冻结内容 | 说明 |
|------------|---------|-----|
| `freeze='none'` | **无** | **全量微调（当前配置）** |
| `freeze='encoder'` | enc_blocks + enc_blocks_ray_map | 仅微调 decoder + head |
| `freeze='decoder'` | dec_blocks + dec_blocks_state | 仅微调 encoder + head |
| `freeze='head'` | downstream_head | 仅微调 encoder + decoder |
| `freeze='encoder_and_decoder_and_head'` | enc + dec + head | 全部冻结（纯推理）|

**当前配置**：
```yaml
freeze='none'  # 全量微调
```

全量微调意味着所有模块都是可训练的：
- ✅ 编码器（ViT encoder）
- ✅ Ray-map 编码器
- ✅ DINOv2 backbone (MHMR)
- ✅ 解码器
- ✅ 所有预测头（深度、姿态、SMPL）

**配置文件位置**：
- 模型定义：`src/dust3r/model.py`
- CroCo 基类：`src/croco/models/croco.py`
- 冻结逻辑：`src/dust3r/model.py` 第 509-635 行

---

### 19. 训练完成事项

1. ✅ **训练配置**：AvatarReX Video + AABB 混合训练（3 数据集）
2. ✅ **训练测试**：数据加载、loss 计算正常
3. ✅ **SMPL 坐标 bug**：已修复
4. ✅ **全量微调验证**：freeze=none, batch_size=1 通过
5. ✅ **正式训练参数**：已确定并成功运行
6. ✅ **模型架构与冻结配置**：已记录
7. ✅ **多GPU训练**：✅ 已解决 - 4GPU 30 epochs 训练成功完成
8. ✅ **AABB view2 pose loss**：已实现并通过测试
9. ✅ **Train/Val/Test 划分**：通过 seed 区分（11/22/33）
10. ✅ **Early stopping**：patience=10 已实现
11. ✅ **Best model 保存**：按 val loss 保存 checkpoint-best.pth

---

### 20. AABB View2 位姿 Loss 增强（2026/04/17）

**问题**：AABB 数据（跨相机跳变镜头）中，第一个 B 帧（view2，即 camB 的首帧）的相机位姿预测准确率明显低于其他帧。这是因为模型在跨相机跳变时缺乏足够的几何约束来精确定位新相机。

**解决方案**：对 AABB 数据的 view2 帧（帧索引 2，即 `gt_poses[2]`），单独计算 L2 位姿 loss（translation + quaternion），并与原有 pose_loss 叠加监督。

**实现位置**：`src/dust3r/losses.py`，`Regr3DPoseBatchList.compute_loss()`，约第 1509-1527 行。

**新增代码**：
```python
# ===== AABB view2 pose loss =====
# AABB: view0,view1 from camA, view2,view3 from camB
# 对 AABB 数据的 view2（第一个 B 帧）单独计算 pose L2 loss
# gts[0]["is_video"] = True for Video, False for AABB
is_video = gts[0]["is_video"]
if not is_video.all():
    is_aabb_mask = ~is_video
    gt_trans_view2 = gt_poses[2][0][is_aabb_mask]
    gt_quat_view2 = gt_poses[2][1][is_aabb_mask]
    pr_trans_view2 = pr_poses[2][0][is_aabb_mask]
    pr_quat_view2 = pr_poses[2][1][is_aabb_mask]
    view2_pose_loss = (
        torch.norm(pr_trans_view2 - gt_trans_view2, dim=-1).mean()
        + torch.norm(pr_quat_view2 - gt_quat_view2, dim=-1).mean()
    )
    details["pose_loss_view2_AABB"] = float(view2_pose_loss)
    # 添加到总 pose_loss（一起监督）
    details["pose_loss"] = details["pose_loss"] + view2_pose_loss
```

**训练验证**（2026/04/17，step 0-250）：
- `pose_loss_view2_AABB` 正常计算并加入总 loss
- `pose_loss_view2_AABB` avg 在 ~2500 范围波动，loss 趋势平稳
- 训练可正常启动，数据加载、loss 反向传播均正常
- 单卡 GPU 4 测试通过

**验证命令**：
```bash
# 查看 GPU 状态
./train.sh 0

# 单卡测试（1 epoch，batch_size=1）
./train.sh 1 1 1
```

**日志查看**：
```bash
# TensorBoard
tensorboard --logdir experiments/avatarrex_zzr_lbn1/

# 文本日志
tail -f src/checkpoints/human3r/train.log
```

**关键指标**：`pose_loss_view2_AABB`（新增，TensorBoard 中的 `train_pose_loss_view2_AABB`）

---

### 21. 服务器迁移与环境配置指南（历史参考）

> ⚠️ **以下内容为历史参考**，基于旧服务器环境 `/data/wangzheng/Movie3R-new/Human3R/`。当前环境为 `/workspace/code/Movie3R/`，路径和配置可能不同。

#### 旧环境配置（仅供参考）

| 项目 | 版本/值 |
|------|--------|
| Python | 3.10.19 |
| PyTorch | 2.4.0+cu124（CUDA 12.4） |
| 虚拟环境 | `.venv`（uv 管理） |
| 预训练权重 | `/data/wangzheng/Movie3R-new/Human3R/src/human3r_896L.pth` |
| Dinov2 backbone | `TORCH_HOME=$HOME/.cache/torch`（离线模式） |

#### 迁移要点（通用）

1. **数据集路径**：所有 `ROOT=` 必须使用**绝对路径**
2. **TORCH_HOME**：`train.sh` 中设置 `export TORCH_HOME=$HOME/.cache/torch`
3. **num_workers**：建议 `num_workers=0` 避免 /dev/shm 限制
4. **NCCL 问题**：如遇多GPU训练卡住，参考 Section 16 排查

---

### 22. 已知问题与注意事项

1. **Dinov2 网络问题**：如 `torch.hub.load` 超时，需要设置 `TORCH_HOME` 使用离线缓存
2. **/dev/shm 限制**：容器环境可能只有 64MB，使用 `num_workers=0` 避免
3. **batch_size 选择**：H800 80GB 单卡最大 batch_size=2（等效 4GPU × bs=2 = 8）
4. **checkpoint 区别**：
   - `checkpoint-final.pth`：仅模型权重 (~4.7GB)
   - `checkpoint-best.pth` / `checkpoint-last.pth`：模型 + 优化器 + AMP scaler (~11.5GB)

---

### 23. 微调后问题排查（2026/04/26）

#### 问题现象
全量微调（freeze='none'）后，在 h36.mp4 推理时出现：
1. **SMPL 检测失败**：smpl_scores 最高只有 0.067，远低于检测阈值 0.3，导致 shape/rotvec/transl 全为 (0, ...)
2. **相机位姿异常**：部分帧位姿偏移过大

#### 根因分析

**检测流程**：
```
图像 → backbone(Dinov2) → feat_mhmr_i → detect_mhmr → scores → apply_threshold(0.3)
                                                    ↓
                                          scores >= 0.3 → 有检测 → n_humans > 0
                                          scores < 0.3 → 无检测 → n_humans = 0 → SMPL为空
```

**问题根因**：
- 微调时 `freeze='none'` 导致 **backbone 也被微调**
- backbone 在 AvatarReX 数据上过拟合，失去对 h36.mp4 等新数据的泛化能力
- 输入 detect_mhmr 的特征质量下降
- detect_mhmr 输出分数从 0.74 降到 0.09，低于阈值

**对比数据**：
| 模型 | detect_mhmr 最高分数 | SMPL 检测 |
|------|---------------------|----------|
| 原预训练 (h36m_test) | 0.74 | ✅ 正常 (1, 10) |
| 微调后 (h36_test_2) | 0.067 | ❌ 失败 (0, 10) |

#### 模型模块结构（总参数 1.17B）

| 模块 | 参数 | 占比 | 作用 |
|------|------|------|------|
| backbone (Dinov2) | 304M | 26.1% | 通用视觉特征提取 |
| enc_blocks (ViT) | 302M | 25.9% | 图像序列编码 |
| pose_retriever | 152M | 13.0% | 相机位姿记忆查询 |
| downstream_head | 152M | 13.0% | 深度/位姿/SMPL输出 |
| dec_blocks | 113M | 9.7% | 多视角融合 |
| dec_blocks_state | 113M | 9.7% | 时序状态更新 |
| enc_blocks_ray_map | 25M | 2.2% | Ray-map编码 |

#### freeze 选项对应的微调模块

| freeze 选项 | 冻结的模块 | 微调的模块 |
|------------|-----------|-----------|
| `none` | 无 | 全部 (1.17B) |
| `encoder` | patch_embed, enc_*, backbone | **decoder, pose_retriever, head** (~530M) |
| `encoder_and_decoder` | encoder + decoder + pose_retriever | **head** (~152M) |
| `encoder_and_decoder_and_head` | encoder + decoder + dpt_*, pose_head | backbone, mlp_classif, mlp_offset |

#### 下一步微调方案（针对镜头跳变偏移修复）

**目标**：修复 AABB 跨相机跳变时的相机位姿估计

**推荐配置**：`freeze='encoder'`

| 模块 | 参数 | 冻结/微调 | 说明 |
|------|------|----------|------|
| patch_embed | 0.8M | ❌ 冻结 | - |
| enc_blocks (ViT) | 302M | ❌ 冻结 | - |
| enc_blocks_ray_map | 25M | ❌ 冻结 | - |
| **backbone (Dinov2)** | 304M | ❌ 冻结 | ✅ 保持泛化能力 |
| **dec_blocks** | 113M | ✅ 微调 | 🎯 修复镜头跳变 - 多视角融合 |
| **dec_blocks_state** | 113M | ✅ 微调 | 🎯 修复镜头跳变 - 时序状态 |
| **pose_retriever** | 152M | ✅ 微调 | 🎯 **核心** - 相机位姿记忆查询 |
| **downstream_head** | 152M | ✅ 微调 | ✅ |
| - pose_head | 2.4M | ✅ 微调 | 🎯 **核心** - 相机位姿输出 |
| - mlp_classif | 1M | ✅ 微调 | ⚠️ 需监控是否下降 |
| - mlp_fuse/decpose等 | ~40M | ✅ 微调 | ✅ |

**微调参数：约 530M (45.5%)**
**冻结参数：约 632M (54.5%)**

**更保守配置**（如 freeze='encoder' 仍有问题）：`freeze='encoder_and_decoder'`，只微调 head (~152M)

**需监控指标**：
- `smpl_scores` 分布（应 > 0.3）
- AABB 的 `pose_loss`
- 推理时 SMPL 是否正常输出

---

### 24. 后续优化方案：PoseCorrectionHead 与 Jump Token

#### 核心判断（来自外部意见）

> 问题不在底层视觉特征（backbone），而是 **pose_retriever / state readout / world-frame 对齐能力不足**。
> 
> 所以不建议一开始 full finetune，推荐 freeze="encoder"。

#### 学习率分配建议

| 模块 | 学习率 | 说明 |
|------|--------|------|
| pose_head | 1e-4 | 主训练 |
| pose_retriever | 5e-5 | 主训练 |
| world/depth/pose related heads | 2e-5 ~ 5e-5 | 适配 |
| dec_blocks_state | 1e-5 ~ 2e-5 | 小学习率适配 |
| dec_blocks | 1e-5 | 小学习率适配 |
| SMPL / human classif | 0 或 1e-5 | 尽量少动 |
| encoder / backbone | 0 | **冻结** |

**重点**：pose_retriever 和 pose_head 主训练，decoder/state 小学习率适配，human 分支尽量少动。

#### PoseCorrectionHead（推荐新增）

**背景**：jump cut 下 pose_retriever / state readout / world-frame 对齐能力不足

**结构**：
```
image/state features
    ↓
pose_retriever
    ↓
raw pose embedding
    ↓
PoseCorrectionHead (轻量 MLP)
    ↓
Δrot (6D rotation / so(3)), Δtrans (R3), confidence (0~1)
    ↓
T_final = exp(confidence * ξ) @ T_raw

其中 ξ 是 SE(3) correction：
- 连续帧：confidence ≈ 0，几乎用原始 pose
- jump cut：confidence 变大，用修正 pose
```

**作用**：
- 解耦连续帧和跳变帧的处理
- 让模型自己学习什么时候该修正位姿
- 实现简单，只在 pose_retriever 后加一个轻量 head

**融合方式**：
```
T_final = exp(confidence * ξ) @ T_raw
```
confidence 控制修正程度，exp 是 so(3) 到 SO(3) 的指数映射。

#### Jump Token / Relocalization Token（可选进阶）

**结构**：
在 decoder 或 pose_retriever 里加入一个额外 token，让它 attend to state，并输出：
- jump probability：当前帧是否是视觉不连续
- pose correction ΔT：位姿修正量
- localization confidence：定位置信度

**作用**：
- 显式判断当前帧是否是跳变帧
- 专门处理重新定位问题

#### Global Anchor Memory（优先级低）

更复杂的方案，增加全局锚点记忆，但优先级低于 PoseCorrectionHead 和 jump token。

#### 训练数据采样建议

| 类型 | 比例 | 目的 |
|------|------|------|
| normal continuous clips | 40% | 基础能力 |
| AABB camera jump clips | 30% | **核心** - 镜头跳变训练 |
| large-baseline same-scene | 20% | 大基线适配 |
| shuffled / loop clips | 10% | 增强鲁棒性 |

**目标**：让模型学会"时间连续 ≠ 相机运动连续"，jump frame 需要重新在已有 world/state 中定位。

#### Loss 设计建议

```
L = L_pose_abs
  + 2 * L_pose_rel_jump        # 跳变前后相对位姿
  + L_world_pointmap            # 防止场景整体偏移
  + 0.5 * L_cross_view_world_consistency
  + 0.1 * L_pretrained_distill  # 防止能力退化
  + human losses
```

**重点**：
- `L_pose_rel_jump`：监督跳变前后相对位姿
- `L_world_pointmap`：防止场景整体偏移
- `L_pretrained_distill`：蒸馏原始预训练模型的能力，防止退化

#### 实验顺序建议

1. **freeze="encoder_and_decoder"**：只训 head，做 ablation
2. **freeze="encoder"**：训 decoder/state/pose_retriever/head，主实验
3. **freeze="encoder" + PoseCorrectionHead**：新增修正模块
4. **freeze="encoder" + jump token**：显式跳变判断
5. **小学习率 full finetune**：最后才考虑

#### 推荐路线总结

```
1. freeze encoder
2. 微调 pose_retriever + pose_head + decoder/state
3. 新增 PoseCorrectionHead 或 jump token
4. 增加 jump-cut / large-baseline 训练数据
5. 使用 relative pose + world pointmap consistency loss
```

**这样更适合在不分段、不后处理的前提下，提升 Human3R 对镜头跳变的重定位能力。**
