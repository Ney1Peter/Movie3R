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

> ⚠️ **此方案已过时（2026/04/27）**：Section 23-24 的 `freeze='encoder'` 方案存在问题，会解冻 CUT3R decoder，破坏 Human3R 的核心假设。正确方向见 Section 25。

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

---

### 25. Shot-Aware Adaptation 方案（正确方向，2026/04/27）

#### 核心原则

1. **不修改 CUT3R 基模**：CUT3R encoder/decoder 全部冻结
2. **模仿 Human3R 的方式**：在冻结的 CUT3R 基础上，增加轻量可学习模块
3. **不破坏原有推理流程**：新模块作为 residual/correction 添加

#### 问题分析

Human3R 在镜头连续、相机运动平滑的视频中表现较好，因为继承了 CUT3R 的 recurrent persistent state 机制。

但在存在明显镜头切换的视频中，模型表现会明显变差：
- 同一场景，时间连续，但相机视角突然变化
- 问题根源：`S_{t-1}` 编码旧镜头视角的空间上下文，与新镜头 F_t 不兼容
- decoder 交互时会从旧 state 读出不适合当前视角的上下文，导致 camera pose / world pointmap / human mesh 偏移

#### 正确方案：Shot-Aware Token + State Gate + LoRA Heads

**新结构：**
```
I_t -> Frozen Encoder -> F_t

F_t, F_{t-1} -> ShotTokenGenerator -> q_t

[z, F_t, H_t, q_t] + S_tilde_{t-1}
  -> Frozen Decoder
  -> [z'_t, F'_t, H'_t, q'_t] + S_t

**Decoder token 排列顺序：[z, F_t, H_t, q_t]（q_t 在最后一位）
q_out = tokens[:, -1:]（取最后一个 token 作为 q'_t）**

LoRA 使用 q'_t（decoder 输出）作为 condition，不是 q_t（decoder 输入）

F'_t + q'_t -> world LoRA
z'_t + q'_t -> pose LoRA
H'_t + q'_t -> human LoRA
```

**关键：q_t 与 q'_t 不混用**
- **q_t**：进入 decoder 前的 shot token，由 ShotTokenGenerator 生成
- **q'_t**：decoder 输出后的 refined shot token，已融合 image/state/camera/human 上下文
- LoRA heads 必须用 **q'_t**

#### 训练数据格式

| 类型 | frame order | camera pattern | shot_label |
|------|-------------|-----------------|------------|
| Video | [t, t+1, t+2, t+3] | [A, A, A, A] | [0, 0, 0, 0] |
| AABB | [t, t+1, t+2, t+3] | [A, A, B, B] | [0, 0, 1, 0] |

**shot_label[i] 表示 frame i-1 → frame i 是否发生 shot change。**
AABB 中 boundary 是 frame1 → frame2，所以 shot_label[2] = 1。

**q_i 始终用相邻时间帧计算：q_2 = ShotGen(F_2, F_1)，不是 F_2 和 F_0。**

#### 模块设计

**1. ShotTokenGenerator V1（第一版）**

使用 decoder 前的 F_t（维度 dec_dim）作为输入，避免额外投影：

```python
class ShotTokenGenerator(nn.Module):
    """Global Difference Token - 使用 dec_dim 特征"""
    def __init__(self, dec_dim=768):
        # 输入: g_curr, g_prev, diff, sim = 3 * dec_dim + 1 (V1)
        # V2 Patch Matching: 4 * dec_dim + 1 (多 d_match)
        self.shot_mlp = nn.Sequential(
            nn.Linear(dec_dim * 3 + 1, 256),  # V1: 3*dec_dim + 1
            nn.GELU(),
            nn.Linear(256, dec_dim),
        )
        # i=0 没有 previous frame，用可学习的 q_init
        self.q_init = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

    def forward(self, feat_curr, feat_prev, i):
        # feat_curr, feat_prev: [B, N, dec_dim]
        if i == 0:
            return self.q_init.expand(feat_curr.shape[0], -1, -1)
        g_curr = feat_curr.mean(dim=1)      # [B, dec_dim]
        g_prev = feat_prev.mean(dim=1)      # [B, dec_dim]
        diff = g_curr - g_prev              # [B, dec_dim]
        sim = F.cosine_similarity(g_curr, g_prev, dim=-1)  # [B]
        x = torch.cat([g_curr, g_prev, diff, sim.unsqueeze(-1)], dim=-1)  # [B, 3*dec_dim+1]
        q_t = self.shot_mlp(x).unsqueeze(1)  # [B, 1, dec_dim]
        return q_t
```

**V2（后续）：Patch Matching Token**
```python
# V2 输入: g_curr, g_prev, diff, d_match, sim = 4 * dec_dim + 1
def forward_v2(self, feat_curr, feat_prev, i):
    if i == 0:
        return self.q_init.expand(feat_curr.shape[0], -1, -1)
    g_curr = feat_curr.mean(dim=1)
    g_prev = feat_prev.mean(dim=1)
    diff = g_curr - g_prev

    # Patch matching
    A = F.softmax(feat_curr @ feat_prev.transpose(-2,-1) / math.sqrt(feat_curr.shape[-1]), dim=-1)
    F_match = A @ feat_prev
    d_match = (feat_curr - F_match).mean(dim=1)  # V2 多这个
    sim = A.max(dim=-1)[0].mean(dim=-1)

    x = torch.cat([g_curr, g_prev, diff, d_match, sim.unsqueeze(-1)], dim=-1)  # 4*dec_dim+1
    return self.shot_mlp_v2(x).unsqueeze(1)
```

**2. StateGate（第一版：scalar alpha）**

```python
class StateGate(nn.Module):
    """S_tilde = alpha * S_prev + (1 - alpha) * S0"""
    def __init__(self, dec_dim=768):
        self.gate_mlp = nn.Sequential(
            nn.Linear(dec_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),  # 输出 scalar alpha
        )

    def forward(self, q_t):
        # q_t: [B, 1, dec_dim]
        # alpha: [B, 1, 1]
        alpha = torch.sigmoid(self.gate_mlp(q_t))  # [B, 1, 1]
        return alpha
```

**第一帧处理（i=0）：**
```python
if i == 0:
    q_t = q_init          # 可学习参数
    S_tilde = S0         # 直接用初始 state，不做 gate
else:
    q_t = ShotTokenGenerator(F_t, F_{t-1})
    alpha = StateGate(q_t)
    S_tilde = alpha * S_prev + (1 - alpha) * S0
```

**State Gate 计算细节：**
```python
# S_prev: [B, N_state, dec_dim]
# S0: [1, N_state, dec_dim] (可学习初始 state)

S0_expand = S0.expand_as(S_prev)  # [B, N_state, dec_dim]
S_tilde = alpha * S_prev + (1 - alpha) * S0_expand  # [B, N_state, dec_dim]
```

**注意：必须加回 S0，不要只做 alpha * S_prev**

#### 3. LoRA Heads（使用 q'_t 作为 condition）

**3.1 Pose LoRA**

Pose 表达：quaternion (4D) + translation (3D) = 7D

```python
class LoRAPoseHead(nn.Module):
    """Pose residual: T_final = T_base + gamma * delta_T
    输入: z'_t [B,1,C], q'_t [B,1,C], pose_base [B,7] (quat4 + trans3)
    """
    def __init__(self, dec_dim=768, shot_dim=768):
        self.gamma = nn.Parameter(torch.tensor(0.01))
        # delta: 4D quaternion residual + 3D translation residual = 7D
        self.lora = nn.Sequential(
            nn.Linear(dec_dim + shot_dim, 128),
            nn.GELU(),
            nn.Linear(128, 7),  # delta_quat(4) + delta_trans(3)
        )

    def forward(self, z_token, q_out, pose_base):
        # z_token: [B,1,dec_dim], q_out: [B,1,shot_dim], pose_base: [B,7]
        x = torch.cat([z_token, q_out], dim=-1)
        delta = self.lora(x)  # [B, 1, 7]
        delta = delta.squeeze(1)

        # quaternion 加 residual 后必须 normalize
        q_base = pose_base[:, :4]      # [B, 4]
        t_base = pose_base[:, 4:]      # [B, 3]
        delta_q = delta[:, :4]         # [B, 4]
        delta_t = delta[:, 4:]         # [B, 3]

        q_final = F.normalize(q_base + self.gamma * delta_q, dim=-1)  # normalize quaternion
        t_final = t_base + self.gamma * delta_t

        return torch.cat([q_final, t_final], dim=-1)  # [B, 7]
```

**3.2 Human LoRA**

**实际 SMPL dict 字段（确认）**：
- `smpl_shape`: (bs, max_humans, 10) - betas shape
- `smpl_transl`: (bs, max_humans, 3) - camera translation
- `smpl_rotmat`: (bs, max_humans, 6, 3, 3) - rotation matrix (from 6D rotvec)
- `smpl_expression`: (bs, max_humans, 10)

**注意**：postprocess_smpl 返回的 key 是 `smpl_shape`，不是 `betas`；是 `smpl_transl`，不是 `cam`。

Human LoRA 实现：

```python
class LoRAHumanHead(nn.Module):
    """Human SMPL residual: y_final = y_base + gamma * delta_y
    输入: H'_t [B,N_humans,C], q'_t [B,1,C]
    Human 输出是 dict: smpl_shape, smpl_transl, smpl_rotmat, smpl_expression
    """
    def __init__(self, dec_dim=768, shot_dim=768):
        # 每个参数单独 gamma
        self.gamma_shape = nn.Parameter(torch.tensor(0.0))
        self.gamma_transl = nn.Parameter(torch.tensor(0.0))
        self.gamma_rotmat = nn.Parameter(torch.tensor(0.0))
        # expression 不做 residual

        in_dim = dec_dim + shot_dim
        self.lora_shape = nn.Linear(in_dim, 10)      # betas: 10
        self.lora_transl = nn.Linear(in_dim, 3)      # transl: 3
        self.lora_rotmat = nn.Linear(in_dim, 54)    # rotmat: 6*3*3 = 54

    def forward(self, smpl_token, q_out, pred_smpl_dict):
        # smpl_token: [B, N_humans, dec_dim]
        # q_out: [B, 1, shot_dim] (q'_t)
        # pred_smpl_dict: dict with keys smpl_shape, smpl_transl, smpl_rotmat, smpl_expression
        q_expand = q_out.expand(-1, smpl_token.shape[1], -1)  # [B, N, shot_dim]
        x = torch.cat([smpl_token, q_expand], dim=-1)  # [B, N, dec_dim+shot_dim]

        # 不要 inplace 修改
        out = pred_smpl_dict.copy()
        out['smpl_shape'] = pred_smpl_dict['smpl_shape'] + self.gamma_shape * self.lora_shape(x)
        out['smpl_transl'] = pred_smpl_dict['smpl_transl'] + self.gamma_transl * self.lora_transl(x)
        # rotmat: [B, N, 6, 3, 3] -> flatten 后 54 维
        delta_rotmat = self.lora_rotmat(x).unsqueeze(-1).unsqueeze(-1)  # [B, N, 54, 1, 1]
        out['smpl_rotmat'] = pred_smpl_dict['smpl_rotmat'] + self.gamma_rotmat * delta_rotmat
        # expression 等其他字段保持不变
        return out
```

**3.3 World LoRA**

**实际 World pts3d 格式（确认）**：BxHxWx3（DPT 深度图格式），不是 BxNx3。

```python
class LoRAWorldHead(nn.Module):
    """World pointmap residual: X_world_final = X_world_base + gamma * Delta_X
    输入: F'_t [B,H,W,C], z'_t [B,1,C], q'_t [B,1,C]
    注意: world pts3d 是 BxHxWx3 格式，不是 BxNx3
    """
    def __init__(self, dec_dim=768, shot_dim=768):
        self.gamma = nn.Parameter(torch.tensor(0.0))
        # 输入: flatten(F'_t) + z'_t + q'_t = H*W*dec_dim + dec_dim + shot_dim
        self.lora = nn.Sequential(
            nn.Linear(dec_dim + shot_dim, 256),  # 先 pooled 特征
            nn.GELU(),
            nn.Linear(256, 3),  # 3D point residual
        )

    def forward(self, img_feat, pose_token, q_out, world_base):
        # img_feat: [B, H, W, dec_dim] - DPT 图像特征
        # pose_token: [B, 1, dec_dim] (z'_t)
        # q_out: [B, 1, shot_dim] (q'_t)
        B, H, W, C = img_feat.shape
        # Global average pool over spatial dimensions
        img_global = img_feat.mean(dim=[1, 2])  # [B, dec_dim]
        img_global = img_global.unsqueeze(1)  # [B, 1, dec_dim]

        x = torch.cat([img_global, q_out], dim=-1)  # [B, 1, dec_dim+shot_dim]
        delta = self.lora(x)  # [B, 1, 3]
        delta = delta.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 3]
        return world_base + self.gamma * delta
```

#### 实现顺序

| Step | 内容 | 位置 | 优先级 |
|------|------|------|--------|
| 1 | ShotTokenGenerator V1 | model.py | 必做 |
| 2 | StateGate | model.py | 必做 |
| 3 | ARCroco3DStereo.__init__ 中创建实例 | model.py | 必做 |
| 4 | 修改 _decoder 接受 f_shot | model.py | 必做 |
| 5 | 修改 _forward_impl 循环，预计算 q_tokens，应用 State Gate | model.py | 必做 |
| 6 | 修改 head split 分离 q_out | model.py | 必做 |
| 7 | LoRAPoseHead | dpt_head.py | 必做 |
| 8 | LoRAHumanHead | dpt_head.py | 必做 |
| 9 | LoRAWorldHead | dpt_head.py | 推荐同时做 |
| 10 | DPTPts3dPoseSMPL.forward 应用 LoRA | dpt_head.py | 必做 |
| 11 | freeze='shot_adaptation' | model.py | 必做 |
| 12 | train.yaml | config/train.yaml | 必做 |

#### 第一版可跳过

- ShotTokenGenerator V2（Patch Matching）
- cam head LoRA
- token-wise alpha
- forward_recurrent_lighter
- 解冻 decoder

#### 第一版实现目标

先保证：
1. 稳定跑通
2. 不破坏原 Human3R 推理流程
3. LoRA residual 形式简单正确

#### 实现注意事项

**1. Pose LoRA 格式依赖**
当前 Pose LoRA 实现假设 pose_base 格式为 quat4 + trans3（共 7D）。
必须先确认原 `downstream_head.pose_head` / `decpose` 的实际输出格式：
- 如果是 axis-angle (3D) + trans3 共 6D，需要调整 residual 维度
- 如果是其他 rotation representation（6D、9D），需要相应修改
- LoRA 输出维度必须与原 pose 维度匹配

**2. Human LoRA 禁止 inplace 修改**
```python
# 错误（inplace 修改）：
pred_smpl_dict['body_pose'] = pred_smpl_dict['body_pose'] + gamma * delta

# 正确（copy 后返回）：
out = pred_smpl_dict.copy()
out['body_pose'] = pred_smpl_dict['body_pose'] + gamma * delta
return out
```

**3. World LoRA 格式适配**
world_base 可能有多种格式：
- `[B, N, 3]`：标准 point cloud 格式，直接加 delta
- `[B, H, W, 3]`：DPT 深度图格式，需要 reshape 后处理
- `dict`：`{'pts3d': ..., 'conf': ...}` 等，需要按字段名处理

实现时必须先检查实际格式，不能假定点 token 是唯一的 3D 输出。

**4. Token 来源明确区分**
- **ShotTokenGenerator**：使用 decoder **输入** token，即 F_t（编码后的图像特征）
- **LoRA heads**：使用 decoder **输出** token，即 F'_t / z'_t / H'_t / q'_t
  - `F'_t`：用于 world LoRA（场景点云）
  - `z'_t`：用于 pose LoRA（相机位姿）
  - `H'_t`：用于 human LoRA（人体 SMPL 参数）
  - `q'_t`：用于所有 LoRA（shot condition）

---

## 2026/04/29

### 1. 移除 StateGate 模块

**commit**: `5fd582e` - refactor: 移除 StateGate 模块，直接使用 S0 重置状态（保留原始代码注释）

**变更**：
- 注释掉 `StateGate` 导入和实例化
- 从训练参数列表中移除 `state_gate`
- 修改状态更新逻辑：直接使用 `S0_expand`，不再使用门控
- `ShotTokenGenerator` 保留，用于生成 `q_t`

**影响**：可训练参数从 ~1.3M 减少到 ~1.2M

### 2. 将 Residual Adapter 改为 LoRA

**commit**: `30bf533` - refactor: 将 Residual Adapter 改为 LoRA（保留原始代码注释）

**变更**：
- `PoseResidualAdapter` → `PoseLoRALayer`
- `HumanResidualAdapter` → `HumanLoRALayer`
- `WorldResidualAdapter` → `WorldLoRALayer`
- LoRA rank=64
- gamma 初始化保持为 0

**LoRA 架构**：
```python
class PoseLoRALayer(nn.Module):
    def __init__(self, dec_dim=768, rank=64):
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.lora_A = nn.Linear(dec_dim * 2, rank, bias=False)  # 1536→64
        self.lora_B = nn.Linear(rank, 7, bias=False)  # 64→7

class HumanLoRALayer(nn.Module):
    def __init__(self, dec_dim=768, rank=64):
        self.gamma_shape = nn.Parameter(torch.tensor(0.0))
        self.gamma_transl = nn.Parameter(torch.tensor(0.0))
        self.lora_A_shape = nn.Linear(dec_dim * 2, rank, bias=False)
        self.lora_B_shape = nn.Linear(rank, 10, bias=False)
        self.lora_A_transl = nn.Linear(dec_dim * 2, rank, bias=False)
        self.lora_B_transl = nn.Linear(rank, 3, bias=False)

class WorldLoRALayer(nn.Module):
    def __init__(self, dec_dim=768, rank=64):
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.lora_A = nn.Linear(dec_dim * 2, rank, bias=False)  # 1536→64
        self.lora_B = nn.Linear(rank, 3, bias=False)  # 64→3
```

### 3. 更新 model.py 以使用 LoRA layers

**commit**: `89337c9` - refactor: 更新 model.py 以使用 LoRA layers

**变更**：
- 导入改为 `PoseLoRALayer, HumanLoRALayer, WorldLoRALayer`
- 模块声明改为 `pose_lora, human_lora, world_lora`
- 训练参数列表和调用位置已更新

### 4. 验证测试

**测试结果**：
- 参数数量：LoRA rank=64 配置下约 ~395K 参数
- Forward pass 正常
- gamma=0 时输出与 base model 一致（验证 residual 形式正确）

**已推送至远程仓库**

---

## 2026/04/30 - 2026/05/01

### 1. rank=128 LoRA 训练

**训练配置**：
- LoRA rank: 64 → 128
- epochs: 30
- batch_size: 2 (per GPU, 4 GPUs = 8)

**参数量变化**：

| 模块 | rank=64 | rank=128 |
|------|---------|----------|
| ShotTokenGenerator | 526K | 526K (不变) |
| PoseLoRA | 99K | 132K |
| HumanLoRA | 197K | 264K |
| WorldLoRA | 98K | 131K |
| **总计** | 789K | **1,053K** |

**commit**: `xxxxxx` - refactor: LoRA rank 64→128

### 2. Inference 代码问题 🚨

**问题**：发现 inference 时没有使用 LoRA！

**原因**：`forward_recurrent_lighter` (inference 路径) 不经过 LoRA layers

**修复尝试**：修改 `forward_recurrent_lighter` 以支持 shot token 生成和 LoRA 应用

**结果**：❌ **推理结果完全错误**，需要回滚

**状态**：待修复

### 3. 后续计划

1. **回滚 inference 修改**：恢复原始 inference 代码
2. **调查 LoRA 在 inference 不生效的原因**
3. **可能方案**：
   - 方案A：重新设计 inference 路径以支持 LoRA
   - 方案B：在 demo.py 中使用 training 路径进行 inference
   - 方案C：添加 StateGate 并重新评估架构

---

## 模型改动总结（2026/04 - 2026/05）

### 架构演进

```
原始 Human3R
    │
    ▼
添加 Shot-Aware Adaptation
    ├── ShotTokenGenerator     — 检测相邻帧不连续程度
    ├── StateGate             — 软门控状态重置
    ├── PoseResidualAdapter  — 位姿修正
    ├── HumanResidualAdapter — SMPL 参数修正
    └── WorldResidualAdapter — 场景点云修正
    │
    ▼
Movie3R v1 (StateGate + Residual Adapter)
    │
    ▼
Movie3R v2 (移除 StateGate，改为 LoRA)
    ├── ShotTokenGenerator — 保留
    ├── StateGate        — ❌ 已移除
    ├── PoseLoRA         — rank=128
    ├── HumanLoRA        — rank=128
    └── WorldLoRA        — rank=128
```

### 改动原因与思路

#### 1. 移除 StateGate
**原因**：
- StateGate 的软门控机制增加了复杂性
- 直接使用 S0 重置状态更简单，可能足够解决问题

**变更**：
- 状态更新从 `S_t = α*S_{t-1} + (1-α)*S0` 简化为 `S_t = S0`（直接重置）

**影响**：减少 ~98K 参数

#### 2. Residual Adapter → LoRA
**原因**：
- LoRA 有更好的正则化效果（低秩约束）
- 参数量更少，适合小数据集微调
- 初始状态更接近原始模型（gamma=0）

**LoRA 形式**：
```python
# 标准 LoRA: W' = W + BA
# delta = B @ A(x)
# output = base + gamma * delta

class PoseLoRALayer(nn.Module):
    def __init__(self, dec_dim=768, rank=64):
        self.gamma = nn.Parameter(torch.tensor(0.0))  # 初始化为0
        self.lora_A = nn.Linear(dec_dim * 2, rank, bias=False)
        self.lora_B = nn.Linear(rank, 7, bias=False)
```

#### 3. LoRA rank: 64 → 128
**原因**：
- rank=64 训练时 loss 很快 plateau
- 增加容量看是否能学到更多

**参数量变化**：
| 模块 | rank=64 | rank=128 |
|------|---------|----------|
| ShotTokenGenerator | 526K | 526K |
| PoseLoRA | 99K | 132K |
| HumanLoRA | 197K | 264K |
| WorldLoRA | 98K | 131K |
| **总计** | **789K** | **1,053K** |

### 当前问题

#### 🚨 Inference 路径不支持 LoRA

**发现**：
- Training 路径 (`_forward_impl`) ✅ 使用 LoRA
- Inference 路径 (`forward_recurrent_lighter`) ❌ 不使用 LoRA

**影响**：
- 训练时 LoRA 正常更新
- 但推理时完全不走 LoRA 路径
- **导致训练效果无法在推理时体现**

**尝试修复**：
- 在 `forward_recurrent_lighter` 中添加 shot token 生成和 LoRA 应用
- ❌ 结果：推理结果完全错误（已回滚）

**待解决**：
1. 分析原始 Human3R inference 架构
2. 确定正确的 inference + LoRA 方案

### Git Commits

| Commit | 内容 |
|--------|------|
| `5fd582e` | refactor: 移除 StateGate 模块 |
| `30bf533` | refactor: 将 Residual Adapter 改为 LoRA |
| `89337c9` | refactor: 更新 model.py 以使用 LoRA layers |
| `94ba8f6` | docs: 同步 git 历史到 work_log.md |
| (rank=128) | refactor: LoRA rank 64→128 (未单独 commit) |

### 关键文件

| 文件 | 改动 |
|------|------|
| `src/dust3r/shot_adaptation.py` | LoRA 类实现 |
| `src/dust3r/model.py` | LoRA 集成、freeze 模式 |
| `config/train.yaml` | 训练配置 |
| `demo.py` | 推理入口 |

---

## 2026/05/06 - LoRA Head V1 修正范围收窄

### 1. State 行为修正

**问题**：此前 `freeze='shot_adaptation'` 训练路径中，`i > 0` 时强制使用 `S0` 重置 recurrent state。

**结论**：这不符合当前设计。当前第一版不引入 StateGate，也不改变 Human3R 原始 recurrent state 行为。

**修正**：
- Shot token 只作为额外 prompt 进入 decoder
- recurrent state 仍默认使用前一帧 `state_feat`
- 后续如需自适应重置，再单独引入 StateGate 或 shot-label 控制

### 2. LoRA Head V1 设计目标

**目标**：优先修正镜头跳变导致的位置/朝向偏移，不追求修改重建细节。

**修正范围**：

| 模块 | V1 修正 | V1 不修正 |
|------|---------|-----------|
| PoseLoRALayer | `camera_pose` translation + quaternion | - |
| HumanLoRALayer | `smpl_transl` | `smpl_shape` / `smpl_rotmat` / `smpl_expression` |
| WorldLoRALayer | `pts3d_in_self_view` + `pts3d_in_other_view` 全局 shift | 局部 pointmap 几何 |

**理由**：
- 镜头跳变主要影响相机位姿、world alignment 和人体在相机/world 中的位置
- 人体 shape、body pose 细节、expression 不应由 shot adaptation 第一版直接修改
- world pointmap 先只做全局 shift，避免破坏局部重建质量

### 3. 实现说明

- `HumanLoRALayer` 移除 active `smpl_shape` 修正分支，只保留 `smpl_transl` 低秩修正
- `WorldLoRALayer` 同时应用到 `pts3d_in_self_view` 和 `pts3d_in_other_view`
- 原始 `shape + transl` HumanLoRA 和旧 model 调用方式已用注释块保留
- 旧 LoRA checkpoint 与该 V1 结构的 HumanLoRA 参数不完全一致，需要基于新结构重新训练

### 4. 仍待解决

1. Inference 路径 `forward_recurrent_lighter` 仍未接入 ShotToken/LoRA
2. LoRA rank 仍硬编码在 `model.py`，后续应写入 config
3. 如需真正检测 shot change，后续应使用 `shot_label` 或增加辅助 loss

### 5. shot_label 使用决策

**当前 V1 决策**：暂不使用 `shot_label` 作为显式监督。

**当前判断机制**：
```
F_i, F_{i-1}
    → ShotTokenGenerator
    → q_i
    → decoder cross-attention
    → q'_i
    → LoRA heads 修正 camera/world/human 输出
    → task loss 反向传播
```

也就是说，当前不是训练一个显式的 shot-change classifier，而是让 `q_i/q'_i` 通过最终任务 loss 隐式学习相邻帧差异和修正量之间的关系。

**理由**：
- V1 优先验证 `shot token + LoRA correction` 主路径是否有效
- 避免一开始引入额外 BCE loss 权重和训练不稳定因素
- `shot_label` 后续可以作为辅助监督增强 shot token 的可解释性

**后续 V2 方案**：
- 在 ShotTokenGenerator 上增加 `shot_logit`
- 用数据集已有 `shot_label` 做 BCE 辅助 loss
- `shot_label` 不直接硬开关 LoRA，只作为辅助监督
- 如重新加入 StateGate，可用预测的 shot probability 控制 state mixing

### 6. 文档同步

已同步更新：
- `docs/movie3r/README.md`：新增当前 LoRA Head V1 简述
- `docs/movie3r/model.md`：新增当前实现、state 行为、LoRA 修正范围、shot_label 使用策略
- `docs/movie3r/training.md`：新增当前 Shot Adaptation V1 训练范围和参数量说明
- `tasklist/TODO.md`：新增 LoRA Head V1 参数估算和 shot_label 后续 TODO

---

## 2026/05/07 - LoRA64 正式训练与推理消融结论

### 1. 正式训练结果

**实验目录**：`experiments/formal_training-4gpu-lora-64`

**配置摘要**：
- `freeze='shot_adaptation'`
- LoRA rank: `64`
- 只训练 `ShotTokenGenerator`、`PoseLoRALayer`、`HumanLoRALayer`、`WorldLoRALayer`
- 原 Human3R/CUT3R 主体冻结

**checkpoint**：
- `checkpoint-best.pth`：epoch 19 保存，`best_so_far=13.534343719482422`
- `checkpoint-final.pth`：训练结束保存

**日志现象**：
- train loss 明显下降
- val/test 的 AvatarReX 指标也有下降
- 但 demo 推理视觉结果严重错误：相机尺度异常、人很小、场景/人体乱成一团

### 2. 推理消融结果

同一段 `data/h36.mp4` 前 8 帧，比较 base 与 LoRA64 checkpoint：

| 模式 | camera 平移均值 | pointmap 范围均值 | SMPL 平移均值 | 结论 |
|------|----------------|------------------|---------------|------|
| base Human3R | `0.010` | `9.049` | `4.935` | 原模型尺度正常 |
| LoRA64 checkpoint，关闭 `enable_shot_adaptation` | `0.010` | `9.049` | `4.935` | checkpoint 中冻结的 base 权重正常 |
| LoRA64 checkpoint，打开 `enable_shot_adaptation` | `0.042` | `3.844` | `3.067` | shot adaptation 分支破坏尺度 |
| LoRA64 checkpoint，LoRA gamma 全置 0，仅保留 trained shot token | `0.020` | `3.844` | `1.966` | pointmap 崩坏主要来自 trained shot token 进入 decoder |

**结论**：
- checkpoint 加载正常，`<All keys matched successfully>`
- 原 Human3R/base 参数没有被破坏
- 错误主要来自 `enable_shot_adaptation=True` 后启用的新增分支
- LoRA residual 不是唯一问题；即使 LoRA gamma 置 0，trained shot token 进入 decoder 后仍会显著改变输出尺度

### 3. 当前判断

当前 V1 设计中，`q_t` 被直接 append 到 frozen decoder token 序列：

```text
[pose, image, human] -> [pose, image, human, q_t]
```

这不是严格 residual-safe 的改动。即使 LoRA residual 初始化或置零，额外 token 仍会通过 decoder self-attention 改变 pose/image/human token。因此“LoRA gamma=0 不破坏 base”的假设只对 LoRA 最后加法成立，不对 shot token 进入 decoder 成立。

### 4. 指标问题

当前 loss 下降不能直接说明 demo 视觉质量提升，原因包括：
- 训练/验证主要统计 AvatarReX 数据上的 task loss，不能覆盖 H36/demo 域外效果
- 日志没有监控 demo 里最敏感的绝对尺度指标，如 camera translation norm、pointmap extent、SMPL translation norm、human/scene scale ratio
- val/test dataset key 存在重复/覆盖风险，`dataset.split("(")[0]` 会让多个 `AvatarReX_Video` 或 `AvatarReX_AABB` 数据集名称混淆
- best checkpoint 监控逻辑更偏向最后一个 val/test key，不等价于整体视觉质量最优

### 5. 下一步方向：先验证 shot token 质量

在继续训练前，需要先独立检查两件事：

1. 数据集标签是否可靠：`AvatarReX_Video` 应全为连续帧，`shot_label=0`；`AvatarReX_AABB` 应为 `[0, 0, 1, 0]`
2. Shot token 是否有可解释区分度：连续帧与跳变帧的 `g_curr/g_prev` 差异、cosine similarity、`q_t` 范数和聚类分布应明显不同

建议新增诊断脚本，对训练前的 pretrained base 和训练后的 checkpoint 分别统计：
- `shot_label`
- `cosine_similarity(g_curr, g_prev)`
- `||g_curr - g_prev||`
- `||q_t||`
- `q_t` 与 `q_{t-1}` cosine
- 连续/跳变二分类的 AUC 或阈值可分性

如果 `g` 特征本身无法区分 AABB 跳变，问题在数据或特征选择；如果 `g` 可分但 `q_t` 不可分，问题在 ShotTokenGenerator 训练；如果 `q_t` 可分但推理仍崩，问题在 `q_t` 注入 decoder 的方式和训练 loss 约束。

---

## 2026/05/07 - Shot Token 质量验证结果

### 1. 新增诊断脚本

**脚本**：`scripts/analyze_shot_token.py`

**用途**：独立验证 AvatarReX `shot_label`、相邻帧 decoder image token 特征差异，以及 `ShotTokenGenerator` 输出 `q_t` 的统计质量。

**典型命令**：

```bash
PYTHONPATH=src:. .venv/bin/python scripts/analyze_shot_token.py \
    --num_samples 20 \
    --output output/shot_token_analysis/labels_only.json

PYTHONPATH=src:. TORCH_HOME=/workspace/cache/torch TORCH_HUB_USE_HEURISTICS=0 \
    .venv/bin/python scripts/analyze_shot_token.py \
    --model_path experiments/formal_training-4gpu-lora-64/checkpoint-best.pth \
    --num_samples 20 \
    --quiet \
    --output output/shot_token_analysis/lora64_best_20_view.json
```

### 2. 数据集格式验证

对三个 root 每类抽样 20 个 sample：
- `/workspace/data/avatarrex_zzr_output`
- `/workspace/data/avatarrex_lbn1_output`
- `/workspace/data/avatarrex_zxc_output`

结果：

| 数据类型 | 期望格式 | 实测结果 | invalid |
|----------|----------|----------|---------|
| `AvatarReX_Video` | `[0, 0, 0, 0]` | 全部 `[0, 0, 0, 0]` | 0 |
| `AvatarReX_AABB` | `[0, 0, 1, 0]` | 全部 `[0, 0, 1, 0]` | 0 |

结构校验也通过：
- Video：4 帧来自同一 sequence，frame 连续
- AABB：view0/1 来自 seqA，view2/3 来自 seqB，seqA != seqB，frame 连续，跳变发生在 view1 -> view2

**结论**：当前未发现 Video/AABB 数据集格式或 `shot_label` 问题。

### 3. Shot token 输入特征验证

统计 `g_curr/g_prev`：

| 指标 | 连续帧 label=0 | 跳变帧 label=1 | AUC |
|------|----------------|----------------|-----|
| `cosine(g_curr, g_prev)` | `0.9990` | `0.9889` | `0.9997`（使用负 cosine） |
| `||g_curr - g_prev||` | `0.697` | `2.445` | `0.9997` |

**结论**：decoder image token 的全局特征本身几乎可以完美区分连续/跳变。问题不在输入特征不可分。

### 4. 训练后 `q_t` 统计

LoRA64 `checkpoint-best.pth` 的 `q_t`：

| 指标 | 连续帧 label=0 | 跳变帧 label=1 | 结论 |
|------|----------------|----------------|------|
| `||q_t||` | `62.17` | `62.21` | 几乎无区分，且幅度非常大 |
| `cos(q_t, q_{t-1})` | median `0.9987` | median `0.9971` | 方向变化很小 |
| view2 `||q_t-q_{t-1}||` 连续 vs 跳变 | `1.21` | `5.22` | 有一定跳变响应 |
| `q_norm` AUC | `0.51` | - | 不能用范数区分跳变 |
| `q_delta_norm` overall AUC | `0.55` | - | 受 view1/q_init 影响，整体可分性弱 |

与 decoder token 尺度比较：

| 指标 | 均值 |
|------|------|
| decoder image token norm | `23.15` |
| pooled image token `g` norm | `17.03` |
| trained `q_t` norm | `63.56` |
| `q_t / decoder token norm` | `2.75x` |

**结论**：
- `q_t` 不是完全没有感知跳变：view2 的 `q_delta_norm` 在跳变处明显变大
- 但 `q_t` 的绝对幅度失控，连续帧和跳变帧都维持约 `62` 的大范数
- 这个大范数 token 被直接 append 到 frozen decoder，容易在所有帧上强行扰动 base 输出
- 这解释了为什么 LoRA gamma 全置 0 时 pointmap 仍然崩：问题来自 trained shot token 本身进入 decoder，而不是仅来自 LoRA output residual

### 5. 当前定性判断

**数据集没有明显问题。输入特征没有明显问题。Shot token 训练结果有问题。**

具体问题不是“shot token 完全没学到跳变”，而是：
- 学到了过强的全局 prompt
- 没有学到连续帧 no-op 行为
- 缺少显式 shot-label/gating/范数约束
- `q_t` 进入 decoder 的方式对 frozen base 不安全

### 6. 下一步改进方向

建议保留 shot token 进入 decoder 的思路，但必须增加约束：

1. 在 `ShotTokenGenerator` 增加 `shot_logit`，用 `shot_label` 做 BCE 辅助监督
2. 用 `shot_prob` 或 `shot_label` gate 控制 `q_t` 强度，连续帧接近 no-op
3. 对 `q_t` 做 `LayerNorm` 或显式范数约束，使其尺度接近 decoder token，而不是 2.7x
4. 增加训练/验证指标：`q_norm`、`q_delta_norm`、`shot_auc`、camera/pointmap/SMPL 尺度监控
5. 如继续 append token，建议从小强度初始化，例如 `q_t = shot_scale * LN(q_raw)`，`shot_scale` 初始接近 0

### 7. 术语和约束方案补充

关于 `cosine(g_curr, g_prev)`：连续帧约 `0.9990`、跳变帧约 `0.9889`，单看数值都接近 1，但在高维 decoder token 特征里该差异很稳定；同时 `||g_curr-g_prev||` 从连续帧 `0.697` 增至跳变帧 `2.445`，AUC 约 `0.9997`，因此可以判断输入特征足够区分 shot change。

关于 no-op：这里指连续帧时 shot token 应尽量“不操作”，即不明显改变原 Human3R 的 camera、pointmap、SMPL 输出；只有在 `shot_label=1` 的跳变帧才允许更强干预。

下一版推荐结构：

```text
q_raw, shot_logit = ShotTokenGenerator(g_curr, g_prev, diff, sim)
shot_prob = sigmoid(shot_logit)
q_t = shot_scale * shot_prob * LayerNorm(q_raw)
```

推荐先实现以下约束：

| 约束 | 目的 | 优先级 |
|------|------|--------|
| `BCE(shot_logit, shot_label)` | 显式训练 shot/change 判断 | 最高 |
| `shot_prob` gate | 连续帧弱化 `q_t`，跳变帧增强 `q_t` | 最高 |
| `LayerNorm(q_raw)` | 控制 `q_t` 尺度，避免大范数 token 破坏 decoder | 高 |
| `shot_scale` 初始接近 0 | 训练初期保持近似 base Human3R | 高 |
| `(1-shot_label) * ||q_t||^2` | 连续帧 no-op 正则 | 高 |
| 输出尺度监控 | 避免 loss 下降但 demo 尺度崩坏 | 高 |

可选输出级 no-op loss：

```text
L_noop = (1 - shot_label) * ||pred_with_shot - stopgrad(pred_without_shot)||
```

该项只约束连续帧，不限制跳变帧；目的是保护原 Human3R 在正常连续视频上的行为。

### 8. 三层改造与提交计划

接下来按三层约束逐层实现，每层都遵循“先注释备份原代码并 commit，再新增实现并 commit”的流程，便于回退。

| 层级 | 目标 | 备份 commit | 实现 commit |
|------|------|-------------|-------------|
| Layer 1 | `ShotTokenGenerator` 输出 `shot_logit`，用 `shot_label` 做 BCE 辅助监督 | 注释保留原 `ShotTokenGenerator.forward` 和当前训练 loss 调用点 | 新增 `shot_logit`、`shot_loss` 监控与训练项 |
| Layer 2 | 给 `q_t` 加 `shot_prob` gate、`LayerNorm` 和 `shot_scale`，控制 token 强度 | 注释保留当前无约束 `q_t` 注入 decoder 逻辑 | 新增 `q_t = shot_scale * shot_prob * LayerNorm(q_raw)` |
| Layer 3 | 连续帧 no-op 输出约束，保护 base Human3R 行为 | 注释保留当前单路 forward/loss 逻辑 | 新增 `pred_with_shot` vs `pred_without_shot` 的连续帧 no-op loss |

推荐最终训练目标：

```text
L = L_task
  + lambda_shot * BCE(shot_logit, shot_label)
  + lambda_q0 * (1 - shot_label) * ||q_t||^2
  + lambda_noop * (1 - shot_label) * ||pred_with_shot - stopgrad(pred_without_shot)||
```

其中 Layer 1/2 先解决“知道哪里是跳变”和“连续帧不要强扰动 decoder”；Layer 3 再直接约束连续帧输出接近 shot-off/base 输出。

---

## 2026/05/09

### 1. ShotToken V5 规划

新增设计文档：`docs/movie3r/shot_token_v5_plan.md`。

V5 目标聚焦两个问题：

```text
1. 跳变帧本身的 camera pose 要算对，尤其是 AABB 的 B1/view2。
2. 整个 AABB 序列的 A1/A2/B1/B2 都要落在同一个 GT world coordinate 下。
```

### 2. V5.1 首选方案

V5.1 采用 interleaved pose-only shot attention：

```text
每层 decoder 正常更新 [pose, image, human]
每层后只让 pose token 和 q_t 做一次 attention
更新后的 pose token 进入下一层 decoder
```

该方案不是最终输出后的单次后处理，而是在 decoder 内部逐层修正 pose token。它比 V4 更早介入 pose 生成过程，同时比 V2 安全，因为 `q_t` 不作为普通 token 直接暴露给 image/human/pointmap 分支。

插入层数可调：

| 配置 | 用途 |
|------|------|
| 每层插入 | V5.1 首版默认，最大化 pose 修正能力 |
| 每 2 层插入 | 如果扰动过强，降低注入频率 |
| 后半层插入 | 如果早期 decoder 被污染，只在语义更稳定阶段注入 |
| 最后几层插入 | 最保守版本，介于 V4 和 V5.1 全层之间 |

### 3. V5.1 loss 同步调整

当前已有 `pose_loss`、`pose_loss_view2_AABB`、`shot_bce`、`shot_q0_loss`、`shot_noop_loss`、`shot_pointmap_keep_loss`。V5.1 需要补充针对跳变边界的显式 camera 监督。

新增核心 loss：

| Loss | 监督目标 |
|------|----------|
| `L_boundary_abs` | 明确监督 A2 和 B1 各自 absolute pose 算对 |
| `L_jump_rel` | 监督 `relative(A2, B1)` 等于 GT 的真实跳变相对位姿 |
| `L_anchor` | 监督 B1/B2 相对于 A2 接回同一个 world coordinate |

重要澄清：`A2 -> B1` 是跳变边界，不是要求 A2 和 B1 位姿相同。正确监督是：

```text
relative(T_pred_A2, T_pred_B1) ≈ relative(T_gt_A2, T_gt_B1)
```

V5.1 首版暂不优先加入全相邻帧 `L_rel_all` 和 supervised residual loss，避免和 `L_jump_rel/L_anchor` 重复或增加调参复杂度。

### 4. No-Harm 约束注意事项

`shot_noop_loss` 和 `shot_pointmap_keep_loss` 需要避免阻止 AABB 的 B 段 correction。建议首版：

```text
noop 主要作用在 AAAA/Video 样本，或至少优先限制 is_video=True 的连续视频。
pointmap keep 保留监控，如果 B 段 pose 修不动，再降低 keep/noop 权重。
```

### 5. V5.2 后备方案

如果 V5.1 仍不能修复 AABB 跳变和 B 段 anchor，则进入 V5.2：改 decoder attention mask，让 ShotToken 真正作为受控 token 进入 decoder。

V5.2 目标权限：

```text
pose token 可以 attend shot token
image tokens 不能 attend shot token
human tokens 默认不能 attend shot token
shot token 不能自由污染 image/human tokens
```

当前工程难点是 `src/dust3r/blocks.py` 的 `Attention`、`CrossAttention`、`DecoderBlock` 没有暴露 `attn_mask` 参数，需要改底层 decoder block 和 `_decoder()` 调用链。该方案更接近理论正确做法，但工程量和回归风险更高，因此作为 V5.1 失败后的后备路线。

---

## 2026/05/13

### 1. RICH AABB Step1：外部 anchor 能映射回 Human3R encoder patch token

验证目标：不修改 encoder / decoder，只确认 XFeat semi-dense + RICH official mesh 得到的真实静态背景 anchors，是否能在原版 Human3R encoder output patch token 中被重新找到。

核心流程：

```text
RICH_4Human3R/Training AABB
-> XFeat semi-dense matches
-> RICH scan mesh + XML calibration 验证 static background anchors
-> 2D anchor 映射到 Human3R crop / patch grid
-> 比较 positive anchor patch token cosine 与 random negative
```

新增脚本：

```text
scripts/verify_rich_anchor_encoder_similarity.py
scripts/verify_rich_aabb_anchor_step1.py
```

AABB boundary 结果：

| 样本 | mesh anchors | unique patch anchors | positive cosine | random cosine | rank median | pos > random |
|------|--------------|----------------------|-----------------|---------------|-------------|--------------|
| `guitar cam06->cam07 f244` | 77 | 41 | 0.594 | 0.249 | 4 | 92.7% |
| `juggle cam02->cam01 f197` | 490 | 179 | 0.750 | 0.282 | 3 | 97.8% |
| `guitar cam01->cam03 f5` | 9 | 7 | 0.486 | 0.315 | 38 | 85.7% |

结论：

```text
1. 外部 XFeat/mesh anchors 不是噪声。
2. Human3R encoder patch token 中已经保留了可用的跨视角静态背景对应信息。
3. 当前阶段不需要修改 encoder，也不需要让 anchor 进入完整 decoder sequence。
4. anchor 多时稳定；anchor 少时仍有信号但 rank 不稳定，需要 quality gate / fallback。
```

### 2. Correction proxy：anchors 能提供 re-anchor 修正信息

新增脚本：

```text
scripts/analyze_rich_aabb_anchor_correction.py
scripts/build_rich_anchor_evidence.py
```

验证目标：测试 boundary anchors 是否能提供比 no-correction 更好的 reference patch lookup prior。

结果：

| 样本 | anchors | no correction | translation | affine | quality gate |
|------|---------|---------------|-------------|--------|--------------|
| `guitar cam06->cam07` | 41 | 3.16 | 4.47 | 1.00 | 0.74 |
| `juggle cam02->cam01` | 179 | 3.16 | 1.00 | 1.00 | 0.81 |
| `guitar cam01->cam03` | 7 | 10.05 | 2.24 | 1.00 | 0.22 |

单位是 patch error，越低越好。

结论：

```text
1. anchors 不只是能找到对应 patch，也能提供 correction evidence。
2. 简单 mean(delta_uv) / translation 不总可靠，可能比 no-correction 更差。
3. affine 作为 coarse re-anchor prior 明显更稳。
4. weak sample 即使 affine 拟合好，也要因为 anchor 数少而降低 gate。
```

### 3. AnchorTokenGenerator 原型：global affine + local residual tokens

当前目标已从“做 correction head”澄清为：把外部 anchors 转成更准确、更有几何含义的 local ShotToken / AnchorToken。

新增脚本：

```text
scripts/prototype_rich_anchor_tokens.py
```

AnchorToken 结构：

```text
AnchorToken_k = {
    key_cur_feature: F_cur[j],
    value_ref_feature: F_ref[i],
    ref_pos_norm: pos_ref[i],
    cur_pos_norm: pos_cur[j],
    delta_uv_norm: pos_cur[j] - pos_ref[i],
    confidence,
    mesh_error_px,
    encoder_cosine
}
```

验证方式：leave-one-out。每次拿掉一个真实 anchor，用剩余 AnchorTokens 预测被拿掉的 current patch 应该对应 reference 的哪个 patch。

结果：

| 样本 | tokens | same-position | affine | token-soft | token-affine-residual |
|------|--------|---------------|--------|------------|-----------------------|
| `guitar cam06->cam07` | 41 | 3.16 | 1.15 | 1.41 | 0.82 |
| `juggle cam02->cam01` | 179 | 3.16 | 0.82 | 1.58 | 0.66 |
| `guitar cam01->cam03` | 7 | 10.03 | 1.05 | 1.46 | 1.14 |

结论：

```text
1. AnchorToken 不应只是 nearest-neighbor memory；token-soft 单独使用不稳定。
2. 最有效形式是：global affine 粗对齐 + local AnchorToken residual 修正。
3. anchor 数充足时，AnchorToken residual 优于纯 affine。
4. anchor 太少时 residual 不稳定，必须由 quality_gate 降权或 fallback。
```

相关输出：

```text
output/rich_aabb_anchor_step1/
output/rich_aabb_anchor_correction_proxy/
output/rich_anchor_evidence/
output/rich_anchor_token_prototype/
```

### 4. Top-K / quality-gate AnchorToken 选择验证

新增脚本：

```text
scripts/validate_rich_anchor_token_selection.py
```

目的：验证实际推理时是否需要保留所有 anchors，还是只保留少量 top-K AnchorTokens 即可。

比较策略：

| 策略 | 说明 |
|------|------|
| `confidence_topk` | 选择 confidence 最高的 K 个 tokens |
| `diverse_topk` | confidence 优先，同时保持空间分散 |
| `random_k` | 随机选择 K 个，作为稳定性 baseline |

结果：

| 样本 | 总 tokens | gate | 最佳策略 | affine error | token residual error | improvement |
|------|-----------|------|----------|--------------|----------------------|-------------|
| `guitar cam06->cam07` | 41 | strong | diverse top-8 | 1.10 | 0.77 | +0.32 |
| `juggle cam02->cam01` | 179 | strong | random top-64 baseline | 0.81 | 0.65 | +0.16 |
| `guitar cam01->cam03` | 7 | fallback | random top-4 | 1.25 | 1.24 | +0.02 |

补充：`juggle cam02->cam01` 中 deterministic 策略也有效，`confidence_topk K=4` 已达到 `affine 0.89 -> token residual 0.66`，说明并非必须大量 tokens。

结论：

```text
1. 推理时不需要保留所有 anchors。
2. anchor 数充足时，8-16 个高质量 / 空间分散 AnchorTokens 已经能提供有效 residual correction。
3. confidence + spatial diversity 更适合作为实际 deterministic selection。
4. anchor 数 < 8 时，residual gain 不可靠，应 fallback 或只弱使用 affine evidence。
```

输出：

```text
output/rich_anchor_token_selection/
```
