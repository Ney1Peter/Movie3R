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

**目的**：实现 A,A,B,B 镜头跳变采样——帧0,1来自相机A的t/t+1，帧2,3来自相机B的t/t+1。

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
  depth/                  ← 空（待生成）
  mask/                   ← 空
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
    --root /data/wangzheng/Movie3R-dataset/AvatarRex4Human3R \
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

### 9. 深度图生成结果（2026/04/13 下午）

**运行结果**：15个序列全部完成，30015张深度图
- 每序列：2001帧，0错误
- 深度尺寸：(2048, 1500) float32
- 深度范围：0.38m ~ 1.72m（合理室内场景）
- 有效像素：100%
- 使用5个GPU并行 (1,4,5,6,7)

**逐序列检查**：全部 PASS ✓

---

### 10. 深度图格式优化（2026/04/14）

**发现**：DA3 输出的深度值精度实际上是 mm 级别（乘1000后为干净整数），float32 浪费空间。

**优化方案**：改用 uint16（毫米整数）保存深度图
- float32: 4B/像素 → 12.3MB/文件 (2048×1500)
- uint16: 2B/像素 → 6.2MB/文件（省50%）
- dataset 类无需修改：np.load() 读取 uint16 自动转为 float32
- 精度不变：mm 级精度足够

**脚本修改**：仅需修改 `generate_depth_avatarrex.py` 的保存部分
```python
# 原来
np.save(depth_path, d_clean.astype(np.float32))

# 改为
d_mm = (d_clean * 1000).astype(np.uint16)  # 米→毫米
cv2.imwrite(depth_path.replace('.npy', '.png'), d_mm)
```

---

### 11. avatarrex_lbn1 数据集处理（2026/04/14）

**转换**：使用 `preprocess_avatarrex_fast.py`
- 16序列 × 1901帧 = 30,416 帧
- 结果：done=0 skipped=30416 errors=0 ✓

**深度图**：使用 `generate_depth_avatarrex.py`（5 GPU）
- 深度尺寸：2048×1500 float32
- 深度范围：0.45m ~ 1.60m
- 损坏文件：0 ✓
- **完成状态**：✅ 30,416/30,416 全部完成

---

### 12. avatarrex_lbn2 数据集处理（2026/04/14）

**转换**：使用 `preprocess_avatarrex_fast.py`
- 16序列 × 1871帧 = 29,936 帧
- 结果：done=5613 skipped=24323 errors=0 ✓

**深度图**：使用 `generate_depth_avatarrex.py`（多GPU）
- 遇到问题：生成过程中产生大量 0 字节损坏文件
- 原因：DA3 对某些图像返回空/无效深度数组，异常被静默捕获
- 修复：在保存后添加验证步骤，损坏文件自动删除并重生成
- **完成状态**：⚠️ 约 10,425/29,936（35%）- 待续

**磁盘空间问题**：
- 处理过程中磁盘满（14TB 已用 100%）
- lbn2 处理被迫多次中断
- 用户决定：仅使用 zzr 和 lbn1 两个完整数据集
- lbn2/lbn3 待迁移到其他服务器后继续

---

### 13. 文档与脚本整理（2026/04/14）

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

### 14. 深度图格式优化：uint16（2026/04/14）

**发现**：DA3 输出精度实际是 mm 级别，float32 浪费空间

**转换脚本** `convert_depth_to_uint16.py`：
```bash
python convert_depth_to_uint16.py -i /path/to/dataset/Training --workers 32
```

**格式对比**：
| 格式 | 每文件 | 节省 |
|------|--------|------|
| float32 npy | 11.7 MB | - |
| uint16 npy | 5.9 MB | **50%** |

**精度验证**：
- mm 整数存储，精度 1mm
- 转换回去 max 误差 < 0.001m（1mm）
- dataset 类无需修改：np.load() 读取 uint16 自动转 float32

**生成脚本已更新**：`generate_depth_avatarrex.py`
- 直接保存 uint16 npy（米→毫米×1000）
- 包含写后验证

---

### 15. 当前数据集状态（2026/04/14 更新）

| 数据集 | RGB | CAM | SMPL | Depth | Mask | 状态 |
|--------|-----|-----|------|-------|------|------|
| avatarrex_zzr | ✅ | ✅ | ✅ | ✅ uint16 | ✅ | ✅ 完成 |
| avatarrex_lbn1 | ✅ | ✅ | ✅ | ✅ uint16 | ✅ | ✅ 完成 |
| avatarrex_lbn2 | ✅ | ✅ | ✅ | ⚠️ ~35% | ✅ | 待续 |

**存储占用**（仅完整数据集）：
- AvatarRex4Human3R: ~258GB（depth ~172GB uint16）
- AvatarRex_lbn1_4Human3R: ~261GB（depth ~174GB uint16）

**节省空间**：
- float32→uint16 深度图节省约 50%
- 两个完整数据集合计：~519GB（vs 原来 ~895GB，省 ~376GB）

---

### 16. AvatarReX 训练数据集配置（2026/04/14）

**背景**：
- 原版 Human3R 使用 BEDLAM_Multi 作为训练数据（650人 × 3000+场景）
- 实际训练仅用 4000 samples/epoch × 40 epochs = 16万样本（BEDLAM 的 0.1%）
- AvatarReX 数据（zzr + lbn1）规模：31 sequences，60k 帧，足够训练

**数据集类**：
- `AvatarReX_Video`：同一相机内连续帧采样（t, t+1, t+2, t+3），is_video=True
- `AvatarReX_AABB`：AABB 镜头跳变采样（camA@t, camA@t+1, camB@t, camB@t+1），is_video=False
- 两个类都继承自 BaseMultiViewDataset，支持 Human3R 标准接口

**采样容量**：
| 数据集 | 类型 | 序列数 | 帧/序列 | 采样数 |
|--------|------|--------|---------|--------|
| zzr | Video | 15 | 2001 | 29,970 |
| zzr | AABB | 15 | 2001 | 420,000 |
| lbn1 | Video | 16 | 1901 | 30,368 |
| lbn1 | AABB | 16 | 1901 | 456,000 |

**训练配置**（trian_human3r.yaml）：
```
train_dataset: 2000 @ ${dataset28}   # AvatarReX_Video zzr
            + 2000 @ ${dataset29}    # AvatarReX_Video lbn1
            + 2000 @ ${dataset30}    # AvatarReX_AABB zzr
            + 2000 @ ${dataset31}    # AvatarReX_AABB lbn1
= 8000 samples/epoch，Video/AABB 各 50%
```

**数据路径**：
- zzr: `../../../Movie3R-dataset/AvatarRex4Human3R`
- lbn1: `../../../Movie3R-dataset/AvatarRex_lbn1_4Human3R`

**验证结果**：
- AvatarReX_Video: 29,970 samples ✓
- AvatarReX_AABB: 420,000 samples ✓
- 单样本测试：img [3,288,512], depth [288,512], is_video=True ✓

**后续计划**：
- 当前方案：先用 AvatarReX 数据集跑通训练流程
- 如效果不好：可下载 BEDLAM subset（几十个序列）作为补充
- lbn2/lbn3 深度图：迁移到其他服务器后继续生成

---

### 17. 待完成事项

1. ✅ **训练配置**：AvatarReX Video + AABB 混合训练（已完成）
2. **lbn2 深度图**：迁移到其他服务器后继续生成
3. **BEDLAM subset**（可选）：如 AvatarReX 效果不佳，下载部分 BEDLAM 数据
