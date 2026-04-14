# AvatarReX → Human3R 格式对比分析

## 1. Human3R (BEDLAM) 数据格式要求

每个 view（一帧）的必要字段（来自 `bedlam.py` 的 `_get_views`）：

| 字段 | 类型/形状 | 说明 |
|------|-----------|------|
| `img` | PIL.Image / np.array (H,W,3) | RGB 图像 |
| `depthmap` | np.array (H,W) float32 | 深度图（米） |
| `camera_pose` | np.array (4,4) float32 | 相机外参（c2w） |
| `camera_intrinsics` | np.array (3,3) float32 | 内参矩阵 K |
| `msk` | np.array (H,W) | 人物 mask（可选） |
| `smpl_mask` | np.array (max_humans,) bool | 人物存在标志 |
| `smplx_root_pose` | np.array (1,3) float32 | 根旋转（旋转向量） |
| `smplx_body_pose` | np.array (21,3) float32 | 21个关节旋转 |
| `smplx_jaw_pose` | np.array (1,3) float32 | 下颌旋转 |
| `smplx_leye_pose` | np.array (1,3) float32 | 左眼旋转 |
| `smplx_reye_pose` | np.array (1,3) float32 | 右眼旋转 |
| `smplx_left_hand_pose` | np.array (15,3) float32 | 左手旋转 |
| `smplx_right_hand_pose` | np.array (15,3) float32 | 右手旋转 |
| `smplx_shape` | np.array (11,) float32 | 形状参数 |
| `smplx_transl` | np.array (3,) float32 | 平移 |
| `smplx_gender_id` | scalar | 性别（0=neutral） |

**目录结构**：
```
Processed_BEDLAM/
  Training/（或 Test/）
    scene_name/
      rgb/        # *.png
      depth/      # *.npy
      mask/       # *.png（可选）
      cam/        # *.npz  → {pose:(4,4), intrinsics:(3,3)}
      smpl/       # *.pkl  → [ {smplx_*: ...}, ... ]
```

**关键约束**：
- `num_views=4`：一次采样4帧作为一个训练样本
- 深度图必须是米制（metric），无效值置0，>200m也置0
- camera_pose 是 **c2w** 矩阵（相机到世界）
- SMPLX 关节顺序：21个body joints + jaw + 2×leye/reye + 2×hands

---

## 2. AvatarReX 数据现状

### 2.1 整体情况
- **16个序列**，每个约 2000 帧（1500×2048 RGB）
- 共享的 `smpl_params.npz`：覆盖 2001 帧（仅一个序列的 SMPL 参数）
- `calibration_full.json`：每序列一组 K, R, T（**不是每帧**）

### 2.2 smpl_params.npz 内容
```
betas:          (1, 10)      ← 注意：SMPL是10维，SMPLX是11维
global_orient:  (2001, 3)    ← 根旋转（旋转向量）
transl:         (2001, 3)    ← SMPL平移
body_pose:      (2001, 63)   ← 23joints×3 = SMPL格式（≠SMPLX的21joints）
jaw_pose:       (2001, 3)
expression:     (2001, 10)   ← SMPLX独有
left_hand_pose: (2001, 45)   ← 15joints×3
right_hand_pose:(2001, 45)
```

### 2.3 calibration_full.json 每序列内容
```python
K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]    # 3×3内参，flattened
R: [r11,...,r33]                      # 3×3旋转矩阵（相机到世界？）
T: [tx, ty, tz]                       # 平移向量（单位待确认）
distCoeff: [0,0,0,0,0]                # 畸变系数（全零）
imgSize: [w, h]                       # [1500, 2048]
```

---

## 3. 格式差距逐项对比

### 3.1 已满足 ✓
| 项目 | 状态 |
|------|------|
| RGB 图像 | ✓ 序列目录下有 JPG |
| 内参 K（3×3） | ✓ calibration_full.json 有 |
| SMPL body_pose / transl / global_orient / jaw | ✓ smpl_params.npz 有 |
| SMPL betas (10维) | ✓ 有，接近 SMPLX 的 11 维 |
| 序列目录结构 | ✓ 16个序列子目录 |

### 3.2 关键缺失 ✗

| 项目 | BEDLAM 要求 | AvatarReX 现状 | 解决方案 |
|------|-------------|----------------|----------|
| **深度图 depthmap** | 每帧一个 *.npy | **完全没有** | 需要用 Depth-Anything-3 或 SMPL→depth渲染生成 |
| **每帧相机外参 camera_pose (4×4)** | 每帧不同 | **所有帧共享同一组 R,T** | 可用 SMPL transl + 相机K反投影估计；或固定为单位矩阵 |
| **SMPLX 参数格式** | 21 joints, 11维betas, leye/reye | SMPL: 23 joints, 10维betas, 无leye/reye | 需要格式转换：23→21 joints, 补零leye/reye, betas补0 |
| **手部姿态** | (15,3)×2（SMPLX格式） | (45,)×2（扁平SMPL格式） | 需要 reshape 并转换 |
| **人物 mask** | 每帧 *.png | **没有** | 需要 SMPL 渲染或手动标注 |
| **SMPL 参数覆盖** | 所有帧都需要 | smpl_params.npz 覆盖全部16序列（共享同一套SMPLX） | 所有16个序列均可处理（22053912少1帧需跳过末尾） |
| **目录名** | scene_name/rgb/... | 直接是序列ID/rgb/... | 重命名为 Training/Test/序列ID |

### 3.3 缺失程度评级

| 级别 | 字段 | 说明 |
|------|------|------|
| 🔴 致命 | 深度图 depthmap | 训练必需，缺失无法训练 |
| 🔴 致命 | 每帧 camera_pose | 4×4外参矩阵，缺失无法计算 ray_map |
| 🟡 中等 | SMPL→SMPLX 格式转换 | 23joints→21joints, betas 10→11, leye/reye补零 |
| 🟡 中等 | 人物 mask | 非必需（代码中有 `if mask exists` 分支） |
| 🟢 轻微 | smpl_gender_id | 可默认为 0 (neutral) |

---

## 4. 正确理解：smpl_params 覆盖全部 16 个序列

AvatarReX 是**同一个人的多视角 Motion Capture 数据集**：
- 16 个序列 = 同一个人的 16 个相机视角，同一动作被 16 台相机同时拍摄
- smpl_params.npz 的 **2001 帧 motion capture 数据对应所有 16 个序列**
- 序列 `22010708[frame_i]`、`22010710[frame_i]`、`...`、`22139907[frame_i]` **共用 smpl_params[frame_i]`
- smpl_params 的 transl 随时间变化（人有动作），不是静止的

**帧数对应关系**：
| 序列 | 图像帧数 | SMPL 参数来源 |
|------|---------|--------------|
| 22010708 | 2001 | smpl_params[0:2001] |
| 22010710 | 2001 | smpl_params[0:2001] |
| ... | 2001 | smpl_params[0:2001] |
| 22053912 | 2000 | smpl_params[0:2000]（少1帧） |
| 其余序列 | 2001 | smpl_params[0:2001] |

**注意**：22053912 仅有 2000 帧，对应 smpl_params[0:2000]。

---

## 5. 转换路线图

### 阶段1（当前可做）：预处理脚本生成
1. **处理全部 16 个序列**（所有序列共享同一套 SMPL 参数）
2. 创建 BEDLAM 风格目录结构
3. 复制图像（jpg→png）
4. 为每个序列生成 smpl/*.pkl（指向同一套 smpl_params，但 frame_idx 按序列内索引）
5. 为每个序列生成 cam/*.npz（使用该序列自己的 R,T 标定，person_transl 取自 smpl_params）
6. 预留 depth/ 目录（待后续填入）
7. 生成 manifest.json 追踪来源

### 阶段2（后续）：深度图生成
- 调用 Depth-Anything-3 批量推理

### 阶段3（后续）：全量微调
- 验证数据可训练性
