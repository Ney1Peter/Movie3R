# Movie3R 数据构建任务——我的理解

## 一、任务背景一句话总结

当前 Human3R 在单目视频流式推理时，对**镜头跳变（shot change）**场景的鲁棒性不足，表现为时序污染导致相机估计漂移。本任务**暂不改模型结构**，而是通过**构造包含明确分镜切换模式的新数据集**，用数据驱动的方式验证"数据分布重构能否解决shot change问题"。

---

## 二、本阶段（E1）核心约束

| 约束项 | 具体要求 |
|--------|----------|
| **不改模型代码** | `src/dust3r/model.py` 等核心代码保持不动 |
| **训练方式** | 全量微调（基于预训练权重继续训练，非从零训练） |
| **优先目标** | 先产出**可训练的数据pipeline**，模型实验放后续 |
| **数据格式依据** | 以第3节A类文件（bedlam.py / preprocess_bedlam.py / smpl_model.py / trian_human3r.yaml）为唯一标准 |

---

## 三、数据策略要点

### 3.1 样本组织形式
- **4帧样本**为一个训练样本（shot模式）
- 优先验证 `A,A,B,B` 视角切换模式（即前两帧来自视角A，后两帧来自视角B）
- 深度真实值暂不生成，先预留目录，后续接 Depth-Anything-3 批量生成

### 3.2 输入数据来源
- **源数据**：AvatarReX（或其他指定人物数据源）
- **目标格式**：Human3R 可训练的数据结构

### 3.3 关键格式字段（参考 bedlam.py）
从 `bedlam.py` 的 `BEDLAM_Multi` 数据集类来看，每个 view 需要包含：

| 字段名 | 含义 | 形状/Dtype |
|--------|------|------------|
| `img` | RGB图像 | H×W×3 uint8 |
| `msk` | 人物mask（可选） | H×W |
| `depthmap` | 深度图 | H×W float32 |
| `camera_pose` | 相机外参（4×4矩阵） | float32 |
| `camera_intrinsics` | 相机内参（3×3矩阵） | float32 |
| `smpl_mask` | 人物是否存在 | max_humans bool |
| `smplx_root_pose` | SMPLX根姿态 | (1,3) float32 |
| `smplx_body_pose` | SMPLX身体姿态 | (21,3) float32 |
| `smplx_jaw_pose` | 下颌姿态 | (1,3) float32 |
| `smplx_leye_pose` / `smplx_reye_pose` | 眼部姿态 | (1,3) float32 |
| `smplx_left_hand_pose` / `smplx_right_hand_pose` | 手部姿态 | (15,3) float32 |
| `smplx_shape` | SMPLX形状参数 | (11,) float32 |
| `smplx_transl` | SMPLX平移 | (3,) float32 |
| `smplx_gender_id` | 性别ID | scalar |

### 3.4 目录结构要求（参考 preprocess_bedlam.py）
预处理后每个scene目录下应有：
```
scene_name/
  ├── rgb/        # PNG图像
  ├── depth/      # NPY深度文件
  ├── mask/       # PNG掩码（可选）
  ├── cam/        # NPZ相机参数
  └── smpl/       # PKL SMPL标注
```

---

## 四、本阶段交付物清单

1. **数据构建脚本**：输入 AvatarReX 源数据，输出 Human3R 可训练的数据结构（可复用）
2. **数据清单与 manifest**：记录每个样本的来源，可追溯
3. **运行说明**：命令行示例 + 参数说明

---

## 五、工作边界（重要）

1. **不做**：模型结构改动、模型代码修改
2. **不做**：方法创新验证（只做数据准备）
3. **格式冲突解决原则**：当发现格式问题时，以第3节A类参考文件为准

---

## 六、关键文件索引

| 用途 | 文件路径 |
|------|----------|
| 训练数据集类定义 | `Human3R/src/dust3r/datasets/bedlam.py` |
| 数据预处理脚本 | `Human3R/datasets_preprocess/preprocess_bedlam.py` |
| SMPL参数处理 | `Human3R/src/dust3r/smpl_model.py` |
| 训练配置入口 | `Human3R/config/trian_human3r.yaml` |
| 任务说明原文 | `/data/wangzheng/Movie3R-new/tasklist/final_ai_brief_for_movie3r.md` |

---

## 七、初步行动建议

1. 先精读上述4个参考文件，确认AvatarReX数据与BEDLAM格式的差异点
2. 设计数据转换脚本，实现 `A,A,B,B` 的4帧样本组织逻辑
3. 预留深度图目录（待Depth-Anything-3生成）
4. 生成manifest文件追踪每个样本来源
5. 提供最小可运行的转换+训练示例
