# Shot-Aware Adaptation 实现计划

## 概述

实现 Section 25 的三个改动，让 Human3R 学会处理镜头跳变（AABB 数据）。

---

## Part 1: 模型结构改动

### 1.1 新增模块文件

**文件**: `src/dust3r/model.py` 或新建 `src/dust3r/blocks/shot_adaptation.py`

#### 1.1.1 ShotTokenGenerator

```python
class ShotTokenGenerator(nn.Module):
    """Global Difference Token - 使用 decoder 维度特征"""
    def __init__(self, dec_dim=768):
        # V1: g_curr, g_prev, diff, sim = 3 * dec_dim + 1
        self.shot_mlp = nn.Sequential(
            nn.Linear(dec_dim * 3 + 1, 256),
            nn.GELU(),
            nn.Linear(256, dec_dim),
        )
        self.q_init = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

    def forward(self, feat_curr, feat_prev, i):
        # feat_curr, feat_prev: [B, N, dec_dim]
        if i == 0:
            return self.q_init.expand(feat_curr.shape[0], -1, -1)
        g_curr = feat_curr.mean(dim=1)
        g_prev = feat_prev.mean(dim=1)
        diff = g_curr - g_prev
        sim = F.cosine_similarity(g_curr, g_prev, dim=-1)
        x = torch.cat([g_curr, g_prev, diff, sim.unsqueeze(-1)], dim=-1)
        q_t = self.shot_mlp(x).unsqueeze(1)
        return q_t
```

**输入**: decoder 前的 F_t 特征（图像编码 token）
**输出**: q_t [B, 1, dec_dim]

#### 1.1.2 StateGate

```python
class StateGate(nn.Module):
    """S_tilde = alpha * S_prev + (1 - alpha) * S0"""
    def __init__(self, dec_dim=768):
        self.gate_mlp = nn.Sequential(
            nn.Linear(dec_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, q_t):
        # q_t: [B, 1, dec_dim]
        # alpha: [B, 1, 1]
        alpha = torch.sigmoid(self.gate_mlp(q_t))
        return alpha
```

**State Gate 计算**:
```python
# S_prev: [B, N_state, dec_dim]
# S0: [1, N_state, dec_dim] (复用 CUT3R/Human3R 原始 initial state)
# S0 不加入 optimizer
S0_expand = S0.expand_as(S_prev)
S_tilde = alpha * S_prev + (1 - alpha) * S0_expand
```

**S0 的来源**: 复用 CUT3R/Human3R 原始的 initial state（从预训练权重加载），不随机初始化，不加入 optimizer。

#### 1.1.3 LoRA Heads（已确认格式）

**LoRAPoseHead**
- 输入: z'_t [B,1,dec_dim], q'_t [B,1,dec_dim], pose_base [B,7]
- 输出: corrected pose [B,7]
- **确认**: pose_base 格式为 trans(3) + quat(4)，共 7D

**LoRAHumanHead**
- 输入: H'_t [B,N_humans,dec_dim], q'_t [B,1,dec_dim], pred_smpl_dict
- 输出: corrected smpl dict
- **确认**: SMPL dict 实际字段为 `smpl_shape`(10D), `smpl_transl`(3D), `smpl_rotmat`(6,3,3), `smpl_expression`(10D)
- **注意**: 使用 `pred_smpl_dict.copy()` 避免 inplace 修改

**LoRAWorldHead**
- 输入: F'_t [B,H,W,dec_dim], z'_t [B,1,dec_dim], q'_t [B,1,dec_dim], world_base
- 输出: corrected world points
- **确认**: world_base 格式为 [B,H,W,3]（DPT 深度图格式），不是 BxNx3
- **注意**: 需要 global average pool 处理空间维度

**gamma 初始化**: 建议初始化为 0，让模型从零开始学习何时应用修正。

#### 1.1.4 Token 提取位置（已确认）

**Decoder 输出 token 顺序（来自 _forward_impl）**：
```python
# dec[dec_depth] 是最终输出
pose_token = dec[dec_depth][:, 0:1]     # z' = 第一个 token
img_tokens = dec[dec_depth][:, 1:-n_humans_i]  # F' = 中间 tokens
smpl_token = dec[dec_depth][:, -n_humans_i:]   # H' = 最后 n_humans 个 tokens
```

**q_out 提取位置**：
```python
# 当前 shot adaptation 模式下，q' 在 decoder 输出末尾
q_out = dec[-1][:, -1:]  # 最后一个 token
```

**注意**: 当前的 decoder 输出结构是 [z', F', H']，q' 还没有插入。当 q_t 插入 decoder 后，结构会变为 [z', F', H', q']。

### 1.2 ARCroco3DStereo 修改

**文件**: `src/dust3r/model.py`

#### 1.2.1 __init__ 中创建模块实例

```python
# 在 ARCroco3DStereo.__init__ 中添加:
self.shot_token_generator = ShotTokenGenerator(dec_dim=self.dec_dim)
self.state_gate = StateGate(dec_dim=self.dec_dim)
# S0 复用 CUT3R/Human3R 原始 initial state（从预训练权重加载）
# 不随机初始化，不加入 optimizer
# LoRA modules 挂在 model 层，forward 后统一处理
self.lora_pose = LoRAPoseHead(dec_dim=self.dec_dim)
self.lora_human = LoRAHumanHead(dec_dim=self.dec_dim)
self.lora_world = LoRAWorldHead(dec_dim=self.dec_dim)
# LoRA 启用标志
self.enable_shot_adaptation = False
```

**LoRA 归属**: LoRA modules 挂在 model 层，在 model forward 结束后统一处理输出。不要混写到 head 内部。

#### 1.2.2 _decoder 修改

修改 `_decoder` 方法签名，增加 `f_shot` 参数:

```python
def _decoder(self, f2d, state, H_c, f_shot=None, **kwargs):
    # 将 f_shot (shot token) 与其他 token 一起传入
    # token 排列: [z, F_t, H_t, q_t]，q_t 在最后
    ...
```

#### 1.2.3 _forward_impl 修改

在循环中预计算 q_tokens 并应用 State Gate:

```python
# 伪代码
# q_tokens 使用 decoder 输入的 image tokens: F_dec[i] 和 F_dec[i-1]
q_tokens = []
for i in range(num_views):
    if i == 0:
        q_tokens.append(self.shot_token_generator(F_dec[i], F_dec[i], i=0))
    else:
        q_tokens.append(self.shot_token_generator(F_dec[i], F_dec[i-1], i))
    # 注意: F_dec 是 decoder 输入的图像 token，不是 output token

# State Gate
S0 = self.initial_state  # 复用 CUT3R 原始 initial state
for i in range(num_views):
    if i > 0:
        alpha = self.state_gate(q_tokens[i])
        S_tilde = alpha * S_prev + (1 - alpha) * S0_expand
    else:
        S_tilde = S0  # 第一帧不用 gate
```

**Decoder token 排列**: [z, F_t, H_t, q_t]
**q_out 提取**: `q_out = tokens[:, -1:]`（取最后一个 token）
**q_tokens 来源**: F_dec[i] = decoder 输入的 image token，F_dec[i-1] =前一帧 decoder 输入 image token

### 1.3 Model 层 LoRA 处理

**文件**: `src/dust3r/model.py`

LoRA modules 挂在 model 层，在 forward 结束后统一处理:

```python
# 在 ARCroco3DStereo.forward 或 _forward_impl 末尾:
if self.enable_shot_adaptation and q_out is not None:
    # q_out = tokens[:, -1:]  从 decoder 输出提取
    # z_out, h_out, f_out  从 decoder 输出提取
    if 'pose' in pred:
        pred['pose'] = self.lora_pose(z_out, q_out, pred['pose'])
    if 'smpl' in pred:
        pred['smpl'] = self.lora_human(h_out, q_out, pred['smpl'])
    if 'pts3d' in pred:
        pred['pts3d'] = self.lora_world(f_out, z_out, q_out, pred['pts3d'])
```

**LoRA 启用条件**: `enable_shot_adaptation and q_out is not None`
- `enable_shot_adaptation`: 全局开关，训练/推理时设置
- `q_out is not None`: 确保 decoder 输出有效

**LoRA 归属**: LoRA 挂在 model，不在 head 内部。head 保持原结构不变。

### 1.4 freeze 选项

**文件**: `src/dust3r/model.py`

在 `set_freeze` 方法中添加:

```python
elif freeze == 'shot_adaptation':
    # 冻结 CUT3R/Human3R 所有原始模块
    self.set_freeze('encoder_and_decoder_and_head')  # encoder + decoder + base heads
    # 只训练新模块（以及 LoRA gamma 参数）
    for p in self.shot_token_generator.parameters(): p.requires_grad = True
    for p in self.state_gate.parameters(): p.requires_grad = True
    # S0 继续冻结（复用原始 initial state，第一版不训练）
    # LoRA heads
    for p in self.lora_pose.parameters(): p.requires_grad = True
    for p in self.lora_human.parameters(): p.requires_grad = True
    for p in self.lora_world.parameters(): p.requires_grad = True
```

**冻结内容**: encoder / decoder / base heads / S0（原始 initial state）
**训练内容**: ShotTokenGenerator / StateGate / LoRA heads / gamma parameters
**S0**: 不加入 optimizer，复用原始 initial state

#### 1.5 两种模式的行为定义

**模式 A: enable_shot_adaptation=False（默认，原 Human3R 路径）**

完全不生成、不插入 q_t，decoder token 顺序不变。

```
[z, F_t, H_t] + S_prev -> Decoder -> [z', F', H']
```

- 不调用 `shot_token_generator`
- 不调用 `state_gate`
- 不向 decoder 追加 q_t
- 不改变 decoder token 顺序
- 不调用 LoRA
- **完全等价原 Human3R forward**

**模式 B: enable_shot_adaptation=True（Shot Adaptation 路径）**

```
F_dec[i], F_dec[i-1] -> ShotTokenGenerator -> q_t
StateGate(q_t) -> alpha
S_tilde = alpha * S_prev + (1 - alpha) * S0
[z, F_t, H_t, q_t] + S_tilde -> Decoder -> [z', F', H', q']
q_out = tokens[:, -1:]
LoRA(z', q_out) / LoRA(h', q_out) / LoRA(f', q_out)
```

- 调用 `shot_token_generator` 生成 q_t
- 调用 `state_gate` 计算 alpha
- S_tilde 替代 S_prev 传入 decoder
- decoder token 顺序为 [z, F_t, H_t, q_t]
- q_out = tokens[:, -1:]
- LoRA 挂在 model 层处理

#### 1.6 freeze='shot_adaptation' 自动启用

在 `set_freeze` 或模型初始化时：

```python
if freeze == 'shot_adaptation':
    self.set_freeze('encoder_and_decoder_and_head')
    self.enable_shot_adaptation = True  # 自动启用
    # 只训练新模块
    for p in self.shot_token_generator.parameters(): p.requires_grad = True
    for p in self.state_gate.parameters(): p.requires_grad = True
    for p in self.lora_pose.parameters(): p.requires_grad = True
    for p in self.lora_human.parameters(): p.requires_grad = True
    for p in self.lora_world.parameters(): p.requires_grad = True
```

推理时也可通过 config/checkpoint 手动控制 `enable_shot_adaptation` 的开关。

---

## Part 2: 数据集改动

### 2.1 当前状态

AvatarReX 数据集 **目前没有返回** `shot_label` 和 `camera_label`。

当前返回字段（见 `avatarrex.py` 第 315-335 行）:
- `is_video`: False for AABB, True for Video
- 没有 `shot_label`
- 没有 `camera_label`

### 2.2 需要添加的字段

#### 2.2.1 shot_label

**定义**: `shot_label[i]` 表示 frame i-1 → frame i 是否发生 shot change

| 数据类型 | shot_label | 说明 |
|----------|------------|------|
| Video | [0, 0, 0, 0] | 无跳变 |
| AABB | [0, 0, 1, 0] | frame1→frame2 跳变 |

**实现**:

```python
# 在 AvatarReX_AABB._load_view 返回的 dict 中添加:
shot_label = np.array([0, 0, 1, 0], dtype=np.int64)  # AABB 固定格式

return dict(
    ...
    shot_label=shot_label,
)
```

对于 Video 数据集，shot_label 全部为 0。

#### 2.2.2 camera_label（可选，用于调试分析）

**定义**: `camera_label[i]` 表示第 i 帧来自哪个相机

| 数据类型 | camera_label | 说明 |
|----------|-------------|------|
| Video | [0, 0, 0, 0] | 同一相机 |
| AABB | [0, 0, 1, 1] | 前2帧相机A，后2帧相机B |

**实现**:

```python
# 在 AvatarReX_AABB._load_view 中:
camera_label = np.array([0, 0, 1, 1], dtype=np.int64)  # AABB

# 在 AvatarReX_Video._load_view 中:
camera_label = np.array([0, 0, 0, 0], dtype=np.int64)  # Video
```

**注意**: camera_label 主要用于分析，可以暂时不添加。

### 2.3 数据集修改清单

**文件**: `src/dust3r/datasets/avatarrex.py`

| 位置 | 修改内容 |
|------|---------|
| `AvatarReX_AABB._load_view()` return dict | 添加 `shot_label=np.array([0,0,1,0])` |
| `AvatarReX_Video._load_view()` return dict | 添加 `shot_label=np.array([0,0,0,0])` |

**注意**: 不需要修改 `AvatarReX_AABB._get_views()` 的 view_specs 逻辑，shot_label 是样本级别标签。

### 2.4 数据集是否需要添加 camera_id？

**不需要**。

当前 AABB 的 view_specs 已经是：
```python
view_specs = [
    (seqA_name, cam, t,  annots_t),   # view 0: 相机A @ t
    (seqA_name, cam, t1, annots_t1),  # view 1: 相机A @ t+1
    (seqB_name, cam, t2, annots_t2),  # view 2: 相机B @ t+2
    (seqB_name, cam, t3, annots_t3),  # view 3: 相机B @ t+3
]
```

相机信息隐含在 seqA_name vs seqB_name 中。

shot_label [0,0,1,0] 已经足够让模型知道第3帧（index 2）是跳变点。

---

## Part 3: 训练配置改动

### 3.1 train.yaml 修改

**文件**: `config/train.yaml`

#### 3.1.1 新增 shot_adaptation freeze 选项

```yaml
# 模型配置中修改:
model: ARCroco3DStereo(ARCroco3DStereoConfig(freeze='shot_adaptation',
  ...其他参数不变...))
```

#### 3.1.2 学习率配置（建议）

| 模块 | 学习率 | 说明 |
|------|--------|------|
| shot_token_generator | 1e-4 | 主训练 |
| state_gate | 1e-4 | 主训练 |
| lora_pose | 1e-4 | 主训练 |
| lora_human | 1e-4 | 主训练 |
| lora_world | 1e-4 | 主训练 |
| 其他 CUT3R 模块 | 0 | **冻结** |

**S0 不加入 optimizer**：S0 复用原始 initial state，不训练，不在 optimizer 中。

```python
# 学习率配置示例
optimizer_params = [
    {'params': model.shot_token_generator.parameters(), 'lr': 1e-4},
    {'params': model.state_gate.parameters(), 'lr': 1e-4},
    {'params': model.lora_pose.parameters(), 'lr': 1e-4},
    {'params': model.lora_human.parameters(), 'lr': 1e-4},
    {'params': model.lora_world.parameters(), 'lr': 1e-4},
    # S0 不加入 optimizer
    # 其他参数（encoder/decoder/heads）lr=0（冻结）
]
```

### 3.2 shot_label 在训练中的使用

**文件**: `src/dust3r/losses.py`

**第一版**: shot_label 仅用于 logging/debug，不参与 loss 加权。

```python
# 在 Regr3DPoseBatchList.compute_loss 中
# 第一版：仅记录，不加权
if 'shot_label' in gts[0]:
    shot_label = gts[0]['shot_label']
    # logging: 记录跳变帧的 loss 分布
    for i, is_shot in enumerate(shot_label[1:]):  # frame 1,2,3
        if is_shot:
            details[f"pose_loss_frame{i+1}_shot"] = float(pose_loss_per_frame[i])
        else:
            details[f"pose_loss_frame{i+1}_normal"] = float(pose_loss_per_frame[i])
```

**如果后续需要加权（必须 frame-wise，不能 sequence-wise）**:
```python
# 错误：给整段 sequence 加权
is_shot = shot_label[1:].any()  # [0,0,1,0] → True → 整段加权

# 正确：frame-wise 加权
for i in range(len(shot_label) - 1):
    if shot_label[i+1] == 1:  # frame i 是跳变帧
        loss[i] = loss[i] * 2.0  # 只加权这一帧
```

---

## Part 4: 实现顺序

### Step 1: 数据集（半天）

- [ ] `AvatarReX_AABB._load_view()` 添加 `shot_label`
- [ ] `AvatarReX_Video._load_view()` 添加 `shot_label`
- [ ] 验证数据集返回正确 shot_label

### Step 2: 模型基础结构（1天）

- [ ] 新建 `src/dust3r/blocks/shot_adaptation.py`
- [ ] 实现 `ShotTokenGenerator`
- [ ] 实现 `StateGate`
- [ ] 在 `ARCroco3DStereo.__init__` 中创建 shot_token_generator / state_gate 实例
- [ ] 修改 `_decoder` 接受 f_shot 参数
- [ ] 修改 `_forward_impl` 预计算 q_tokens（使用 F_dec[i] 和 F_dec[i-1]）
- [ ] 复用 `self.initial_state` 作为 S0（不新增参数）
- [ ] freeze='shot_adaptation' 选项

### Step 3: LoRA Heads（1天）

- [ ] `LoRAPoseHead` - pose_base 格式 Bx7 (trans3+quat4)
- [ ] `LoRAHumanHead` - smpl dict 字段: smpl_shape(10), smpl_transl(3), smpl_rotmat(6,3,3), smpl_expression(10)
- [ ] `LoRAWorldHead` - world_base 格式 BxHxWx3（DPT 格式）
- [ ] LoRA 挂在 model 层，在 forward 末尾统一处理（不在 head 内部）
- [ ] 验证 LoRA 输出维度正确

### Step 4: 训练调试（1-2天）

- [ ] train.yaml 配置 freeze='shot_adaptation'
- [ ] 设置各模块学习率
- [ ] 单卡测试 loss 正常回传
- [ ] 多卡训练
- [ ] 监控 q_token, alpha, S_tilde 的值分布
- [ ] 监控 LoRA 输出是否合理

### Step 5: Ablation（可选）

- [ ] freeze='encoder' 对比
- [ ] 有/无 LoRA 对比
- [ ] 有/无 StateGate 对比

---

## Part 5: 已确认格式

### 5.1 Pose 格式（已确认）

- **格式**: `Bx7` = trans(3) + quat(4)
- **来源**: `PoseDecoder.mlp` 输出，来自 `postprocess_pose` 处理后
- **字段名**: `camera_pose`

### 5.2 SMPL 格式（已确认）

- **smpl_shape**: (bs, max_humans, 10) - betas
- **smpl_transl**: (bs, max_humans, 3) - camera translation
- **smpl_rotmat**: (bs, max_humans, 6, 3, 3) - rotation matrix (from 6D rotvec)
- **smpl_expression**: (bs, max_humans, 10)
- **来源**: `postprocess_smpl(pred_smpl, self.depth_mode)` 返回

### 5.3 World pts3d 格式（已确认）

- **格式**: `BxHxWx3`（DPT 深度图格式，不是 BxNx3）
- **来源**: `postprocess(self_out, self.depth_mode, self.conf_mode)` 返回的 `pts3d`

### 5.4 Decoder Token 顺序（已确认）

```python
# dec[dec_depth] 最终输出结构：
# [pose_token, img_tokens..., smpl_tokens...]
#   pose_token = dec[dec_depth][:, 0:1]     # z'
#   img_tokens = dec[dec_depth][:, 1:-n_humans_i]  # F'
#   smpl_token = dec[dec_depth][:, -n_humans_i:]   # H'

# 当前 shot adaptation 模式下，q' 还未插入 decoder
# LoRA q_out = dec[-1][:, -1:]（最后一个 token）
```

---

## Part 6: 预期结果

### 6.1 成功标准

1. **训练正常**: loss 下降，无 NaN/Inf
2. **模块激活**: alpha 在 shot frame 相比 non-shot frame 有**明显不同分布**（不一定是接近 0）
3. **LoRA 生效**: LoRA 输出 delta 的 magnitude 合理（不是 0 也不是爆炸）
4. **默认模式等价原 Human3R**: `enable_shot_adaptation=False` 时完全不插入新模块，输出与原 Human3R 一致
5. **可切换**: freeze='none' 和 freeze='shot_adaptation' 都能正常训练

### 6.2 监控指标

| 指标 | 位置 | 含义 |
|------|------|------|
| `alpha` distribution | StateGate output | shot frame vs non-shot frame 有**不同分布**，而非绝对值接近 0 |
| `q_t` norm | ShotTokenGenerator | shot token 的强度 |
| `lora_delta` norm | LoRA output | 修正量大小 |
| `pose_loss_frame*_shot` | loss breakdown | 跳变帧的 pose loss（logging） |
| `pose_loss_frame*_normal` | loss breakdown | 正常帧的 pose loss（logging） |

### 6.3 回归验证

**验证默认模式等价原 Human3R**:

```python
# enable_shot_adaptation=False 时：
# - 不生成 q_t
# - 不使用 StateGate
# - 不改变 decoder token 顺序
# - 不启用 LoRA
# - 完全走原始 Human3R forward 路径

if not self.enable_shot_adaptation:
    # 等价原 Human3R forward
    pass
```

**验证 shot adaptation 模式**:

```python
# enable_shot_adaptation=True 时：
# - 生成 q_t
# - 使用 StateGate 调制 S_prev
# - decoder token 顺序为 [z, F_t, H_t, q_t]
# - q_out = tokens[:, -1:]
# - LoRA 生效
```

---

## 附录: 文件修改清单

| 文件 | 修改类型 | 改动量 |
|------|---------|--------|
| `src/dust3r/blocks/shot_adaptation.py` | 新建 | ~200 行 |
| `src/dust3r/heads/dpt_head.py` | 无修改 | 0 行（LoRA 在 model 层） |
| `src/dust3r/model.py` | 修改 | ~200 行（含 LoRA 处理逻辑） |
| `src/dust3r/datasets/avatarrex.py` | 修改 | ~10 行 |
| `src/dust3r/losses.py` | 修改 | ~20 行（可选）|
| `config/train.yaml` | 修改 | ~5 行 |

**总改动量**: ~450 行代码
**预计实现时间**: 2-3 人天
**LoRA 归属**: LoRA modules 在 model 层处理，不修改 head
