# Shot-Aware Adaptation 实现文档

## 概述

本文档详细描述 Shot-Aware Adaptation 模块的实现设计，用于处理 Human3R/CUT3R 在镜头跳变时的重建偏移问题。

**核心原则**: 不修改 CUT3R 基模，并冻结 CUT3R/Human3R 原始参数，只训练新增轻量模块。

**冻结内容**: encoder / decoder / base heads / 原始 initial state S0
**训练内容**: ShotTokenGenerator / StateGate / LoRA heads / gamma parameters

---

## 1. 模块架构

### 1.1 三个新模块

| 模块 | 类型 | 位置 | 训练参数 | 作用 |
|------|------|------|----------|------|
| ShotTokenGenerator | nn.Module | model.py | ~0.79M | 生成 shot token q_t |
| StateGate | nn.Module | model.py | ~0.10M | 控制状态混合比例 |
| LoRA Pose/Human/World | nn.Module | model.py | ~0.45M | 对输出做微调修正 |

**总训练参数**: 1,301,788 (≈1.3M，已实际打印确认)

**参数估算**:
- ShotTokenGenerator: 788,480 ≈ 787K
- StateGate: 98,561 ≈ 99K
- LoRAPoseHead: 197,640 ≈ 198K
- LoRAHumanHead: 19,983 ≈ 20K (第一版只修 shape/transl)
- LoRAWorldGlobalShift: 197,124 ≈ 197K

### 1.2 数据流

```
# Shot Adaptation 启用时 (enable_shot_adaptation=True)

F_dec[i], F_dec[i-1]  →  ShotTokenGenerator  →  q_t [B,1,D]
                                                        ↓
                                              StateGate  →  α [B,1,1]
                                                        ↓
                                          S_tilde = α·S_prev + (1-α)·S0
                                                        ↓
                      [z, F_t, H_t, q_t] + S_tilde  →  Decoder
                                                        ↓
                                              [z', F', H', q'] + S_t
                                                        ↓
                                        q_out = tokens[-1:]  ← 最后一个 token
                                                        ↓
                        LoRA(z', q_out), LoRA(H', q_out), LoRA(pooled(F'), q_out)
                                                        ↓
                                              修正后的输出
```

---

## 2. 模块详细设计

### 2.1 ShotTokenGenerator

**文件**: `src/dust3r/shot_adaptation.py`

**输入**:
- `feat_curr`: [B, N, D] - 当前帧 decoder 输入特征 (F_dec[i])
- `feat_prev`: [B, N, D] - 上一帧 decoder 输入特征 (F_dec[i-1])
- `i`: int - 帧索引

**输出**:
- `q_t`: [B, 1, D] - shot token

**公式**:
```
g_curr = mean(feat_curr, dim=1)      # [B, D] 全局平均池化
g_prev = mean(feat_prev, dim=1)      # [B, D]
diff = g_curr - g_prev               # [B, D] 差异特征
sim = cosine_similarity(g_curr, g_prev)  # [B] 相似度

x = concat([g_curr, g_prev, diff, sim])  # [B, 3D+1]
q_t = MLP(x)  →  [B, 1, D]
```

**特殊情况**: i=0 时，返回可学习的 `q_init`（无 previous frame）

**MLP 结构**:
```
Linear(3*D+1, 256) → GELU → Linear(256, D)
```

---

### 2.2 StateGate

**文件**: `src/dust3r/shot_adaptation.py`

**输入**:
- `q_t`: [B, 1, D] - shot token

**输出**:
- `alpha`: [B, 1, 1] - 门控值 (0~1)

**公式**:
```
alpha = sigmoid(MLP(q_t))  → [B, 1, 1]
S_tilde = alpha * S_prev + (1 - alpha) * S0_expand
```

**MLP 结构**:
```
Linear(D, 128) → GELU → Linear(128, 1) → Sigmoid
```

**S0 来源**:
- S0 = `init_state_feat`（来自 `_init_state` 的初始 state）
- S0_expand = S0.expand_as(S_prev)
- S0 **不训练**，只复用原始初始状态

---

### 2.3 LoRA Heads

**文件**: `src/dust3r/shot_adaptation.py`

所有 LoRA 使用 residual 形式: `y_final = y_base + gamma * delta_y`

**gamma 初始化**: 0.01（工程建议值）

**注意**: 若 gamma=0，LoRA MLP 的梯度会被 gamma 乘成 0，训练初期学不到东西。0.01 是工程平衡值。

**token slicing 安全措施**: decoder 输出 token 的 slicing 逻辑封装在 `_slice_decoder_tokens` helper 函数中，避免直接写 slice 索引出错。

#### 2.3.1 LoRAPoseHead

**输入**:
- `z_token`: [B, 1, D] - decoder 输出的 pose token (z')
- `q_out`: [B, 1, D] - decoder 输出的 shot token (q')
- `pose_base`: [B, 7] - trans(3) + quat(4)

**输出**:
- `pose_final`: [B, 7] - trans(3) + quat(4)

**公式**:
```
x = concat([z_token, q_out])  → [B, 1, 2D]
delta = MLP(x)  →  [B, 1, 7]  →  delta.squeeze(1)  →  [B, 7]

t_base = pose_base[:, :3]      # [B, 3]
q_base = pose_base[:, 3:7]    # [B, 4]
delta_t = delta[:, :3]        # [B, 3]
delta_q = delta[:, 3:7]       # [B, 4]

t_final = t_base + gamma * delta_t
q_final = normalize(q_base + gamma * delta_q)

pose_final = concat([t_final, q_final])  →  [B, 7]
```

#### 2.3.2 LoRAHumanHead

**输入**:
- `smpl_token`: [B, N_humans, D] - decoder 输出的人体 token (H')
- `q_out`: [B, 1, D] - decoder 输出的 shot token (q')
- `pred_smpl_dict`: dict - SMPL 参数字典

**SMPL 字典实际字段**:
- `smpl_shape`: [B, N, 10] - betas
- `smpl_transl`: [B, N, 3] - camera translation
- `smpl_rotmat`: [B, N, 6, 3, 3] - rotation matrix (不直接修改)
- `smpl_expression`: [B, N, 10]

**输出**:
- `smpl_final`: dict - 同结构

**第一版只修改 transl 和 shape，不修改 rotmat**:

```python
q_expand = q_out.expand(-1, N, -1)  →  [B, N, D]
x = concat([smpl_token, q_expand])  →  [B, N, 2D]

delta_shape = Linear(x)  →  [B, N, 10]
delta_transl = Linear(x)  →  [B, N, 3]

out = pred_smpl_dict.copy()  # 不 inplace 修改
out['smpl_shape'] = pred_smpl_dict['smpl_shape'] + gamma_shape * delta_shape
out['smpl_transl'] = pred_smpl_dict['smpl_transl'] + gamma_transl * delta_transl
# rotmat / expression 保持不变
```

**rotmat 修正说明**: rotation matrix 直接相加后通常不再是合法旋转矩阵。第一版不修 rotmat，后续如需修应用 axis-angle / 6D rotation residual。

#### 2.3.3 LoRAWorldGlobalShift

**名称说明**: 当前实现本质上是给整张 pointmap 加一个全局 3D 平移 residual，能修全局 world alignment / camera offset，但不能修局部几何。

**输入**:
- `img_tokens`: [B, N, D] - decoder 输出的图像 token (F')，内部做全局平均池化
- `pose_token`: [B, 1, D] - decoder 输出的 pose token (z')
- `q_out`: [B, 1, D] - decoder 输出的 shot token (q')
- `world_base`: [B, H, W, 3] - DPT 输出的 pts3d

**输出**:
- `world_final`: [B, H, W, 3]

**公式**:
```
# 全局池化在内部做，失去空间信息，只修全局偏移
img_global = mean(img_tokens, dim=1, keepdim=True)  →  [B, 1, D]

x = concat([img_global, q_out])  →  [B, 1, 2D]
delta = MLP(x)  →  [B, 1, 3]
delta = delta.squeeze(1).unsqueeze(-1).unsqueeze(-1)  →  [B, 1, 1, 3]

world_final = world_base + gamma * delta  →  [B, H, W, 3]
```

---

## 3. 模型修改

### 3.1 ARCroco3DStereo.__init__

新增模块实例:
```python
self.shot_token_generator = ShotTokenGenerator(dec_dim=self.dec_embed_dim)  # D=768
self.state_gate = StateGate(dec_dim=self.dec_embed_dim)
self.lora_pose = LoRAPoseHead(dec_dim=self.dec_embed_dim)
self.lora_human = LoRAHumanHead(dec_dim=self.dec_embed_dim)
self.lora_world_global_shift = LoRAWorldGlobalShift(dec_dim=self.dec_embed_dim)
self.enable_shot_adaptation = False  # 默认关闭
```

### 3.2 _decoder 修改

**修改前签名**:
```python
def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_smpl, pos_smpl, use_ttt3r=False)
```

**修改后签名**:
```python
def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_smpl, pos_smpl, f_shot=None, use_ttt3r=False)
```

**f_shot 插入位置**: 在 f_smpl（人体 token）之后

**修改前 token 顺序**:
```
tokens = [z, F_t, H_t]  (f_smpl 在 F_t 之后)
```

**修改后 token 顺序** (当 f_shot is not None):
```
tokens = [z, F_t, H_t, q_t]  (q_t 在最后)
```

### 3.3 _forward_impl 修改

#### 3.3.1 循环前：预计算 q_tokens

```python
if self.enable_shot_adaptation:
    # F_dec[i] = decoder 输入的图像 token (经过 decoder_embed)
    f_dec = [self.decoder_embed(f) for f in feat]  # list of [B, N, D]
    q_tokens = []
    for i in range(len(views)):
        if i == 0:
            q_tokens.append(self.shot_token_generator(f_dec[0], f_dec[0], i=0))
        else:
            q_tokens.append(self.shot_token_generator(f_dec[i], f_dec[i-1], i))
    S0 = init_state_feat  # 复用，不训练
```

#### 3.3.2 循环中：应用 StateGate（reset 优先）

```python
# reset_mask 优先于 StateGate：reset=True 时跳过 blend，直接用 S0
reset_mask_frame = views[i].get("reset", None)
if self.enable_shot_adaptation and i > 0:
    S0_expand = S0.expand_as(state_feat)
    if reset_mask_frame is not None and reset_mask_frame.any():
        # reset 优先：跳过 blend，直接用 S0
        state_for_recurrent = S0_expand
    else:
        alpha = self.state_gate(q_tokens[i])  # [B, 1, 1]
        state_for_recurrent = alpha * state_feat + (1 - alpha) * S0_expand
else:
    state_for_recurrent = state_feat  # i==0 或关闭时使用原始 state
```

**注意**: reset 与 StateGate 的顺序很重要。reset 表示镜头跳变后需要回到初始状态，应优先于 StateGate 的混合计算。当 reset=True 时，直接使用 S0，不再进行 blend。

#### 3.3.3 循环中：传递 f_shot 给 decoder

```python
f_shot = q_tokens[i] if self.enable_shot_adaptation else None

new_state_feat, dec, _ = self._recurrent_rollout(
    state_for_recurrent,
    state_pos,
    feat_i,
    pos_i,
    pose_feat_i,
    pose_pos_i,
    smpl_feat_i,
    smpl_pos_i,
    init_state_feat,
    img_mask=views[i]["img_mask"],
    reset_mask=views[i]["reset"],
    update=views[i].get("update", None),
    f_shot=f_shot,  # 新增参数
)
```

#### 3.3.4 循环中：提取 tokens 并应用 LoRA

**Token Slicing Helper**: 使用 `_slice_decoder_tokens` 函数安全地提取 decoder 输出中的各个 token，避免直接写 slice 索引出错。

```python
# _slice_decoder_tokens(dec, n_humans, enable_shot_adaptation)
# 返回: (z_out, img_tokens, smpl_token, q_out)

# 当 enable_shot_adaptation=True 时:
# Token 顺序: [z', F', H', q'] → q' 在最后
z_out = dec[-1][:, 0:1]      # pose token
q_out = dec[-1][:, -1:]     # shot token

# 当 enable_shot_adaptation=False 时:
# Token 顺序: [z', F', H'] → 无 q_t
z_out = dec[-1][:, 0:1]
q_out = None
```

**LoRA 应用**:
```python
if self.enable_shot_adaptation and q_out is not None:
    z_out, img_tokens, h_token, q_out = self._slice_decoder_tokens(
        dec, n_humans_i, enable_shot_adaptation=True)

    # Pose LoRA
    if 'camera_pose' in res:
        res['camera_pose'] = self.lora_pose(z_out, q_out, res['camera_pose'])

    # Human LoRA
    if n_humans_i > 0 and 'smpl_shape' in res:
        res = self.lora_human(h_token, q_out, res)

    # World LoRA (全局平移)
    # img_tokens [B,N,D] 直接传入，LoRA 内部做 pooling
    if 'pts3d_in_self_view' in res:
        res['pts3d_in_self_view'] = self.lora_world_global_shift(
            img_tokens, z_out, q_out, res['pts3d_in_self_view'])
```

**base head 输入显式去掉 q_out**: head_input 的 slicing 使用 `[:-n_humans-1]` 显式排除 q_t（当 enable_shot_adaptation=True 时），确保 base head 收到的输入结构与原 Human3R 完全一致。

**当 enable_shot_adaptation=False 时**:
- 不生成 q_tokens
- f_shot = None
- 不改变 decoder token 顺序
- 不应用 LoRA
- **完全等价原 Human3R forward**

### 3.4 freeze='shot_adaptation'

```python
elif freeze == 'shot_adaptation':
    # 第一步：冻结全部参数
    for p in self.parameters():
        p.requires_grad = False

    # 第二步：只打开新模块
    for module in [
        self.shot_token_generator,
        self.state_gate,
        self.lora_pose,
        self.lora_human,
        self.lora_world_global_shift,
    ]:
        for p in module.parameters():
            p.requires_grad = True

    # S0 不训练，不进入 optimizer

    # 自动启用
    self.enable_shot_adaptation = True
```

**冻结内容**: encoder / decoder / base heads / S0（原始 initial state）
**训练内容**: ShotTokenGenerator / StateGate / LoRA heads / gamma parameters

---

## 4. 数据集格式

### 4.1 shot_label

**定义**: `shot_label[i]` 表示 frame i-1 → frame i 是否发生 shot change

| 数据类型 | shot_label | 说明 |
|----------|------------|------|
| Video | [0, 0, 0, 0] | 无跳变 |
| AABB | [0, 0, 1, 0] | frame1→frame2 跳变 |

**AABB 数据结构**:
```
view0: camA @ t      → frame 0
view1: camA @ t+1    → frame 1
view2: camB @ t+2    → frame 2 (shot change)
view3: camB @ t+3    → frame 3
```

**shot_label[2]=1** 表示 frame1→frame2 发生了相机跳变

---

## 5. 两种模式对比

### 5.1 enable_shot_adaptation=False（默认，原 Human3R 路径）

```
[z, F_t, H_t] + S_prev → Decoder → [z', F', H']
                                            ↓
                               不生成 q_t，不使用 StateGate
                                            ↓
                               不改变 token 顺序
                                            ↓
                               不应用 LoRA
                                            ↓
                               输出等价原 Human3R
```

### 5.2 enable_shot_adaptation=True（Shot Adaptation 路径）

```
F_dec[i], F_dec[i-1] → ShotTokenGenerator → q_t
                                            ↓
                                  StateGate → α
                                            ↓
                          S_tilde = α·S_prev + (1-α)·S0
                                            ↓
                    [z, F_t, H_t, q_t] + S_tilde → Decoder
                                            ↓
                                  [z', F', H', q']
                                            ↓
                              q_out = tokens[-1:]
                                            ↓
                    LoRA(z', q_out), LoRA(H', q_out), LoRA(F', q_out)
                                            ↓
                               修正后的输出
```

---

## 6. 回归验证

### 6.1 默认模式等价原 Human3R

当 `enable_shot_adaptation=False` 时:
- 不调用 ShotTokenGenerator
- 不调用 StateGate
- 不向 decoder 追加 q_t
- 不改变 decoder token 顺序
- 不调用 LoRA
- **完全走原始 Human3R forward 路径**

### 6.2 gamma=0 时输出不变

当 LoRA gamma=0 时:
- delta_y = 0（无论输入如何）
- y_final = y_base + 0 = y_base
- LoRA 不产生任何修正

---

## 7. 实现检查清单

### 7.1 已实现

- [x] ShotTokenGenerator 类
- [x] StateGate 类
- [x] LoRAPoseHead 类 (gamma=0.01)
- [x] LoRAHumanHead 类 (gamma=0.01, 只修正 shape/transl)
- [x] LoRAWorldGlobalShift 类 (gamma=0.01, 全局平移, 内部做 pooling)
- [x] model.py 导入新模块
- [x] __init__ 中创建实例
- [x] _decoder 接受 f_shot 参数
- [x] _decoder 在 f_smpl 后插入 q_t
- [x] _recurrent_rollout 传递 f_shot
- [x] _forward_impl 预计算 q_tokens
- [x] _forward_impl reset 优先于 StateGate
- [x] _forward_impl 传递 f_shot 给 decoder
- [x] _slice_decoder_tokens helper 函数
- [x] _forward_impl 应用 LoRA 修正
- [x] base head 输入显式去掉 q_out
- [x] freeze='shot_adaptation' 选项

### 7.2 待验证

- [x] 数据集添加 shot_label
- [ ] 训练配置 freeze='shot_adaptation'
- [ ] 单卡测试 loss 正常回传
- [ ] 多卡训练
- [ ] alpha 在 shot frame 的分布
- [ ] LoRA delta 的 magnitude

---

## 8. 已知限制

### 8.1 World LoRA 简化

当前 World LoRA 使用全局平均池化的图像 token，而非真正的空间感知特征:
- 简化版: `img_global = mean(img_tokens, dim=1)` → 失去空间信息
- 理想版: 需要从 DPT 输出特征中提取 HxWxD 格式

### 8.2 q_t 位置

q_t 作为额外 token 插入 decoder，会改变 decoder 的 token 序列长度，可能影响 attention 计算效率。

---

## 9. 文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/dust3r/shot_adaptation.py` | 新建 | ShotTokenGenerator, StateGate, LoRA Heads |
| `src/dust3r/model.py` | 修改 | 导入、__init__、_decoder、_forward_impl、freeze |
| `src/dust3r/datasets/avatarrex.py` | 待修改 | 添加 shot_label |
| `config/train.yaml` | 待修改 | freeze='shot_adaptation' |
