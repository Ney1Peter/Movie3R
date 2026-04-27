# Shot-Aware Adaptation 设计与实现

## 1. 概述

**目标**: 解决 Human3R/CUT3R 在镜头跳变时的重建偏移问题。

**核心原则**:
- 不修改 CUT3R 基模，冻结 encoder/decoder/base heads/原始 initial state S0
- 只训练新增轻量模块 (~1.3M 参数)

---

## 2. 模块架构

### 2.1 新增模块

| 模块 | 参数量 | 作用 |
|------|--------|------|
| `ShotTokenGenerator` | ~787K | 生成 shot token q_t |
| `StateGate` | ~99K | 控制状态混合比例 alpha |
| `LoRAPoseHead` | ~198K | 修正相机位姿 (trans+quat, 7D) |
| `LoRAHumanHead` | ~20K | 修正 SMPL 参数 (shape/transl) |
| `LoRAWorldGlobalShift` | ~197K | 全局平移修正点云 |
| **总计** | **~1.3M** | |

### 2.2 数据流

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

## 3. 模块详细设计

### 3.1 ShotTokenGenerator

**文件**: `src/dust3r/shot_adaptation.py`

**输入**: F_dec[i], F_dec[i-1] (decoder 输入的图像 token, [B,N,D])
**输出**: q_t [B,1,D]

**公式**:
```
g_curr = mean(feat_curr, dim=1)      # 全局平均池化
g_prev = mean(feat_prev, dim=1)
diff = g_curr - g_prev               # 差异特征
sim = cosine_similarity(g_curr, g_prev)  # 相似度
x = concat([g_curr, g_prev, diff, sim])  # [B, 3D+1]
q_t = MLP(x)  →  [B, 1, D]
```

**MLP 结构**: `Linear(3*D+1, 256) → GELU → Linear(256, D)`

**特殊情况**: i=0 时，返回可学习的 `q_init`

### 3.2 StateGate

**输入**: q_t [B,1,D]
**输出**: alpha [B,1,1] (0~1)

**公式**:
```
alpha = sigmoid(MLP(q_t))
S_tilde = alpha * S_prev + (1 - alpha) * S0_expand
```

**S0 来源**: 复用 `_init_state` 的初始 state，不训练

### 3.3 LoRA Heads

所有 LoRA 使用 residual 形式: `y_final = y_base + gamma * delta_y`
gamma 初始化为 0.01

#### LoRAPoseHead

- **输入**: z_token [B,1,D], q_out [B,1,D], pose_base [B,7] (trans+quat)
- **输出**: pose_final [B,7]
- **公式**: t_final = t_base + gamma * delta_t; q_final = normalize(q_base + gamma * delta_q)

#### LoRAHumanHead

- **输入**: smpl_token [B,N,D], q_out [B,1,D], pred_smpl_dict
- **输出**: smpl_final dict
- **第一版只修改**: smpl_shape (10D) 和 smpl_transl (3D)，不修改 rotmat

#### LoRAWorldGlobalShift

- **输入**: img_tokens [B,N,D], pose_token [B,1,D], q_out [B,1,D], world_base [B,H,W,3]
- **输出**: world_final [B,H,W,3]
- **注意**: 内部做全局平均池化，只修全局偏移

---

## 4. 模型修改

### 4.1 ARCroco3DStereo.__init__

```python
self.shot_token_generator = ShotTokenGenerator(dec_dim=self.dec_embed_dim)
self.state_gate = StateGate(dec_dim=self.dec_embed_dim)
self.lora_pose = LoRAPoseHead(dec_dim=self.dec_embed_dim)
self.lora_human = LoRAHumanHead(dec_dim=self.dec_embed_dim)
self.lora_world_global_shift = LoRAWorldGlobalShift(dec_dim=self.dec_embed_dim)
self.enable_shot_adaptation = False  # 默认关闭
```

### 4.2 _decoder 修改

**签名变更**: 增加 `f_shot=None` 参数

```python
def _decoder(self, ..., f_shot=None, ...):
    ...
    if f_shot is not None:
        f_img = torch.cat([f_img, f_shot], dim=1)  # q_t 插入末尾
        pos_shot = torch.zeros_like(f_shot)[:, :, :2]
        pos_img = torch.cat([pos_img, pos_shot], dim=1)
```

**Token 顺序**: [z, F_t, H_t] → [z, F_t, H_t, q_t] (当 f_shot is not None)

### 4.3 _forward_impl 修改

#### 预计算 q_tokens (循环前)

```python
if self.enable_shot_adaptation:
    f_dec = [self.decoder_embed(f) for f in feat]
    q_tokens = []
    for i in range(len(views)):
        if i == 0:
            q_tokens.append(self.shot_token_generator(f_dec[0], f_dec[0], i=0))
        else:
            q_tokens.append(self.shot_token_generator(f_dec[i], f_dec[i-1], i))
    S0 = init_state_feat  # 复用，不训练
```

#### 应用 StateGate (循环中，reset 优先)

```python
reset_mask_frame = views[i].get("reset", None)
if self.enable_shot_adaptation and i > 0:
    S0_expand = S0.expand_as(state_feat)
    if reset_mask_frame is not None and reset_mask_frame.any():
        state_for_recurrent = S0_expand  # reset 优先
    else:
        alpha = self.state_gate(q_tokens[i])
        state_for_recurrent = alpha * state_feat + (1 - alpha) * S0_expand
```

#### 传递 f_shot 给 decoder

```python
f_shot = q_tokens[i] if self.enable_shot_adaptation else None
new_state_feat, dec, _ = self._recurrent_rollout(..., f_shot=f_shot)
```

#### 提取 tokens 并应用 LoRA

```python
if self.enable_shot_adaptation:
    q_out = dec[-1][:, -1:]  # 最后一个 token 是 q'
    z_out = dec[-1][:, 0:1]

    if 'camera_pose' in res:
        res['camera_pose'] = self.lora_pose(z_out, q_out, res['camera_pose'])
    if n_humans_i > 0 and 'smpl_shape' in res:
        res = self.lora_human(h_token, q_out, res)
    if 'pts3d_in_self_view' in res:
        res['pts3d_in_self_view'] = self.lora_world_global_shift(...)
```

### 4.4 freeze='shot_adaptation'

```python
elif freeze == "shot_adaptation":
    freeze_all_params(to_be_frozen["encoder_and_decoder_and_head"])
    freeze_all_params(to_be_frozen["encoder"])
    freeze_all_params(to_be_frozen["mhmr"])
    freeze_all_params([self.downstream_head])
    freeze_all_params([self.masked_smpl_token, self.mhmr_masked_smpl_token])

    fix_all_params([
        self.shot_token_generator,
        self.state_gate,
        self.lora_pose,
        self.lora_human,
        self.lora_world_global_shift,
    ])
    self.enable_shot_adaptation = True
```

---

## 5. 数据集格式

### shot_label

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

---

## 6. 两种模式对比

| 模式 | enable_shot_adaptation=False | enable_shot_adaptation=True |
|------|------------------------------|----------------------------|
| q_t | 不生成 | 预计算后传入 decoder |
| StateGate | 不使用 | alpha 控制 S_tilde |
| LoRA | 不应用 | 修正 pose/human/world |
| 输出 | 等价原 Human3R | 修正后的输出 |

---

## 7. 已知限制

1. **World LoRA 简化**: 使用全局平均池化，只修全局偏移，不修局部几何
2. **q_t 位置**: 作为额外 token 插入 decoder，改变 token 序列长度

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `src/dust3r/shot_adaptation.py` | ShotTokenGenerator, StateGate, LoRA Heads |
| `src/dust3r/model.py` | 集成新模块、_decoder、_forward_impl、freeze |
| `src/dust3r/datasets/avatarrex.py` | shot_label 添加 |
