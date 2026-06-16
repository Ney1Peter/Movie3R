# V9 Model Architecture Details

这份文档专门记录 V9 在原版 Human3R 模型结构上增加了什么，以及 PPT 画模型图时建议使用的符号。

## 1. 总体改动

原版 Human3R 的核心流程可以简化成：

```text
image tokens F_t
+ pose token z_t
+ human tokens H_t
+ recurrent state S_t
  -> Human3R recurrent decoder
  -> refined pose / image / human tokens
  -> pose head 输出 camera pose
  -> human head 输出 SMPL / smpl_transl
```

V9 增加了一条 UniCon-style 的 relation correction branch：

```text
F_t, z_t, H_t, S_t, pose memory, previous correction state
  -> relation correction prompt builder Phi_corr
  -> A_corr,t

[z_t ; A_corr,t ; F_t ; H_t]
  -> Human3R recurrent decoder
  -> refined pose token, refined human token, refined A_corr,t
  -> pose latent correction
  -> human latent correction
```

核心思想是：不直接预测一个新的 camera matrix，也不直接手工移动 SMPL，而是通过新增 correction prompt token，让 decoder 自己学习“当前帧的人、场景、历史是否对齐”，然后输出 latent residual 去修正原 Human3R 的 pose token 和 human token。

## 2. Correct Token 的符号

建议 PPT 里使用：

```text
A_corr,t
```

表示第 `t` 帧的 correction prompt。更准确地说，它不是单个 token，而是一组 relation correction tokens：

```text
A_corr,t = [a_sem,t ; a_align,t ; a_mom,t]
```

其中：

| 符号 | 名称 | 作用 |
|---|---|---|
| `a_sem,t` | semantic relation token | 看当前视觉、人、pose 和历史 memory 是否一致 |
| `a_align,t` | pose alignment token | 看当前 pose latent 和上一帧 pose latent 的变化是否合理 |
| `a_mom,t` | temporal momentum token | 记录上一帧 correction / delta / gate 的时间先验 |

PPT 中可以把构造模块写成：

```text
Phi_corr
Attn + MLP + Gate
```

正式公式可以写成：

```text
A_corr,t = Phi_corr(
  F_t, z_t, H_t, S_t, M_t,
  A_corr,t-1, Delta z_t-1, g_t-1
)
```

符号说明：

| 符号 | 含义 |
|---|---|
| `F_t` | image tokens |
| `z_t` | pose token |
| `H_t` | human tokens |
| `S_t` | recurrent state memory |
| `M_t` | pose retriever memory |
| `A_corr,t-1` | previous refined correction prompt |
| `Delta z_t-1` | previous pose-token residual |
| `g_t-1` | previous correction gate |

## 3. A_corr,t 是怎么构造的

代码位置：

```text
src/dust3r/v8_pose_prompt.py
V82PoseRelationPrompt
```

当前主线使用 `v8_pose_prompt_variant='relation_v8_2'`，也就是 `V82PoseRelationPrompt`。

### 3.1 Semantic Token

输入：

```text
current context = image tokens + pose token + human tokens
memory context  = recurrent state + pose memory + previous corr token
```

先用 learned query 分别对当前 context 和 memory context 做 attention：

```text
current_token = Attention(query, current context)
memory_token  = Attention(query, memory context)
```

然后用 learned gate 做软融合：

```text
gamma_sem = sigmoid(MLP([current_token, memory_token]))

a_sem,t = gamma_sem * current_token
        + (1 - gamma_sem) * memory_token
```

这个 token 负责表达：当前图像、人、pose 和历史世界信息是否一致。

### 3.2 Alignment Token

输入：

```text
z_t
z_t-1
z_t - z_t-1
memory_token
```

经过 MLP：

```text
a_align,t = MLP([z_t, z_t-1, z_t - z_t-1, memory_token])
```

这个 token 负责表达：当前 pose token 相对历史 pose token 的变化是否像正常相机运动。

### 3.3 Momentum Token

输入：

```text
A_corr,t-1
Delta z_t-1
g_t-1
```

经过 MLP：

```text
a_mom,t = MLP([A_corr,t-1, Delta z_t-1, g_t-1])
```

这个 token 负责表达：上一帧是否修过、修了多少、当前帧是否应该延续这个修正趋势。

### 3.4 最终进入 decoder 的是什么

三个 token 构造完成后，会各自加 token type embedding，再过 LayerNorm：

```text
A_corr,t = LayerNorm([
  a_sem,t   + type_embed_sem,
  a_align,t + type_embed_align,
  a_mom,t   + type_embed_mom
])
```

所以进入 decoder 的不是原始信息拼接，而是经过 attention、MLP、gate、type embedding、LayerNorm 后得到的一组 correction prompt tokens。

## 4. 为什么 A_corr,t 放在 pose token 后面

decoder 拼接顺序是：

```text
X_dec,t = [z_t ; A_corr,t ; F_t ; H_t]
```

代码位置：

```text
src/dust3r/model.py
_decoder()
```

把 `A_corr,t` 放在 pose token 后面，主要是工程上最稳：

1. 原 Human3R 默认第 0 个 token 是 pose token。
2. pose head、pose memory update、camera pose 解码都默认 `dec[-1][:, 0:1]` 是 pose token。
3. 如果把 correction token 放到最前面，会破坏原 Human3R 的 token 约定。
4. 放在 pose token 后面，既不破坏第 0 位 pose token，又方便 decoder attention 让 pose 和 correction prompt 直接交互。
5. 后续切 image / human token 时，只要显式跳过 corr token，就不会污染原 image/human head 输入。

所以这个位置不是理论上唯一可行的位置，但它是对原 Human3R 侵入最小、最稳定的接入方式。

## 5. Decoder 输出后怎么修 camera

decoder 后得到 refined correction prompt：

```text
A_tilde_corr,t
```

Pose Correct Head 输入 refined correction prompt，输出：

```text
Delta z_t_raw
g_t
```

其中：

```text
Delta z_t_raw = MLP(mean(A_tilde_corr,t))
drift_logit   = MLP(mean(A_tilde_corr,t))
g_t           = sigmoid(drift_logit)
Delta z_t     = g_t * Delta z_t_raw
```

然后修正 pose token：

```text
z_hat_t = z_tilde_t + Delta z_t
```

最后仍然使用原 Human3R pose head：

```text
T_hat_t = PoseHead(z_hat_t)
```

重点：

- Pose Correct Head 不直接输出 4x4 camera pose。
- 它输出的是 pose token latent residual。
- 原 Human3R pose head 负责把 corrected pose token 解码成 camera pose。

## 6. Decoder 输出后怎么修 human

当前主线使用的是 implicit human latent correction，不是显式 `smpl_transl += delta`。

Human Latent Correct Head 输入：

```text
refined human token H_tilde_t
refined correction prompt A_tilde_corr,t
corrected pose token z_hat_t
```

先构造 context：

```text
context = [H_tilde_t, mean(A_tilde_corr,t), z_hat_t]
```

然后输出 human latent residual：

```text
Delta H_t_raw = MLP(context)
```

human 分支本身也有 learned gate，但当前主线配置使用 shared gate：

```text
gate_mode = "shared"
```

也就是人体修正共享 pose branch 的 `g_t`：

```text
Delta H_t = g_t * Delta H_t_raw
H_hat_t   = H_tilde_t + Delta H_t
```

最后仍然使用原 Human3R human head：

```text
SMPL_hat_t = HumanHead(H_hat_t)
```

重点：

- Human Correct Head 不直接输出最终 SMPL。
- 它输出的是 human token latent residual。
- corrected human token 再交给原 Human3R human head 解码出 body pose、shape、expression、`smpl_transl` 等。
- 因此这比直接手写 `smpl_transl` 更接近 UniCon-style：修 latent，由原 head 解释 latent。

## 7. 两个 Head 是否都会输出 residual 和概率

更准确地说：

| 分支 | 输入 | 输出 residual | 输出 gate / probability | 当前是否主线使用 |
|---|---|---|---|---|
| Pose Correct Head | refined `A_corr,t` | `Delta z_t_raw` | `g_t = sigmoid(drift_logit)` | 是 |
| Human Latent Correct Head | refined human token + refined `A_corr,t` + corrected pose token | `Delta H_t_raw` | 有 learned gate，但主线用 shared pose gate | 是 |
| Explicit Human Trans Head | human token + corr token + pose token + `smpl_transl` | `Delta transl` | learned/shared gate | 诊断分支，当前主线关闭 |

当前配置里：

```text
v8_human_trans_corr = False
v8_human_latent_corr = True
v8_human_latent_corr_gate_mode = "shared"
```

所以当前真正生效的是：

```text
Pose branch 输出 g_t
Human branch 使用同一个 g_t 去控制 Delta H_t
```

## 8. Gate 的含义

`g_t` 可以理解成 drift / correction probability：

```text
g_t 低：当前帧基本保持原 Human3R 输出
g_t 高：当前帧可能发生 drift，需要较大 correction
```

它不是人工写死“前两帧不修、后两帧修”。当前设计是 learned gate，通过训练中的 drift/gate 监督学出来。

在长序列里，如果第 3 帧刚遇到跳变，第 4 帧又回到前两帧附近，可能说明：

1. 第 3 帧刚发生边界变化，history 还主要来自第 1、2 帧。
2. 第 4 帧已经接收到第 3 帧的 `A_corr / Delta z / gate` 信息。
3. Momentum token 会把上一帧 correction 状态带到当前帧。

这说明当前 correction branch 已有流式时间先验，但长序列边界行为仍需要专门评估和训练。

## 9. LoRA 在哪里

LoRA 不是 `A_corr,t` 本身。

`A_corr,t` 是新增 correction prompt branch。LoRA 是给原 Human3R head 加低秩残差，让原 head 在冻结主权重的情况下有小幅适配能力。

当前最新 pose-human-lora 配置：

```text
v8_pose_head_lora = True
v8_human_head_lora = True
v8_head_lora_rank = 8
v8_head_lora_alpha = 8.0
v8_head_lora_dropout = 0.0
```

所以：

```text
LoRA scaling = alpha / rank = 8 / 8 = 1.0
```

接入位置：

| LoRA | 作用位置 |
|---|---|
| pose head LoRA | `downstream_head.pose_head` |
| human head LoRA | `downstream_head.deccam / decpose / decshape / decexpression` |

LoRA 的作用是小幅微调原 pose/human head，而不是替代 correction branch。

## 10. PPT 推荐画法

推荐模型图分成三段。

第一段：构造 correction prompt

```text
F_t, z_t, H_t, S_t, M_t, A_corr,t-1, Delta z_t-1, g_t-1
        |
        v
Phi_corr: Attention + MLP + Gate
        |
        v
A_corr,t = [a_sem,t ; a_align,t ; a_mom,t]
```

第二段：进入 decoder

```text
X_dec,t = [z_t ; A_corr,t ; F_t ; H_t]
        |
        v
Human3R Recurrent Decoder
        |
        v
z_tilde_t, H_tilde_t, A_tilde_corr,t
```

第三段：两个 correction heads

```text
Camera branch:
A_tilde_corr,t
  -> Pose Correct Head
  -> Delta z_t, g_t

z_hat_t = z_tilde_t + g_t * Delta z_t_raw
T_hat_t = PoseHead(z_hat_t)
```

```text
Human branch:
H_tilde_t, A_tilde_corr,t, z_hat_t
  -> Human Latent Correct Head
  -> Delta H_t

H_hat_t = H_tilde_t + g_t * Delta H_t_raw
SMPL_hat_t = HumanHead(H_hat_t)
```

## 11. 最容易说错的点

1. 不要说 `A_corr,t` 是从 encoder 直接提取出来的 token。
   它是由 encoder/decoder-side tokens、memory、history 共同构造出来的 relation correction prompt。

2. 不要说 `A_corr,t` 是一个单 token。
   它当前是 3 个 relation tokens 组成的一组 prompt。

3. 不要说 camera branch 直接输出 camera delta。
   它输出 pose-token latent residual，camera pose 仍由原 Human3R pose head 输出。

4. 不要说 human branch 直接输出 SMPL delta。
   当前主线输出 human-token latent residual，SMPL 仍由原 Human3R human head 输出。

5. 不要把 LoRA 和 correction branch 混为一谈。
   Correction branch 负责生成 latent residual；LoRA 只是让原 head 有小幅可训练适配。

6. 不要说 gate 是人工指定后两帧修。
   当前 gate 是 learned drift/correction probability。

