# V6 AnchorPoseAdapter

## 0. 文档定位

本文档记录 Movie3R 下一步实现版本：`V6 AnchorPoseAdapter`。

该版本不是继续强化旧 `ShotToken`，而是把镜头跳变处的外部局部匹配证据转成 `AnchorToken`，并让 decoder 结束后的 `pose token` 单独读取这些 AnchorTokens，最后只修正 `camera_pose`。

本文档用途：

- 作为下一步代码实现的设计说明。
- 重新评估当前 AnchorToken 携带的信息是否过多、过泛。
- 对比 Human3R 的 human token，明确 AnchorToken 应该模仿什么、不应该模仿什么。
- 明确第一版 smoke test 的最小实现范围。

当前代码状态：

| 项目 | 当前状态 |
|---|---|
| 当前训练接入 | `ShotTokenGenerator + LayerwisePoseShotAdapter` |
| 当前默认训练方向 | V5.1-LAST2，但已决定停止作为主线 |
| 新主线 | V6 AnchorPoseAdapter |
| 旧 ShotToken | 不再作为主方案 |
| V5 layerwise shot adapter | 不再继续扩展 |
| 新 token | AnchorToken，来自局部静态背景匹配 |
| 新 adapter | decoder 后 pose-only attention + pose residual |
| 影响范围 | 只影响 `camera_pose` |
| 不影响范围 | image tokens、human tokens、pointmap、SMPL body |

相关文件：

| 文件 | 作用 |
|---|---|
| `src/dust3r/model.py` | 当前 ARCroco3DStereo 主模型，后续接入 V6 的位置 |
| `src/dust3r/shot_adaptation.py` | 当前 ShotToken/LoRA 模块，后续可新增 AnchorPoseAdapter 模块 |
| `src/dust3r/datasets/avatarrex.py` | 后续加入 anchor cache-aware loading |
| `scripts/batch_generate_rich_guitar_anchor_cache.py` | 当前 guitar AnchorToken cache 生成脚本 |
| `scripts/prototype_rich_anchor_tokens.py` | 当前 AnchorToken prototype 和 leave-one-out 验证脚本 |
| `docs/movie3r/shot_token_v6_plan.md` | 旧 V6 实验记录和历史方案，保留用于追溯 |

## 1. 核心结论

V6 应该采用如下方向：

```text
Human3R frozen backbone / decoder / heads
    + offline or runtime static scene anchors
    + AnchorTokenProjector
    + decoder-after PoseAnchorAttention
    + bounded PoseDeltaHead
    -> corrected camera_pose only
```

最重要的工程约束：

```text
AnchorTokens 不进入主 decoder sequence。
AnchorTokens 不被 image tokens 读取。
AnchorTokens 不被 human tokens 读取。
AnchorTokens 不直接进入 pointmap head。
AnchorTokens 只由 pose token 在 decoder 结束后读取。
```

这和旧 ShotToken 的区别：

| 方面 | 旧 ShotToken | V6 AnchorPoseAdapter |
|---|---|---|
| token 数量 | 1 个 global token | K 个 local anchor tokens |
| token 语义 | 当前帧和上一帧整体差异 | 具体静态背景 patch pair 的重定位证据 |
| 空间锚点 | 无明确锚点 | 有 ref/current patch 位置 |
| 匹配关系 | 无明确对应 | 有 ref patch 和 cur patch 对应 |
| 接入位置 | 曾经进入 decoder 或 layerwise pose path | decoder 结束后，只给 pose token 读 |
| 影响范围 | 容易污染 pose/image/human/pointmap | 只改 camera pose |
| no-op 安全性 | 依赖 q gate，历史上不稳定 | anchor 缺失或 gate 低时严格 no-op |

当前结论：AnchorToken 方向是合理的，但第一版 token payload 必须收窄。不要把所有 cache 字段都作为模型输入。cache 可以存很多调试字段，但模型读入的内容必须是 typed、局部、可解释、可消融的 re-anchor evidence。

## 2. 为什么不继续旧 ShotToken

旧 ShotToken 的问题不是“完全没有信号”，而是“语义太泛、权限太大”。

旧版本中 `q_t` 来自相邻帧 decoder image token 的全局池化差异：

```text
g_curr = mean(F_curr)
g_prev = mean(F_prev)
q_t = MLP([g_curr, g_prev, g_curr - g_prev, cosine])
```

这个 token 能区分跳变，但它不知道：

```text
当前帧哪个 patch 应该接回历史帧哪个 patch。
哪些区域是静态背景。
哪些匹配是可靠的。
局部重定位 residual 在哪里。
```

如果把这种 global token 放进 decoder full attention，它就可能同时影响：

```text
pose token
image tokens
human tokens
pointmap head
SMPL branch
recurrent state
```

这和近期失败现象一致：base Human3R 正常，开启 shot adaptation 后出现尺度、pointmap、人体分支污染。

V6 的核心变化是把问题拆开：

| 职责 | V6 中的承担者 |
|---|---|
| 判断是否有可靠跨镜头重叠 | anchor cache / quality gate |
| 提供局部对应证据 | AnchorTokens |
| 汇总局部证据并决定 camera correction | pose token + PoseAnchorAttention |
| 保护 reconstruction 和 human branch | adapter 接在 decoder 后，pose-only |

## 3. Human3R human token 到底携带什么

评估 AnchorToken 前，需要先看 Human3R 的 human token 真实携带什么。当前代码里 human token 不是一个泛泛的“人体类别 token”，也不是直接塞入 GT SMPL 参数，而是一个明确锚定到人体局部位置的 prompt。

### 3.1 human token 的生成路径

当前主路径在 `src/dust3r/model.py` 中：

```text
RGB image
  -> CUT3R encoder tokens feat [B, N, 1024]
  -> MHMR / DINOv2 backbone feat_mhmr [B, N_mhmr, 1024]
  -> MHMR detection score 找人体中心 / head-like patch
  -> 从 MHMR feature map 取 central token: smpl_tk_mhmr [B, Nh, 1024]
  -> 从 CUT3R feature map 对应位置取 central token: smpl_tk_cut3r [B, Nh, 1024]
  -> concat([smpl_tk_mhmr, smpl_tk_cut3r]) [B, Nh, 2048]
  -> downstream_head.mlp_fuse
  -> human token / smpl_query [B, Nh, 768]
  -> [pose, image, human] 进入 recurrent decoder
```

关键代码位置：

| 位置 | 内容 |
|---|---|
| `src/dust3r/model.py:372-379` | 初始化 MHMR/DINOv2 backbone，得到 `backbone_dim=1024` |
| `src/dust3r/model.py:1311-1424` | `smpl_tokenizer_mhmr`，做人检测、offset、取 MHMR central token |
| `src/dust3r/model.py:1426-1506` | `smpl_tokenizer_cut3r`，在 CUT3R patch grid 中取对应 token |
| `src/dust3r/model.py:1508-1533` | `token_fuse`，把 MHMR token 和 CUT3R token 融合到 768 维 |
| `src/dust3r/model.py:1105-1113` | decoder token 顺序为 `[pose, image, human]` |
| `src/dust3r/model.py:1689-1710` | decoder 后取 refined human token，并拼回 MHMR token 给 SMPL heads |
| `src/dust3r/heads/dpt_head.py:420-433` | SMPL heads 用 refined human token 预测 pose/shape/cam/expression |

### 3.2 human token 的信息组成

当前 `output_mode='pts3d+pose+smpl'` 主路径下，一个 human token 主要包含：

| 信息 | 是否在 human token 中 | 来源 | 说明 |
|---|---|---|---|
| 人体局部外观特征 | 是 | MHMR/DINOv2 central token | 来自人体检测位置附近的视觉 token |
| 3D/scene context 特征 | 是 | CUT3R central token | 同一位置的 CUT3R scene token |
| 人体空间锚点 | 是 | `pos_cut3r` / `smpl_pos_i` | human token 在 decoder 中有对应位置 |
| 人体检测 score | 间接使用 | `mlp_classif` | 用于选择 token，不直接作为主要语义向量 |
| offset refined 位置 | 间接使用 | `mlp_offset` | 用于定位中心点，再取 token |
| GT SMPL pose/shape/transl | 否 | 训练 label | 不直接塞进 token，由 head 预测 |
| 完整人体 mask | 否 | 可选 msk head | 不作为 human token 主输入 |
| 全图 global image diff | 否 | 无 | human token 是局部对象 prompt |

decoder 后，SMPL heads 实际看到的是：

```text
refined human token H' [B, Nh, 768]
    + original MHMR token [B, Nh, 1024]
    -> smpl_token [B, Nh, 1792]
    -> decpose / decshape / decexpression

refined human token H' 的前 768 维
    -> deccam
```

这里有一个关键点：Human3R 保留了原始 MHMR 人体先验作为 SMPL head 的输入，但这个先验仍然是人体局部视觉 token，不是 GT SMPL 参数。

### 3.3 human token 为什么可以进入 decoder

human token 能进入 decoder，原因不是“它是 human 这个类别”，而是它满足几个条件：

| 条件 | human token 是否满足 |
|---|---|
| 有明确对象 | 是，一个 token 对应一个人 |
| 有明确局部锚点 | 是，来自检测到的人体中心 patch |
| 有任务专用强先验 | 是，MHMR/DINOv2 human prior |
| 有明确下游消费者 | 是，SMPL heads |
| 信息不是泛泛全局控制 | 是，主要是局部人相关 prompt |
| 模型主体经过原训练适配 | 是，Human3R 原本就这样训练 |

这说明 V6 不能简单说“human token 可以进 decoder，所以 anchor token 也可以进 decoder”。AnchorToken 是新加的，原 decoder 没有见过它，且它服务 camera alignment，不服务 image reconstruction 或 SMPL。因此第一版必须比 human token 更保守。

## 4. AnchorToken 应该模仿 human token 的哪部分

AnchorToken 应该模仿的是 human token 的“局部锚点 + 专家先验 + 轻量融合”思想，而不是模仿“把 token 放进 decoder full attention”。

对照关系：

| Human3R human token | Movie3R V6 AnchorToken |
|---|---|
| 一个 token 对应一个人 | 一个 token 对应一个静态背景 match |
| MHMR/DINOv2 提供人体先验 | XFeat + mesh/fundamental 提供匹配先验 |
| CUT3R token 提供 scene context | Human3R/CUT3R encoder token 提供内部表示 |
| head/pelvis patch 作为空间锚点 | ref/current patch pair 作为空间锚点 |
| mlp_fuse 融合成 768 维 prompt | AnchorTokenProjector 融合成 768 维 evidence token |
| decoder 和 SMPL heads 消费 human token | pose token 和 PoseDeltaHead 消费 AnchorToken |
| 服务人体参数恢复 | 服务 camera pose re-anchor |

AnchorToken 不应该模仿的是：

```text
直接作为普通 token 拼进 [pose, image, human] decoder sequence。
让 image tokens / human tokens 自由读取 AnchorToken。
让 AnchorToken 同时修 camera、pointmap、human、state。
```

第一版更安全的类比是：

```text
Human token: local human evidence -> SMPL branch
AnchorToken: local scene re-anchor evidence -> pose/camera branch
```

## 5. 当前 AnchorToken 设计是否合理

### 5.1 当前 cache / prototype 携带的信息

当前 Step3 prototype 中保存过更完整的结构：

| 字段 | shape | 语义 |
|---|---|---|
| `key_cur_feature` | `[K, 1024]` | current patch 的 encoder token |
| `value_ref_feature` | `[K, 1024]` | reference patch 的 encoder token |
| `ref_pos_norm` | `[K, 2]` | reference patch 归一化位置 |
| `cur_pos_norm` | `[K, 2]` | current patch 归一化位置 |
| `delta_uv_norm` | `[K, 2]` | current - reference 的 2D 位移 |
| `confidence` | `[K]` | mesh error / fundamental inlier 形成的置信度 |
| `encoder_cosine` | `[K]` | Human3R encoder token 相似度 |
| `mesh_error_px` | `[K]` | mesh-verified reprojection error |

当前 high-overlap cache 存得更轻：

| 字段 | shape | 语义 |
|---|---|---|
| `ref_patch_idx` / `cur_patch_idx` | `[K]` | patch index，用于训练时在线 gather encoder tokens |
| `ref_patch_xy` / `cur_patch_xy` | `[K, 2]` | patch grid 坐标 |
| `ref_pos_norm` / `cur_pos_norm` | `[K, 2]` | normalized 2D position |
| `delta_uv_norm` | `[K, 2]` | current - reference |
| `confidence` | `[K]` | match 置信度 |
| `mesh_error_px` | `[K]` | teacher 验证误差 |
| `fundamental_inlier` | `[K]` | epipolar inlier 标记 |
| `affine_forward` / `affine_inverse` | `[2, 3]` | sample-level coarse affine |
| `quality_gate` | `[1]` | sample-level 是否启用 AnchorPoseAdapter |

这个 cache 作为离线数据是合理的，因为它保留了调试、筛选、ablation 需要的信息。但模型第一版不应该把所有字段都喂进去。

### 5.2 是否携带信息太多

结论：cache 信息不算太多，模型输入如果照单全收就太多。

需要区分三层信息：

| 层级 | 可多存吗 | 可直接喂模型吗 | 说明 |
|---|---|---|---|
| cache/debug 字段 | 可以 | 不一定 | 方便复现、筛选、画图 |
| gate/weight 字段 | 可以 | 有限制 | 可作为 sample/token 权重，但不一定作为 token content |
| token content 字段 | 必须收窄 | 是 | 只保留模型实际需要的 re-anchor evidence |

不建议第一版直接作为 token content 的字段：

| 字段 | 原因 | 推荐用法 |
|---|---|---|
| `mesh_error_px` raw value | 依赖 RICH mesh teacher，最终推理未必有 | 只用于训练权重、cache 筛选、quality gate |
| `fundamental_inlier` raw bool | 过于离散，容易变成规则 shortcut | 可参与 confidence，不作为单独强特征 |
| `affine_forward` / `affine_inverse` per-token repeat | sample-level 信息重复进每个 token 会放大其影响 | 作为 sample-level coarse prior 或 residual 计算依据 |
| camera id / frame id / sequence id | 数据集 shortcut，容易过拟合 | 禁止进入模型 |
| GT camera residual | 直接泄漏目标 | 禁止进入 AnchorToken |
| full-frame pooled image diff | 回到旧 global ShotToken 问题 | 禁止进入 AnchorToken |

可以保留给模型的最小信息：

| 信息 | 建议保留 | 原因 |
|---|---|---|
| current patch encoder token | 是 | 让 token 和当前 pose token / current scene 对齐 |
| reference patch encoder token | 可选，建议做 ablation | 提供 match identity，但可能增加 appearance shortcut |
| current normalized position | 是 | 告诉 pose token 证据来自当前视野哪里 |
| reference normalized position | 是或转成 residual | 表达应该接回哪里 |
| local residual after affine | 是，优先 | 表达局部 correction，不只是全局 affine |
| confidence | 是 | 让低质量 anchor 权重小 |
| token mask | 是 | 支持 padding 和缺失 anchor |
| sample quality gate | 是 | 低 overlap 时严格 fallback |

### 5.3 是否语义太泛

结论：现在的 AnchorToken 比旧 ShotToken 明确得多，但还需要在实现上保持 typed token，不要退化成 global prompt。

AnchorToken 的语义应该是：

```text
这个 current 静态背景 patch，和 reference 中这个 patch 是同一个局部区域。
在 coarse affine re-anchor 后，它还需要这个 local residual。
这个证据有 confidence，pose token 可以决定是否采用。
```

AnchorToken 不应该表达：

```text
这里发生了 shot change。
当前 frame 整体应该怎么动。
pointmap 应该怎么变。
human 应该怎么变。
recurrent state 应该怎么重置。
```

判断它不是泛泛 token 的证据来自 Step4 specificity controls：

| 对照 | 现象 | 说明 |
|---|---|---|
| correct token 优于 affine-only | strong sample 成立 | token 提供局部 residual evidence |
| shuffled value 退化 | strong sample 成立 | key-value 绑定有意义 |
| wrong boundary 退化 | strong sample 成立 | 不是泛泛“有 anchor”就行 |
| weak sample 不稳定 | 也成立 | 需要 gate/fallback |

因此当前 AnchorToken 思路合理，下一步不是增加更多信息，而是控制信息边界和影响范围。

## 6. V6 AnchorPoseAdapter 总体结构

### 6.1 数据流

第一版只处理 AABB boundary：

```text
views = [A@t, A@t+1, B@t+2, B@t+3]
boundary = A@t+1 -> B@t+2
ref_view_idx = 1
cur_view_idx = 2
```

训练时数据流：

```text
AvatarReX_AABB sample
  -> load anchor cache for start frame
  -> returns anchor indices / positions / confidence / quality_gate
  -> model encodes all 4 views normally
  -> gather F_ref[ref_patch_idx], F_cur[cur_patch_idx]
  -> AnchorTokenProjector builds anchor_tokens [B, K, 768]
  -> recurrent decoder runs original [pose, image, human] path
  -> final pose token z_out reads anchor_tokens
  -> PoseDeltaHead outputs bounded delta camera pose
  -> only res['camera_pose'] is updated
```

如果没有 cache、anchor 数量不足、quality gate 低、或当前不是 boundary frame：

```text
anchor_mask = all false
anchor_gate = 0
delta_pose = 0
camera_pose_final = camera_pose_base
```

### 6.2 模块划分

建议新增或重构为三个小模块：

| 模块 | 输入 | 输出 | 职责 |
|---|---|---|---|
| `AnchorTokenProjector` | gathered patch features + anchor geometry | `anchor_tokens [B,K,768]` | 把 typed raw anchor fields 投影成 token |
| `PoseAnchorAttention` | `z_out [B,1,768]`, `anchor_tokens [B,K,768]` | `anchor_context [B,1,768]` | pose token 单独读取 anchor evidence |
| `PoseDeltaHead` | `z_out`, `anchor_context`, gate | `camera_pose_final`, info | 输出 bounded camera residual |

建议 API：

```python
anchor_tokens, anchor_info = self.anchor_token_projector(
    ref_feat=ref_anchor_feat,          # [B, K, 1024]
    cur_feat=cur_anchor_feat,          # [B, K, 1024]
    ref_pos_norm=ref_pos_norm,         # [B, K, 2]
    cur_pos_norm=cur_pos_norm,         # [B, K, 2]
    local_residual_norm=residual_norm, # [B, K, 2]
    confidence=confidence,             # [B, K]
    anchor_mask=anchor_mask,           # [B, K]
)

pose_context = self.pose_anchor_attention(
    pose_token=z_out,                  # [B, 1, 768]
    anchor_tokens=anchor_tokens,       # [B, K, 768]
    anchor_mask=anchor_mask,           # [B, K]
)

camera_pose, pose_anchor_info = self.pose_delta_head(
    pose_token=z_out,
    anchor_context=pose_context,
    pose_base=res["camera_pose"],
    quality_gate=quality_gate,         # [B, 1]
)
```

### 6.3 推荐 token content

第一版推荐把 raw cache 字段分成三类。

模型必须使用：

| 字段 | shape | 来源 | 说明 |
|---|---|---|---|
| `cur_feat` | `[B,K,1024]` | 当前 encoder token gather | 当前视图中的局部证据 |
| `cur_pos_norm` | `[B,K,2]` | cache | 当前 patch 位置 |
| `ref_pos_norm` | `[B,K,2]` | cache | reference patch 位置 |
| `local_residual_norm` | `[B,K,2]` | `ref_pos - affine_inverse(cur_pos)` | affine 后的局部 residual |
| `confidence` | `[B,K]` | cache | token 权重 |
| `anchor_mask` | `[B,K]` | collate/padding | 支持可变 K |
| `quality_gate` | `[B,1]` | cache | sample-level fallback |

建议作为 ablation 加入：

| 字段 | 用途 | 风险 |
|---|---|---|
| `ref_feat` | 判断 ref/current visual match identity | 可能增加外观 shortcut 和过拟合 |
| `delta_uv_norm` | 直接表达 2D displacement | 与 ref/cur/residual 冗余 |
| `encoder_cosine` | 表达内部 token similarity | 可能变成低维 shortcut |

不建议进入模型：

| 字段 | 原因 |
|---|---|
| `mesh_error_px` | teacher-only / dataset-specific |
| `fundamental_inlier` | 应并入 confidence，不单独强输入 |
| `camera_id` / `frame_id` / `sequence_name` | shortcut |
| GT camera delta | target leakage |
| full image global feature diff | 回到旧 ShotToken |

### 6.4 local residual 的定义

cache 中已有 `affine_inverse`，它表示从 current normalized position 回到 reference normalized position 的 coarse 2D transform。

对第 k 个 anchor：

```text
base_ref_k = affine_inverse(cur_pos_norm_k)
local_residual_k = ref_pos_norm_k - base_ref_k
```

AnchorToken 不应该只存 `delta_uv = cur - ref`，因为 `delta_uv` 混合了全局视角变化和局部 residual。更推荐把 coarse affine 和 local residual 分开：

```text
affine_inverse: sample-level coarse re-anchor
local_residual: per-token local correction evidence
```

这样和 Step3 prototype 的结论一致：strong sample 中 `affine + local residual` 优于 `affine-only`。

## 7. Pose-only 接入方式

### 7.1 不进入 decoder sequence

V6 第一版不要修改 `_decoder()` 的 token sequence。

保持原始 Human3R：

```text
f_img = [pose token, image tokens, human tokens]
```

不要做：

```text
f_img = [pose token, image tokens, human tokens, anchor tokens]
```

原因：当前 decoder 是 frozen 的，没训练过 anchor token。直接加入新 token 即使 scale 很小，也可能通过 attention 改变 image/human token 和 pointmap head。

### 7.2 decoder 后只修 camera pose

接入点建议在 `_forward_impl()` 中 `_downstream_head()` 之后：

```text
dec[-1][:, 0:1] -> z_out
res = self._downstream_head(...)
res['camera_pose'] = self.anchor_pose_adapter(z_out, anchor_tokens, res['camera_pose'])
```

这保证：

```text
pointmap 已由原 head 产生，不被 AnchorToken 影响。
SMPL 已由原 human token path 产生，不被 AnchorToken 影响。
image tokens 和 human tokens 没有读 AnchorToken。
只有 camera_pose 被 bounded residual 修正。
```

### 7.3 residual 形式

推荐第一版只做 bounded residual：

```text
t_final = t_base + gate * scale_t * max_delta_t * tanh(delta_t)
q_final = normalize(q_base + gate * scale_q * max_delta_q * tanh(delta_q))
```

参数建议：

| 参数 | 初始值 | 说明 |
|---|---|---|
| `anchor_pose_scale` | `0.0` | 初始严格等于 base Human3R |
| `max_delta_t` | `0.25` 或更小 | 先保守，避免大幅拉尺度 |
| `max_delta_q` | `0.05` | 小角度 residual |
| `quality_gate` | cache 给定 | 低质量样本 fallback |
| `anchor_mask` | dataset/collate 给定 | 没有 token 时 no-op |

第一轮 smoke 可先只修 translation：

```text
camera_pose[:3] += bounded delta_t
camera_pose[3:7] 保持 base
```

如果 translation-only 稳定，再启用小幅 quaternion residual。

## 8. Dataset 和 cache-aware loading

### 8.1 第一版 sample 选择

当前 high-overlap guitar cache：

```text
/data/wangzheng/iJCV-CODE/data/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1
```

当前统计：

| 指标 | 数值 |
|---|---|
| cached samples | 185 |
| skipped samples | 0 |
| top_k_tokens | 16 |
| mean quality_gate | 0.792 |
| mean unique anchor patch pairs | 122.74 |
| quality_gate min/max | 0.711 / 0.834 |
| camera pairs | `1-2,3-4,4-5,5-6,6-7` |

第一版训练不要直接覆盖完整 `AvatarReX_AABB 20,720` samples。建议只采样 cache-covered high-overlap samples，先验证 forward/backward 和 pose-only correction。

### 8.2 Dataset 返回字段

建议 `AvatarReX_AABB` 在 cache-aware mode 下额外返回：

```text
anchor_valid: bool
anchor_ref_view_idx: int, default 1
anchor_cur_view_idx: int, default 2
anchor_ref_patch_idx: [K]
anchor_cur_patch_idx: [K]
anchor_ref_pos_norm: [K, 2]
anchor_cur_pos_norm: [K, 2]
anchor_local_residual_norm: [K, 2]
anchor_confidence: [K]
anchor_quality_gate: [1]
anchor_mask: [K]
```

collate 后：

```text
anchor_ref_patch_idx: [B, Kmax]
anchor_cur_patch_idx: [B, Kmax]
anchor_ref_pos_norm: [B, Kmax, 2]
anchor_cur_pos_norm: [B, Kmax, 2]
anchor_local_residual_norm: [B, Kmax, 2]
anchor_confidence: [B, Kmax]
anchor_quality_gate: [B, 1]
anchor_mask: [B, Kmax]
```

### 8.3 crop / resize 对齐风险

cache patch index 必须和训练时 encoder patch grid 对齐。

第一版必须固定：

```text
human3r_size = 512
patch_size = 16
resolution / crop policy = cache 生成时一致
aug_crop = disabled or fixed
random resolution = disabled
```

如果训练中仍使用随机 resolution/crop，cache 中的 patch index 会指向错误位置，AnchorToken 会变成噪声。

## 9. 训练 loss 和开关

### 9.1 必须关闭的旧分支

V6 主线中应关闭：

| 旧模块 / loss | V6 处理 |
|---|---|
| `ShotTokenGenerator` | 不训练，不调用 |
| `LayerwisePoseShotAdapter` | 不训练，不调用 |
| `PoseLoRALayer` | 关闭 |
| `HumanLoRALayer` | 关闭 |
| `WorldLoRALayer` | 关闭 |
| `shot_loss_weight` | 设 0 或不计算 |
| `shot_q0_loss_weight` | 设 0 或不计算 |
| `shot_noop_loss_weight` | 设 0 或替换为 anchor no-op |
| `shot_pointmap_keep_loss_weight` | 第一版不需要，因为 pointmap 不被改 |
| `shot_pose_residual_loss_weight` | 可替换为 anchor pose residual regularization |

### 9.2 推荐训练目标

第一版仍以已有主任务 loss 为主：

```text
L = L_task
  + lambda_delta * ||delta_pose||
  + lambda_no_anchor * (1 - anchor_valid) * ||delta_pose||
```

其中 `L_task` 继续使用已有 camera/pointmap/SMPL 监督，尤其是 AABB boundary camera pose 相关 loss。

由于 V6 不改 pointmap/human branch，第一版不需要额外 pointmap preservation loss。若后续发现 camera pose 修正和 pointmap 坐标系不一致，再单独加 camera/world consistency loss。

### 9.3 no-op 约束

V6 的 no-op 应该比旧 ShotToken 简单：

```text
if anchor_valid == False or quality_gate < threshold:
    delta_pose = 0
```

实现上用 gate 保证：

```text
effective_gate = anchor_quality_gate * has_any_anchor * learnable_scale
```

当没有 anchor 时，`effective_gate=0`，即使 adapter 内部输出非零，也不会改变 camera pose。

## 10. 推荐实现步骤

### Step 1: 新增配置

新增 guitar-only config，例如：

```text
config/train_v6_anchor_pose_guitar.yaml
```

配置要求：

| 项 | 推荐值 |
|---|---|
| `pretrained` | `/data/wangzheng/iJCV-CODE/Movie3R/src/human3r_896L.pth` |
| dataset root | `/data/wangzheng/iJCV-CODE/data/RICH_4Human3R` |
| dataset | cache-covered `AvatarReX_AABB` first |
| batch size | 1 |
| num workers | 0 |
| resolution | fixed 512-compatible |
| freeze | new `anchor_pose_adaptation` or reuse `shot_adaptation` carefully |
| old shot losses | 0 |

### Step 2: Dataset cache-aware loading

在 `AvatarReX_AABB` 增加 cache-aware mode：

```text
anchor_cache_root=/data/wangzheng/iJCV-CODE/data/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1
anchor_cache_only=True
anchor_top_k=16
```

第一版可以只返回 cache-covered samples，不做全量 fallback。等 smoke test 稳定后，再支持 cache missing 时 `anchor_valid=False`。

### Step 3: 新增 AnchorPoseAdapter 模块

建议放在 `src/dust3r/shot_adaptation.py` 或新文件 `src/dust3r/anchor_pose_adapter.py`。

最小模块：

```text
AnchorTokenProjector
PoseAnchorAttention
AnchorPoseDeltaHead
```

初始化要求：

```text
delta head 最后一层 zero init
learnable scale init 0
LayerNorm all projected features
confidence/gate clamp to [0, 1]
```

### Step 4: model.py 接入

接入点：

```text
_forward_impl()
  -> after _downstream_head()
  -> before _attach_shot_info() / append res
```

接入逻辑：

```text
if self.enable_anchor_pose_adapter and 'camera_pose' in res:
    anchor_tokens = build from feat[ref_view_idx], feat[cur_view_idx], cache indices
    res['camera_pose'], anchor_info = self.anchor_pose_adapter(z_out, anchor_tokens, res['camera_pose'])
    res.update(anchor_info)
```

注意：`feat` 是 encoder 1024 维 token，`z_out` 是 decoder 768 维 pose token。AnchorTokenProjector 负责把 1024 维 patch token 和 geometry 投影到 768 维。

### Step 5: freeze mode

新增 freeze mode 更清晰：

```text
freeze='anchor_pose_adaptation'
```

训练参数只包括：

```text
anchor_token_projector
pose_anchor_attention
anchor_pose_delta_head
```

冻结：

```text
encoder
decoder
state decoder
pose retriever
downstream heads
MHMR/DINO backbone
old shot modules
human/world/pose LoRA
```

### Step 6: smoke test

先做 1 GPU / batch size 1 / 20-50 iterations：

检查项：

| 检查 | 期望 |
|---|---|
| forward | 无 shape error |
| backward | loss finite |
| trainable params | 只包含 anchor pose modules |
| no-anchor sample | camera_pose 完全等于 base 或 delta=0 |
| gate low sample | delta 接近 0 |
| pointmap | 不被 adapter 直接修改 |
| smpl outputs | 不被 adapter 直接修改 |
| anchor delta norm | 有日志，且受 `max_delta` 限制 |

## 11. 推荐 ablation

为了回答“AnchorToken 信息是否太多或太泛”，必须做 payload ablation。

| 实验 | token content | 目的 |
|---|---|---|
| A0 | affine-only，无 AnchorToken | coarse prior baseline |
| A1 | geometry-only: `cur_pos`, `ref_pos`, `local_residual`, `confidence` | 看几何 anchor 是否足够 |
| A2 | A1 + `cur_feat` | 看当前 encoder feature 是否提供额外选择能力 |
| A3 | A2 + `ref_feat` | 看 ref/current appearance match 是否进一步提升 |
| A4 | A3 + shuffled value control | 验证不是泛泛 prompt |
| A5 | A3 + wrong boundary control | 验证 boundary specificity |
| A6 | A3 but no quality gate | 验证 gate 的必要性 |

推荐第一轮实现 A1 和 A2：

```text
A1 如果已经有效，说明局部几何 correspondence 是主信息。
A2 如果进一步提升，说明 Human3R encoder feature 对 local residual aggregation 有帮助。
A3 暂时不要作为默认，防止 token payload 过宽。
```

## 12. 与 Human Token 的最终对比结论

human token 和 AnchorToken 的共同点：

| 共同点 | 说明 |
|---|---|
| 都是局部 prompt | human token 锚定人，AnchorToken 锚定静态背景 match |
| 都依赖专家先验 | human 用 MHMR，anchor 用 XFeat/mesh/fundamental |
| 都融合主模型 token | human 融合 CUT3R token，anchor gather Human3R/CUT3R encoder token |
| 都压到 decoder 维度 | human 用 `mlp_fuse` 到 768，anchor 用 projector 到 768 |
| 都有明确下游任务 | human 给 SMPL，anchor 给 camera pose |

关键差异：

| 差异 | human token | AnchorToken |
|---|---|---|
| 原模型是否训练过 | 是 | 否，新加模块 |
| 是否可进主 decoder | 原路径可以 | 第一版不可以 |
| 目标分支 | SMPL/person branch | pose/camera branch |
| 是否应该影响 pointmap | 由原模型学习决定 | 第一版不直接影响 |
| token 数量 | 每人一个，通常少 | 每 boundary 多个，K=8/16 |
| 风险 | 人体检测错会影响 SMPL | anchor 错会误修 camera |

因此 AnchorToken 的设计原则是：

```text
像 human token 一样具体。
不像旧 ShotToken 一样全局。
不像普通 decoder token 一样有全权限。
```

## 13. 当前风险和处理策略

| 风险 | 原因 | 处理 |
|---|---|---|
| cache patch index 与训练 crop 不对齐 | cache 固定 512 crop，训练随机 resolution/crop | 第一版固定 resolution/crop |
| weak overlap anchor 误导 pose | anchor 少或 affine residual 大 | quality gate fallback |
| token payload 太强，模型走 shortcut | 输入 mesh_error/camera id/GT residual | 禁止这些字段进入 token |
| ref_feat 引入 appearance overfit | 跨镜头 appearance 不稳定 | 先 geometry-only / cur_feat ablation |
| camera pose 修了但 pointmap 坐标不一致 | 第一版只改 camera_pose | 先评估 camera loss，后续再做 consistency |
| high-overlap cache coverage 小 | 当前只有 185 samples | 第一版作为 smoke/ablation，不作为正式训练规模 |
| depth 数据有空值 | AABB 部分 views depth nonzero 为 0 | 第一版不要依赖这些 depth 做 anchor token content |

## 14. V6 第一版定义

第一版 V6 AnchorPoseAdapter 的硬定义：

```text
Input:
    frozen Human3R encoder/decoder outputs
    cache-provided static scene anchor correspondences

AnchorToken:
    per-token local re-anchor evidence
    not a global shot label
    not a camera residual label

Adapter:
    decoder-after
    pose-token query only
    anchor-token key/value only
    bounded camera pose residual only

No-op:
    no anchors or low quality gate -> exact base Human3R camera_pose

Frozen:
    encoder, decoder, downstream heads, MHMR/DINO, pointmap branch, SMPL branch

Trainable:
    AnchorTokenProjector
    PoseAnchorAttention
    PoseDeltaHead
```

一句话版本：

```text
V6 用多个局部静态背景匹配 token，给最终 pose token 提供受控 re-anchor evidence，只修 camera pose，不再让 shot/anchor 信息污染 reconstruction 和 human branch。
```
