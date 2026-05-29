# Movie3R V8.1 UniCon-Style Implementation Plan

本文档只写执行计划，不开始改实现代码。目标是先把要改哪里、怎么改、怎么训练、怎么验证说清楚，确认后再动手。

## 1. 当前目标

V8.1 要做的不是 decoder 后的 pose 后处理，也不是单独挂一个 sidecar head，而是更接近 UniCon3R 的 decoder-in prompt：

```text
当前帧 image / pose / human tokens
+ recurrent state / pose memory
+ 新增 pose correction prompt: A_corr_t
  -> Human3R decoder
  -> refined pose token + refined A_corr_t
  -> residual head 输出 pose latent residual 和 gate
  -> corrected pose token
  -> 原 pose head
  -> corrected camera pose
```

核心是：`A_corr_t` 必须进入 decoder，和原来的 pose / image / human / state tokens 发生 attention 交互。它不是直接预测一个显式 `delta T` 后处理相机位姿。

## 2. 已读代码结论

### 2.1 Human3R 干净参考

参考文件：

```text
src/dust3r/model_human3r.py
```

关键流程：

```text
_forward_impl
  -> _encode_views_mhmr
  -> smpl_tokenizer_mhmr / smpl_tokenizer_cut3r
  -> token_fuse 得到 human tokens
  -> pose_retriever 得到当前 pose token
  -> _recurrent_rollout
  -> _decoder
  -> decoder 输出切片
  -> _downstream_head
```

`model_human3r.py` 中 decoder 的 token 顺序很清晰：

```text
[pose token, image tokens, human tokens]
```

然后 head 里默认把 `x[-1][:, 0]` 当作 pose token，送进 pose head 输出 camera pose。

### 2.2 当前实际要改的 model.py

目标文件：

```text
src/dust3r/model.py
```

这个文件已有 V2-V7 的残留：

```text
shot adaptation
anchor decoder tokens
anchor pose token adapter
v7 pose adapter
decoder 后 pose adapter / LoRA 注释残留
```

所以 V8.1 不能继续混在旧后处理逻辑里。计划是新增一个独立开关：

```text
enable_v8_pose_prompt
```

默认关闭。只有 `freeze="v8_pose_prompt"` 或显式配置时才启用。旧 V6/V7 分支先保留，不直接删除；如果必须修改旧逻辑，先注释旧路径，再新增 V8.1 路径。

### 2.3 head 的约束

文件：

```text
src/dust3r/heads/dpt_head.py
```

`DPTPts3dPoseSMPL.forward()` 里有硬约定：

```python
pose_token = x[-1][:, 0].clone()
pose = self.pose_head(pose_token)
```

因此第一版不改 pose head。V8.1 只在进入 downstream head 前，把 final decoder output 的第 0 个 pose token 替换成 corrected pose token。

### 2.4 loss 现状

文件：

```text
src/dust3r/losses.py
```

当前已有：

```text
Regr3DPose / Regr3D
V5 shot pose loss 残留
V7PosePseudoLoss
```

V8.1 不复用 V7 pseudo delta loss 作为主方案，因为 V8.1 不是 decoder 后显式 delta pose adapter。计划新增独立 loss 类，例如：

```text
V81PosePromptLoss
```

它监督最终 `pred["camera_pose"]` 接近 GT pose，同时监督 V8.1 branch 的 gate / residual 不要乱修。

### 2.5 AvatarReX_AABB 现状

文件：

```text
src/dust3r/datasets/avatarrex.py
```

当前 `AvatarReX_AABB` 已经能构造四帧：

```text
A_t, A_t+1, B_t+2, B_t+3
```

并且已经修过 SMPL transl 到相机坐标系的问题。当前缺口是：

```text
还没有按 A/B 相机夹角过滤样本
```

V8.1 dataloader 需要能优先选大角度 AABB，例如 `>= 60 deg`，必要时用 `90 deg` 或 `180 deg` 做更明显的 overfit 测试。

## 3. 模型改动计划

### 3.1 新增模块位置

建议新增：

```text
src/dust3r/v8_pose_prompt.py
```

第一版包含几个小模块，不拆太碎，避免一开始工程复杂：

```text
V81PosePromptConfig
V81BodyPartPromptBuilder
V81HistoryTokenizer
V81CameraMotionTokenizer
V81ReliabilityTokenizer
V81PoseCorrectionPrompt
V81PoseLatentResidualHead
```

后续如果稳定，再拆成目录：

```text
src/dust3r/modules/pose_prompt/
```

### 3.2 A_corr_t 的第一版组成

先做最小可行版本：

```text
A_corr_t =
  A_body_part_t
  + A_history_human_t
  + A_camera_motion_t
  + A_reliability_gate_t
```

含义：

| token | 当前第一版怎么来 | 为什么要它 |
|---|---|---|
| `A_body_part_t` | learned body-part queries 从当前 image/human tokens 读取 pelvis、torso、left foot、right foot 相关隐式 token | 对应我们已经验证过最强的四个人体 anchor |
| `A_history_human_t` | 上一帧 corrected anchors / body-part state，经 MLP token 化 | 给当前帧一个历史人体参照 |
| `A_camera_motion_t` | 上一帧 raw/corrected pose、pose memory、当前 coarse pose token，经 MLP token 化 | 给当前帧一个相机运动先验 |
| `A_reliability_gate_t` | human score、body-part token confidence、history consistency、上一帧 confidence，经 MLP token 化 | 判断当前是否该强 correction |

注意：当前帧 pointmap、当前帧 raw pose、当前帧 SMPL joints 都是 decoder/head 后才有，不能用于构造当前帧 decoder-in `A_corr_t`。它们可以用来做监督、日志和可视化。

### 3.3 token 插入顺序

建议 V8.1 使用显式 token layout：

```text
[pose, corr, image, anchor(optional old), human, shot(optional old)]
```

原因：

1. `pose` 仍然在第 0 位，方便兼容 pose head；
2. `corr` 紧跟 pose，切片和替换 pose token 都更直观；
3. image/human token 的相对含义保持清楚；
4. 后面若保留旧 anchor/shot 分支，也能通过 layout 统一管理。

需要把现有 `_decoder_token_layout()` 扩展为支持：

```text
n_corr
n_anchor
n_humans
has_shot
```

并用 helper 统一生成：

```text
pose_start / pose_end
corr_start / corr_end
img_start / img_end
anchor_start / anchor_end
human_start / human_end
shot_start / shot_end
```

### 3.4 decoder 修改点

修改 `src/dust3r/model.py::_decoder()`：

新增参数：

```python
f_corr=None
pos_corr=None
```

当 `enable_v8_pose_prompt=True` 时，拼接：

```text
token_parts = [f_pose, f_corr, f_img, ...]
pos_parts   = [pos_pose, pos_corr, pos_img, ...]
```

`pos_corr` 用 dummy 位置即可，类似 pose token：

```text
-1 或 0 的 2D position，dtype 和 pos_img 保持一致
```

### 3.5 forward 修改点

修改 `src/dust3r/model.py::_forward_impl()`：

每一帧循环里：

1. 正常得到当前 `feat_i`、`pose_feat_i`、`smpl_feat_i`、`state_feat`、`mem`；
2. 调用 V8 prompt builder 构造 `f_corr_i, pos_corr_i, corr_info_i`；
3. 把 `f_corr_i, pos_corr_i` 送进 `_recurrent_rollout()` / `_decoder()`；
4. 从 decoder final tokens 里切出：

```text
refined_pose_token = tokens[:, pose_start:pose_end]
refined_corr_token = tokens[:, corr_start:corr_end]
```

5. residual head：

```text
delta_pose_latent, gate = V81PoseLatentResidualHead(refined_corr_token)
corrected_pose_token = refined_pose_token + gate * delta_pose_latent
```

6. 进入 downstream head 前，把 head input 里的 pose token 替换成 `corrected_pose_token`；
7. `pose head` 原样输出 corrected `camera_pose`；
8. 在 `res` 里记录辅助信息：

```text
v8_pose_prompt_gate
v8_pose_prompt_delta_norm
v8_pose_prompt_corr_token
v8_pose_prompt_pose_token_raw
v8_pose_prompt_pose_token_corrected
```

这些只用于 loss、日志、debug，可按需 detach。

### 3.6 history/memory 更新

第一版 history 先保持简单：

```text
第 0 帧：
  没有上一帧 corrected history
  A_history_human_t 用 learned no-history token 或全零 token

第 1 帧以后：
  使用上一帧输出后缓存的信息
```

缓存内容：

```text
previous corrected pose token
previous predicted camera pose
previous human token summary
previous body-part prompt summary
previous gate
previous confidence
```

如果当前 batch 有 reset，则清空该样本 history。

第一版不强求把上一帧 SMPL joints 作为训练输入，因为当前主线要先验证 token-level decoder-in prompt。上一帧 SMPL / joints 可以作为可选显式 history ablation，不作为默认主路径。

## 4. 训练参数计划

### 4.1 新增 freeze 模式

新增：

```text
freeze="v8_pose_prompt"
```

行为：

```text
freeze Human3R / CUT3R encoder
freeze decoder
freeze downstream heads
freeze MHMR / SMPL heads
freeze 旧 V2-V7 adapters
train only:
  V81PoseCorrectionPrompt
  V81PoseLatentResidualHead
  可能还有 body-part query 参数
```

注意：decoder 参数冻结，但 decoder forward 仍然参与计算图，让 loss 梯度能回传到 `A_corr_t` 和 residual head。

### 4.2 第一版训练什么

第一版只训练：

```text
1. 构造 A_corr_t 的小 MLP / learned queries
2. refined A_corr_t -> pose latent residual 的 residual head
3. gate head
```

不训练：

```text
encoder
decoder
pose head
DPT head
human head
SMPL head
旧 shot/anchor/v7 adapters
```

这样可以最大程度保持 Human3R 原结构不被破坏。

## 5. Loss 设计计划

### 5.1 主 loss

使用 AvatarReX GT camera pose 监督最终 corrected camera pose：

```text
L_pose = lambda_t * L_translation + lambda_R * L_rotation
```

translation：

```text
SmoothL1(t_pred, t_gt)
```

rotation：

```text
geodesic(R_pred, R_gt)
或 quaternion distance
```

要注意和现有 Human3R 一样，以 view0 为 reference，使用相对 pose：

```text
T_gt_rel_i = inverse(T_gt_0) @ T_gt_i
T_pred_i = pred["camera_pose"]
```

具体编码需要和当前 `Regr3DPose.compute_pose_loss()` 对齐，避免坐标系/scale 写错。

### 5.2 V8 prompt 辅助 loss

建议第一版加轻量约束：

```text
L_residual_small = ||delta_pose_latent||^2
L_gate_sparse = mean(gate) 或 gate target loss
```

gate target 可以先用 AABB 结构生成弱标签：

```text
view0: 0
view1: 0
view2: 1   # A -> B 第一帧，最需要修
view3: 0/soft  # B -> B 可能仍需要轻微修，可设 0.3 或先不监督
```

第一版可以先不用复杂 gate label，只记录 gate，看它是否自然在 view2 变大。

### 5.3 可选 keep loss

因为 `A_corr_t` 进入 decoder 后可能影响 image/human tokens，虽然 decoder/head 冻结，但 token 交互可能扰动 pointmap / human 输出。

如果发现 scene/human 输出明显坏掉，再加：

```text
L_keep = frozen baseline 输出与 V8 输出保持接近
```

第一版可以先不加，减少训练复杂度。

### 5.4 初始总 loss

第一版建议：

```text
L = L_pose
  + 0.01 * L_residual_small
  + 0.1  * L_gate_sparse_or_gate_target
```

如果训练不稳定，再逐步加 `L_keep`。

## 6. Dataloader 计划

### 6.1 数据源

只使用：

```text
/data/wangzheng/iJCV-CODE/data/Avatarrex_output
```

格式沿用：

```text
AvatarReX_AABB
```

### 6.2 AABB 采样

四帧顺序：

```text
seqA frame t
seqA frame t+1
seqB frame t+2
seqB frame t+3
```

保留现有逻辑，不改成 `A_t, A_t+1, B_t, B_t+1`，因为当前代码和实验已经围绕 `t,t+1,t+2,t+3` 跑通过。

### 6.3 新增角度过滤

给 `AvatarReX_AABB` 加可选参数：

```text
min_view_angle_deg=60
max_view_angle_deg=None
max_samples=None
pair_strategy="all" / "fixed" / "top_angle"
manifest_path=None
```

角度计算建议：

```text
读取 seqA/seqB 对应 frame 的 camera c2w
取相机 forward/view direction
计算夹角 angle_deg
保留 angle_deg >= min_view_angle_deg
```

如果 c2w/w2c 方向不确定，先写 dataloader 检验脚本可视化相机朝向，确认后再用于训练。

### 6.4 固定小样本

先生成一个 manifest：

```text
output/v8_1_aabb_manifest/train_20.json
output/v8_1_aabb_manifest/val_8.json
```

每条记录包含：

```json
{
  "seqA": "...",
  "seqB": "...",
  "start_frame": 820,
  "frames": [820, 821, 822, 823],
  "view_angle_deg": 90.0
}
```

这样每次训练/可视化都能复现同一批 AABB。

### 6.5 Dataloader 检验脚本

新增脚本：

```text
scripts/v8_1_check_aabb_dataloader.py
```

输出：

```text
1. 打印选中的 A/B sequence、frames、angle
2. 检查 rgb/depth/mask/smpl/cam 是否完整
3. 保存 4 帧 montage
4. 保存相机 frustum 小图或 json
5. 保存 SMPL/mask bbox 对齐检查
```

这个脚本先跑通，再开始训练。

## 7. 小训练流程

### 7.1 Overfit sanity check

先只用 1-2 个 AABB 样本：

```text
train samples: 1 or 2
steps: 200 - 500
batch size: 1
freeze: v8_pose_prompt
```

目标：

```text
看 corrected pose loss 是否明显下降
看 gate 是否在 view2 或高误差帧变大
看 viewer 中 B 段相机是否被拉回
```

如果 1-2 个样本都不能 overfit，说明模型接入、loss 或坐标系有问题。

### 7.2 小批量训练

overfit 通过后：

```text
train: 20-50 个 AABB
val: 8-16 个 AABB
angle: >= 60 deg，必要时先 >= 90 deg
```

指标：

```text
raw pose translation / rotation error
corrected pose translation / rotation error
view2 boundary error
view3 continuation error
gate mean per view
delta latent norm per view
```

可视化：

```text
原版 Human3R viewer
V8.1 corrected viewer
raw camera 灰色叠加
corrected camera 彩色显示
```

## 8. 代码执行顺序

确认本文档后，建议按下面顺序提交，每一步单独 commit：

1. **docs/log commit**
   - 把本计划同步到 V8.1 log。
   - 显式目标：
     - V8.1 的主线目标明确写成 UniCon-style decoder-in prompt；
     - 文档中不再把 sidecar / decoder 后处理作为主方案；
     - 后续实现顺序、loss、dataloader 和验收标准都有对应记录。

2. **dataloader commit**
   - 给 `AvatarReX_AABB` 增加角度过滤和 manifest/fixed sample 支持；
   - 新增 `scripts/v8_1_check_aabb_dataloader.py`；
   - 不改模型。
   - 显式目标：
     - 能从 `/data/wangzheng/iJCV-CODE/data/Avatarrex_output` 中稳定选出 AABB 四帧样本；
     - 能筛选 `view_angle_deg >= 60` 的相机对；
     - 能保存固定 train/val manifest，保证每次训练样本一致；
     - 检验脚本能输出 montage、样本列表、角度信息和基础文件完整性检查。

3. **model module commit**
   - 新增 `src/dust3r/v8_pose_prompt.py`；
   - 只包含模块定义和简单 shape test；
   - 不接入主 forward。
   - 显式目标：
     - `A_corr_t` builder 能输入 image / pose / human / state / history 相关张量；
     - 输出 shape 为 `[B, n_corr, dec_dim]`；
     - residual head 能输出 `delta_pose_latent: [B, 1, dec_dim]` 和 `gate: [B, 1, 1]`；
     - 最小 shape test 通过，且不依赖当前帧 pointmap / raw pose / SMPL head 输出。

4. **model integration commit**
   - 在 `src/dust3r/model.py` 加 `enable_v8_pose_prompt`；
   - 扩展 decoder token layout；
   - 接入 `A_corr_t` 和 latent residual；
   - 默认关闭，确保原版路径不变。
   - 显式目标：
     - `enable_v8_pose_prompt=False` 时，原 Human3R forward 输出不变；
     - `enable_v8_pose_prompt=True` 时，decoder token layout 为 `[pose, corr, image, human]` 或兼容旧 anchor/shot 的扩展形式；
     - downstream head 前第 0 个 pose token 被替换成 corrected pose token；
     - `res` 中能看到 gate、delta norm、raw/corrected pose token debug 信息；
     - forward 可以在 1 个 AABB batch 上跑通。

5. **loss commit**
   - 新增 `V81PosePromptLoss`；
   - 不删除旧 loss，只明确 V8.1 使用新 loss。
   - 显式目标：
     - 能计算 corrected camera pose 相对 GT pose 的 translation / rotation loss；
     - 能记录 raw vs corrected pose error；
     - 能记录 gate mean、delta latent norm；
     - loss 在 batch size 1 和 batch size > 1 下都不出现 shape / squeeze 错误；
     - 不影响原有 `Regr3DPose`、SMPL loss 和 V7 loss 的可用性。

6. **train/check commit**
   - 加小训练配置或命令记录；
   - 跑 1-2 sample overfit；
   - 保存日志和可视化。
   - 显式目标：
     - 只训练 V8.1 prompt / residual / gate 分支，打印 trainable params 验证；
     - 1-2 个 AABB 样本上 pose loss 能明显下降；
     - corrected pose error 低于 raw Human3R；
     - gate 在 view2 或高误差帧有更强响应；
     - 能保存原版和 V8.1 corrected viewer，用于直观看 A->B 相机是否被拉回。

## 9. 主要风险和检查点

| 风险 | 检查方式 |
|---|---|
| token 插入后切片错位，human token 或 image token 被误送入 head | 用 `_decoder_token_layout()` 打印每段 shape，并写最小 shape check |
| pose head 仍然读第 0 个 token，corrected pose token 没有真正替换进去 | 在 `res` 里记录 raw/corrected pose token norm 和 camera pose 差异 |
| 当前帧 decoder 前错误使用了当前帧 head 输出 | 明确只允许当前 tokens、state、pose memory、上一帧 cached outputs |
| freeze 后 residual branch 没有梯度 | 打印 trainable params 和 grad norm |
| GT pose 坐标系与 `pred["camera_pose"]` 不一致 | 先复用 `Regr3DPose` 的相对 pose 逻辑，单独写 pose loss sanity check |
| 大角度样本不够或选错角度 | dataloader manifest 先保存 angle 和相机朝向图 |
| 进入 decoder 的 prompt 影响 scene/human 输出 | 先观察 viewer，必要时加 `L_keep` |

## 10. 第一轮验收标准

第一轮不是要求泛化，只要求工程和逻辑跑通：

```text
1. dataloader 能稳定选出 >=60 deg 的 AABB 样本；
2. V8.1 模型启用后，decoder token layout 正确；
3. 只有 V8.1 prompt/residual/gate 分支参与训练；
4. 1-2 个样本 overfit 时，corrected pose error 能明显低于 raw Human3R；
5. viewer 中 A->B 后的相机偏移有可见改善；
6. 原版 Human3R 路径在 enable_v8_pose_prompt=False 时不受影响。
```
