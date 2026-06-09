# V8.6 复盘与修复计划：统一评估协议、数据输入和几何一致性

## 版本定位

当前进入 `V8.6`。

- `V8.4`：建立 mixed AABB/AAAA 训练和 pose benchmark。
- `V8.5`：AIST/ASIT 多视角数据预处理。
- `V8.6`：暂停继续堆训练，先修复今天暴露出来的评估、dataloader、数据集和 SMPL/pointcloud 一致性问题。

V8.6 的核心目标不是马上让模型更复杂，而是先保证：

```text
同一个输入格式
同一个 raw Human3R 定义
同一个 viewer world gauge
同一套 GT / metric 计算方式
```

否则训练结果、可视化结果和指标会互相矛盾，无法判断模型到底有没有进步。

## 今天暴露的主要问题

### 1. Raw Human3R 可视化来源不统一

今天 THuman viewer 里出现了一个明显问题：灰色 raw Human3R 相机位姿看起来完全不对。

原因是 viewer 中的 raw 并不一定是真正通过原版 `demo.py` 跑出来的 Human3R raw；有时它是 benchmark/dataloader 路径里临时生成的 `v8_raw_camera_pose`。这会带来两个风险：

- raw / corrected / GT 不在同一个输入预处理下。
- raw / corrected / GT 不在同一个 world gauge 下。

后续规则：

```text
raw、corrected、GT 必须明确来自同一个输入和同一个坐标基准。
不允许把 demo.py raw、dataloader raw、pose dump camera、saved output scene 随意混合叠加。
```

需要给 viewer 加一个更严格的 metadata / assert：

- 输入模式：`demo` 还是 `dataloader`
- resize 模式：是否 `human3r_demo`
- raw 来源：原版 `demo.py` 还是同模型 `v8_raw_camera_pose`
- viewer gauge：以哪一个 frame-0 camera 为世界基准
- corrected pointcloud / SMPL 是否已经按 pose dump camera 做 frame-wise transform

### 2. Dataloader resize / crop 曾经导致 raw 不可信

旧版本 dataloader 曾经使用类似 `resolution=(512, 288)` 的固定横板输入。对于 AvatarReX 这种竖图，会裁掉大量人体上下文，导致：

- 人体不完整，例如头、脚被裁掉。
- 原版 Human3R demo 和 dataloader raw 不是同一个输入分布。
- 后续用这个 raw 做对比，结论不可靠。

现在应该统一使用：

```text
resolution=512
resize_mode='human3r_demo'
```

这和 `demo.py` 的 `load_images(size=512)` 更一致。代价是不同数据集尺寸不一样：

```text
AvatarReX 竖图: 约 368x512
THuman 横/近方图: 约 512x432
```

因此在没有 aspect-bucket sampler 前，真实 batch size 仍建议先用 1，再用 gradient accumulation 增大等效 batch。

### 3. 只修 camera pose 不等于完整 3D 几何正确

今天单 clip overfit 说明了一件重要事情：

```text
pose-only branch 可以把 camera pose 拉近 GT，
但 SMPL 和点云并不会自动一起正确。
```

原因是 Human3R 输出中至少有三套独立量：

```text
camera_pose
  决定 camera space 如何放到 world space。

pts3d_in_self_view
  决定每个像素在当前 camera space 中的深度和 3D 位置。

smpl_transl
  决定人体 SMPL 在当前 camera space 中的整体平移。
```

最终可视化大致是：

```text
world_point = camera_pose @ pts3d_in_self_view
world_smpl  = camera_pose @ SMPL(smpl_pose, smpl_shape, smpl_transl)
```

所以：

- 改 `camera_pose`：点云和人体都会被放到新的 world 坐标，但它们各自在 camera space 内的深度/平移不变。
- 改 `smpl_transl`：只移动人体，不移动点云。
- 改 `pts3d_in_self_view`：只移动/改变点云，不移动人体。

这解释了今天的现象：

```text
camera pose 已经接近 GT，
但后两帧人体和点云在深度上仍然没有完全对齐。
```

### 4. 手动 SMPL 平移验证了缺失的 human translation correction

今天做了一个 oracle 手动实验：

```text
delta_world = GT_SMPL_head_world - Pred_SMPL_head_world
delta_cam   = R_c2w^T @ delta_world
smpl_transl = smpl_transl + delta_cam
```

只改 `smpl_transl`，不改 camera pose，也不改 pointcloud。

结果：

- `postjump_only`：只修后两帧，后两帧 head 对齐 GT，但前两帧仍有约 0.43m 的人体位置误差。
- `all4`：四帧都按 GT head 平移后，人体整体位置明显更对。
- 但 pelvis / mean joints 仍有几厘米误差，因为只做整体平移，没有改人体姿态、shape 或 root rotation。

这说明：

```text
当前 pose-only 设计缺少 human translation residual / SMPL anchor correction。
只看 camera metric 会高估模型效果。
```

### 5. Pointcloud 也可能有类似问题

由于 `pts3d_in_self_view` 独立于 `camera_pose`，点云也可能存在和 SMPL 类似的问题：

```text
camera pose 对了，
但每帧 self-view pointmap/depth 的尺度或前后深度仍然不一致。
```

因此后续不能只看相机指标，还要看：

- corrected pose 下的 pointcloud 是否跨帧一致。
- 人体附近点云是否和 SMPL 相对位置合理。
- 地面/墙面等大平面是否在 A/B 前后连续。
- 点云是否因为 camera pose correction 被整体投到错误深度。

AvatarReX 因为旧裁剪/视野问题不适合单独看地板点云，THuman / AIST 更适合检查 pointcloud。

### 6. 数据集处理仍需统一

当前训练数据核心目录已经统一到：

```text
/data/wangzheng/iJCV-CODE/data/Training
```

现有数据源：

```text
AvatarReX: lbn1, zzr
THuman: thuman00, thuman02
AIST/ASIT: asit
```

当前 V8.6 主线决策：

```text
先只使用 AvatarReX + THuman。
ASIT/AIST 预处理结果可以保留在磁盘上，但暂时不进入训练。
训练主线先回到 pose-only correction，不引入 native SMPL 分支。
```

原因：

- AvatarReX + THuman 当前数据已经足够做下一轮问题定位。
- ASIT 是 native SMPL，不是 SMPL-X；为了兼容它而改训练 GT 生成逻辑，会增加主线冲突风险。
- 当前先把 camera / SMPL-X / pointcloud 的评估协议稳定，再决定是否重新启用 ASIT。

当前主线允许的数据格式：

```text
AvatarReX: SMPL-X-like smplx_* fields, camera-space transl target
THuman:   SMPL-X-like smplx_* fields, world-space human params + T_w2c transform
```

当前主线暂不允许：

```text
ASIT native smpl_* fields
SMPL native forward branch in update_smpl_gt()
ASIT group in training manifest
```

需要继续保留 raw metadata：

```text
/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1
/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr
```

今天对 AIST/ASIT 的关键记录：

- 原始视频是 60 FPS。
- 只保留前 3 秒。
- 每 4 帧取 1 帧，得到约 45 帧，减少重复。
- RGB、SMPL、camera 必须按同一个 source frame 对齐。
- `c05` 曾经发现视频动作延迟/时间戳不对，应跳过或显式 blacklist。
- AIST 使用 SMPL，不是 SMPL-X，需要训练/验证代码明确兼容 native SMPL 和 SMPL-X 两种格式。
- mask 暂时可以为空或占位，后续可再用检测/分割生成。

### 7. 指标不够完整

目前 benchmark 主要指标是 camera pose：

```text
raw/corrected translation error
raw/corrected rotation error
gate
delta_norm
improvement
```

这些不够。V8.6 必须增加几何一致性指标：

```text
SMPL head world error
SMPL pelvis world error
SMPL mean joint world error
SMPL translation residual magnitude
pointcloud cross-frame centroid drift
human-near pointcloud drift
corrected camera + predicted SMPL consistency
corrected camera + predicted pointmap consistency
AAAA no-correction stability
```

这些指标可以用 GT 做 evaluation，但不能在 inference 中使用 GT。

### 8. 早停和 best checkpoint 选择目前不可靠

当前训练代码的 early stopping 逻辑是：

```text
每个 epoch 开头先 eval
用 val/test 的 loss_med 判断 best
连续 early_stopping_patience 次没有变好就停止
停止后仍然保存 checkpoint-final.pth
```

这对常规训练可以用，但对当前 pose correction 任务不够可靠：

- 我们关心的不只是 loss，还包括 AABB 是否修正、AAAA 是否少修、gate 是否合理、SMPL/pointcloud 是否一致。
- `loss_med` 下降不等于 viewer 变好，也不等于相机、人体和点云都变好。
- `early_stopping_patience=4` 太激进，模型可能还没学会 correction 就停了。
- `checkpoint-final.pth` 不一定代表完整训练到设定 epoch，它可能只是 early stop 时的最终保存，名字容易误导。

因此之前 pose-only 训练“final 更好但训练轮次不足”的结果，只能说明这个方向有信号，不能作为充分收敛的结论。

后续早停原则：

```text
早停不能只看 loss_med。
在评估协议稳定前，可以先关闭早停，或只把早停作为提醒，不自动终止。
best checkpoint 必须同时记录 best-by-loss、best-by-camera、best-by-SMPL、best-by-composite 和 final。
```

### 9. pose head 全量微调风险过大

之前 pose-head 版本是全量打开 pose head 训练，结果 viewer 明显不稳定。这个问题的本质是：

```text
pose head 是原 Human3R 已经学好的输出头。
全量微调很容易破坏它原来的能力，尤其是在小数据和偏分布数据上。
```

后续如果要动 pose head，应优先考虑：

```text
冻结主干 + pose correction branch
只训练 residual / gate / relation head
如确实需要动 pose head，使用 LoRA / adapter / last-layer 小范围微调
```

不再把“全量微调 pose head”作为主线方案。

### 10. 数据分布会掩盖 correction 能力

当前训练数据里并不是所有 AABB 都真的需要纠正：

- AvatarReX AABB 更接近我们要解决的 drift / jump case。
- THuman 背景纹理更丰富，原版 Human3R 在很多 AABB 上本来就不错。
- AAAA 是正常连续帧，主要用于教模型少修或不修。

如果训练集中“需要大幅修正”的比例太低，模型会倾向于保守输出：

```text
少修正通常更安全，
但会导致真正大跳变时修不够。
```

后续必须显式统计：

```text
AvatarReX AABB drift ratio
THuman/AIST AABB already-good ratio
AAAA no-drift ratio
large-angle / small-angle ratio
```

并在 benchmark 中分别报告，不能只报混合平均值。

### 11. gate 的设计和监督还不够闭环

目前 gate 的作用是控制 residual 修正幅度：

```text
gate 越大，允许 correction branch 改得越多。
gate 越小，尽量保持 raw Human3R。
```

但当前 gate 更多是通过整体 loss 间接学出来的，还没有非常强的显式协议证明它真的学会了：

```text
哪里需要修
哪里不应该修
修多少才合理
```

后续 gate 不能只看 viewer，需要单独评估：

- AABB drift 上 gate 应该高。
- AAAA 上 gate 应该低。
- 原版 Human3R 已经正确的 AABB 上 gate 也应该低。
- gate 高但 correction 没变好，要判为失败。
- gate 低但 raw 本来错误，也要判为失败。

可以考虑增加显式 drift score / improvement margin / residual small 等辅助监督，但不能让 gate 直接依赖推理阶段拿不到的 GT。

### 12. batch size、aspect ratio 和训练效率还没有工程化解决

之前为了加速训练尝试过增大 batch size，但现在仍有两个限制：

- AvatarReX 竖图和 THuman/AIST 横图尺寸不同，直接混 batch 容易触发 padding/crop/shape 问题。
- 旧 dataloader 为了固定尺寸曾经裁掉人体，导致 raw baseline 不可信。

短期规则：

```text
优先保证输入和坐标正确，再追求 batch size。
batch size 变大前，必须做 bz=1 vs bz=N 的同样本一致性 smoke test。
```

中长期可以做：

```text
aspect-ratio bucket sampler
same-shape batch
gradient accumulation
```

### 13. 旧指标和旧输出要降级为历史参考

凡是满足下面任一条件的旧结果，不能再作为当前结论：

```text
raw 来源不清楚
使用旧 512x288 / 固定横板裁剪
viewer 没有写明 world gauge
GT camera 坐标系没有经过确认
corrected camera / SMPL / pointcloud 来源混合
只看 camera metric，没有 SMPL / pointcloud 诊断
```

这些结果可以帮助回忆实验过程，但不能用于证明模型有效。

### 14. 输入信息泄漏问题必须继续防住

V8.3 已经暴露过一个问题：如果训练/推理输入里保留 `shot_label`、`ray_map`、`ray_mask=True` 这类额外信息，模型可能不会真正学习“当前图像和历史是否对齐”，而是学到捷径：

```text
看到 shot boundary label -> 修
看到固定帧位置 -> 修
看到 GT-like ray geometry -> 按几何提示修
```

这不符合最终目标。最终方法应尽量只使用推理时真实可得的信息：

```text
当前图像 token
当前人体/pose token
上一帧 recurrent state / pose memory / human anchor memory
上一帧模型自己输出的 scene / SMPL / pose 结果
```

不能依赖人工给出的 shot label，也不能依赖由 GT 相机或 GT 几何构造出来的强提示。

### 15. overfit 实验只能验证模型容量，不能证明泛化

单 clip overfit 的价值是：

```text
验证结构有没有能力表达修正。
验证 loss 和反向传播是否能推动 correction。
验证坐标系和 viewer 是否大体闭环。
```

但它不能证明：

```text
模型真的学会了通用 pose correction。
模型真的会根据人体/场景/历史关系判断。
模型能泛化到新人物、新场景、新相机和新动作。
```

所以后续汇报时要分清：

- `overfit success`：说明这条结构有拟合能力。
- `same-dataset success`：说明对相似分布有效，但可能仍有数据集偏置。
- `unseen-dataset success`：才更接近真正泛化。
- `AAAA no-op success`：说明模型知道正常连续帧不要乱修。

## V8.6 硬性规则

### Rule 1: viewer 必须先声明坐标基准

每个 viewer case 必须保存：

```text
viewer_gauge
raw_camera_source
corrected_camera_source
gt_camera_source
scene_payload_source
smpl_payload_source
input_mode
resize_mode
model_path
checkpoint_type
```

没有这些 metadata 的旧 viewer 输出不能再作为证据。

### Rule 2: raw / corrected 必须同源

允许两种模式，但不能混用：

```text
Mode A: demo mode
  raw        = 原版 Human3R demo.py 输出
  corrected  = 同一批 demo input 上的 correction 输出

Mode B: dataloader mode
  raw        = dataloader 输入下的 v8_raw_camera_pose
  corrected  = 同一 dataloader 输入下的 corrected camera_pose
```

不允许：

```text
demo raw + dataloader corrected + pose dump GT 混合可视化
```

除非脚本明确做了 frame-wise gauge transform，并把 transform 写入 metadata。

### Rule 3: camera metric 不能单独代表模型成功

以后报告必须同时包含：

```text
camera pose metric
SMPL world metric
pointcloud consistency diagnostic
qualitative viewer
```

如果只修好了 camera pose，但 SMPL / pointcloud 错，结论只能写：

```text
pose branch 修正了 camera，但完整几何仍不一致。
```

### Rule 4: 旧的 dataloader raw 结果暂时不作为结论依据

凡是使用旧 `512x288` 或未注明 resize_mode 的结果，只能作为历史参考，不能再和当前结果直接比较。

### Rule 5: early stopping 暂时不能作为主训练终止标准

在 composite benchmark 稳定前：

```text
训练可以保存 best，但不要因为单一 loss_med 自动停掉关键实验。
```

如果必须早停，至少要同时看：

```text
val loss
AABB camera improvement
AAAA no-op stability
SMPL world error
gate separation
```

### Rule 6: checkpoint 命名必须反映真实含义

后续每次训练至少保存并标注：

```text
checkpoint-best-loss.pth
checkpoint-best-camera.pth
checkpoint-best-smpl.pth
checkpoint-best-composite.pth
checkpoint-final-epoch{N}.pth
```

如果 early stop 发生，必须在 log 和 checkpoint metadata 里写清楚：

```text
stopped_epoch
stop_reason
best_metric_name
best_metric_value
```

### Rule 7: 不再主线使用 pose head 全量微调

主线实验优先：

```text
frozen Human3R backbone
frozen pose head
train correction token / decoder prompt / residual heads
```

如要微调原 pose head，只允许作为对照实验，并优先用 LoRA / adapter / last-layer 微调。

### Rule 8: train / eval / inference 的 GT 边界必须清楚

```text
inference 不能用 GT。
GT 只用于训练 loss、benchmark metric 和 oracle diagnostic。
viewer 里 GT 只能作为红色参考，不参与 corrected 输出生成。
```

### Rule 9: 禁止使用会泄漏答案的输入提示

主线训练和推理中不能使用：

```text
shot_label
GT-derived ray_map
ray_mask=True 的强几何提示
任何只在构造 AABB 时知道、真实推理时不知道的 label
```

如果某个实验为了诊断临时使用这些信息，必须标为 oracle / diagnostic，不能和主线 image-only / streaming 方法混在一起比较。

## 下一步执行计划

### Step 1. 固化 canonical viewer 协议

目标：

写一个或修改一个 viewer 脚本，让它只支持清晰的两种模式：

```text
demo-consistent
dataloader-consistent
```

输出必须包含：

```text
viewer_record.json
viewer_coordinate_metadata.json
raw/corrected/gt camera matrices
scene payload source
smpl payload source
```

验收标准：

- 同一个 sample 上 raw/corrected/GT 三相机能稳定复现。
- 不再出现 raw 相机来自错误格式的问题。
- 脚本遇到混合来源时直接报错。

### Step 2. 重建一批可信 raw Human3R baseline

目标：

对 benchmark 中选定的少量样本重新生成 raw baseline：

```text
AvatarReX large angle
THuman large angle
THuman small angle / AAAA
AIST large angle
```

每个 baseline 都要明确：

```text
input_mode
resize_mode
raw_camera_source
raw_scene_source
```

验收标准：

- raw Human3R 单独 viewer 看起来和预期一致。
- corrected viewer 中的 gray raw 和单独 raw viewer 一致。

### Step 3. 增加 SMPL 几何诊断

目标：

在 benchmark eval 中加入 SMPL world metric：

```text
head error
pelvis error
mean joint error
root/transl error
```

并且拆分保存三种诊断：

```text
pred camera + pred SMPL
GT camera   + pred SMPL
pred camera + GT SMPL
```

用途：

- 判断错误来自 camera 还是 SMPL。
- 判断是否需要 human translation residual。

验收标准：

在同一个样本上能解释：

```text
camera 已对但人体仍偏
人体平移 oracle 后是否变对
```

### Step 4. 增加 pointcloud 几何诊断

目标：

对 corrected pose 下的 pointcloud 做跨帧一致性检查：

```text
全局点云 centroid / bbox
人体附近点云 centroid / bbox
地面/墙面候选平面
frame 1 -> frame 2 jump drift
AAAA drift
```

优先用 THuman / AIST 检查，因为它们比 AvatarReX 更适合看完整场景和地面。

验收标准：

- 能判断点云是否和 SMPL 一样存在 camera-space depth/scale residual。
- 能判断 pointcloud 是否需要额外 residual 或 loss。

### Step 5. 重新定义 V8.6 模型目标

当前 pose-only 目标不够。下一版模型应考虑：

```text
pose residual
+ human translation residual
+ optional pointmap/global scale residual
+ geometry consistency losses
```

可能的结构：

```text
correct token -> decoder -> relation latent
relation latent -> pose residual head
relation latent -> human transl residual head
relation latent -> optional pointmap/global shift head
```

训练 loss 不应只看 camera pose，还应加入：

```text
L_camera_pose
L_smpl_head_world
L_smpl_pelvis_world
L_smpl_joint_world
L_human_history_alignment
L_pointcloud_history_alignment
L_aaaa_noop
L_residual_small
```

### Step 6. 重做训练协议和 checkpoint 选择

目标：

把训练流程从“只看 loss 早停”改成“固定训练 + 多指标评估 + 多 best 保存”。

需要增加：

```text
每个 epoch 后跑固定 benchmark subset
保存 camera / SMPL / pointcloud / gate 指标
按不同指标保存 best checkpoint
early stopping 默认关闭，或只在 composite metric 稳定后启用
```

验收标准：

- 能清楚回答一个 checkpoint 为什么被选中。
- 不再出现 `checkpoint-final.pth` 实际上是 early stop 产物但看起来像完整训练的情况。
- viewer 选择 checkpoint 时能明确说明是 best-loss、best-camera、best-SMPL 还是 final。

### Step 7. 重新审计数据分布

目标：

给每个 train/test manifest 输出统计表：

```text
dataset
clip_type: AABB / AAAA
source_subject
angle
raw_camera_error
raw_smpl_error
is_drift_candidate
is_already_good
```

验收标准：

- AABB 中真正有 drift 的比例明确。
- AAAA 和 already-good AABB 的比例明确。
- AvatarReX / THuman / AIST 不再混成一个平均数直接判断。

### Step 8. 检查输入字段白名单

目标：

给 dataloader 和 model forward 加一个 V8 主线输入白名单，明确哪些字段可以进模型，哪些只能用于 loss/eval。

允许进模型的字段应只包括推理时真实可得的信息：

```text
image
intrinsics if original Human3R path requires it
previous recurrent state
previous model outputs used as streaming memory
```

只能用于训练/评估的字段：

```text
GT camera
GT SMPL
GT mesh / joints
GT masks
dataset labels
clip type labels
```

验收标准：

- 训练 log 里写出本次 forward 实际使用了哪些字段。
- benchmark 里写清楚 GT 只用于 metric，不参与 inference。
- 不再出现 `shot_label` / `ray_map` 无意中进入模型主线的情况。

### Step 9. 再考虑新一轮训练

只有在 Step 1-4 的评估协议修好后，才开始新的训练。

初始训练建议：

```text
数据: lbn1 + zzr + thuman00 + thuman02 + asit verified subset
clip: AABB + AAAA
训练: 先 pose + human transl residual，不急着动完整 pointmap
评估: camera + SMPL + pointcloud + viewer
```

## 当前清理状态

已清理：

- `output/v8_5*` 临时输出。
- v8.5 临时 viewer 进程。
- `config/` 中旧实验配置在本地清理，并通过 `skip-worktree` 避免进入 commit。

当前保留配置：

```text
config/train.yaml
config/train_v8_4_mixed_aabb_aaaa_image_only_pose_relation.yaml
config/train_v8_4_mixed_no_zxc_bs10_long.yaml
```

注意：

这些清理不代表旧实验结论无效，而是旧配置/旧输出不再作为当前主线入口。后续需要引用历史结果时，应从文档和 committed code 中找，而不是依赖散落的临时 output。

## 当前结论

今天最重要的结论是：

```text
V8 pose correction token 目前只证明了 camera pose 可以被修正。
但完整 3D 几何是否正确，必须同时检查 SMPL 和 pointcloud。
```

下一步不应该继续只扩大训练，而是先把：

```text
raw baseline
dataloader input
viewer coordinate
SMPL metric
pointcloud metric
```

全部统一。

只有这些基础稳定后，再讨论新的 pose correction token 是否有效、是否需要 human residual、是否需要 pointmap residual。
