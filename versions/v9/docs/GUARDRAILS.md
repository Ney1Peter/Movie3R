# V9 Guardrails

这份文档专门记录容易反复出错的细节。做 V9 训练、推理、可视化、改代码之前先看这里。

## 1. 坐标系规则

坐标系错误是目前最大风险。只要 camera / SMPL / pointcloud / viewer 任何一个环节混用坐标，loss 可能看起来下降，但可视化会出现人倒置、漂浮、相机在地板下、raw Human3R 和 demo 不一致等问题。

### 1.1 AvatarReX

AvatarReX 的最终 pose 监督和 viewer GT camera 必须来自 raw calibration，而不是预处理后 `cam/*.npz` 里的 processed `camera_pose`。

raw calibration 约定：

```text
X_cam = R_w2c @ X_world + T_w2c
R_c2w = R_w2c.T
t_c2w = -R_w2c.T @ T_w2c
raw_camera_pose = c2w(R_c2w, t_c2w)
```

四帧相对 pose target：

```text
T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i
```

训练配置里：

```text
pose_key='raw_camera_pose'
load_da3_depth=False
raw_calibration_root={...}
```

不要把下面这个当 AvatarReX 最终 pose GT：

```text
/data/wangzheng/iJCV-CODE/data/Training/<group>/<seq>/cam/*.npz camera_pose
```

它可以用于数据组织和部分预处理，但不能直接作为 V9 camera loss 或 GT camera viewer 的最终标准。

### 1.2 THUman

THUman 的官方 camera / SMPL 投影已经单独验证过。THUman 没有 AvatarReX 那种额外 raw calibration 映射时，`cam/*.npz` 里的 c2w 可以作为 `raw_camera_pose` 使用。

dataloader 里如果检测到 THUman：

```text
raw_camera_pose = camera_pose
human_params_are_world = True
T_w2c = inv(raw_camera_pose)
```

关键是 camera GT 和 SMPL GT 必须在同一个 gauge 下监督。不要一边用 THUman camera，另一边把 SMPL 当成 AvatarReX camera-space 参数。

### 1.3 SMPL / Human

Human3R 的人体输出不是只由 camera pose 决定。SMPL 自己有 `smpl_transl` / human latent，camera pose 对了以后，人仍可能在高度或深度上错。

因此：

- camera loss 只保证相机对。
- human trans loss 才保证人体位置对。
- viewer 中必须同时画 corrected SMPL 和 GT SMPL，不要只看相机。

V9 的正确目标是：

```text
corrected_camera_pose @ corrected_human
  ~= GT_camera_pose @ GT_human
```

而不是只让：

```text
corrected_camera_pose ~= GT_camera_pose
```

### 1.4 Viewer 颜色和对齐

统一 viewer 颜色：

```text
gray   = raw Human3R
yellow = corrected V9 output
red    = GT
```

正确 viewer 必须满足：

- raw Human3R 灰色相机和单独运行原版 `demo.py` 的结果一致。
- GT 红色相机不能跑到人和场景底下。
- corrected 黄色相机、点云、人体必须使用同一套 frame-0 world gauge。
- 如果 raw Human3R 在对比 viewer 里和原版 demo 不一致，先修 viewer / saved-output 坐标，不要继续分析模型效果。

修正输出中可能同时保存：

```text
v8_raw_camera_pose = correction 前 Human3R pose
camera_pose        = correction 后 pose
```

这是为了 metric 和可视化，不代表推理时用了 GT 或未来帧。

## 2. Dataloader 规则

### 2.1 Resize / crop

AvatarReX 曾经因为 crop / resize 导致头、脚或身体边缘被裁掉，进而影响 Multi-HMR 检测和可视化。

V9 AvatarReX 单 clip / full-body 检查优先用：

```text
resize_mode='resize_only_16'
```

或者等价 no-crop 路径：

```text
resize_mode='no_crop'
resize_mode='human3r_no_crop'
```

`human3r_demo` 会模仿 demo 的 long-edge resize + center crop，适合对比 demo，但可能裁掉竖版 AvatarReX 的头或脚。使用时必须明确记录。

旧的 dataset crop 路径可能和真实 demo 推理不一致，不要作为 V9 成功标准。

### 2.2 DA3 depth

默认：

```text
load_da3_depth=False
```

DA3 / monocular pseudo-depth 不是跨相机 metric GT，不能用来验证世界坐标是否对齐，也不应该作为 V9 pose/human correction 的监督标准。

如果 depthmap 被 zero-filled，这是符合 pose/human correction 实验预期的；Human3R 自己预测的 pointmap 仍然可以用于可视化。

### 2.3 Dataloader 不等于真实推理

真实推理只能看到 RGB 图片/视频和模型历史 state。训练 dataloader 里如果保留了推理时没有的信息，会导致 benchmark 虚高。

V9 不应该依赖：

- shot label 作为输入。
- ray map /特殊数据标签作为 correction cue。
- GT camera pose 作为模型输入。
- GT SMPL 作为模型输入。
- DA3 metric depth 假设。

GT 只能进入 loss / metric / viewer overlay。

### 2.4 每次换数据集都要做 sanity check

新数据集进入训练前，至少做：

1. 保存 dataloader resize 后的 4 张输入图。
2. 用 GT camera + GT SMPL 投影到图像，检查人是否对齐。
3. 用 viewer 画 raw Human3R / corrected / GT 三套 camera。
4. 检查 corrected 点云和 corrected SMPL 是否在同一 world gauge。
5. 检查 raw Human3R viewer 是否和原版 `demo.py` 单独结果一致。

没有通过这些检查，不要开始大训练。

## 3. 训练规则

### 3.1 新训练从原版 Human3R 权重开始

除非明确做 continuation 或 ablation，新训练必须从原版 Human3R checkpoint 初始化。

不要这样做：

```text
old V8/V9 experiment checkpoint
  -> 继续训练
  -> 声称是新结构本身的能力
```

正确做法：

```text
original Human3R checkpoint
  -> train V9 correction branch
  -> evaluate
```

历史中已经出现过从旧实验权重继续训练导致判断混乱的问题。V9 必须避免。

### 3.2 训练顺序

推荐顺序：

1. 单 clip overfit，验证结构和坐标系。
2. 5 clip overfit，验证不是单个样本偶然。
3. 小批量 train/test，显式保留未见 test clips。
4. 扩大数据集，加入 AABB 和 AAAA。
5. 再做 LoRA / head 微调 ablation。

不要一开始直接大训练。坐标系或 dataloader 错时，大训练只会浪费时间。

### 3.3 当前应训练什么

V9 baseline：

```text
train:
  v8_pose_prompt / A_corr_t builder
  v8_pose_residual_head
  v8_human_latent_corr_head

freeze by default:
  Human3R backbone
  decoder
  original pose head
  original human head
```

后续可以做 LoRA：

```text
baseline correction branch only
correction + pose head LoRA
correction + human head LoRA
correction + pose head LoRA + human head LoRA
```

不推荐把 pose head / human head 全量解冻作为主线，因为之前全量 pose head 训练破坏了原版能力。

### 3.4 训练时要看的指标

不要只看 total loss。至少记录：

- `v82_trans_err`
- `v82_rot_err_deg`
- `v82_raw_trans_err`
- `v82_raw_rot_err_deg`
- `v82_gate_mean`
- `v82_delta_norm`
- `v82_human_trans_err`
- `v82_raw_human_trans_err`
- `v82_human_latent_delta_norm`
- best / final checkpoint 的 test benchmark

判断好坏时看：

```text
corrected camera error < raw camera error
corrected human trans error < raw human trans error
AAAA / stable samples gate low and delta small
AABB / drift samples gate higher and correction useful
```

## 4. 可视化规则

每次展示模型效果前，先单独跑或加载原版 Human3R raw 结果。raw 对了，才看 corrected。

如果出现以下情况，优先怀疑 viewer / coordinate，而不是模型：

- raw Human3R 在对比 viewer 中和原版 demo 不一样。
- GT camera 在地板下方或方向反了。
- corrected camera 看起来对，但 corrected human 飞起来。
- 点云和人体不是同一个 world 里的相对位置。
- 同一 clip 换 viewer 脚本后结果反了。

Multi-HMR 有时会对一个真实人输出多个候选人体。单人场景可视化时可以使用 top-1 SMPL filtering，但必须说明这是可视化过滤，不是场景里真的有多个人。

## 5. 代码修改标准

为了避免历史代码混乱，改代码按两步：

1. 先把不再使用的旧代码注释掉，commit 一次。
2. 再新增 V9 代码，commit 一次。

不要在一个 commit 里同时大量删除旧逻辑和增加新逻辑，否则之后很难 review 和回退。

Commit 格式：

```text
<type>: <description>
```

允许的 type：

```text
feat:     新功能
fix:      bug 修复
docs:     文档修改
refactor: 代码重构
test:     测试相关
chore:    维护/杂项
```

例子：

```text
docs: add v9 method overview
fix: align avatarrex smpl loss gauge
feat: add implicit human latent correction
test: add v9 single clip benchmark
chore: archive old v8 outputs
```

## 6. Checkpoint / output 清理规则

长期保留：

- 原版 Human3R 权重。
- 当前 V9 best / final。
- 明确命名的 milestone checkpoint。
- 能复现关键结果的 config / manifest / metric。

可以清理：

- 中间 epoch checkpoint。
- 旧 viewer 临时输出。
- 旧 tmp 训练目录。
- 已归档且确认不再使用的测试图片。

不要清理：

- 其他人的文件。
- 原始数据或预处理数据，除非已经确认映射关系和替代路径。
- 没有记录来源的 calibration / smpl 参数文件。

## 7. V9 开工前最短检查表

```text
[ ] 从 original Human3R checkpoint 开始。
[ ] AvatarReX 使用 raw_camera_pose，THUman 使用官方 c2w。
[ ] load_da3_depth=False。
[ ] resize/crop 路径和 demo 对比一致，必要时保存 resize 后图片。
[ ] raw Human3R viewer 与原版 demo 一致。
[ ] corrected / raw / GT 三套 camera 在同一 viewer gauge。
[ ] corrected SMPL 和 GT SMPL 都画出来。
[ ] inference 不读取 GT。
[ ] loss、metric、viewer 输出路径明确。
[ ] best/final checkpoint 保存，临时 checkpoint 后续清理。
```
