# V14.1: First-Post-Cut Event-Only Correction

## 1. 阶段定位

V14.1 只验证一个最小问题：

> 已知 camera cut 发生位置时，复用 V9 的 correct-token decoder refine、camera
> correction head、human correction head 和 head LoRA，只纠正第一张 post-cut
> frame，是否具备同时修正 camera 与 human 显式输出的能力？

本阶段是 capacity probe，不是完整 V14 系统。它暂不解决：

- 自动 cut detector；
- post-cut hard-reset state 与 corrected state 如何合并；
- 第一帧 correction 如何转换成整个 shot 的固定 SE(3) Boundary；
- 后续帧是否继续使用 corrected recurrent state；
- V13 multi-human identity 和 consensus；
- pointmap/scene 的显式统一变换；
- 跨 sequence、跨 subject 的正式泛化。

因此，V14.1 成立只能说明“第一帧隐式纠正值得继续”，不能直接说明完整 V14 已成立。

---

## 2. 当前输入协议

每个训练样本只有 3 帧：

```text
view 0: camera A, frame t-1    shot_label = 0
view 1: camera A, frame t      shot_label = 0
view 2: camera B, frame t      shot_label = 1
```

即：

```text
A(t-1) -> A(t) -> B(t)
```

选择同步的 `A(t) -> B(t)` 是为了第一轮尽量隔离 camera viewpoint jump，避免同时
引入较大人体 motion。

当前固定：

```text
Kpre = 2
Kpost = 1
num_views = 3
shot_labels = [0, 0, 1]
```

不再使用：

- AAAA；
- AABB；
- 固定四帧输入；
- normal/cut pattern 分类；
- post-cut future frame；
- segment loss。

camera cut 由外部 `shot_label` 显式提供。V14.1 不训练 detector。

---

## 3. 模型结构

### 3.1 冻结主干

初始化 checkpoint：

```text
checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth
```

冻结：

- Human3R image encoder；
- recurrent decoder 的原始参数；
- scene/pointmap head；
- base camera pose head；
- base human/SMPL-X head；
- Multi-HMR backbone。

V14.1 继续使用 Human3R 的流式 recurrent forward，不读取 future frame。

### 3.2 两个 correct tokens

V9 原始 relation prompt 可以产生：

```text
semantic token
alignment token
momentum token
```

V14.1 设置：

```text
v8_pose_prompt_token_ablation = no_momentum
```

因此 event frame 只插入两个 token：

```text
CorrectToken = [SemanticToken, AlignmentToken]
```

其中：

- SemanticToken 聚合当前 image/pose/human token 与已有 recurrent state/pose memory；
- AlignmentToken 使用当前 pose token、上一帧 pose token、二者 latent difference 和
  memory context；
- MomentumToken 不进入 decoder；
- previous corrected token、previous correction residual 和 previous correction gate
  不构成本阶段的纠正 token。

保留 state 和 pose memory 的原因是 correct token 必须通过 decoder attention 进行
refine，而不是退化为一个直接预测 SE(3) 的浅层 MLP。

### 3.3 Event-only decoder refine

对 context frame：

```text
shot_label = 0
-> no correct token appended
-> no pose latent correction
-> no human latent correction
-> pose/human head LoRA disabled
```

对第一张 post-cut frame：

```text
shot_label = 1
-> append semantic + alignment correct tokens
-> correct tokens participate in full recurrent decoder attention
-> obtain refined pose/correct/human tokens
```

correct token 不是仅在 decoder 后使用。它必须进入完整 decoder attention，才能影响
pose token、human token 与 recurrent representation 的联合 refine。

### 3.4 两条 corrected head 路径

Camera：

```text
refined correct token
-> pose latent residual head
-> corrected pose token
-> camera pose head with pose-head LoRA
-> explicit camera translation + quaternion
```

Human：

```text
refined correct token + refined human token + corrected pose token
-> human latent residual head
-> corrected human latent
-> SMPL-X head with human-head LoRA
-> explicit human parameters and translation
```

V14.1 第一轮固定：

```text
pose correction gate = 1
human correction gate = 1
```

原因是 cut 已由外部 detector 给出，本阶段只验证纠正容量，不重复训练是否需要纠正。

### 3.5 Event-only LoRA

新增 `v14_1_event_only_head_lora`：

- 每个 view 开始前先关闭 pose/human head LoRA；
- 只有当前 view 实际插入 correct tokens 时才启用；
- 防止上一条样本以 event frame 结束后，LoRA 状态泄漏到下一条样本的 context frame。

新增 `v14_1_freeze_unused_prompt_params`：

- momentum MLP；
- unused human/body-part prompt modules；
- unused pooling/gate heads；
- 其他未进入当前两-token 路径的模块；

均不进入 optimizer。

---

## 4. 当前状态语义

V14.1 的 forward 是 causal、online 的：

```text
frame t 只读取 frame <= t
fixed Kpre = 2
no future frame
no global optimization
```

但当前阶段尚未冻结正式 V14 的 state commit 规则。

当前 capacity probe 中，event frame 仍运行在现有 V9 recurrent forward 内，correct token
参与 decoder refine。它不是最终设计中的“双 state”或“shadow transaction”实现。

必须区分：

```text
V14.1 已验证：
第一张 post-cut frame 能否被隐式 correct-token 路径纠正

V14.1 未验证：
纠正后的 recurrent state 是否应提交给后续帧
或是否只提取一次显式 Boundary 后回到 reset raw state
```

后一个问题应在 V14.2 中单独比较，不能由单帧过拟合结果代替。

---

## 5. Loss

仅监督 `shot_label=1` 的 event frame。

配置：

```text
camera translation weight = 1
camera rotation weight    = 5
human translation weight = 10
latent residual L2 weight = 1e-5
```

context frame：

```text
shot_label = 0
-> forward only
-> no V14.1 camera/human loss
```

event frame：

```text
shot_label = 1
-> camera pose loss
-> human translation loss
-> small latent residual regularization
```

当前不训练：

- pointmap loss；
- depth loss；
- scene consistency；
- no-cut identity loss；
- segment propagation loss；
- detector/gate loss；
- post-cut later-frame loss。

实现开关：

```text
V82PoseRelationLoss(supervise_shot_label_only=True)
```

同一 batch 的同一个 view 必须共享 event/non-event 语义；混合 event mask 会直接报错，
避免 silent supervision mismatch。

---

## 6. 数据

数据根目录：

```text
/data/wangzheng/iJCV-CODE/data/Training
```

明确排除：

```text
/data/wangzheng/iJCV-CODE/data/Training/asit
```

### 6.1 单样本 capacity probe

```text
lbn1/22053926 frame 1191
lbn1/22053926 frame 1192
lbn1/22010716 frame 1192
camera angle = 132.853 degrees
```

manifest：

```text
config/manifests/v14_1_cut_event/single/lbn1_1192.jsonl
```

### 6.2 10-event pilot

来源：

```text
AvatarReX: 3
THuman:    2
MVHuman:   5
total:    10
```

manifests：

```text
config/manifests/v14_1_cut_event/ten/avatarrex.jsonl
config/manifests/v14_1_cut_event/ten/thuman.jsonl
config/manifests/v14_1_cut_event/ten/mvhuman100.jsonl
config/manifests/v14_1_cut_event/ten/mvhuman200.jsonl
```

训练和 test dataset 长度必须分别为：

```text
3 @ AvatarReX
2 @ THuman
3 @ MVHuman100
2 @ MVHuman200
```

不能写成四个 `1 @ Dataset`，否则评测只覆盖 4 个事件。

10-event pilot 仍是小规模训练集 capacity test，不是正式 held-out 泛化结论。

---

## 7. 三阶段训练

### Stage A: smoke

目标：

- dataloader 输出 `[0, 0, 1]`；
- event frame 插入 2 个 correct tokens；
- context frame head LoRA 关闭；
- event frame head LoRA 开启；
- 只有 event frame 产生监督；
- forward/backward 无 NaN。

### Stage B: one-event overfit

目标：

- camera translation 显著低于 raw；
- camera rotation 显著低于 raw；
- human translation 显著低于 raw；
- 可视化中第一张 post-cut frame 的 camera/human 同向改善。

当前结果：

```text
40 epochs, GPU 2, 2m26s

loss:                  1.1098 -> 0.0099
camera translation:    1.1773 m -> 0.0891 m
camera rotation:       17.245 deg -> 0.0969 deg
human translation:     0.5238 m -> 0.0086 m
event gate:            1.0
```

checkpoint：

```text
output/v14_1/v14_1_cut_event_single_lbn1_1192/checkpoint-best.pth
```

结论：单样本纠正容量成立，但该结果是 overfit upper bound。

### Stage C: 10-event pilot

目标：

- 完整覆盖 10 个 events；
- 分来源报告 raw/corrected camera translation、rotation 和 human translation；
- 检查是否存在某个数据来源被其他来源牺牲；
- 与原版 Human3R、V9 和 single-event upper bound 做相同输入可视化。

正式大规模训练只能在 10-event pilot 显示稳定同向改善后开始。

2026-07-27 pilot 已完成；完整结果见：

```text
versions/v14/docs/V14_1_INITIAL_PILOT_RESULTS_20260727.md
```

---

## 8. 配置与命令

基础配置：

```text
config/train_v14_1_cut_event.yaml
```

单样本：

```bash
cd src
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=..:. \
  ../.venv/bin/python train.py --config-name train_v14_1_cut_event_single
```

10-event：

```bash
cd src
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=..:. \
  ../.venv/bin/python train.py --config-name train_v14_1_cut_event_ten
```

demo：

```bash
PYTHONPATH=src:. .venv/bin/python demo.py \
  --model_path <checkpoint.pth> \
  --seq_path <three-frame-directory> \
  --cut_indices 2 \
  --viewer_port 8093
```

`--cut_indices` 使用 zero-based frame index。

---

## 9. 必须报告的指标

每个来源至少报告：

```text
corrected camera translation error
raw camera translation error
corrected camera rotation error
raw camera rotation error
corrected human translation error
raw human translation error
event gate
pose-head LoRA norm
human-head LoRA norm
```

还必须报告：

- mean 与 median；
- 每个 event 的结果，而不只报告合并均值；
- 单样本 overfit 与 10-event pilot 分开；
- train events 与未来 held-out events 分开；
- 3D viewer 主观观察。

---

## 10. 进入 V14.2 的条件

只有以下条件同时成立，才继续研究 state/Boundary：

1. 单样本 camera 和 human 均可稳定过拟合；
2. 10 events 上大多数样本同方向改善；
3. 不只依赖一个数据来源；
4. event-only LoRA 不泄漏到 context frame；
5. viewer 中 camera 与 human 的改善一致，而不是只优化数值编码；
6. 不出现明显 mesh 崩坏或 scene/human 分离。

V14.2 应比较：

```text
A. corrected state commit
B. corrected output only, raw reset state commit
C. 从 corrected-vs-raw 第一帧提取固定 SE(3) Boundary
```

并验证 post-cut 后续帧，而不是继续增加单帧训练复杂度。

---

## 11. 当前结论边界

V14.1 当前可以声称：

> 在显式已知 cut 的条件下，V9 风格的 semantic/alignment correct tokens 可以只在第一张
> post-cut frame 进入 decoder，并通过 pose/human latent residual 与 event-only head
> LoRA 显著降低该帧的 camera 和 human error。

当前不能声称：

- 已获得 shot-persistent Boundary；
- 已解决 hard reset 后 world alignment；
- 已解决多人物身份；
- 已证明跨数据泛化；
- 已证明后续帧无需纠正；
- 已完成完整 V14。
