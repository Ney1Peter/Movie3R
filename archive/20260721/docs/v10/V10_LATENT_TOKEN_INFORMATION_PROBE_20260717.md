# V10 Human3R 隐式 Token 信息探针实验

日期：2026-07-17

## 1. 实验目的

本实验不设计最终 shot correction 网络，也不训练 Human3R 主体。目标是回答四个问题：

1. Camera、scene、human 和 recurrent state 中分别有哪些信息可以被小 probe 读取。
2. 这些信息在未见场景和 AvatarReX 跨数据集零样本上是否仍然成立。
3. Token 只是与输出相关，还是对 camera、pointmap、SMPL-X 具有真实因果作用。
4. 后续 Shot Prompt 应读取什么，并插入 decoder 的什么位置。

Human3R encoder、decoder、camera head、pointmap head 和 SMPL-X head 全部冻结。

## 2. 实现

独立实验脚本：

```text
scripts/v10_latent_token_cache.py
scripts/v10_latent_token_probe.py
scripts/v10_latent_activation_patching_probe.py
scripts/v10_latent_activation_patching_video_probe.py
scripts/v10_latent_cross_view_retrieval_probe.py
```

没有修改 Human3R 默认推理路径。实验脚本通过临时 forward hook 读取或替换 activation，退出上下文后恢复原函数。

缓存内容包括：

```text
24 层 Encoder image token
12 层 Decoder image token
12 层 Decoder state token
initial / refined camera token
human prompt / refined human token
persistent state / new state
Multi-HMR DINO token
边界前后少量 sampled patch token
```

每层完整 token 不全量落盘，只保存 mean+std pooled summary；边界前后各保存 16 个 sampled patch。最终 180 个 case 缓存为 474 MB。

严格原版推理中的 `cross_attn_states` 返回 `None`，因此本轮没有伪造 attention weight，而是分析每层 decoder state activation。若后续必须研究 attention map，需要在独立模型副本中显式暴露 attention 权重。

## 3. 数据与划分

使用与 Boundary Gauge Probe 相同的 180 个 AABB case：

| 数据源 | Case 数 | 用途 |
|---|---:|---|
| AvatarReX | 48 | 完全零样本测试 |
| THuman | 48 | 训练与未见 group 验证 |
| MVHuman100 | 48 | 训练与未见 group 验证 |
| MVHuman200 | 36 | 训练与未见 group 验证 |

划分不是随机按帧进行：

```text
训练：90 case
未见场景验证：42 case
AvatarReX 零样本：48 case
```

验证组为：

```text
THuman: thuman02
MVHuman100: 100005
MVHuman200: 200005
```

Camera 和 human 目标使用数据集标定及 world SMPL-X/body25 标注。

当前 180 个 case 没有统一验证过的静态背景 GT depth/mesh。Patch depth、world coordinate、normal 和 confidence 只能以 Human3R 自己的 pointmap 输出作为 head-readout 伪目标，不能当成真实 scene geometry GT，也不能报告真实 10/20/50 cm correspondence 正确率。

## 4. Token 可解码性

### 4.1 Probe 设置

比较：

```text
Linear Probe: PCA + Ridge / Logistic Regression
Small MLP: PCA + 64-32 两层 MLP
```

Linear 和 Small MLP 都覆盖了全部 24 层 encoder、12 层 decoder image、12 层 decoder state 以及 camera、human、state、DINO token。另有一份代表层快速结果用于调试，但最终结论使用全层结果。

Skill 定义为相对训练集常数预测器的误差改善：

```text
skill = 1 - probe_MAE / constant_baseline_MAE
```

`skill > 0` 表示优于常数基线，`skill <= 0` 表示没有稳定可读信息。

### 4.2 Boundary 物理信息

Linear Probe 的最佳结果：

| 目标 | 未见场景 skill | AvatarReX 零样本 skill | 判断 |
|---|---:|---:|---|
| Boundary rotation | 0.091 | 0.055 | 很弱，不足以直接回归旋转 |
| Translation direction | 0.121 | 0.185 | 有少量方向信息，但精度有限 |
| Translation norm | 0.622 | -0.892 | 训练域可读，跨数据集完全失效 |
| Explicit translation error | 0.708 | 0.618 | 可较稳定判断平移误差大小 |
| Explicit rotation error | 0.573 | 0.843 | 可稳定判断旋转误差大小 |

Small MLP 没有改变结论：

```text
AvatarReX boundary rotation 最佳 skill: 0.046
AvatarReX translation direction: 全部 MLP layer 均未超过常数基线
AvatarReX explicit rotation error 最佳 skill: 0.856
AvatarReX catastrophic failure 最佳 classification skill: 0.270
```

因此现有 token 不适合从零预测完整 SE(3)，但适合预测显式结果的误差、难度和可靠性。

### 4.3 Camera 信息

逐帧 camera absolute/relative rotation 和 relative translation 在未见场景上整体接近或低于常数基线。Camera token 对当前 camera head 输出有控制作用，但其中的物理量不容易跨场景直接解码。

这两点不冲突：

```text
camera token 是 head 的控制变量
不等于 camera token 是一个可跨数据集线性读取的显式 SE(3) 容器
```

### 4.4 Human 信息

Human world root 在 pooled image token 上可读：

```text
未见场景最佳 skill: 0.781
AvatarReX 零样本最佳 skill: 0.550
```

但 torso heading 基本不可读，未见场景最佳 skill 为 `-0.007`。因此 human/image token 能表达人体大致位置或图像布局，但当前没有证据表明 human token 能稳定提供跨镜头 torso yaw。

Human angular velocity 在部分 token 上看起来可读，但四个 source 大多为原地动作，目标分布较窄，不能据此证明真实长距离运动建模能力。

### 4.5 Scene head-readout 信息

Decoder patch token 对 Human3R 自己的 pointmap 输出具有强可解码性：

| 伪目标 | 未见场景 skill | AvatarReX 零样本 skill |
|---|---:|---:|
| Predicted depth | 0.804 | 0.699 |
| Predicted world coordinate | 0.790 | 0.710 |
| Predicted normal | 0.474 | 0.497 |
| Pointmap confidence | 0.691 | 0.435 |

最强层主要位于 decoder 中后段，world coordinate 在训练域以 `decoder_patch_l11` 最强，AvatarReX 则以 `decoder_patch_l04` 最强。

这说明 scene 几何信息确实在 decoder image token 中形成，但它只是证明 token 能被原 head 读取，不能证明跨相机物理对应稳定。

## 5. 跨视角全局稳定性

Training-free 全局检索使用 cut 后 token 查询 cut 前 token：

```text
最佳 exact-case Recall@1: 6.1%
最佳 exact-case Recall@5: 18.9%
最佳 exact-case median rank: 18-24
```

这表明 token 不能可靠找回精确的跨相机 case。

同 group 检索明显更高：

```text
encoder_image_l21 same-group excluding-self Recall@1: 80.6%
Recall@5: 97.8%
```

但 group 同时包含 capture session、人物和场景风格，所以这更像粗粒度 session/identity/style 信息，不是精确物理重定位。Human prompt/refined token 的 exact-case Recall@1 只有 `1.1%/0.6%`，没有优于 image token。

此前 6 个 AvatarReX case 的局部 token matching probe 也得到：

```text
token confidence 最高接近 0.99
oracle 诊断下物理正确 match 比例均值约 0.67%
```

综合两组结果，raw token 不适合直接做 patch nearest-neighbor 或从匹配点求 SE(3)。

## 6. Activation Patching 因果实验

### 6.1 设置

使用电影片段连续 16 帧，在第 9 帧前人为 reset：

```text
Teacher: 前 8 帧正常连续运行
Student: 从第 9 帧 fresh state 运行
边界处 Teacher 与 Student 输入完全相同 RGB
只替换 latent activation
评估边界帧和后续 7 帧
```

Recovery Ratio：

```text
(reset error - patched error) / reset error
```

### 6.2 结果

Reset raw 相对 continuous teacher 的平均误差：

```text
camera translation: 0.3449 m
camera rotation: 5.47 deg
world pointmap: 1.0301 m
human world root: 0.8240 m
```

核心结果：

| Patch | Camera R recovery | World pointmap recovery | Human root recovery |
|---|---:|---:|---:|
| Refined camera token only | 0.239 | 0.005 | 0.001 |
| Persistent state only | 0.690 | 0.675 | 0.705 |
| Initial camera token + state | 0.813 | 0.857 | 0.896 |
| Final scene token + state | 0.690 | 0.676 | 0.705 |
| All key tokens | 0.813 | 0.857 | 0.896 |

最关键的逐帧现象：

1. 只替换 refined camera token 时，边界帧 camera error 接近零，但下一帧立刻重新漂移。
2. 只替换 persistent state 时，边界帧没有完全恢复，但后续帧 camera、scene 和 human 明显持续接近 teacher。
3. 同时替换 teacher 的 initial camera token 和 persistent state，边界帧几乎完全恢复，后续 7 帧也保持大幅改善。
4. 加入 final scene/human token 后几乎没有超过 `camera + state`，说明主要因果变量是 state 与 decoder 输入端 camera token。

这证明：

```text
refined camera token 控制当前帧输出
persistent state 控制后续 recurrent trajectory
initial camera token + state 共同决定当前 gauge 与后续传播
```

### 6.3 控制实验

静态 AvatarReX case 的控制结果：

```text
self-token replacement: 近似 0 变化
random final scene token: world pointmap recovery -8.70
spatially shuffled scene token: world pointmap recovery -8.14
other-video key tokens: world pointmap recovery -1.98
```

因此恢复不是 hook 本身造成，也不是任意 token 替换都能产生。正确历史 state/camera activation 具有内容特异性。

Decoder image token 单层 patch 在运动视频上最高只恢复约 `19.7%` world pointmap，并且不同层结果不稳定。Scene token 有几何控制作用，但单独不足以恢复 world gauge。

## 7. 最终判断

### A. Camera token

Refined camera token 是当前帧 camera head 的直接控制点，但只修它不会传播到未来。后续 Shot Prompt 不能只放在最终 camera head 前做一次 residual。

### B. Persistent state

Persistent state 是当前最强的 world-context 因果载体。它能同时恢复 camera、scene 和 human，并把作用传播到后续帧。

但是最终方法不能在真实 camera cut 后继续写旧 state。合理设计应是：

```text
旧 state 只读
fresh image/state 正常建立新 shot
Shot Prompt 查询旧 state 和 fresh token
生成受限 camera initialization / state-write guidance
后续只更新 fresh state
```

### C. Image/scene token

Decoder image token 确实编码 pointmap head 所需的局部几何，但跨视角 exact retrieval 和局部匹配不稳定。它适合作为 state-query 的当前观测，不适合直接做 raw nearest-neighbor correspondence。

### D. Human token

Human token 可以作为人体存在、粗位置、身份/session 和运动辅助信息，但当前没有证据支持它独立恢复 torso heading 或完整 SE(3)。

### E. Reliability

Token 对 explicit translation/rotation error 的回归明显强于对 SE(3) 本身的回归。即使最终不做 latent gauge correction，token 仍适合：

```text
显式候选评分
失败预测
safe fallback
是否等待额外帧
```

## 8. 推荐 Shot Prompt 架构

下一版最值得验证的是小型 read-only state-query 模块：

```text
cut 前只读 persistent state summary
        +
cut 后 fresh encoder/decoder image observation
        +
fresh human token，可选辅助
        ↓
cross-attention / gated projector
        ↓
1. initial camera token 的受限 residual
2. fresh state 第一次 write 的受限 guidance
3. reliability / fallback score
```

插入位置优先级：

1. Decoder 前或最早层，修正 initial camera token。
2. 第一次 fresh state write 前，修正 state update，而不是直接覆盖整份 state。
3. 不优先在最终 camera output 后做大 residual。

训练目标可以来自连续视频的 teacher-reset 自蒸馏：

```text
Teacher: 正常连续 state
Student: 同一帧前 reset
监督 Student 的 camera token、new state 和最终 camera/scene/human 输出恢复 Teacher
```

这种训练不需要真实跨分镜 GT，也不会要求 raw token 具有显式三维坐标。

## 9. 输出

```text
output/v10_latent_token_probe/token_cache/
output/v10_latent_token_probe/probe_results_linear/
output/v10_latent_token_probe/probe_results_mlp_full/
output/v10_latent_token_probe/probe_results_mlp_selected/  # 代表层快速调试
output/v10_latent_token_probe/cross_view_retrieval/
output/v10_latent_token_probe/moving_video_clip01_f090_105/activation_patching/
output/v10_latent_token_probe/activation_patching_static_control/
```

主要图：

```text
activation_recovery_matrix.png
decoder_layer_recovery_curves.png
boundary_linear_zero_shot_avatarrex_heatmap.png
patch_linear_zero_shot_avatarrex_heatmap.png
cross_view_retrieval_top15.png
```
