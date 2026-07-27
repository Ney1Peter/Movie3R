# Movie3R V9 / V12 / V13 统一技术总结

> 文档用途：给后续 AI、合作者或论文分析会话提供一个自包含的技术入口。
>
> 更新日期：2026-07-27。
>
> 正式版本：Movie3R-Learned V9.0、Movie3R-Single V12.0、
> Movie3R-Multi V13.0。
>
> 重要提醒：V10、V11、V14、V16、V20 是历史实验或组件编号，不是当前并列发布版本。

## 0. 项目总目标与统一术语

### 0.1 Movie3R 要解决什么问题

Movie3R 建立在 Human3R 之上，目标是处理单目、多分镜、存在 camera cut 的人体三维视频。
系统希望在严格流式、因果条件下，逐帧输出：

- camera pose；
- scene depth / pointmap；
- SMPL-X、人体 joints 和 vertices；
- 跨镜头尽可能一致的 world coordinate system。

Human3R 在普通连续镜头内通常可以稳定地联合预测 camera、scene 和 human，但 camera cut
会同时产生两个不同问题：

1. **State contamination**：新镜头如果继续读取旧 recurrent state，旧镜头的
   scene/camera memory 会污染新镜头第一帧和后续轨迹。
2. **World-gauge discontinuity**：即使在 cut 处 hard reset，新镜头也只会得到一个干净但
   独立的 shot-local reconstruction。它和旧 world 之间仍可能存在 rotation、translation
   和 metric scale 差异。

因此，当前最清楚的问题分解是：

```text
camera cut
-> local recurrent-state transition
-> fresh shot-local reconstruction
-> shot-to-world re-anchoring
-> global camera + human + scene output
```

V9、V12、V13 对这个问题采用了三种不同研究路线：

| 版本 | 主要思想 | 是否训练新增模块 | 主要研究对象 | 当前状态 |
|---|---|---:|---|---|
| V9 | 在 Human3R decoder 内用 relation prompt 学习 camera/human latent correction | 是 | 4-frame AABB 学习式纠正 | 已训练并冻结 |
| V12 | cut 时 hard reset，再显式求一个单人 shot-level similarity Boundary | 否 | 单人 short-shot camera-human alignment | 当前单人主版 |
| V13 | 多个人分别提供 Boundary candidate，再融合成一个 shared Boundary | 否；身份探针也保持冻结 | GT-ID 多人几何与自动身份可行性 | 几何通过，自动 WHO 未通过 |

### 0.2 术语

#### AABB / AAAA

- `AABB`：四帧训练模式，前两帧来自镜头/视角 A，后两帧来自镜头/视角 B，用于模拟
  camera cut 或跨视角跳变。
- `AAAA`：四帧均来自连续稳定镜头，用于约束模型不要过度纠正正常视频。

#### Shot-local gauge

Human3R 在一个 fresh shot 中输出的 camera、pointmap 和 human 共享一个局部坐标系，但该
坐标系不保证与上一个 shot 的 world gauge 一致。

#### Boundary

Boundary 是把新 shot-local reconstruction 接回旧 predicted world 的固定变换。V12 使用
similarity：

```text
X_world = R * (s * X_local) + t
```

V13 为隔离多人几何贡献，固定 `s=1`，使用 shared rigid Boundary：

```text
X_world = R * X_local + t
```

#### One shared Boundary

同一个 shot 中的 camera、pointmap、所有 human roots、body offsets、joints 和 vertices
必须使用同一组变换。禁止 camera、scene 和 human 各用一套看似更优、实际互不一致的
world transform。

#### Align-Then-Commit

新 shot 的当前预测必须先完成 Boundary alignment，再写入长期 world memory 或 identity
memory。未经 alignment 的 local prediction 不能污染长期世界状态。

### 0.3 共同基础模型 Human3R

三个版本都以 `src/human3r_896L.pth` 为基础。其 SHA-256 为：

```text
1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377
```

Human3R 的抽象流程为：

```text
RGB
-> image encoder / image tokens
-> recurrent encoder-decoder state
-> pose token + image tokens + human tokens
-> camera head      -> camera pose
-> scene head       -> pointmap/depth/confidence
-> human head       -> SMPL-X/root/joints/vertices
```

V9 修改 decoder 输入和 pose/human latent；V12/V13 冻结 Human3R，只控制 cut 时的 state
生命周期并在输出侧求显式 Boundary。

---

# 1. Movie3R-Learned V9.0

## 1.1 版本定位

V9 是一个已经完成训练并冻结保存的学习式版本。它的目标是让 Human3R 在 AABB camera-cut
模式中自动判断是否发生 drift，并用很小的 latent residual 同时纠正 camera pose 和 human
placement。

V9 的核心问题意识是：

```text
只修 camera pose 不够。
Human3R 的 human head 还有自己的 smpl_transl / human latent placement。
因此 camera 和 human 必须同时纠正。
```

V9 不包含 V12 的 pre-decode hard reset、显式 Boundary、V16 torso rotation、DA3 shared
scale，也不包含 V13 的多人 consensus。

## 1.2 原理想法

V9 借鉴 UniCon3R 的 contact-token 范式。UniCon3R 为 human-scene contact 构造 relation
prompt，让 prompt 进入 decoder 与原始 tokens 交互，然后用 refined prompt 预测 human latent
residual。V9 将这个范式改造成：

```text
contact relation prompt
-> camera-human alignment relation prompt

只修 human
-> 同时修 pose token 和 human token
```

V9 不直接回归新的 4x4 camera matrix，也不直接把 SMPL-X 平移硬改成某个数。它先构造
一组 correction prompt tokens，让它们与 Human3R 原 token 在 decoder 中交互，再输出：

- pose-token latent residual；
- human-token latent residual；
- learned correction gate。

最终 camera 和 SMPL-X 仍由 Human3R 原有 heads 解码。

该设计希望保留 Human3R 已学到的重建能力，只学习“什么时候修、在 latent space 中往哪里
小幅移动”。

## 1.3 输入

### 1.3.1 推理时外部输入

V9 runner 接收连续 RGB 图像流。正式训练的主要分布是四帧：

```text
frame 0: A
frame 1: A
frame 2: B
frame 3: B
```

主要输入条件：

- 4-frame RGB；
- Human3R 标准图像预处理；
- `resize_only_16`，不做强制正方形 padding；
- 单人主训练设置 `max_humans=1`；
- Human3R recurrent state 和 pose memory；
- 前一帧 correction history。

推理不读取：

- GT camera；
- GT SMPL-X；
- GT shot label；
- GT depth；
- DA3 depth；
- 未来帧。

虽然当前 runner 可以接收更长序列，但 25 帧、多次 cut 属于训练分布外压力测试，不能替代
V9 的 4-frame AABB 结论。

### 1.3.2 模型内部输入

第 `t` 帧构造 correction prompt 时使用：

| 符号 | 内容 |
|---|---|
| `F_t` | 当前 image tokens |
| `z_t` | 当前 pose token |
| `H_t` | 当前 human tokens |
| `S_t` | Human3R recurrent state memory |
| `M_t` | pose retriever memory |
| `A_corr,t-1` | 上一帧 refined correction prompt |
| `Delta z_t-1` | 上一帧 pose-token correction |
| `g_t-1` | 上一帧 correction gate |

形式上：

```text
A_corr,t = Phi_corr(
    F_t, z_t, H_t, S_t, M_t,
    A_corr,t-1, Delta z_t-1, g_t-1
)
```

### 1.3.3 训练监督输入

GT 只进入 loss、metric 和 viewer overlay：

- AvatarReX camera 使用 raw calibration c2w；
- 四帧 target 为 `inv(raw_camera_pose_0) @ raw_camera_pose_i`；
- THuman 使用已经验证的官方 camera/SMPL gauge；
- GT human translation 用于 human correction loss。

训练时 `load_da3_depth=False`。DA3 pseudo depth 不能作为跨相机 metric GT。

## 1.4 模型架构设计

### 1.4.1 冻结配置中的 Human3R 主体

V9 resolved config 中的关键结构为：

| 项目 | 配置 |
|---|---|
| 模型类 | `ARCroco3DStereo` |
| state size | 768 |
| encoder dim/depth/heads | 1024 / 24 / 16 |
| decoder dim/depth/heads | 768 / 12 / 12 |
| image backbone | `dinov2_vitl14` |
| Multi-HMR image resolution | 896 |
| output mode | pointmap + pose + SMPL-X |
| views | 4 |

V9 使用原始 Human3R 权重初始化，训练 correction branch，并给 pose/human heads 添加 LoRA，
而不是全量解冻 backbone 和 decoder。

### 1.4.2 Relation correction prompt builder

`A_corr,t` 由三个 relation tokens 组成：

```text
A_corr,t = [a_sem,t ; a_align,t ; a_mom,t]
```

#### Semantic token

它聚合当前 image/pose/human context 与历史 state/pose/correction memory：

```text
current_token = Attention(query, current context)
memory_token  = Attention(query, memory context)

gamma_sem = sigmoid(MLP([current_token, memory_token]))

a_sem,t = gamma_sem * current_token
        + (1 - gamma_sem) * memory_token
```

职责：判断当前视觉、人、pose 与历史世界是否一致。

#### Alignment token

```text
a_align,t = MLP([z_t, z_t-1, z_t-z_t-1, memory_token])
```

职责：判断 pose latent 的变化是否像正常连续相机运动，还是发生了异常跳变。

#### Momentum token

```text
a_mom,t = MLP([A_corr,t-1, Delta z_t-1, g_t-1])
```

职责：将上一帧“是否修过、修了多少”的状态带到当前帧，形成流式 correction prior。

三个 token 加 type embedding 和 LayerNorm 后进入 decoder。

### 1.4.3 Decoder 接入

decoder token 顺序为：

```text
X_dec,t = [z_t ; A_corr,t ; F_t ; H_t]
```

`A_corr,t` 放在 pose token 后，既保留 Human3R 第 0 个 token 是 pose token 的原约定，又允许
correction prompt 与 pose/image/human tokens 发生 attention 交互。

整体数据流：

```text
RGB frames
   |
   +-> Human3R image encoder -> F_t
   +-> Multi-HMR branch      -> H_t
   +-> pose/state pathway    -> z_t, S_t, M_t
                                |
history corr/delta/gate --------+
                                v
                  Relation Prompt Builder Phi_corr
                                |
                 A_corr,t = semantic/alignment/momentum
                                |
             [z_t ; A_corr,t ; F_t ; H_t]
                                |
                  Human3R recurrent decoder
                                |
        +-----------------------+-----------------------+
        |                       |                       |
 refined pose token      refined corr tokens     refined human tokens
        |                       |                       |
        |             pose residual + gate              |
        |                       |                       |
        +-> corrected pose token                        |
        |                                               |
        +---------------- corrected pose context --------+
                                                        |
                                           human latent residual
                                                        |
                                           corrected human token
        +----------------------+------------------------+
        |                      |                        |
     pose head             scene head               human head
        |                      |                        |
    camera pose          pointmap/conf             SMPL-X
```

### 1.4.4 Camera correction branch

refined correction prompt 先做 pooling，再输出 pose latent residual 和 gate：

```text
Delta z_raw = MLP(pool(A_tilde_corr,t))
g_t         = sigmoid(drift_logit)
Delta z_t   = g_t * Delta z_raw
z_hat_t     = z_tilde_t + Delta z_t
T_hat_t     = PoseHead(z_hat_t)
```

关键点：Pose Correct Head 不直接输出 camera SE(3)，而是修正 pose token，camera pose 仍由
Human3R pose head 输出。

### 1.4.5 Human latent correction branch

V9 主线关闭直接 `smpl_transl += delta` 的显式诊断分支，启用 implicit human latent
correction：

```text
context = [H_tilde_t, pool(A_tilde_corr,t), z_hat_t]
Delta H_raw = HumanCorrectHead(context)
Delta H_t   = g_t * Delta H_raw
H_hat_t     = H_tilde_t + Delta H_t
SMPL_hat_t  = HumanHead(H_hat_t)
```

human branch 默认共享 pose branch 的 gate。这样 camera 和 human correction 在触发时机上
耦合，但各自拥有不同 latent residual。

### 1.4.6 LoRA

冻结版本开启：

```text
pose head LoRA  = on
human head LoRA = on
rank            = 8
alpha           = 8
dropout         = 0
```

LoRA 只给原 pose/human heads 提供小幅适配能力。它与 correction prompt 不是同一模块：

- correction prompt/head 决定 latent residual；
- LoRA 让原 head 更好地解释 corrected latent。

## 1.5 输出

V9 每帧输出：

- corrected camera pose；
- Human3R pointmap/depth/confidence；
- corrected SMPL-X parameters；
- corrected human root translation；
- joints、vertices、mask；
- correction gate 和 latent residual diagnostics；
- recurrent state，供后续帧使用。

V9 没有独立显式 scene correction head，也没有 shot-level fixed Boundary。它主要通过 decoder
relation prompt、pose token 和 human token 改善 camera-human alignment。

## 1.6 训练设计

### 1.6.1 正式冻结训练

正式训练配置：

```text
config/train_v9_mixed_avatarrex_thuman_60h_pose_human_lora_bs10.yaml
```

主要设置：

| 项目 | 设置 |
|---|---|
| 初始化 | 原版 `human3r_896L.pth` |
| 数据 | AvatarReX + THuman |
| source 权重 | AvatarReX 0.6，THuman 0.4 |
| 每 source batch | 10 |
| views | 4 |
| epoch | 72 |
| steps/epoch | 100 |
| optimizer steps | 7200 |
| resize | `resize_only_16` |
| gradient checkpointing | 开启 |
| 主训练样本 | AABB + AAAA |

每个 source 的规划训练组成是 `8000 AABB + 2000 AAAA`。AvatarReX 和 THuman 分辨率
分别 forward/backward，再合并梯度做一次 optimizer step，避免强制 padding 到同一比例。

### 1.6.2 Loss

冻结 resolved config 的主要损失包括：

| Loss | 职责 |
|---|---|
| pose translation/rotation | corrected camera 接近 GT |
| drift/gate loss | gate 学会区分 drift 与稳定帧 |
| improvement loss | corrected camera 应优于 raw Human3R |
| pose residual regularization | 防止 pose latent residual 无界增大 |
| human translation loss | corrected human placement 接近 GT |
| human latent delta regularization | 保持 human correction 为 residual-like |
| LoRA norm regularization | 限制 head 适配幅度 |

冻结正式 config 的关键权重是：translation `1.0`、rotation `5.0`、drift `0.05`、
improvement `0.05`、human translation `10.0`。

## 1.7 实验结果

### 1.7.1 单 clip 能力验证

AvatarReX 单 clip overfit：

| 指标 | Raw Human3R | Corrected V9 |
|---|---:|---:|
| Human translation error | 0.708 m | 0.0037 m |
| Camera translation error | 0.592 m | 0.016 m |
| Camera rotation error | 24.04 deg | 0.119 deg |
| Gate mean | - | 0.339 |

该实验说明 architecture 和监督链路具备纠正能力，但单 clip overfit 不是泛化结论。

### 1.7.2 60h loss-sweep 中的 H3 结果

H3 是和正式 V9 同架构、同 AvatarReX+THuman 60h setting 的 loss 变体。它使用 deadzone gate、
meaningful-improvement margin 和更弱的 human-delta regularization。H3 不是
`versions/v9/manifest.json` 指向的冻结正式 config，因此以下数值应作为 V9 架构的 loss
消融证据，不能直接冒充 `checkpoint-final.pth` 的独立最终表。

| Dataset/subset | Camera T raw -> corrected | Camera R raw -> corrected | Gate | Human T raw -> corrected |
|---|---:|---:|---:|---:|
| AvatarReX AABB | 0.3136 -> 0.1400 m | 5.20 -> 4.27 deg | 0.499 | 0.0640 -> 0.0333 m |
| AvatarReX AAAA | 0.0043 -> 0.0040 m | near zero | 0.009 | stable |
| THuman AABB | 0.1688 -> 0.0464 m | 未单列 | 0.445 | 0.0655 -> 0.0581 m |
| THuman AAAA | 0.0022 -> 0.0022 m | near zero | 0.001 | stable |
| zxc held-out AABB | 0.2559 -> 0.1617 m | 5.28 -> 4.81 deg | 0.498 | 0.1048 -> 0.1247 m |
| zxc held-out AAAA | 0.0037 -> 0.0036 m | near zero | 0.009 | stable |

该表证明：

- learned gate 可以在 AABB 上开启、AAAA 上接近关闭；
- camera correction 可以泛化到 zxc held-out；
- held-out human translation 可能反而变差，说明 camera 与 human correction 并未形成稳定的
  完整 world closure。

### 1.7.3 Token/pooling 消融

`benchmark_mixed_small18` 中：

| Variant | AABB camera T | Improvement | Human T | AAAA gate | Loss |
|---|---:|---:|---:|---:|---:|
| global weighted | 0.580 -> 0.403 m | 0.177 m | 0.271 -> 0.120 m | 0.147 | 1.412 |
| all-concat/contact-style | 0.487 -> 0.274 m | 0.213 m | 0.297 -> 0.116 m | 0.055 | 0.892 |

小规模结果支持保留 semantic/alignment/momentum 的完整信息，而不是过早压成单 token。
但较大训练中 concat 也出现 AAAA 过修风险，所以冻结正式版本仍使用稳定 baseline。

### 1.7.4 冻结身份

| 项目 | 值 |
|---|---|
| release | Movie3R-Learned V9.0 |
| commit | `6eb64cb2158fb443d53cd4f1713af1899fe5a026` |
| tag | `movie3r-v9-trained` |
| 推荐权重 | `checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth` |
| checkpoint SHA-256 | `3fb2799420f7fd3caa63a47c9cde73090a6f93383520363484eb5158e446fceb` |

现有正式文档没有为该 `checkpoint-final.pth` 单独整理一张包含 camera、human、MPJPE、PVE
的统一最终 benchmark 表。后续 AI 做论文数值分析时，应把“冻结权重存在”和“消融实验中
出现过某个数字”分开处理。

## 1.8 V9 已证明、未证明与主要限制

已证明：

- relation prompt 中存在可学习的 camera-human correction signal；
- pose-only 不够，需要 human latent correction；
- AABB/AAAA 联合训练和 learned gate 能减少正常帧过修；
- latent residual + 原 Human3R head 是可行的学习式接入方式。

未证明或限制：

- 训练分布主要是 4-frame AABB，不是长期多 cut；
- recurrent state 在 cut 后仍可能被污染；
- camera 指标改善不保证 human/scene 一起改善；
- token correction 上限受训练数据、GT 和 latent 压缩影响；
- V9 没有显式解决 shot-level world gauge。

---

# 2. Movie3R-Single V12.0

## 2.1 版本定位

V12 是当前效果优先的单人 camera-cut 主版，准确范围是：

> Short-horizon camera-human re-anchoring after sparse camera cuts.

它冻结 Human3R，不训练新的 SE(3) 网络，而是在 cut 时对 fresh shot-local reconstruction
显式求一次 similarity transform。历史 V10.1、V16、V11.4 和 V14.x 都是 V12 的组件来源
或审计编号。

## 2.2 原理想法

V12 的核心判断是：camera cut 不是一个普通 per-frame refinement 问题，而是：

```text
旧 state 是否应该继续存在？
+
新 shot 的 local gauge 如何接回旧 world？
```

因此它明确分离：

- scene/camera local state：cut 时必须 reset；
- human physical motion/history：允许作为 Human3R 外部的跨 shot anchor；
- fresh post-cut reconstruction：负责新 shot 内局部几何；
- Boundary：只在新 shot 开始时估计一次，整个 short shot 固定复用。

当前默认方法只包含两个主要 alignment correction blocks：

1. V16 bounded torso-motion rotation；
2. V11.4 fused shared shot scale。

Fixed Explicit 提供 coarse initialization/fallback，translation 由显式方程求解。DA3 和
Keypoint R-CNN 是 V11.4 内部 cue，不是两个独立 Boundary networks。

## 2.3 输入与状态

### 2.3.1 外部输入

每帧输入：

- streaming RGB image `I_t`；
- 与 resize/crop 一致的 intrinsics `K_t`；
- Human3R recurrent state；
- external cut trigger `c_t`；
- 已经到达的历史帧，不读取未来 shot。

主评测预处理：

- Human3R resolution：`512 x 288`；
- resize mode：`human3r_demo`；
- DA3 process resolution：504；
- `max_humans=1`。

当前使用 GT cut index 作为**触发信号**。它只告诉系统 cut 在哪里，不提供 camera、human、
rotation、translation 或 scale。自动 cut detector 尚未作为 V12 的验证结论。

### 2.3.2 Cut 前保留的外部历史

允许保留：

- 上一帧/上一 shot 的 predicted world human root；
- pre-cut torso orientation/motion history；
- 上一 shot 已确定的 scale/gauge；
- 可选 canonical human memory。

必须删除：

- old scene recurrent state；
- old camera recurrent state；
- old shot-local decoder history。

## 2.4 完整模型架构设计

### 2.4.1 总体数据流

```text
Streaming RGB + intrinsics + external cut trigger
                         |
                         v
                 Frozen Human3R
                         |
           +-------------+-------------+
           |                           |
       no camera cut                camera cut
           |                           |
 original Human3R path       save allowed world/human history
           |                 reset local state before decode
           |                           |
           |                 fresh post-cut Human3R output
           |                           |
           |                 Fixed Explicit coarse anchor
           |                           |
           |                 V16 torso-motion rotation
           |                    bound = 20 deg
           |                           |
           |           +---------------+---------------+
           |           |                               |
           |   Keypoint R-CNN                   DA3Metric-Large
           |   human/root pixels               metric depth cue
           |           |                               |
           |           +---------------+---------------+
           |                           |
           |                 V11.4 fused shot scale s
           |                           |
           |              explicit translation solve
           |                           |
           |               one fixed Boundary B
           |                           |
           +---------------------------+
                                       |
                   camera + pointmap + complete SMPL-X
                   all transformed into one world gauge
                                       |
                            optional continuity
                                       |
                              Align-Then-Commit
```

### 2.4.2 Normal-frame 路径

无 cut 时运行原始 Human3R recurrent path。审计中：

```text
camera max diff   = 0
pointmap max diff = 0
SMPL-X max diff   = 0
```

因此 V12 的新增逻辑不应改变正常连续帧。

### 2.4.3 Pre-decode Hard Reset

正确时序：

```text
detect cut
-> preserve only allowed external history
-> clear Human3R recurrent state
-> decode first post-cut RGB from fresh state
-> estimate Boundary from current output and past history
-> transform and emit current output
```

必须在第一张 post-cut frame 进入 recurrent decoder 前 reset。只在最终输出层修正一个已经读取
旧 state 的结果，无法消除后续 trajectory contamination。

### 2.4.4 Fixed Explicit coarse anchor

输入：

- pre-cut human root rotations/translations；
- first post-cut fresh human root；
- pre-cut non-human/background pointmap history；
- first post-cut background pointmap。

处理：

1. 对 pre-cut root rotations 求稳健平均；
2. 对 pre-cut root translations 求 median；
3. 用历史人体目标和 post-cut current root 生成 coarse rigid transform；
4. 用稀疏背景 pointmap 做小范围、固定预算的 robust refinement。

输出：coarse Boundary rotation/translation，以及 V16 的初始化。V16 后会重新显式求最终
translation，因此 Fixed 的初始 translation 不等于最终 translation。

Fixed 也是 post-cut human 无效时的 fallback。当前尚未完成“另一种 coarse initializer +
同一 V16/V11.4”的 clean replacement ablation。

### 2.4.5 V16 torso-motion rotation

核心假设：camera observation 在 cut 处不连续，但人的物理运动在时间上连续。

输入：

- Fixed coarse rotation；
- pre-cut Human3R/SMPL-X 3D torso frames；
- first post-cut fresh torso frame；
- fixed correction bound `20 deg`。

处理：

1. 从 pre-cut torso frames 估计 robust angular motion；
2. 外推 cut 后当前时刻的 target torso heading；
3. 比较 Fixed 映射后的 current torso 与 target torso；
4. 求 bounded yaw/heading residual；
5. 在 Fixed rotation 上应用一次 correction。

V16 只负责 rotation，不预测 scale，不直接回归 translation，也不逐帧重估。

### 2.4.6 Keypoint R-CNN cue

冻结 Torchvision Keypoint R-CNN 在 cut/reference RGB 上输出：

- person bbox；
- 17 个 COCO keypoints；
- detection/keypoint confidence。

它的职责是定位 pelvis/root/torso pixels，让 DA3 从正确人体位置读取 metric depth。它不直接
输出 Boundary，也不参与 V16。

当前证据没有证明 Torchvision Keypoint R-CNN 本身不可替代；Human3R projected joints 或
更轻 detector 仍可能替换它。

### 2.4.7 DA3Metric-Large cue

DA3 只在 shot/cut reference frame 上运行，不逐帧运行。它提供两类 scale evidence：

#### Human-root cue

在 Keypoint R-CNN 的 torso/root pixels 读取 DA3 metric depth，与 Human3R raw root depth
比较，得到 `s_h`。

#### Background cue

排除人体 mask，在 Human3R 高置信背景像素上计算：

```text
s_bg = robust median(DA3_depth / Human3R_depth)
```

DA3 background-only 和 human-root-only 在公平消融中都没有达到显著 camera gain。保留的是
它们组成的 V11.4 joint rule。

### 2.4.8 V11.4 fused uniform shot scale

定义：

```text
s_h  = human/root metric scale
s_bg = background median scale
q    = s_bg / s_h
```

安全限制：

```text
s_h clipped to [0.35, 3.0]
s_bg_bounded = s_h * clip(q, 0.85, 1.15)
```

冻结规则：

```text
if q < 0.95:
    s = s_bg_bounded
else:
    s = s_h
```

若 background 有效像素不足，则 fallback 到 human-root scale。

同一个 `s` 必须共同缩放：

- camera relative translation；
- pointmap；
- SMPL-X camera-frame root；
- root-centered body offsets；
- joints；
- vertices。

只缩放 root 而不缩放身体会破坏人体尺寸、接触和投影。围绕 camera origin 同比缩放完整
3D geometry 时，perspective projection 理论上保持不变：

```text
pi(K * (sX)) = pi(K * X)
```

审计中的 projection invariance max error 为约 `9.35e-6 px`。

### 2.4.9 Explicit translation 与 Boundary

定义：

- `C_0^L`：first post-cut Human3R local camera-to-world pose；
- `s`：shot scale；
- `R_B`：Fixed + V16 得到的 Boundary rotation；
- `r_0^C`：post-cut raw camera-frame human root；
- `a_pre^W`：pre-cut predicted world human anchor。

先得到最终 camera rotation：

```text
R_C^W = R_B * R_C^L
```

共享尺度人体 root：

```text
r_scaled^C = s * r_0^C
```

显式 camera translation：

```text
c_C^W = a_pre^W - R_C^W * r_scaled^C
```

于是 first post-cut target camera pose 为：

```text
C_0^W = [R_C^W, c_C^W]
```

最终 Boundary：

```text
B = C_0^W * inverse(ScalePose(C_0^L, s))
```

该方程使 first post-cut final world root 闭合到 `a_pre^W`：

```text
r_world = R_C^W * r_scaled^C + c_C^W = a_pre^W
```

因此 human-root 指标主要受 pre-cut anchor/motion model 控制，不能把 root gain 全归因于
V11.4 scale。

### 2.4.10 Shot 内固定复用

对后续第 `i` 帧：

```text
C_i^W       = B * ScalePose(C_i^L, s)
X_scene_i^W = B * (s * X_scene_i^L)
```

human root、body offsets、joints 和 vertices 使用相同 `B` 与 `s`。系统不逐帧重估
Boundary，不做 BA、loop closure 或完整 trajectory optimization。

### 2.4.11 默认关闭的模块

#### V14.2 continuity memory

只在 alignment 后平滑 canonical shape、body scale 和 root-centered local pose。它改善
continuity，但不改善 Boundary accuracy，默认关闭。

#### Conditional VGGT

可在困难 wide-baseline case 中提供 rotation-tail candidate，不预测 scale/translation。
它有效但增加 checkpoint、显存、cut latency，并存在少量 harmful trigger，默认关闭；只有
显式 `--enable_vggt` 才运行。

#### V14.3/V14.4 coupled-root branches

它们改善投影方程 closure，但在统一 world metrics 中没有超过 V11.4 raw-root 主线，保留为
诊断和负结果。

## 2.5 输出

每帧最终输出：

- world-space camera-to-world pose；
- Boundary `s/R/t`；
- world-space pointmap/depth/confidence；
- complete SMPL-X parameters；
- world root/global orientation；
- joints 和 vertices；
- cut index、scale cue、V16 correction、fallback 状态等 diagnostics。

输出重点是 camera-human placement。V12 不会自动修复 Human3R 本身已经存在的 local
SMPL-X pose、脚地、悬空或 pointmap depth error。

## 2.6 训练方式

V12 是 training-free/frozen geometry pipeline：

| 部件 | 是否训练 |
|---|---:|
| Human3R | 否 |
| Human3R encoder/decoder/heads | 否 |
| Keypoint R-CNN | 否，预训练冻结 |
| DA3Metric-Large | 否，预训练冻结 |
| Fixed Explicit | 无可学习参数 |
| V16 | 固定几何规则和 20 deg bound |
| V11.4 | 固定融合规则 |
| Continuity | 固定 EMA/blending，默认关闭 |
| VGGT | 预训练冻结，默认关闭 |

V12 的研究贡献主要是状态路由、模态连续性职责分离、显式共享几何约束和固定 Boundary
协议，而不是训练一个新 backbone。

## 2.7 实验结果

### 2.7.1 180-cut 四源统一评测，VGGT off

数据：

| Source | Cases |
|---|---:|
| AvatarReX | 48 |
| THuman | 48 |
| MVHuman100 | 48 |
| MVHuman200 | 36 |
| 总计 | 180 |

Scene 共同有效子集为 `147/180`。

| Method | Camera T mean/P90/P95 | Rotation | Root | Joints | Vertices | Scene | Camera success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Explicit | 0.712/1.509/1.703 m | 24.20 deg | 0.234 m | 0.290 m | 0.285 m | **0.483 m** | 41.1% |
| Fixed + V16, raw scale | 0.518/0.934/1.314 m | **16.04 deg** | **0.163 m** | **0.223 m** | **0.215 m** | 0.526 m | 55.0% |
| V12 Full / V11.4 fused scale | **0.463/0.918/1.088 m** | **16.04 deg** | **0.163 m** | 0.225 m | 0.218 m | 0.536 m | **60.6%** |

V16 相对 Fixed：

- camera translation `-0.194 m`，`p=2.20e-14`；
- rotation `-8.17 deg`，`p=3.10e-15`；
- human joints `-0.068 m`，`p=2.90e-10`；
- scene `+0.043 m`，显著变差。

V11.4 相对 V16 raw scale：

- camera `0.518 -> 0.463 m`，`p=0.00107`；
- P95 `1.314 -> 1.088 m`；
- success `55.0% -> 60.6%`；
- root 不变；
- joints `0.223 -> 0.225 m`，不显著；
- scene `0.526 -> 0.536 m`，`p=0.0380`，轻微但显著变差。

### 2.7.2 Scale cue 公平消融

所有方法共享 V16 rotation 和相同 translation equation，只改变 scale cue：

| Scale branch | Camera T | Camera paired p vs V16 | Scene |
|---|---:|---:|---:|
| V16 raw scale | 0.518 m | - | 0.526 m |
| DA3 background only | 0.480 m | 0.0684 | 0.545 m |
| DA3 + Keypoint root | 0.492 m | 0.1169 | 0.542 m |
| Keypoint projection only | 0.504 m | 0.1285 | 0.539 m |
| V11.4 fused scale | **0.463 m** | **0.00107** | 0.536 m |

结论：三个单独 cue 都没有达到显著 camera gain。有用的是 DA3 metric depth、human keypoint
定位和有界 background gate 组成的联合尺度规则。

### 2.7.3 Capture-disjoint 60-cut holdout，VGGT off

| Metric | Fixed | V12 Full |
|---|---:|---:|
| Camera translation | 0.663 m | **0.508 m** |
| Camera rotation | 23.05 deg | **17.62 deg** |
| Human root | 0.234 m | **0.195 m** |
| Human joints | 0.291 m | **0.245 m** |
| Human vertices | 0.285 m | **0.240 m** |
| Scene discontinuity | **0.475 m** | 0.547 m |
| Camera success | 41.7% | **60.0%** |

Holdout 复现了 `camera/human improvement + scene trade-off`，而不是三者全面改善。

### 2.7.4 Conditional VGGT 可选结果

显式开启 VGGT 后：

| Protocol | No VGGT | Conditional VGGT |
|---|---:|---:|
| 180-cut rotation | 16.04 deg | 12.09 deg |
| 180-cut camera T | 0.463 m | 0.403 m |
| Untouched 60 rotation | 17.62 deg | 14.08 deg |
| Untouched 60 camera T | 0.508 m | 0.450 m |

这些是可选最高精度结果，不是 V12 默认结果。

### 2.7.5 真实 recurrent multi-cut

每次 cut 都使用上一次 predicted world 作为下一次 anchor，不恢复 GT gauge：

| Cuts | Camera drift | Rotation drift | Human-root drift |
|---:|---:|---:|---:|
| 1 | 0.229 m | 7.81 deg | 0.093 m |
| 2 | 0.326 m | 23.97 deg | 0.094 m |
| 4 | 0.698 m | 37.99 deg | 0.134 m |
| 8 | 0.946 m | 59.03 deg | 0.193 m |

因此 V12 的有效范围是 1-2 个稀疏 cuts 的 short horizon；4-8 cuts 是累计漂移压力测试，
不是长期稳定证据。

### 2.7.6 Runtime

NVIDIA L20 审计：

| 项目 | Mean | Median | P90/P95 |
|---|---:|---:|---:|
| No-VGGT cut cue | 0.609 s | 0.407 s | 1.084/1.106 s |
| Triggered VGGT cut cue | 3.615 s | 2.319 s | 6.878/7.171 s |
| Normal Human3R | 3.570 FPS | 3.571 FPS | 3.579/3.580 FPS |

No-VGGT cut 第一帧总时间约 `0.609 + 0.280 = 0.889 s`。它满足因果流式定义，但不满足
25/30 FPS 实时视频要求。

## 2.8 V12 已证明、未证明与主要限制

已证明：

- pre-decode reset 可以消除旧 state 对新 shot trajectory 的污染；
- V16 是最强、最明确的独立 alignment gain；
- V11.4 shared scale 对 camera 有较小但显著的额外收益；
- camera、pointmap 和完整 SMPL-X 可以使用一个 projection-preserving shared similarity；
- 所有关键横向结果可在统一 evaluator 和 capture-disjoint holdout 上复现。

未证明或限制：

- scene consistency 明确退化；
- 不能宣称完整 camera-human-scene closure；
- 当前 cut trigger 不是自动 detector；
- Fixed coarse initializer 尚无 clean replacement ablation；
- Keypoint R-CNN 的具体模型必要性未证明；
- 4-8 cuts 后累计漂移明显；
- 没有 loop closure、BA、map reuse 或长期 gauge correction；
- Full 相比 Lite 的约 5.5 cm camera gain是否值得 DA3+Keypoint 复杂度，仍是重要方法选择。

---

# 3. Movie3R-Multi V13.0

## 3.1 版本定位

V13 是多人 shared-Boundary 研究版。它研究的问题不是“多输出几个人”，而是：

> 在人物身份关联正确时，多个人能否为同一个 camera-cut Boundary 提供冗余几何约束，
> 从而比任意一个可部署单人 anchor 更稳定？

核心原则：

```text
Identity answers WHO.
Geometry answers WHERE.
All humans share ONE Boundary.
```

当前状态：

```text
GT-ID multi-human shared-Boundary geometry: VALIDATED
Deployable automatic cross-shot identity: NOT VALIDATED
Default token_reid: false
```

## 3.2 原理想法

单个人的 Human3R root、torso 和 background refinement 可能有噪声或视角歧义。如果 cut 前后
有多个相同身份的人，每个人都能独立给出一个 Boundary candidate：

```text
human i -> (R_i, t_i)
```

若 identity 正确，这些候选应围绕同一个真实 shot Boundary 分布。对多个 rotation 在 SO(3)
上取均值、对 translation 取均值，可以抵消单个人的 torso、root 和 pointmap 噪声，特别是
减少 rotation ambiguity。

V13 不允许每个人拥有独立 world transform，因为 camera 和 scene 只有一个世界。多人只提供
冗余约束，最终必须合成：

```text
ONE B = [R, t]
```

## 3.3 输入与实验范围

### 3.3.1 Phase 1/2 几何输入

每个 cut 输入：

```text
5 pre-cut full-frame RGB images
+
1 first post-cut full-frame RGB image
```

MultiHuman `three` 中原始 2048x2048 全图缩放到 512x512，不做 per-person crop。Human3R
直接对全画面进行多人 detection 和 SMPL-X reconstruction。

实验规模：

- `three`：3 人、6 cameras、7 timestamps、9 camera pairs、offset 0/1/2/4/8，共 315 cuts；
- `dance`：2 人、36 cuts，独立序列 pilot/frozen evaluation；
- `box`：2 人、36 cuts，自动身份 frozen evaluation；
- EgoHumans `001_legoassemble`（本地历史路径位于 `data/EgoBody/`）：鱼眼、多 cut、人数变化 stress test；
- EgoBody val（本地 `data/EgoHuman/`）：双人 3/5 路 Kinect GT 已审计，但 RGB 尚不可用，未进入当前实验。

### 3.3.2 当前启用与关闭

启用：

- frozen Human3R；
- pre-decode hard reset；
- strict GT-ID association，仅当前几何 Oracle；
- Fixed Explicit；
- per-human V16 torso rotation，20 deg bound；
- explicit per-human translation candidate；
- uniform multi-human consensus；
- one shared Boundary。

关闭：

- DA3；
- Keypoint R-CNN；
- V11.4 shared scale；
- VGGT；
- V14.2 continuity；
- scene refinement；
- learned fusion；
- token Re-ID；
- learned identity adapter。

V13 固定 `s=1`，目的是隔离“多人冗余”本身，不把 V12 Full 的外部 scale cue 混入结论。

## 3.4 完整模型架构设计

### 3.4.1 总体数据流

```text
5 pre-cut full RGB frames
             |
             v
 Frozen Human3R recurrent inference
             |
 camera + pointmap + all detected SMPL-X humans
             |
 preserve per-identity root/torso history in old predicted world
             |
        camera cut trigger
             |
 reset scene/camera state before post frame decode
             |
             v
 first post-cut full RGB -> fresh Human3R multi-human output
             |
             v
 strict GT mesh-projection identity association
        answers WHO only
             |
    +--------+--------+--------+
    |                 |        |
 human 1           human 2   human N
    |                 |        |
 root-motion       root-motion ...
 torso target      torso target
 Fixed coarse      Fixed coarse
 V16 bounded R     V16 bounded R
 explicit t        explicit t
    |                 |        |
  (R_1,t_1)       (R_2,t_2) ...(R_N,t_N)
    +-----------------+--------+
                      |
             Uniform consensus
                      |
     R = SO(3) mean; t = arithmetic mean
                      |
             ONE shared B = [R,t]
                      |
       +--------------+----------------+
       |              |                |
    camera         pointmap       every SMPL-X human
       |              |                |
       +--------------+----------------+
                      |
                 world output
```

### 3.4.2 Strict GT-ID association

Human3R detection index `D0/D1/D2` 和 native `smpl_id` 都不是可靠的跨 camera-cut identity。
V13 当前几何 Oracle 使用 GT SMPL-X mesh projection 做严格对应：

- 将 GT mesh 投影到相机图像；
- 比较预测 mesh 与对应 GT mesh 的投影/对应 vertex geometry；
- 建立 pre/post identity assignment；
- identity 只回答“这个 detection 是谁”。

GT identity/mesh projection 不参与：

- root-motion anchor 数值；
- torso rotation candidate；
- Boundary `R/t`；
- fusion weight；
- camera/human world placement。

GT camera 和 GT SMPL-X 除 association 外只用于 evaluator。由于 WHO 使用 GT，这条路径不是
可部署推理系统。

### 3.4.3 每个人的历史状态

对每个 matched identity 保存：

- cut 前最多 5 帧 predicted world root；
- root orientation；
- torso frame；
- detection score；
- mesh completeness；
- 稀疏 background point cloud。

cut 时 scene/camera recurrent state reset，但这个外部 per-identity history 仍用于几何预测。

### 3.4.4 Root-motion anchor

从 cut 前 roots 计算稳健速度：

```text
v_i = robust_velocity(root_i history)
a_i = root_i(last) + delta_frame * v_i
```

`a_i` 表示人物 `i` 在旧 predicted world 中，post-cut 当前时刻应到达的位置。offset 0 主要
排除真实人体运动；offset 1/2/4/8 检查短时 motion extrapolation。

### 3.4.5 Torso-motion target

从相邻 torso frames 的 SO(3) relative rotations 估计 angular velocity，并外推：

```text
T_i_target = Exp(delta_frame * median_angular_velocity) * T_i_last
```

它提供人物 `i` 在当前 physical time 的目标 torso orientation。

### 3.4.6 Per-human Fixed + V16 candidate

对每个人：

1. 用最近历史 root orientations 的 SO(3) mean 与 post root orientation 得到 initial rotation；
2. 用 `a_i` 和 post root 得到 initial translation；
3. 从相同背景 point cloud 出发做固定预算 Fixed refinement；
4. 使用当前 post torso 与 `T_i_target` 求 V16 heading residual；
5. residual 限制在 `+-20 deg`；
6. 得到最终 `R_i`。

不同人物给出的人体初值不同，因此同一 background refinement 可能落到不同局部 branch。
多人 fusion 正是要减少这种单人不稳定。

### 3.4.7 Per-human explicit translation

rotation 固定后：

```text
t_i = a_i - R_i * r_i_post
```

使人物 `i` 的候选满足：

```text
R_i * r_i_post + t_i = a_i
```

每个人只产生一个候选，不拥有独立最终 Boundary。

### 3.4.8 Uniform multi-human consensus

冻结默认：

```text
R = SO3Mean(R_1, R_2, ..., R_N)
t = Mean(t_1, t_2, ..., t_N)
B = [R, t]
```

rotation 不是对欧拉角逐分量平均，而是在 SO(3) 上求 mean。translation 平均的是每个人在
自身 `R_i` 下显式求出的 raw `t_i`。

人数处理：

```text
N >= 2 valid matches -> multi-human consensus
N == 1               -> single-human Fixed + V16
N == 0               -> identity-free Fixed fallback / 当前几何不可用
```

V13 比较过 confidence weighting、geomedian、trimmed、Huber、layout selection、one-reject 和
soft uncertainty，但都没有在 held-out 上稳定超过 naive mean。

### 3.4.9 Boundary 的统一应用

固定 `s=1`：

```text
C_post_world = B * C_post_local
X_scene_world = R * X_scene_local + t
r_i_world = R * r_i_local + t
J_i_world = R * J_i_local + t
V_i_world = R * V_i_local + t
```

所有 post-cut humans、camera 和 pointmap 使用同一个 `B`。不允许 per-person Boundary、
独立 root correction、foot translation 或不同 human/scene scale。

## 3.5 自动身份研究模块

V13 的 deployability 瓶颈不是多人几何，而是跨 shot WHO。

### 3.5.1 Phase 3 native identity bridge

比较的 Human3R/人体表示包括：

- refined human token `H'`，768D；
- CUT3R head token，1024D；
- Multi-HMR head token，1024D；
- fused human prompt，768D；
- predicted beta，10D；
- root-centered local pose，468D。

匹配比较：

- raw L2、normalized L2、cosine；
- Hungarian、Sinkhorn；
- last、5-frame mean、5-frame medoid prototype；
- dustbin、TTL=8、new/unmatched identity；
- Match-Then-Align、Align-Then-Commit。

开发集冻结的最强规则是：

```text
last local pose + cosine + Hungarian
```

但 local pose 是 short-horizon motion compatibility，不是稳定 appearance identity。

### 3.5.2 Phase 4 precision-first appearance bridge

使用冻结 DINOv2 ViT-S/14：

```text
Human3R predicted mesh
-> project to predicted bbox
-> 8% padding
-> 224x224 letterbox crop
-> DINOv2 CLS + mean patch feature
-> normalized 768D appearance
```

冻结 gate 使用：

- appearance + beta + local pose；
- five-frame mean prototype；
- cosine distance；
- mutual nearest；
- absolute distance；
- row/column margin；
- five-frame vote；
- beta compatibility；
- pose compatibility；
- 至少 3 个有效 crop observations。

身份模块只能输出 association/dustbin，不能输出 SE(3)、scale、Boundary 或 fusion weight。

## 3.6 输出

GT-ID 几何版输出：

- post-cut shared Boundary `R/t`；
- global camera pose；
- global pointmap；
- 所有人的 global root、orientation、joints、vertices；
- 每个人的 `R_i/t_i` candidate 和质量诊断；
- GT-ID assignment，仅实验记录；
- multi/single/fallback 状态。

自动身份实验还输出：

- external identity ID；
- match cost/confidence；
- accepted/dustbin；
- ID switches、IDF1、coverage；
- identity memory/TTL 状态；
- catastrophic Boundary failure diagnostics。

## 3.7 训练方式

V13 几何主路径没有新增训练：

- Human3R frozen；
- Fixed/V16/translation/consensus 全部显式；
- fusion 不训练；
- Phase 3 native token matcher 不训练；
- Phase 4 DINOv2 frozen；
- Phase 4B shot-invariant adapter 因 Phase 4A gate 失败而未启动。

因此 V13 的当前有效贡献是 GT-ID 条件下的几何可行性和严格的 identity-negative audit，
不是一个已经训练好的多人 Re-ID 网络。

## 3.8 实验结果

### 3.8.1 Strict GT-ID 几何主结果

旧 Phase 1 曾使用 bbox IoU/center 做不严格 ID，人物重叠时发生 person swap，导致虚假的
150-180 deg 多人失败。V2 改用严格 mesh-projection ID 后重评全部 315 cuts。

在 308-case common support 上：

| Metric | Highest-confidence single | Naive multi mean |
|---|---:|---:|
| Camera translation | 0.565 m | **0.517 m** |
| Camera rotation | 9.96 deg | **7.01 deg** |
| Camera composite | 0.764 | **0.657** |
| Human joints | 0.402 m | **0.380 m** |
| Human vertices | 0.392 m | **0.372 m** |

paired composite improvement rate 为 74.0%，`p=1.20e-16`。没有 catastrophic failure。

但 GT evaluator 选出的 Oracle Best Single composite 为 `0.633`，仍优于 multi mean 的
`0.657`。因此原始严格 gate `multi > Oracle Best Single` 仍是 FAIL；这不否定“多人优于
可部署单人选择器”的结论。

#### V13 数值口径提醒

上表采用当前正式入口 `versions/v13/README.md` 的 release-canonical 数值。更早或更细的
V20 V2/Phase 2 缓存报告对 highest-confidence single 给出了另一组数值：

```text
V20 V2 paired 308 support: camera 0.613 m, composite 0.811
Phase 2 full 315 cuts:     camera 0.616 m, composite 0.814
```

它们与正式 README 的 `0.565 m / 0.764` 不能直接混在同一张绝对数值表中。现有文档没有
单列一次对这些 cache/evaluator 修订的完整 reconciliation，但几份报告对以下结论一致：

- multi mean 为 `0.517 m / 7.01 deg / 0.657`；
- multi 显著优于 highest-confidence single；
- paired improvement rate 74.0%，`p=1.20e-16`；
- multi 没有超过 GT Oracle Best Single。

后续论文如果使用 V13 single-baseline 的绝对数值，应先用冻结 runner 和同一 JSON 重新生成
一张唯一表；在此之前，以正式 README 作为版本摘要口径，不把 `0.764` 和 `0.811/0.814`
互换。

### 3.8.2 人数消融

在 212 个三人均有效的 samples 上，正式 README 的固定人数比较为：

| 人数 | Camera T | Rotation | Composite |
|---:|---:|---:|---:|
| 1 | 0.594 m | 10.80 deg | 0.810 |
| 2 | 0.560 m | 8.81 deg | 0.737 |
| 3 | **0.549 m** | **7.49 deg** | **0.699** |

人数增加带来单调改善，支持收益来自独立几何冗余，而不是某个固定人物恰好较好。

里程碑文档还报告过另一种 all-subset/aggregation 表，composite 为
`0.843/0.681/0.611`。它和上表不是同一统计聚合，后续引用时也应保留来源，不能选择性
混用。

### 3.8.3 多人收益来自什么

Phase 2 fusion-only 报告在它自己的缓存/统计口径下给出：

| Fusion | Composite | 解释 |
|---|---:|---|
| Highest-confidence single | 0.814 | Phase 2 全集基线 |
| Translation-only | 0.814 | 不改善 |
| Rotation-only | 0.728 | 明显改善 |
| Rotation + translation naive mean | **0.657** | 最佳 |

多人最明确的独立贡献是减少 torso rotation ambiguity。translation 只有在多人 rotation
也共同融合时才产生进一步收益。

### 3.8.4 Fusion optimization 与 `dance` pilot

在 `three` held-out 上，development 选择的 soft uncertainty rule：

```text
naive mean composite = 0.647
soft rule composite  = 0.650
paired p             = 0.589
```

soft rule 的 P90 略好，但 mean 和 rotation 退化，没有通过预定义门槛。

`dance` 36-cut pilot：

- 25 cuts 有两人，可以 fusion；
- 11 cuts 只有一人，自动退化；
- multi support 上 highest-confidence single composite `0.809`；
- two-human naive mean `0.745`；
- soft rule `0.762`；
- 无 catastrophic failure。

因此默认 fusion 保留全部有效人物的 naive mean。

### 3.8.5 Phase 3 native/pose automatic identity

跨 shot feature probe 中，`three` 最佳 IDF1：

| Feature | Best IDF1 | 解释 |
|---|---:|---|
| local pose | 0.934 | 短时动作兼容，不是长期 identity |
| beta | 0.877 | shape cue |
| Multi-HMR token | 0.657 | 最强 native 单 token，但不足 |
| refined `H'` | 0.488 | 原生 tracking feature 跨 cut 失效 |
| CUT3R head token | 0.401 | 失效 |

端到端 camera composite：

| Sequence | Single | GT-ID multi | Automatic-ID multi | ID switches | Catastrophic |
|---|---:|---:|---:|---:|---:|
| three | 0.814 | **0.664** | 0.850 | 52 | 3.17% |
| dance | 0.802 | **0.758** | 0.885 | 2 | 5.56% |
| box | 0.720 | 0.614 | **0.612** | 0 | 0% |

`box` 证明 WHO 完全正确时 automatic path 可以兑现 GT-ID 收益；`three/dance` 证明极少量
ID swap 会被 one shared Boundary 放大成 catastrophic geometry failure。

`three` 尾部：

```text
281/315 无错误 accepted ID:
automatic composite = 0.655, catastrophic = 0%

34/315 至少一个错误 accepted ID:
automatic composite = 2.463, catastrophic = 29.4%
```

因此 Phase 3 未通过部署 gate，默认 `token_reid=false`。

### 3.8.6 Phase 4 precision-first appearance

冻结 gate 的 identity precision/coverage：

| Sequence | Accepted precision | Accepted coverage | Multi coverage |
|---|---:|---:|---:|
| three | 100% | 14.37% | 7.62% |
| dance | 100% | 13.11% | 2.78% |
| box | 100% | 26.87% | 5.56% |
| EgoHumans | N/A，0 accepted | 0% | 0% |

它实现了零错误 accepted match，但 coverage 太低。完整端到端结果：

| Sequence | Single | GT-ID | Precision-first | Catastrophic |
|---|---:|---:|---:|---:|
| three | 0.814 | 0.664 | 3.882 | 55.6% |
| dance | 0.802 | 0.758 | 3.359 | 52.8% |
| box | 0.720 | 0.614 | 2.930 | 41.7% |

失败原因不是 accepted identity 错，而是大多数 cut 没有 accepted identity，进入较弱的
identity-free Fixed fallback。该 fallback 在 wide-view cut 上产生严重误差。

只看实际启用 multi 且 identity 全正确的 cuts：

| Sequence | Cuts | Single | GT-ID | Precision-first | Catastrophic |
|---|---:|---:|---:|---:|---:|
| three | 24 | 0.654 | 0.577 | 0.613 | 0% |
| dance | 1 | 0.745 | 0.485 | 0.485 | 0% |
| box | 2 | 0.496 | 0.419 | 0.419 | 0% |

这再次证明多人几何没有被推翻，真正瓶颈是安全 identity 的覆盖率和 fallback。

### 3.8.7 EgoHumans `3 -> 1 -> 3` stress test

Phase 4 冻结规则：

```text
accepted / matchable           = 0 / 16
multi activation              = 0 / 6 cuts
inactive identities recovered = 0 / 2
fallback                      = Fixed 6 / 6
```

主要问题包括：

- DINOv2/general appearance 的跨数据域偏移；
- Human3R predicted bbox 在鱼眼、遮挡和贴边人物上不稳定；
- crop 包含大量背景或只覆盖部分人体；
- 冻结 threshold 在 EgoHumans 上拒绝全部 proposal。

## 3.9 V13 已证明、未证明与主要限制

已证明：

- 在 strict GT-ID 下，多人 shared-Boundary 明显优于可部署单人选择器；
- 两人有效，三人进一步改善；
- 多人最主要减少 rotation ambiguity；
- naive mean 比当前手工 robust/soft fusion 更稳定；
- WHO 正确时，automatic pipeline 可以兑现 GT-ID 几何收益。

未证明或限制：

- GT identity/GT mesh projection 不可部署；
- native Human3R token 不能安全跨 wide-view cut Re-ID；
- local pose 是 motion cue，不是长期 identity；
- 少量 wrong accept 会污染所有人的 shared Boundary；
- precision-first appearance 只能用极低 coverage 换取零 wrong accept；
- EgoHumans 上 coverage 为零，进入/离开/遮挡/重新出现未解决；
- V13 当前没有 V12 Full scale，也没有形成 single/multi 统一产品路径；
- MultiHuman 缺少 GT scene，因此当前不报告 scene accuracy。

## 3.10 下一步准入条件

当前文档冻结的下一步顺序是：

1. 改善不依赖 GT bbox 的可部署 person crop；
2. 测试真正冻结的 person-ReID encoder，并单独统计其 latency/memory；
3. 在 capture/subject/camera-pair-disjoint 数据上验证 same/different separation；
4. 只有在非零 coverage 下保持接近零 wrong accept，才考虑轻量 shot-invariant adapter；
5. identity feature 只能回答 WHO，不能预测 SE(3)、scale、Boundary 或 fusion weight；
6. 保持 Match-Then-Align 和 Align-Then-Commit；
7. 单人或无可靠匹配时需要一个真正可靠的 V12/Lite fallback，而不是当前较弱的
   identity-free Fixed fallback。

---

# 4. 三个版本的关系与分析边界

## 4.1 它们不是简单的 V9 -> V12 -> V13 单线升级

| 关系 | 正确理解 |
|---|---|
| V9 vs V12 | V9 是 learned decoder correction；V12 是 frozen explicit geometry。两者是不同路线，不应把结果直接混成同一模型消融。 |
| V12 vs V13 | V13 继承 Hard Reset、Fixed、V16 和 shared-Boundary 原则，但关闭 V12 Full 的 DA3/Keypoint/V11.4 scale，以隔离多人贡献。 |
| V13 是否替代 V12 | 否。V13 的几何依赖 GT-ID，automatic WHO 失败。当前单人输入和真实部署仍应优先 V12。 |

## 4.2 当前最强的统一研究结论

```text
1. Camera cut 同时包含 recurrent-state transition 和 world-gauge transition。
2. Scene/camera local state 应在 cut 前解码边界明确 reset。
3. Human physical history 可以跨 shot 保存，但必须与 local scene/camera state 隔离。
4. 新 shot 应只估计一个 fixed Boundary，而不是逐帧随意 correction。
5. Camera、scene 和完整 human geometry 必须共享同一个 Boundary。
6. 单人 torso history 能显著减少 rotation error。
7. 多个正确匹配的人能进一步减少 rotation ambiguity。
8. Identity association 必须先于多人 geometry；wrong WHO 会让 shared WHERE 灾难性失败。
9. Projection consistency 是必要条件，但不等于 world/scene consistency。
10. 当前方法是 short-horizon re-anchoring，不是 unlimited-horizon mapping。
```

## 4.3 当前最重要的开放问题

给后续 AI 做研究分析时，优先考虑：

1. V12 Lite 的简洁性是否比 V12 Full 的约 5.5 cm camera gain 更有论文价值？
2. V11.4 的 shared scalar 为什么改善 camera，却系统性伤害 scene？
3. 是否能找到无需 DA3/Keypoint、但保留 V16 主要收益的更简洁 scale/translation formulation？
4. Fixed Explicit 是否可以被更强但仍严格因果、固定预算的 coarse initializer 替代？
5. 自动 cut detection 如何加入而不把 detector error 与 alignment error 混在一起？
6. long-horizon 是否需要 loop closure/BA/global optimizer，而不是继续堆 cut-time cues？
7. V13 是否可以先使用成熟 person-ReID reference 建立 WHO 上界，再决定是否训练 adapter？
8. 低 identity coverage 时，fallback 应该如何安全退化到 V12/Lite？
9. 能否用 uncertainty 表达“拒绝 multi，但仍使用可靠单人”，避免 identity-free Fixed？
10. 在不破坏 one-shared-Boundary 原则下，如何处理人物进入、离开、遮挡和重新出现？

---

# 5. 冻结身份、入口与关键文档

## 5.1 V9

```text
release: Movie3R-Learned V9.0
commit:  6eb64cb2158fb443d53cd4f1713af1899fe5a026
tag:     movie3r-v9-trained
runner:  versions/v9/run.py
viewer:  versions/v9/viewer.py
manifest: versions/v9/manifest.json
```

关键文档：

- `versions/v9/docs/METHOD_OVERVIEW.md`
- `versions/v9/docs/MODEL_ARCHITECTURE_DETAILS.md`
- `versions/v9/docs/EXPERIMENT_RECAP_20260630.md`
- `versions/v9/docs/EXPERIMENT_60H_SWEEP_RESULTS_20260629.md`
- `versions/v9/docs/GUARDRAILS.md`

## 5.2 V12

```text
release: Movie3R-Single V12.0
commit:  af92478
tag:     movie3r-v12-single
runner:  versions/v12/run.py
viewer:  versions/v12/viewer.py
manifest: versions/v12/manifest.json
```

关键文档：

- `versions/v12/LATEST_MODEL.md`
- `versions/v12/docs/CURRENT_MODEL_FULL_ARCHITECTURE_AND_ABLATION.md`
- `versions/v12/docs/V14_7_SHOT_AWARE_UNIFORM_SIMILARITY_REANCHORING.md`
- `versions/v12/docs/V14_6_ALIGNMENT_COMPONENT_NECESSITY_AUDIT.md`
- `versions/v12/docs/V14_5_FINAL_GEOMETRY_STREAMING_AUDIT.md`

## 5.3 V13

```text
release: Movie3R-Multi V13.0
commit:  e45e2af
tag:     movie3r-v13-multi
runner:  versions/v13/gt_id_consensus.py
viewer:  versions/v13/viewer.py
manifest: versions/v13/manifest.json
```

关键文档：

- `versions/v13/MULTIHUMAN_GEOMETRY_VALIDATED.md`
- `versions/v13/docs/V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md`
- `versions/v13/docs/V13_PHASE2_MULTIHUMAN_FUSION_OPTIMIZATION.md`
- `versions/v13/docs/V13_PHASE3_CROSS_SHOT_IDENTITY_BRIDGE.md`
- `versions/v13/docs/V13_PHASE4_PRECISION_FIRST_IDENTITY.md`
- `versions/v13/docs/V13_PHASE5_CAUSAL_IDENTITY_STATE.md`
- `versions/v13/docs/V13_EGOHUMANS_EGOBODY_DATASET_AUDIT_20260727.md`

---

# 6. 给后续 AI 的最短上下文

如果上下文窗口有限，可以只使用下面这段：

```text
Movie3R 基于 frozen/finetuned Human3R，研究单目多分镜视频中的 camera-human-scene
跨 shot 世界坐标连续性。

V9 是已训练的 4-frame AABB learned correction：semantic/alignment/momentum relation
tokens 进入 Human3R decoder，refined prompt 预测 pose-token residual、human-token residual
和 shared learned gate，pose/human heads 使用 LoRA。它证明 latent correction 可行，但没有
显式解决 recurrent-state contamination 和 shot-level world gauge。

V12 是当前单人主版：known cut trigger -> pre-decode hard reset -> fresh Human3R local
reconstruction -> Fixed Explicit coarse anchor -> V16 bounded torso-motion rotation -> V11.4
DA3/Keypoint fused shared scale -> explicit translation -> one fixed similarity Boundary ->
camera/pointmap/complete SMPL-X in one gauge -> Align-Then-Commit。180-cut camera 从
0.712/24.20deg 改善到 0.463m/16.04deg，但 scene 从 0.483 变差到 0.536m；8-cut
达到 0.946m/59.03deg，因此只适用于 short shot/sparse cuts。

V13 是 GT-ID 多人研究版：5 pre-cut + 1 fresh post-cut full RGB，正确身份下每个人产生
(R_i,t_i)，R 用 SO(3) mean、t 用 arithmetic mean，所有人/camera/pointmap 使用 ONE
shared Boundary，s=1。308-case common support 上 highest-confidence single 到 multi mean：
camera 0.565->0.517m，rotation 9.96->7.01deg，composite 0.764->0.657；人数 1/2/3
单调改善。automatic WHO 未通过：native/local-pose bridge 会接受少量 catastrophic swaps；
DINOv2 precision-first 虽零 wrong accept，但 multi coverage 只有 2.8%-7.6%，EgoHumans
为 0。因此 GT-ID geometry validated，但 V13 不能部署，默认 token_reid=false。
```
