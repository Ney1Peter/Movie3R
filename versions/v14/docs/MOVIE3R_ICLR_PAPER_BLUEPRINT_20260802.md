# Movie3R：面向 ICLR 的论文故事、方法蓝图与收敛计划

日期：2026-08-02

文档性质：论文总纲 / 研究决策文档 / 后续实验执行依据

目标会议：ICLR（具体年份与 deadline 以官方通知为准）

本文只整理思路、证据和计划，不把尚未完成的实验写成结果，也不启动新的训练、评测或可视化。

---

## 0. Executive judgment：项目现在处于什么位置

### 0.1 一句话判断

Movie3R 已经形成了一个有 ICLR 潜力、且与现有工作有明确区别的核心问题和方法思想：

> **Camera cut 不只是一个相机位姿跳变，而是一次必须区分“状态所有权”和“世界坐标所有权”的事务。Movie3R 在 cut 后提交干净的 reset state，只把旧状态当作只读的隐式 proposal，再通过显式几何验证，把可信的低维 gauge / human correction 提交到新 shot。**

英文 slogan：

> **Reset the state, recover the gauge, verify the geometry, and commit only what is safe.**

更短的机制概括：

```text
Implicit proposal -> Explicit verification -> Transactional commit
```

### 0.2 当前不是“没有答案”，而是“答案的骨架已经明确，端到端证据尚未闭环”

目前已经明确的不是某个单独阈值或 solver，而是三个层次必须被分开：

```text
Level 1: recurrent state transition
         cut 后旧 state 不能继续写入，新 shot 必须 hard reset

Level 2: shot/world gauge transition
         reset 后的局部重建需要一个显式 B0 接回旧 world

Level 3: person-structure correction
         camera gauge 正确不代表 Human3R 的 camera-relative root、orientation、scale、pose 正确
```

这三个层次分别对应目前最强的结果：

- non-committing shadow transaction：解决 state contamination；
- cross-source first-post-cut `B0`：提供 learned coarse camera/world proposal；
- BRTC-LC + person-local Kabsch：在相机冻结后显式修正人体 root/layout/orientation。

### 0.3 当前离 ICLR 完整投稿还差什么

最大的缺口不是继续想一个更复杂的新网络，而是完成以下证据闭环：

1. **统一 checkpoint 闭环**：当前 cross96 的 camera 结果与 BRTC/Kabsch 的人体结果来自不同 B0，不能直接拼成端到端表格。
2. **利用 shadow human gain**：`shadow_event` 的 human error 为 `0.4083 m`，但 camera-only `b0_runtime` 为 `1.2288 m`；当前部署路径丢掉了最有价值的人体修正信息。
3. **camera proposal 安全性**：cross96 frozen180 仍有 `86/180` catastrophic case，尤其集中在 MVHuman wide-view。
4. **外部对标**：当前 EgoHumans 只是同数据源、自建三条短链、provisional evaluator，不能宣称已经超过 Multi-THuMBS。
5. **真正 multi-cut、automatic cut、runtime/memory 和 scene 指标**：核心实现和论文系统性证据仍需补齐。

### 0.4 推荐的论文主线

不建议将论文写成：

```text
Human3R + correct token + Hungarian + triangulation + Kabsch
```

这会很容易被审稿人评价为工程模块拼接。

建议统一为：

> **Movie3R is a transactional reconstruction framework that decouples recurrent-state continuity from world-gauge continuity. It obtains expressive but unsafe corrections from a read-only shadow rollout, projects them into typed geometric residuals, explicitly verifies those residuals, and commits only the verified low-dimensional updates to a clean streaming trajectory.**

其中 B0、BRTC 和 Kabsch 都是“typed verified commit”的具体实例，而不是互不相关的技巧。

---

## 1. 推荐标题、核心命名与论文定位

### 1.1 首选标题

> **Movie3R: Transactional State–Gauge Decoupling for Streaming Human–Scene Reconstruction Across Camera Cuts**

### 1.2 更强调机制的备选标题

> **Movie3R: Propose, Verify, and Commit for Streaming Human–Scene Reconstruction Across Camera Cuts**

> **Movie3R: Verified Shadow Transactions for Causal 4D Reconstruction Beyond Video Shots**

### 1.3 论文中的核心术语

| 术语 | 推荐定义 |
|---|---|
| `Clean Reset Trajectory` | cut 后由原版 Human3R hard reset 得到、唯一允许继续写 recurrent state 的 shot-local trajectory |
| `Shadow Transaction` | 只读使用 pre-cut state、只在 first-post-cut frame 运行、永不提交其 recurrent state 的 correction rollout |
| `Coarse Gauge Proposal B0` | 由 shadow/raw camera 差值得到的显式 shot-level local-to-world proposal |
| `Typed Residual` | 被拆成 camera gauge、person root、person orientation、scale/shape 等语义明确的低维修正 |
| `Explicit Verification` | 用 ray observability、layout consistency、torso correspondence、trust region 和 fallback 判断修正是否可提交 |
| `Transactional Commit` | 只提交 clean raw state、固定 Boundary，以及通过验证的人体低维 correction；shadow state/geometry 不直接提交 |
| `Verified Shadow Projection` | 下一阶段重点：将 shadow human 相对 `B0(raw human)` 的增益投影为可验证的 root/orientation 等低维 residual |

### 1.4 任务定位

输入是一条可能包含 abrupt camera cuts 的单目 RGB stream。系统在线输出：

- camera trajectory；
- dense scene pointmaps；
- 多人 SMPL-X meshes / joints / roots；
- 所有可提交输出位于一个持续的 world coordinate system 中。

约束是：

- causal / online；
- 不读取 post-cut future frames；
- 不回写历史输出；
- fixed-size recurrent/external state；
- first-post-cut-only correction；
- 不依赖 DA3、VGGT、SLAM、ReID 等新增预训练模型作为主方法；
- 没有可靠证据时必须允许精确 fallback 或 abstention。

---

## 2. 证据纪律：全文必须区分的三种状态

本文以及后续论文草稿中的每个结论都应标记为：

| 状态 | 含义 |
|---|---|
| **[Established]** | 已有实现、可复现结果和适用范围清晰，可以作为当前事实 |
| **[Partial]** | 有正向证据，但数据、checkpoint、协议或安全性尚不完整 |
| **[Planned]** | 合理且可执行的目标方法/实验，当前不能写成论文结果 |

最重要的证据隔离规则：

> **cross96 B0 与当前 BRTC/Kabsch 不属于同一条已验证 end-to-end pipeline。投稿前必须重新执行 `cross96 -> BRTC-LC -> Kabsch -> full metrics`，否则不能把两组数字放在同一行或同一摘要 claim 中。**

另一个 provenance 注意点：

- 当前 cross96 明确从 `v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth` 初始化；其 manifest/config 名称明确是 AvatarReX + THuman mixed 60h。
- V9 研究阶段还进行了包含 MVHuman100/200 的四来源充分探索，因此“隐式路线能力上限是粗对齐”是跨多来源实验后的研究判断。
- 论文中必须区分“具体 checkpoint 的训练来源”和“整个 V9 路线曾覆盖的数据”，不能把两者写成一句含糊的 provenance。

---

## 3. 整个项目的发展脉络：每个版本解决了什么

### 3.1 Human3R：项目的统一流式基座

Human3R 基于 CUT3R recurrent state，将每张输入图像依次处理为：

```text
RGB frame
-> DINOv2 / image encoder tokens
-> pose token + image/scene tokens + human tokens
-> recurrent decoder 与 persistent state 交互
-> camera head: camera pose/intrinsics
-> scene head: dense world/self pointmaps + confidence
-> human head: multi-person SMPL-X parameters, roots, joints, vertices
-> update recurrent state
```

Human3R 的优势是统一、在线、固定内存、无需独立 SLAM/depth/HMR pipeline；但它默认输入属于一个连续 shot。camera cut 会把不相关的新视角送入旧 recurrent state，产生两种不同失败：

```text
State contamination:       旧 state 与新 shot 不兼容
World-gauge discontinuity: hard reset 虽干净，但新 shot 回到独立局部坐标系
```

### 3.2 V9：隐式 relation-token correction

V9 受 UniCon3R “关系 cue 作为内部 corrective prompt”启发，在 Human3R decoder 内加入 correction relation tokens：

```text
semantic token:
    current image/pose/human tokens + recurrent context

alignment token:
    current pose token + previous pose token + latent difference

momentum token:
    previous correction token/delta/gate
```

这些 tokens 与 image / pose / human tokens 一起进入 decoder attention。refined correction representation 分别驱动：

```text
pose latent residual + gate -> corrected pose token -> original camera head
human latent residual + gate -> corrected human token -> original human head
```

pose head 和 human head 使用 LoRA 适配 corrected latent。

V9 的贡献和结论：

- **[Established]** 隐式 token correction 能同时影响 camera 与 human latent；
- **[Established]** 在多来源充分训练/探索后，其能力上限更适合作为 coarse re-anchoring，而不是精确最终对齐；
- **[Established]** 连续逐帧使用 correction 会把 corrected token/state 继续写入 recurrent memory，存在污染和漂移；
- **[Established]** latent correction 本身不保证 camera、scene、human 使用同一个显式刚体变换。

因此 V9 不应被“重新训练一次然后直接一步到位”。它应被重新定位为 expressive proposal generator。

### 3.3 V12：hard reset、固定 shot Boundary 与 shared transform

V12 建立了重要的显式协议：

```text
camera cut
-> hard reset Human3R
-> 在 cut 处估计一次 shot-level Boundary
-> camera + pointmap + all humans 共用同一个 Boundary
-> 整个 post shot 固定传播，不逐帧重估
```

V12 证明：

- cut 后 hard reset 是正确的 state lifecycle；
- world alignment 应是显式、shot-level、固定的；
- camera/scene/humans 的共同 world gauge 必须共享；
- 单人 torso 可提供 rotation cue。

但单人 anchor 会把 Human3R root-depth bias 直接带入整个 shot，并且 multi-cut 会累计误差。

### 3.4 V13：多人 shared-Boundary、WHO/WHERE 分离与失败边界

V13 将多个 matched humans 转为多个 Boundary candidates，再使用 shared consensus：

```text
WHO: 哪些 pre/post detections 是同一个人
WHERE: 这些已匹配的人共同支持怎样的 shot Boundary
```

正确身份下，多人能够降低单人 rotation ambiguity；但自动 identity 不可靠时，错误匹配会产生 catastrophic shared-Boundary failure。更关键的是，从 person roots 重新求一个完整 Boundary 会覆盖已经较准的 B0，并吸收 Human3R 的 camera-relative root-depth bias。

因此 V13 给出的科学结论是：

- 多人信息有价值；
- WHO 和 WHERE 必须分开；
- 不能用不可靠 identity/human root 无条件重写全局 camera/scene Boundary；
- human refinement 应在 camera gauge 冻结后作为 person-structure correction，而不是再次估完整 shot transform。

### 3.5 V14：state–gauge decoupling

V14 将 V9、V12、V13 的有效部分重新组合为事务：

```text
first post-cut frame
├─ clean raw-reset branch
│    correction OFF
│    state can be committed
│
└─ read-only shadow branch
     pre-cut state + V9 correct tokens
     expressive corrected camera/human proposal
     state must never be committed

camera difference -> explicit B0
discard shadow state
propagate clean raw state + fixed B0
```

这一步首次把“使用旧状态恢复 world gauge”和“允许旧状态继续控制未来”分离开。

### 3.6 BRTC-LC：冻结相机后的 root/layout 精对齐

GT 可视化和误差分解揭示：camera 对齐后，人仍可能严重不齐，因为 Human3R 的人体相对相机深度本身有偏差。于是 BRTC-LC 不再修改相机，而是：

```text
B0-aligned anonymous association
-> last-pre / first-post torso-joint camera rays
-> two-view closest-ray triangulation
-> signed person-root residual
-> observable gate
-> group shift + layout-selected individual residual
-> accepted human rigid translation
-> rejected/unmatched exact B0
```

这一步将“camera gauge”与“person root gauge”明确分离。

### 3.7 Person-local Kabsch：root 后的 orientation 精对齐

BRTC 只平移人体，无法修正不同 shot 独立预测出的 global orientation 漂移。Kabsch 使用 BRTC accepted 人的 root-centred hips/shoulders 显式对应，求有界 per-person SO(3)：

```text
BRTC corrected root
-> root-centred torso4 correspondences
-> SO(3) Kabsch
-> half rotation, cap 25 degrees
-> rotate joints/vertices around native root
-> camera/root/rejected/unmatched unchanged
```

它验证了 person orientation 是独立于 camera 和 root 的有效 correction 类型。

### 3.8 cross96：跨来源 first-post-cut proposal 训练

最新实验保持 V9/V14 architecture 和 loss 不变，只扩大 cut 监督：

```text
formal V9 checkpoint
-> AvatarReX / THuman / MVHuman100 / MVHuman200
-> 96 cut events per source, 384 total
-> first-post-cut-only event training, 6 epochs
```

cross96 在 frozen180 上把 camera composite 从 raw reset 的 `5.5173` 降到 `1.7333`，相对旧单 Avatar checkpoint 降低 `24.2%`。这证明跨来源 cut supervision 确实学到 held-out gauge recovery，但 `86/180` catastrophic 表明 learned proposal 仍不能无条件 commit。

---

## 4. 论文要讲的完整科学故事

### 4.1 传统直觉为什么不够

对普通视频，recurrent reconstruction 假设相邻帧在同一个持续世界中。对剪辑视频，这个假设在 cut 处突然失效。两个直接方案都不成立：

```text
继续 recurrent state:
    保留 world memory，但新 shot 污染旧 state

直接 hard reset:
    state 干净，但新 shot 的 local gauge 与历史世界断开
```

即使相机重新对齐成功，Human3R 仍可能把人预测在错误的 camera-relative depth、orientation 或 scale。因此“找到一个 SE(3) 乘整个 shot”也不是完整答案。

### 4.2 核心观察

项目已有实验支持四个观察：

1. 旧 recurrent state 中确实含有跨视角 world prior，但它不适合成为新 shot 的长期 state。
2. shadow branch 的隐式表示比 camera-only B0 更 expressive：它将 human error 从 `1.1046 m` 降到 `0.4083 m`。
3. clean raw branch 更适合长期传播，但简单 B0 不会保留 shadow human gain，human error 反而为 `1.2288 m`。
4. camera、person root/layout、person orientation 是不同误差自由度，必须由不同证据验证。

### 4.3 论文核心 insight

> **A camera cut should be processed as a typed transaction, not as an ordinary recurrent update or a single rigid alignment. The model may use stale state to propose a correction, but it should commit only a clean state and explicitly verified, type-specific geometric residuals.**

### 4.4 主方法的统一表达

```text
                   Read-only old state
                           |
First post-cut RGB --------+--------------------+
        |                                       |
        v                                       v
Clean raw-reset rollout                    Shadow rollout
state owner: new shot                      state owner: none
        |                                       |
        |                         camera + human proposals
        |                                       |
        +---------------+-----------------------+
                        v
             Typed residual decomposition
          shot gauge / root / orientation / ...
                        |
                        v
             Explicit geometric verification
          confidence / rays / layout / SO(3) / cap
                        |
                        v
                 Transactional commit
        clean raw state + fixed B0 + verified humans
                        |
                        v
                Causal shot propagation
```

---

## 5. Abstract 草稿

### 5.1 English abstract：当前证据版本

> Streaming human–scene reconstruction models maintain a compact recurrent state to jointly recover cameras, dense geometry, and people from monocular video. Their state continuity assumption, however, breaks at camera cuts: carrying the state across a cut contaminates the new shot, while resetting it produces a clean reconstruction in an unrelated world coordinate system. We introduce **Movie3R**, a transactional framework that decouples recurrent-state continuity from world-gauge continuity. At the first frame after a cut, Movie3R runs a clean reset trajectory together with a read-only shadow correction that accesses the previous state. The shadow trajectory is never committed; instead, it proposes an explicit coarse gauge bridge between the reset shot and the existing world. Movie3R then decomposes the remaining mismatch into typed human residuals and commits only corrections that pass explicit geometric verification, including camera-ray observability, multi-person layout consensus, and bounded person-local orientation alignment. This proposal–verify–commit design preserves a clean recurrent state, requires no future frames or history rewriting, and adds only fixed per-cut computation and memory. Cross-source first-cut training on AvatarReX, THuman, and MVHuman reduces held-out camera composite error by 68.6% over hard reset and 24.2% over a single-source checkpoint, while retaining bit-exact no-cut behavior. On a separately frozen multi-human evaluation, explicit root/layout verification reduces root and world-joint errors by 38.8% and 33.3% without changing cameras, and bounded orientation refinement further improves joint and mesh accuracy. These results expose a central limitation of camera-only alignment and establish transactional state–gauge–structure factorization as a promising route to causal 4D reconstruction across edited shots.

这版 abstract 中最后两组结果来自不同 B0，只能作为“当前项目证据摘要”，不能原样成为投稿 abstract。完成统一 end-to-end 评测后，应替换为同 checkpoint 数字。

### 5.2 English abstract：投稿占位模板

> ... On held-out multi-shot benchmarks, the complete Movie3R pipeline reduces camera error by **[X%]**, world-frame human error by **[Y%]**, and cross-shot identity switches by **[Z%]** over the strongest causal baseline, while preserving **[exact/no-regression]** no-cut behavior and using **[runtime/memory]**. It further **[matches/outperforms]** Multi-THuMBS under **[official/protocol-matched]** evaluation without future-frame optimization or additional pretrained geometry models.

只有完成对应实验后才能填这些位置。

### 5.3 中文摘要

现有流式人—场景重建模型依赖一个持续更新的 recurrent state，从单目视频中联合恢复相机、稠密场景和多人三维人体。但 camera cut 会破坏其状态连续性假设：继续使用旧状态会污染新镜头，而 hard reset 虽能得到干净重建，却会产生与历史世界无关的新坐标系。Movie3R 将切镜建模为一次事务，显式分离 recurrent-state continuity、world-gauge continuity 与 person-structure consistency。在 cut 后第一帧，系统同时运行可提交的 clean reset branch 和只读旧状态的 shadow correction branch。shadow state 永不提交，而是产生粗 world-gauge 及人体修正 proposal；随后系统把修正分解为 shot gauge、人体 root/layout 和 orientation 等有明确语义的低维 residual，并通过射线可观察性、多人布局一致性和有界 SO(3) 对齐进行显式验证，只将可信结果提交到 clean stream。该方法不使用未来帧、不回写历史、保持固定外部状态，并使无 cut 输入与原 Human3R 数值一致。当前实验已经验证跨来源 first-post-cut 训练显著改善 held-out camera gauge，显式 BRTC-LC 能在冻结相机下大幅降低人体 root/layout 误差，person-local Kabsch 可进一步改善人体朝向；完整统一 checkpoint 的端到端结果仍是投稿前的首要闭环任务。

---

## 6. Introduction 应如何展开

### 6.1 第一段：任务价值

在线视频、电影、直播切换和短视频都由多个 shot 组成。若能从这类视频中持续恢复统一世界中的 camera、scene 和 multiple humans，将直接服务于运动理解、电影编辑、AR/VR、数字人和具身学习。Human3R/CUT3R 已经表明统一 recurrent model 可以实时、固定内存地处理连续视频，但真实编辑视频的 camera cuts 仍是结构性失败点。

### 6.2 第二段：两难问题

需要用一个反例图说明：

```text
carry state -> 新 shot 被旧场景拖坏
hard reset  -> 每个 shot 自己看起来合理，但放不到同一 world
camera align -> camera 变准，人仍因 root-depth/orientation bias 对不齐
```

这里明确提出：

> `camera cut = state transition + gauge transition + structured human residual`

### 6.3 第三段：为什么现有方法不能直接解决

- CUT3R/Human3R 假设正常连续 stream，不规定 cut 后 state 所有权；
- TTT3R、TTSA3R、ReCal3R 改善长序列 state update/forgetting，但不回答 abrupt cut 后“旧 state 不能提交、却仍含 world cue”的矛盾；
- HumanMM 和 Multi-THuMBS 解决 multi-shot human motion，但依赖独立 SLAM/VGGT、mesh/scene registration、Re-ID、多阶段优化、传播或全局 smoothing，不是统一 recurrent human–scene model 的 first-frame causal transaction；
- 一个 camera-only shared transform 无法修复 base model 的 camera-relative human structure bias。

### 6.4 第四段：方法概述

Movie3R 在 cut 后第一帧复制一次只读状态用于 shadow proposal，同时从空状态产生 clean raw trajectory。它不提交 shadow state，而是把 shadow/raw 差异转为显式 B0；再把人体 mismatch 拆为 root/layout/orientation typed residual，并仅在显式证据支持时 commit。

### 6.5 第五段：贡献

推荐贡献写为四点：

1. **Problem formulation.** 我们首次从 recurrent-state ownership 与 world-gauge ownership 的角度形式化 camera cuts，并进一步揭示 camera gauge 与 person structure residual 的层级差异。
2. **Transactional architecture.** 提出 non-committing shadow transaction：旧 state 只生成 correction proposal，新 shot 只提交 clean reset state；由此兼得 state purity 与 world continuity。
3. **Verified typed commit.** 将隐式 latent proposal 投影成可审计的 shot gauge、person root/layout 与 orientation residual，并通过 BRTC-LC、layout consensus、有界 Kabsch 及 fallback 进行显式验证。
4. **Causal evaluation.** 建立 first-post-cut、no-future、fixed-memory、multi-cut 的评测协议，并在 cross-source camera cuts 与 real multi-human sequences 上系统评估 camera、human、scene、identity、安全 tail 和 streaming cost。

第一点中的“首次”投稿前需再做一次系统文献检索，最终可改成更保守的 “to our knowledge, the first ...”。

---

## 7. Related Work 的组织方式与差异

### 7.1 Streaming 3D reconstruction

**CUT3R** 使用 persistent recurrent state 在线预测统一 world 中的 camera 和 pointmaps；**Human3R** 在此基础上统一输出 multiple SMPL-X humans、scene 和 camera。它们提供 Movie3R 的基座，但默认观测属于可持续更新的同一状态过程。

Movie3R 的区别不是提出另一个普通 recurrent backbone，而是定义 camera cut 处的 state ownership 和 commit protocol。

### 7.2 Long-context state maintenance

**TTT3R** 将 recurrent state 视为 fast weights，以 confidence-aware learning rate 平衡历史保留和新观测；**TTSA3R** 从 temporal state evolution 与 spatial observation quality 自适应更新；**ReCal3R** 根据 token reliability 校准更新率。这些工作主要解决长连续 stream 中逐渐遗忘、覆盖和 drift。

Movie3R 解决的是不同问题：在 abrupt cut 后，关键不是“给旧 state 多大更新率”，而是“旧 state 根本不应拥有新 shot 的写权限；它只能作为一次只读 proposal source”。这组 baseline 应包含 “TTT/ReCal-style update across cut” 与 hard reset 对照，证明平滑更新不能替代事务切换。

### 7.3 Unified human–scene reconstruction

**Human3R** 追求 all-at-once online human/scene/camera reconstruction；**UniCon3R** 用 contact prompt 和 latent feedback 把人—场景关系作为内部 correction cue；**GUSH3R** 将统一人景重建扩展到可渲染 Gaussians；**Trophies** 使用同步多视角、multi-branch reconstruction、Sim(3) alignment、bundle adjustment 和 contact optimization 获得全局一致结果。

UniCon3R 对 Movie3R 的最直接启发是：关系信息应进入 latent refinement，而不是仅作为 auxiliary output。Movie3R 的新问题是如何在 streaming cut 处安全使用这种 expressive latent correction：proposal 可以隐式，commit 必须显式且可验证。

Trophies 依赖同步多视角与全局优化，设定与 first-post-cut causal monocular streaming 不同。

### 7.4 Human motion across shots

**HumanMM** 面向 multi-shot 单人长运动，结合 shot detector、增强 SLAM/camera estimation、HMR、跨 shot alignment 和 motion integration。它没有同时维护统一 recurrent scene state，也不以固定内存 first-frame transaction 为核心。

**Multi-THuMBS** 面向 multi-person 3D human mesh tracking beyond shots。它使用 VGGT 在两个 boundary frames 建立 shared 3D space，Grounded-SAM/ViTPose/4DHumans 等模块提供 mask/keypoint/mesh，三阶段优化 root、orientation 和 camera，再用 appearance/geometry/pose Re-ID、DROID-SLAM propagation 和全序列 smoothing/cross-camera optimization。150-frame video 报告约 10 分钟优化。

Movie3R 与其目标最接近，但范式不同：

| 维度 | Multi-THuMBS | Movie3R 目标 |
|---|---|---|
| 基础模型 | 多模块 HMR + VGGT + SLAM + ReID | 一个 Human3R recurrent backbone + 小型 correction |
| 边界处理 | 两帧 shared space + 多阶段 optimization | first-post-cut shadow proposal + closed-form verification |
| 时间 | 全序列传播/后处理 | 严格 causal，不读 future |
| 状态 | 非 recurrent-state transaction | 显式 state ownership/commit |
| 额外模型 | VGGT、Grounded-SAM、ViTPose、4DHumans、DROID-SLAM 等 | 主线不新增外部预训练模型 |
| 输出 | human tracking / camera | unified camera + scene + humans |
| 内存 | 视频级 optimization | fixed-size streaming state |

不能把“速度更快”作为唯一创新；真正差异是 transaction semantics 和 typed verified commit。

### 7.5 Multi-view global reconstruction

Trophies、HSfM 等利用同步多视角、Sim(3) 和 global optimization 缓解尺度及人体—场景对齐。Movie3R 只在编辑边界获得 last-pre/first-post 两个时间相邻但视角突变的观测，并要求立即提交新 shot，因此需要 precision-first 的局部、因果方案。

---

## 8. Problem formulation

### 8.1 符号

设输入流为图像 `I_1, ..., I_T`，cut indicator 为 `e_t`。第 `k` 个 shot 的局部坐标系为 `L_k`，持续输出世界坐标系为 `W`。

Human3R recurrent forward 写为：

\[
(Y_t,S_t)=F_\theta(I_t,S_{t-1}),
\]

其中 `Y_t` 包括 camera `C_t`、scene pointmap `X_t` 和 people `H_t={H_{i,t}}`。

### 8.2 cut 处的 clean transition

若 `e_t=1`，clean raw branch 从初始化状态运行：

\[
(Y_t^r,S_t^r)=F_\theta(I_t,S_\emptyset).
\]

`S_t^r` 是唯一允许写入未来的 recurrent state。

### 8.3 non-committing shadow proposal

shadow correction 使用 pre-cut state：

\[
(\widetilde{Y}_t,\widetilde{S}_t)
=F_{\theta,\phi}^{corr}(I_t,S_{t-1};e_t=1),
\]

其中 `phi` 包括 correct-token builder、pose/human correction heads 和 LoRA。强制：

\[
\widetilde{S}_t \text{ is never committed.}
\]

### 8.4 coarse gauge proposal

在 camera-to-world 约定下，局部 bridge 为：

\[
\Delta B_t = \widetilde{C}_t(C_t^r)^{-1}.
\]

若上一 shot 的 local-to-world Boundary 为 `G_{k-1}`，则当前 coarse world Boundary 为：

\[
B_{0,k}=G_{k-1}\Delta B_t.
\]

单 cut 评测将上一 world 归一为单位阵时，常简写为 `B0 = C_shadow @ inverse(C_raw)`。

### 8.5 typed person correction

camera-only aligned raw human 为：

\[
H_{i,t}^{0}=B_{0,k}\odot H_{i,t}^{r}.
\]

最终 person correction 不再被错误地称为另一套 shot Boundary，而写成固定 world camera 下的人体预测 refinement：

\[
H_{i,t}^{*}=\Phi( H_{i,t}^{0};\delta r_i,R_i,\delta s_i,\delta q_i),
\]

分别表示 root translation、global orientation、scale 和 articulation/shape residual。当前已验证 `delta r_i` 的 BRTC-LC 和 `R_i` 的 bounded Kabsch；scale 与 articulation 尚未晋级。

### 8.6 优化目标

目标不是无约束最小化单个 mean，而是在 streaming contract 下优化：

\[
\min E_{cam}+\lambda_h E_{human}+\lambda_s E_{scene}+\lambda_l E_{layout}
\]

并满足：

```text
state purity
no future leakage
no history rewrite
bounded memory/compute
tail-risk constraint
exact fallback
```

---

## 9. 方法：输入、模块、处理和输出

### 9.1 总模块表

| 模块 | 输入 | 操作 | 输出 | 状态写入 | 当前状态 |
|---|---|---|---|---|---|
| M0 Cut Router | current RGB, causal history | 判断是否发生 cut | event flag | 只写有限 detector state | **[Planned for full system]** |
| M1 Clean Raw Reset | first-post RGB | correction OFF，从空 state 运行 Human3R | raw camera/scene/humans, clean state | **允许** | **[Established]** |
| M2 Shadow Proposal | first-post RGB, read-only pre state | V9 correct tokens + decoder + corrected heads | shadow camera/humans/scene diagnostics | **禁止** | **[Established]** |
| M3 Coarse Gauge `B0` | raw/shadow camera | `C_shadow @ inv(C_raw)` | fixed shot bridge | 写一个 4x4 Boundary | **[Established, not yet safe]** |
| M4 Shadow Residual Decomposition | shadow human, B0(raw human) | 拆 root/orientation/scale/articulation residual | typed proposals | 不直接写 | **[Planned, P0 research]** |
| M5 Anonymous Association | B0-aligned pre/post humans | root+torso+centred-joint Hungarian + dustbin | matched index pairs | 写有限 track relation | **[Partial]** |
| M6 BRTC-LC | matched people/cameras | ray triangulation + gate + layout consensus | person translation | accepted only | **[Established on older B0]** |
| M7 Kabsch | BRTC accepted torso joints | bounded root-centred SO(3) | person orientation | accepted only | **[Qualified on older B0]** |
| M8 Verified Shadow Projection | learned typed proposals + explicit proposals | agreement/uncertainty/trust-region selection | final typed residuals | verified only | **[Planned main novelty completion]** |
| M9 Transaction Commit | clean state, B0, accepted residuals | atomic ownership update | new shot state | clean/verified only | **[Partial integration]** |
| M10 Shot Propagation | later frames + cached Boundary/person state | normal Human3R + fixed transforms | world outputs | normal clean state | **[Established in probes]** |

### 9.2 M0：cut router

主实验首先使用 GT/oracle cut timestamp，以隔离 alignment。完整系统再加入 causal detector，例如 PySceneDetect 或基于现有 encoder token change 的轻量 detector。detector 只触发状态机，不估计 Boundary。

论文必须同时报告：

- oracle-trigger alignment；
- automatic detector precision/recall/F1；
- detector 错误对最终 camera/human 指标的影响。

### 9.3 M1：clean raw-reset branch

输入：first-post RGB。

处理：在 decoder 前 hard reset；所有 correct token、correction head residual 和 LoRA routing 关闭，运行原 Human3R。

输出：

```text
C_raw, X_raw, H_raw, S_raw
```

其中 `S_raw` 是新 shot 唯一的 recurrent state owner。

### 9.4 M2：read-only shadow branch

输入：同一 first-post RGB、pre-cut recurrent state 和固定长度 correction context。

处理：

```text
image/pose/human tokens
+ semantic/alignment/momentum correct tokens
+ old recurrent state
-> recurrent decoder attention
-> refined pose/human/correction tokens
-> corrected camera and human heads
```

输出：

```text
C_shadow, H_shadow, X_shadow(diagnostic), confidence signals
```

shadow state、pose memory、human memory、scene geometry均不得直接写入正式流。

### 9.5 M3：camera-derived coarse B0

`B0` 只承担 identity-free coarse WHERE：把新 shot 从任意 local gauge 移入旧 world 的正确 basin。它不承担最终 human root、pose、shape 或 scene surface 修正。

同一个 `B0` 作用于 raw camera、raw pointmap 和所有 raw humans，以保持 shot-level coordinate transform 一致。后续 frames 只复用一个缓存的 `B0`，不再运行 shadow。

### 9.6 M4：Verified Shadow Projection（投稿强版本的关键新增点）

这是由当前最强证据冲突直接推导出的下一主线，不是另起炉灶。

先比较：

```text
H_shadow
vs.
B0(H_raw)
```

将差异分解为：

```text
delta_root_shadow
delta_global_orientation_shadow
delta_scale/depth_shadow
delta_articulation_shadow
delta_shape_shadow
scene-relative residual
```

原则：

- 不直接复制 shadow mesh；
- 不提交 shadow state；
- 不把所有 latent gain 压成一个新的全局 SE(3)；
- 只考虑有独立观测验证、低维且可设置 trust region 的分量。

推荐第一版只做：

```text
shadow root translation proposal
shadow global orientation proposal
```

并与 BRTC/Kabsch 做 agreement verification：

```text
if shadow-root agrees with BRTC and BRTC observable:
    robustly fuse / choose lower-risk bounded residual
elif only BRTC observable:
    use frozen BRTC
else:
    exact B0

if shadow-orientation agrees with Kabsch and lowers predicted torso residual:
    bounded orientation commit
elif Kabsch alone passes frozen rule:
    use frozen Kabsch
else:
    keep BRTC orientation
```

这个机制把 learned shadow 和 explicit geometry 真正连成一条论文主线。

### 9.7 M5：B0-aligned anonymous association

关联必须在 B0 后进行，因为 direct local gauges 下 root/torso 不可比较。当前 cost：

```text
root distance
+ torso orientation disagreement
+ root-centred joint geometry distance
-> normalized cost matrix
-> Hungarian one-to-one assignment
```

必须保留当前负面结论：仅靠几何 dustbin 不能解决一般 identity replacement。`box` 中出现了几何上高置信但身份完全交换的反例。因此最终论文有三种选择：

1. 主 claim 限制为 same-visible identities / reliable association setting；
2. 使用 Human3R 已有 appearance/native tokens 作为正交 identity cue，不新增 ReID 模型；
3. precision-first abstention，对不确定 replacement 不提交 person refinement。

不能声称 automatic multi-human identity 已解决。

### 9.8 M6：BRTC-LC root/layout verification

对 matched person 的 pelvis、左右 hips、左右 shoulders 构造相机射线：

\[
d_{a,j}=\frac{J_{a,j}-C_a}{\|J_{a,j}-C_a\|},\qquad
d_{b,j}=\frac{J_{b,j}-C_b}{\|J_{b,j}-C_b\|}.
\]

求两条直线 `C_a+s_a d_a` 与 `C_b+s_b d_b` 的 closest points，以中点作为 joint triangulation。减去 post joint 相对 pelvis 的向量，得到 pelvis 候选；只保留其沿当前 pelvis ray 的 signed residual，多关节用 median 汇聚。

冻结 gate：

```text
joint set            = pelvis + hips + shoulders
minimum valid joints = 1
median ray gap       <= 0.20 m
joint residual MAD   <= 0.40 m
median parallax sine >= 0.025
residual cap         = +/-2.0 m
```

多人时将 individual shift 写成：

\[
s_i=g+\lambda(s_i^{ind}-g),
\]

其中 `g` 是同 cut accepted shifts 的坐标 median；`lambda` 从 `{0,.25,.5,.75,1}` 中按 pre predicted pairwise layout consistency 选择。未通过 gate 或 unmatched 的人保持 exact B0。

### 9.9 M7：person-local orientation Kabsch

对 BRTC accepted person 取 torso4：左右 hips 与 shoulders，分别减 native root，求：

\[
R_{raw}=\arg\min_{R\in SO(3)}\|X_{post}R^T-X_{pre}\|.
\]

通过 SVD 和 reflection correction 获得 SO(3)。冻结策略：

```text
applied angle = min(0.5 * raw angle, 25 degrees)
apply only if predicted torso correspondence decreases
```

rotation 围绕每帧自己的 translated native root 传播；camera、root 和 pair-root layout 不变。

### 9.10 M8-M10：commit 和 propagation

cut 时原子顺序应为：

```text
1. snapshot pre-cut read-only context
2. initialize clean raw state
3. run raw and shadow first-post rollouts
4. compute B0
5. decompose and verify typed residuals
6. discard every shadow-owned state/geometry object
7. commit clean raw state + fixed B0 + accepted person residual state
8. emit current world output
9. later frames run normal raw Human3R and reuse fixed committed transforms
```

下一次 cut 只使用上一 shot 已提交的 world Boundary 和 causal person state，不能重放未来或重写过去。

### 9.11 推理伪代码

```python
def process_frame(image, event, stream_state):
    if not event:
        raw, clean_state = human3r(image, stream_state.clean_state)
        world = apply_committed_geometry(raw, stream_state)
        stream_state.clean_state = clean_state
        return world, stream_state

    pre_context = readonly_snapshot(stream_state)

    raw, clean_state = human3r(image, init_state(), correction=False)
    shadow, _ = human3r_corrected(image, pre_context, correction=True)

    b0 = camera(shadow) @ inverse(camera(raw))
    typed_shadow = decompose_shadow(shadow.humans, apply_b0(raw.humans, b0))

    matches = associate_after_b0(pre_context.humans, raw.humans, b0)
    geometric = brtc_then_kabsch(pre_context, raw, b0, matches)
    verified = verify_and_project(typed_shadow, geometric)

    stream_state = atomic_commit(
        clean_state=clean_state,
        shot_boundary=b0,
        person_residuals=verified,
    )
    return apply_committed_geometry(raw, stream_state), stream_state
```

### 9.12 可以作为 appendix proposition 的性质

1. **No-cut invariance.** event router 不触发时，Movie3R 与原 Human3R 数值一致。
2. **State purity.** 未来 recurrent state 仅是 clean reset trajectory 的函数，与 shadow state 数值无关。
3. **First-frame causality.** cut 输出只依赖 `I_{<=t}`，截断 `I_{>t}` 不改变结果。
4. **Fixed-memory property.** 每个 shot 只保存固定 recurrent state、一个 Boundary 和有限 person state。
5. **Gauge equivariance.** 对输入 pre world 施加共同 rigid transform，BRTC/Kabsch 结果应相应变换而不改变相对 correction。
6. **Exact fallback.** 任何验证失败的 typed residual 都退回其上一级已提交结果，而不是产生未定义输出。

这些性质需要用单元测试和数值审计支持，不应只写成文字保证。

---

## 10. Training design

### 10.1 当前 cross96 训练协议

每个 event：

```text
frames:      [t-1, t, t]
sequences:   [camera A, camera A, camera B]
shot_labels: [0, 0, 1]
```

只对 first-post-cut event frame 启用 correction token 与 pose/human head LoRA；两个 pre frames 使用冻结普通 Human3R 路径。

训练来源：

```text
AvatarReX
THuman
MVHuman100
MVHuman200
96 events/source, 384 total, 6 epochs
```

当前 `max_humans=1`，因此 cross96 证明的是 coarse single-person cut proposal 泛化，不是 multi-human training。

### 10.2 当前 loss

保留 V9/V14 event-only correction supervision，并加入 geometry preservation：

```text
camera correction supervision
human correction supervision
gate / residual regularization
self pointmap keep weight   = 20.0
shared pointmap weight      = 0.1
human parameter keep weight = 0.1
```

### 10.3 下一阶段训练原则

先做无训练的 shadow residual decomposition，再决定是否训练。只有当某个 causal feature 对 signed residual 显示跨来源稳定关系时，才训练小模块。

允许训练的目标：

- typed residual uncertainty；
- proposal reliability / acceptance probability；
- BRTC/shadow/Kabsch 的 bounded weight；
- 小幅 root/orientation residual。

不允许重新训练成：

- 从头预测完整 Boundary 的大 MLP；
- 不可解释地复制 shadow mesh；
- 需要未来 post frames 的 transformer；
- 新的 DA3/VGGT/ReID 主干依赖。

推荐 loss：

\[
L=L_{proposal}+\lambda_{type}L_{typed}+\lambda_{unc}L_{calib}
+\lambda_{safe}L_{harm}+\lambda_{keep}L_{fallback}.
\]

其中 `L_harm` 应强调错误方向和 tail，而不是只最小化平均值；`L_fallback` 保证低置信 case 输出接近上一级冻结结果。

---

## 11. 实验设计：ICLR 论文必须回答的问题

### 11.1 Research questions

| RQ | 问题 |
|---|---|
| RQ1 | camera cut 是否真的包含独立的 state failure 和 gauge failure？ |
| RQ2 | non-committing shadow 是否比 carry/reset/普通 adaptive update 更好地兼顾 state purity 与 alignment？ |
| RQ3 | first-post-cut-only proposal 是否能传播整个 shot？ |
| RQ4 | cross-source supervision 是否提高 mean、tail 和 held-out source generalization？ |
| RQ5 | camera-only B0 为什么不能保证 human/scene 对齐？ |
| RQ6 | typed root/layout/orientation verification 是否在相机不变时改善人体？ |
| RQ7 | shadow human gain 能否通过 Verified Shadow Projection 安全进入 clean stream？ |
| RQ8 | 方法是否保持 causal、fixed-memory、multi-cut、no-cut invariance？ |
| RQ9 | 在 Multi-THuMBS-style 和官方协议上，与离线多模块方法差距多大？ |

### 11.2 数据与 split

#### Controlled/cross-source camera cuts

- AvatarReX；
- THuman；
- MVHuman100；
- MVHuman200；
- frozen10 / frozen180 camera-pair-disjoint evaluation；
- 必须新增 actor/capture-disjoint final test，避免只做 camera-pair disjoint。

#### MultiHuman

- `three offset0`：仅作为 development；
- `three offset1`：当前 candidate-specific confirmation；
- `dance/box`：已经被研究过程打开，只能作为 post-hoc support；
- 必须新增至少 2 个从未用于调参的新 multi-human sequences 作为 pristine test；
- 必须包含 variable visibility、identity replacement、1->N/N->1 和 wide-view cuts。

#### EgoHumans / Multi-THuMBS overlap

- 当前只有 `001_legoassemble`；
- 三条 15-frame、2-cut chain 是 local protocol；
- 它属于 EgoHumans，但不能确认是 Multi-THuMBS 官方 capture/split；
- 需要扩展为更长、更多 camera pair/cut/identity 的 manifest；
- 官方 code/supplementary 发布后锁 commit，替换 adapter 和 manifest 重跑。

#### In-the-wild edited videos

- 只做 qualitative 时要同时展示 success / median / failure；
- 若没有 GT，可报告 cut stability、identity、2D reprojection、temporal discontinuity 和用户可视化，不混入定量主表。

### 11.3 Baselines

最低需要：

```text
Original Human3R carry-state
Original Human3R hard reset
Formal V9 continuous correction
V14 old one-Avatar first-cut model
cross96 shadow direct diagnostic
cross96 camera-only B0 runtime
B0 + BRTC-LC
B0 + BRTC-LC + Kabsch
B0 + proposed Verified Shadow Projection
```

外部/适配 baseline：

```text
CUT3R/Human3R base
TTT3R-style adaptive update across cut
ReCal3R or TTSA3R-style update across cut
HumanMM (where executable/comparable)
Multi-THuMBS official numbers/code when available
Multi-THuMBS component-level public reproduction if license/protocol allows
```

重要：不能只与 raw Human3R 比；Multi-THuMBS 对标必须使用当前完整 Movie3R 方法的同一 forward/caches。

### 11.4 指标

#### Camera

- translation error；
- rotation error；
- camera composite `T + 0.02 R`；
- ATE（明确 SE(3)/Sim(3)）；
- P50/P90/P95；
- catastrophic rate。

#### Human global alignment

- W-MPJPE；
- WA-MPJPE；
- fixed-world root/joint/vertex；
- per-cut first-post error；
- post-shot trajectory error。

#### Human local structure

- pelvis-aligned MPJPE/MPVPE；
- PA-MPJPE/PA-MPVPE；
- global orientation error；
- body scale/bone consistency。

#### Multi-human layout and identity

- pairwise root distance；
- pairwise root vector；
- association precision/recall；
- IDs / identity switches；
- variable-visibility coverage；
- wrong confident commit rate。

#### Temporal / physical

- acceleration error；
- jitter；
- foot sliding / penetration（若 evaluator 口径明确）；
- boundary discontinuity and multi-cut drift。

#### Scene/full geometry

- pointmap/depth AbsRel、delta1（有 GT 时）；
- cross-shot point cloud Chamfer/F-score；
- background-only overlap；
- scene/camera/human joint success rate；
- 不能再只按 camera composite 选择可视化。

#### Streaming/safety

- FPS、first-cut latency、amortized latency；
- peak GPU memory；
- fixed external-state size；
- no-cut parity；
- no-future leakage；
- exact fallback count；
- accepted coverage、improve rate、harm >5 cm。

### 11.5 统计要求

- 至少报告 mean/median/P90/P95 与 paired delta；
- confidence intervals 或 bootstrap；
- trainable module 至少 3 seeds；
- per-source/per-sequence 和 worst cases；
- failure taxonomy；
- 所有 threshold 在 dev 冻结，test 不回调；
- 同一主表只能使用同 checkpoint、同 forward、同 detection/association 和同 aggregation。

---

## 12. 当前实验结果：可以怎样进入论文

### 12.1 cross96 camera proposal

Frozen180：

| 方法 | T (m) | R (deg) | Composite | P90 | P95 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| raw hard reset | 3.1006 | 120.838 | 5.5173 | 8.5929 | 9.2995 | 180/180 |
| old one-Avatar | 1.0984 | 59.455 | 2.2875 | 4.9074 | 5.6680 | 107/180 |
| cross24 | 1.0260 | 46.707 | 1.9602 | 4.2133 | 5.2791 | 97/180 |
| cross96 | **0.9073** | **41.302** | **1.7333** | **3.9670** | **4.7186** | **86/180** |

相对 old one-Avatar，cross96：

```text
translation -17.4%
rotation    -30.5%
composite   -24.2%
P90         -19.2%
P95         -16.8%
catastrophic 107 -> 86
```

分来源：

| Source | N | T | R | Composite | Catastrophic |
|---|---:|---:|---:|---:|---:|
| AvatarReX | 48 | 1.0462 | 35.481 | 1.7558 | 19/48 |
| THuman | 48 | 0.2935 | 1.985 | 0.3332 | 2/48 |
| MVHuman100 | 48 | 1.0751 | 63.249 | 2.3401 | 36/48 |
| MVHuman200 | 36 | 1.3169 | 72.222 | 2.7613 | 29/36 |

论文解释：跨来源监督有效，但 learned implicit proposal 没有内生安全保证，wide-view MVHuman 是主要 tail。

### 12.2 camera/human 核心冲突

| Output | Camera composite | Human head error |
|---|---:|---:|
| raw reset | 5.5173 | 1.1046 m |
| shadow event | 1.7333 | **0.4083 m** |
| camera-only B0 runtime | 1.7333 | **1.2288 m** |

这是整篇论文最有价值的 motivating result：

```text
shadow branch knows useful human correction
but its recurrent state is unsafe to commit
camera-only projection preserves state purity
but discards the human gain
```

它直接支持 Verified Shadow Projection，而不是继续扩大 camera-only correct token。

### 12.3 no-cut/state purity

cross96 对原 Human3R 的 no-event sequence：camera、pointmap、confidence、SMPL fields 全部 `max_abs=0.0`；B0 后 camera 与 shadow camera 的 4x4 disagreement 小于 `2.4e-7`。

这是可以进入主文的强结构性结果，而不只是工程测试。

### 12.4 BRTC-LC 的 fresh confirmation

`three offset1`，42 cuts / 125 people，自动匹配：

| 指标 | B0 | BRTC-LC | 相对改善 |
|---|---:|---:|---:|
| Root | 0.3779 | **0.2314** | 38.8% |
| World joint | 0.4117 | **0.2745** | 33.3% |
| World vertex | 0.3891 | **0.2525** | 35.1% |
| Pairwise distance | 0.1341 | **0.0984** | 26.7% |
| Pairwise vector | 0.3297 | **0.2588** | 21.5% |

附加审计：

```text
coverage                       88.0%
accepted residual sign correct 87.3%
root harm >5cm                 7.2%
anonymous matching             125/125
camera max numerical change    0.0
```

必须注明：这是此前冻结 B0，不是 cross96 B0。

### 12.5 Kabsch

MultiHuman 中，BRTC 后 Kabsch 在 `three offset1/dance/box` 均降低 joint/vertex mean，root 和 camera 不变。EgoHumans local provisional：

| 方法 | W | WA | pelvis MPJPE | pelvis MPVPE | fixed joint | fixed vertex |
|---|---:|---:|---:|---:|---:|---:|
| BRTC | 314.059 | 202.461 | 109.266 | 129.960 | 384.729 | 385.238 |
| + Kabsch | **312.769** | **200.029** | **101.526** | **119.928** | **383.933** | **383.791** |

严格零容差下 mapped-pelvis fixed-root 退化 `0.034 mm`，因此 strict decision 是 NO-GO；native Human3R root 实际 bit-exact，若对 topology-conversion proxy 允许 `0.1 mm` tolerance，则它是 qualified candidate。论文应透明报告这层 dual decision。

### 12.6 剩余误差告诉我们什么

Kabsch 后 Ego post-shot：

```text
fixed root                         326.984 mm
oracle remove per-frame shared     148.279 mm
shared squared-error fraction      71.6%
directly remove camera-error       334.016 mm (worse)
```

结论：剩余最大成分是 shared person-root bias，但它不是 camera drift vector。不能再整体移动相机；应估计 human group/person residual。orientation 有价值但不是全部，scale 当前也不是第一优先级。

### 12.7 Multi-THuMBS provisional comparison

当前同一 EgoHumans local forward：

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | ATE | IDs/stream |
|---|---:|---:|---:|---:|---:|---:|
| raw reset | 1088.2 | 405.1 | 109.3 | 130.0 | 1.848 | 5.67 |
| old B0 | 350.6 | 235.2 | 109.3 | 130.0 | 0.119 | 1.00 |
| old B0+BRTC | 314.1 | 202.5 | 109.3 | 130.0 | 0.119 | 1.00 |
| old B0+BRTC+Kabsch | 312.8 | 200.0 | 101.5 | 119.9 | 0.119 | 1.00 |
| Multi-THuMBS paper EgoHumans | 279.0 | 166.0 | 228.3 | 262.2 | 0.7 | 0.97 |

当前只能比较 W/WA 的大概差距：`+33.8/+34.0 mm`。本地 MPJPE/MPVPE/ATE 数值虽然更低，因 split、visibility、miss/FP、alignment、aggregation 和官方 evaluator 未公开，不能宣称胜出。

Multi-THuMBS 的官方 supplementary/code/split 当前不可得；本地只能称：

```text
same source dataset
locally constructed cross-camera cuts
protocol-matched as far as public information
```

---

## 13. 当前已经建立、部分建立和完全缺失的 claim matrix

### 13.1 Established

1. camera cut 可被实现为 raw/shadow 双分支的 non-committing transaction。
2. no-cut 与原 Human3R bit-exact。
3. first-post-cut-only shadow 足以产生可传播的显式 B0。
4. cross-source cut supervision 显著改善 held-out camera mean/tail。
5. camera-only B0 不等于完整 human/scene success。
6. BRTC-LC 在冻结旧 B0 下显著改善 root/layout，并保持 camera 不变。
7. bounded person-local Kabsch 能在 root 不变下改善 orientation-sensitive human metrics。
8. 仅靠当前几何 cue 的 identity dustbin 无法处理一般 equal-count replacement。

### 13.2 Partial

1. cross96 是有用 coarse proposal，但 catastrophic 仍过高。
2. B0-before-WHO 对同可见人数很强，但 variable visibility/identity replacement 未解决。
3. BRTC/Kabsch 有跨 split 正向证据，但尚未接到 cross96。
4. EgoHumans W/WA 接近 Multi-THuMBS，但只是一份 provisional local comparison。
5. multi-cut person correction 已在短链跑通，但没有大规模状态/漂移审计。

### 13.3 Missing

1. cross96+BRTC+Kabsch 同 checkpoint end-to-end 表。
2. shadow human residual 的完整分解与 Verified Shadow Projection。
3. precision-first cross96 camera gate/uncertainty。
4. scene full-geometry 指标。
5. pristine multi-human test sequences。
6. automatic cut detector 下的完整系统。
7. external baseline 可运行对比。
8. official Multi-THuMBS protocol/code 对榜。
9. 正式 runtime、memory、multi-cut、no-future audit。
10. 3-seed training 与统计置信区间。

---

## 14. 为什么这个故事具有创新性，以及如何避免被说成工程拼接

### 14.1 真正的创新点

1. **状态与世界连续性的正交化。** 现有 streaming work 通常只优化怎样更新 state；Movie3R 明确规定 cut 后哪些 state 没有提交权，并用独立显式 gauge 恢复 world continuity。
2. **Non-committing shadow transaction.** 旧状态仍可以提供有价值的跨镜先验，但只在 read-only branch 中使用，这是对 recurrent reconstruction state lifecycle 的新设计。
3. **Implicit-to-explicit projection.** learned latent correction 不是最终不可审计输出，而是 proposal；最终 commit 被投影到 camera/root/orientation 等有明确语义的低维几何空间。
4. **Typed ownership.** shot Boundary 对 camera/scene/all humans 共享；person correction 被明确限定为固定 camera world 中的人体预测 refinement，避免混淆 gauge transform 与 structure correction。
5. **Safety as method, not appendix.** exact fallback、tail risk、no-cut parity、state purity 和 no-future leakage 是算法定义的一部分。

### 14.2 单独不够新颖的部分

以下不能单独作为贡献：

- hard reset；
- camera pose 相乘；
- Hungarian；
- ray triangulation；
- Kabsch；
- fixed shot transform；
- PySceneDetect。

它们的论文价值来自事务框架中的角色和互相验证关系。

### 14.3 必须做的关键消融

为了证明不是模块堆叠，至少要有：

```text
carry old state
hard reset only
commit full shadow state/output
shadow camera -> B0 only
shadow human direct copy, no verification
explicit geometry only, no shadow
shadow + typed decomposition, no verification
shadow + verification + transactional commit (full)
```

还要有：

```text
one global Boundary only
+ person root
+ person orientation
+ verified shadow projection
```

这样才能实证每一层 factorization 的必要性。

---

## 15. 预判审稿人质疑与回答策略

### Q1：这不就是 hard reset 加一个 alignment 吗？

回答：普通 alignment 没有区分 recurrent state 的读/写所有权，也无法解释 `shadow human 0.408 m` 与 `B0 runtime 1.229 m` 的差异。Movie3R 的核心是 clean state 与 expressive proposal 的事务分离，以及 typed verified commit。

### Q2：BRTC 和 Kabsch 都是经典几何，创新在哪里？

回答：它们是显式 verifier，而不是论文单点创新。贡献是把不可安全提交的 latent correction 投影为可验证的低维类型，并证明 camera gauge、root/layout 和 orientation 必须分解处理。

### Q3：为什么不直接提交 shadow human？

回答：shadow decoder 使用与新 shot 不兼容的旧 recurrent state，其当前帧输出可能好，但继续传播会污染 state。需要实验比较 direct shadow commit 的短期收益与后续 drift，并证明 transactional projection 保留收益而不污染未来。

### Q4：为什么不使用 VGGT/DA3/SLAM？

回答：目标是保留统一、在线、固定内存的 Human3R paradigm。Multi-THuMBS 已展示多模型离线路线；Movie3R 研究的是一个 recurrent foundation model 如何自身跨 cut。外部模型可作为 baseline，不是主方法依赖。

### Q5：cross96 还有 86/180 catastrophic，是否说明方法失败？

回答：它说明 learned proposal 有效但不安全，恰好支持 proposal–verify–commit 论点。投稿前必须证明 verifier/gate 明显降低 tail；若做不到，cross96 只能作为 analysis，不足以支撑强论文。

### Q6：Multi-THuMBS 数字并不公平。

回答：当前文档明确只称 provisional same-source comparison。投稿时要么使用官方协议，要么公开自建 manifest/evaluator、重跑可执行 baselines，并把 paper-only numbers 分表展示。

### Q7：identity replacement 还没解决，能叫 multi-person 吗？

回答：不能过度声称。可以采用 precision-first abstention 并明确 coverage；强版本应从 Human3R 已有 native appearance tokens 中提取正交 identity cue，而不是继续调几何阈值。

### Q8：person-local transform 是否破坏“one shared Boundary”？

回答：不破坏。`B0` 仍是唯一 shot local-to-world Boundary，作用于 camera、scene 和所有 humans。BRTC/Kabsch 是在固定 world camera 下修正 base model 的 per-person structure bias，相当于 human-head refinement，不是第二套 shot coordinate system。

---

## 16. 投稿主表和主图设计

### Figure 1：任务与失败

四列：continuous Human3R、carry-state cut failure、hard-reset gauge discontinuity、Movie3R unified output。必须同时画 camera、scene 和 humans。

### Figure 2：Transactional architecture

突出 raw/shadow state ownership、typed proposal、verification、commit 和 propagation。用实线表示提交，虚线/灰色表示 read-only discarded branch。

### Figure 3：误差分解

展示同一个 case：

```text
raw reset
-> B0 camera aligned but human wrong
-> BRTC root/layout
-> Kabsch orientation
-> Verified Shadow Projection
```

### Figure 4：关键证据冲突

camera composite 与 human error 的 raw/shadow/B0 三柱图；这是方法动机的核心。

### Figure 5：risk–coverage / failure taxonomy

展示 accepted coverage 与 error/harm/catastrophic，证明 safety gate 不是只优化 mean。

### Main Table 1：cross-source camera alignment

raw、carry、formal V9、old V14.1、cross24、cross96、full verified method；分 source + overall + tail。

### Main Table 2：完整 human/scene/multi-person

同 checkpoint 的 B0、BRTC、Kabsch、VSP；W/WA/fixed world/layout/scene/IDs。

### Main Table 3：external comparison

HumanMM/Multi-THuMBS/其他可运行 baseline；明确 causal、future、额外模型、优化时间、memory 和指标协议。

### Main Table 4：streaming contract

no-cut parity、multi-cut、no-future、FPS、cut latency、peak memory、external-state size。

---

## 17. 下一步可完成计划

以下按依赖顺序执行。任何阶段失败都记录，不跨过证据断点继续包装下游结果。

### Phase 0：冻结和证据统一（2–3 天）

目标：建立唯一可信的 checkpoint/data/evaluator matrix。

任务：

1. 给 cross96、旧 B0、BRTC/Kabsch policy 记录 SHA256、config、manifest、commit。
2. 创建统一 prediction schema：每帧保存 raw/shadow/B0 camera、pointmap confidence、humans、native/stable IDs、joints/vertices、visibility。
3. 明确 train/dev/test overlap：camera pair、actor、capture、frame。
4. 新建 pristine test manifest，旧 `three/dance/box` 不再冒充完全 blind test。
5. 统一 native root、SMPL-X->SMPL mapped pelvis 和各 evaluator 语义。

完成条件：任意主表 row 能追溯到一个 checkpoint、一次 forward 和一个 manifest。

### Phase 1：cross96 全链闭环（3–5 天）

目标：回答 BRTC/Kabsch 在 cross96 上是否仍成立。

固定执行：

```text
cross96 raw/shadow
-> camera-only B0
-> frozen BRTC-LC, parameters unchanged
-> frozen Kabsch, parameters unchanged
-> camera/human/layout/scene/safety metrics
```

先在 existing dev 做 compatibility，不调参数；再打开新增 pristine confirmation。

Go：

- camera 保持 cross96 B0 数值不变；
- BRTC root gain >= 8%；
- layout-vector gain >= 5%；
- root harm >5 cm <= 10%；
- Kabsch joint/vertex 非退化；
- 至少两个不同 source/sequence family 正向。

No-Go：若 cross96 camera error 导致 BRTC observability/association 崩溃，则必须先做 camera proposal verification，不继续调 BRTC gate。

### Phase 2：shadow human gain 分解（3–5 天，CPU 优先）

目标：解释并利用 `0.4083 vs 1.2288 m`。

对每个 matched human 计算：

```text
H_shadow - B0(H_raw)
```

分解 root、global orientation、scale、articulation、shape、camera-relative depth。对每个分量统计：

- signed GT residual correlation；
- oracle commit upper bound；
- source/camera-angle dependence；
- 与 BRTC/Kabsch proposal 的 cosine、magnitude ratio、agreement；
- improve rate、P90/P95、catastrophic；
- 是否可由预测 confidence/gate 识别。

停止规则：只有跨至少两个 source、signed correlation 稳定且 tail 可门控的分量进入 Phase 3。预计优先级为 root translation、global orientation；scale/articulation 暂缓。

### Phase 3：Verified Shadow Projection（5–8 天）

按复杂度递增：

```text
VSP-0: shadow root only, bounded
VSP-1: agreement-gated shadow root vs BRTC
VSP-2: shadow orientation vs Kabsch
VSP-3: root + orientation typed commit
VSP-4: small learned uncertainty/weight head（仅前三者稳定后）
```

每个候选都必须 exact fallback 到 frozen cross96+BRTC+Kabsch。

Go：

- W 和 WA 相对 baseline 至少改善 5%，或 fixed-world root/joint 至少改善 8%；
- camera bit-exact；
- harm >5 cm <=10%；
- P90/P95 不退化；
- 至少两个 source/sequence family 均正向；
- no-cut 和 rejected case exact parity。

### Phase 4：camera proposal verification（与 Phase 2/3 可交错，5–7 天）

目标：显著降低 cross96 的 MVHuman catastrophic tail。

不新增外部模型，优先测试：

1. correction-token gate/calibration；
2. raw-shadow correction magnitude/rotation trust region；
3. camera-human proposal consistency；
4. raw/shadow pointmap confidence and geometry preservation residual；
5. causal forward/reverse boundary cycle verification：只使用 last-pre 和 first-post，不读未来；
6. cross96 与旧 coarse checkpoint 的 disagreement/selection（仅作为成本 ablation）。

报告 risk–coverage，不允许只删除难例后报告低 mean。

强 Go 目标：catastrophic `86/180 -> <=43/180`，同时 overall composite 不退化超过 1%。

最低 Go：catastrophic 至少相对下降 30%，P95 同时改善，且有可部署 confidence/abstention 定义。

### Phase 5：multi-human identity 与 variable visibility（4–7 天）

不再扫描纯几何 dustbin 阈值。优先：

- 从 Human3R 已有 native human/image tokens 提取 frozen appearance similarity；
- geometry 仅负责 gauge-aware WHERE，appearance 负责正交 WHO；
- dustbin/mutual/margin abstention；
- 1->N、N->1、equal-count replacement 单独报告；
- 无可靠 match 时允许 B0-only 或 BRTC unmatched fallback。

Go：wrong committed association 接近 0，correct coverage 明确且不通过错误 refinement 换取均值。

### Phase 6：Multi-THuMBS 对标闭环（5–10 天）

1. 扩展 EgoHumans local manifest，不再只用 3x15 frames。
2. 同 forward 运行 raw/B0/BRTC/Kabsch/VSP。
3. 冻结公开 provisional evaluator 和 miss/FP/visibility 规则。
4. 报 W/WA/MPJPE/MPVPE/Accel/ATE/IDs 与 Movie3R fixed-world/safety 两层指标。
5. 官方 code/supplementary 可得后锁版本并切换 official adapter。

中间目标：local provisional W/WA 从 `312.8/200.0` 降到 `<295/<185 mm`。

强目标：同公平协议达到或超过 Multi-THuMBS `279/166 mm`，同时保持 online、no future、无新增预训练模型。

### Phase 7：完整 streaming 和论文实验（7–10 天）

- automatic cut detector；
- 长 multi-cut stream；
- no-future truncation test；
- detection permutation；
- zero-match fallback；
- scene metrics；
- runtime/memory/FPS；
- 3 seeds / bootstrap；
- success/median/failure demo-style visualization；
- 所有 baselines 与 ablations。

### Phase 8：写作与复现（与实验并行）

- 主文 9 页以内的故事收敛；
- appendix 放公式、协议、negative results、完整 per-source tables；
- config/checkpoint/manifest hash；
- 一键 evaluator 和 demo export；
- README 中声明 official/provisional 指标边界；
- release 前清理 hard-coded GT/path 和易失 `/dev/shm` artifact。

---

## 18. ICLR Go/No-Go gates

### 18.1 最小可投稿版本

必须同时满足：

1. 统一 cross96 checkpoint 下，B0+BRTC+Kabsch 端到端同时改善 camera 与 human/layout。
2. no-cut bit-exact，shadow state 永不提交，first-post-only 和 no-future 有自动测试。
3. 至少一个真正 pristine multi-human test family。
4. 与 carry/reset/V9 continuous/strong B0 baseline 的完整消融。
5. local Multi-THuMBS-style 同 forward 对比透明、可复现。
6. runtime/memory/multi-cut 指标齐全。

如果只有这些，论文可以主打 state–gauge transaction，BRTC/Kabsch 作为 verified refinement；创新强度中等，依赖实验规模和写作质量。

### 18.2 强 ICLR 版本

在最小版本上再满足：

1. Verified Shadow Projection 把 shadow human gain 安全投影到 clean stream，并有明显端到端收益。
2. camera verification 将 catastrophic tail 至少减半或形成很强 risk–coverage。
3. automatic identity/variable visibility 有 precision-first 结果。
4. 与 Multi-THuMBS 在公平协议上接近或超过，同时展示数量级更低的 latency/memory 和严格 causality。
5. scene、human、camera 的联合 full-geometry success，而非只优化 camera。

### 18.3 必须停止投稿包装并继续研究的情况

- cross96+BRTC/Kabsch 在新 test 上不稳定或总体退化；
- 无法利用 shadow human gain，也无法解释它只是 evaluator artifact；
- catastrophic 仍过高且没有可观测 gate；
- 只有一个 capture 的自建短链结果；
- 只能与 raw Human3R 比，缺少近邻 baseline；
- 最终方法依赖大量 test-specific threshold；
- scene 明显恶化而论文仍称 unified human–scene reconstruction。

---

## 19. 推荐的最短实验顺序

后续按本文继续工作时，严格按以下顺序：

```text
E0 provenance + schema + pristine split freeze
E1 cross96 -> BRTC -> Kabsch compatibility
E2 shadow vs B0(raw) typed residual decomposition
E3 root-only Verified Shadow Projection
E4 orientation-only Verified Shadow Projection
E5 root+orientation verified commit
E6 cross96 camera confidence / cycle verification
E7 variable-visibility identity with existing native tokens
E8 expanded EgoHumans provisional Multi-THuMBS protocol
E9 full multi-cut + automatic detector + runtime
E10 final baselines, ablations, figures and statistics
```

每个实验记录：

```text
hypothesis
causal inputs
checkpoint/config/manifest hash
dev/test split
frozen parameters
metrics and paired deltas
coverage/tail/worst cases
decision: reject / revise / promote
next experiment implied by evidence
```

---

## 20. 最终论文故事的精炼版本

### 20.1 一句话

> Movie3R treats every camera cut as a transaction: it resets the recurrent state, uses the old state only to propose a world bridge, explicitly verifies typed camera and human residuals, and commits only safe corrections to the new streaming trajectory.

### 20.2 三句话

1. Camera cut 同时破坏 recurrent state continuity 和 world gauge continuity，而 camera 对齐后仍存在 person-relative structure bias。
2. Movie3R 运行 clean raw reset 与 non-committing shadow proposal，把隐式 correction 分解为 shot gauge、person root/layout 和 orientation。
3. 系统只提交通过显式几何验证的低维修正，从而保持 no-cut parity、state purity、causality 和 fixed memory，同时恢复跨 shot 的 camera/human/scene 连续性。

### 20.3 当前最诚实的结论

项目已经找到一个清晰、可证伪且有创新潜力的 ICLR 论文问题，也已经分别证明 coarse gauge、root/layout 和 orientation correction 的可行性。现在没有理由再回到“让 B0 单模型一步到位”的路线，也不应继续无结构地堆更多预训练模型。下一阶段唯一主线应是：

```text
统一 cross96 全链证据
-> 分解 shadow human gain
-> 用显式几何验证 typed residual
-> 完成安全 transactional commit
-> 在真正 held-out multi-human 和 Multi-THuMBS-style 协议上闭环
```

如果这条链成立，Movie3R 的贡献将不只是一个更准的对齐模块，而是一个适用于 recurrent 3D foundation models 的新 camera-cut state-management paradigm。

---

## 21. 论文最终目录、Limitations 与 ICLR reproducibility

### 21.1 建议主文目录

```text
1. Introduction
2. Related Work
3. Camera Cuts as State–Gauge Transactions
4. Movie3R
   4.1 Clean Reset and Read-Only Shadow Proposal
   4.2 Explicit Coarse Gauge Recovery
   4.3 Typed Human Residuals
   4.4 Geometric Verification and Transactional Commit
   4.5 Causal Propagation and Complexity
5. Experimental Setup
6. Results
   6.1 Cross-Source Gauge Recovery
   6.2 Human Root/Layout/Orientation Alignment
   6.3 Multi-Shot and Multi-Person Reconstruction
   6.4 Safety, Streaming, and Ablations
7. Limitations
8. Conclusion
```

篇幅紧张时，将完整公式、per-source tables、negative experiment ledger、协议审计和更多可视化放入 appendix；主文始终围绕 proposal–verify–commit 展开。

### 21.2 必须主动写出的 limitations

1. 当前主实验首先假设 cut timestamp 已知，automatic detector 不是核心贡献。
2. last-pre / first-post 几何默认 cut 两侧时间间隔足够小；大 frame gap 时真实人体动作会与 alignment residual 混淆。
3. BRTC 在低 parallax、ray gap 大、人体严重截断时会 fallback，因此 coverage 不是 100%。
4. Kabsch 可能把真实 torso motion误认为 orientation drift，只适用于相邻 boundary 和有界 correction。
5. 纯几何 association 无法可靠处理外观相似、位置互换的 equal-count identity replacement。
6. 当前 scale、articulation/shape 和 scene-surface residual 尚未解决；不能把 camera/root 改善描述成完整 photorealistic reconstruction。
7. 每次 cut 的 Boundary 和 person state 仍可能在长 multi-cut 视频中积累漂移，需要正式长序列实验。
8. 训练数据以受控多相机人类数据为主，对真实电影中的运动模糊、焦距变化、遮挡和非同步剪辑的泛化尚待验证。

主动报告这些边界不会削弱论文；它能说明方法的精确适用范围，并让 precision-first fallback 合理化。

### 21.3 Reproducibility checklist

投稿 artifact 至少应包含：

- 每个 checkpoint、policy、manifest 和代码版本 hash；
- 完整训练配置、数据来源、事件构造、epoch/seed/compute；
- oracle-cut 与 automatic-cut 两套入口；
- raw/shadow/B0/typed residual 的统一 prediction schema；
- metric 公式、单位、alignment、visibility、miss/FP 和 aggregation；
- frozen dev/test split 与 overlap audit；
- no-cut/no-future/state-purity/multi-cut 的自动测试；
- 每个主表 row 的一键复现命令；
- failure cases 和 rejected experiments，而不只发布最佳配置；
- inference FPS、cut latency、memory 与硬件信息。

### 21.4 Conclusion 草稿

> We studied streaming human–scene reconstruction across camera cuts and showed that the central challenge is not a single pose discontinuity, but the coupled failure of recurrent-state ownership, world gauge, and camera-relative human structure. Movie3R addresses this challenge as a transaction: it commits a clean reset trajectory, uses the previous state only for a read-only shadow proposal, and projects the proposal into explicitly verified, typed geometric updates. This design preserves causal fixed-memory inference and ordinary Human3R behavior while enabling cross-shot camera and human continuity. More broadly, our results suggest that stateful 3D foundation models should separate what they may use for inference from what they are allowed to commit—a principle that may extend beyond edited videos to relocalization, scene re-entry, and abrupt domain transitions.

---

## 22. 关键证据索引

```text
versions/v14/cut_first_cross_source/EXPERIMENT_HANDOFF_20260802.md
versions/v14/cut_first_cross_source/RESULTS.md
output/v14_cut_first_cross_source/eval_cross96_180/four_source_b0_evaluation.md

versions/v14/docs/V14_B0_TWO_VIEW_TRIANGULATION_FINAL_20260731.md
versions/v14/docs/V14_BRTC_PERSON_LOCAL_ORIENTATION_KABSCH_20260801.md
versions/v14/docs/V14_BRTC_GLOBAL_ORIENTATION_KABSCH_EGOHUMANS_20260801.md
versions/v14/docs/V14_BRTC_KABSCH_RESIDUAL_DECOMPOSITION_20260801.md

versions/v14/docs/V14_MULTITHUMBS_PUBLIC_PROTOCOL_DATA_OVERLAP_AUDIT_20260801.md
versions/v14/docs/V14_BRTC_MULTITHUMBS_EGOHUMANS_20260801.md
versions/v14/docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md

versions/v14/docs/V14_ICLR_FINALIZATION_PLAN_20260729.md
versions/v14/docs/V14_FULL_METHOD_DESIGN_FOR_REVIEW_20260729.md
versions/v14/docs/Movie3R-V14.MD
versions/v9/docs/METHOD_OVERVIEW.md

/data/wangzheng/iJCV-CODE/paper/CUT3R.pdf
/data/wangzheng/iJCV-CODE/paper/Human3R-ori.pdf
/data/wangzheng/iJCV-CODE/paper/UniCon3R.pdf
/data/wangzheng/iJCV-CODE/paper/TTT3R.pdf
/data/wangzheng/iJCV-CODE/paper/TTSA3R.pdf
/data/wangzheng/iJCV-CODE/paper/ReCal3R.pdf
/data/wangzheng/iJCV-CODE/paper/HumanMM.pdf
/data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf
/data/wangzheng/iJCV-CODE/paper/Trophies.pdf
```
