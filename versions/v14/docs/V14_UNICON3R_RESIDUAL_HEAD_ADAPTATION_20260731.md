# V14 UniCon3R Residual Head 到 Movie3R 人体精对齐的实现映射

> 日期：2026-07-31
> 论文：`/data/wangzheng/iJCV-CODE/paper/UniCon3R.pdf`，arXiv:2604.19923v2
> 本地源码状态：未发现 UniCon3R 官方源码或 checkpoint；论文写明 code/models 将在接收后发布
> 本文目标：提取论文中 contact/residual pathway 的精确结构，并设计一个不改 B0 相机、只修 camera-local human root/orientation 的最小适配，不把“UniCon-style”当成泛泛的 token 命名。

## 1. 结论先行

UniCon3R 对当前问题最有价值的不是“预测脚接触”，而是下面这个已经被消融验证的因果
路径：

```text
当前人 token
+ 当前图像/历史 scene state
+ 明确的局部 metric geometry
+ 上一帧 contact memory
→ per-person contact prompt
→ 和 human token 一起通过冻结 decoder
→ contact-conditioned latent residual
→ 原 Human3R SMPL-X head
→ 改变人体 root translation 和 global/body orientation
```

论文在 RICH moving-camera 40 sequences 上把 residual 置零后，只重跑最终 SMPL-X head，
得到：

- world-frame body root 平均变化 `0.3 m`；
- mean per-joint angular shift `2.8°`；
- maximum per-joint angular shift `10.2°`；
- contact logits 差异严格为 `0.0`。

这直接证明：contact branch 的 residual 不是一个无效 auxiliary head；它拥有改变 Human3R
人体 camera-local placement/pose 的能力，而不需要改 camera pose head。

但必须纠正一个容易造成错误实现的说法：

```text
UniCon3R 原版 residual head 不直接输出 Δroot 或 Δorientation。
它输出与 human-head 输入同维的 latent residual ΔH；共同微调的完整 SMPL-X head
再解码 root、pose、shape、expression。
```

Movie3R 已经证明 B0 camera 是合格粗对齐，且 V9 隐式模型已经充分训练到粗对齐上限。
因此不能再次解冻相机或让一个全自由 latent residual 同时承担 camera/human gauge。最小正确
适配应为：

```text
冻结 B0 camera、CUT3R/Human3R backbone/decoder、scene heads 和 Multi-HMR
→ 构造真正 per-person、scene-grounded 的 registration/contact token
→ 只输出有界 camera-local Δt 和 global-root ΔR
→ shape、expression、local body pose、camera、pointmap bit-exact 不变
```

这个版本可以看成 **UniCon3R 的 amortized contact correction** 与 **Multi-THuMBS 的
per-person boundary optimization** 的交集：Multi-THuMBS 给出显式优化变量和 teacher/oracle，
UniCon3R 给出一次前向学习这些修正的架构。

## 2. UniCon3R 原版的精确输入与张量流

### 2.1 基础 Human3R/CUT3R 流

论文沿用 Human3R/CUT3R：

- 当前 RGB 编码为 image tokens `F_t`；
- persistent scene state 为 `S_{t-1}`；
- pose token 为 `z`；
- 当前帧检测到 `N_t` 个人，每人一个 human prompt：

  \[
  H_t\in\mathbb{R}^{N_t\times c},\qquad c=768.
  \]

- Multi-HMR ViT-L encoder 提供冻结的人体先验特征 `F^u_HMR,t`；
- 两组 interleaved CUT3R 4D decoders：depth 12、12 attention heads、decoder/state
  dimension 768；
- CUT3R image encoder 是 ViT-L，token dimension 1024。

Human3R 基础路径为：

\[
[F'_t,z'_t,H'_t],S_t=
\mathrm{Decoders}([F_t,z,H_t],S_{t-1}),
\qquad Y_t=\mathrm{Head}_{human}(H'_t).
\]

`Y_t` 是完整 SMPL-X 参数，包括 pose、shape、expression 和 camera-local root
translation；camera pose `T_t` 由独立 pose head 解码。

### 2.2 Semantic scene context

每个人用 human prompt 分别查询当前帧和历史 scene state：

\[
U_{curr}=\mathrm{CA}_{curr}(H_t,F_t),\qquad
U_{mem}=\mathrm{CA}_{mem}(H_t,S_{t-1}).
\]

学习一个逐通道 soft gate：

\[
\gamma_t=\sigma\bigl(
\mathrm{MLP}(H_t\oplus U_{curr}\oplus U_{mem})
\bigr),
\]

\[
U_{scene}=\gamma_t\odot U_{curr}
+(1-\gamma_t)\odot U_{mem}.
\]

所以 `γ_t` 不是单个 accept/reject scalar，而是 `[N_t,768]` 的 element-wise gate，决定
每个人、每个 latent channel 更依赖当前图像还是 persistent memory。

### 2.3 Explicit metric geometry token

论文把上一帧 world pointmap 简写为：

\[
X_{t-1}:=X^{world}_{t-1}.
\]

对当前第 `n` 个人的 2D anchor `u_t^n`，在 `X_{t-1}` 上取局部 square window：

- window half-size：24 pixels，即约 `49×49` 输入区域；
- RoIAlign 输出：`7×7`；
- pointmap channel：3D XYZ；
- 对 `7×7×3` 的坐标做 spatial mean，得到：

  \[
  \phi_{geo}(X_{t-1},u_t^n)\in\mathbb{R}^{3};
  \]

- 两层 GELU MLP 投影到 768 维：

  \[
  G_t=\{\mathrm{MLP}(\phi_{geo}(X_{t-1},u_t^n))\}_{n=1}^{N_t}
  \in\mathbb{R}^{N_t\times768}.
  \]

这里不是 pointmap loss，也不是输出后取一个 median depth；geometry 已经作为 prompt 输入
进入人体修正的计算图。

### 2.4 Temporal contact momentum

上一帧 refined contact token `C'_{t-1}` 被对齐到当前人数：

\[
M_t=\mathrm{MLP}(\mathrm{Align}(C'_{t-1};N_t))
\in\mathbb{R}^{N_t\times768}.
\]

论文的 `Align` 很简单：保留前 `min(N_{t-1},N_t)` 行，人数减少则截断，人数增加则补零。
它不是完整 Re-ID。在连续、单人居多的数据上可用，但不能原样用于 Movie3R 多人 shot；
Movie3R 必须按已经确定的 identity association 重排 token。

### 2.5 Contact prompt fusion

最终 contact prompt 是：

\[
C_t=\mathrm{MLP}
(H_t\oplus U_{scene}\oplus G_t\oplus M_t)
\in\mathbb{R}^{N_t\times768}.
\]

附录说明：除特别注明外，论文中所有 MLP 都是 two linear layers + GELU，hidden
dimension 768。

## 3. Exact residual/contact head

### 3.1 Contact token 进入冻结 decoder

UniCon3R 不是在 Human3R 输出后独立读取 contact：

\[
[F'_t,z'_t,H'_t,C'_t],S_t=
\mathrm{Decoders}([F_t,z,H_t,C_t],S_{t-1}).
\]

即 token layout 是：

```text
[pose z, image F, human H, contact C]
```

论文明确称这组 4D decoders 为冻结的 Human3R/CUT3R decoder-and-memory pipeline。
附录的 attention audit 也确认 human/contact 双向交互：早层 `H→C` 最强，后层 `C→H`
最强，且显著高于同尺寸 image-token slice control。

### 3.2 拼接冻结 Multi-HMR prior

decoder 后，human/contact token 分别拼接同一个 Multi-HMR feature：

\[
\widetilde H_t=H'_t\oplus F^u_{HMR,t},\qquad
\widetilde C_t=C'_t\oplus F^u_{HMR,t}.
\]

主模型用 ViT-L/896。结合 Human3R 配置可知 decoder token 是 768、Multi-HMR ViT-L
feature 是 1024，最终 human/contact head 输入是 1792 维；附录也明确 human head 的 hidden
dimension 是 `4×1792`。

### 3.3 Contact head

\[
s_t^v=\mathrm{Head}_{contact}(\widetilde C_t)
\in\mathbb{R}^{N_t\times6890},
\qquad
\hat c_t^v=\sigma(s_t^v).
\]

它预测的是 **SMPL 6890 vertices** 的 dense binary contact，而 human regressor 仍在
SMPL-X 空间。这和当前 Movie3R 的 SMPL-X 10475 vertices 不是同一 topology，适配时必须
显式提供 SMPL-X→SMPL mapping，不能用 array index 假装一致。

### 3.4 Latent residual head

\[
\Delta H_t=\mathrm{Head}_{residual}(\widetilde C_t),
\qquad
\overline H_t=\widetilde H_t+\Delta H_t,
\]

\[
Y_t=\mathrm{Head}_{human}(\overline H_t).
\]

`Head_residual` 的输出必须和 `H~` 同维，因而是 per-person human-head latent residual，
不是 3D translation。按照附录统一约定，它是 two-layer GELU MLP、hidden 768。

Human head 内部为彼此独立的 two-layer MLP，hidden `4×1792=7168`，分别回归：

- body/root pose；
- shape；
- expression；
- root translation。

所以原版 residual 有能力同时改变 root translation、global/root orientation、local pose，
甚至经共同微调的 head 改变 shape/expression。它的物理 grounding 收益很强，但这也是
Movie3R 第一阶段不能直接照搬全自由 residual 的原因。

## 4. 原版 loss、数据与冻结策略

### 4.1 总损失

\[
L_{total}=L_{4D}+L_{SMPLX}+\lambda_cL_{contact},
\qquad \lambda_c=1.0.
\]

`L4D`：

- CUT3R confidence-aware L2,1 pointmap loss，`α=0.2`；
- per-frame RGB MSE；
- camera/appearance 等继承的 4D reconstruction supervision。

`LSMPLX`：

- Human3R L1 SMPL-X supervision；
- 主文列出 parameter、mesh、joint、2D reprojection 项。

contact loss：

\[
L_{contact}=L_c^{3D}+\lambda_pL_p,
\qquad \lambda_p=1.5.
\]

- `L_c^{3D}=FocalBCE(s_t^v,c_{gt}^v)`；
- focal `γ=2.0`、`α=0.25`；
- positive-class weight 10；
- `L_p` 是 DECO 风格的 part-level contact loss。

### 4.2 数据

训练用 RICH + BEDLAM balanced mixture：

- RICH：multi-camera video、scanned scene、SMPL-X、dense world-frame body-scene
  contact；用 official split，把 moving-camera subset 留作测试；
- BEDLAM：去掉只用 panoramic HDRI 表示 environment 的序列，剩 2700 training
  sequences；BEDLAM 没有 contact label，所以只优化 `L4D+LSMPLX`；
- contact supervision 只在 RICH 样本启用。

本地数据现状必须单独记录：

- `/data/wangzheng/iJCV-CODE/data/RICH` 当前不存在；
- `/data/wangzheng/iJCV-CODE/data/BEDLAM` 当前目录为空；
- 本地有 AvatarReX、THuman、MVHuman 的 SMPL-X/metric geometry 训练路径，但没有现成
  6890-vertex RICH contact labels。

因此现在可以训练 Movie3R root/orientation residual，但不能声称严格复现 UniCon3R
contact head；要复现 dense contact branch，需要补 RICH 或生成经过审计的 scene-contact
pseudo labels。

### 4.3 优化

- AdamW：`β1=0.9, β2=0.95, weight_decay=0.05`；
- 前 5 epochs linear warmup：`1e-6 → 1e-5`；
- 之后 cosine decay 回 `1e-6`；
- 共 100 epochs；
- per-GPU batch 4，gradient accumulation 2；
- 8×A5000，effective batch 64；
- mixed precision + gradient checkpointing。

### 4.4 冻结/训练模块

冻结：

- CUT3R ViT-L scene/image backbone；
- 两组 4D recurrent decoders；
- persistent state machinery；
- Multi-HMR encoder。

训练：

- `CA_curr`、`CA_mem` scene-context cross-attention；
- prompt-construction MLPs：gate、geometry、momentum、fusion；
- contact head；
- latent residual head；
- Human3R SMPL-X prediction head；
- Human3R segmentation head。

这里的关键不是“只训练一个孤立 residual MLP”。论文同时微调 human head，使它学会
解释 contact-conditioned latent；Parallel Readout 用相同 contact supervision 却不反馈到 human
reconstruction，WA/W 为 `97.9/153.5 mm`，几乎不优于 Human3R† 的 `97.5/153.2 mm`。
完整 latent feedback 才达到 `81.5/129.8 mm`。

## 5. 为什么它可能解决当前 camera-local human root bias

### 5.1 当前误差的自由度正好在 human head

Human3R 的世界人体可写成：

\[
P^{world}_t=T_tP^{cam}_t.
\]

B0 已经负责把 shot 后的 `T_t` 拉回正确粗 gauge；现有 GT 可视化和实验表明，剩余误差
主要在 `P_cam`：root depth、lateral root、scale/body structure 和 global orientation。

UniCon3R residual 插在 human token 与 `Head_human` 之间。它可以改变 human head 的
translation decoder 和 pose decoder，却不必改变 pose token/camera head。因此它作用在当前
已经确认的正确自由度，而不是再次把 camera 与 human 一起整体移动。

### 5.2 它补的是“关系证据”，不是更大的隐式模型

V9 已经是充分训练的 Human3R relation-prompt/latent-correction baseline。它证明隐式 RGB、
pose、human token 和 memory 可以做到粗对齐，但上限仍是 B0。再次把同类 token 加宽、训练
更久，不会自动产生缺失的 metric root evidence。

UniCon3R 相比 V9 真正新增的是：

```text
明确的 person-conditioned scene context
+ 显式 local metric geometry G_t
+ contact supervision
+ contact latent 必须反馈到 human output
```

论文消融也支持这个分解：

| Variant | WA | W | Foot sliding | Jitter |
|---|---:|---:|---:|---:|
| Parallel readout | 97.9 | 153.5 | 35.3 | 262.5 |
| + scene context | 89.4 | 140.6 | 35.0 | 260.1 |
| + explicit geometry | 85.9 | 134.5 | 33.1 | 244.6 |
| + momentum | 85.3 | 135.1 | 32.0 | 231.8 |
| + latent refinement | **81.5** | **129.8** | **31.5** | **221.4** |

这说明下一步不能再只有 semantic correction token；必须让独立 scene/root relation 成为输入，
并用明确人体 residual loss 训练。

### 5.3 适用边界

contact 只在人体与可靠 rigid support surface 有关系时提供强约束。它对以下情况不是万能的：

- 跳跃、悬空、躺在软体物体上；
- scene geometry 错误或人体把背景完全遮住；
- shot 跨不同物理场景；
- 人物真实移动使 pre/post translation 不连续。

所以它应是 gated person residual，不是强制每个人贴地。论文也承认 scene error 会传播到
contact/body alignment。

## 6. 不能原样复制 UniCon3R 的三个原因

### 6.1 原版是连续视频，不是 shot boundary

原版在上一帧 `Xworld_{t-1}` 的**当前人 anchor 像素**附近做 RoIAlign，依赖相邻帧视觉
连续性。shot 后当前人的 `u_t` 与上一镜头相同像素没有对应关系。直接用
`RoIAlign(Xworld_pre,u_post)` 会采到任意背景。

V14 的 CUT3R virtual-view probe 已实际证明，从 pre state 用 post B0 camera query 的
virtual pointmap主要返回远处低置信静态背景：两名可见人的 virtual root-ray evidence
`+2.133/+1.665 m`，而 GT residual 为 `-0.259/-0.246 m`，方向一致率 0%。因此不能用
virtual pointmap 假装恢复了 UniCon3R 的连续-frame geometry。

### 6.2 Human3R internal pointmap 与人体 head 共享 monocular bias

现有 180-cut/400-person 和 near/far stress 已表明，简单 person-mask pointmap median 的
>5 cm harm 可达约 17%–34%。UniCon3R 把 geometry 送进 learned residual，并有 contact/SMPL
监督；它并没有证明“median depth 后处理”安全。Movie3R geometry token 必须包含独立、
可校准的 scene evidence，或至少由训练监督学会拒绝共享 bias。

### 6.3 原版 full latent residual 会改太多人体自由度

原版需要改善 local HMR，允许重训完整 human head。Movie3R 当前目标更窄：B0 camera 和
Human3R local body details 已经可用，只需 root/orientation 精对齐。第一版若直接改 1792 维
latent并解冻完整 head，会再次出现 V9 的不可解释 gauge 和 body-detail 污染。

## 7. Movie3R 的可实现模块映射

### 7.1 本地已有 hook

| UniCon3R | Movie3R 当前代码 | 可复用方式 |
|---|---|---|
| `H_t [B,N,768]` | `src/dust3r/model.py` 中 fused `smpl_query` / decoder `smpl_token` | 直接作为 per-person human token |
| `F^u_HMR [B,N,1024]` | `smpl_tk_mhmr` | 与 human/contact token 拼接或只给输出 head |
| `F_t` | CUT3R `feat_i` / decoder image tokens | current scene semantic context |
| `S_{t-1}` | `state_feat` | 只读 scene memory；不允许 query 写 state |
| identity momentum | `StreamingHumanMemory`、`smpl_id`、pre human token history | 按 ID 对齐，不能按 row 截断 |
| latent residual | `V8HumanLatentResidualHead` | 可作 latent-control baseline，不作第一部署版 |
| direct translation residual | `V8HumanTranslationCorrectionHead` | 可扩成受限 `Δt+ΔR` sidecar |
| Human head | `downstream_head.deccam/decpose/decshape/decexpression` | 第一阶段完全冻结；第二阶段仅 root 路径 LoRA |

当前 `V8HumanLatentResidualHead` 已实现：

```text
[human token 768, corr token 768, pose token 768]
→ LN + 2-layer GELU context MLP
→ 768-dim delta + scalar gate
→ corrected human token
→ concat Multi-HMR 1024
→ original 1792-dim Human3R head
```

它是接入点，不是新证据。此前 V9 的 corr token 主要是 semantic/alignment/momentum；新
模块必须替换为 person-scene registration evidence。

### 7.2 建议的 `PersonSceneRegistrationPrompt`

每个 post 人 `i` 构造：

```text
h_i       : decoder human token                         [768]
u_curr_i  : CA(h_i, current image/scene tokens)         [768]
u_mem_i   : CA(h_i, B0-aligned read-only scene memory)  [768]
g_i       : explicit local body-scene geometry          [768]
m_i       : ID-aligned pre contact/registration memory  [768]
r_i       : reliability/visibility/scale metadata       [small -> 768]
```

然后复用 UniCon3R gate/fusion：

\[
\gamma_i=\sigma(MLP(h_i\oplus u^{curr}_i\oplus u^{mem}_i\oplus r_i)),
\]

\[
u_i=\gamma_i\odot u^{curr}_i+(1-\gamma_i)\odot u^{mem}_i,
\]

\[
c_i=MLP(h_i\oplus u_i\oplus g_i\oplus m_i\oplus r_i).
\]

这里的 geometry 不能只是 Human3R person pointmap median。优先输入：

1. frozen B0 post camera 下的 person mesh/body-part sample；
2. 有效的独立 scene points、support plane/normal、signed body-to-scene distances；
3. feet/hands/pelvis 各自的 nearest-scene displacement、visibility 和 confidence；
4. DA3 若使用，必须先做整对共享尺度和坐标审计；失败则对应 geometry token 置零并把
   reliability 设为 false；
5. first-post 2D joints/silhouette reprojection 作为语义/训练约束，不把 bbox size 当 depth。

为了贴近论文又适应 shot，推荐 `K=5` typed body queries：pelvis、left/right foot、
left/right hand。每个 part 收集 `[relative_xyz, distance, normal, confidence, visible]`，经过
共享 MLP 得 part tokens，再 attention/concat 融合到 `g_i`。这比一个错误的 7×7 pre-image
crop更符合当前可观测性。

### 7.3 第一版受限输出

建议新建 `RootOrientationResidualHead`：

```text
[h_i, c_i, F_HMR_i, reliability_i]
→ LN
→ Linear/GELU/Linear hidden 768
→ delta_t_raw [3], delta_omega_raw [3], gate_logit [1]
```

参数化：

\[
\Delta t_i =
a_r r_i+a_u e^u_i+a_v e^v_i,
\]

其中 `r_i` 是当前 root viewing ray，`e_u/e_v` 是 camera-local 横向正交基。用 `tanh`
分别限制：

- root-ray：建议初始 cap `0.50 m`；
- lateral：建议初始 cap `0.15 m`；
- root rotation axis-angle：建议初始 cap `15°`。

最终：

\[
t'_i=t_i+g_i\Delta t_i,
\qquad
R'_i=\exp(g_i\Delta\omega_i)R_i.
\]

必须保持：

```text
camera_pose' == camera_pose_B0                 bit-exact
shape' == shape_raw                            bit-exact
expression' == expression_raw                  bit-exact
local body pose excluding root' == raw         bit-exact
pointmaps/conf' == raw                          bit-exact
```

这比原版 UniCon3R 更受限，但正好对应已证实的剩余错误，并且输出可解释、可单元测试。

### 7.4 第二阶段 latent 版本

若 direct head 证明 geometry 有 oracle-free signal，再做更接近原论文的版本：

```text
c_i + Multi-HMR feature
→ V8HumanLatentResidualHead / new 1792-d residual
→ only deccam + root-pose rows of decpose through LoRA
```

禁止第一步就解冻完整 `decpose/decshape/decexpression`。必须用 direct `Δt/ΔR` 作为 teacher
或 auxiliary readout，确认 latent branch 真正在修 root，而不是改变四肢来降低 loss。

## 8. 最小训练阶段

### Stage 0：冻结特征缓存与 oracle 上限

目标：不训练大模型，确认输入包含可预测的 camera-local residual。

1. 固定现有 V9/B0 checkpoint；
2. 对 AvatarReX、THuman、MVHuman 的 AABB/shot pair 跑 raw first-post；
3. 缓存 `h_i, F_HMR_i, current scene features, state, predicted mesh, B0 camera`；
4. GT 计算 teacher target：

   \[
   \Delta t^*=t^{cam}_{GT}-t^{cam}_{raw},
   \quad
   \Delta R^*=R^{cam}_{GT}(R^{cam}_{raw})^{-1};
   \]

5. 另跑固定相机的 Multi-THuMBS-style 显式 optimizer，得到 deployable constraints 下的
   `Δt_opt/ΔR_opt`，用来区分“GT 可修”与“当前 scene evidence 可修”；
6. camera target 和人体 target 分离，禁止 GT root 反向生成 B0。

停止条件：geometry-only oracle/linear probe 在 held-out clip 上不能显著预测 residual 方向，
则不进入大训练，先换 geometry 来源。

### Stage 1：最小 sidecar root/orientation head

只训练：

- `CA_curr/CA_mem`；
- geometry/semantic/momentum/fusion MLP；
- reliability gate；
- `RootOrientationResidualHead`；
- 可选 part-level contact auxiliary head。

冻结全部 Human3R/CUT3R/Multi-HMR、B0、camera/scene/human base heads。先不把 contact token
插入共享 decoder，直接读取 frozen tokens，确保不会污染 camera/state。

建议 loss：

\[
L_{root}=\|t'_i-t^*_{i,cam}\|_1,
\]

\[
L_{orient}=d_{SO(3)}(R'_i,R^*_{i,cam}),
\]

\[
L_{joint}=\frac1J\sum_j\|J'_{ij}-J^*_{ij}\|_1,
\]

\[
L_{repr}=\|\Pi(K,J'_i)-j^{2D}_i\|_1,
\]

\[
L_{res}=\|\Delta t_i\|_1+\eta\|\Delta\omega_i\|_2^2.
\]

再加：

- `L_improve=max(0, error_corrected-error_raw+margin)`；
- AAAA/continuous no-op 与 false-contact fallback loss；
- 有 RICH/pseudo contact 时加入论文 exact focal/part loss；
- 有 Multi-THuMBS optimizer teacher 时加入 `L_distill(Δ_pred,Δ_opt)`。

第一轮训练顺序：

```text
single clip overfit
→ small mixed AvatarReX/THuman/MVHuman
→ three development
→ freeze thresholds/checkpoint
→ dance/box untouched evaluation
```

### Stage 2：UniCon-style decoder-in contact token

只有 Stage 1 有稳定 held-out 收益才做：

- 把 `C_i` 作为 typed token 插入 decoder；
- decoder 权重继续冻结；
- attention mask 只允许 `H↔C`、`C→F/S`，禁止 pose token `z` 和 state update 从 `C`
  接收写入，保证 camera/state 不变；
- 训练 contact prompt、latent residual、root-only human-head LoRA；
- direct `Δt/ΔR` sidecar保留为 auxiliary head和可解释性监控。

### Stage 3：identity-aligned momentum 与在线 gate

最后再加入跨 shot memory：

- 用 B0 后 WHO association 对齐 `C'_{pre,id}`；
- 人物没匹配、场景跨物理空间、geometry 低置信时 momentum 置零；
- gate 失败必须返回 raw B0 human bit-exact；
- 不允许简单按检测 row 使用 UniCon3R 原始 `Align`。

## 9. 与 Multi-THuMBS 显式优化的关系

两者解决的是同一自由度，但计算方式不同：

| 维度 | Multi-THuMBS | UniCon3R | Movie3R 建议 |
|---|---|---|---|
| camera | 边界处一起优化 | camera由统一模型输出，backbone冻结 | B0后彻底冻结 |
| human变量 | root translation + global orientation | latent residual后完整SMPL-X head | 只允许 `Δt_cam+ΔR_root` |
| scene evidence | VGGT person depth、silhouette、2D joints | current/state context + local pointmap geometry + contact | B0坐标中的可靠person/support geometry |
| 推理 | 500 iterations/边界，完整流程非因果 | 一次在线前向，2.4 FPS | 一次 first-post per-person residual |
| 监督 | test-time显式 objective | RICH contact + SMPL-X/scene GT | GT residual/contact或optimizer pseudo-label |

更准确的组合方式不是二选一：

```text
Multi-THuMBS-style fixed-camera optimizer
→ 在训练集产生可解释 Δt_opt/ΔR_opt 和失败标签
→ UniCon3R-style person relation/contact prompt
→ amortize optimizer into one feed-forward residual
→ inference 不再迭代
```

这可以命名为：

```text
Amortized Person-Conditioned Boundary Scene Registration
```

创新点不在“先相机再人体”这个顺序，而在：一个 B0 coarse gauge 后，使用 scene-grounded
contact/registration token，把逐人显式优化蒸馏成一次因果 residual，并严格隔离 camera 与
body-local details。

## 10. 必做 ablation 与接受标准

按 UniCon3R 的因果消融复现：

```text
A0 B0 raw human
A1 semantic human token only                 复现V9上限
A2 + current/state scene context
A3 + explicit metric geometry
A4 + contact/part supervision
A5 + root/orientation residual feedback
A6 + ID-aligned temporal momentum
```

必须报告：

- camera-local root L1、ray-depth error、lateral error；
- root orientation geodesic error；
- world root/joint/vertex error；
- physical grounding：penetration、float、foot sliding；
- improve rate、>5 cm harm、P90/P95；
- gate coverage；
- camera/pointmap/shape/local-pose bit-exact hash；
- near/far、遮挡、多人重叠、有/无可靠 support 分组。

最低接受线：

1. `three` 相对 B0 的 camera-local root 和 orientation 均改善；
2. `dance/box` frozen 后仍改善，且 >5 cm harm 不超过预先冻结阈值；
3. continuous/AAAA 不修时输出 bit-exact；
4. camera 始终 bit-exact；
5. A3 必须明显优于 A1，证明收益来自 explicit person-scene evidence，不是再训练一次 V9；
6. A5 必须优于 Parallel Readout/A4，证明 contact 真正反馈到人体；
7. 无 support/低置信 case 可可靠 fallback，不能强行贴地。

## 11. 最终建议

UniCon3R 可以作为 Movie3R 最终精对齐的学习型主线，但不能直接复制它的 continuous-frame
RoIAlign，也不能回到“同类隐式 token + 完整 human/camera joint correction”。正确落点是：

```text
B0 camera frozen
→ per-person current scene/contact geometry
→ UniCon-style semantic + geometry + identity momentum prompt
→ bounded camera-local root/orientation residual
→ optional root-only latent/LoRA refinement
→ unreliable evidence exact fallback
```

当前第一优先级不是立刻跑 100 epochs，而是 Stage 0/1：证明一个冻结 camera、只读 scene
evidence 的小 head 能在 held-out `three` 上预测 residual 方向。若做不到，说明 geometry 仍不
可观测，应继续找独立 scene evidence；若做到，再把它升级为 decoder-in contact token 和
Multi-THuMBS teacher distillation。
