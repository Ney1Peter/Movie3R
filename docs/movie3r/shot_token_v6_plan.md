# ShotToken V6 Plan: Local Scene Re-Anchor Tokens

## 1. 核心目标

V6 的目标是把现在的 global ShotToken：

```text
一个无空间锚点的全局控制 token
```

改成：

```text
多个有明确位置、有明确匹配关系、有明确监督的 local scene re-anchor tokens
```

原来的 global ShotToken 更像在告诉模型：

```text
镜头跳了，整体相机应该调整。
```

V6 希望改成：

```text
当前帧这些静态背景 patch，应该接回历史帧 / keyframe 里的这些位置。
```

也就是说，ShotToken 不再直接像“全局遥控器”一样控制 camera / world，而是提供局部重定位证据，让 camera / pose token 根据这些证据自己推理对齐关系。

## 2. 为什么要改

之前实验说明，global ShotToken 如果直接进入 decoder full attention，会污染很多分支：

```text
image token
pointmap
background reconstruction
human token
camera pose
recurrent state
```

主要原因是它没有明确空间位置，也没有明确局部对象。它表达的是全局变化，但 decoder 里所有 token 都可能读取它，于是它很容易变成一个不受控的全局控制信号。

Human3R 的 human token 能进入 decoder，是因为它有明确锚点和明确任务：

```text
head patch -> human prompt -> 恢复这个具体的人
```

所以 V6 希望 ShotToken 也变成类似形式：

```text
static background patch / matched patch -> re-anchor prompt -> 帮助 camera/world 对齐
```

## 2.1 当前 V6 的两条主线

经过 V2 / V4 / V5.1 / V5.1-LAST2 的验证，目前 V6 主要分成两条主流路线。

```text
路线一：完善 ShotToken，让它从 global command token 变成 scene-anchored token。
路线二：增加 attention mask，限制 ShotToken / anchor token 只能影响 pose/camera token。
```

路线一解决的是“token 到底表达什么”的问题。

```text
原来的 q_t 只知道当前帧和上一帧整体不一样，
但不知道当前帧哪个背景点应该接回旧世界里的哪个背景点。
```

因此路线一的目标是把 ShotToken 改成多个 local background anchor tokens，每个 token 都来自具体静态背景 patch，并携带局部匹配证据。

路线二解决的是“token 能影响谁”的问题。

```text
anchor token <-> pose token: yes
anchor token <-> image token: no
anchor token <-> human token: no
```

两条路线可以分开验证，也可以最终结合：local anchor 提供可靠几何证据，attention mask 防止这些证据污染 image / human / pointmap。

## 2.2 重新评估：优先引入 XFeat 作为 scene matching prior

基于对 Human3R 真实做法的重新理解，V6 当前主线需要从“自己训练一个简单 MLP descriptor”调整为“优先引入 pretrained scene matching prior”。

Human3R 表面上像是从 CUT3R encoder token 中检测 head patch，再生成 human token。但更准确地说，它不是强迫 CUT3R raw token 自己学会人体检测和人体先验，而是额外引入了经过人体数据训练的 Multi-HMR / ViT-DINO human encoder。

Human3R 的有效组合是：

```text
pretrained Multi-HMR human prior
+ CUT3R 3D scene / camera / metric prior
+ lightweight prompt fusion
```

对应到 V6 的 scene-anchor 问题，raw CUT3R/Human3R encoder token 已经在 MVP 中暴露出局限：

```text
1. 同一相机连续帧匹配正确。
2. AABB 跨镜头 / 大视角变化时，raw encoder token cosine matching 几何错误。
3. 这说明 raw CUT3R token 适合 reconstruction / metric scene / temporal state，
   但不一定天然适合 wide-baseline local feature matching。
```

因此，V6 更合理的类比是：

```text
pretrained XFeat scene matching prior
+ CUT3R / Human3R 3D scene / camera prior
+ lightweight re-anchor prompt / camera-token adapter
```

这里 XFeat 的角色类似 Human3R 中的 Multi-HMR：它不是最终输出模块，而是提供任务相关的强先验。XFeat 负责产生可靠局部背景对应，CUT3R/Human3R 仍负责 online camera/world reconstruction。

当前判断：

```text
1. XFeat inference-time prior 应作为 V6 首选主线。
2. trained MLP descriptor 仍可保留为 ablation，但不应作为第一主方案。
3. XFeat teacher-only 可以作为后续轻量化路线，而不是第一步就承担最终推理能力。
```

这样更容易解释方法贡献：不是重新发明 local matcher，而是把 lightweight pretrained matching prior 转成 scene re-anchor tokens，并以受控方式融入 Human3R/CUT3R 的 camera-token path。

这也不等价于回退到传统 SLAM/SfM pipeline。V6 不做完整建图、不做 bundle adjustment、不用 XFeat 单独输出 camera；XFeat matches 只是 sparse static scene anchors，最终 camera/world 仍由 Human3R/CUT3R 完成。

## 2.3 RICH 几何验证结论（2026/05/12）

RICH `depth/*.npy` 不能直接当跨相机 GT depth 使用。当前 `RICH_4Human3R` 预处理只导出 `rgb/cam/smpl`，`depth/*.npy` 是 Depth Anything 生成的伪深度，不是与 RICH XML camera calibration 严格对齐的 metric GT depth。

因此，RICH 跨相机 scene-anchor 验证应使用官方静态扫描 mesh：

```text
/workspace/data/RICH/scan_calibration/BBQ/scan_camcoord.ply
/workspace/data/RICH/scan_calibration/BBQ/calibration/*.xml
```

正确验证流程：

```text
1. XFeat 只负责从两张 RGB 图产生候选局部匹配。
2. RICH scan mesh + XML calibration 负责提供真实 3D 静态背景几何。
3. 同一个 mesh vertex 分别投影到两个相机，经过 z-buffer 可见性过滤后，形成真实点对点对应。
4. XFeat match 与 mesh 投影对应比较 reprojection error，得到 mesh geometry inlier。
```

重要发现：

```text
BBQ_001_juggle cam03@101 -> cam04@102
XFeat raw matches: 936
Homography RANSAC inliers: 342
Fundamental RANSAC inliers: 709
Mesh geometry inliers: 108
Mesh inliers inside homography: 0 / 108
Mesh inliers inside fundamental: 94 / 108
```

结论：

```text
1. DA3 depth 不能作为 RICH 跨相机 metric GT depth。
2. 同相机连续帧 depth projection sanity 会产生假阳性，因为固定相机下任意 depth 都会近似投回原像素。
3. 跨相机非平面场景不应使用 homography RANSAC 作为最终几何筛选。
4. V6 scene-anchor 应优先使用 XFeat + fundamental/essential geometry + mesh/depth validation。
5. 没有 mesh/GT depth/SfM depth 时，camera calibration 只能给极线约束，不能唯一确定点对点对应。
```

相关脚本：

```text
/workspace/code/accelerated_features/scripts/visualize_rich_mesh_projection.py
/workspace/code/accelerated_features/scripts/visualize_rich_mesh_correspondences.py
/workspace/code/accelerated_features/scripts/test_rich_aabb_xfeat_mesh_geometry.py
```

## 3. Token 形式

V6 不再使用单个：

```text
q_t = ShotGenerator(F_{t-1}, F_t)
```

而是生成 K 个局部 scene anchor tokens：

```text
R_t = {R_1, R_2, ..., R_K}
```

每个 anchor token 表达：

```text
当前帧某个静态背景 patch
对应历史帧 / keyframe / memory 里的某个静态背景 patch
这个匹配有多可靠
```

可以写成：

```text
R_k = Projection(
    F_cur[u_k],
    F_ref[v_k],
    PE_cur[u_k],
    PE_ref[v_k],
    match_confidence_k,
    optional depth / 3D / visibility info
)
```

这里的 `R_k` 不直接携带完整 SE(3) camera correction。它只提供局部匹配证据，例如：

```text
当前帧这个墙角 patch，对应旧帧里的那个墙角。
当前帧这个门框 patch，对应旧世界里的那个门框。
当前帧这块地面 patch，对应历史帧里的那块地面。
```

camera / pose token 再根据多个 anchor evidence 共同推理相机应该如何重新对齐。

## 4. 职责拆分

V6 不让一个 token 同时负责所有事情，而是拆成三类模块。

```text
1. Global transition gate
2. Local scene anchor tokens
3. Camera / pose token
```

Global transition gate 只负责：

```text
是否发生 shot change
当前是否 low-overlap
是否需要启用 correction
correction gate 应该多大
```

Local scene anchor tokens 负责：

```text
当前 patch 对应哪个历史 patch
这个 patch 是否是静态背景
这个匹配是否可靠
当前帧和参考帧是否有有效 overlap
```

Camera / pose token 负责：

```text
汇总 anchor evidence
推理 refined camera pose
完成 world re-anchor
```

这样可以避免一个 global token 同时承担“判断跳变、表达跳变、修 pose、更新 world、更新 state”等过多职责。

## 5. V6.0 安全版本

V6.0 建议先不要让 anchor tokens 进入主 decoder sequence。

采用更安全的 pose-only adapter：

```text
z_l = z_l + gamma_l * g_t * Adapter(z_l, R_t)
```

其中：

```text
z_l: 第 l 层 decoder 后的 pose token
g_t: global transition gate
R_t: local scene anchor tokens
```

这个版本的约束是：

```text
scene anchor tokens 只影响 pose / camera token
image token 不直接 attend anchor tokens
human token 不直接 attend anchor tokens
pointmap head 不直接吃 anchor tokens
```

这样最接近当前 V5.1 的结构，也最容易验证 anchor evidence 是否真的帮助 camera alignment，同时最大程度保护 pointmap / background / human branch。

## 6. V6.1 冻结背景 Encoder

Human3R 额外引入了冻结的 Multi-HMR encoder 来提取人体相关 token。V6 也可以类似引入一个冻结的背景 / 匹配 encoder，用来提供更可靠的静态场景 anchor。

候选方向包括：

```text
DINOv2 / MAE 类通用视觉特征
SAM / semantic segmentation 类背景区域提取
SuperPoint + LightGlue 类局部匹配
DUSt3R / MASt3R 类几何匹配特征
Depth Anything / monocular depth 只作弱辅助过滤，不能当跨相机 metric GT
```

这个 encoder 初期可以冻结，不参与训练，只作为 anchor proposal 和 matching 的辅助来源。

它的作用是：

```text
帮助选择可靠背景 patch
帮助排除人体和动态区域
帮助判断 patch match 是否可信
提供比主模型 image token 更稳定的局部匹配特征
```

这和 Human3R 使用专门人体 encoder 的思路类似：主模型不一定自己承担所有感知任务，可以借助冻结专家模型产生更稳定的 prompt / anchor。

## 6.1 轻量内部 Anchor Detector 优先方案

在引入 XFeat / LightGlue / SuperPoint 这类外部匹配器之前，建议先尝试一个完全基于 Human3R/CUT3R 内部特征的 lightweight anchor detector。

核心原因是：如果内部 encoder token 已经能找到可靠背景对应关系，那么最终方法会更轻、更统一，也更符合 Human3R one-stage 风格。

### 6.1.1 当前代码里可复用的信息

当前 Human3R 代码里已经有不少可以复用的 tensor。

| 信息 | 是否可用 | 位置 |
|---|---|---|
| 当前帧 encoder tokens `F_cur` | 可用 | `src/dust3r/model.py::_encode_image()` / `_encode_views_mhmr()` |
| 参考帧 encoder tokens `F_ref` | 可用，需要缓存 | forward 中已有每帧 `feat[i]` |
| patch 2D 位置 `pos` | 可用 | `_encode_image()` 返回 `pos` |
| pose/camera token `z_l` | 可用 | `_decoder()` 中 `f_img[:, 0:1]` |
| last2 / last4 decoder 插入点 | 可用 | `_apply_layerwise_pose_shot()` |
| GT camera pose | 可用 | `AvatarReX` 返回 `camera_pose` |
| GT depth | 可用 | `AvatarReX` 返回 `depthmap` |
| GT / dataset human mask | 可用 | `AvatarReX` 返回 `msk` |
| 模型预测 pointmap confidence | 可用，但在 head 后 | `pred["conf_self"]` / `pred["conf"]` |

需要注意的是，`conf_self` 和 `conf` 是 downstream head 输出后的结果，不能直接用于 decoder 前的 anchor selection；第一版可以先用 GT depth valid mask、human mask 和图像梯度过滤。

### 6.1.2 Anchor Detector 设计

可以增加一个轻量 MLP：

```text
a_i = sigmoid(MLP_anchor(F_cur[i]))
```

这里的 `a_i` 不是简单表示“是不是背景”，而是表示这个 patch 是否适合作为 static scene anchor。

好的 anchor 应该满足：

```text
静态背景
不在人身上
有纹理或边缘
depth / pointmap 可靠
能在 reference frame 中找到对应 patch
对 camera relocalization 有帮助
```

第一版 final score 可以写成：

```text
final_anchor_score_i =
    anchor_score_i
  * (1 - human_mask_i)
  * depth_valid_i
  * optional texture_or_gradient_score_i
```

其中 human mask 可以先用 AvatarReX 的 `msk`，再下采样到 patch grid；texture / gradient score 可以先不加，或者只用简单 Sobel/边缘强度做过滤。

MLP 输入维度建议使用 encoder token 维度或 decoder token 维度，输出 1 个 anchor score；初始化时最后一层 bias 可以偏负，让训练初期 anchor 数量少一些，更接近 no-op。

### 6.1.3 内部 Patch Matching 设计

不使用外部 matcher 时，第一版可以直接做 encoder token cosine matching。

流程如下：

```text
1. 当前帧根据 final_anchor_score_cur 选 top-K 候选 U。
2. 参考帧根据 final_anchor_score_ref 选 top-K 候选 V。
3. 对 F_cur[U] 和 F_ref[V] 做 L2 normalize。
4. 计算 S = f_cur @ f_ref.T。
5. 用 mutual nearest neighbor 或 top similarity 选匹配。
6. 过滤 similarity 低的 pair。
7. 得到 anchor pairs: (u_k, v_k, sim_k)。
```

建议第一版 `K=16` 或 `K=32`，不要一开始用太多 anchor，因为错误 anchor 比 anchor 少更危险。

这种方法的优势是实现轻、推理快、和现有模型特征空间一致；风险是 encoder token 是 patch-level，不一定有 XFeat / SuperPoint 那种 keypoint-level 精度。

### 6.1.4 Anchor Token 构造

每个 anchor token 不直接预测完整 SE(3) camera correction，只提供局部匹配证据。

建议构造为：

```text
R_k = Projection(
    F_cur[u_k],
    F_ref[v_k],
    PE_cur[u_k],
    PE_ref[v_k],
    sim_k,
    optional depth / confidence / visibility
)
```

Projection 输出维度建议为 `768`，和 decoder pose token 维度一致。

为了防止污染，建议加：

```text
LayerNorm
learnable type embedding
match confidence gate
learnable scale gamma，初始化为 0 或很小
```

这样 anchor token 初始接近 no-op，不会一开始就破坏原模型。

### 6.1.5 接入方式

第一版不要让 anchor tokens 进入主 decoder full attention。

更安全的方式是只让 pose token 读 anchor tokens：

```text
for selected decoder layers l:
    delta_z_l = CrossAttn(
        query = z_l,
        key/value = R_t
    )
    z_l = z_l + gamma_l * g_t * delta_z_l
```

约束是：

```text
只在 last2 或 last4 decoder layers 加 adapter
只更新 pose/camera token
不修改 image tokens
不修改 human tokens
不修改 world head / human head
```

当前 V5.1-LAST2 的结构已经证明，减少层数可以保护 background / pointmap；V6 可以复用这个安全插入点，只是把 global q_t 换成多个 local anchor tokens。

### 6.1.6 Anchor 伪标签生成

AvatarReX AABB 数据很适合生成 anchor supervision，因为它有同一 world coordinate 下的 GT camera pose、depth 和 human mask。

对 B1 的 patch `u` 可以这样生成标签：

```text
1. 如果 u 在 human mask 内，标为 invalid。
2. 如果 depth 无效，标为 invalid。
3. 用 B1 的 GT depth 把 u 反投影到 3D world。
4. 把这个 world point 投影到参考帧 A2。
5. 如果投影点 v 不在图像内，标为 no-overlap。
6. 如果 v 在 human mask 内，标为 invalid。
7. 检查 A2 depth consistency。
8. 如果一致，则 u 是 positive anchor，v 是 GT match。
```

可以得到：

```text
anchor_label[u]
match_label[u] = v
inlier_label[u, v]
overlap / visibility label
```

这里最容易出 bug 的地方是 crop/resize 后的 pixel 坐标、depth、mask 和 intrinsics 是否完全对齐。

### 6.1.7 推荐 Losses

第一阶段建议只训练 anchor 是否可靠，不接入 decoder。

优先 loss：

```text
Anchor Detection BCE
Match CE / InfoNCE
Inlier BCE
```

第二阶段再接入 pose-only adapter，并使用：

```text
Jump Boundary Pose Loss
Post-jump Anchor Pose Loss
AAAA no-op / identity loss
```

更重的 reconstruction preservation loss 和 background pointmap loss 可以后加，因为它们可能压住 camera correction。

### 6.1.8 推荐训练流程

建议分三步做。

```text
Stage 0: 离线验证内部 token matching
Stage 1: 只训练 MLP_anchor + matching / inlier heads
Stage 2: 接入 pose-only anchor adapter，冻结 Human3R 主体
```

Stage 0 最重要：先不训练模型，只取 A5B5 的 A2-B1，提取 encoder tokens，做 top-K matching，可视化匹配线，并用 GT camera/depth 检查匹配是否正确。

如果 Stage 0 都找不到可靠 anchor，就说明内部 token matching 不够，需要外部 matcher 做 teacher 或 upper-bound。

### 6.1.9 XFeat / LightGlue 作为备选

旧判断：XFeat / LightGlue 不建议第一步就作为主方法，但很适合作为备选。

最新补充：上面这句话是早期“先验证内部 token matching”的保守排序。经过内部 MVP 结果和 Human3R 强先验路线的重新评估后，当前 V6 主线调整为：优先使用 XFeat 作为 inference-time lightweight scene matching prior；内部 MLP descriptor 和 teacher-only XFeat 作为 ablation / 后续轻量化路线保留。

原因是：Human3R 并没有只靠 CUT3R raw token 和简单 MLP 学出人体先验，而是引入 Multi-HMR human prior。V6 的 scene-anchor 也不应强迫 raw CUT3R token 从零学 wide-baseline matching，尤其当前 raw token cosine matching 在 AABB / cross-camera 场景中已经出现 0 几何 inlier。

当前两种 XFeat 用法的定位：

```text
方案 A: inference-time XFeat
    作为正式方法的一部分，优先验证是否能解决 abrupt transition。
    XFeat 输出 sparse / semi-dense static scene matches，后续转成 re-anchor tokens。

方案 B: teacher-only XFeat
    训练阶段生成 pseudo matches / anchor labels。
    推理阶段由轻量 MLP_anchor / MLP_desc 复现匹配行为。
    作为轻量化 ablation 或后续优化，不作为第一阶段主路线。
```

论文表述上，XFeat 应被定位为 lightweight scene matching prior，而不是传统 SfM/SLAM 子系统。

备选一是 teacher-only：

```text
训练阶段用 XFeat / LightGlue 生成高质量 pseudo labels。
推理阶段不用外部 matcher，仍然使用内部 MLP_anchor + token matching。
```

这个方案能保持最终模型轻量；但在当前新路线中，它更适合作为后续轻量化 ablation，而不是第一阶段主方法。

备选二是 inference-time matcher baseline：

```text
推理阶段也用 XFeat / LightGlue 生成 anchor pairs。
```

这个方案可以作为当前优先主线，也可以在 ablation 中作为 upper-bound / debug baseline；代价是引入额外依赖和推理开销，需要单独报告 runtime。

当前更新后的顺序是：先用 inference-time XFeat 验证 scene-anchor 是否有效；如果有效，再尝试 teacher-only / MLP descriptor 蒸馏来降低推理依赖。

### 6.1.10 最小 MVP

最小可行版本建议是：

```text
1. 从当前帧和参考帧拿 encoder tokens。
2. 用 human mask / depth valid 过滤无效 patch。
3. top-K 选背景候选 patch。
4. cosine similarity + mutual nearest neighbor 做 patch matching。
5. 生成 K 个 anchor pairs。
6. Projection MLP 得到 anchor tokens。
7. last2 decoder layers 中 pose token cross-attend anchor tokens。
8. 只训练 anchor projection + pose adapter，主模型冻结。
```

这个 MVP 的目标不是马上达到最好效果，而是先验证：

```text
局部 scene anchor evidence 是否比 global q_t 更可靠。
```

## 7. V6.2 Masked Decoder

如果 V6.0 / V6.1 证明 anchor tokens 有效，再考虑让 anchor tokens 进入 decoder。

但必须加 attention mask：

```text
anchor token <-> pose token: yes
anchor token <-> image token: no
anchor token <-> human token: no
```

否则即使 token 是 local anchor，也仍然可能污染 reconstruction。

V6.2 的目标是同时做到两点：

```text
token 语义更精准：local scene evidence
token 权限更受控：只服务 camera / pose branch
```

这会比 V6.0 工程量更大，因为当前 decoder attention 还不支持 mask，需要改底层 attention 调用链。

## 8. 主要困难

### 8.1 Anchor 选什么

这是 V6 最大难点。

理想 anchor 应该是：

```text
静态背景
纹理足够明显
不在人体 mask 内
不在动态物体区域
不是天空 / 纯色墙 / 低纹理地面
在当前帧和历史帧都有可见对应
match confidence 高
```

错误 anchor 会很危险，因为它会给 camera token 提供错误几何证据。

### 8.2 Shot Change 未必有 Overlap

电影镜头切换时可能出现：

```text
A2 和 B1 几乎没有重叠背景
相机视角完全不同
背景遮挡严重
```

这种情况下不能强行 re-anchor，否则会把 camera pose 拉错。

因此 V6 需要 overlap / validity 判断：

```text
有可靠 overlap -> 启用 anchor correction
没有可靠 overlap -> gate 变小或关闭
```

### 8.3 Mask 不一定可靠

如果人体区域或动态区域没有排干净，anchor 可能选到人身上。

这会造成：

```text
把动态人体当作静态世界点
camera token 根据错误点对齐
pose 更乱
```

所以需要可靠的人体 mask、动态区域过滤，或者 anchor inlier 判断。

### 8.4 匹配监督不直接可用

即使数据集有 GT camera pose，也不一定直接有 patch-level correspondence 标签。

实现时需要考虑：

```text
patch-level GT correspondence 不一定直接可用
depth / pointmap 可能有噪声
不同相机视角下同一背景 patch 不一定容易匹配
low-overlap 时可能根本没有可靠匹配
```

初期可以用 weak supervision 和几何一致性约束辅助训练。

### 8.5 工程成本更高

V6 比 V5.1 大很多，涉及：

```text
anchor proposal
anchor matching
frozen encoder 接入
anchor token projection
pose adapter
可能的 attention mask
新的监督和日志
```

因此建议分阶段做，不要一次性全部加上。

## 9. 推荐路线

建议按下面顺序推进：

基于最新评估，当前更推荐的 V6 主路线是：

```text
V6-A: XFeat inference-time scene matching prior
    用 XFeat 直接产生跨镜头背景匹配，验证 scene anchor 是否能修 abrupt transition。

V6-B: XFeat matches -> scene re-anchor tokens -> camera-token adapter
    不做 PnP 后处理，不做 BA，只把 verified matches 转成 tokens，受控影响 pose token。

V6-C: teacher-only XFeat / MLP descriptor ablation
    如果 V6-A/B 有明显收益，再研究是否能把 XFeat 蒸馏成内部轻量 descriptor。
```

旧的内部 token matching 路线仍然保留作为必要 ablation：

```text
V6.0: local anchor tokens + pose-only adapter
V6.1: frozen background / matching encoder 辅助 anchor 选择
V6.2: masked decoder with anchor prompts
```

这样可以逐步回答三个问题：

```text
局部 anchor evidence 是否比 global ShotToken 更有用？
冻结专家 encoder 是否能提高 anchor 可靠性？
anchor tokens 是否可以安全进入 decoder attention？
```

## 10. 最终目标

V6 的最终目标是把 ShotToken 从 global command token：

```text
镜头跳了，整体应该这样变。
```

改成 local scene evidence tokens：

```text
当前帧这些静态背景 patch，对应旧世界里的这些位置。
```

这样做的好处是：

```text
1. token 有明确来源和空间锚点。
2. token 有明确任务和监督。
3. camera correction 来自局部证据，而不是全局命令。
4. image / human / pointmap 更不容易被污染。
5. 方法更容易解释，也更适合作为论文方法。
```
