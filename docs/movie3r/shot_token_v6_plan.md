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

## 2.4 最新设计：XFeat/mesh 作为 patch correspondence teacher

最新判断是：V6 不应只把 XFeat 当成 inference-time matcher，也可以把它作为监督信号，教 CUT3R/Human3R encoder patch token 学会可匹配的 scene-anchor descriptor。

这和 Human3R 的思路更一致：

```text
Human3R:
    CUT3R encoder token
  + Multi-HMR pretrained human prior
  + lightweight human prompt fusion

Movie3R V6:
    CUT3R encoder patch token
  + XFeat + mesh-verified static scene correspondence teacher
  + lightweight scene-anchor descriptor / prompt fusion
```

这里要区分两件事：

```text
1. XFeat 不能直接对 encoder patch token 做特征提取。
   XFeat 的输入是 RGB image，输出是 2D keypoint / descriptor matches。

2. XFeat + RICH mesh 可以生成高可信 patch correspondence pseudo labels。
   这些 labels 用来监督 encoder patch token 上的小 descriptor head。
```

建议流程：

```text
1. RGB pair -> XFeat raw matches。
2. RICH scan mesh + XML calibration -> 验证哪些 XFeat matches 是真实 static background matches。
3. 把通过验证的 2D match 映射到 encoder patch grid。
4. 得到 patch-level positive pairs: (patch_A_i, patch_B_j)。
5. 用这些 positive pairs 监督 encoder token descriptor head。
6. 推理时可以逐步从 XFeat inference-time 过渡到 teacher-only / internal descriptor matching。
```

descriptor head 的最小形式：

```text
d_i = normalize(MLP_desc(F_i))
S_ij = d_i @ d_j
```

监督目标：

```text
1. mesh-verified patch pairs 的 similarity 高。
2. random / wrong / human / low-overlap pairs 的 similarity 低。
3. mutual nearest neighbor 后的 match 能复现 XFeat+mesh teacher 的高可信 anchor。
```

因此，XFeat + mesh 的角色是：

```text
teacher / pseudo-label generator / validation source
```

encoder patch token 的角色是：

```text
student descriptor source / eventual inference-time anchor source
```

### 2.4.1 Anchor 不是 offset，offset 来自 matched positions

需要明确：patch anchor 本身不是 camera/world 偏移。anchor 只说明：

```text
frame A 的 patch_i 和 frame B 的 patch_j 是同一个静态背景点 / 区域。
```

变化应该由 matched patch 的位置差和几何差表示。

最小 2D 表示：

```text
delta_uv_k = pos_B[j] - pos_A[i]
```

这个量可以看作 sparse patch flow。它能告诉模型同一个静态背景 anchor 在图像平面上移动了多少，但它不是唯一的真实 camera offset，因为 2D 位移同时受 rotation、translation、depth、parallax 和 intrinsics 影响。

建议第一版 anchor token 包含：

```text
R_k = MLP_anchor(
    F_A[i],
    F_B[j],
    F_B[j] - F_A[i],
    pos_A[i],
    pos_B[j],
    delta_uv_k,
    match_confidence_k,
    optional overlap / visibility / background score
)
```

其中 `delta_uv_k` 是最直接的“变化”信号，`F_B[j] - F_A[i]` 只是辅助特征差，不应被解释成物理 offset。

### 2.4.2 3D / pose-level offset 的后续版本

如果 decoder 已经有 pointmap / depth / 3D prediction，可以把 2D patch anchor 升级成 3D anchor：

```text
X_A[i] <-> X_B[j]
```

然后用多个 3D anchors 拟合一个刚体变换：

```text
T_A_to_B = argmin_T Σ_k || T * X_A[i_k] - X_B[j_k] ||
delta_pose_token = MLP(log_SE3(T_A_to_B))
```

这个 `delta_SE3` / `delta_pose_token` 才更接近 camera/world offset。它可以作为更强的 pose prior 输入到：

```text
PoseResidualAdapter
WorldResidualAdapter
StateGate / Global transition gate
```

推荐分阶段：

```text
Stage A: 只用 XFeat+mesh teacher 验证真实 patch correspondence 数量。
Stage B: 用 teacher 监督 encoder patch descriptor head。
Stage C: 用 internal descriptor matching 生成 patch anchor tokens，先只携带 delta_uv。
Stage D: 如果 C 有效，再基于 predicted pointmap / depth 拟合 delta_SE3。
Stage E: 把 anchor evidence 以 pose-only adapter 方式注入 camera / pose token。
```

关键实验：

```text
mesh-verified XFeat anchors 对应的 encoder patch token similarity
是否显著高于 random negatives / wrong matches。
```

如果正负分不开，说明 raw encoder token 不适合作为 descriptor，需要训练 `MLP_desc`；如果能分开，则可以直接做 internal patch matching ablation。

### 2.4.3 Step1 结论：外部 anchor 可以映射回 encoder patch token

截至 2026/05/12，已完成第一阶段 AABB 验证。验证目标不是修改 encoder 或 decoder，而是确认：

```text
XFeat semi-dense + RICH mesh verified anchors
是否能在 Human3R encoder output patch token 中被重新找到。
```

实验流程：

```text
1. 使用 /workspace/data/RICH/RICH_4Human3R/Training 读取 AABB 样本。
2. 对 [A@t, A@t+1, B@t+2, B@t+3] 分别评估：
   A@t   -> A@t+1     contiguous reference pair
   A@t+1 -> B@t+2     shot boundary pair
   B@t+2 -> B@t+3     contiguous current pair
3. 对每个 pair 使用 XFeat semi-dense 产生 2D matches。
4. 使用 RICH official static scan mesh + XML calibration 过滤真实 static background anchors。
5. 把通过 mesh 验证的 2D anchors 映射到 Human3R encoder patch grid。
6. 比较 positive anchor patch token cosine 与 random / shuffled negative patch token cosine。
```

关键结果：

```text
BBQ_001_guitar cam06/cam07 start=244 boundary:
    mesh anchors: 77
    unique patch anchors: 41
    positive encoder cosine mean: 0.594
    random negative cosine mean: 0.249
    true match rank median: 4
    positive > random: 92.7%

BBQ_001_juggle cam02/cam01 start=197 boundary:
    mesh anchors: 490
    unique patch anchors: 179
    positive encoder cosine mean: 0.750
    random negative cosine mean: 0.282
    true match rank median: 3
    positive > random: 97.8%

BBQ_001_guitar cam01/cam03 start=5 boundary:
    mesh anchors: 9
    unique patch anchors: 7
    positive encoder cosine mean: 0.486
    random negative cosine mean: 0.315
    true match rank median: 38
    positive > random: 85.7%
```

结论：

```text
1. 外部 XFeat/mesh anchors 不是噪声。
2. Human3R encoder patch token 中已经保留了可用的跨视角静态背景对应信息。
3. 当前阶段不需要修改 encoder，也不需要让 anchor 进入 decoder 主体。
4. anchor 数多时，encoder token 对应关系非常稳定；anchor 数少时仍有信号，但 rank 和 correction 稳定性下降。
5. 因此 anchor 不是 data type classifier，而是 shot boundary 上的 geometric evidence。
```

当前推荐 gate：

```text
unique_anchor_patch_pairs >= 16: 启用 anchor evidence
8 <= unique_anchor_patch_pairs < 16: 降权启用 / 弱启用
unique_anchor_patch_pairs < 8: fallback，不强行 re-anchor
```

相关脚本：

```text
/workspace/code/Movie3R/scripts/verify_rich_anchor_encoder_similarity.py
/workspace/code/Movie3R/scripts/verify_rich_aabb_anchor_step1.py
```

相关输出：

```text
/workspace/code/Movie3R/output/rich_anchor_encoder_step1/
/workspace/code/Movie3R/output/rich_aabb_anchor_step1/
```

### 2.4.3.1 Step1 通俗解释：为什么说 anchor 能映射回 encoder patch token

这一阶段验证的不是“只靠 encoder token 自动找出所有 anchors”。验证的是：

```text
如果外部 XFeat/mesh 已经告诉我们真实对应位置，
这些对应位置在 Human3R encoder 输出的 patch token 中是否仍然保留对应关系信号。
```

更具体地说，Human3R 输入图像后会先被切成 patch，然后经过 encoder 多层 self-attention：

```text
image -> patch embedding -> encoder attention -> contextual patch tokens
```

所以 encoder 输出的每个 token 已经不再是原始小图块，而是带有上下文信息的 patch 表示。我们关心的是：

```text
原图里由 XFeat/mesh 找到的同一个真实静态背景点，
经过 resize / crop / patch mapping 后落到 patch_i 和 patch_j，
那么 encoder output 里的 F_A[i] 和 F_B[j] 是否仍然相似？
```

实验步骤可以理解为：

```text
1. XFeat/mesh 在原图上说：
   图 A 的这个位置 = 图 B 的那个位置。

2. 我们把这两个位置换算到 Human3R encoder patch grid：
   图 A 的位置 -> patch_i
   图 B 的位置 -> patch_j

3. 再看 encoder 输出：
   F_A[i] 和 F_B[j] 是否比随机 patch 更相似。
```

结果显示：

```text
F_A[i] 和 F_B[j] 明显比 random negative 更相似。
```

因此可以说：

```text
图片即使经过 patch 切分、token 化、encoder 全局交互，
外部 anchor 对应的 patch token 仍然携带可用的对应特征。
```

但需要明确边界：

```text
已经证明：
    外部 anchor 可以落到 encoder patch token 上，
    且这些 token 中仍保留对应关系信号。

尚未证明：
    不用 XFeat，只靠 Human3R encoder token 自己就能稳定找出所有 anchors。
```

所以当前分工仍然是：

```text
XFeat/mesh: 负责找真实位置关系。
Human3R encoder token: 负责提供模型内部可用的 patch 表示。
AnchorTokenGenerator: 负责把“外部位置关系 + 内部 patch token”合成 local AnchorToken。
```

### 2.4.4 下一步：anchor correction proxy

Step1 只证明了“anchor 能找到”。下一步要证明的是：

```text
这些 anchors 是否能提供可用的 correction 信息。
```

最小验证不直接训练模型，而是做 post-encoder proxy：

```text
1. 从 boundary anchors 得到 patch-level correspondences: pos_A[i] -> pos_B[j]。
2. 用少量 anchors 拟合简单 2D correction：translation / affine。
3. 把 correction 应用到 held-out anchors 的 ref patch positions。
4. 比较 no-correction error 与 corrected error。
5. 用 4 / 8 / 16 / 32 个 anchors 重复抽样，观察 correction 是否稳定。
```

这个实验回答的是用户关心的问题：

```text
一帧内是否只需要少量 anchor，就能给其他 patch 提供大致修正方向。
```

注意，2D translation / affine 仍只是 proxy。它不是最终 camera correction，也不是 SE(3)。如果 proxy 有效，才进入下一阶段：把 anchor evidence 变成 pose-only adapter 的输入。

### 2.4.5 Correction proxy 初步结果

已新增脚本：

```text
/workspace/code/Movie3R/scripts/analyze_rich_aabb_anchor_correction.py
```

输入是 `verify_rich_aabb_anchor_step1.py` 的输出目录。该脚本不重复跑 XFeat / mesh / Human3R，只读取已保存的 boundary anchors，并测试：

```text
1. no correction: 直接把 ref patch 位置当作 cur patch 位置。
2. translation correction: 用 anchors 拟合一个全局平均 2D 位移。
3. affine correction: 用 anchors 拟合一个 2D affine patch transform。
4. 4 / 8 / 16 / 32 anchors 抽样，评估 held-out anchors 的 patch error。
```

初步结果：

```text
BBQ_001_guitar cam06/cam07 start=244 boundary, 41 patch anchors:
    no correction median error: 3.16 patches
    translation median error: 4.32 patches
    affine median error: 1.04 patches
    8-anchor affine held-out median: 1.32 patches
    16-anchor affine held-out median: 1.23 patches

BBQ_001_juggle cam02/cam01 start=197 boundary, 179 patch anchors:
    no correction median error: 3.16 patches
    translation median error: 1.04 patches
    affine median error: 0.76 patches
    8-anchor affine held-out median: 0.81 patches
    16-anchor affine held-out median: 0.77 patches

BBQ_001_guitar cam01/cam03 start=5 boundary, 7 patch anchors:
    no correction median error: 10.03 patches
    translation median error: 2.47 patches
    affine median error: 0.48 patches
    4-anchor affine held-out median: 0.89 patches
```

解释：

```text
1. anchors 确实携带 correction 信息：no correction error 可以被明显降低。
2. 简单平均 translation 不总可靠。guitar cam06->cam07 中 translation 反而更差，说明不同区域存在视角变化 / parallax / crop geometry，不能假设所有 patch 用同一个平移。
3. affine correction 在 3 个样本中都明显优于 no correction，说明 anchors 更适合先聚合成轻量 2D geometric evidence，而不是直接平均 offset。
4. 8-16 个可靠 anchors 通常已经能给出稳定 correction proxy；少于 8 个 anchors 可以工作，但应强 gate / 降权。
```

当前设计更新：

```text
不要把 anchor 简化成 mean(delta_uv)。
更合理的是把 anchors 聚合成：
    anchor_count
    confidence statistics
    weighted delta statistics
    affine / local transform parameters
    residual error statistics
再由 pose-only adapter 或 correction head 使用这些 evidence。
```

相关输出：

```text
/workspace/code/Movie3R/output/rich_aabb_anchor_correction_proxy/
```

### 2.4.5.1 Affine correction 通俗解释

这一阶段的目的不是单纯找匹配点，而是验证：

```text
已经找到的 anchors 能不能提供“怎么纠正 shot boundary 偏移”的信息。
```

最简单的想法是平均平移 translation：

```text
delta_mean = mean(pos_B[j] - pos_A[i])
```

也就是假设：

```text
整张图所有 patch 都移动同一个 dx, dy。
```

这个假设在跨镜头 / 跨相机时经常不成立。原因是不同位置的背景点会因为视角变化、深度差、parallax、crop 几何差异而产生不同方向和大小的位移。例如：

```text
左边墙角可能向右移动 1 个 patch；
右边门框可能向右移动 5 个 patch；
上方背景和下方地面还可能有不同的竖直变化。
```

如果强行取平均，就会变成：

```text
所有 patch 都向右移动 3 个 patch。
```

这对很多区域反而是错的。因此在 `guitar cam06->cam07` 样本中，translation 比 no-correction 还差。

Affine 的作用是用 anchors 拟合一个更灵活的 2D 变换：

```text
x_B = a * x_A + b * y_A + tx
y_B = c * x_A + d * y_A + ty
```

它比 translation 多表达了：

```text
平移 translation
缩放 scale
旋转 rotation
倾斜 / 剪切 shear
```

通俗理解：

```text
translation 问的是：整张图平均移动了多少？
affine 问的是：整张图大概发生了什么二维几何变化？
```

在实验里，我们用 mesh-verified anchors 作为已知对应点，拟合一个 affine，使得：

```text
Affine(pos_A[i]) 尽量接近 pos_B[j]
```

也就是找一个最合适的二维整体变换，让参考帧 A 的 anchor 位置尽可能对齐到当前帧 B 的 anchor 位置。

然后我们用 held-out anchors 检查它是否真的能预测未参与拟合的对应位置。如果 affine 后误差显著下降，说明：

```text
anchors 不只是匹配点，确实携带可用的 correction 信息。
```

重要结论：

```text
1. 不应该把 anchors 简化成 mean(delta_uv)。
2. affine 可以作为 coarse re-anchor prior。
3. affine 仍然不是最终 camera pose / SE(3)，只是 2D proxy。
4. 后续 AnchorToken residual 是在 affine 粗对齐基础上继续补局部误差。
```

### 2.4.6 Anchor evidence vector 与 reference lookup 验证

下一步已把 correction proxy 转成更接近模型输入的固定维度 evidence。

新增脚本：

```text
/workspace/code/Movie3R/scripts/build_rich_anchor_evidence.py
```

该脚本读取 AABB Step1 输出中的 boundary anchors，然后重新跑原版 Human3R encoder，仅用于验证：

```text
当前帧某个 patch，能否通过 anchor affine evidence 找回参考帧中应该读取的 patch。
```

比较对象：

```text
same_position:
    不做 correction，current patch 直接找 reference 中相同归一化位置。

translation:
    用 anchors 的平均位移反推 reference patch。

affine:
    用 anchors 拟合 affine，再反推 reference patch。

oracle_anchor:
    mesh-verified anchor 的真实 reference patch，作为上限参考。
```

初步结果：

```text
BBQ_001_guitar cam06/cam07 start=244, 41 patch anchors:
    same_position reference lookup error: 3.16 patches
    translation reference lookup error: 4.47 patches
    affine reference lookup error: 1.00 patch
    oracle anchor cosine median: 0.642
    affine lookup cosine median: 0.514
    quality_gate: 0.74

BBQ_001_juggle cam02/cam01 start=197, 179 patch anchors:
    same_position reference lookup error: 3.16 patches
    translation reference lookup error: 1.00 patch
    affine reference lookup error: 1.00 patch
    oracle anchor cosine median: 0.782
    affine lookup cosine median: 0.823
    quality_gate: 0.81

BBQ_001_guitar cam01/cam03 start=5, 7 patch anchors:
    same_position reference lookup error: 10.05 patches
    translation reference lookup error: 2.24 patches
    affine reference lookup error: 1.00 patch
    oracle anchor cosine median: 0.508
    affine lookup cosine median: 0.517
    quality_gate: 0.22
```

这说明：

```text
1. anchor evidence 不只是能解释已有 anchors，还能给 current patch 提供 reference lookup prior。
2. affine evidence 明显优于 same-position，也比纯 translation 更稳。
3. 弱样本虽然 affine 几何 lookup 看起来很好，但 anchors 太少，因此 quality_gate 应该低，不能强行信任。
4. 这个 evidence 可以作为后续 pose-only adapter / correction head 的输入，而不是直接进入 encoder 或 decoder 主序列。
```

当前 evidence vector 维度为 24，包含：

```text
count / log_count / quality_gate
anchor cosine statistics
mesh reprojection error statistics
visible overlap statistics
translation dx, dy
affine residual parameters: a-1, b, tx, c, d-1, ty
no-correction / translation / affine residual error statistics
```

输出：

```text
/workspace/code/Movie3R/output/rich_anchor_evidence/
```

重点文件：

```text
anchor_evidence_summary.jpg
*/anchor_lookup_comparison.jpg
*/lookup_error_chart.jpg
*/affine_correction_field.jpg
*/anchor_evidence_vector.npy
*/anchor_evidence_summary.json
```

### 2.4.7 AnchorTokenGenerator 原型验证

当前目标已从“做 correction head”澄清为：

```text
把外部 anchors 转成更准确、更有几何含义的 local ShotToken / AnchorToken。
```

新增独立原型脚本：

```text
/workspace/code/Movie3R/scripts/prototype_rich_anchor_tokens.py
```

该脚本仍不修改 encoder / decoder / 模型主路径。它从 AABB boundary anchors 和 Human3R encoder tokens 构造结构化 AnchorTokens：

```text
AnchorToken_k = {
    key_cur_feature: F_cur[j],
    value_ref_feature: F_ref[i],
    ref_pos_norm: pos_ref[i],
    cur_pos_norm: pos_cur[j],
    delta_uv_norm: pos_cur[j] - pos_ref[i],
    confidence: mesh/token confidence,
    mesh_error_px,
    encoder_cosine
}
```

验证方式是 leave-one-out：

```text
1. 每次拿掉一个真实 anchor，当作 held-out target。
2. 用剩余 anchors 构造 AnchorTokens。
3. 让 held-out current patch 通过 AnchorTokens 预测 reference patch。
4. 和真实 mesh-verified reference patch 比较 patch error。
```

比较方法：

```text
same_position:
    不使用 AnchorToken，current patch 找 reference 同位置 patch。

translation:
    使用平均平移。

affine:
    使用全局 affine。

anchor_token_soft:
    current patch 根据 feature + spatial score attend 到 AnchorTokens，直接读 ref_pos。

anchor_token_affine_residual:
    先用 affine 给出粗位置，再从 AnchorTokens 读取局部 residual 修正。

oracle_anchor:
    真实 mesh anchor，对应上限。
```

结果：

```text
BBQ_001_guitar cam06/cam07 start=244, 41 AnchorTokens:
    same_position error: 3.16 patches
    affine error: 1.15 patches
    anchor_token_soft error: 1.41 patches
    anchor_token_affine_residual error: 0.82 patches
    oracle error: 0.00 patches
    token_residual cosine median: 0.592

BBQ_001_juggle cam02/cam01 start=197, 179 AnchorTokens:
    same_position error: 3.16 patches
    affine error: 0.82 patches
    anchor_token_soft error: 1.58 patches
    anchor_token_affine_residual error: 0.66 patches
    oracle error: 0.00 patches
    token_residual cosine median: 0.823

BBQ_001_guitar cam01/cam03 start=5, 7 AnchorTokens:
    same_position error: 10.03 patches
    affine error: 1.05 patches
    anchor_token_soft error: 1.46 patches
    anchor_token_affine_residual error: 1.14 patches
    oracle error: 0.00 patches
    token_residual cosine median: 0.520
```

解释：

```text
1. AnchorToken 不是一个普通 correction head，而是局部 scene re-anchor memory。
2. 只用 feature/spatial 直接读 ref_pos 的 anchor_token_soft 不一定优于 affine，说明 AnchorToken 不能只当 nearest-neighbor memory。
3. 更有效的是 anchor_token_affine_residual：先用 affine 捕捉全局 shot transition，再用 AnchorTokens 提供局部 residual correction。
4. 在 anchor 数充足的两个样本中，AnchorToken residual 明显优于纯 affine。
5. anchor 太少时，AnchorToken residual 不稳定，应由 quality gate 降权或 fallback。
```

当前设计结论：

```text
V6 AnchorToken 不应表达“整张图平均偏移”。
它应表达：
    这个 current patch / local region
    应该参考哪个 ref patch / local region
    在 global affine re-anchor 基础上还需要什么 local residual。

因此，推荐的第一版模型集成是：
    global affine evidence 作为 coarse re-anchor prior
    local AnchorTokens 作为 residual re-anchor tokens
    quality_gate 控制启用强度
    只进入 pose/camera path，不进入 encoder，不进入完整 decoder token sequence
```

输出：

```text
/workspace/code/Movie3R/output/rich_anchor_token_prototype/
```

重点文件：

```text
anchor_token_prototype_summary.jpg
*/anchor_token_leave_one_out_chart.jpg
*/anchor_token_lookup_overlay.jpg
*/anchor_tokens_structured.npz
*/anchor_token_summary.json
```

### 2.4.8 Top-K / quality-gate AnchorToken 选择验证

下一步验证实际推理时是否可以只保留少量 AnchorTokens，而不是把所有 anchors 都传给后续 pose/camera path。

新增脚本：

```text
/workspace/code/Movie3R/scripts/validate_rich_anchor_token_selection.py
```

比较策略：

```text
confidence_topk:
    选择 confidence 最高的 K 个 AnchorTokens。

diverse_topk:
    先按 confidence 排序，再做空间分散选择，避免 token 全堆在一个局部区域。

random_k:
    随机选择 K 个，作为稳定性 baseline。
```

验证方式：

```text
1. 选出 K 个 AnchorTokens 作为 token bank。
2. 用这 K 个 token 估计 affine + local residual。
3. 在未被选择的 held-out anchors 上测试 reference lookup error。
4. 对比纯 affine 与 anchor_token_affine_residual。
```

结果：

```text
BBQ_001_guitar cam06/cam07 start=244, total 41 tokens:
    gate: strong
    best deterministic strategy: diverse_topk, K=8
    affine error: 1.10 patches
    token residual error: 0.77 patches
    improvement: 0.32 patches

BBQ_001_juggle cam02/cam01 start=197, total 179 tokens:
    gate: strong
    confidence_topk K=4:
        affine error: 0.89 patches
        token residual error: 0.66 patches
    diverse_topk K=8:
        affine error: 0.86 patches
        token residual error: 0.71 patches
    random_k K=64:
        affine error: 0.81 patches
        token residual error: 0.65 patches

BBQ_001_guitar cam01/cam03 start=5, total 7 tokens:
    gate: fallback
    K=4 random token residual error: 1.24 patches
    K=4 affine error: 1.25 patches
    improvement: 0.02 patches, not reliable
```

结论：

```text
1. 不需要保留所有 anchors。8-16 个高质量 / 空间分散 AnchorTokens 已经可以提供有效 residual correction。
2. anchor 很多时，top-K deterministic selection 和 random-K 都可工作；实际推理应优先 confidence + spatial diversity，而不是随机。
3. anchor 很少时，即便局部结果看起来不差，也不能强启用 AnchorToken residual；应 fallback 到 affine 或 base path。
4. 推荐第一版 gate：
   unique_anchor_patch_pairs >= 16: strong enable
   8 <= unique_anchor_patch_pairs < 16: weak enable / lower weight
   unique_anchor_patch_pairs < 8: fallback
```

输出：

```text
/workspace/code/Movie3R/output/rich_anchor_token_selection/
```

重点文件：

```text
anchor_token_selection_summary.jpg
*/anchor_token_selection_chart.jpg
*/anchor_token_selection_summary.json
```

### 2.4.9 AnchorToken specificity / negative-control 验证

用户提出的关键问题是：

```text
AnchorToken 进入 decoder / pose path 之前，是否真的像 human token 一样携带明确、有用的信息？
还是只是一个泛泛的 anchor / shot label？
```

为此新增 decoder-before proxy 脚本：

```text
/workspace/code/Movie3R/scripts/validate_anchor_token_specificity.py
```

该脚本仍不修改模型主路径，只使用 frozen Human3R encoder patch tokens 和已验证的 AABB boundary anchors。验证方式是 leave-one-out：拿掉一个真实 anchor，用剩余 anchors 构造 token bank，预测 held-out current patch 应该回看 reference 的哪个 patch。

对照组：

```text
same_position:
    不做 correction，current patch 找 reference 同位置。

affine_only:
    只用 global affine coarse re-anchor，不使用 local token residual。

correct_anchor_token:
    使用正确 boundary 的 AnchorTokens，feature + spatial attention 后读取 local residual。

spatial_only_token:
    使用正确 residual，但 attention 忽略 feature，只靠空间位置。

shuffled_value_token:
    保留正确 key / position，但打乱 residual value，验证 key-value 对应是否重要。

wrong_boundary_token:
    使用另一个 AABB boundary 的 token residual，验证 token 是否只是泛泛先验。

oracle_anchor:
    held-out mesh anchor 本身，作为上限。
```

结果如下，单位是 median patch error，越低越好：

```text
BBQ_001_guitar cam06/cam07 start=244, 41 tokens:
    same_position: 3.16
    affine_only: 1.15
    correct_anchor_token: 0.77
    spatial_only_token: 0.84
    shuffled_value_token: 1.18
    wrong_boundary_token: 1.33
    oracle_anchor: 0.00

BBQ_001_juggle cam02/cam01 start=197, 179 tokens:
    same_position: 3.16
    affine_only: 0.82
    correct_anchor_token: 0.65
    spatial_only_token: 0.68
    shuffled_value_token: 0.82
    wrong_boundary_token: 0.78
    oracle_anchor: 0.00

BBQ_001_guitar cam01/cam03 start=5, 7 tokens:
    same_position: 10.03
    affine_only: 1.05
    correct_anchor_token: 1.11
    spatial_only_token: 1.13
    shuffled_value_token: 1.16
    wrong_boundary_token: 1.13
    oracle_anchor: 0.00
```

解释：

```text
1. 在两个 strong samples 中，correct_anchor_token 明显优于 affine_only。
2. shuffled_value_token 和 wrong_boundary_token 都变差，说明 token 的 key-value 绑定和 boundary specificity 是有意义的。
3. spatial_only_token 也有效，说明几何位置本身携带强信号；但 correct_anchor_token 仍更好，说明 encoder feature 可以补充局部选择。
4. weak sample 只有 7 个 anchors，correct token 不优于 affine，符合 quality_gate/fallback 预期。
5. 因此 AnchorToken 不是泛泛 shot label，也不是只有“有 anchor”这个二值信息；它携带可测试的 local residual correction evidence。
```

当前更精确的结论：

```text
已经证明：
    AnchorToken 在 decoder 前具备明确、可验证的局部 re-anchor 信息；
    正确 token 优于 affine-only；
    打乱 value 或换错 boundary 会退化。

尚未证明：
    AnchorToken 接入 Movie3R pose/camera path 后一定改善最终 3D / camera / SMPL 输出。

下一步：
    用离线 cache 接入 dataset / loader，先做 pose/camera path 的受控小模型实验。
```

输出：

```text
/workspace/code/Movie3R/output/rich_anchor_token_specificity/
```

重点文件：

```text
anchor_token_specificity_summary.jpg
summary.json
*/specificity_summary.json
```

### 2.4.10 Guitar offline AnchorToken cache 小规模生成

为避免训练时在线跑 XFeat + RICH mesh verification，已开始离线生成 AnchorToken cache。当前先只做 `BBQ_001_guitar`，保存到 RICH 数据目录内。

新增脚本：

```text
/workspace/code/Movie3R/scripts/batch_generate_rich_guitar_anchor_cache.py
```

当前不是相机全排列，而是 high-overlap 相邻相机 pair：

```text
camera_pairs = 6-7, 5-6, 4-5, 3-4, 1-2
```

原因：

```text
1. 8 个相机全有向排列是 8 * 7 = 56 个 pair，乘以时间后样本会快速膨胀。
2. low-overlap pair 往往 anchors 少、质量不稳定，不适合作为第一批主训练数据。
3. high-overlap pair 先保证训练信号干净，再逐步加入 medium-overlap pair 增加多样性。
```

已生成两批：

```text
小批 smoke cache:
    path: /workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_v1/
    candidates: 20
    cached: 20
    skipped: 0
    frame_stride: 30
    top_k_tokens: 16
    quality_gate_mean: 0.793
    unique_anchor_patch_pairs_mean: 111.5
    size: 145K

high-overlap cache:
    path: /workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1/
    candidates: 185
    cached: 185
    skipped: 0
    frame_stride: 10
    top_k_tokens: 16
    quality_gate_mean: 0.793
    unique_anchor_patch_pairs_mean: 120.49
    size: 923K
```

每个 `.npz` 只保存 top-K AnchorToken metadata 和 affine evidence，不保存整帧 encoder tokens：

```text
ref_patch_idx
cur_patch_idx
ref_patch_xy
cur_patch_xy
ref_pos_norm
cur_pos_norm
delta_uv_norm
confidence
mesh_error_px
fundamental_inlier
affine_forward
affine_inverse
ref_grid_hw
cur_grid_hw
quality_gate
```

采样策略建议：

```text
第一阶段：high-overlap pairs + frame_stride 10，作为 clean training signal。
第二阶段：加入 medium-overlap pairs，降低采样权重，增加 shot diversity。
第三阶段：low-overlap pairs 只作为 hard validation / fallback 测试，不大量训练。
```

## 2.5 V6.1 当前实现和诊断结果（2026/05/18）

V6.1 不是新的训练结构，而是对 V6-A decoder-after 路线增加真实视频验证闭环。

当前 V6.1 代码路径：

```text
video mp4
  -> scripts/detect_video_shot_changes.py 找 shot boundary 候选
  -> scripts/build_video_anchor_cache.py 用 XFeat + Fundamental RANSAC 生成 anchor npz
  -> demo.py --anchor_path 注入 anchor fields
  -> forward_recurrent_lighter() 缓存每帧 CUT3R encoder feat
  -> _apply_anchor_pose_adapter() gather ref/current patch feat
  -> AnchorPoseAdapter decoder-after 修 camera_pose translation residual
```

V6.1 和 V6-A 的关系：

| 项 | V6-A | V6.1 |
|---|---|---|
| 核心结构 | decoder-after AnchorPoseAdapter | 不改变核心结构 |
| 数据来源 | AvatarReX/RICH anchor cache | 真实视频外部 XFeat anchor npz |
| 推理路径 | full forward 训练路径 | `forward_recurrent_lighter` demo 路径 |
| 目的 | 训练 AnchorPoseAdapter | 验证真实视频是否实际触发、是否有效 |
| 当前修正 | 只修 translation | 仍只修 translation |
| rotation | 默认不修 | 不修，h36 上 rotation jump 保持不变 |

### 2.5.1 V6.1 代码修正结论

最初 `demo.py` 使用 `inference_recurrent_lighter()`，而 V6-A 的 `_apply_anchor_pose_adapter()` 只接在 full recurrent path 中，因此真实视频 with-anchor 实际没有触发 AnchorPoseAdapter。

已修正：

```text
src/dust3r/model.py::forward_recurrent_lighter
  -> 新增 anchor_feats 缓存
  -> downstream head 后调用 _apply_anchor_pose_adapter()
```

修正后 h36 `62 -> 63` 上可以看到 adapter 真实触发：

```text
anchor_pose_gate = 0.4903
anchor_pose_valid = 1.0
anchor_pose_delta_t_norm = 0.04927
anchor_pose_delta_q_norm = 0.0
```

### 2.5.2 h36 / AIST / guitar 诊断结果

当前真实视频 anchor 质量：

| 视频 | boundary | raw matches | F inliers | top-K anchors | quality_gate | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `h36_new.mp4` | `62 -> 63` | 263 | 54 | 16 | 0.4903 | 可触发，但 anchor 有错配 |
| `aist_ms_000000_9s_13s.mp4` | `70 -> 71` | 193 | 102 | 16 | 0.0 | 检测正确，但 gate 判无效 |
| guitar cam01->cam05 | `29 -> 30` | 198 | 29 | 16 | 0.6399 | 中等压力样本 |
| guitar cam01->cam06 | `29 -> 30` | 158 | 17 | 15 | 0.4599 | 压力样本，anchor 少 |

h36 decoder-after 数值效果：

```text
no-anchor boundary translation jump   = 6.5091
with-anchor boundary translation jump = 6.4650
with-anchor camera translation diff   = 0.04927
rotation jump                         = 112.62 deg，不变
```

结论：

```text
1. V6.1 确认 adapter 已经触发。
2. 当前 checkpoint 输出 residual 太小，视觉上基本不可见。
3. translation-only 不能处理 h36 这种大 rotation jump。
```

### 2.5.3 手动 correction 和可视化结论

为了区分“camera_pose 后处理无效”和“adapter 学得太弱”，做了手动 translation 测试：

```text
manual encoded delta_t = [0.5, 0, 0]
camera_translation_shift_norm = 0.5
point_shift_mean_norm = 0.5
```

结论：`camera_pose` 后处理链路是通的。如果输出足够大的正确 residual，viewer/点云会移动。

随后用 h36 的 16 个 anchor 做最简单 3D rigid alignment 可视化：

```text
before anchor 3D residual:
  mean   = 2.43
  median = 0.93
  max    = 11.88

after manual rigid correction:
  mean   = 1.46
  median = 1.43
  max    = 3.36
```

解释：

```text
手动刚体修正能拉近部分极端 outlier，因此 mean/max 下降。
但 anchor 中存在错配，部分原本较好的点被拉坏，median 反而变差。
```

因此当前 h36 anchor 不是“可直接几何修 pose”的干净约束，只能说明存在部分可用信号。

可视化输出：

```text
output/anchor_pose_visualization/h36_new/frame_0062_ref_anchor_points.jpg
output/anchor_pose_visualization/h36_new/frame_0063_cur_anchor_points.jpg
output/anchor_pose_visualization/h36_new/frame_0062_0063_anchor_matches.jpg
output/anchor_pose_visualization/h36_new/manual_anchor_correction_before_after_xz.jpg
```

### 2.5.4 当前判断和下一步 gate

当前不能直接根据 h36 真实视频失败就否定 AnchorToken 思路，因为有两个混杂因素：

```text
1. h36 anchor 质量不够干净。
2. V6-A decoder-after 影响力可能太弱。
```

下一步必须先做 decoder-after overfit gate：

```text
固定 1 个或少量 high-quality anchor 样本
只训练 AnchorPoseAdapter
看 camera_pose 是否能 overfit 到 GT / target pose
```

go/no-go 标准：

| 结果 | 结论 | 后续动作 |
|---|---|---|
| 单样本 overfit 成功 | decoder-after 有容量，问题主要是数据/训练/anchor 质量 | 再决定是否扩大数据或加入 rotation |
| 单样本 overfit 失败 | decoder-after 结构影响力不足或监督不对 | 舍弃 V6-A，直接做 AnchorToken 进 decoder |
| 只能 translation overfit，rotation 不行 | translation-only 设计不足 | 新版本必须让 rotation/pose token 更早读 anchor |

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
V6.1 gate: decoder-after overfit test
    先固定 1 个或少量高质量 anchor 样本，确认当前 AnchorPoseAdapter 是否能 overfit camera pose。

V6-A: XFeat inference-time scene matching prior + decoder-after adapter
    只在 overfit gate 成功时继续扩大训练或加入 rotation residual。

V6-B: XFeat matches -> scene re-anchor tokens -> camera-token adapter
    如果 V6.1 gate 失败，停止 decoder-after，改为让 AnchorTokens 更早进入 decoder/pose-token 更新。

V6-C: teacher-only XFeat / MLP descriptor ablation
    如果 V6-A/B 有明显收益，再研究是否能把 XFeat 蒸馏成内部轻量 descriptor。
```

关于 V6-B 的结构选择，当前记录两个候选：

| 候选 | 方式 | 风险 | 当前优先级 |
|---|---|---|---|
| V6-B1 | pose-only cross attention，anchor 不进 full sequence | 影响力可能仍偏弱 | 中 |
| V6-B2 | AnchorTokens append 到 decoder sequence，但 head slicing 丢弃 anchor tokens | 可能影响 image/human/pointmap，需要 mask/no-op 保护 | 高，若 V6.1 overfit 失败则尝试 |

V6-B2 的最小原则：

```text
1. AnchorTokens 只在有可靠 anchor 的 boundary frame 加入。
2. no-anchor / gate=0 时必须严格等价 base Human3R。
3. downstream head 不直接预测 anchor token，head slicing 必须排除 anchor tokens。
4. 先用单样本 overfit 验证，再扩大训练。
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
