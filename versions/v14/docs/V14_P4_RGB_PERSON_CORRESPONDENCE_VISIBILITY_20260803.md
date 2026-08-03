# V14 P4：无额外预训练模型的 RGB 人体纹理对应可观测性

日期：2026-08-03  
状态：**探索性 runtime-only visibility audit；尚无结果。**

## 动机

P3 已说明 Human3R 预测 joint ray 不能再提供比 BRTC 更高维的刚体约束。P4 检验一条与
预测 3D joint 几何正交的最小路线：RGB 中的人体衣物/纹理特征。它不引入 ReID、DINO、
ViTPose 或深度模型，只使用 OpenCV SIFT（手工局部描述子）和已经存在的 runtime person
bbox / geometry association。

若同一人的 pre/post RGB 匹配足够密集且分散，可在后续的新 protocol 中把 pre predicted
mesh surface 作为 3D source、post SIFT pixels 作为 2D observation，做双向 mesh-PnP 的
independent rigid proposal。这里**不做 PnP、不选择阈值、更不提交人体动作**；先问这个
前提是否存在。

## 固定诊断

复用 P1 schema-2 的 36-event `three` runtime cache 和其保存的输入 RGB。每张 image
按原 Human3R input 的长边 `512` resize；对 B0 geometry-Hungarian `(pre i, post j)`：

```text
pre/post grayscale RGB
 -> SIFT features over full image
 -> one-way BF L2 2-NN Lowe ratio < 0.70
 -> both endpoints must lie in their predicted bbox 的内缩 10%区域
 -> count pair-local RGB matches and 3x3 spatial occupied cells
```

这条检查只读取 runtime record、image、bbox 与 anonymous association。它不读取 P1/P2
evaluator fields、GT identity/mesh/camera、future frame 或任何新网络；B0、camera、scene、
human geometry 和 state 均不被修改。

为避免“几条同位置纹理匹配”冒充可解 PnP，固定定义一个 **PnP-observable pair** 为：至少
`8` 条 filtered matches，且 pre、post 各至少占据 `3` 个 3×3 bbox cell。数字在运行此 audit
前固定；不是一个部署 gate。

## 决策

这是在已经读取该 development cache 后进行的探索性 feasibility audit，不是 blind
confirmation。它的唯一 promotion 条件是：PnP-observable pair coverage 至少为全部 matched
runtime rows 的 `20%`（并保存 case-level count/spread）。低于此值则：

```text
NO_GO_RGB_PERSON_SIFT_OBSERVABILITY
```

并停止对相同 SIFT+bbox candidate 调 ratio、bbox margin、match count 或 PnP RANSAC；低
coverage 不能通过放宽 gate 伪造成通用精对齐。只有满足 coverage，才允许在一个新分离的
selection split 上定义 mesh rasterization、双向 PnP verification 和 exact fallback。

## 已完成结果（2026-08-03）

结论：

```text
NO_GO_RGB_PERSON_SIFT_OBSERVABILITY
```

在完整的 36-event cache 上，96 个 geometry-Hungarian runtime person rows 的 bbox-local
RGB match count 为 mean `1.57`、median `1`、maximum `13`。仅有两个 rows 同时达到预注册的
至少 8 条 match、两端至少 3 个 spatial cells：

```text
dev_three_t0700_c0_c3, pre/post index 0/1: 13 matches, 6/4 cells
dev_three_t0700_c3_c0, pre/post index 1/0:  8 matches, 3/4 cells
```

所以 PnP-observable coverage 为 `2/96 = 2.08%`，远低于 `20%` promotion 条件。该 probe
从头到尾没有读取 evaluator / GT、future post frame 或 P2 labels，也没有修改 B0 camera、
人体、scene 或 recurrent state。

这不是“再放宽 ratio 或 box 就能解决”的结果：高跨视角、低分辨率人体 crop 中，手工局部
纹理本来就不能提供通用的跨 shot correspondence。为覆盖率而降低 Lowe ratio、扩大 bbox 或
把少数匹配当 dense correspondence，只会主要引入背景和错误 person feature，不能形成论文
级精对齐证据。因此不进入 mesh-PnP，也不在这个已读 split 上扫 SIFT/PnP 参数。若未来允许
新的独立 learned correspondence / ReID observation，应作为不同方法和新 protocol 重启；它
不属于当前“不额外引入预训练模型”的主线。
