# ShotToken 生成流程解释

这份文档解释当前 Movie3R 里 ShotToken 是怎么从图片输入一步步生成出来的，以及它在进入 decoder 前后到底携带了什么信息。

## 1. 图片进入模型

单帧图片输入模型时，形状可以理解为：

```text
image: [3, H, W]
```

以当前常用的 AABB 可视化样本为例，输入分辨率是：

```text
[3, 288, 512]
```

模型不会把整张图片直接当作一个整体特征，而是先把图片切成 patch。

## 2. 图片被切成 patch

当前 patch size 是 `16`，所以对于 `288 x 512` 的图像：

```text
288 / 16 = 18
512 / 16 = 32
```

总 patch 数是：

```text
18 x 32 = 576
```

也就是说，一帧图像会变成一个 `18 x 32` 的 patch token 网格：

```text
image
  -> 18 x 32 patches
  -> 576 patch tokens
```

每个 patch 对应一个 token。

## 3. Encoder 处理 patch token

这些 patch token 会进入 CUT3R / CroCo encoder。

刚切 patch 时，每个 token 更像一个局部图像块特征。但经过 transformer encoder 之后，每个 patch token 已经不只是自己的局部 patch 信息了。因为 encoder 里有 self-attention，每个 patch token 可以和其它 patch token 交互。

所以 encoder 输出的 token 更准确地说是：

```text
带全局上下文的 patch token
```

形状大概是：

```text
feat[i]: [B, 576, 1024]
```

含义是：

```text
B    = batch size
576  = patch token 数
1024 = 每个 patch token 的特征维度
```

## 4. 投影到 decoder 维度

ShotToken 使用的是进入 decoder 前的 image token，不是最原始的 encoder token。

代码里会做：

```python
f_dec = self.decoder_embed(feat)
```

这一步把每个 patch token 从 `1024` 维投影到 `768` 维：

```text
encoder output: [B, 576, 1024]
decoder input:  [B, 576, 768]
```

ShotTokenGenerator 使用的就是这个 `f_dec[i]`。

## 5. 对当前帧所有 patch token 做平均

假设当前帧是第 `i` 帧：

```text
f_dec[i]: [B, 576, 768]
```

ShotTokenGenerator 会直接对 `576` 个 patch token 做平均：

```python
g_curr = feat_curr.mean(dim=1)
```

得到：

```text
g_curr: [B, 768]
```

通俗理解就是：

```text
把当前帧 576 个 patch token 混成一个全局向量
```

这里没有 learned weight，没有 attention pooling，也没有挑重点区域。它就是无权重平均：

```text
g_curr[d] = (patch_0[d] + patch_1[d] + ... + patch_575[d]) / 576
```

## 6. 对前一帧也做同样平均

前一帧也做同样的 mean pooling：

```python
g_prev = feat_prev.mean(dim=1)
```

得到：

```text
g_prev: [B, 768]
```

此时有两个全局特征：

```text
g_prev: 前一帧整体特征
g_curr: 当前帧整体特征
```

## 7. 比较前后两帧的全局特征

ShotTokenGenerator 会计算两类比较信息：

```python
diff = g_curr - g_prev
sim = F.cosine_similarity(g_curr, g_prev, dim=-1)
```

含义是：

```text
diff: 当前帧和上一帧整体差了什么
sim:  当前帧和上一帧整体像不像
```

形状是：

```text
diff: [B, 768]
sim:  [B]
```

## 8. 拼成一个 MLP 输入向量

然后把这些信息拼起来：

```python
x = torch.cat([g_curr, g_prev, diff, sim.unsqueeze(-1)], dim=-1)
```

所以 ShotTokenGenerator 的 MLP 看到的信息是：

```text
当前帧整体特征
上一帧整体特征
两帧整体差值
两帧整体相似度
```

输入维度是：

```text
768 + 768 + 768 + 1 = 2305
```

## 9. MLP 生成 raw shot token

拼接后的 `x` 会进入一个小 MLP：

```python
q_raw = self.shot_mlp(x).unsqueeze(1)
```

输出是：

```text
q_raw: [B, 1, 768]
```

这才是原始的 shot token。

所以要注意：

```text
ShotToken 不是当前帧平均特征本身。
ShotToken 是由当前帧平均特征、上一帧平均特征、两者差值和两者相似度经过 MLP 生成的新 token。
```

## 10. 另一个 MLP 预测 shot_prob

同时，ShotTokenGenerator 还会用另一个 MLP 预测当前帧是不是 shot change：

```python
shot_logit = self.shot_logit_mlp(x).squeeze(-1)
shot_prob = torch.sigmoid(shot_logit)
```

输出是：

```text
shot_prob: [B]
```

理论上：

```text
连续帧 shot_prob 应该低
镜头跳变帧 shot_prob 应该高
```

但当前可视化显示，`shot_prob` 对 AABB 边界的区分并不明显。

## 11. q_raw 经过 norm、gate、scale

实际传给后续模块的不是 `q_raw`，而是 `q_t`：

```python
q_t = self.shot_scale * shot_prob * self.shot_norm(q_raw)
```

可以理解为：

```text
q_t = 被 shot_prob 控制强弱的 shot token
```

直观设计意图是：

```text
如果 shot_prob 很小，q_t 应该很弱
如果 shot_prob 很大，q_t 应该更强
```

但是当前 V5.1-LAST2 里还有一个需要注意的问题：`LayerwisePoseShotAdapter` 内部又会对 `q_t` 做一次 `LayerNorm`。这意味着 `shot_scale * shot_prob` 对 q_t 幅度的控制，可能在进入 pose-shot attention 前被重新归一化掉。

当前可视化里已经看到：

```text
q_norm 很小，但 adapter context_norm(q_t) 后变大很多
```

这说明 gate/scale 未必真的控制了 q_t 对 pose attention 的影响强度。

## 12. q_t 怎么进入 decoder

当前 V5.1-LAST2 里，`q_t` 不作为普通 decoder token 拼进完整 token 序列。

也就是说，它不是这样进入 decoder：

```text
[pose token, image tokens, human tokens, shot token]
```

而是这样：

```text
decoder 正常处理:
[pose token, image tokens, human tokens]

在最后两层 decoder block 后:
pose token 单独看一下 q_t
只更新 pose token
```

具体机制可以理解为：

```text
pose token attends to [pose token, q_t]
```

也就是说：

```text
q_t 只影响 camera pose token
q_t 不直接影响 image token
q_t 不直接影响 human token
q_t 不直接影响 pointmap token
```

这样设计的目的是避免 ShotToken 污染重建分支，只让它参与 camera pose 修正。

## 总结

当前 ShotToken 生成流程可以概括为：

```text
image
  -> patch embedding
  -> encoder patch tokens
  -> decoder_embed
  -> decoder input patch tokens: [B, 576, 768]
  -> 对当前帧 576 个 patch token 做无权重平均，得到 g_curr
  -> 对上一帧 576 个 patch token 做无权重平均，得到 g_prev
  -> 计算 diff = g_curr - g_prev
  -> 计算 sim = cosine(g_curr, g_prev)
  -> 拼接 [g_curr, g_prev, diff, sim]
  -> MLP 生成 q_raw
  -> 另一个 MLP 生成 shot_prob
  -> q_t = shot_scale * shot_prob * LayerNorm(q_raw)
  -> q_t 只进入 pose-only adapter，影响 camera pose token
```

最关键的问题是：

```text
所有 patch token 被平均成一个全局向量以后，空间信息基本丢失。
模型不知道变化来自哪里。
它不知道是人体变了、背景变了、相机跳了，还是局部遮挡。
```

这也是为什么后续需要继续做 patch contribution 可视化、human/background 分解，以及更局部的 scene-anchor / feature matching。
