# V14 CUT3R Virtual-View 人体锚点可观测性 Probe 设计

> 日期：2026-07-31
> 状态：完成本地代码只读审计，尚未运行模型实验
> 范围：只验证 CUT3R/Human3R persistent scene state 能否在指定 post-shot 相机下产生可用于人体精对齐的独立 metric geometry；不修改模型代码，不把该信号直接接入最终方法。

## 1. 结论先行

本地 Human3R/CUT3R **保留了 ray-map virtual-view 的训练/完整前向路径**。可以把
cut 前 RGB views 和一个只含目标相机 ray map 的 query view 放在同一个 full forward
中，读取 query 的 self/world pointmap 与 confidence。由于 query 的 `img_mask=False`，
它会读取已有 recurrent state 并生成输出，但不会把 query 写回 global state 或 pose
memory。

但是，当前 Movie3R demo 的真实流式入口 `forward_recurrent_lighter()` 完全没有读取
`ray_map`/`ray_mask`。仓库也没有一个可工作的“输入已有 `state_args`，执行一个 ray
query”的公开 API。因此在不改模型代码的前提下，最小 probe 必须：

```text
pre-shot RGB views
        +
最后一个 ray-only virtual query
        ↓
full model.forward(..., ret_state=True, inference=True)
        ↓
读取 pred[-1] 的 pointmap/conf
```

也就是说，当前能做的是**从头重放短 pre-shot context 后查询**，不能直接复用已经由
demo 轻量流式推理得到的 persistent state。这不妨碍做小规模 observability probe，
但在 probe 成功之前不应为它改线上模型接口。

## 2. 需要回答的唯一科学问题

已有实验已经否定了 Human3R first-post internal pointmap median、mask size 和
apparent-size ratio 作为安全 root-depth 修正。virtual-view probe 不是再做一次相同
统计，而是检查下面这个不同来源的证据：

```text
cut 前 persistent scene state
+ 指定的 post-shot 相机 rays
→ 从 post 相机观察 pre-shot world 的预测 pointmap
→ 该 pointmap 在人物/人体邻域是否仍保留正确的 metric depth 或 support geometry
```

必须先回答：

1. ray-only query 是否能稳定输出有限、非退化的 pointmap 和 confidence；
2. 它是否保留动态人体，还是主要只重建静态背景；
3. 在 first-post person mask/2D joints 内，它的 depth 是否与 GT root-depth residual
   相关；
4. 大视角变化、遮挡和人物运动时，confidence 是否能识别失败；
5. query 是否真的不改变 pre-shot state。

如果 2–4 不成立，这条路线应停止，不能因为 virtual pointmap 看起来平滑就接入 root
修正。

## 3. 已定位的确切代码入口

### 3.1 Ray map 生成

文件：`src/dust3r/datasets/base/base_multiview_dataset.py`

```python
get_ray_map(c2w1, c2w2, intrinsics, h, w)
```

输入：

- `c2w1`: reference/首帧 camera-to-world，`[4,4]`；
- `c2w2`: 目标 virtual camera 的 camera-to-world，`[4,4]`；
- `intrinsics`: 目标相机内参，`[3,3]`；
- `h, w`: query 输出尺寸。

函数先计算：

```python
c2w = np.linalg.inv(c2w1) @ c2w2
```

所以 ray map 的参考坐标是首帧相机坐标。返回值是 `np.float32[H,W,6]`：

- `[...,0:3]`: 广播到每个像素的 ray origin；
- `[...,3:6]`: 按仓库现有公式生成并归一化的 ray field。

这里必须直接复用该函数，不要自行换成“常规的 `R K^-1 p` direction”实现。仓库代码
在旋转结果上加了 ray origin 后再归一化，这是训练时使用的确切表示；另写一个数学上
更常见的版本会造成训练/测试分布变化。

模型 view 中加入 batch 维后的输入应为：

```text
ray_map: torch.float32 [B,H,W,6]，probe 使用 B=1
```

模型在 `model.py` / `model_human3r.py::_encode_views_mhmr()` 内部再执行：

```python
ray_maps = ray_maps.permute(0, 3, 1, 2)  # [B,6,H,W]
```

然后调用 `_encode_ray_map(ray_map, true_shape)`。

### 3.2 Full-forward 调用链

能够真正消费 ray map 的最小现有调用链是：

```text
src/dust3r/inference.py::inference(groups, model, device)
→ loss_of_one_batch(..., inference=True)
→ model(batch, ret_state=True, inference=True)
→ src/dust3r/model.py::_forward_impl()
  或 src/dust3r/model_human3r.py::_forward_impl()
→ _encode_views_mhmr()
→ _encode_ray_map()
→ _recurrent_rollout()
→ _downstream_head()
```

当前 Movie3R 主模型实现在 `src/dust3r/model.py`；
`src/dust3r/model_human3r.py` 中的原始 Human3R 路径具有相同的 ray 编码和基本 state
更新语义。

### 3.3 Full-forward 返回值

`src/dust3r/inference.py::inference()` 返回：

```python
result, state_args = inference(groups, model, device)
query_pred = result["pred"][-1]
```

对标准 `dpt + pts3d+pose+smpl` Human3R head，query 的关键输出为：

| key | shape | 坐标/含义 |
|---|---:|---|
| `pts3d_in_self_view` | `[1,H,W,3]` | virtual query 相机局部坐标中的 pointmap |
| `conf_self` | `[1,H,W]` | self pointmap confidence |
| `pts3d_in_other_view` | `[1,H,W,3]` | 首帧/reference camera 坐标中的共享 pointmap |
| `conf` | `[1,H,W]` | shared/cross pointmap confidence |
| `camera_pose` | `[1,7]` | 模型自身为 query 解码的 reference-relative camera pose |

`pts3d_in_other_view` 的 reference-frame 语义可以由
`src/dust3r/losses.py::get_all_pts3d()` 确认：GT cross point 被显式变换到
`inv(gts[0]["camera_pose"])`，即 view-1 camera frame。

当前默认 `conf_mode=("exp",1,+inf)`，所以 `conf_self/conf` 不是 `[0,1]` 概率，不能
直接以 `0.5` 阈值解释。probe 应先用分位数、均值和与实际误差的 calibration 曲线，
再确定 gate。

## 4. Query view 的最小张量约定

full forward 会 `torch.stack()` 所有 view，所以 pre-shot image views 和 query view 必须
具有相同的 batch、`H,W` 和基础字段。query view 建议直接构造，不依赖隐藏默认值：

```python
query = {
    # ray-only query 不会编码 RGB；这里仍需提供同形占位张量
    "img": torch.zeros(1, 3, H, W),
    "img_mhmr": torch.zeros_like(pre_views[0]["img_mhmr"]),
    "K_mhmr": pre_views[0]["K_mhmr"].clone(),
    "true_shape": torch.tensor([[H, W]], dtype=torch.int32),

    # get_ray_map(...) 返回 [H,W,6]，这里显式增加 batch 维
    "ray_map": torch.from_numpy(ray_map).unsqueeze(0).float(),
    "img_mask": torch.tensor([False]),
    "ray_mask": torch.tensor([True]),

    # 双重关闭写状态，避免以后 mask 语义变更时意外写入
    "update": torch.tensor([False]),
    "update_state": torch.tensor([False]),
    "update_mem": torch.tensor([False]),
    "update_v8_history": torch.tensor([False]),
    "reset": torch.tensor([False]),

    "idx": query_index,
    "instance": str(query_index),
    "camera_pose": torch.eye(4).unsqueeze(0),
}
```

对每个 RGB view，也必须有形状为 `[1,H,W,6]` 的 ray-map 占位和
`ray_mask=False`。不要沿用 `demo.py::prepare_input()` 在 image-only 分支创建的
`[B,6,H,W]` 占位；当 image view 与 ray view 被一起 `torch.stack()` 时，这两种布局
不兼容。最安全的方式是把所有非 ray view 的占位统一为：

```python
view["ray_map"] = torch.zeros(1, H, W, 6)
view["ray_mask"] = torch.tensor([False])
```

`demo.py::prepare_input(..., raymaps=..., img_mask=..., raymap_mask=...)` 提供了 mixed
view 的字段模板，但其 ray-only `true_shape` 使用
`raymaps[k].shape[1:-1][::-1]`。对明确为 `[B,H,W,6]` 的 ray map，这会把 `H,W`
反转。因此 probe 脚本应显式覆盖为 `[[H,W]]`，不能未经核对直接使用其结果。

## 5. 为什么 query 不更新 persistent state

`model.py::_forward_impl()` 和 `model_human3r.py::_forward_impl()` 在产生输出以后才提交
state：

```python
img_mask = views[i]["img_mask"]
update = views[i].get("update", None)
update_mask = (img_mask & update) if update is not None else img_mask
update_mask = update_mask[:, None, None].float()

state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
mem = new_mem * update_mask + mem * (1 - update_mask)
```

ray-only query 设置 `img_mask=False`，因此即使 decoder 会计算一个
`new_state_feat/new_mem` 来生成当前输出，提交 mask 仍为 0，旧 `state_feat/mem` 被保留。
额外设置所有 `update*=False` 是防御性约束。

`ret_state=True` 返回的每个元素是：

```text
(state_feat, state_pos, init_state_feat, mem, init_mem)
```

返回列表在 index 0 保存初始化状态，之后每处理一个 view 追加一次。因此 query 位于
最后一个 view 时，应检查：

```python
before = state_args[-2]
after = state_args[-1]
assert torch.equal(before[0], after[0])  # state_feat
assert torch.equal(before[1], after[1])  # state_pos，若非 None
assert torch.equal(before[3], after[3])  # pose memory
```

同时检查所有张量 finite。若不相等，probe 立即失败，不应继续评估 pointmap。

## 6. 当前无法直接使用的入口与具体阻碍

### 6.1 `forward_recurrent_lighter()` 不支持 ray query

demo 实际调用：

```text
demo.py
→ src/dust3r/inference.py::inference_recurrent_lighter()
→ model.forward_recurrent_lighter()
```

`model.py` 和 `model_human3r.py` 的 lighter 实现逐帧只编码：

```python
view["img"]
view["img_mhmr"]
```

函数内没有读取 `view["ray_map"]` 或 `view["ray_mask"]`，也没有调用
`_encode_ray_map()`。把 ray-only view 直接塞进 demo 默认入口不会得到 virtual-view
query，只会把占位 RGB 当成普通帧，结论无效。

### 6.2 `inference_step()` 包装器在本地不可工作

`src/dust3r/inference.py::inference_step()` 会调用：

```python
model.inference_step(view, state_feat, state_pos, init_state_feat, mem, init_mem)
```

但本地 `src/dust3r/model.py` 和 `src/dust3r/model_human3r.py` 均没有实现
`inference_step` 方法。因此不能把 full forward 返回的 `state_args` 直接交给这个包装器。

### 6.3 Full `forward()` 不接受外部 state

`model.forward(views, ret_state=True, inference=True)` 总是从第一个输入 view 初始化：

```python
state_feat, state_pos = self._init_state(feat[0], pos[0])
```

没有 `state_args=` 参数。返回的 state tuple 也只覆盖 state feature 和 pose memory，
不包含 lighter 路径中的 tracking、Movie3R human memory 与部分 V8/V9 局部历史。故当前
不能无损地把已运行 demo 的完整在线上下文迁移到 full ray-query path。

### 6.4 Human token 对 ray-only query 没有可靠语义

full Human3R 的 MHMR tokenizer 仍会对每个 view 的 `img_mhmr` feature 做人体检测。ray-only
view 使用的是 masked MHMR token，并没有真实 RGB。因此 query 输出中的 SMPL/human
detection 不应使用。这个 probe 只读取 pointmap/conf，并使用 first-post RGB 上已有的
person mask、2D joints 或 B0 SMPL 投影在 query pointmap 中取样。

## 7. 不改模型代码的最小调用路径

伪代码只展示调用顺序；probe 实现时应复用项目既有模型加载、图像预处理和 checkpoint
配置，避免引入另一个 preprocessing domain。

```python
from src.dust3r.datasets.base.base_multiview_dataset import get_ray_map
from src.dust3r.inference import inference

# 1. 取一个短的 cut 前 context，保持和 B0 相同的 resize/pad。
pre_views = build_pre_views(pre_rgb_paths)

# 2. DA3/B0 提供的 post camera 必须先转成与 c2w_ref 同一世界坐标的 c2w。
ray_np = get_ray_map(
    c2w_ref,
    c2w_post_query,
    K_post_resized,
    H,
    W,
).astype("float32")

# 3. 规范所有 RGB view 的 ray 占位布局，并追加只读 query。
for view in pre_views:
    view["ray_map"] = torch.zeros(1, H, W, 6)
    view["ray_mask"] = torch.tensor([False])

query = build_query_view(ray_np, H, W, pre_views[0])
groups = pre_views + [query]

# 4. 必须走 full inference，不能走 inference_recurrent_lighter。
result, state_args = inference(groups, model, device)
pred_q = result["pred"][-1]

point_world = pred_q["pts3d_in_other_view"]  # [1,H,W,3]
conf_world = pred_q["conf"]                  # [1,H,W]
point_local = pred_q["pts3d_in_self_view"]  # [1,H,W,3]
conf_local = pred_q["conf_self"]            # [1,H,W]

# 5. query 必须保持 state/mem bit-exact。
verify_state_unchanged(state_args[-2], state_args[-1])
```

相机约定是这一 probe 最容易发生 silent failure 的地方：

- `get_ray_map` 要求两个 pose 都是 `camera-to-world`；
- 两个 pose 必须处于同一个世界/参考系；
- `K_post_resized` 必须对应 Human3R 实际输入的 resize/crop 后像素，而不是原图 K；
- 不能把 DA3 的 `world-to-camera` 直接传入；
- 先用 identity query（`c2w2 == c2w1`）做 sanity check，再测试真实 post camera。

## 8. 最小 Probe 矩阵与通过标准

本轮只建议选少量已知 GT、人物可见的 boundary，不跑完整 180 cuts：

| Probe | 输入 query | 目的 |
|---|---|---|
| P0 | identity camera | 验证 ray layout、K、坐标和输出 finite |
| P1 | 小平移/小旋转 synthetic camera | 验证输出随 rays 连续变化而不是忽略 ray map |
| P2 | GT post camera（仅 evaluator 可见 GT） | 测 virtual geometry 的上限与可观测性 |
| P3 | B0/DA3 post camera | 测真实 deployable camera 下的信号 |
| P4 | 同一个 query，打乱/置零 ray map | 证明输出确实使用 ray，而非只复现 state prior |

每个 query 至少记录：

- state/mem 是否 bit-exact 不变；
- pointmap finite ratio；
- `conf/conf_self` 分位数；
- first-post person mask 内的有效点比例；
- mask 内 point depth 的 median/MAD，但**不直接把 median 当修正结果**；
- 与 GT root depth 的 signed residual、absolute residual 和 Spearman correlation；
- near/far、遮挡、多人重叠和大视角变化分组；
- `P3` 相对 B0 raw 的 potential correction harm rate。

判定为“值得进入下一阶段”的最低标准：

1. P0–P1 通过，且 query 不更新 state；
2. P2/P3 在 person region 内有足够的有效点，不只是静态背景；
3. depth evidence 与 GT root-depth residual 有稳定同号关系或显著 rank correlation；
4. confidence/visibility 特征能把大部分严重失败隔离出去；
5. 在 frozen validation boundary 上，简单保守 gate 的 >5 cm harm rate 明显低于现有
   Human3R internal pointmap 方法。

若 virtual pointmap 只保留静态场景，也不一定完全失败：可进一步检查脚底 support plane
与 penetration，但不能再把人物 mask 内的背景深度误当成人体 root depth。

## 9. 与 V14 主线的关系

这个 probe 不是新主线，而是给以下精对齐优化寻找独立观测：

```text
冻结 B0/DA3 camera
→ 对每个人只优化 Δz / Δxy / 小 ΔR
→ virtual-view person/support evidence 提供可选约束
→ 证据不可靠时严格 fallback 到 B0
```

如果 probe 证明 virtual geometry 对人物 depth 有稳定信号，它可以进入
`Person-Conditioned Boundary Scene Registration`，或作为 UniCon3R-style learned
residual head 的输入。如果 probe 失败，则回到 DA3 person geometry、2D reprojection、
contact/penetration 和学习型 residual head，不修改 CUT3R state 接口。

## 10. 最终实现可行性判断

| 问题 | 当前答案 |
|---|---|
| 本地 checkpoint/model 是否包含 ray encoder | 是，`_encode_ray_map()` 已连接 full forward |
| 能否输出 pointmap/conf | 是，四个关键张量都由现有 downstream head 返回 |
| ray-only query 是否理论上只读 state | 是，`img_mask=False` 使 state/mem update mask 为 0 |
| demo 默认 lighter path 能否直接 query | 否，它完全绕过 ray encoder |
| 能否直接复用已跑 demo 的 `state_args` | 否，缺少可工作的单步 API，full forward 也不接收外部 state |
| 不改模型能否做小 probe | 能，短 pre-shot context + ray query 一次 full-forward 重放 |
| 现在是否值得改线上模型接口 | 否，先验证动态人体/支撑面可观测性和失败 gate |
