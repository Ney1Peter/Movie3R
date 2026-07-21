# V10 因果式 State-query Shot Prompt 验证

## 1. 实验目的

本轮实验验证一个严格流式的架构假设：camera cut 后冻结分镜前的旧状态，只读查询旧状态；新镜头从 fresh state 开始，并只提交自己的第一次 state write。实验不修改 Human3R 默认推理，Human3R 全部冻结，只训练独立的小型 first-write adapter。

重点问题是：

1. 可部署的分镜前状态 `S_{t-1}` 是否包含旧世界信息；
2. early camera query 和第一次 state write 分别控制什么；
3. 学到的 state transition 是否能持续改善后续帧；
4. state prompt 能否与固定 shot-level 显式 SE(3) 互补。

## 2. 阶段零：因果位置审计

代码审计和 activation capture 确认：

- 之前有效的 persistent state 是当前帧 decoder 之前的 `S_{t-1}`，没有看过 cut 后当前帧，部署时可获得；
- teacher 当前帧结束后的 `S_t` 包含当前帧，只能作为 first-write Oracle；
- 所谓 initial camera activation 不是共享参数，而是当前图像查询旧 pose memory 后、进入 decoder layer 0 前的 history-conditioned early camera query；
- early camera query 位于第一次 state write 之前；
- Human3R 实际有 scene/world persistent state 和 pose retriever memory 两类历史。

## 3. 阶段一：State Transition Oracle

### 3.1 数据与设置

从三段连续单目视频各取 24 帧，在第 12 帧构造 pseudo-cut。Teacher 正常连续运行；Student 从同一张 RGB 开始 fresh reset。评估边界以及 cut 后第 1、2、4、8 帧。

### 3.2 关键结果

三段平均结果：

| 方法 | Camera T | Camera R | World pointmap | Human root | Camera R 恢复 |
|---|---:|---:|---:|---:|---:|
| Reset baseline | 0.351 m | 25.74 度 | 2.151 m | 2.413 m | 0% |
| Late camera only | 0.309 m | 23.33 度 | 2.034 m | 2.203 m | 7% |
| Early camera only | 0.324--0.336 m | 24.39--25.86 度 | 2.019--2.086 m | 2.252--2.342 m | 约 0% |
| Correct first write `S_t` | 0.079 m | 2.63 度 | 0.271 m | 0.288 m | 79% |
| Late camera + correct first write | 0.108 m | 1.04 度 | 0.210 m | 0.125 m | 81% |
| 可用 `S_{t-1}` 直接作为 active state | 0.201 m | 6.06 度 | 0.430 m | 0.498 m | 68% |
| `S_{t-1}` + history camera query | 0.108 m | 1.04 度 | 0.147 m | 0.096 m | 80% |
| 不可用 post-update `S_t` + camera | 0.102 m | 1.21 度 | 0.137 m | 0.085 m | 81% |

逐帧现象：

- late refined camera 在边界帧恢复 100%，下一帧立即回到接近 0% 恢复；
- correct first-write 不改变边界输出，但第 1、2、4、8 帧 camera rotation 恢复约 79%、83%、86%、89%；
- 对应 world pointmap 恢复约 90%、89%、88%、86%；
- 对应 human root 恢复约 94%、92%、93%、93%；
- 只把旧 state 读成一个 camera guidance、再写 fresh state 的无训练双 pass 几乎无效。

因此，第一次 state write 是后续漂移的核心；但旧 state 中的信息不能通过简单硬替换一个 camera token 自动转移到 fresh state，需要一个学习到的 state transition。

## 4. 阶段二：最小可训练 Prompt

### 4.1 模块

新增独立模块：

```text
只读旧 S_{t-1}
+ fresh branch 第一次原始 write
+ 当前图像 token 的 mean/std
+ 当前人体 token 的 mean/std
+ early camera query
+ 旧 pose memory 的 mean/std
        ↓
共享 token-wise 低秩 MLP
        ↓
gate × first-write residual
        ↓
只提交 corrected fresh state
```

Human3R encoder、decoder、camera head、pointmap head、SMPL-X head 和 Multi-HMR 均冻结。正常无 cut 帧不触发该模块，因此默认路径输出完全不变。

分别训练两个版本：

- Raw first-write：使用 reset 分支的共享初始 camera query；
- Early-query first-write：当前图像先查询只读旧 pose memory，再预测 first-write residual。

### 4.2 训练数据

- 三段连续视频，每段 24 帧；
- 每段边界为 6、9、12、15，共 12 个 pseudo-cut；
- clip01、clip02 的 8 个样本训练；
- clip03 的 4 个样本作为未见片段验证；
- 两个 adapter 分别在 GPU 上训练 1200 步；
- Human3R 前向、缓存和 rollout 评测均在 GPU 上完成。

### 4.3 Latent 拟合

| Adapter | 训练 latent recovery | 未见 clip03 latent recovery |
|---|---:|---:|
| Raw first-write | 69.3% | 42.6% |
| Early-query first-write | 68.8% | 42.1% |

两个版本均能学习 first-write residual，但 early query 没有明显提高 latent 泛化。

## 5. Rollout 结果

### 5.1 未见 clip03

| 方法 | Camera T | Camera R | Camera-frame pointmap | World pointmap | Human root |
|---|---:|---:|---:|---:|---:|
| Hard reset | 0.511 m | 49.33 度 | 0.525 m | 3.551 m | 4.569 m |
| Boundary output Oracle | 0.258 m | 5.11 度 | 0.525 m | 0.784 m | 0.518 m |
| Raw State-query Prompt | 0.275 m | 22.16 度 | 0.598 m | 1.849 m | 2.280 m |
| Early State-query Prompt | 0.255 m | 21.48 度 | 0.630 m | 1.840 m | 2.251 m |
| Early Prompt + output Oracle | 0.548 m | 28.85 度 | 0.630 m | 2.010 m | 2.526 m |
| First-write `S_t` Oracle | 0.149 m | 5.24 度 | 0.140 m | 0.469 m | 0.563 m |

### 5.2 因果控制

- 正确旧 state 的 Early Prompt 将 rotation 从 49.33 度降到 21.48 度；
- old state 置零后 rotation 变为 51.87 度，world pointmap 变为 3.799 m；
- 打乱旧 state 后虽然个别 camera 指标偶然变好，但 world pointmap 和 human trajectory 明显不稳定，逐帧出现大幅负恢复；
- 说明模块确实读取旧 state，而不是只依赖 fresh 当前帧。

### 5.3 Early camera 的作用

Early-query 相比 Raw Prompt 只有小幅改善：

- rotation：22.16 度降到 21.48 度；
- translation：0.275 m 降到 0.255 m；
- world pointmap：1.849 m 降到 1.840 m。

因此，当前设计下 early camera query 不是主要瓶颈，first-write guidance 更重要。

## 6. 最关键的失败：与显式 SE(3) 不互补

Boundary output Oracle 单独已经把未见 clip03 rotation 降到 5.11 度；Early Prompt 单独为 21.48 度，但二者直接组合反而变成 28.85 度。

更重要的是，即使使用正确 teacher `S_t` 的 First-write Oracle，单独 rotation 为 5.24 度，与 Boundary output Oracle 组合后也会恶化到 44.63 度。

原因不是模块容量不足，而是目标定义冲突：

- 当前 first-write adapter 被监督去恢复 teacher 的绝对 world gauge；
- 固定 boundary SE(3) 也在修正同一个 world gauge；
- 两者顺序叠加会发生重复坐标变换；
- prompt 改变后续 state gauge 后，边界第一帧与后续 state 不再处在同一种 correction convention 中。

所以当前方案不是“latent state transition + explicit metric alignment”的真正互补，而是两个模块重复修同一件事。

## 7. 明确结论

1. **旧 `S_{t-1}` 可以作为可部署的只读 world context。** 正确旧 state 明显优于 zero/shuffle 控制，且可学习模块在未见片段上恢复约 42% latent error、约 55% camera rotation error。

2. **first-write guidance 是必要且有持续作用的。** Oracle first-write 对第 1、2、4、8 帧均有强恢复，证明 state transition 确实控制后续 camera、scene 和 human rollout。

3. **early camera correction 当前不是必要主模块。** 它只有小幅增益，不能单独阻止漂移；应作为消融或辅助输入保留。

4. **Shot Prompt 相比纯 output correction 提供了时序信息，但当前精度仍较差。** 它能改变后续 relative trajectory，而 output correction 不能；但在当前 pseudo-cut 上，Boundary output Oracle 单独更准确。

5. **当前 Shot Prompt 与显式 SE(3) 没有形成真实互补。** 直接组合会双重修正 gauge，触发本轮停止条件。

6. **暂时不进入 180 个真实跨相机样本和 Multi-THuMBS 正式对比。** 当前架构目标尚未定义正确，扩大训练只会放大 gauge 冲突。

## 8. 下一步调整

下一版 Prompt 不应回归完整 teacher `S_t`，而应只监督 gauge-neutral 的动态量：

- 相对 camera motion；
- camera-frame pointmap / depth consistency；
- root-centered human motion；
- torso heading 的相对变化；
- first-write 中去除全局 SE(3) 后的 residual state；
- 显式 SE(3) 统一负责绝对 world gauge。

推荐新的组合顺序：

```text
fresh Human3R local reconstruction
        ↓
State-query Prompt 只修正 gauge-neutral motion/state transition
        ↓
提交 fresh local state
        ↓
固定 shot-level 显式 SE(3) 只负责 world gauge
        ↓
camera / pointmap / SMPL-X 统一输出
```

只有当这个版本在 pseudo-cut 上满足“Prompt + Explicit 优于 Explicit-only”，才值得扩展到 180 个 AvatarReX、THuman、MVHuman 跨分镜样本。

## 9. 代码与结果

核心代码：

```text
src/dust3r/v10_causal_state_query_prompt.py
scripts/v10_causal_state_transition_oracle_probe.py
scripts/v10_causal_state_query_prompt_validation.py
scripts/v10_train_causal_state_query_adapter.py
scripts/v10_eval_causal_state_query_prompt.py
scripts/v10_merge_causal_state_query_prompt.py
```

结果目录：

```text
output/v10_latent_token_probe/causal_state_transition_oracle_v2/
output/v10_latent_token_probe/causal_state_query_prompt/
```
