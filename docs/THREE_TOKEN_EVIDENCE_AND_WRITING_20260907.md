# Shot3R 三 Token 旧实验审计与论文写法建议

日期：2026-09-07

## 1. 结论

现有记录可以支持以下较窄但可信的判断：

> 语义、相机对齐和时间上下文三类信息具有不同作用；完整表示在早期小样本训练探针中具有更好的综合稳定性，并在新的 MVHuman 大视角敏感性测试中取得更低的跨镜头接缝和相机旋转误差。

现有记录不能支持以下强结论：

> 三个 token 缺一不可，且完整三 token 在所有数据集和所有指标上均最优。

原因是：唯一对各结构独立训练的 V9 实验复用了同一组 10 条训练/验证/测试样本；现有 EgoHumans 和 MVHuman 实验则是在同一完整 checkpoint 上进行推理时 token 屏蔽，而不是独立重训。不同指标的结论也不一致。

因此，论文应把三 token 写成历史条件对齐路径中的一种“分类型互补证据表示”，不要把“为什么恰好必须是三个”提升为核心理论主张。Shot3R 的核心贡献仍然是对齐信息与递归状态传播具有不同生命周期，以及由此形成的 alignment--state decoupling。

## 2. 找到的旧实验记录

### 2.1 V9 Small10 独立训练探针

关键记录：

- `versions/v9/docs/IMPROVEMENT_EXPLORATION_SUMMARY_20260705.md`
- `versions/v9/docs/EXPERIMENT_RECAP_20260630.md`
- `config/archive/v9_legacy/train_v9_small10_token_ablation_base.yaml`
- `config/archive/v9_legacy/train_v9_small10_ablate_{all_mean,no_semantic,no_alignment,no_momentum,single_token,learned_pooling}.yaml`
- `scripts/training/run_v9_small10_token_ablation.sh`

协议：

- 7 个 AvatarReX + 3 个 THuman 四帧 AABB 样本；
- `all / no semantic / no alignment / no momentum` 分别初始化并训练；
- 完整版本记录于第 63 epoch，删除版本记录于第 60 epoch；
- 训练、验证和测试指向相同的 10 条 manifest 记录，只改变随机种子。

记录结果：

| 设置 | AABB avg ↓ | AAAA avg ↓ | AABB pose ↓ | AABB human ↓ |
|---|---:|---:|---:|---:|
| 完整三个 token | **0.0813** | **0.0154** | 0.0451 | **2.74e-4** |
| 去掉语义 token | 0.1071 | 0.0343 | 0.0185 | 1.30e-3 |
| 去掉相机对齐 token | 0.1021 | 0.0385 | 0.0185 | 1.10e-3 |
| 去掉时间上下文 token | 0.0924 | 0.0305 | **0.0120** | 7.57e-4 |

可以得出的结论：完整表示在综合 correction loss、连续序列上的不过度修正以及人体分支 loss 上更稳。

不能得出的结论：完整表示并非每个分项都最低；尤其删除 token 后 AABB pose loss 更低。由于样本复用，该结果不能证明 held-out 泛化。

权重审计：原来的各删除版本 checkpoint 和结果级 JSON 已不在仓库中；目前只找到配置、启动脚本和汇总记录。`checkpoints/v9_small10_pose_human_lora_baseline` 也只剩 Hydra 配置与日志，没有可用于重跑对照的正式权重。因此这些数字不宜直接作为论文正式表格。

### 2.2 V9 早期旧 benchmark

`IMPROVEMENT_EXPLORATION_SUMMARY_20260705.md` 还记录：

| 设置 | Camera translation ↓ | Camera rotation ↓ |
|---|---:|---:|
| 完整三个 token | 0.1543 m | 3.86° |
| 去掉语义 token | 0.5447 m | 7.51° |
| 去掉相机对齐 token | **0.1215 m** | **2.64°** |
| 去掉时间上下文 token | **0.0968 m** | 4.45° |
| 单 token | 0.1758 m | 5.16° |
| learned pooling | 0.4015 m | 6.08° |

该记录说明语义信息和多 token 分解可能有价值，但完整表示仍不在所有单项上最好。由于缺少完整协议、样本数、结果文件和 checkpoint，该表只能作为内部设计历史，不应出现在投稿论文中。

### 2.3 V14.1 单事件上限实验

关键记录：

- `versions/v14/docs/V14_1_INITIAL_PILOT_RESULTS_20260727.md`
- `versions/v14/docs/V14_1_ONE_SHOT_SHADOW_CORRECTION_TRAINING.md`

在一个 132.9° 的单事件 overfit 上：

| 设置 | Camera translation ↓ | Camera rotation ↓ | Human translation ↓ |
|---|---:|---:|---:|
| 简化 V14.1 | 0.082 m | **0.13°** | 0.0126 m |
| V9-parity | **0.048 m** | 0.25° | **0.0050 m** |

这不是有效的 token 消融。简化版本除了删除时间 token，还同时关闭 reliability、改变 correction gate/human gate，并改变 context 上的 LoRA 行为。原记录也明确说明差异不能整体归因于 momentum。该结果不应在论文中用于证明第三个 token 的必要性。

### 2.4 EgoHumans Formal-90 推理时屏蔽

关键记录：

- `output/bridge3r_egohumans_ablation_v1/formal90_final/internal/formal90_ablation_summary.json`
- `output/bridge3r_egohumans_ablation_v1/formal90_final/internal/FORMAL90_ABLATION_SUMMARY.md`
- `versions/v19/egohumans/run_formal_ablation_queue.py`

协议：固定 90 个片段、27 个 capture、同一 checkpoint、同一 detector stream，只在推理时改变 token 输入。

主要结果：

| 设置 | W ↓ | WA ↓ | Seam ↓ | IDF1 ↑ |
|---|---:|---:|---:|---:|
| 仅语义 token | 879.2 | 342.5 | 1.205 | 0.560 |
| 仅相机对齐 token | **853.5** | **323.9** | **1.151** | **0.574** |
| 语义 + 相机对齐 | 867.9 | 333.5 | 1.220 | 0.571 |
| 完整三个 token | 866.8 | 336.9 | 1.227 | 0.571 |

该结果不能证明完整三个 token 最好。捕获级配对 bootstrap 也显示，完整模型相较 alignment-only 的 WA 明显更差，而多数其他差异区间跨零。它只能说明冻结模型对 token 组合的响应具有数据集和指标依赖性。

当前稿件中存在 `manuscript/artifacts/egohumans_formal90/formal90_token_ablation.tex`，但正文和补充材料没有 `\input` 它，因此当前 PDF 实际未展示该表。若未来使用，标题必须从 `Correction-token ablation` 改成 `Inference-time token sensitivity`。

### 2.5 新 MVHuman Extreme-10 推理时屏蔽

关键记录：

- `output/bridge3r_mvhuman_token_sensitivity_extreme_v1/aggregate.json`
- `output/bridge3r_mvhuman_token_sensitivity_extreme_v1/aggregate.csv`
- `output/bridge3r_mvhuman_token_sensitivity_extreme_v1/MVHUMAN_EXTREME_TOKEN_SENSITIVITY_REPORT.md`

协议：10 个未参与训练的 MVHuman extreme 样本，159.7°--165.1°，每例 150 帧；固定真实边界以隔离 detector 影响；同一最终 checkpoint，仅做推理时 leave-one-out 屏蔽。

| 设置 | Anchor ↓ | Root ↓ | Seam ↓ | Camera rotation ↓ | Camera translation ↓ |
|---|---:|---:|---:|---:|---:|
| 完整三个 token | 295.6 | 240.2 | **516.3** | **142.5** | 2.065 |
| 去掉语义 token | **238.8** | **180.1** | 541.8 | 153.6 | 2.131 |
| 去掉相机对齐 token | 263.8 | 205.4 | 531.9 | 144.6 | **2.049** |
| 去掉时间上下文 token | 290.0 | 233.3 | 519.2 | 142.7 | 2.057 |

完整表示在 Seam 和 camera rotation 上均最低，并且 Seam 分别在 9/10、8/10、8/10 个配对样本上优于三种 leave-one-out 设置。语义与相机对齐 token 的影响较清楚，时间上下文 token 的影响较小。Anchor/Root 指标则呈现相反趋势。

该结果可以作为 held-out 大视角敏感性证据，但必须说明它是固定 checkpoint 的 inference-time masking，并使用受控真实边界；不能称为 independently retrained ablation。

## 3. 为什么旧实验和现在看起来不一致

1. **所问问题不同。** V9 测量各结构经过独立训练后在小训练域内的拟合与稳定性；EgoHumans/MVHuman 测量完整模型在推理时突然缺少某类已训练输入后的敏感性。
2. **数据不同。** V9 是 10 个四帧训练样本，并包含 AABB/AAAA；MVHuman 是 150 帧 extreme cut；EgoHumans 是更复杂的真实多人片段。
3. **指标不同。** V9 的 aggregate loss同时惩罚跨镜头误差和连续序列过度修正；新测试分别报告世界轨迹、接缝和相机误差。这些目标不保证同向变化。
4. **第三个 token 的边际作用本来就较弱。** V9 的 no-momentum 已是三个删除版本中最接近 full 的；MVHuman 也只观察到 2.8 mm Seam 和 0.22° camera-rotation 的小幅变化。
5. **完整模型追求的是跨镜头关系而非每个局部人体指标。** 更好的相机/接缝关系不保证 first-shot-anchored human trajectory error 同时降低。

## 4. 推荐的论文写法

### 4.1 Method：解释信息分工，不宣称三个都必不可少

推荐英文：

> To expose distinct boundary cues without prematurely pooling them into a single summary, the history-conditioned path uses three typed representations: a semantic token compares current observation evidence with the preceding memory, a camera-alignment token encodes the latent camera change, and a temporal-context token summarizes the preceding correction. These tokens are decoded jointly with the native image, camera, and human representations to estimate the temporary boundary transform.

对应中文：

> 为了在过早汇聚为单一表示之前保留不同类型的边界线索，历史条件路径采用三种带类型的表示：语义 token 比较当前观测证据与先前记忆，相机对齐 token 编码相机潜变量变化，时间上下文 token 汇总此前的校正信息。它们与原有图像、相机和人体表示共同解码，用于估计临时的镜头边界变换。

这里应避免：

- `Each token is indispensable.`
- `All three tokens are necessary for accurate reconstruction.`
- `The complete token set consistently outperforms all alternatives.`

### 4.2 Main experiments：最多只作一句指向补充材料

推荐英文：

> We further examine the internal boundary representation in the supplement. Fixed-checkpoint masking shows that the complete token set gives the lowest boundary-seam and camera-rotation errors on the extreme-view MVHuman subset, although the effect is not uniform across all trajectory metrics.

对应中文：

> 我们在补充材料中进一步分析内部的镜头边界表示。固定 checkpoint 的输入屏蔽结果表明，完整 token 集合在 MVHuman 极大视角子集上取得最低的边界接缝和相机旋转误差，但这种改善并未一致延伸到所有轨迹指标。

该句不应出现在 Contributions 或摘要中。三 token 不是论文最需要被记住的贡献。

### 4.3 Supplement：使用“敏感性分析”，不使用“正式消融”

建议只展示 `full / w/o semantic / w/o camera alignment / w/o temporal context` 四行，避免同时混入 `semantic only / alignment only` 而使表格过长。PA-MPJPE 在所有设置下相同，应删去该列。

推荐标题：

> **Inference-time sensitivity of the boundary representation.**

对应中文：

> **镜头边界表示的推理时敏感性分析。**

推荐表注英文：

> Frozen-checkpoint token sensitivity on the 10 extreme-view MVHuman cases (159.7°--165.1°). Each variant masks one token input at inference without retraining. Annotated transition timing is used only to isolate the alignment representation from transition detection. Lower is better.

对应中文：

> MVHuman 中 10 个极大视角样本（159.7°--165.1°）上的固定 checkpoint token 敏感性分析。每个变体仅在推理时屏蔽一种 token 输入，不进行重新训练。标注的镜头切换时刻仅用于将对齐表示与切换检测的影响分离。所有指标均为越低越好。

推荐结果分析英文：

> The complete representation yields the lowest boundary-seam and camera-rotation errors under every leave-one-out mask. Removing the semantic or camera-alignment token has the largest effect, whereas removing temporal context produces only a small change. The trend reverses for first-shot-anchored human errors, indicating that the typed representation primarily stabilizes the inter-shot relation rather than uniformly improving every downstream metric. We therefore interpret the three tokens as complementary boundary evidence, not as components that individually dominate all metrics.

对应中文：

> 与三种 leave-one-out 屏蔽相比，完整表示均取得最低的边界接缝和相机旋转误差。删除语义或相机对齐 token 的影响最大，而删除时间上下文只带来较小变化。首镜头锚定的人体误差呈现相反趋势，说明这种带类型的表示主要用于稳定镜头间关系，而不是一致改善每一个下游指标。因此，我们将三个 token 解释为互补的边界证据，而不声称每个组件都在所有指标上占优。

## 5. 是否应把 V9 数字写入论文

不建议。理由：

1. train/val/test 使用相同 10 条样本；
2. 不同设置记录的 epoch 不完全一致；
3. 原 checkpoint 与结果级 JSON 已缺失，无法按当前协议复核；
4. AABB/AAAA loss 是内部训练指标，不如当前世界坐标、接缝与相机指标直观。

V9 数字应保留在项目实验审计文档中，作为选择三 token 结构的开发依据。如果必须在 rebuttal 中说明，可以称为 `an early independently trained in-domain probe`，同时主动说明其规模和用途，但不应包装成主要实验。

## 6. 最终建议

当前时间有限时，最合理的处理是：

1. 保留 Method 中三 token 的公式和信息分工；
2. 将论述从“必须是三个”收窄为“以三种带类型表示保留互补边界证据”；
3. 不把 V9 小样本数字放进论文；
4. 在补充材料加入 MVHuman leave-one-out 表，但明确标为固定 checkpoint 的 inference-time sensitivity；
5. 正文最多增加一句指向该分析；
6. 不在摘要、标题或 Contributions 中强调 three-token design；
7. 若之后有训练预算，再从共同初始化分别训练 `full / w/o semantic / w/o camera alignment / w/o temporal context`，届时用正式结果替换敏感性表。

