# Movie3R 长时任务完成报告：因果 detector、策略与 V9 决策

日期：2026-08-05  
状态：**模块冻结，可进入大规模论文指标阶段**

本报告承接 `ADAPTIVE_AND_V9_RESEARCH_PLAN_20260805.md`，记录本轮真正执行的验证、失败分支和最终保留的 ICLR 主线。运行时推理不读取 GT；GT 只出现在离线训练标签或评估器中。

## 1. 工作区冻结

工作区已清理，没有未提交文件。按功能保留并提交了所有可复现实验资产：

| Commit | 内容 |
|---|---|
| `c326e58` | V9 decoder/token 探针、可恢复缓存、最终报告 |
| `ec9a4f0` | ICLR blueprint、边界几何与 camera-safety 协议、pair-disjoint manifests |
| `1d298eb` | camera-human boundary evaluator、人体锚点与 demo payload 工具 |
| `4fe285a` | B0 safety、显式 residual、VSP/dual-proposal 研究探针 |

当前最新主线仍是此前冻结的 `8e058a1`/`f646453` 之后版本：B0+BRTC+C1 与 adaptive joint 没有被实验分支覆盖。

## 2. 最终模块

### 2.1 在线流程

```text
RGB 流
  → causal image detector（当前帧和有限历史）
  → Human3R/V9 的 B0 粗 gauge proposal
  → BRTC-LC + C1 人体局部稳定
  → 边界人体几何 gate
       ├─ residual 小/匹配不可靠：exact B0+BRTC+C1 fallback
       └─ residual 大且可信：共享 Kabsch + root-ray 的 camera-human joint update
  → post shot 持续复用一次修正，pre 帧完全不变
```

### 2.2 Detector

新增了 `versions/v14/train_causal_detector.py` 与 `causal_image_detector.py`。输入是相邻 RGB 帧的廉价统计：RGB/灰度变化、颜色直方图、光流、ORB 匹配等；明确排除了 GT 相机角度、SMPL、Human3R 输出和未来帧。模型只保留长度为 3 的 pair-feature 历史，输出当前帧 `p_cut`。

四源、按 source 留出验证（408 pairs）的结果：

| 模型 | macro F1 | macro FPR | macro Brier |
|---|---:|---:|---:|
| temporal-feature logistic | 0.926 | 0.151 | 0.074 |
| causal MLP | 0.953 | 0.057 | 0.037 |
| causal GRU（选中） | **0.982** | **0.031** | **0.015** |

选择规则是 held-out F1 不下降、FPR 不上升且校准误差不恶化。完整审计见 [REPORT.json](/data/wangzheng/iJCV-CODE/Movie3R/output/v14/detector_learning_audit/REPORT.json)，模型 artifact 为 `output/v14/detector_learning_audit/SELECTED_MODEL.pt`。

在 AvatarReX 30 帧低纹理序列和多人 30 帧序列上，GRU 与旧 logistic 都只预测第 5 帧，未产生额外边界。GRU 是可选接入：`streaming_detector_joint_boundary.py --detector-model ...`；不传该参数时保留历史 logistic baseline。

### 2.3 自适应策略结论

尝试过的学习/混合策略包括 B0 abstention tree、dual-B0 selector、固定 SE(3) mixture、VSP root-agreement 和显式 residual head。它们共同遵守 pair-disjoint development、source-group CV、错误更新高惩罚和 exact fallback。

结论是：学习模块可以可靠地判断“是否可能是 cut”，但目前没有证据证明它能安全地直接回归世界坐标 residual。显式 residual head 的重跑结果为 No-Go：最佳 MLP 在 dev 上平均 composite 仍下降 9.44%，catastrophic count 反而增加 8.16%；其余候选更差。因而最终策略保留“可学习 detector + 可解释几何 gate”，不把不安全的 residual regressor 放进主线。

## 3. V9 严格验证结论

V9 token 深度/读取方式的 robust re-audit 使用 384 个四源训练 cuts、10 个冻结评估 cuts、4-fold source/pair grouped CV。结果再次支持：decoder L11 pose token 是最有效的隐式 correction carrier；早期 DINO/CUT3R token、手工 correct-token mean 和多层拼接都没有稳定增益。

正式 V9 的收益明显依赖数据源：AvatarReX/THuman 有改善，MVHuman 接近中性。175k pose-relation head 虽把 frozen mean composite 从 1.862 降到 1.624、P90 从 4.894 降到 3.945，却在 MVHuman200 产生新的 catastrophic case（0.511 → 3.871），因此不晋升。

最终定位：**V9 保留为 B0 的隐式粗对齐/人体先验，不宣称其是通用低纹理相机校正器。** 低纹理相机失败由后续人体锚点联合几何模块处理；这比继续扩大 token 或盲目重训 V9 更安全，也与已有 decoder evidence 一致。完整证据见 `versions/v9/decoder_correct_token_probe/FINAL_REPORT.md` 与 re-audit JSON。

## 4. 当前几何模块证据

AvatarReX 单人低纹理 30 帧案例中，人体 gate 接受 66.27° residual，联合 shadow root-ray 后首 post 相机误差约 0.011 m / 0.40°，25 帧平均约 0.054 m / 0.44°；B0+BRTC+C1 原始结果约 1.697 m / 66.51°。三人高纹理四个案例的 residual 为 3.12°–10.25°，全部拒绝修正并 exact fallback，因此不会破坏原本相机已经正确的多人场景。

这给出最终论文最重要的行为分解：

1. 高纹理多人：背景/B0 已可靠，人体局部 BRTC/C1 修正，adaptive joint abstains；
2. 低纹理单人：背景不可靠，人体成为锚点，joint solver 同时纠正 camera 和 human；
3. 证据不足：宁可保持 baseline，也不提交危险的全局 gauge。

## 5. Multi-THuMBS-style 当前对标

官方 Multi-THuMBS supplementary/evaluator 未公开，因此本项目只报告透明的本地 provisional protocol，不能当作官方复现。当前 cross96 EgoHumans 5 chains/10 cuts 的同一 checkpoint 结果：

| 方法 | W-MPJPE (mm) | WA-MPJPE (mm) | pelvis MPJPE (mm) | pelvis MPVPE (mm) | ATE (m) | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| raw reset | 1155.6 | 388.9 | 129.1 | 152.4 | 1.788 | 28 |
| B0 | 443.7 | 258.2 | 129.1 | 152.4 | 0.198 | 6 |
| B0+BRTC-LC | **434.4** | **257.1** | 129.1 | 152.4 | **0.198** | **6** |

多人 P0 checkpoint 的 B0+BRTC-LC 为 W-MPJPE 329.3 mm、WA-MPJPE 222.6 mm、ATE 0.162 m、ID switches 6。论文表中的 279/166 mm 等数字只作为 reference，因数据 split、visibility、matching 和公式未知，暂不能宣称可直接比较或已经超过论文。

## 6. 最终 ICLR 模块与论文主张

最终可写成：

> **Causal Adaptive Camera–Human Gauge Correction**：在 camera cut 处把旧 recurrent state 仅作为只读 proposal，使用因果 detector 提供事件先验，再用人体/背景可观测性 gate 选择是否提交一个可解释的共享 SE(3)；相机与人体只有在同一个可信 gauge 更新中同步改变，否则 exact fallback。

创新点不是“又训练一个大 encoder”，而是：

- state ownership 与 world-gauge ownership 的 transactional separation；
- detector、geometry verification、abstention 的因果自适应组合；
- 低纹理时的人体锚点联合 camera-human correction，并对高纹理多人保持零破坏 fallback；
- 在不依赖额外大型预训练模型和离线 bundle adjustment 的情况下保持在线执行。

明确限制：不专门解决完全不可见人体、严重遮挡下的全新 Re-ID，也不声称复现 Multi-THuMBS 的离线全序列优化。

## 7. 失败记录与下一阶段入口

- 显式 learned SE(3) residual：pair-disjoint dev No-Go，已记录并冻结为 ablation；
- VSP/dual-B0 mixture：没有同时改善均值、尾部和所有 source，已关闭；
- Multi-THuMBS self-test 在当前公共 shell 因缺少 `gsplat/smplx` 依赖失败；已有 GPU/CPU 生成的 B0+BRTC report 可复核，环境修复后只需重跑 evaluator，不应改变方法结论；
- 当前 detector GRU artifact 尚未替换 demo 默认 logistic，避免无 artifact 时改变旧行为；论文主实验应固定 artifact checksum 后再切换到 GRU。

至此，方法模块和决策已冻结。下一阶段不再做开放式架构搜索，直接进行 detector artifact checksum 固定、AvatarReX/THuman/MVHuman/MultiHuman 大规模分层测试，并按 Multi-THuMBS 透明 provisional 指标导出论文表格。
