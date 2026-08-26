# Harmony4D 保留、重放与下一步计划（2026-08-25）

## 目的与原则

Harmony4D 已完成正式的单次视角切换主结果，并完成一个仅作补充
证据的四 capture、八 boundary 多切换控制。本文件规定在交替上传数据集的
磁盘约束下，哪些资产必须保留、哪些可以在验证后清理，以及下一轮实验的
优先顺序。

**原则：** 不复制或展开完整的 328 GB 数据集。主结果所需的冻结预测、
manifest 和汇总文件与原始数据分开保留；需要重放多切换实验时，只恢复其
冻结的四个 capture。本文档不授权删除任何数据。

## 冻结的正式单切换结果（不得删除）

| 项目 | 位置 | 作用 |
|---|---|---|
| 原始数据 | `data/Harmony4D.zip`（328 GB） | 在完成 Harmony4D 的新增 evaluator-only 实验、全量消融或正式 replay 包前，仍是唯一完整输入来源。 |
| 正式汇总与 evaluator report | `output/v17_harmony4d/unified_half_translation_audit/`（约 11 MB） | 论文中的 Harmony4D-CS150、共同可评 `N=88` 结果的汇总、per-case evaluator report 与审计 metadata；它本身不存放大 prediction cache。 |
| 正式 BRIDGE3R prediction cache | `output/v17_harmony4d/full_test/predictions/`（约 43 GB） | 统一 audit 读取的正式 BRIDGE3R cache；不得删除。 |
| 正式内部参考 prediction cache | `output/v15_harmony4d/predictions/test_*`（约 19 GB） | 统一 audit 的 Strict Human3R / reset-only 等内部参考输入；`06_sword3_a` 与 `06_sword3_b` 均被引用，不得删除。 |
| 正式汇总 | `output/v17_harmony4d/unified_half_translation_audit/paper/summary.json` | 由 `PAPER_METHOD_LOCK.json` 绑定；SHA-256：`51ebde2f0ca2d70e54bd3ba948ab4ea47bc2821aabc6dd34b4ec34728520b7da`。 |
| 历史全量报告 | `versions/v17/HARMONY4D_V17_FULL_TEST_RESULT_MANIFEST_20260820.json` | 记录原始 v17 全量运行；不替代统一正式路径。 |
| 主论文结果绑定 | `publication/bridge3r_iclr2027/PAPER_METHOD_LOCK.json` | 锁定 checkpoint `de2430ed...828265`、因果 detector 与正式结果摘要。 |

完整数据包的 SHA-256 尚未计算；在磁盘空闲、且不与大规模数据上传或 GPU
读取竞争时再计算。当前应至少将 `Harmony4D.zip` 视为不可修改的完整来源。

## 多切换重放保留包（已制作，待最终校验）

原先的 staging 目录为四个父 action archive 的完整展开，实际占用约 52 GB；
其中大部分 capture 不属于冻结的 multi-cut protocol。已从其中抽取并打包下列
**四个完整 capture 目录**，以支持完全相同的 RGB 推理与 evaluator-only 评测：

| source archive | capture | shot cameras | frames | boundaries |
|---|---|---|---:|---:|
| `train/10_karate2.zip` | `10_karate2/001_karate2` | `cam18, cam15, cam19` | 150 | 50, 100 |
| `train/11_karate3.zip` | `11_karate3/001_karate3` | `cam18, cam15, cam19` | 150 | 50, 100 |
| `train/14_mma3.zip` | `14_mma3/008_mma3` | `cam01, cam11, cam20` | 150 | 50, 100 |
| `train/15_mma4.zip` | `15_mma4/006_mma4` | `cam01, cam11, cam20` | 150 | 50, 100 |

| 项目 | 位置 | 状态 |
|---|---|---|
| 精简重放包 | `data/Bridge3R_harmony4d_retention/Harmony4D_multicut_replay_v1.tar.gz`（5,540,952,591 bytes） | SHA-256：`a2ea8faa9e810673b8c63030e7b8a828e9d2a7182c45825f52831195f9948682`；`*.sha256` 伴随文件已生成，`*.tar-list-ok` 已确认 gzip/tar 可完整列出。它现在可作为旧 staging 的替代品。 |
| 冻结 multi-cut manifest | `publication/bridge3r_iclr2027/multicut/manifests/harmony4d_multicut_v1.jsonl` | 必须保留。 |
| no-cut control manifest | `publication/bridge3r_iclr2027/multicut/manifests/harmony4d_nocut_v1.jsonl` | 必须保留。 |
| 推理/评测报告 | `publication/bridge3r_iclr2027/multicut/runs/` 与 `aggregate_v1.json` | 必须保留；该目录约 793 MB。 |
| 旧完整 staging | `data/Bridge3R_multicut_harmony4d/staging/`（约 52 GB） | **暂不删除。** 重放包校验通过且作者确认后，可清理以净释放约 46--47 GB。 |

上述多切换结果仅能表述为四个预注册 capture、八个 hard-cut boundary 的补充
控制；不能代替三数据集单切换主表，也不支持跨物理场景的结论。

## 其他 Harmony4D 占用的处置顺序

| 资产 | 当前约占用 | 当前处理 |
|---|---:|---|
| `output/v18_harmony4d/` | 34 GB | 首选的候选清理项，但要先逐项核对它没有被正式论文、multi-cut 或后续消融脚本引用。 |
| `output/v16_harmony4d/` | 1.6 GB | 较小；先保留到 v17/论文表格重放路径被独立验证。 |
| `output/v15_harmony4d/` | 30 GB | 其中约 19 GB 是被正式统一 audit 直接引用的 `predictions/test_*`；其余开发/定性内容须在逐目录引用核对后才可考虑清理。 |
| `data/Harmony4D_work_v17_full_test/` | 16 GB | 含主结果的 staging/metadata；在正式 replay 资产重建前不删。 |

任何清理都应在本文件对应条目更新为“已替代、已校验”后执行，且只清理明确
列出的单一路径，绝不使用宽泛的递归删除命令。

## 建议的后续实验顺序

1. **Harmony4D 的直接关联证据已完成；保留脚本与输入供跨数据集复用。**
   `evidence/harmony4d_boundary_association/formal_v1/final_v1.json` 已对
   88 个最终绑定 case 完成 evaluator-only 的 first-post-cut
   correspondence、runtime abstention 与 framewise oracle-ID upper bound。
   该项对应 D09 的 Harmony4D 部分；EgoBody/EgoHumans 恢复后必须以同一定义
   重放，才可将它从补充材料的单数据集组件证据升级为跨数据集结论。
2. **准备但不抢跑完整模块消融。** 对 D31 的六级链式消融，先建立同一
   小 replay subset、固定 checkpoint 和 manifest；正式全量消融应优先在
   EgoBody/EgoHumans 恢复后统一执行，避免只凭 Harmony4D（它在 W-MPJPE
   上是 mixed result）支撑核心模块结论。
3. **准备 Harmony4D 的相机指标分解。** 先检查现有 evaluator 能否从缓存
   无重推导出 ATE-Sim(3)、ATE-SE(3)、rotation、translation、scale 的
   一致分解。只有定义和复现核对通过才填入补充材料；否则维持 ATE-Sim3
   与显式 `TODO-EXPERIMENT`。该项对应 D23。
4. **待 EgoBody 恢复后优先完成跨数据集证据。** recording-level paired
   bootstrap/permutation、win rate 与 median improvement（D11），随后在
   development-only subset 上做 `lambda=0.5` sweep（D03）。这两项的论文
   价值高于新增 Harmony4D 试跑。
5. **最后统一做运行时/内存与 release replay。** 标准化 GPU、分辨率、
   precision、batch size 的 runtime/memory（D08），并在三个数据集可用时
   复建匿名可重放包（D22/D26）。

## 与论文状态表的对应关系

本文件只管理数据资产和实验调度。论文决策仍以
`ICLR-paper/bridge3r_iclr2027/REVISION_DECISIONS_V6.md` 为准，完成状态仍
以 `ICLR-paper/bridge3r_iclr2027/REVISION_EXECUTION_STATUS_V6.md` 为准。多
切换保留包及校验完成后，应在状态表的 D12/D22 证据列补充其路径与 hash。
