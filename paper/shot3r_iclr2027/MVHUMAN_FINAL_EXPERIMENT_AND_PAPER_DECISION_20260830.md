# MVHuman Test1/Test2 最终实验审计与投稿决策

**日期：** 2026-08-30  
**状态：** 数据与正式协议完成；PromptHMR 正式实验完成；GVHMR pilot 按预注册门槛停止；MVHuman 不作为 BRIDGE3R 的正文正向证据。  
**原则：** 固定样本、不替换失败案例、不从 Test 选择阈值、不把 evaluator-timed control 写成自动方法。

## 1. 数据结论

- 接收归档为 `data/Mvhuman-test1.zip`（8,876,412,777 bytes）和
  `data/Mvhuman-test2.zip`（3,614,782,728 bytes）。两个 ZIP、内部十个
  `tar.gz` 和归档成员均通过完整性及路径安全检查。
- 共十个 capture：`100021--100025` 与 `200021--200025`，包含同步 RGB、
  人体 mask、相机内外参、SMPL-X、3-D joints 和 mesh；总计 131,600 张 RGB。
- correction-module fine-tuning 使用的是 `100001--100005` 与
  `200001--200005`。新数据在 capture、event 和 frame/member 层面无交集，
  但仍共享 MVHuman 数据域和原相机 rig。
- 因此正式表述只能是：

  > ten held-out MVHuman captures whose capture and frame members are
  > disjoint from the correction-module training manifests, while retaining
  > the dataset's original camera rigs

- 不得写成 unseen source、unseen rig、unseen subject、官方 test split，或
  未出现在继承 backbone 预训练语料中的全新数据。
- 缺少 scene depth/mesh GT，适用于 Camera--Human 定量，不适用于 scene
  reconstruction 定量。
- 可称为 `controlled weak-texture studio setting`。正式 low-texture benchmark
  需要高纹理对照与预定义阈值；目前只保留已计算的背景梯度和 Laplacian
  统计，不作更强外推。

完整数据审计见 `MVHUMAN_TEST12_DATA_AUDIT_20260830.md`。

## 2. 冻结协议

- 正式协议为 50 cases，而非早期计划的 40 cases：十个 capture 各包含
  small、medium、large、very-large 和 extreme 五个 SO(3) 角度档。
- 每例 150 帧：前后两个物理相机各 75 帧，切换发生在输出帧 74/75 之间。
- 相机旋转覆盖约 6--180 度。所有方法使用完全相同的 50-case manifest、
  RGB、帧序、GT evaluator 和固定分母。
- 冻结协议：
  `Movie3R/output/bridge3r_mvhuman_v1/protocol_freeze/protocol_freeze.json`
- 文件 SHA-256：
  `15e47e58cbe4287609f1b1628f95176c5063dee7e9a17ef4d7c18d52ff4e356f`

## 3. 自动运行的正式内部结果

以下为 capture-macro，人体误差单位为 mm、旋转为 degree、平移为 m。

| Method | PA-MPJPE | Anchor-MPJPE | Seam-root | Seam-orient. | Camera rot. | Camera trans. | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 27.5 | 217.5 | 442.3 | 88.9 | 95.0 | 1.733 | 99.8% |
| BRIDGE3R, automatic event route | 27.6 | 261.0 | 434.5 | 89.0 | 87.7 | 1.897 | 99.8% |

与 Strict Human3R 相比，自动 BRIDGE3R 的 Seam-root 改善 1.8%，Camera
rotation 改善 7.6%；但 Anchor-MPJPE 变差 20.0%，Camera translation 变差
9.5%，PA 基本持平。十个 capture 的 paired bootstrap 区间整体跨零，不能
写成统计可靠的全面领先。

正式聚合：
`Movie3R/output/bridge3r_mvhuman_v1/internal/formal_aggregate.json`  
文件 SHA-256：
`42098d8db328ba4d217a762ca8d12913031eed09707e2d86364296bdb4a14e9b`

## 4. 自动 detector 完整性结论

- GT boundary 在所有 50 例均为 75；自动 first-positive 仅 10/50 恰好为
  75，其余 40/50 在同一 shot 内提前触发。
- detector 的逐帧汇总为 TP=50、FP=399、FN=0，precision=0.111、
  recall=1.000、F1=0.200。
- `run_case.py` 将 first-positive 直接传入 `run_transaction`，并在该位置执行
  clean reset、coarse gauge、association 和 shared translation。故提前触发
  不是日志或索引记账问题，而是实际改变了推理输出。
- 不能只修改论文文字称“自动触发”，也不能以 GT boundary 静默替换正式
  自动运行。若未来重做自动结果，须在独立 detector train/validation 上冻结
  新 detector，再在同一 50-case Test 上完整重跑；不得从本 Test 选择阈值。

## 5. PromptHMR 正式 50-case 结果

PromptHMR 官方 offline full-video 路线完成 50/50，失败 0，Coverage 与
completion 均为 100%，并通过 native output、SPEC camera、conversion、
固定分母、代码和 checkpoint 哈希审计。

| Method | PA-MPJPE | Anchor-MPJPE | Seam-root | Seam-orient. | Camera rot. | Camera trans. | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| PromptHMR, official offline | 28.5 | 137.6 | 20.3 | 53.6 | 103.9 | 1.789 | 100.0% |

正式 ledger：
`Movie3R/output/bridge3r_mvhuman_v1/prompthmr_formal50/audit/formal_ledger.json`  
文件 SHA-256：
`0ef775fff8c5e1535ab630cedf7285ac8612de951372984babc2cb7031a82a0c`

该路线可读取未来帧，不能与 BRIDGE3R 写成同信息条件的 causal 排名；但在
Anchor 和 Seam 指标上明显更强。因此 MVHuman 也不能被包装成 BRIDGE3R
优于所有公开单人方法的证据。

## 6. GVHMR 和 TRACE 的准入决定

- GVHMR 使用预注册 12-case pilot。十例存在可恢复的官方 native output，
  两例在官方 SimpleVO 产生 native reconstruction 前失败，completion 为
  10/12=83.3%，未通过全量扩展门槛。
- 失败案例保留在固定分母，未替换、未以插值结果掩盖、未扩展到 50 cases。
- gate 文件：
  `Movie3R/output/bridge3r_mvhuman_v1/gvhmr_official/gvhmr_pilot_gate_failed_final.json`
- 文件 SHA-256：
  `d1d0f61ba77eae44cc542a6edc63f77eb9dbf358d03a21409289ff076407b283`
- TRACE 没有与本 Camera--Human 协议一致的、可验证的物理相机轨迹输出。
  它在既有多人实验中保留为人体轨迹/coverage reference，但不进入本数据集
  的 camera leaderboard。由于 MVHuman 已未通过内部正向证据门槛，额外
  运行 TRACE local-only 结果不会回答本轮的 Camera--Human 研究问题，故停止。

## 7. Evaluator-timed control

为区分 detector failure 与 alignment failure，运行了同一 capture、五个角度
档的 evaluator-timed control。该控制显式接收 event=75，未来帧访问仍为零，
但它不是自动部署结果。

| Method/control | PA-MPJPE | Anchor-MPJPE | Seam-root | Seam-orient. | Camera rot. | Camera trans. |
|---|---:|---:|---:|---:|---:|---:|
| Strict Human3R | 27.9 | 297.2 | 322.8 | 99.6 | 106.0 | 1.458 |
| Clean reset at true boundary | 27.9 | 224.8 | 239.2 | 97.5 | 106.0 | 1.623 |
| Full BRIDGE3R at true boundary | 27.9 | 451.8 | 676.1 | 79.1 | 72.2 | 2.249 |

正确 boundary 能改善 orientation/camera rotation，但 Anchor、Seam-root 和
translation 仍明显退化。因此停止扩展到 50 例。该结果只用于内部诊断；若
公开，必须位于 ablation/control，而不能进入 baseline ranking。

控制输出：
`Movie3R/output/bridge3r_mvhuman_v1/internal_timed_control_pilot5`  
run-summary SHA-256：
`376fbd4eec2e353eb27c8a9f9bd63c4df80b81eb6501014480340fd171361207`

## 8. 最终投稿决策

1. **不将 MVHuman 升级为正文第四个 benchmark。** 当前证据不支持
   BRIDGE3R 在该单人弱纹理域的全面优势，而且自动 detector 的 end-to-end
   有效性门槛未通过。
2. **不选择性只报告 Camera rotation。** 这样会隐藏 Anchor、Seam 和
   translation 的反向结果，也会把错误事件路由的输出误解成真实切镜响应。
3. **正文继续使用三个多人数据集。** EgoBody、EgoHumans、Harmony4D 的
   严格配对结果和 extreme/farthest 分析直接对应论文的多人 Camera--Human
   主张；AIST++ 只保留单人迁移压力测试。
4. **MVHuman 结果只进入 Supplement 的压力审计。** v029 完整报告自动
   50-case 内部表、PromptHMR 正式表、detector first-trigger failure 和 GVHMR
   availability gate；不报告 timed-control pilot，也不把任何一列升级为
   BRIDGE3R 的正向主张。
5. **若后续必须恢复 MVHuman 正向证据，需重新开发而非修饰文字：**
   使用独立 train/validation 冻结 shot detector；针对单人场景重新训练或
   验证 coarse/fine alignment；随后在原 50-case manifest 上一次性重跑，并
   同时报告 Anchor、Seam、camera translation、rotation 和 Coverage。

## 9. 与论文版本的关系

论文的科学主线、九页正文、真实大视角定性图和可编辑 SVG 已在 v028 完成。
v029 保留三多人证据链作为正文中心，并将 MVHuman 作为明确限定的 Supplement
failure/stress audit；这不是正向 benchmark。Evaluator-timed pilot 仍只留在本
私有文档中，避免后续把自动结果、timed control 或 pilot 条件均值混淆。
