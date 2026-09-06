# Shot3R / Movie3R 对比基线审计与实验建议

> 日期：2026-09-06（Asia/Shanghai）  
> 项目：`/data/wangzheng/iJCV-CODE/Movie3R`  
> 对应论文版本：`ICLR-paper/bridge3r_iclr2027/versions/v037_20260906_supplement_revision/manuscript/`  
> 文档目的：供另一位 AI 或合作者独立审阅，重点判断本文对 JOSH、JOSH3R、SLAHMR 及其他候选基线的定位是否合理。

## 1. 希望审阅者回答的问题

请不要只根据方法名称或论文摘要判断，而应结合公开代码、输入条件、时间访问方式和原生输出回答：

1. JOSH 是否应作为 Shot3R 的强离线全系统参考？
2. JOSH3R 是否有资格进入 EgoBody、EgoHumans、Harmony4D 的多人主表？
3. SLAHMR 官方实现是否能原生输出跨两个 hard-cut shot 的统一世界坐标结果？
4. 是否存在比 JOSH 更匹配、且当前真正可执行的 multi-shot multi-person 方法？
5. 如果计算资源只允许新增一个多人 baseline 和一个单人 baseline，应该选择谁？
6. 下述实验协议是否避免了 GT 泄漏、逐例调参和跨镜头结果的人工修复？

## 2. 结论摘要

当前建议是：

1. **P0：JOSH。**先做冻结的 12-case pilot；通过后仅先扩展 EgoHumans 90 例。它应作为 `offline full-sequence` 参考单独报告，不能与 Shot3R 的 causal streaming 结果混成一个公平排行榜。
2. **P0（代码可得时）：Multi-THuMBS。**它在任务定义上比 JOSH、SLAHMR 更接近 Shot3R；目前缺少可执行代码、完整 manifest 和 evaluator，因此应优先联系作者，而不是混用论文中的异协议数字。
3. **P1：AIST++ 增加一个强单人 baseline。**首选 WHAC，其次 GVHMR；JOSH3R 可作为额外实验性参考，但不建议把它作为最重要的单人对比。
4. **P2：SLAHMR 只做 1--3 例兼容性诊断。**若官方路径仍只返回单个 shot，则跨镜头 W/WA、完整相机误差、IDF1 和 seam 均记为 `N/A`，不再扩展。
5. **P3：GLAMR、WHAM、TRAM、Multishot。**仅作为 fallback、单人附录或 Related Work，不建议默认做三数据集全量实验。

一句话概括：**最缺的不是 SLAHMR，而是可执行的 Multi-THuMBS/HumanMM；在现有公开条件下，最现实且最有审稿价值的新增组合是“JOSH + 一个 AIST++ 单人 baseline”。**

## 3. Shot3R 的任务边界

以论文 v037 为准，Shot3R 研究：

- 按时间顺序到达的单目 RGB；
- 由 hard cut 连接的多个 shot；
- 多人 4D 重建；
- 统一世界坐标下的人体运动；
- 可独立评测的物理相机轨迹；
- 跨镜头持续身份；
- 首个 post-cut frame 到达时立即处理；
- 不访问未来帧，不回改历史结果，不做逐视频全局优化。

当前三个主要协议并不是任意跨场景电影剪辑，而是由同步相机数据构造的、同一物理场景中的 sequential shots：

- EgoBody：129 个固定 case；
- EgoHumans：90 个固定 case；
- Harmony4D：88 个固定 case。

因此，一个方法仅仅能做连续视频 world-coordinate HMR，并不等价于能完成这里的任务。主表方法还需要区分：

- temporal mode：causal / semi-online / offline；
- 是否输出独立物理相机；
- 是否显式处理 shot transition；
- 是否支持多人和 native identity；
- 两个 shot 是否处于同一个原生 world gauge；
- prediction coverage 和失败率。

## 4. 已有基线，不应重复认定为缺失

最新版主表已经包含：

- Human3R：同骨干、同在线递归范式的主要受控基线；
- OnlineHMR：三个多人数据集上的同输入外部参考；
- TRACE；
- PromptHMR-SPEC。

其中 Human3R 与 Shot3R 的差异集中在 cut 处的状态处理，因而仍然是最重要的受控对比。OnlineHMR、TRACE、PromptHMR-SPEC 则补充不同时间访问方式和不同输出可用性的外部参考。

注意：旧版 `BASELINE_AUDIT.md` 或旧实验计划中关于“OnlineHMR 尚未完成”的判断已经过时，应以 v037 主表为准。

## 5. 候选方法适配性总表

| 方法 | 多人支持 | 时间访问 | 原生跨镜头统一世界 | 独立相机轨迹 | 当前建议 |
|---|---:|---|---:|---:|---|
| JOSH | 是 | 离线全序列、逐视频优化 | 无显式 shot-aware 机制，但可尝试完整序列 | 是 | 最高优先级离线参考 |
| JOSH3R | 否；当前 demo 只读取第一条 TRAM track | 图像对推理，非严格零未来帧 | 仅适合单人实验性测试 | 当前保存接口不完整 | 仅 AIST++ 附录 |
| SLAHMR | 单 shot 内多人 | 离线序列优化 | 否；官方路径选择一个 `shot_idx` | 单 shot 有 | 1--3 例诊断，不全量 |
| Multi-THuMBS | 是 | 离线全序列 | 是，任务高度匹配 | 论文报告完整相机注册 | 若获得代码，优先级高于 SLAHMR |
| HumanMM | 否 | 离线 multi-shot integration | 是 | 论文报告有 | 代码开放后用于 AIST++ |
| WHAC | 否 | 完整视频 | 连续视频 world HMR，无显式 cut | 是 | 推荐补到 AIST++ |
| GVHMR | 官方 tracker 为 top-1 | 完整视频 | 无显式 cut | 可从 global pipeline 评测 | AIST++ 次选 |
| GLAMR | 是 | 离线全序列 | 无显式 shot 处理 | 有全局运动估计 | JOSH 失败时的 fallback |
| WHAM / TRAM | 单人或逐 track | 完整视频 | 无显式 shot 处理 | 有 | 优先级低于 WHAC/GVHMR |
| Multishot (CVPR 2022) | 公开 demo 偏单 track | 离线 | 关注跨镜头人体一致性 | 缺少独立物理相机输出 | 只适合局部/边界指标 |

## 6. 对 JOSH 的判断

### 6.1 为什么值得比较

JOSH 是当前最值得补充的强离线系统参考，因为它联合处理：

- 多个人体轨迹；
- 相机运动；
- 场景几何；
- 人体--场景接触；
- 完整序列优化。

它能回答一个对审稿人很有意义的问题：

> 即使允许查看完整视频、利用未来帧并进行逐视频优化，一个强联合优化系统在 hard cut 下能够做到什么程度？

这与 Shot3R 的优势来源不同，因此结果无论好坏都具有信息量：

- 如果 JOSH 更准，说明完整序列优化仍提供明显上界式参考；
- 如果 JOSH 在 cut 处失败，说明连续视频优化系统并不会自动解决镜头切换；
- 如果 JOSH 人体较好但 identity/camera seam 较差，可以进一步说明显式 boundary operation 的必要性。

### 6.2 为什么不能作为完全公平的 causal baseline

JOSH：

- 使用完整序列；
- 可访问未来帧；
- 进行逐视频优化；
- README 允许调整优化超参数；
- 没有针对 hard cut 的显式状态转换机制。

因此不能把 JOSH 与 Shot3R 混在同一 `causal online` 分区中直接宣称全面胜负。正确角色是：

```text
Offline full-sequence methods on the same RGB clips
```

并同时报告：

- temporal access；
- 是否逐视频优化；
- completion rate；
- runtime；
- W/WA 的有效支持数；
- camera、IDF1 和 coverage 的原生可用性。

### 6.3 执行风险

1. JOSH 虽然遍历多个 TRAM tracks，但跨 cut 的 tracker ID 可能不稳定；必须保留 native ID，不能人工合并。
2. JOSH 没有显式 hard-cut 机制，其 SLAM、matching 或人体轨迹模块可能在边界失败；这种失败必须进入完整分母。
3. 对 200 帧以上输入，当前实现会分 chunk 后进行较简单的拼接，并没有完整 global bundle adjustment。
4. 不允许根据每个 case 的输出效果单独调整优化超参数。
5. 官方仓库目前没有清晰的顶层 LICENSE；发表数值结果与重新分发修改代码是不同问题，后者应向作者确认。

### 6.4 关于 `max_frames`

旧版 baseline 计划曾建议把 `max_frames=21` 直接改到 150。根据对新版代码的重新核查，这个建议不应机械执行：其采样逻辑对 100/150 帧输入会覆盖完整原始时间跨度，但只选择较稀疏的关键帧。

正式实验前应先确认：

- 采样使用了哪些原始 frame indices；
- 优化结果如何插值或映射回完整 150 帧；
- 两个 shot 是否均有输出；
- 改变 `max_frames` 是否只是输出密度变化，还是会改变优化预算和结果质量。

只有完成这些核查后，才能冻结一套全局配置。

## 7. 对 JOSH3R 的判断

JOSH3R 不是 JOSH 的快速多人等价版本。当前发布代码具有以下限制：

- 只读取排序后的第一条 `tram/*.npy`；
- 只生成一条 SMPL 轨迹；
- 固定使用人物身份 0；
- 通过间隔约三帧的图像对预测相对人体变换，因此不是严格零未来帧；
- mesh 会在内存中生成，但当前 `scene.pkl` 主要保存相机和内参，仍需额外的无几何修复 adapter 才能进入统一 evaluator。

由此得到明确结论：

- **不能进入 EgoBody/EgoHumans/Harmony4D 多人主表；**
- 不能把它的固定 ID 0 当成多人 identity 输出；
- 可以进入 AIST++ 单人 single-cut 或 multi-cut 附录；
- 应标为 `single-track, non-causal/offline reference`；
- 论文说服力低于完整 JOSH 或成熟的 WHAC/GVHMR。

## 8. 对 SLAHMR 的判断

SLAHMR 是重要的 multi-person offline optimizer，但官方数据路径明确选择单个 shot：

- `external_baselines/SLAHMR/slahmr/data/vidproc.py` 中，相机预处理根据 `cfg.shot_idx` 选择一个 shot interval，并只在该区间运行 SLAM；
- `external_baselines/SLAHMR/slahmr/data/dataset.py` 中，`MultiPeopleDataset` 也只加载指定 `shot_idx` 的图像。

所以，SLAHMR 官方路径不能原生输出两个 shot 共用的一个 world gauge。

以下做法不应使用：

```text
shot 1 单独跑 SLAHMR
+ shot 2 单独跑 SLAHMR
+ GT / 相机标定 / Shot3R Boundary 对齐
= “SLAHMR baseline”
```

因为额外对齐已经创造了一个新方法，而不是评测原始 SLAHMR。

合理方案是：

1. 只选 1--3 个 development case；
2. 尝试不修改核心算法的 custom-video 路径；
3. 检查是否返回一个覆盖两个 shot 的原生 camera--human world；
4. 如果只返回一个 shot 或两个独立 gauge，则停止扩展；
5. per-shot PA-MPJPE/coverage 可以作为诊断；
6. 跨镜头 W/WA、完整 camera ATE、IDF1、seam 应标为 `N/A`。

因此，SLAHMR 适合 Related Work 和补充材料中的协议兼容性说明，不适合默认做三个数据集的完整 numerical baseline。

## 9. 其他更值得关注的方法

### 9.1 Multi-THuMBS：任务最匹配，但受公开性阻塞

Multi-THuMBS 直接处理：

- multi-shot；
- multi-person；
- shared world；
- cross-shot identity；
- 完整相机注册；
- boundary 处的人体 root/orientation 与相机联合对齐；
- 全序列 smoothing 和 cross-camera reprojection。

它与当前同一场景、近时间相邻镜头切换的协议高度一致。但截至当前审计：

- 没有找到作者关联的完整可执行实现；
- 没有公开固定序列、camera pair、cut frame 和 manifest；
- 没有公开完整 person matching、ATE、ID 和聚合协议；
- 论文数字来自不同数据与评测定义，不能直接混入同输入主表。

建议直接联系作者索取：

- inference code 和 checkpoints；
- EgoHumans manifest；
- evaluator；
- camera alignment 定义；
- identity matching 与 miss/FP 处理规则。

如果获得完整官方实现，它应比 SLAHMR 更优先，且任务匹配程度高于 JOSH。

### 9.2 HumanMM：适合单人 multi-shot，但目前不可复现

HumanMM 直接研究 multi-shot global human motion，但它是单人、离线方法。当前公开仓库不足以完成同输入推理和统一评测，因此：

- 当前保留为 Related Work；
- 不混用论文原始数字；
- 若作者开放代码，优先加入 AIST++ 单人协议，而不是多人主表。

### 9.3 WHAC：推荐新增的单人 baseline

WHAC 的优点是：

- 已正式发表；
- 输出 world-coordinate human motion；
- 输出相机运动；
- 适合完整视频的单人 global HMR；
- 当前论文 Related Work 尚未引用。

限制是官方实现只保留检测最多的一条 track，不适合多人主表，也没有显式 hard-cut 机制。因此最合适的位置是 AIST++ 单人附录。

### 9.4 GVHMR

本地已经存在 GVHMR 环境和 availability pilot。它是成熟的 global HMR 参考，但官方 tracker 是 top-1 person，因此同样只建议用于 AIST++。

若资源只允许选择一个单人 baseline：

1. 首选 WHAC：输出与论文缺口更匹配，且当前未引用；
2. 次选 GVHMR：本地执行条件更成熟；
3. 再次是 JOSH3R：任务名称接近，但当前更像单轨 demo。

### 9.5 低优先级或不适合 numerical baseline 的方法

- **GLAMR：**支持离线多人全局重建，但环境较老且无显式 cut 处理；仅作为 JOSH 失败时的 fallback。
- **WHAM / TRAM：**单人或逐 track，且 PromptHMR 已覆盖/继承部分 TRAM 路线；新增证据有限。
- **Multishot (CVPR 2022)：**公开 demo 需要指定单个 PHALP track，缺少可独立评测的物理相机；只适合人体局部或边界指标。
- **EmbodMocap：**需要双视角同步采集、额外 scene recording/calibration 和人工同步信息，输入条件不同。
- **HSfM：**静态多视图，不输出 150 帧动态人体轨迹。
- **ShowMak3r：**更适合电视内容的离线场景/人物定性讨论。
- **UniCon3R、GUSH3R、TTT3R、TTSA3R、ReCal3R：**属于连续观测下的状态维护或通用场景重建，不直接满足当前跨切多人 camera--human 输出协议。

## 10. 为什么 AIST++ 值得增加单人基线

当前 AIST++ single-cut 正式结果为：

| 方法 | Temporal mode | Anchor-MPJPE (mm) |
|---|---|---:|
| Human3R | causal streaming | 492.9 |
| Shot3R | causal streaming | 556.3 |
| PromptHMR | offline full-video | 462.6 |

Shot3R 在 Anchor-MPJPE 上弱于 Human3R 和 PromptHMR。因此新增 WHAC、GVHMR 或 JOSH3R 可能继续形成压力，但这是有价值且诚实的：它能帮助论文更准确地限定贡献为：

- 多人；
- 相机；
- 身份；
- hard-cut boundary operation；
- causal streaming；

而不是宣称在所有单人世界轨迹指标上都优于离线方法。

## 11. 建议的实验顺序与停止条件

### 阶段 A：JOSH 12-case 冻结 pilot

选择与方法无关的 12 个 case：

- EgoBody、EgoHumans、Harmony4D 各四个；
- 每个数据集分别覆盖 small、medium、large、extreme viewpoint change；
- pilot 名单先冻结，再运行任何 JOSH 结果。

扩展门槛：

1. 至少 10/12 个 clip 无人工干预完成；
2. 两个 shot 均有 finite native predictions；
3. 两个 shot 均有非零 evaluated coverage；
4. camera-to-world 与 world-human 坐标约定通过 projection audit；
5. 同一套配置覆盖全部角度档位；
6. runtime、失败和输出支持可以无例外地报告。

通过后：

1. 先扩展 EgoHumans 90 例；
2. 单独放入 offline full-sequence 表；
3. 若 completion、时间和有效支持可接受，再考虑 EgoBody/Harmony4D；
4. 不评测三个数据集没有统一可靠 GT 的 scene reconstruction error。

### 阶段 B：AIST++ 单人 baseline

优先顺序：

1. WHAC；
2. GVHMR；
3. JOSH3R，可作为额外行而非替代成熟 baseline。

所有方法使用相同 AIST++ single-cut/multi-cut manifest，并分别标注 causal、pair-based 或 full-video access。

### 阶段 C：SLAHMR 兼容性诊断

- 仅 1--3 个 development case；
- 不把两个 shot 独立结果做额外对齐；
- 若没有完整跨 shot world，立即停止；
- 把结论写入 supplement，而不是制造不可比较的主表数字。

### 阶段 D：争取 Multi-THuMBS/HumanMM

- 联系作者；
- 获得代码后先做同样的 adapter 和 availability audit；
- 未获得完整协议前，不引用其原论文数字作为同表 numerical comparison。

## 12. 所有新增 baseline 的公平协议

### 12.1 输入

1. 使用完全相同、冻结的 ordered RGB manifests；
2. 不提供 GT cut、相机、人物身份、人数、mask、box 或 depth；
3. baseline 内部原生 detector、tracker、shot detector、SLAM 可以保留；
4. 不删除 hard cut；
5. 不把两个 shot 分别处理后再人工拼接；
6. 固定一套全局配置，禁止逐视频调参。

### 12.2 输出 adapter

只允许无几何修复的格式转换：

```text
cameras_c2w       [T,4,4]
vertices_world    [T,P,6890,3]
joints_world      [T,P,24,3]
persistent_ids    [T,P]
native_ids        [T,P]
valid             [T,P]
```

允许：

- 固定坐标系 convention 转换；
- 有文档记录的 SMPL/SMPL-X topology 转换；
- 将原生输出索引映射回输入帧。

禁止：

- 单独拟合 post-cut transform；
- 用 GT 选择最佳人物 track；
- 人工修复 ID；
- 借用 Shot3R 或其他方法的相机/scene 输出；
- 根据最终评测结果选择每个 case 的超参数。

### 12.3 聚合与报告

- crash、empty output、miss、ID switch 和 zero coverage 保留在完整分母；
- W/WA 等条件误差同时报告有效支持数；
- 报告 coverage 和 completion rate；
- causal、semi-online/pair-based、offline 分区报告；
- 不把不同数据、不同 alignment、不同 evaluator 的原论文数字混入同输入结果表。

## 13. 论文文字建议

### 13.1 补充 JOSH 的 Related Work 定位

当前 JOSH 只在 Introduction 中作为 complete-sequence/per-video optimization 方法被引用。虽然 changelog 声称已把 JOSH 加入 Related Work，但最新版 `sections/03_related_work.tex` 的正文并未真正展开。

建议在 SLAHMR/TRAM 段加入类似表述：

> JOSH further jointly optimizes people, cameras, scene geometry, and human--scene contact over complete videos, providing a strong offline full-system reference but relying on future frames and per-video refinement.

### 13.2 补充 WHAC

当前 Related Work 提到 SLAHMR、TRAM、WHAM、GVHMR 和 OnlineHMR，但没有 WHAC。建议将 WHAC 放在 learned/global human--camera motion recovery 这一脉络中，并说明这些方法主要假设 temporally continuous input。

### 13.3 明确三类比较角色

建议论文中始终区分：

1. **Controlled causal baseline：**Human3R；
2. **Same-input public references：**OnlineHMR、TRACE、PromptHMR-SPEC；
3. **Offline full-sequence references：**JOSH，以及未来可能获得的 Multi-THuMBS/HumanMM。

这样可以避免审稿人把“离线方法使用未来帧”误解为不公平遗漏，也避免把不同协议的方法强行做成一个总排行榜。

## 14. 当前最终推荐

如果只允许新增两个实验：

```text
多人/主任务：JOSH，12-case pilot -> 通过后 EgoHumans 90
单人/AIST++：WHAC；若执行受阻则 GVHMR
```

同时：

```text
Multi-THuMBS：立即联系作者，拿到代码后优先级最高
SLAHMR：1--3 例诊断，不做三数据集全量
JOSH3R：只作为 AIST++ 可选额外行
```

## 15. 主要证据位置

- Movie3R 项目说明：`/data/wangzheng/iJCV-CODE/Movie3R/README.md`
- 最新实验设置：`/data/wangzheng/iJCV-CODE/ICLR-paper/bridge3r_iclr2027/versions/v037_20260906_supplement_revision/manuscript/sections/05_experiments.tex`
- 最新 Related Work：`/data/wangzheng/iJCV-CODE/ICLR-paper/bridge3r_iclr2027/versions/v037_20260906_supplement_revision/manuscript/sections/03_related_work.tex`
- 最新多人主表：`/data/wangzheng/iJCV-CODE/ICLR-paper/bridge3r_iclr2027/versions/v037_20260906_supplement_revision/manuscript/tables/main_results.tex`
- AIST++ single-cut 表：`/data/wangzheng/iJCV-CODE/ICLR-paper/bridge3r_iclr2027/versions/v037_20260906_supplement_revision/manuscript/artifacts/aist_cs150_formal/aist_cs150_formal_table.tex`
- SLAHMR shot 选择代码：`/data/wangzheng/iJCV-CODE/external_baselines/SLAHMR/slahmr/data/vidproc.py`
- SLAHMR dataset：`/data/wangzheng/iJCV-CODE/external_baselines/SLAHMR/slahmr/data/dataset.py`
- Multi-THuMBS 审计：`/data/wangzheng/iJCV-CODE/Movie3R/versions/v14/docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md`
- 旧 baseline 扩展计划：`/data/wangzheng/iJCV-CODE/ICLR-paper/bridge3r_iclr2027/versions/v034_20260906_onlinehmr_extensions/BASELINE_EXTENSION_PLAN_20260902.md`

## 16. 请审阅者重点反驳或确认

请优先检查以下可能影响最终决策的点：

1. JOSH 最新官方代码是否确实能在不修改核心方法的情况下输出完整 150 帧、多 track 的 world human 与 camera；
2. JOSH 的 `max_frames`、关键帧采样和完整帧回填是否会使 W/WA/IDF1 评测失真；
3. JOSH 的 TRAM track 是否在 hard cut 后保留同一人物 identity，或是否会自然产生两个 track；
4. JOSH3R 是否已有晚于本次审计的新多人入口或更完整输出保存逻辑；
5. SLAHMR 是否存在官方、未被当前仓库暴露的 multi-shot joint-world 模式；
6. Multi-THuMBS/HumanMM 是否已经公开了新的 executable、checkpoint 或 evaluator；
7. WHAC 与 GVHMR 中哪一个能以最少非算法性 adapter 输出完整 AIST++ 人体与相机指标；
8. 是否存在遗漏的、公开可执行且同时满足 multi-shot、multi-person、camera、identity 的新方法。

如果上述事实没有发生变化，则本文给出的优先级和停止条件应当成立。
