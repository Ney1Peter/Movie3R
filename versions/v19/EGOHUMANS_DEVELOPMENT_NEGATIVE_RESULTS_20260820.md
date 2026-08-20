# EgoHumans development negative results

本文档只记录在 development 上被否决的分支。它们不进入 holdout/test，也不改变冻结协议。

## 1. Causal identity re-tracking

证据范围：`basketball/004_basketball` 的四个预注册跨度，其中三个可评；extreme case 对所有方法均无法形成初始人体匹配。

Parent 的三例均值为 IDF1 0.365、IDs 29.67、Coverage 0.681。有限 identity 网格中，最好 IDF1 为 0.380，但 IDs 变为 30.0；其余多数配置降低 IDF1 或增加 IDs。例如 `id050` 的 IDF1 为 0.295、IDs 为 37.33。几何平移候选的 W 改善不能归因于 identity tracker。

结论：单个 case 上曾观察到 IDF1 从约 0.343 提升到 0.471，但跨三个可评跨度不能同时稳定改善 IDF1 和 IDs。该分支不进入主线，最终方法沿用 B0 已有的因果边界对应，不额外重写整段身份轨迹。

原始报告：

- `output/v19_egohumans/development/captures/basketball__004_basketball/v19_identity_grid_early.json`
- `output/v19_egohumans/development/captures/basketball__004_basketball/v19_identity_grid_early.csv`

## 2. Per-person post-shot translation

同一组三例中，parent 的 W-MPJPE 为 1939.9 mm。逐人平移 blend 0.50/0.75/1.00 的 W 分别为 1921.2/1924.0/1974.4 mm：弱 blend 的改善不足 1%，强 blend 反而恶化；blend 1.00 还把 Coverage 从 0.681 降至 0.644。它没有形成可复现的全局收益。

结论：逐人强制贴合容易把检测/身份噪声写入完整 post-shot 轨迹。该分支被否决，保留“对所有人体共享同一跨-shot gauge 变换”的主线。

原始报告：

- `output/v19_egohumans/development/captures/basketball__004_basketball/v19_person_grid_early.json`
- `output/v19_egohumans/development/captures/basketball__004_basketball/v19_person_grid_early.csv`

## 3. SE(3) versus translation-only boundary correction

在前四个 development 动作、15 个可评 case 上，完整 translation 的核心五指标几何均值比 parent 为 0.641，并通过全部 development 安全门槛；完整 SE(3) 为 0.662，但因最坏 case 安全条件未通过而被否决。translation 的 W/ATE-SE3 更稳，SE(3) 虽偶尔改善 seam 或 WA，却更容易把人体方向噪声传播到整段 post-shot。

结论：EgoHumans 的 development 证据支持先只校正共享平移 gauge；旋转保留 parent 的因果估计。后续只在预注册的 shared-translation 与 causal root filter 组合中选择。

中间聚合：`output/v19_egohumans/development/interim_four_actions/`。

## 4. 论文使用边界

- 以上均为 development 负结果，可用于消融与 failure analysis；
- 不把单例改进当作最终结论；
- holdout 和 test 不再打开 identity/person/SE(3) 新网格；
- 若所有新候选未通过独立 holdout，按协议回退冻结 v17 MultiCue-Safe。
