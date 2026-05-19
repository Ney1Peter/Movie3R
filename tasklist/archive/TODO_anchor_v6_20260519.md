# Movie3R TODO

更新时间：2026-05-14

本文只保留当前真实待办。旧版长 TODO 已归档到：

```text
tasklist/archive/TODO_legacy_20260514.md
```

## 当前主线：AnchorToken V6

目标：把 mesh/XFeat 验证过的 static background anchors 转成受控的 local AnchorToken evidence，用于 pose/camera re-anchor。第一版仍然不改 encoder，也不把 anchor token 插入完整 decoder token sequence。

### 1. 数据与 cache 接入

- [ ] 在 dataset / loader 中接入 offline anchor cache manifest。
- [ ] 读取 cache 中的 `ref_patch_idx`、`cur_patch_idx`、`ref_pos_norm`、`cur_pos_norm`、`delta_uv_norm`、`affine_forward`、`affine_inverse`、`quality_gate`。
- [ ] 支持 cache 缺失时 fallback 到 base path，不中断训练。
- [ ] 先使用 `/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_high_overlap_v1/` 做小规模受控实验。

### 2. 模型接入

- [ ] 新增最小 AnchorToken adapter，只影响 pose/camera path。
- [ ] 保持 Human3R encoder 不变。
- [ ] 保持完整 decoder token sequence 不变。
- [ ] 使用 `global affine evidence + local AnchorToken residual + quality_gate` 作为第一版输入。
- [ ] `unique_anchor_patch_pairs >= 16` 强启用，`8-15` 弱启用，`<8` fallback 或低权重。

### 3. 训练与验证

- [ ] 先冻结主模型，只训练 AnchorToken 相关小模块。
- [ ] 设计小规模 smoke training，确认 loss、梯度、fallback 都正常。
- [ ] 对比 base / affine-only / AnchorToken residual 三组输出。
- [ ] 增加 camera translation、rotation、SMPL translation、pointmap extent 等尺度监控。
- [ ] 明确记录 weak anchor 样本的 fallback 行为。

### 4. Cache 扩展

- [ ] 在 guitar high-overlap 稳定后，加入 medium-overlap camera pairs。
- [ ] low-overlap camera pairs 先作为 hard validation / fallback 测试，不作为第一批主训练数据。
- [ ] 统计不同 overlap、anchor count、quality gate 与最终收益的关系。

### 5. 推理方案

- [ ] 设计 inference-time XFeat semi-dense matching 流程。
- [ ] 加 lightweight geometry / confidence filtering，替代训练时的 RICH mesh teacher。
- [ ] 选择 top-K 空间分散 anchors，默认保留 8-16 个高质量 tokens。
- [ ] 当 overlap 或 anchor count 不足时自动 fallback。

## 文档整理

- [ ] 迁移完成后再决定是否把 `MIGRATION_NEW_SERVER.md` 归档。
- [ ] 后续如需更新入口文档，再统一修正 `README.md` / `CLAUDE.md` 中的环境文档链接。
- [ ] `tasklist/work_log.md` 保留为随笔和原始流水账，不要求持续整理。

## 已归档但暂不删除

```text
tasklist/archive/TODO_legacy_20260514.md
tasklist/archive/work_compact.md
tasklist/archive/training_record.md
```

环境文档已移动并重命名为：

```text
docs/env_setup_h800_cuda124.md
```
