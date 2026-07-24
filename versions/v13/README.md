# Movie3R-Multi V13.0

## 定位

V13 是独立的多人 shared-Boundary 正式研究目录，不是 V12 的一个小消融。它由历史
V20 Phase 1 v2 实验冻结，用于验证：在人物身份已知时，多个人能否共同提供比可部署
单人选择器更稳定的 camera-cut 几何约束。

当前路线状态已经冻结为：**GT-ID 多人几何可行性通过，值得进入 cross-shot WHO/Re-ID
阶段；可部署多人身份关联尚未完成。** 里程碑边界见
[`MULTIHUMAN_GEOMETRY_VALIDATED.md`](MULTIHUMAN_GEOMETRY_VALIDATED.md)。

```text
5 pre-cut RGB frames + 1 fresh post-cut RGB frame
-> frozen Human3R multi-human reconstruction
-> strict GT mesh-projection identity association (WHO, Oracle)
-> each matched human produces one R_i/t_i candidate (WHERE)
-> SO(3) mean rotation + arithmetic mean translation
-> ONE shared Boundary for camera, pointmap and every SMPL-X
```

核心原则是：

```text
Identity answers WHO.
Geometry answers WHERE.
All humans share ONE Boundary.
```

## 冻结身份

| 项目 | 值 |
|---|---|
| 正式名称 | Movie3R-Multi V13.0 |
| 几何实现 commit | `e45e2af` |
| Git tag | `movie3r-v13-multi` |
| Backbone | frozen `src/human3r_896L.pth` |
| Human3R SHA-256 | `1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377` |
| 数据 | MultiHuman Real-World-Capture `three` |
| 规模 | 315 cuts，3 人，6 相机，offset 0/1/2/4/8 |
| 当前身份模块 | strict GT-ID projection Oracle |
| 当前最佳融合 | all-valid-human naive mean |

## 当前效果

在 308-case common support 上，多人 mean 相比 highest-confidence single：

- camera translation：`0.565 -> 0.517 m`；
- camera rotation：`9.96 -> 7.01 deg`；
- composite：`0.764 -> 0.657`；
- human joints：`0.402 -> 0.380 m`；
- human vertices：`0.392 -> 0.372 m`。

在 212 个三人均有效样本上，人数增加呈单调改善：

| 人数 | Camera T | Rotation | Composite |
|---:|---:|---:|---:|
| 1 | 0.594 m | 10.80 deg | 0.810 |
| 2 | 0.560 m | 8.81 deg | 0.737 |
| 3 | 0.549 m | 7.49 deg | 0.699 |

多人 mean 没有超过读取 GT evaluator 的 Oracle Best Single（`0.657` vs `0.633`），
因此原预注册的严格 gate 仍是 FAIL；但它显著优于所有已测试的可部署单人选择策略，
证明多人冗余几何本身有价值。

Phase 2 进一步比较了 soft confidence、motion uncertainty、candidate dispersion、
layout weighting 和 rotation/translation 分解。开发集选出的 soft rule 在 held-out 上
没有超过 naive mean，因此当前 fusion 仍冻结为 naive mean。`dance` 两人 36-cut pilot
也得到相同路线决策：多人平均有帮助，手工 soft weighting 尚不稳定。

## 路线决策与下一阶段

当前版本正式记录为“多人 geometry validated”，原因是 `three` 和 `dance` 都表明：在
严格身份正确时，多人 shared-Boundary 优于可部署单人 anchor，且人数增加呈单调改善。
这满足继续研究多人路线的前置条件。

下一阶段只解决 WHO：Human3R native token 的跨 shot Re-ID、dustbin、新人/消失人物、
tracklet TTL 和 geometry verification。Token 只参与 identity association，不能直接回归
Boundary；通过 tentative match 后，仍由多人显式几何求唯一 shared Boundary，并执行
Match-Then-Align / Align-Then-Commit。

## 当前启用与关闭

启用：frozen Human3R、pre-decode hard reset、Fixed Explicit、V16 20 度约束、显式
translation、一个 shared Boundary。

关闭：DA3、Keypoint R-CNN、V11.4 scale、VGGT、continuity、token Re-ID、learned ID
adapter 和 scene refinement。Shot scale 固定为 `s=1`。

## 主入口

完整 GT-ID 几何评测：

```bash
PYTHONPATH=src:. .venv/bin/python versions/v13/gt_id_consensus.py \
  --data_root /data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted \
  --device cuda:0 \
  --output_dir output/v13/multihuman
```

可用 `--sequence three|dance|box` 选择 Real-World-Capture 序列。`dance/box` 自动使用
两个人；`three` 使用三个人。

Phase 2 fusion-only 复评：

```bash
PYTHONPATH=src:. .venv/bin/python \
  versions/v13/experiments/fusion_optimization.py
```

仅使用已有 cache 重评 V2 identity：

```bash
.venv/bin/python versions/v13/gt_id_consensus.py \
  --evaluation_only \
  --output_dir output/v13/multihuman \
  --timestamps 500 700 900 1000 1100 1300 1500 \
  --camera_pairs 0-1 1-2 2-3 3-4 4-5 5-0 0-3 1-4 2-5 \
  --offsets 0 1 2 4 8
```

Viewer：

```bash
PYTHONPATH=src:. .venv/bin/python versions/v13/viewer.py \
  --result_dir output/v13/multihuman \
  --method multi \
  --case three_t0900_c0_c3_k0 \
  --port 8080
```

Native Human3R token 和 EgoBody 数据探针分别位于：

- `versions/v13/native_token_probe.py`
- `versions/v13/egobody_probe.py`

它们源自历史 V20 实验，是身份/数据诊断入口，不代表 token Re-ID 已经通过验证。

## 不能宣称的内容

- 当前不是可部署多人版本，因为 GT identity/GT mesh projection 进入 WHO association。
- 当前 robust Huber/layout/reject 比 naive mean 更差，不能作为默认 consensus。
- 还没有完成 token Re-ID、dustbin、TTL、进入/离开和遮挡恢复。
- 当前只有 MultiHuman `three` 主调试序列和 `dance` 小规模跨序列 pilot，不是最终跨数据
  benchmark。
- V13 不能替代 V12 当前单人默认路径；单人输入应自动退化为 V12/Lite 逻辑，
  这一完整集成尚未冻结。

## 详细文档

- `versions/v13/MULTIHUMAN_GEOMETRY_VALIDATED.md`，当前路线里程碑与下一阶段准入决定
- `versions/v13/docs/V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS_V2.md`，当前有效结论
- `versions/v13/docs/V20_PHASE1_GT_ID_MULTIHUMAN_CONSENSUS.md`，旧错误 ID 报告，仅作审计
- `versions/v13/docs/V20_EGOBODY_MULTIHUMAN_DATASET_GUIDE.md`
- `versions/v13/docs/V20_EGOBODY_LEGOASSEMBLE_FEASIBILITY.md`
- `versions/v13/docs/V13_PHASE2_MULTIHUMAN_FUSION_OPTIMIZATION.md`
