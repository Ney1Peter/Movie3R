# V7 Implicit Token Adapter Validation

## 1. 验证目的

本轮实验验证的问题是：

```text
Human3R 内部 token 中是否已经包含足够的 human + scene 信息，
让一个轻量 adapter 在不读取 decoded SMPL、显式平面和未来帧的情况下，
预测 shot-change 后的 camera pose correction。
```

换句话说，显式 human-scene geometry teacher 只负责离线生成 pseudo label；最终 student / inference 只看 Human3R forward 中已有的隐式 token。

## 2. 具体流程

当前 overfit 不是“修改 token 再重新过 pose head”，而是：

```text
Human3R forward
  -> dump pose / human / scene / memory tokens
  -> adapter 读取 tokens + raw camera pose
  -> adapter 预测 alpha_t 和 delta_xi_t
  -> T_corr = exp(alpha_t * delta_xi_t) @ T_raw
  -> 和 teacher pseudo GT 对齐训练
```

student 输入不包含：

```text
decoded SMPL-X bodies as explicit anchors
explicit background planes / normals
future frames
post-shot stable window
global BA / pose graph / chunk alignment
```

## 3. 已完成实验

### 3.1 H36M boundary63

```text
case: h36m_test_boundary63
source: data-V7-test/h36m/h36_new.mp4
raw output: output/human3r_h36m_test
boundary: 62 -> 63
teacher target frames: 63 / 64 / 65
tokens: output/v7_h36m_pseudo_gt_smoke/h36m_test_boundary63/v7_tokens.npz
```

Token dump 内容：

```text
frames: 120
pose_tokens: (120, 1, 768)
scene_tokens: (120, 992, 768)
human_tokens: (120, 1, 768)
memory_tokens: (120, 768, 768)
target frames: [63, 64, 65]
teacher delta_t_norm: [1.7406, 0.2724, 0.0767]
```

Overfit best metrics：

| input mode | best step | best loss | target err t | target err r | no-op delta t |
| --- | ---: | ---: | ---: | ---: | ---: |
| human_scene | 750 | 0.00000936 | 0.00027 | 0.0045 deg | 0.00105 |
| human | 1250 | 0.00922784 | 0.00220 | 0.0371 deg | 0.00157 |
| scene | 1500 | 0.00938343 | 0.00231 | 0.0050 deg | 0.00180 |
| pose | 1500 | 0.01860497 | 0.00522 | 0.0194 deg | 0.00231 |

Viewer 输出：

```text
corrected viewer:
  output/v7_h36m_pseudo_gt_smoke/h36m_test_boundary63/v7_implicit_viewer_human_scene_lr5e4

raw camera overlay:
  output/v7_h36m_pseudo_gt_smoke/h36m_test_boundary63/v7_implicit_viewer_raw_subset

viewer frames:
  original frames 55..85
  target frames 63..65 -> viewer index 8..10
```

### 3.2 H36M boundary91

```text
case: h36m_18s_boundary91
source: data-V7-test/h36m/h36m_ms_000020_18s_25s.mp4
raw output: output/human3r_h36m_18s
boundary: 90 -> 91
teacher target frames: 91 / 92 / 93
tokens: output/v7_h36m_pseudo_gt_smoke/h36m_18s_boundary91/v7_tokens.npz
```

Token dump 内容：

```text
frames: 210
pose_tokens: (210, 1, 768)
scene_tokens: (210, 768, 768)
human_tokens: (210, 1, 768)
memory_tokens: (210, 768, 768)
target frames: [91, 92, 93]
teacher delta_t_norm: [2.4761, 0.4738, 0.0548]
```

Overfit best metrics：

| input mode | best step | best loss | target err t | target err r | no-op delta t |
| --- | ---: | ---: | ---: | ---: | ---: |
| human_scene | 1000 | 0.00001627 | 0.00332 | 0.0533 deg | 0.00258 |
| human | 1000 | 0.00519180 | 0.01718 | 0.1885 deg | 0.00234 |
| scene | 750 | 0.00536734 | 0.00165 | 0.0181 deg | 0.00473 |
| pose | 1250 | 0.01054729 | 0.00392 | 0.0314 deg | 0.00863 |

Viewer 输出：

```text
corrected viewer:
  output/v7_h36m_pseudo_gt_smoke/h36m_18s_boundary91/v7_implicit_viewer_human_scene_lr5e4

raw camera overlay:
  output/v7_h36m_pseudo_gt_smoke/h36m_18s_boundary91/v7_implicit_viewer_raw_subset

viewer frames:
  original frames 83..113
  target frames 91..93 -> viewer index 8..10
```

## 4. 阶段性结论

当前结果是正向 sanity check：

```text
Human3R internal pose / human / scene / memory tokens 中确实有可用信号，
轻量 adapter 可以从这些 token 中恢复 teacher camera correction。
```

更具体地说：

```text
1. human_scene 在两个 clip 上总 loss 都最低，说明同时看人体和场景 token 是合理的。
2. human / scene / pose 单独也能在单 clip 上拟合，但这更接近记忆化 overfit，不能证明泛化。
3. no-op delta 较小，说明 adapter 在正常帧上可以学到基本不乱改。
4. viewer 中 corrected camera / pointcloud / human mesh 相对 raw camera 有可见修正效果。
```

当前不能声称已经解决泛化问题。单 clip overfit 只能证明“token 中有信号”，不能证明“未见视频也能修”。

## 5. Pseudo GT 可靠性说明

当前 pseudo GT 仍来自 Human3R saved output 上的离线 teacher，因此它不是人工真值，也不是最终评测真值。

它适合用于：

```text
1. 验证 teacher 是否能构造一个合理的 correction upper bound。
2. 验证 causal student 是否能从内部 token 中复现 teacher correction。
3. 做早期大规模弱监督预训练或蒸馏。
```

它不适合单独用于最终 claim：

```text
1. teacher 可能受错误 SMPL、错误 pointmap 或错误 plane match 影响。
2. pseudo GT 可能继承 Human3R 的系统性偏差。
3. 如果没有 held-out validation 和可视化检查，student 可能只是在学习 teacher bias。
```

后续应逐步加入更强验证：

```text
held-out clips
manual visual audit
teacher reliability filtering
RICH / AvatarReX / H36M / MS-AIST 跨数据验证
少量真实或人工校验标注，如果后续可获得
```

## 6. 下一步：MS-AIST Shot2 Pilot

下一步从单 clip overfit 转向 multi-clip held-out validation。优先使用：

```text
/data/wangzheng/iJCV-CODE/data/data-V7-shot-change-clips/ms-aist/videos/shot2
```

当前该目录包含：

```text
mp4 clips: 99
total video size: about 115 MB
manifest: /data/wangzheng/iJCV-CODE/data/data-V7-shot-change-clips/ms-aist/manifest.json
```

建议分三步扩大：

```text
Stage A: 5 clips
  目的：检查 Human3R raw output、teacher pseudo label、token dump 是否稳定。

Stage B: 20 train + 5 val
  目的：第一次验证未见 clip 泛化。

Stage C: 80 train + 19 val
  目的：覆盖完整 shot2，评估不同动作和背景。
```

评估必须包含：

```text
human_scene / human / scene / pose / all ablation
target correction error
normal no-op error
alpha_t 是否只在 boundary / settling frames 打开
held-out viewer visual audit
```

## 7. 存储策略

MS-AIST 原始 clip 不大，但 Human3R saved-output 会快速变大。后续批量实验应采用最小存储策略。

推荐保留：

```text
per clip:
  pseudo_gt_labels.npz
  v7_tokens.npz
  summary / metrics json

debug subset only:
  viewer-ready corrected output
  viewer-ready raw camera overlay
```

Human3R raw saved-output 只在 teacher 构造和可视化诊断时需要。生成 pseudo labels 和 tokens 后，可以只保留少量失败样本的完整 raw output，其余样本保留 token / label / metrics。

如果需要 viewer 输出，corrected output 不应复制大文件，应复用 hardlink / symlink：

```text
camera/*.npz: 写入 corrected pose
color/depth/conf/smpl: hardlink 或 symlink 到 raw output
raw camera overlay: 只用于对比 camera，不显示 raw 点云 / raw 人体
```

## 8. 当前相关脚本

```text
scripts/build_v7_h36m_pseudo_gt_smoke.py
scripts/build_post_shot_local_gauge_teacher.py
scripts/dump_v7_implicit_tokens.py
scripts/overfit_v7_implicit_token_student.py
scripts/export_v7_implicit_student_viewer_output.py
scripts/view_human3r_saved_output.py
```

当前新增能力：

```text
1. dump Human3R internal V7 tokens。
2. overfit implicit token student，并保存 best checkpoint / best predictions。
3. 导出 adapter corrected viewer output，只显示 corrected 点云和人体，叠加 raw camera。
```
