# V7 Human-Scene Pose Correction Experiment Log

本文只记录实验过程、现象和阶段性判断。V7 主计划仍见：

```text
docs/movie3r/v7/online_human_scene_pose_correction_plan.md
```

## 1. 当前结论

目前最有效方向是：

```text
Human-Guided, Geometry-Guarded Pose Correction
```

核心判断：

```text
人体 anchor 能发现 shot boundary 处的人体瞬移，但不能单独决定正确相机位姿。
背景几何能保护地面 / 墙面 / 世界方向，进一步约束左右滑动和 yaw。
最可靠的方向不是 human-only，也不是 background-only，而是 human + scene geometry joint correction。
```

当前最好的 smoke test 是：

```text
scripts/overfit_single_boundary_frame_scene_geometry.py
```

它只使用上一帧和当前帧，只修当前跳变帧，不使用未来帧、不做全局优化。

## 2. 测试数据

### 2.1 H36M 18s-25s clip

```text
video:  data-V7/h36m/h36m_ms_000020_18s_25s.mp4
output: output/human3r_h36m_18s
frames: 210
shot boundary: 90 -> 91
boundary arg: 91
```

该数据中人会蹲下再起立，因此不能依赖“整段人体静止”的强假设。

### 2.2 H36M h36_new clip

```text
video:  data-V7/h36m/h36_new.mp4
output: output/human3r_h36m_test
frames: 120
shot boundary: 62 -> 63
boundary arg: 63
```

自动诊断显示主跳变是 `62 -> 63`，不是 30 附近。`63 -> 64` 还有明显 settling jump。

## 3. 实验版本汇总

### 3.1 Long-Reference Human Anchor

脚本：

```text
scripts/overfit_human_anchor_pose_correction.py
```

方法：

```text
用 pre-shot / 前半段人体作为 long-term reference。
post-shot 每一帧都被拉向这个人体 reference。
只改 camera pose，不改 depth / color / conf / SMPL。
```

观察：

```text
优点：在人体基本静止的数据上，能显著降低人体漂移。
缺点：对会运动的人不可靠，容易压制真实动作。
```

在 `human3r_h36m_18s` 上：

```text
boundary_foot_jump: 1.539 -> 0.113
post_foot_err:      0.887 -> 0.111
```

但它依赖“人整体基本不动”，不适合作为最终主线。

### 3.2 Boundary-Pair Human Consistency

脚本：

```text
scripts/overfit_boundary_pair_human_correction.py
```

方法：

```text
只强调 shot boundary 前后人体连续。
不把整个 post-shot 段都拉向 long reference。
```

观察：

```text
优点：更符合“人可以运动，但一帧内不应瞬移”的假设。
缺点：只修 boundary，两帧之后传播不足；后续人体方向可能仍不稳定。
```

在 `human3r_h36m_18s` 上：

```text
boundary_foot_jump: 1.539 -> 0.088
post_foot_err:      0.887 -> 0.283
```

### 3.3 Causal Memory Propagation

脚本：

```text
scripts/overfit_memory_human_correction.py
```

方法：

```text
boundary 修正后，把 corrected human anchor 写回 causal memory。
后续帧读取上一帧 corrected memory，再预测 correction。
```

观察：

```text
优点：形式更接近 online。
缺点：如果逐帧都修，容易把后续本来较准的 Human3R 轨迹改坏。
```

在 `human3r_h36m_18s` 上：

```text
boundary_foot_jump: 1.539 -> 0.100
post_foot_err:      0.887 -> 0.143
```

### 3.4 Learnable Transient Gate v2

脚本：

```text
scripts/overfit_transient_gate_human_correction.py
```

方法：

```text
把 correction 拆成 transient residual + learnable gate。
模型学习 shot 后每一帧需要修多少。
可靠帧 gate 被压低，避免后续整段被改坏。
```

关键公式：

```text
delta_xi_t = delta_xi_persistent + alpha_t * delta_xi_transient
T_corr_t = exp(delta_xi_t) @ T_hat_t
```

在 `human3r_h36m_18s` 上，gate 学到：

```text
91: 0.9982
92: 0.0000
93: 0.0048
94+: ~0
```

观察：

```text
优点：非常符合“91 最坏，92 后恢复”的现象。
缺点：仍是 human-only，可能出现人对齐但场景倾斜 / 上下漂移。
```

### 3.5 Single-Frame Full SE(3) Human-Only

脚本：

```text
scripts/overfit_single_boundary_frame_human_anchor.py
```

方法：

```text
只用 boundary-1 和 boundary 两帧。
只修 boundary 当前帧。
完整 SE(3) 自由度。
只用人体 stable joints / feet / torso orientation 作为监督。
```

观察：

```text
人能对齐，但场景会出现明显倾斜 / 上下偏移。
证明 human anchor 单独约束 camera pose 不充分。
```

在 `human3r_h36m_18s` 上：

```text
boundary_foot_jump: 1.539 -> 0.122
```

但视觉上场景变斜，说明人体提供的约束存在 SE(3) 歧义。

### 3.6 Single-Frame Planar Human-Only

脚本：

```text
scripts/overfit_single_boundary_frame_planar_human_anchor.py
```

方法：

```text
仍然只修 boundary 当前帧。
只允许水平平移 + yaw。
禁止 vertical translation / pitch / roll。
```

观察：

```text
场景不容易倾斜，但人体对齐不足，仍有上下/左右错位。
说明简单限制自由度不能替代背景几何约束。
```

在 `human3r_h36m_18s` 上：

```text
boundary_foot_jump: 1.539 -> 0.430
```

### 3.7 Single-Frame Human + Single Scene Normal

脚本：

```text
scripts/overfit_single_boundary_frame_scene_normal.py
```

方法：

```text
只用 boundary-1 和 boundary 两帧。
只修 boundary 当前帧。
人体提供对齐监督。
背景高置信点云去掉人体 mask 后，RANSAC 找 dominant plane。
用 dominant plane normal / offset 防止场景倾斜和上下漂移。
```

在 `human3r_h36m_18s` 上，效果非常正确：

```text
boundary_foot_jump:      1.539 -> 0.143
boundary_stable_jump:    1.611 -> 0.119
post_camera_max_step_t:  3.659 -> 0.552
```

观察：

```text
证明 scene geometry 对保护场景姿态非常有效。
但在 h36m_test 上仍有左右/yaw 偏移，说明 single normal 只解决倾斜，不足以约束平面内滑动。
```

### 3.8 Single-Frame Human + Multi-Plane / Background Geometry Guard

脚本：

```text
scripts/overfit_single_boundary_frame_scene_geometry.py
```

方法：

```text
在 single scene normal 基础上增加：
1. top-K 背景平面，而不是只用最大平面；
2. 背景点云 weak Chamfer，用于约束平面内左右滑动；
3. 适当降低人体权重，避免 human anchor 单独主导 horizontal/yaw。
```

在 `human3r_h36m_test` 上匹配到两个背景平面：

```text
plane 0: dot 0.997, weight 0.482
plane 1: dot 0.953, weight 0.063
```

指标：

```text
boundary_foot_jump:
  raw          1.108
  scene-normal 0.063
  multi-plane  0.113

boundary_camera_step_r_deg:
  raw          112.62
  scene-normal 152.16
  multi-plane  134.13

post_camera_max_step_t:
  raw          2.642
  scene-normal 1.505
  multi-plane  0.182

settle 63->64 camera_step_t:
  scene-normal 1.505
  multi-plane  0.106
```

视觉判断：

```text
multi-plane/background geometry guard 明显改善了左右偏移和 63->64 过渡。
这是当前最好的方向。
```

## 4. 当前最佳方法的理解

当前最佳方法不是“单独背景 align”，也不是“单独人体 align”，而是 joint correction：

```text
人体告诉模型：这一帧的人不应该瞬移。
背景 plane normal 告诉模型：地面/墙面不能歪。
多个 plane 和背景点云告诉模型：不能沿平面左右乱滑。
camera prior 告诉模型：能小修就不要大修。
```

当前 overfit 还是优化形式，但使用的数据保持 causal：

```text
只用 boundary-1 和 boundary。
不使用 boundary+1。
不使用未来帧。
不做全局 alignment。
```

## 5. 下一步建议

### 5.1 与 transient gate 合并

最合理的下一步是：

```text
Learnable Transient Gate
+ Human Anchor
+ Multi-Plane / Background Geometry Guard
```

期望形式：

```text
异常 boundary 帧: gate 打开，human + scene geometry 共同修正
恢复后的正常帧: gate 关闭，尽量保留 Human3R 原始输出
```

### 5.2 从 overfit 转为网络预测

当前脚本验证的是 loss 和 cue 是否有效。最终不应使用 per-sample optimization，而应训练 correction head：

```text
input:
  Human3R raw pose / pointmap / confidence
  human joints / human token
  online scene geometry cues
  previous corrected memory

output:
  correction gate alpha_t
  bounded delta_xi_t
  scene/human reliability weights
```

### 5.3 Scene geometry 的网络化路线

第一阶段可以使用 deterministic online geometry module：

```text
pointmap/conf/mask -> top-K planes / background geometry features
```

第二阶段再 distill 成 neural normal / plane head：

```text
scene features -> plane normals / confidence
```

推理目标仍是：

```text
每帧 forward 末端立即输出 corrected pose。
```

## 6. 新增方向：Offline Teacher, Online Student

2026-05-22 新增一个重要训练约束：可以用未来帧做 teacher / analysis，但最终模型绝对不能在推理时读取未来帧。

### 6.1 为什么需要 teacher

当前观察显示，shot 后第一帧和 cam2 后续稳定帧应处在同一个局部 gauge 中：

```text
boundary frame 91 不应继续留在 cam1 gauge；
也不应被硬设成后续某一帧的 pose；
它应被修到 cam2 local gauge 附近，同时保留合理相对运动。
```

为了知道这个 local gauge 应该长什么样，teacher 阶段可以读取 post-shot stable window：

```text
94..120 等未来稳定帧
```

teacher 的作用是回答：

```text
91 / 92 / 93 的 pseudo target 应该在哪里；
哪些 scene planes / background geometry cue 可靠；
correction gate 应该如何从 boundary 到 stable frames 衰减。
```

### 6.2 不能违反的最终约束

未来帧只能用于 teacher / pseudo label / oracle upper bound，不能作为最终模型输入。

最终 student 必须保持 causal：

```text
input at frame t:
  current Human3R raw outputs
  current pointmap / confidence / SMPL
  previous corrected camera pose
  previous human / scene memory

output at frame t:
  alpha_t
  delta_xi_t
  T_corr_t = exp(alpha_t * delta_xi_t) @ T_hat_t
```

推理时禁止：

```text
读取 t+1 / future frames；
用 post-shot stable window 反修 boundary；
做整段 cam2 alignment；
做全局 BA / pose graph。
```

### 6.3 推荐训练范式

训练路线应写成：

```text
offline geometric teacher -> causal online student
```

teacher 可以使用未来帧估计 pseudo target：

```text
post-shot local gauge
delta_xi*_t
alpha*_t
scene reliability*_t
```

student 训练时只看当前和过去，但拟合 teacher target：

```text
student_input_t = {current cues, previous memory}
student_target_t = teacher_pseudo_label_t
```

这使得 teacher 能帮助定义“正确修正”，但最终部署仍是实时前馈。

### 6.4 当前需要验证的问题

下一步 teacher smoke test 关注：

```text
用 94..120 的 post-shot stable window 建立 cam2 local gauge；
反向生成 91 / 92 / 93 的 pseudo corrected pose；
检查是否比只修 91 更自然衔接；
检查 91 是否靠近 cam2 local gauge，但不被硬等同于后续帧。
```

### 6.5 第一版 teacher smoke test

脚本：

```text
scripts/build_post_shot_local_gauge_teacher.py
```

测试设置：

```text
input: output/human3r_h36m_18s
boundary: 91
target frames: 91 / 92 / 93
future stable window: 94..120
gauge anchor frame: 94
```

该测试是 offline teacher / oracle：

```text
允许使用未来稳定窗口估计 cam2 local gauge；
只用于 pseudo label / upper bound；
不能作为最终实时推理方法。
```

关键结果：

```text
raw:
  boundary_foot_jump      1.539
  boundary_stable_jump    1.611
  settle_camera_step_t    3.659
  settle_foot_jump        1.935

causal 91-only conservative multi-plane:
  boundary_foot_jump      0.099
  boundary_stable_jump    0.138
  settle_camera_step_t    0.898
  settle_foot_jump        0.340

offline teacher 91/92/93 -> post-shot local gauge:
  boundary_foot_jump      0.103
  boundary_stable_jump    0.177
  settle_camera_step_t    0.500
  settle_foot_jump        0.263
```

结论：

```text
teacher 保持了 91 的人体修正质量，同时显著改善了 91->92 的 settling 过渡。
这支持“91/92/93 应靠近 cam2 local gauge，但不能硬等同后续帧”的建模。
下一步应将 teacher 产生的 delta_xi / alpha / reliability 作为 pseudo label，训练只看当前和过去 memory 的 causal student。
```

### 6.6 第一版 causal student smoke test

脚本：

```text
scripts/overfit_causal_student_from_teacher.py
```

测试设置：

```text
input: output/human3r_h36m_18s
teacher target: output/human3r_h36m_18s_post_shot_local_gauge_teacher_corrected/post_shot_local_gauge_teacher_metrics.json
train frames: 90..120
positive targets: 91 / 92 / 93 teacher delta_xi
negative no-op frames: 90 and 94..120
```

student 输入保持 causal，只包含每个 frame `t` 和 `t-1` 可得信息：

```text
raw camera step / acceleration
camera-frame human motion
stable / foot camera-frame Chamfer
torso camera-frame orientation step
current-vs-previous scene plane / background summary
boundary time encoding
```

student 不读取：

```text
94..120 future stable window
teacher gauge anchor
future corrected poses
```

关键结果：

```text
raw:
  boundary_foot_jump      1.539
  settle_camera_step_t    3.659
  settle_foot_jump        1.935

offline teacher:
  boundary_foot_jump      0.103
  settle_camera_step_t    0.500
  settle_foot_jump        0.263

causal student:
  boundary_foot_jump      0.106
  settle_camera_step_t    0.522
  settle_foot_jump        0.275
```

teacher target fitting error：

```text
mean target translation error: 0.008
mean target rotation error:    0.080 deg
pre-boundary no-op delta norm: 0.025 on frame 90
```

结论：

```text
在单样本 overfit smoke test 上，offline teacher 生成的 correction 可以被只看当前/过去的 causal student 近似复现。
这支持 offline teacher -> online student 的训练路线。
下一步应在更多样本上生成 teacher pseudo labels，并训练非记忆化的 correction head / gate head。
```

## 7. Implicit Token Adapter Overfit

2026-05-25 进一步验证了更接近最终 V7 形态的 implicit token adapter：student 不再读取显式 SMPL joints / 背景平面 / future stable window，而是只读取 Human3R forward 内部 tokens 和 raw camera pose。

脚本：

```text
scripts/dump_v7_implicit_tokens.py
scripts/overfit_v7_implicit_token_student.py
scripts/export_v7_implicit_student_viewer_output.py
```

输入：

```text
pose token
human token
scene/image tokens
recurrent memory tokens
raw camera pose
```

输出：

```text
alpha_t
delta_t
delta_rotvec
r_human
r_scene
T_corr = exp(alpha_t * delta_xi_t) @ T_raw
```

该 adapter 不修改 decoder token，也不重新跑 pose head，而是直接预测 pose residual。

### 7.1 H36M boundary63

```text
case: h36m_test_boundary63
target frames: 63 / 64 / 65
best human_scene loss: 0.00000936
target_err_t: 0.00027
target_err_r: 0.0045 deg
noop_delta_t: 0.00105
```

### 7.2 H36M boundary91

```text
case: h36m_18s_boundary91
target frames: 91 / 92 / 93
best human_scene loss: 0.00001627
target_err_t: 0.00332
target_err_r: 0.0533 deg
noop_delta_t: 0.00258
```

### 7.3 当前判断

```text
1. Human3R 内部 token 中有 correction 信号。
2. human_scene 在两个 clip 上总 loss 最低，支持人体 token + 场景 token 联合使用。
3. pose-only 也能单 clip 拟合，说明下一步必须做 held-out validation，排除记忆化。
4. viewer 中 corrected camera / pointcloud / human mesh 相对 raw camera 有可见修正。
```

详细表格、viewer 路径和下一步计划见：

```text
docs/movie3r/v7/implicit_token_adapter_validation.md
```
