# Multi-THuMBS 公开协议、数据重合与可执行对标审计

> 核验日期：2026-08-01
>
> 论文：`/data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf`，arXiv:2607.01626v1
>
> 范围：只核对公开资源、本地数据与已有 CPU 评测；没有使用 GPU，也没有重新训练或推理。

## 1. 可以立即采用的结论

本地数据与 Multi-THuMBS 在**数据集层面确定重合**：双方都使用 EgoHumans；本地
`001_legoassemble` 也能由 EgoHumans 官方仓库的同名配置、相机命名和人物命名确认归属。

但在**论文 benchmark 的具体 capture/split 层面不能确认重合**。Multi-THuMBS 主文只说它从
EgoHumans、EgoBody、Harmony4D 的多视角序列构造 multi-shot benchmark，没有公布所选
sequence、camera pair、cut timestamp、clip 数或 manifest。因此当前本地实验只能准确称为：

```text
same source dataset / locally constructed cross-camera cuts /
protocol-matched-as-far-as-public-information
```

不能称为 Multi-THuMBS 官方 split，也不能仅凭数值大小宣称正式击败论文。

当前仍值得直接测试，因为它已经给出同源 EgoHumans 的真实跨域证据，并能用来冻结我们自己的
公开协议；待作者代码/补充材料发布，只需替换 manifest 和官方 metric adapter，不必重做模型。

## 2. Multi-THuMBS 官方资源公开状态

截至核验时：

- arXiv API 仅指向 v1 PDF 和项目页；项目页为
  `https://on-jungwoan.github.io/projects/multi-thumbs/`；
- 项目页资源栏只有 arXiv 和 Video，没有 Code 或 Supplementary 链接；BibTeX 仍显示
  `Coming soon!`；
- 第一作者公开 GitHub 账号 `On-JungWoan` 没有 Multi-THuMBS 仓库；
- GitHub repository 搜索没有找到官方 Multi-THuMBS 实现；
- arXiv v1 source 包只有 `main.tex`、五个正文 section、五个 table 和 figure/style 文件，
  没有 supplementary、评测代码或数据 manifest。

论文第 11 页却明确说 dataset construction、implementation settings 和 evaluation protocol
在 supplementary。因此目前缺失的不是本地查找，而是作者尚未公开关键协议。

## 3. 本地 capture 的归属与内容

本地路径虽然历史命名为 `EgoBody`：

```text
/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble
```

但它正式属于 EgoHumans，证据为：

1. EgoHumans 官方仓库存在
   `egohumans/configs/legoassemble/001_legoassemble.yaml`；
2. 该配置声明 `SEQUENCE: '001_legoassemble'`，人物为
   `aria01/aria02/aria03`，与本地逐字一致；
3. 官方下载说明的数据树是
   `colmap + ego/aria* + exo/cam* + processed_data`，与本地同构；
4. 官方 EgoHumans test big-sequence 包含 `02_legoassemble`。但 Multi-THuMBS 是否从中选了
   `001_legoassemble`，公开材料没有说明。

本地实际内容核验：

- 8 路 exo camera：`cam01` 到 `cam08`；
- 每路 601 张同步图像：`00001.jpg` 到 `00601.jpg`；
- `processed_data/smpl` 和 `processed_data/poses3d` 各 601 个时间戳；
- SMPL 标注每帧含 `aria01/aria02/aria03`，字段包括 `global_orient`、`transl`、
  `body_pose`、`betas`、`vertices`、`joints`；
- `poses3d` 每人是 `17 x 4`；SMPL mesh 为 6890 vertices，现有 evaluator 使用 24 joints；
- 当前 `/data/wangzheng/iJCV-CODE/data` 下未发现第二个可运行的 EgoHumans/EgoBody 或
  Harmony4D RGB+GT capture。

## 4. 当前本地 cut 与论文 split 的关系

已有三条 15-frame chain：

```text
cam01 296-300 -> cam06 300-304 -> cam07 304-308
cam02 176-180 -> cam05 180-184 -> cam08 184-188
cam03 416-420 -> cam04 420-424 -> cam01 424-428
```

每个 boundary 两侧重复同一个 dataset timestamp。这相当于把同步多相机视图人为编辑成 shot
change，非常适合测试“跨相机 gauge 是否接上”，也符合论文所述由 multi-view sequence 构造
multi-shot benchmark 的总体思路。但以下内容都不是论文确认事实：

- 论文是否使用了这一个 capture；
- 是否使用了同样的 camera pair 和 cut timestamp；
- 是否在 cut 两侧重复同一 timestamp；
- 训练/验证/测试划分和 clip 聚合权重。

所以它可以作为同数据源 overlap 测试，不能冒充 exact split reproduction。

## 5. 论文公开指标与缺失公式

主文明确公开的只有：

| 指标 | 论文说明 | EgoHumans 参考值 |
|---|---|---:|
| W-MPJPE ↓ | initial-frame alignment，轨迹一致性 | 279.0 |
| WA-MPJPE ↓ | trajectory-level alignment，shape accuracy | 166.0 |
| MPJPE ↓ | pose | 228.3 |
| MPVPE ↓ | pose/mesh | 262.2 |
| Accel ↓ | temporal smoothness | 27.3 |
| ATE ↓ | camera localization | 0.7 |
| IDs ↓ | identity switches | 0.97 |

主文没有公开：

- W/WA 使用 SE(3) 还是 Sim(3)，initial 是 1 帧还是 2 帧，是否分 chunk；
- MPJPE/MPVPE 是否 pelvis-align、joint/vertex topology、visibility 和漏检惩罚；
- Accel 的差分公式、坐标系、fps 和单位；
- ATE 的 SE(3)/Sim(3)、尺度、单位与 clip aggregation；
- IDs 对 miss/FP/进入退出的处理和小数结果的聚合方式；
- PCK*、Jitter、Foot Sliding 的阈值、normalizer 和实现。

因此本地 evaluator 中的公式必须标为 provisional，不能写成 Multi-THuMBS 官方代码实现。

## 6. 已可直接运行的同源对比

### 6.1 公式自测

```bash
cd /data/wangzheng/iJCV-CODE/Movie3R
.venv/bin/python versions/v14/eval_multithumbs_protocol.py --self_test
```

2026-08-01 实测通过：`>> self-test passed`。

### 6.2 旧 raw Human3R 的论文命名指标诊断

```bash
cd /data/wangzheng/iJCV-CODE/Movie3R
.venv/bin/python versions/v14/eval_multithumbs_protocol.py --device cpu
```

该入口使用三个现有 15-frame cache，不含 B0/BRTC。它按稳定 GT identity 分轨，报告本地
GVHMR/Human3R-style W、WA、pelvis MPJPE/MPVPE、两种 Accel、Sim(3) camera-center ATE
与 native track-ID switches，并明确记录所有口径假设。

### 6.3 当前 B0 与 B0+BRTC 的同-forward 连续链结果

已有 CPU 结果，不需重新 forward：

| Method | W | WA | pelvis MPJPE | pelvis MPVPE | ATE | IDs/stream |
|---|---:|---:|---:|---:|---:|---:|
| raw reset | 1088.2 | 405.1 | 109.3 | 130.0 | 1.848 | 5.67 |
| B0 | 350.6 | 235.2 | 109.3 | 130.0 | 0.119 | 1.00 |
| B0+BRTC | 314.1 | 202.5 | 109.3 | 130.0 | 0.119 | 1.00 |

BRTC 相对 B0 的 W/WA 改善是真实的同-forward内部增益；camera bit-exact 不变，所以 ATE
相同。刚性人体平移会被 pelvis alignment 抵消，所以 MPJPE/MPVPE 不变也是预期结果。

与论文数字不能直接判胜负：本地只有一个 capture 的三条短链、visibility/漏检/聚合和官方
split 未知。尤其本地 MPJPE/MPVPE 比论文小不能解释为更强，它们只统计成功匹配帧，且 pose
口径可能不同。

## 7. 最小公平对标计划

1. 现在冻结并版本化本地 manifest、GT identity association、漏检/FP 规则和 evaluator；
2. 对每个方法保存同一 forward 的逐帧 camera、稳定/native identity、24 joints、6890
   vertices 与 visibility，不允许混拼不同 checkpoint cache；
3. 同时报两层结果：
   - `paper-named provisional`：W/WA/MPJPE/MPVPE/Accel/ATE/IDs；
   - `Movie3R fixed-world`：root、world joint/vertex、pair distance/vector 和 harm audit；
4. 以 B0、frozen BRTC 和后续精对齐方法做同输入、同检测、同关联、同聚合比较；
5. 作者发布 supplementary/code 后，先锁定其 commit，再只替换 official split/metric adapter，
   重跑所有方法；此时才使用 `<279.0/<166.0/<228.3/<262.2/<27.3/<0.7/<0.97`
   作为 EgoHumans 正式胜负线。

## 8. 证据路径

本地：

```text
/data/wangzheng/iJCV-CODE/paper/Multi-THuMBS.pdf
versions/v14/eval_multithumbs_protocol.py
versions/v14/eval_brtc_multithumbs_egohumans.py
versions/v14/docs/V14_MULTITHUMBS_AUDIT_AND_EGOHUMANS_BASELINE_20260731.md
versions/v14/docs/V14_BRTC_MULTITHUMBS_EGOHUMANS_20260801.md
output/v14/fine_alignment_research/multithumbs_protocol/
output/v14/fine_alignment_research/brtc_multithumbs_egohumans/
/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble
```

公开网页：

```text
https://arxiv.org/abs/2607.01626
https://on-jungwoan.github.io/projects/multi-thumbs/
https://github.com/On-JungWoan
https://github.com/rawalkhirodkar/egohumans
https://github.com/rawalkhirodkar/egohumans/blob/main/egohumans/configs/legoassemble/001_legoassemble.yaml
https://github.com/rawalkhirodkar/egohumans/blob/main/assets/DOWNLOAD.md
```
