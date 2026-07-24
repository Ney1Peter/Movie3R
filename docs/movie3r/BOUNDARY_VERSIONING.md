# Movie3R Versioning

## 正式版本

仓库从现在起只维护三个对用户可见的正式版本号：

| 正式版本 | 目录 | 定位 | 状态 |
|---|---|---|---|
| Movie3R-Learned V9.0 | `versions/v9/` | 4-frame AABB 学习式 correction | 冻结训练版 |
| Movie3R-Single V12.0 | `versions/v12/` | 单人 short-shot similarity re-anchoring | 当前单人主版 |
| Movie3R-Multi V13.0 | `versions/v13/` | 多人 shared-Boundary | GT-ID Oracle 研究版 |

V12 和 V13 是在整理时选定的正式发布编号，因此不要求和此前实验日志连续。后续如果
冻结新的独立产品版本，应从 V14 开始顺序增长；普通消融不再占用新的正式整数版本。

## 历史实验编号

过去的编号按研究问题增长，曾出现 V11、V14、V20 以及更早的 V46/V47/V53。这些
编号不是多个同时维护的软件版本。它们现在只出现在版本内部的报告、脚本和缓存字段中。

| 历史编号 | 归属正式版本 | 含义 |
|---|---|---|
| V9 | V9 | 已训练 relation-correction/LoRA 模型 |
| V10.1 Fixed Explicit | V12 | cut 后显式 coarse Boundary |
| V11.1 / V16 | V12 | bounded torso-motion rotation |
| V11.4 | V12 | DA3/Keypoint fused uniform shot scale |
| V14.1-V14.7 | V12 | reset、continuity、coupled-root、统一评测和冻结审计 |
| V20 Phase 1 v2 | V13 | strict GT-ID 多人几何共识验证 |
| V46/V47/V53 | V12 历史 | 旧名称，分别映射到 contact、rotation、uniform similarity 实验 |

## 使用规则

1. 对外方法名、目录、运行入口和 Git tag 使用 V9、V12、V13。
2. 历史文件不批量改名，避免旧报告、结果 JSON 和实验命令失去可追溯性。
3. 正式代码不得再散落为根目录下的 `v14_*.py` 或 `v20_*.py`；它们分别归入
   `versions/v12/experiments/` 和 `versions/v13/`。
4. V12 适合 short shot 和稀疏 cut，不宣称无限长度 multi-shot mapping。
5. V13 仍使用 GT identity Oracle，不宣称已经完成可部署多人 Re-ID。
