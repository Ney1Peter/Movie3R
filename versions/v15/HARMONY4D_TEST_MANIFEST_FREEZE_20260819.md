# Harmony4D test manifest 冻结记录（首次 forward 前）

日期：2026-08-19  
状态：**7 个 test 序列与 28 个 cases 已冻结；尚未运行任何 Harmony4D test forward / test metric**

## 1. 冻结边界

- 方法与阈值依据：`HARMONY4D_DEV_FREEZE_20260819.md`；
- runtime provenance 代码基线：`b84686a`；
- 协议：`Movie3R-Harmony4D-CrossShot-v1`；
- 固定 seed：`20260818`；
- 每个 case 为 75 pre + 75 post，共 150 帧；
- 每个序列固定 small / medium / large / extreme 各一个 camera pair；
- capture 按结构索引的冻结 SHA256 顺序选取首个 projection-valid 候选；
- capture、边界帧与相机对只使用 GT calibration/visibility，不读取 Movie3R 预测。

首次 test forward 前检查：

```text
test prediction directories = 0
test cases = 28
test sequences = 7
small / medium / large / extreme = 7 / 7 / 7 / 7
```

## 2. 单序列冻结清单

投影 gate 为 camera median ≤ 5 px 且 camera P95 ≤ 15 px。

| Archive entry | Frozen capture | Audit attempts | Cases | Max camera median px | Max camera P95 px | Manifest SHA256 |
|---|---|---:|---:|---:|---:|---|
| `test/01_hugging.zip` | `01_hugging/002_hugging` | 1 | 4 | `0.372` | `0.808` | `cd256f8215d048f350c2c4d6916d7231970aa490fbefad26b741d268097c8825` |
| `test/03_grappling2.zip` | `03_grappling2/028_grappling2` | 1 | 4 | `0.297` | `0.593` | `a9748540bec7443561ffcf5e94921afbea8b91705fdd7ece27298a79cd193065` |
| `test/05_sword2.zip` | `05_sword2/009_sword2` | 1 | 4 | `0.259` | `0.530` | `0ea9fbc0e987d612d29daa99149c568828f34ae698ffa1b107d56d5692dad80b` |
| `test/06_sword3.zip` | `06_sword3/004_sword3` | 1 | 4 | `4.980` | `10.499` | `d37e9049b7e50babf8a1c9ddbdf0c67ba73d22c714b7cbfad9c5192dafb61aca` |
| `test/08_ballroom2.zip` | `08_ballroom2/009_ballroom2` | 1 | 4 | `1.048` | `2.797` | `d2df8d4c735cb0ababac216a38bbf32595062a0535413484e84c266e9c56f8d1` |
| `test/15_mma4.zip` | `15_mma4/016_mma4` | 1 | 4 | `0.938` | `4.320` | `5db309ddc02bd2cbf4cf7551c6fc3b47cbcd31ed3f813e2052586cb4ded23706` |
| `test/16_mma5.zip` | `16_mma5/001_mma5` | 3 | 4 | `0.608` | `1.101` | `7448d5f0dae8a3252d75bdf94af83b269cf3fb82c10b64dd4486f357da19ec13` |

`mma5` 的前两个结构候选被透明拒绝：`cam15` 的 P95 分别为 18.821 px
与 25.391 px，超过预注册的 15 px gate。第三个候选通过全部坐标检查。
该选择发生在任何模型 forward 前，失败 attempt 与 traceback 均保存在 staging ledger，
不是根据 Movie3R 指标排除样本。

## 3. 全局 manifest

文件：

```text
versions/v15/harmony4d/protocols/h4d_cs150_test.jsonl
```

精确文件 SHA256：

```text
9c5cacfadb7a50d2618415b119c286cf23582c2d185e58b768323302e86638d2
```

`build_manifest.py` 从七个 selected audit 重新构建全局文件。冻结检查确认：

- 全局 28 行与七个单序列 manifest 的并集逐字段完全一致；
- 28 个 `case_id` 全部唯一；
- 七个 sequence 均为 4 cases；
- `selection_depends_on_model_result=false`；
- `boundary_index=75`、`clip_length=150`；
- 外层 `/data/wangzheng/iJCV-CODE/data/Harmony4D.zip` 只读保留；
- 七个 nested ZIP 均通过发布 size/SHA256 与完整 CRC 检查。

## 4. Test 执行约束

从该冻结点开始：

1. 不再修改 checkpoint、detector、ID 策略、gate 或阈值；
2. 不根据 test 指标删除或替换 case；
3. 技术异常可修复并断点重跑，数据不适配特例必须显式记入 ledger；
4. runtime 逐 case 保存 Git commit、runner SHA、manifest SHA、checkpoint SHA、
   GPU/CUDA/cuDNN、wall time、VRAM/RAM 与 causal contract；
5. 所有 17 个内部方法行由同一组 forward cache 派生；
6. 测试汇总固定使用 10,000 次 bootstrap 与 20,000 次 paired permutation test。

