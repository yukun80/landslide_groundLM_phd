# 检查清单、不确定性与交付

> 本页保留先前工作记录；当前范围与写法见[533条复核](revision-2026-11-12.md)。
[返回描述改进 v2 入口](README.md) · [实施流程](workflow.md) · [规则与辅助图口径](rules.md) · [训练读取](training.md)

## 一、实际运行通过的检查

| 检查 | 命令 / 产物 | 结果 |
| --- | --- | --- |
| 原答案与掩膜核验（阶段 B） | `desc_v2.py verify` → `originals_verified.json` | 1000 条逐字一致，位置/bbox/面积与像素计算一致 |
| 辅助图规格 | `desc_v2.py aux-spec` → `aux_spec_verified.json` | 三来源视图数、波段、显示范围核实；单位未确认 |
| 试点三轮 | `desc_v2.py pilot-review` → `pilot_review.md` | 12 条 `verified` / 0 `failed` |
| 逐批闸门 | `desc_v2_batch.py check --batch 0..9` | 10 批各 100 条全部 `verified`，134 张拼图无缺失 |
| 全量静态检查 | `check_desc_v2.py --scope full` → `checks.json` | **24 通过 / 0 失败**（`apply` 前后各跑一次） |
| 文档短文与入口 | `check_desc_v2_docs.py` | 6 篇短文，行数上限、本地链接、两个 README 入口通过 |
| 工具语法 | `python -m py_compile` × 5 | 通过 |
| 覆盖二次断言 | `desc_v2.py apply` → `apply_report.json` | train 800 / val 100 / test 100 原子覆盖；新旧字节哈希已记录 |
| 覆盖后复核 | 逐行比对三份 jsonl | 1000 条 id 唯一、17 字段顺序一致、`answer`/`answer_optical` 与 `records` 一致、非答案字段指纹全一致 |
| loader 数据冒烟 | `smoke_desc_v2_loader.py --mode data` → `loader_smoke.json` | **1000 行 PASS / 0 错**（用 `domain.py` 权威 `training_answer()` 复核数值句与后缀） |

`checks.json` 覆盖：1000 条 id 唯一且全覆盖、`records` 全 `verified`、非答案字段指纹未变、3900 张 PNG 与 `display_ranges.json` / `screening.jsonl` / `summary.json` 字节未变、必需看图角色与证据齐全、光学版无辅助来源词、两版相同条件正确、活动判别只在多图版且有授权留痕、后缀逐字等于程序值且官方解析器得 1 个有效框、禁止词/长度/句数、套话重复、`select_answer` 三分支、`domain_runtime.py:127` 未变、`domain_train.py` 用 `select_answer_row` 注入答案、规则与 spec 一致、视图产物齐全、初稿/未人工确认留痕如实。

## 二、尚未验证 / 不确定

- **真实 tokenizer 截断**：`--mode data` 已在本机通过；但 `--mode encode` 需要 `torch` / `transformers` 与本地 processor，本机不具备，无法执行。字符数 ≤200 不等于 token 数 ≤`maximum_answer_tokens=160`，也不能直接证明 `maximum_sequence_length=2048` 内不截断。需在 WSL 侧运行：
  ```bash
  scripts/domain.sh tools/smoke_desc_v2_loader.py --mode encode --limit 20
  ```
  该模式只加载 processor 调 `build(training=True)` 检查编码长度与答案 token，不加载模型权重、不前向、不写 checkpoint、不启动正式训练；结果写 `loader_smoke.json`。运行前请确认该脚本的 `local_files_only` 能找到本地 processor。
- **外观描述准确性**：自动检查只验证格式与部分一致性，不能证明 533 条外观句都与图片相符，需要人工逐图复核。
- **活动判别正确性**：唯一 1 条定性判别是用户授权的领域解释，本地编码值文件无法核实 `|v|>20 mm/yr`，不等于已证实的活动性结论。
- **视觉模型初稿**：`view_checked_by` 记为视觉模型，`human_confirmed=false`、`draft_is_fact=false`、`needs_recheck=true`；逐样本 `uncertain` 记录看不清/证据不足。
- **InSAR 物理量**：单位、scale/offset、零点、符号未确认；0–255 编码值与 `display_ranges.json` 的 `[12,130]` 都不能换算成 `m/yr`。

## 三、不可声称项

- 未启动正式训练，未做独立评估，**不声称模型效果提升**。
- 不声称人工逐图确认完成，不声称所有外观描述准确，不声称活动判别正确。
- 「描述更丰富」（533 条新增外观句、320 条两版差异）与「模型效果更好」是两件事。
- 不利用验证/测试集预测成绩反复调整描述规则；本次未做任何模型预测。

## 四、人工复核入口

- 逐样本记录：`datasets/landslide_lens_positive_v1/description_v2/records.jsonl`（含 `original_answer`、两版新答案、`images_viewed`、`evidence`、`uncertain`）。
- 逐样本视图：`.../description_v2/views/<id>/`（`overlay.png`、`optical_at_input.png`、`aux_i_at_input.png`、`contact_sheet.png`、`crops/target_zoom.png` 及图例）。
- 分批复核拼图与行号映射：`.../description_v2/sheets/batchX/index.json` + `sheet_*.png`。
- 新旧对照：`docs/description-v2/examples.md` 与 `.../description_v2/pilot_review.md`。
