# 范围与实施流程

> 本页保留先前工作记录；当前范围与写法见[533条复核](revision-2026-11-12.md)。
[返回描述改进 v2 入口](README.md) · [规则与辅助图口径](rules.md) · [检查结论](checks.md)

## 一、范围

- 数据集：`datasets/landslide_lens_positive_v1`，train 800 + val 100 + test 100 = 1000 条。
- 不重新抽样、不改划分、不增加纯背景样本、不修改 `Xiaomi-Lens-0-raw`。
- 3900 张原始 PNG、`mask.png` / `valid.png`、`display_ranges.json` 字节未变（已做逐文件哈希）。
- 只改三份 jsonl 的 `answer`，新增 `answer_optical`；其余字段按指纹逐条比对未变。
- 不启动正式训练、不做独立评估。

## 二、每条样本实际查看什么

按固定列序拼成一张「复核行」，列即下列面板：

| 列 | 内容 | 作用 |
| --- | --- | --- |
| `opt raw` | 无覆盖标注的真彩光学原图 | 看真实地表外观，不把叠加色当颜色 |
| `mask` | 二值掩膜 | 确认区域 |
| `overlay` | GT 叠加图 | 确认要描述的位置；品红只是标注显示色 |
| `opt@input` | 模型接收尺寸光学图 | 外观细节必须在此尺寸下仍有依据 |
| `auxN raw` / `auxN@in` | 每张实际辅助图的原图与接收尺寸图 | 按真实波段含义解读 |

局部放大 `crops/target_zoom.png` 只用于定位，不产生新细节。

## 三、分批看图流水线

- `tools/desc_v2_batch.py sheets` 按来源分组生成拼图，行号与 `id` 写入 `index.json`，不靠文件顺序对应：
  - GDCLD：4 列 × 9 行 / 张；landslide4sense：6 列 × 8 行 / 张；multimodal：8 列 × 7 行 / 张。
- 共 10 批，每批 100 条；10 个 `sheets/batchX/` 目录合计 134 张拼图。
- `metrics` 由程序算掩膜像素、覆盖率、光学均值/方差、各辅助图均值/方差/黑像素比例，写入 `obs/batchX_metrics.json`，作为逐样本客观证据。
- 逐行看图后，只把「确实看得见」的句子写入 `obs/batchX_observations.json`；看不清的样本不写外观句。
- `drafts` 把紧凑观察展开成完整的 `viewed` / `crops_viewed` / `evidence` / `uncertain` 结构，再交给 `desc_v2.py write-drafts` 走同一套校验，不绕过闸门。

## 四、状态机与断点续跑

- `records.jsonl` 一行一条记录，按 `id` 关联（追加写，取最后状态），不靠行号。
- 状态：`pending → viewed → drafted → verified`，失败记 `failed`。
- 只有 `verified` 且 `suffix_verified` 为真才能进入最终 `apply`。
- 试点 12 条（3 来源 × 4 种情况）先做三轮试写，用户确认并冻结规则后才开始全量；已 `verified` 的试点样本在后续批次草稿中跳过，不重复写。
- 每批独立 `check --batch X`，中断后可从 `records.jsonl` 与 `last_batch` 继续。

## 五、数量

| 批次 | 总数 | 新增外观句 | 两版不同 |
| --- | --- | --- | --- |
| batch0 | 100 | 48 | 0 |
| batch1 | 100 | 41 | 0 |
| batch2 | 100 | 54 | 0 |
| batch3 | 100 | 60 | 34 |
| batch4 | 100 | 53 | 51 |
| batch5 | 100 | 56 | 56 |
| batch6 | 100 | 52 | 52 |
| batch7 | 100 | 54 | 54 |
| batch8 | 100 | 61 | 36 |
| batch9 | 100 | 54 | 37 |
| **合计** | **1000** | **533** | **320** |

证据不足（目标过小、全黑、低对比）时按规则不补写外观句，两版等于原答案，不用统一套话填满。
