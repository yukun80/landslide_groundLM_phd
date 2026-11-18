# 文字质量追加复查

[返回总览](README.md) · [使用方式](usage.md) · [代码说明](code-quality-review.md)

## 范围与状态

2026-11-18处理生成任务失败后的10条缺项，全部补齐；另保存逐张原图核查后的修订，不覆盖原返回。
全量18,987条按当前采用来源筛查，包括999条复用答案与27条旧会话答案。
重复句、无光学外观、泛泛背景对比、小目标纹理、颜色疑点及10条修复样本去重后，需要复查11,876条。
这不是错误样本数量；重复句可能正确，留空也可能合理。
旧 18 条试跑未放行。此后另行定稿的 1,314 张已合并，实际改写 1,174 张。其中 1,245 张与旧候选清单重合，后续生成会跳过已采用定稿。全量仍待确认的 12,392 张以[现用全量清单说明](description-audit.md)为准。

## 保存与核查规则

- 原数据、划分、模型及旧答案只读；新记录在 `review/quality_review/`。
- `baseline.jsonl` 固定原采用答案及文件身份；`candidates.jsonl` 固定逐条复查原因。
- 光学句忽略空格标点后，至少20个样本出现时纳入复查。
- 没有光学外观的样本全部复查，包含只有辅助句的样本。
- 先检查18条试跑，覆盖三类来源及主要风险；通过实际图片核查后才运行全量。
- 失败试跑保存在 `pilot_rounds/`，不作为合格结果。
- 当前试跑将光学补充收窄为可确认颜色，不补形状、纹理或地物名称；辅助信息另看图核查。
- 每句单独实际核查；无候选仍保存第二次真实返回，但不制造通过句。
- 无效返回最多实际重试三次，仍失败则保留失败，禁止默认为完成。
- 实际看图修订标明Codex查看，不冒充本地模型返回，也不称人工专家确认。

## 命令

```bash
scripts/full_dataset.sh tools/review_full_dataset_quality.py inventory
scripts/full_dataset.sh tools/review_full_dataset_quality.py pilot
scripts/full_dataset.sh tools/review_full_dataset_quality.py approve-pilot --input 实际看图记录.json
scripts/full_dataset.sh tools/review_full_dataset_quality.py run
scripts/domain.sh tools/full_dataset_status.py
```

放行记录必须包含18条的答案身份、实际查看图片及具体说明，不是自动通过标记。
`run` 重复执行只处理未保存项；任务锁避免入口相互覆盖。
已保存记录不能直接重写，失败试跑先归档，再重新生成。

## 最终交付

以下是原候选清单的发布流程。已合并定稿且仍有全量待确认项时，旧发布入口会拒绝覆盖现用清单。

在原流程中，全部候选完成后，按来源和处理结果各抽最多3条实际查看，保存 `final_visual_review.json`。
发布要求完整记录、试跑及最终抽查都通过；随后仍须运行数据与训练读取检查。

```bash
scripts/full_dataset.sh tools/review_full_dataset_quality.py publish
scripts/domain.sh tools/check_full_dataset.py raw
scripts/domain.sh tools/check_full_dataset.py encode --device cpu
scripts/domain.sh tools/check_full_dataset.py final
```

复查未结束时，上述发布不能通过；旧入口禁止绕过待复查项。
修订只作用于新全量数据的文字副本，原1000条答案和独立评测原件仍受保护。

## 验证范围

9项追加复查边界测试与14项原接续边界测试通过，不代表文字准确率。
10条缺项已实际生成并看图核对；颜色修订与原位置面积核对通过。
全量生成、最终读取与抽查尚未通过，不能把试跑当作交付完成。
