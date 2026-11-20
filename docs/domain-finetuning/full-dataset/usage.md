# 本地接续操作

[返回总览](README.md) · [上一节：文字](descriptions.md) · [下一节：代码](code.md)

在项目根目录使用已有 `qwen3vl` 环境，不另建环境或下载模型。
以下是文字生成，不是微调；模型生成时会使用显卡，训练格式检查使用普通处理器。

## 查看现用清单或重做本批合并

在根目录运行 `scripts/full_dataset.sh tools/merge_description_final.py`。输入为全量准备清单、对应的原采用记录、逐图审查表和 1,314 张定稿。首次先在独立目录验证，再开放训练、验证和测试三个清单入口。再次运行会核对输入及现用清单，并复用同一批文字已通过的读取结果。

`python tools/full_dataset_status.py` 同时显示已采用的修改数和全量待确认数。合并及清理明细在数据目录的 `review/merged_descriptions/`。使用当前合并版进行训练时采用[手动训练入口](../full-training.md)，仍保留待确认状态。需要全量复核时，按现用答案和待确认项继续；旧候选发布命令不覆盖本批已采用的文字。

下列为历史生成方式，可按[质量复查流程](quality-review.md)处理；下列为原接续流程，不能绕过新复查发布。

## 首次恢复与试跑

```bash
scripts/full_dataset.sh tools/describe_full_dataset.py init
scripts/full_dataset.sh tools/describe_full_dataset.py audit
scripts/full_dataset.sh tools/describe_full_dataset.py pilot
```

`init` 固定模型、规则、旧文字与27条Codex记录；`audit` 检查全部旧本地答案，记录重新查看理由。
`pilot` 对固定12张训练图实际生成并再次核查，逐条保存在 `review/local_resume/samples/`。
当前会话实际打开每张独立图片，核对答案后手写查看说明，再保存试跑放行记录：

```bash
scripts/full_dataset.sh tools/describe_full_dataset.py approve-pilot --input 实际核查记录.json
scripts/domain.sh tools/check_local_resume_pilot.py
scripts/domain.sh tools/check_full_dataset.py encode --pilot --device cpu
```

查看说明包含12个编号、当前答案摘要、实际查看的全部图片及其摘要、判断和说明。
脚本不代写看图声明；试跑不通过时不得开始全量任务。
需要改规则时，用 `revise-pilot --input 实际失败核查.json` 归档失败轮次后重做；放行后规则冻结。

## 批量执行和中断恢复

```bash
scripts/full_dataset.sh tools/describe_full_dataset.py run
scripts/domain.sh tools/full_dataset_status.py
scripts/domain.sh tools/complete_full_dataset.py
```

`run` 跳过本轮已完成项；每条保存真实模型返回，失败条目保持缺失并报告。
只允许一个生成任务；显存不足时缩小同批数量，不删辅助图片，单条仍失败则停止该次任务。
`complete_full_dataset.py` 检查放行后接续生成、统一导出并执行全部检查；失败保持未完成。
此入口允许后台接续本地模型，不启动Codex、在线接口或正式训练。

## 分步最终检查

```bash
scripts/full_dataset.sh tools/describe_full_dataset.py publish
scripts/domain.sh tools/check_full_dataset.py raw
scripts/domain.sh tools/check_full_dataset.py encode --device cpu
scripts/domain.sh tools/check_full_dataset.py final
scripts/domain.sh tools/write_full_dataset_validation.py
scripts/domain.sh tools/check_full_dataset_docs.py
```

仅全部必修项完成后允许导出。最终检查通过、`summary.json` 中 `complete: true` 才能用于训练。
历史Codex入口保留，但当前模式下不能继续提交或发布，防止两种流程同时改写当前数据。
