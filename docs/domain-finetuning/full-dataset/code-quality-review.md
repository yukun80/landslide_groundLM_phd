# 追加质量复查代码

[返回文字复查](quality-review.md) · [返回代码目录](code.md)

## 名单与来源

[quality_review.py](../../../src/xiaomi_lens/full_dataset/quality_review.py)：

- `normalized`：忽略空格标点比较重复句。
- `current_text`：从实际采用记录或复用答案提取两版文字。
- `reasons_for`：形成无外观、重复及风险复查原因。
- `inventory`：固定全量来源、候选范围和18条试跑。
- `load`：核对准备数据与清单身份。
- `merged_entries`：读取已采用的定稿并核对来源。
- `status`：分别统计旧候选清单的保存数、已采用定稿数和剩余数。

## 实际读图

[quality_generate.py](../../../src/xiaomi_lens/full_dataset/quality_generate.py)：

- `protocol`：记录本轮提问规则身份。
- `quality_paths`：生成原像素局部核对图，保留原图和定位图。
- `quality_issues`：拒绝当前范围外的纹理、形状和泛泛背景句。
- `receipt`：保留真实返回，失败最多实际重试三次。
- `phase`：光学与辅助分别生成，再逐句判断；空候选另行核查。
- `phases`：批量计算独立请求，按真实请求对应返回。
- `run`：逐条保存；正式批量要求试跑看图记录。

## 核对与交付

[quality_delivery.py](../../../src/xiaomi_lens/full_dataset/quality_delivery.py)：

- `apply_visual`：保存实际看图后的追加修订，不覆盖既有记录。
- `validate_record`：核对原句、实际图片、模型返回和位置面积。
- `final_sample_ids`：固定按来源和处理结果抽查的编号。
- `publish`：全部记录和最终抽查齐全后导出，等待实际读取检查。
- `validate_final`：逐条核对最终清单与采用来源，保护非文字内容。

[命令入口](../../../tools/review_full_dataset_quality.py)选择阶段并取得任务锁；
[边界检查](../../../tools/check_quality_review.py)覆盖缺项、失败、否定判断和来源隔离。
