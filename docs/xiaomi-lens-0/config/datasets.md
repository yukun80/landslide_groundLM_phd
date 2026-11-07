# 数据清单

[返回上级](README.md)

| 文件 | 用途与关键内容 |
|---|---|
| [stage1.yaml](../../../Xiaomi-Lens-0-raw/examples/train/stage1.yaml) | 指向人造数据 `toy/stage1.jsonl` 和 `toy/images`，标签为 segmentation |
| [stage2.yaml](../../../Xiaomi-Lens-0-raw/examples/train/stage2.yaml) | 训练与评估都指向 `toy/stage2.jsonl`；评估取前 16 条。仅供检查流程，不能当独立测试集 |
| [sample.yaml](../../../Xiaomi-Lens-0-raw/examples/train/sample/sample.yaml) | 指向同目录两条真实样本及 images 子目录，可用于理解读取过程 |

共同字段：`name` 是日志用名称；`path` 是样本文件；`image_folder` 是图片根目录；`tag` 是用途标签；`sampling_strategy` 支持全部、前若干、后若干或随机若干，也可用百分比。相对路径以清单所在目录为起点，不是任意终端所在位置。
