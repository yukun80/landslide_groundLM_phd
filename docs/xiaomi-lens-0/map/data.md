# 训练数据与示例

[返回上级](README.md)

步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。

下表两张真实样例图片仅在本地保留，不随仓库上传；新下载仓库需先按[上传内容说明](../../maintenance.md)准备图片，再打开对应链接或运行真实样例。

| 文件 | 阅读步骤 | 用途与使用时机 |
|---|---|---|
| [examples/train/README.md](../../../Xiaomi-Lens-0-raw/examples/train/README.md) | 4 | 训练数据格式英文说明；理解示例、配置与启动方法。 |
| [examples/train/README_CN.md](../../../Xiaomi-Lens-0-raw/examples/train/README_CN.md) | 4 | 训练数据格式中文说明；配合真实样本逐字段阅读。 |
| [examples/train/make_toy_data.py](../../../Xiaomi-Lens-0-raw/examples/train/make_toy_data.py) | 4 | 画简单形状，并生成对应的问题、答案和区域标注，用于理解数据格式与检查训练流程。 |
| [examples/train/sample/images/COCO_train2014_000000580562.jpg](../../../Xiaomi-Lens-0-raw/examples/train/sample/images/COCO_train2014_000000580562.jpg) | 4 | 灰色自卸卡车示例图片，与 sample.jsonl 第二条对应；查看真实训练输入时用。 |
| [examples/train/sample/images/COCO_train2014_000000581739.jpg](../../../Xiaomi-Lens-0-raw/examples/train/sample/images/COCO_train2014_000000581739.jpg) | 4 | 右侧花瓶示例图片，与 sample.jsonl 第一条对应；本次也用于预测前后对照。 |
| [examples/train/sample/sample.jsonl](../../../Xiaomi-Lens-0-raw/examples/train/sample/sample.jsonl) | 4 | 两条真实图文训练示例；参考答案与区域按顺序对应，字段详解见配套文档。 |
| [examples/train/sample/sample.yaml](../../../Xiaomi-Lens-0-raw/examples/train/sample/sample.yaml) | 4 | 附带真实样本的清单；解析同目录样本和 images 中图片。 |
| [examples/train/stage1.yaml](../../../Xiaomi-Lens-0-raw/examples/train/stage1.yaml) | 4 | 人造数据第一阶段清单；生成 toy 数据后使用。 |
| [examples/train/stage2.yaml](../../../Xiaomi-Lens-0-raw/examples/train/stage2.yaml) | 4 | 人造数据第二阶段清单；评估复用训练前 16 条，仅供流程检查。 |
| [src/xiaomi_lens/train/data/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/__init__.py) | 4 | 汇总数据相关的对外名称，调用时再去相应文件查找实现。 |
| [src/xiaomi_lens/train/data/io.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/io.py) | 4 | 读取数据清单，把样本文件和图片位置整理成统一形式。 |
| [src/xiaomi_lens/train/data/jsonl_index.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/jsonl_index.py) | 4 | 只记住每条样本在文件中的位置，需要时再读取，避免一次载入全部数据。 |
| [src/xiaomi_lens/train/data/preprocess.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/preprocess.py) | 4 | 把单条图文样本变成数字输入，找出要学习的回答位置，并对齐正确区域。 |
| [src/xiaomi_lens/train/data/render.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/render.py) | 4 | 把不同来源的对话整理成模型认识的格式，并对齐图片占位符。 |
| [src/xiaomi_lens/train/data/segmentation.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/segmentation.py) | 4 | 把压缩存储的区域标注还原成黑白像素图，并缩放到模型处理图片的大小。 |
| [src/xiaomi_lens/train/data/sft_format.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/sft_format.py) | 4 | 把框坐标与区域标记拼成训练答案，保证文字中的区域顺序与标注一致。 |
