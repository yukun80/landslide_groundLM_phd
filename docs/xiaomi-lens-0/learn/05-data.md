# 先认识训练数据

[返回上级](../README.md)

按这个顺序打开：

1. [真实小样本](../../../Xiaomi-Lens-0-raw/examples/train/sample/sample.jsonl) 与 [字段说明](../config/README.md)：理解一行就是一条图文对话及其正确区域。
2. [make_toy_data.py](../../../Xiaomi-Lens-0-raw/examples/train/make_toy_data.py)：看简单几何形状怎样产生图片、问题和不含歧义的正确区域。
3. [io.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/io.py) → [jsonl_index.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/jsonl_index.py)：定位数据文件，只在需要时读一行。
4. [render.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/render.py)：对齐图片占位符，把问题与答案排成模型熟悉的对话。
5. [preprocess.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/preprocess.py) 的 `encode_sample`：转数字、决定哪些回答位置需要学习、对齐每个区域与图片。
6. [segmentation.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/segmentation.py)：还原压缩的正确区域；[sft_format.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/sft_format.py) 负责拼答案格式。
7. [collator.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/collator.py)：多条样本补齐长度后放成一批。

正确区域是训练题的“参考答案”，不是模型的预测结果。`segmentation[k]` 对应助手回答中第 `k` 个区域标记，连思考部分的区域也按出现顺序计入。用户消息里引用的同名标记不应挤占助手标注的位置。

`labels=-100` 是“不考这个位置”。问题、补齐位置和部分模板前缀不参与文字评分。太长的样本需要截断，但不能留下半张图的占位符；被截掉的区域仍保留顺序记录，只是失去有效资格。

自测：两张不同大小的图片各有一个区域，能否统一按第一张图的大小制作标注？**不能。每个区域要先对齐它所属图片，再为同一批处理补齐。**

[上一节](04-mask.md) · [下一节](06-training.md)
