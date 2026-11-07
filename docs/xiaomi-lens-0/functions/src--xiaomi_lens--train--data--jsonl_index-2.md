# src/xiaomi_lens/train/data/jsonl_index.py（第 2 页）

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/jsonl_index.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/jsonl_index.py)。搜索下面的名称可定位原文件。

- `JsonlIndexDataset.__iter__`：按当前数据集编号逐个产生样本，实际读取与重试都交给 __getitem__。
- `JsonlIndexDataset.column_names`：返回数据对象对外提供的字段名称，供训练工具识别样本结构。
- `JsonlIndexDataset.lengths`：返回每条所选样本的行字节长度，用作分组时的粗略长短估计；它不等于真正的文字单元数。
- `JsonlIndexDataset.modalities`：返回各样本所属来源的图文类型估计，辅助将纯文字与图文样本分组。
- `build_jsonl_index_dataset`：接收数据来源清单，逐文件建立索引、应用抽样规则并按进程切分；返回按需读取的数据集对象。

[上一页](src--xiaomi_lens--train--data--jsonl_index.md)
