# src/xiaomi_lens/train/data/io.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/io.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/io.py)。搜索下面的名称可定位原文件。

- `DatasetEntry`：记录一个数据来源的样本文件、图片目录、名称、标签和抽样规则，不存放图片像素。
- `DatasetEntry.display_name`：返回显式的数据来源名称；没有名称时使用样本文件名，便于记录时识别来源。
- `parse_path_spec`：把冒号分隔的多个路径拆成列表，忽略空项；没有输入时返回空列表。
- `resolve_dataset_entries`：优先从数据清单读取来源，否则解析命令行中的样本路径和图片目录；返回统一的数据来源列表。
- `resolve_sample_images`：将 image 或 images 统一成图片路径列表，并接上图片根目录；返回样本副本，纯文字样本的图片列表为空。
- `resolve_dataset_config_path`：依次检查直接给出的清单路径和包内预设位置，返回存在的文件路径；两者都没有则报错。
- `_relative_to`：接收基准目录与路径；绝对路径原样返回，相对路径拼到基准目录后返回。
- `load_dataset_config`：读取 YAML 清单，选择训练或评估分组并验证字段；把相对路径解析好后返回 DatasetEntry 列表。
