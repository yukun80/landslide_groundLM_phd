# 清洗与复制代码

[返回代码地图](code.md)

[common.py](../../../src/xiaomi_lens/full_dataset/common.py) 定义三类输入组合、路径、种子与原子保存。

- `source_paths`：限制来源必须在共享数据目录内。
- `file_meta`、`digest`：记录文件和设置的内容身份。
- `old_rows`、`pilot_rows`：读取旧数据，确定每类4张训练试跑图。
- `protection`、`check_protection`：保护旧数据及历史证据。
- `initialize`：建立或核对构建约定，防止混合不同设置。

[prepare.py](../../../src/xiaomi_lens/full_dataset/prepare.py) 按来源准备新数据。

- `inspect_one`、`scan`：逐条实际读取影像与标注，记录排除原因。
- `group_key`、`bounds`、`SpatialIndex`：父图标识、真实尺寸范围与重叠检查。
- `verify_near_identities`：相似候选的真实结构比较，避免只凭相似删图。
- `partition`：固定旧划分、去重并分配三部分；质量冲突单独记录。
- `export_one`、`export`：复制完整源文件，按固定范围制图，复用匹配的旧答案。

入口：[prepare_full_dataset.py](../../../tools/prepare_full_dataset.py)。
每条完成记录最后原子保存；复制中断后可继续，只有写入完整的样本才进入准备清单。
同类准备任务有互斥保护，避免两个进程同时写同一份副本。
