# src/xiaomi_lens/train/data/jsonl_index.py（第 1 页）

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/jsonl_index.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/jsonl_index.py)。搜索下面的名称可定位原文件。

- `JsonlIndex`：保存非空样本行的字节起点与长度，以及源文件大小和修改时间。
- `JsonlIndex.__len__`：返回索引中非空样本行的数量，供 len(index) 使用。
- `scan_jsonl_offsets`：逐行扫描样本文件，跳过空白行并记录每行起点和长度；返回索引，不把全部样本内容留在内存。
- `apply_sampling_strategy`：根据 all、first、end 或 random 规则挑选样本编号；返回选中编号，百分比向上取整，随机选择受 seed 控制。
- `stride_for_rank`：按进程编号隔位领取样本，必要时裁掉不能平分的尾部；返回本进程编号列表，样本少于进程数时保留原列表。
- `JsonlIndexShard`：管理一份样本文件的只读映射与索引，提供按行读取及关闭操作。
- `JsonlIndexShard.__init__`：保存索引与图片目录等来源信息，并打开当前进程的只读文件映射。 构造完成后通过对象保存这些状态，不返回预测结果。
- `JsonlIndexShard._open`：打开只读文件并建立按需读取的内存映射；更新对象状态，已经打开时直接返回。
- `JsonlIndexShard.__getstate__`：返回可传递给另一进程的索引与来源信息，不传递已打开的文件句柄。
- `JsonlIndexShard.__setstate__`：接收另一进程恢复的状态，重建本进程自己的文件映射；无返回值，避免跨进程复用无效句柄。
- `JsonlIndexShard.__len__`：返回这一份样本文件的有效行数，转交给底层索引统计。
- `JsonlIndexShard.get`：按指定行号查索引，只解析这一行，再补齐图片路径与来源信息；返回一条样本。
- `JsonlIndexShard.default_modality`：读取首条样本来估计该来源是否含图片并缓存；返回布尔值，这是来源级估计，不逐条检查。
- `JsonlIndexShard.close`：释放只读映射与文件描述符并清空状态，可重复调用；不修改样本文件。
- `JsonlIndexShard.__del__`：对象销毁时调用 close 释放文件资源，不执行数据写入。
- `open_jsonl_shard`：对一个样本文件先建索引，再创建带图片目录与来源名称的读取对象；返回可按行读取的分片。
- `JsonlIndexDataset`：把多个样本文件拼成统一编号空间，并为损坏记录提供有限重试。
- `JsonlIndexDataset.__init__`：建立总编号到分片行号的对应表，保存抽样结果，并生成遇到坏行时的备用读取顺序。 构造完成后通过对象保存这些状态，不返回预测结果。
- `JsonlIndexDataset.__len__`：返回抽样和分片后当前数据集的样本数量，不一定等于原文件行数总和。
- `JsonlIndexDataset.__getitem__`：根据总编号定位分片和行；坏行会按固定备用顺序有限次重试，成功返回样本，全部失败则报错。

[下一页](src--xiaomi_lens--train--data--jsonl_index-2.md)
