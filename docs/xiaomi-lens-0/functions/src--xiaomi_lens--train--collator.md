# src/xiaomi_lens/train/collator.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/collator.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/collator.py)。搜索下面的名称可定位原文件。

- `LazyCollator`：在取到一批样本后才处理图片与文字，并补齐长度；不在启动时一次处理完整数据集。
- `LazyCollator.__init__`：保存处理器、区域符号编号和长度像素限制，初始化错误提示计数。 构造完成后通过对象保存这些状态，不返回预测结果。
- `LazyCollator._fallback_token_id`：为整批失败的占位输入寻找可用文字编号，优先补齐符号，再尝试结束或起始符号，最后返回 0。
- `LazyCollator._sample_tag`：从样本中提取来源名称、标签和图文数量，返回简短文字用于错误记录。
- `LazyCollator._dummy_batch`：接收原批次大小，构造不含图片且所有标签均忽略的占位批次；返回字典，让多进程训练保持同步而不学习坏样本。
- `LazyCollator.__call__`：接收原始样本列表，逐条编码、跳过失败记录并补齐成功样本；返回批次字典，全失败时返回零损失占位批次。
- `LazyCollator._pad`：接收短序列、填充值和目标长度，在右侧补齐并返回新列表；文字补 0，学习标签补 -100。
