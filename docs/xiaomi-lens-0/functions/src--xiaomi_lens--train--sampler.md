# src/xiaomi_lens/train/sampler.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/sampler.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/sampler.py)。搜索下面的名称可定位原文件。

- `_length_grouped_indices`：先随机打乱样本编号，再在大组中按长度排序；返回新顺序，以减少同一批次的补齐浪费。
- `_modality_length_grouped_indices`：把图文与纯文字样本分别按长度分组，再混洗完整组并合并余数；返回完整样本顺序。
- `BatchSamplerShard`：负责每个进程每一批取得哪些样本编号，支持打乱与长短分组。
- `BatchSamplerShard.__init__`：保存批次与进程设置，检查分组参数，计算当前进程负责的样本量。 构造完成后通过对象保存这些状态，不返回预测结果。
- `BatchSamplerShard.rank`：返回当前进程编号，优先使用显式设置；没有多进程环境时返回 0。
- `BatchSamplerShard.world_size`：返回参与分配样本的进程数，优先使用显式设置；单进程默认 1。
- `BatchSamplerShard.__iter__`：使用本轮种子生成顺序、领取当前进程份额，再逐批产生编号列表；按设置保留或丢掉最后不满一批的部分。
- `BatchSamplerShard.set_epoch`：接收训练轮数并更新随机种子；让不同轮次的顺序变化，同时在同一设置下可复现。
- `BatchSamplerShard.__len__`：根据本进程样本量、批次大小与是否丢尾返回批次数，便于计算训练进度。
