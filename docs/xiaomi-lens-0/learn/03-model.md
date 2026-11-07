# 理解模型怎样连接

[返回上级](../README.md)

打开 [modeling_xiaomi_lens.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/modeling_xiaomi_lens.py)，先读模型类的 `forward`，再看 `_prepare_multimodal_inputs` 和 `_decode_pixel_masks`。

图片特征有两个出口：合并后的较短序列进入语言模型；合并前更细的网格进入区域分支。这样不用再另建一个完整的图片编码器。语言模型输出的每个位置都有上下文状态，区域分支提取的是 `<|mask|>` **所在位置经过图文理解后的状态**，而不是从词表里拿一个固定向量。

`_extract_pre_merge_features` 还会恢复小块的空间排列。数组看起来有相同数量的元素，不代表它们已经按原图从左到右、从上到下排列；这一步如果错了，区域会在空间上错位。

多图样本中，每个区域默认对应它前面最近的图片；显式传入图片索引时按索引处理。同一张图的多个区域共用图片特征，只换目标查询。

[上一节](02-inference.md) · [下一节](04-mask.md)
