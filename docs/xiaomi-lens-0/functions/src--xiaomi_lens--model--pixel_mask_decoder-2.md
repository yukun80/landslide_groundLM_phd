# src/xiaomi_lens/model/pixel_mask_decoder.py（第 2 页）

[返回上级](README.md)

对应源码：[src/xiaomi_lens/model/pixel_mask_decoder.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/pixel_mask_decoder.py)。搜索下面的名称可定位原文件。

- `_ChunkedBilinearInterpolate4x.forward`：接收区域分数并放大四倍，同时保存输入尺寸与设备信息，供反向传播还原对应关系。
- `_ChunkedBilinearInterpolate4x.backward`：接收放大后的梯度，按批次与通道分块计算输入梯度；返回原始尺寸的梯度。
- `_bilinear_interpolate_4x`：接收区域分数，将高宽分别放大四倍；需要反向传播且超过 CUDA 限制时走分块版本，返回放大后的分数。
- `PixelMaskDecoder`：把文字中的目标表示与图片特征变成区域分数图；每个区域标记对应一张分数图。
- `PixelMaskDecoder.__init__`：读取文字和视觉宽度，创建查询变换、小块注意力、降通道和两次放大层。 构造完成后通过对象保存这些状态，不返回预测结果。
- `PixelMaskDecoder._remap_legacy_keys`：加载旧权重时改写历史参数名称，使它们对应当前层名；修改传入的权重映射，不改变训练出来的数值。
- `PixelMaskDecoder._maybe_checkpoint`：接收要执行的模块和输入，训练需要节省显存时通过重算保留必要信息；返回模块结果。
- `PixelMaskDecoder.forward`：接收 B×K×D 的区域查询和 B×(h·w)×V 的图片特征；输出 B×K×(16h)×(16w) 原始像素分数。
- `PixelMaskDecoder._validate_inputs`：检查区域查询和图片特征的维数、批次及网格数量是否匹配；合法时返回批次大小，不匹配则报错。

[上一页](src--xiaomi_lens--model--pixel_mask_decoder.md)
