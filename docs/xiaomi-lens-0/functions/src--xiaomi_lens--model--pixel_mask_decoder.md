# src/xiaomi_lens/model/pixel_mask_decoder.py（第 1 页）

[返回上级](README.md)

对应源码：[src/xiaomi_lens/model/pixel_mask_decoder.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/pixel_mask_decoder.py)。搜索下面的名称可定位原文件。

- `_resolve_attention_backend`：读取环境变量指定的注意力计算方式并检查取值；返回选用的方式名称，供各条计算路径共用。
- `_flash_warn_once`：接收警告标识和文字，每种原因只记录一次；不返回计算结果，避免相同回退提示刷屏。
- `_get_flash_funcs`：按需导入并缓存 FlashAttention 函数；返回函数组合或 None，让上层可以回退到可用方式。
- `MLP`：把输入表示经过两层线性变换与中间非线性变换，转换成所需宽度；用于生成区域查询。
- `MLP.__init__`：接收输入、中间和输出宽度，创建两层线性变换与激活函数。 构造完成后通过对象保存这些状态，不返回预测结果。
- `MLP.forward`：接收最后一维为输入宽度的数组，依次做线性变换、GELU 和线性变换，返回目标宽度的表示。
- `LayerNorm2d`：在每个图片位置对通道做归一化，让不同特征数值尺度较稳定，保持高宽不变。
- `LayerNorm2d.__init__`：接收通道数和防止除零的小量，创建每通道可学习的缩放与偏置。 构造完成后通过对象保存这些状态，不返回预测结果。
- `LayerNorm2d.forward`：接收 B×C×H×W 特征，按通道计算均值与方差并归一化；返回同尺寸特征。
- `PatchSelfAttention`：让图片的小块彼此交流信息的一层模块；没有区域查询到图片的交叉注意力，也不负责放大图片。
- `PatchSelfAttention.__init__`：接收宽度、头数和中间宽度，创建归一化、注意力与前馈层。 构造完成后通过对象保存这些状态，不返回预测结果。
- `PatchSelfAttention.forward`：接收 B×P×D 小块序列和可选补齐标记，执行归一化、注意力与前馈变换并加回原输入；返回同尺寸序列。
- `PatchSelfAttention._self_attention`：根据设置选择图片小块之间的注意力实现；返回更新后的特征，加速不可用时回退到普通实现。
- `PatchSelfAttention._mha_self_attention`：使用 PyTorch 的多头注意力处理图片小块，传入补齐位置标记；返回每个位置融合邻近和远处信息后的特征。
- `PatchSelfAttention._project_qkv`：用已有权重从每个小块特征生成查询、键和值；返回三组按批次、小块、头、头内宽度排列的数组。
- `PatchSelfAttention._project_output`：把各注意力头的结果合回模型宽度，并应用输出投影；返回与后续残差相加兼容的特征。
- `PatchSelfAttention._sdpa_self_attention`：使用 PyTorch 的缩放点积注意力计算小块关系；正确转换补齐标记的真假含义，返回投影后的特征。
- `PatchSelfAttention._flash_self_attention`：检查设备、类型等条件后使用快速注意力；条件不支持时返回 None，通知上层回退。
- `PatchSelfAttention._flash_varlen_attention`：保留所有查询位置，只打包有效的键和值，执行不同有效长度的注意力；返回恢复成规则排列后的特征。
- `_ChunkedBilinearInterpolate4x`：提供高宽四倍插值的自定义反向计算；分块避免 CUDA 单次启动规模过大。

[下一页](src--xiaomi_lens--model--pixel_mask_decoder-2.md)
