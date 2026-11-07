# src/xiaomi_lens/model/modeling_xiaomi_lens.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/model/modeling_xiaomi_lens.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/modeling_xiaomi_lens.py)。搜索下面的名称可定位原文件。

- `XiaomiLensOutput`：整理一次模型计算的返回值，包括文字分数、区域分数、有效像素和两类训练损失。
- `XiaomiLensQwen3_5ForConditionalGeneration`：继承基础图文模型并接入区域分支，统一文字生成、像素预测与联合训练。
- `XiaomiLensQwen3_5ForConditionalGeneration.__init__`：先构造基础 Qwen 图文模型，再创建区域解码器和区域损失函数。 构造完成后通过对象保存这些状态，不返回预测结果。
- `XiaomiLensQwen3_5ForConditionalGeneration._get_vision_config`：转交给统一设置读取函数，返回图像部分设置，避免在模型类中重复处理兼容名称。
- `XiaomiLensQwen3_5ForConditionalGeneration._get_spatial_merge_size`：读取图像小块合并的边长倍数；返回整数，用于恢复合并前的空间排列。
- `XiaomiLensQwen3_5ForConditionalGeneration._flatten_image_features`：从视觉输出提取供语言模型使用的合并后特征，统一返回列表；缺失必要特征时报错。
- `XiaomiLensQwen3_5ForConditionalGeneration._extract_pre_merge_features`：按每张图片的网格尺寸切分视觉特征，再将合并块顺序还原成逐行顺序；返回每张图片的细网格特征列表。
- `XiaomiLensQwen3_5ForConditionalGeneration._prepare_multimodal_inputs`：把文字编号转换成表示，把图片特征填入图片占位符，并准备空间位置信息；返回混合输入、视觉输出和位置编号。
- `XiaomiLensQwen3_5ForConditionalGeneration._decode_pixel_masks`：按样本及所属图片收集区域查询，补齐不同大小的网格后统一解码；返回区域分数和有效像素范围，补齐区域不参与评分。
- `XiaomiLensQwen3_5ForConditionalGeneration.forward`：接收图文输入及可选正确答案；先计算共享图文状态，再生成文字和区域分数，训练时返回两类加权损失。
