# src/xiaomi_lens/model/config.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/model/config.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/config.py)。搜索下面的名称可定位原文件。

- `get_qwen_vision_config`：从整体设置中查找 vision_config 或兼容名称 visual_config；返回图像部分设置，缺失时明确报错。
- `build_mask_decoder_config`：读取文字宽度、注意力头数与图像宽度，返回构造区域解码器的设置字典；默认中间通道 256、像素表示宽度 32。
