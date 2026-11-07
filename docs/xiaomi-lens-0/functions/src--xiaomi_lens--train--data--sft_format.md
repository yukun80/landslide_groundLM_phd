# src/xiaomi_lens/train/data/sft_format.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/sft_format.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/sft_format.py)。搜索下面的名称可定位原文件。

- `box_token`：把四个相对坐标拼成 <bbox>…</bbox> 文字并返回；这里不负责从图片推算坐标。
- `mask_block`：在框文字后加上区域特殊标记，返回一个实例的标准答案片段。
- `build_completion`：按输入框的顺序拼接答案片段，可选择是否带区域标记；返回用于训练的完整助手文字。
