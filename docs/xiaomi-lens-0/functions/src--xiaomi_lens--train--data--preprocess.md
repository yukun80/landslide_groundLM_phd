# src/xiaomi_lens/train/data/preprocess.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/preprocess.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/preprocess.py)。搜索下面的名称可定位原文件。

- `encode_sample`：接收一条图文标注，依次整理对话、读取图片、转数字、截断、确定回答位置和区域归属；返回单条训练输入，无法处理时返回 None。
- `_pixels_per_token`：根据图片小块边长和合并倍数计算一个图像文字位置覆盖多少源像素；返回面积，默认例子为 (16×2)²=1024。
- `_truncate_keeping_whole_images`：接收文字序列、长度上限和图片网格；把截断点退到完整图片边界，返回保留文字长度和保留图片数。
- `_assistant_spans`：扫描对话起止标记，定位每一轮助手回答的内容区间；返回起止位置列表，让历史回答也参与学习。
- `_unpack_masks`：将按位打包的正确区域还原成像素数组；返回展开结果，是训练器在设备上解包过程的 CPU 参照。
