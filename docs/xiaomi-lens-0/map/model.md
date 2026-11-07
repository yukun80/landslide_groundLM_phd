# 模型核心

[返回上级](README.md)

步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。

| 文件 | 阅读步骤 | 用途与使用时机 |
|---|---|---|
| [src/xiaomi_lens/model/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/__init__.py) | 3 | 对外提供模型、解码器和损失类，并延迟导入较重的依赖。 |
| [src/xiaomi_lens/model/config.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/config.py) | 3 | 从基础模型的设置中取出文字和图像宽度，构造区域解码器需要的尺寸设置。 |
| [src/xiaomi_lens/model/lm_loss.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/lm_loss.py) | 5 | 衡量下一个文字单元是否预测正确，同时提供节省显存的等价计算路径。 |
| [src/xiaomi_lens/model/losses.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/losses.py) | 5 | 衡量预测区域与正确区域的差距：既看整体重叠，也看每个像素是否分对。 |
| [src/xiaomi_lens/model/modeling_xiaomi_lens.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/modeling_xiaomi_lens.py) | 3 | 把基础图文模型与区域解码器接起来，是理解算法的核心文件。 |
| [src/xiaomi_lens/model/pixel_mask_decoder.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/pixel_mask_decoder.py) | 3 | 把目标的文字状态与图片各位置的特征结合，输出每个像素属于目标的原始分数。 |
