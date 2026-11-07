# src/xiaomi_lens/model/losses.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/model/losses.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/losses.py)。搜索下面的名称可定位原文件。

- `_global_num_masks`：接收本进程的有效区域数，汇总所有进程后返回损失归一化分母；最小为 1，避免没有区域时除零。
- `DiceLoss`：衡量预测区域与正确区域的整体重叠差距；分数越小表示越接近正确区域。
- `DiceLoss.forward`：接收原始分数、0/1 标注和有效范围；内部转概率后计算 1－(2倍重叠+1)/(预测面积+标注面积+1)，返回归一化损失。
- `BCEWithLogitsLoss`：逐像素检查前景与背景的预测差距，直接接收原始分数，内部才转成概率。
- `BCEWithLogitsLoss.forward`：接收原始分数与标签，先按每个区域的有效像素数平均，再按区域数归一化；返回逐像素分类损失。
- `MaskLoss`：组合整体重叠与逐像素两种损失，支持有效区域和分块计算。
- `MaskLoss.__init__`：保存两种评分权重和分块大小，创建评分模块；默认重叠权重 1、逐像素权重 2。 构造完成后通过对象保存这些状态，不返回预测结果。
- `MaskLoss.forward`：核对预测、标注和有效像素形状，按区域分小批计算两种损失并求和；返回一个可反向传播的总分。
- `MaskLoss._compute_chunk`：将一小批预测和标注临时转成较高精度，计算重叠差距与逐像素差距；返回两项加权结果，供所有小批求和。
