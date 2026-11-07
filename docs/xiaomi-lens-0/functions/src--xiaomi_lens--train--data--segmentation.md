# src/xiaomi_lens/train/data/segmentation.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/segmentation.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/segmentation.py)。搜索下面的名称可定位原文件。

- `is_coco_rle`：检查对象是否具有区域游程编码所需的 size 和 counts，返回真假，不执行像素解码。
- `split_segmentation_masks`：把空值、单个区域或区域列表统一成按答案顺序排列的列表；不接受无法解释的标注类型。
- `_decode_compressed_rle_counts`：读取 COCO 压缩字符串中的分段整数与差分，返回连续黑白段的长度列表，供解码器还原像素。
- `decode_coco_rle`：把连续黑白段长度按列优先还原成高×宽的 0/1 数组；检查大小和计数，不要求安装 pycocotools。
- `segmentation_to_pixel_mask`：接收标注与目标画布大小，先解码，再按最近邻缩放；多个标注取并集，返回画布上的 0/1 区域。
