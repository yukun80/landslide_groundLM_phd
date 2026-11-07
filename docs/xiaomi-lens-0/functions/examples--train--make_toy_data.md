# examples/train/make_toy_data.py

[返回上级](README.md)

对应源码：[examples/train/make_toy_data.py](../../../Xiaomi-Lens-0-raw/examples/train/make_toy_data.py)。搜索下面的名称可定位原文件。

- `encode_rle`：将 0/1 区域按列优先数连续段，返回未压缩的 COCO 游程编码，方便人读懂与检查。
- `per_mille_box`：从区域像素求外接框并换算到 0～1000 相对坐标，返回四个整数。
- `grounding_block`：把一个区域转成框坐标与区域标记组成的答案片段并返回，用于合成训练样本。
- `draw_image`：按随机种子画互不重叠的简单形状，返回图片和各形状的信息与区域；几何规则直接给出正确答案。
- `stage1_row`：从一张合成图与形状标注构建第一阶段样本，返回用于学习区域分支的图文记录。
- `stage2_rows`：围绕同一张合成图构造指代、多个实例、计数等不同问答，返回第二阶段样本列表。
- `write_jsonl`：把样本逐条写为一行 JSON 并保存；无预测过程，便于之后按行索引读取。
- `main`：读取输出位置和样本规模，按固定种子画图并写出两个阶段的训练文件与清单。
