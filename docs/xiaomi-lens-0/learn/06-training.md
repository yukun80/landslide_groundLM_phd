# 模型如何学习

[返回上级](../README.md)

先看两个启动脚本，再看 [train.py](../../../Xiaomi-Lens-0-raw/train.py) 的 `main`，随后转到 [trainer.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/trainer.py)。

| 阶段 | 从哪里出发 | 哪些部分学习 | 用什么判断错误 |
|---|---|---|---|
| 第一阶段 | 准备过特殊符号的基础 Qwen 模型 | 区域解码器和 `<|mask|>` 的词表行 | 区域损失 |
| 第二阶段 | 第一阶段输出 | 全部参数 | 文字损失＋区域损失 |

在 `trainer.py` 先看 `_build_mask_inputs` → `compute_loss`，理解正确区域如何展开并送入模型，再回头看 `_apply_param_trainability`、初始化和保存逻辑。真正循环取批次、反向求导、更新参数的大框架来自外部 `Trainer`，不是在本文件里重新从头实现。

[losses.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/losses.py) 将两种区域差距组合：Dice 关注区域整体重叠，BCE 关注每个像素的前景背景。默认区域损失内部是 `1×Dice＋2×BCE`，外面还乘 `mask_loss_weight`。[lm_loss.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/lm_loss.py) 负责下一个词的差距，正确标签要错开一位。

一次训练可以理解为：做预测→与参考答案比较→算出参数应该调整的方向→小幅调整。冻结表示“不更新参数”，不代表这一部分完全不需要计算；如果要让前面的新符号学会工作，误差信息可能仍需经过冻结部分传回来。

最后再读 [sampler.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/sampler.py) 的分批、[preflight.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/preflight.py) 的检查报告和日志保存。它们影响如何组织训练，不是区域预测公式本身。

自测：损失下降能否直接证明新图片上识别准确？**不能。还需要未用于训练的数据和正确评估。示例清单中的训练内评估重复用了训练文件，只适合检查流程。**

[上一节](05-data.md) · [下一节](07-evaluation.md)
