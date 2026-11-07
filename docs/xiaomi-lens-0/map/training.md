# 训练组织

[返回上级](README.md)

步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。

| 文件 | 阅读步骤 | 用途与使用时机 |
|---|---|---|
| [scripts/train_stage1.sh](../../../Xiaomi-Lens-0-raw/scripts/train_stage1.sh) | 5 | 第一阶段启动配方；只训练区域解码器与特殊符号行。 |
| [scripts/train_stage2.sh](../../../Xiaomi-Lens-0-raw/scripts/train_stage2.sh) | 5 | 第二阶段启动配方；从第一阶段继续学习全部参数。 |
| [src/xiaomi_lens/train/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/__init__.py) | 5 | 对外提供训练器和训练设置，按需加载以降低导入成本。 |
| [src/xiaomi_lens/train/collator.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/collator.py) | 4 | 把多条长短不同的样本临时加工并拼成一批；失败样本会记录并跳过。 |
| [src/xiaomi_lens/train/config.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/config.py) | 5 | 定义训练、数据和模型三个方面可调整的选项及默认值。 |
| [src/xiaomi_lens/train/configs/zero1.json](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/configs/zero1.json) | 5 | 第一阶段 DeepSpeed 存储分配设置，逐字段解释见配置与示例详解。 |
| [src/xiaomi_lens/train/configs/zero2.json](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/configs/zero2.json) | 5 | 第二阶段 DeepSpeed 存储分配设置，逐字段解释见配置与示例详解。 |
| [src/xiaomi_lens/train/preflight.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/preflight.py) | 5 | 训练开始前整理数据规模、可训练参数和训练设置，让人先核对任务是否符合预期。 |
| [src/xiaomi_lens/train/sampler.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/sampler.py) | 5 | 决定每一轮如何打乱、分组和分配样本，使各训练进程得到自己的批次。 |
| [src/xiaomi_lens/train/trainer.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/trainer.py) | 5 | 组织模型加载、可训练部分选择、数据读取、每步损失计算、记录和保存。 |
| [train.py](../../../Xiaomi-Lens-0-raw/train.py) | 5 | 训练总入口：读取训练选项、确定数据来源、创建训练器，并启动或继续训练。 |
