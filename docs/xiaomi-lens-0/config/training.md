# 两份训练设置

[返回上级](README.md)

[zero1.json](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/configs/zero1.json) 用于第一阶段，[zero2.json](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/configs/zero2.json) 用于第二阶段。它们告诉 DeepSpeed 如何分配训练存储和计算。`auto` 表示由启动选项与训练框架补齐，不是文字占位错误。

| 字段 | 两份文件中的含义 |
|---|---|
| `fp16` | 半精度训练的一组设置；实际是否启用由 `enabled` 决定 |
| `fp16.enabled` | `auto`，跟随训练选择 |
| `fp16.loss_scale` | 0 表示使用动态损失缩放，帮助小梯度不被半精度舍入掉 |
| `fp16.loss_scale_window` | 1000，动态缩放时观察连续稳定步骤的窗口 |
| `fp16.initial_scale_power` | 16，初始缩放按 2 的 16 次方设置 |
| `fp16.hysteresis` | 2，缩小损失缩放值前的容忍次数设置 |
| `fp16.min_loss_scale` | 1，动态缩放的下限 |
| `bf16.enabled` | `auto`，允许框架按启动选项启用 bfloat16；本项目两个配方均传入 `--bf16` |
| `train_micro_batch_size_per_gpu` | 每张卡每次处理多少样本，由启动参数确定 |
| `train_batch_size` | 考虑卡数和梯度累积后的总批次大小，由框架确定 |
| `gradient_accumulation_steps` | 累积几次小批次再更新一次参数 |
| `zero_optimization.stage` | 第一份为 1，分担优化器状态；第二份为 2，还分担梯度。它不等于本项目的训练阶段编号，恰好配方采用了相同数字 |
| `zero_optimization.overlap_comm` | false，不启用这里的通信计算重叠选项 |
| `zero_optimization.contiguous_gradients` | true，用连续空间整理梯度，降低碎片影响 |
| `zero_optimization.sub_group_size` | 仅第二份出现，1e9，是框架分组处理参数的规模选项；是否实际使用由所选阶段实现决定，不是图片分块大小 |
| `zero_optimization.reduce_bucket_size` | 仅第二份出现，auto，由框架确定梯度归约缓冲区规模 |

恢复训练需要与保存时的分片方式匹配；“重新加载模型参数”和“恢复优化器及训练进度”不是一回事。这里只增加说明，没有改变任何设置值。
