# 训练启动选项

[返回上级](README.md)

[第一阶段](../../../Xiaomi-Lens-0-raw/scripts/train_stage1.sh) 与 [第二阶段](../../../Xiaomi-Lens-0-raw/scripts/train_stage2.sh) 都按“路径→机器与卡→训练命令”组织。源码中已按段加注释；连续的反斜杠命令中不要随意插入注释，否则会截断命令。

| 一组选项 | 含义 |
|---|---|
| `MODEL_PATH / DATA_CONFIG / OUTPUT_DIR` | 从哪里读模型、从哪里读训练清单、保存到哪里 |
| `DEEPSPEED_CONFIG` | 选择上面解释的分布式训练设置 |
| `NPROC_PER_NODE / NNODES / NODE_RANK` | 每台机器进程数、机器总数、当前机器编号 |
| `MASTER_ADDR / MASTER_PORT` | 多进程协调地址和端口；不是模型输出地址 |
| `attn_implementation / bf16` | 注意力计算方式与较省显存的计算精度 |
| `max_pixels / min_pixels / model_max_length` | 图片像素范围和整条输入长度上限；脚本为 4194304、65536、8192 |
| `group_by_modality_length` | 按图文类型与长度分组，减少批次补齐浪费 |
| `dataloader_num_workers` | 几个读取进程帮助准备样本，配方为 8 |
| `dataloader_persistent_workers / dataloader_prefetch_factor` | 保留读取进程，并提前准备若干批次；这里预取因子为 4 |
| `dataloader_drop_last` | 丢掉最后不满一批的样本批次 |
| `unfreeze_*` | 哪些部分允许学习；逐项含义见 [config.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/config.py) 的中文旁注 |
| `lm_loss_weight / mask_loss_weight` | 文字和区域损失的外层权重；第一阶段文字权重为 -1，关闭文字评分 |
| `mask_loss_chunk_size` | 每次处理 8 个区域的评分，降低临时存储 |
| `learning_rate` | 调整参数的步幅，第一阶段 1e-4，第二阶段 5e-6 |
| `weight_decay` | 此配方为 0，不施加这项权重衰减 |
| `warmup_steps` | 此配方传 0.03；已核对本机 Transformers 5.4.0：小于 1 时按总训练步数的比例计算并向上取整，即预热约前 3% 的步骤 |
| `lr_scheduler_type` | cosine，学习率随进度按余弦规律变化 |
| `num_train_epochs` | 遍历训练数据的轮数，配方为 1 |
| `per_device_train_batch_size` | 每卡每小批样本数，第一阶段 8、第二阶段 4 |
| `gradient_accumulation_steps` | 连续累积 4 个小批次再更新 |
| `gradient_checkpointing` | 少存中间结果、反向时重新计算，以时间换显存；与保存训练进度不同 |
| `auto_resume` | 自动找输出目录最新进度继续；显式恢复选项优先 |
| `save_steps / save_total_limit` | 每 2000 步保存，保留数量限制为 2 |
| `logging_steps / report_to / run_name` | 记录频率、记录工具和本次训练名称 |
| 最后一行 `"$@"` | 将用户额外传入的选项放到命令尾部，可覆盖前面的默认配方 |
