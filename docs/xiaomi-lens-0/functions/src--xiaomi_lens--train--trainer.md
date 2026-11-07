# src/xiaomi_lens/train/trainer.py（第 1 页）

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/trainer.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/trainer.py)。搜索下面的名称可定位原文件。

- `_FreezeNonTrainableEmbedRows`：为词嵌入挂接梯度处理，只允许指定新符号行更新。
- `_FreezeNonTrainableEmbedRows.__init__`：根据允许学习的词编号构造梯度行掩码，并登记反向传播回调。 构造完成后通过对象保存这些状态，不返回预测结果。
- `_FreezeNonTrainableEmbedRows._hook`：接收词嵌入梯度，在支持的完整形状下将其他词行清零并返回；遇到无法对应行的分片形状则警告并返回 None，不执行屏蔽。
- `XiaomiLensTrainer`：扩展通用训练器，把图文模型与区域标注处理接入原有训练循环。
- `XiaomiLensTrainer.__init__`：加载处理器与模型、添加区域符号、设置可训练部分，再构建训练/评估数据和拼批处理器。 构造完成后通过对象保存这些状态，不返回预测结果。
- `XiaomiLensTrainer._disable_unneeded_gradient_checkpointing`：只训练区域分支时，关闭冻结主干上无用的重算连接；调整设置以减少不必要的反向计算。
- `XiaomiLensTrainer._set_requires_grad`：对模块中满足条件的参数设置是否参与学习；原地修改标志，不返回预测结果。
- `XiaomiLensTrainer._apply_param_trainability`：先冻结全部参数，再按选项开放指定部分；检查共享词表权重的冲突，防止开放与冻结规则互相抵消。
- `XiaomiLensTrainer._mean_init_new_token_embeddings`：把新增特殊符号的初始表示设为原词表平均值；原地初始化新增行，并处理输入输出共享权重的情况。
- `XiaomiLensTrainer._build_model`：接收模型位置或已有模型及训练设置；加载模型并配置两种损失与计算方式，返回可训练模型。
- `XiaomiLensTrainer._configure_image_processor`：把图片像素上限和下限写入图片处理器，兼容不同设置保存形式；原地更新，不读取具体图片。
- `XiaomiLensTrainer._prepare_inputs`：保留样本中不规则的区域信息到 compute_loss 阶段处理，返回输入，避免过早用通用搬运逻辑处理它们。
- `XiaomiLensTrainer.get_train_dataloader`：根据数据是否已分片选择正确的批次分配方式，并组合处理器和数据读取进程；返回训练用数据加载器。
- `XiaomiLensTrainer._build_mask_inputs`：将每条样本的区域位置、所属图片和压缩标注拼成规则数组；返回区域输入字典，记录有效像素以排除补齐部分。
- `XiaomiLensTrainer.compute_loss`：把文字、图片和解包后的正确区域送入模型，取得并记录两种损失；按调用要求返回总损失或总损失与模型输出。
- `XiaomiLensTrainer._to_device`：把非空数组搬到当前计算设备，返回搬运结果；空值和零元素数组统一返回 None。
- `XiaomiLensTrainer.prediction_step`：评估训练样本时复用相同的区域准备与损失计算，关闭梯度；返回损失和按需要保留的文字预测。
- `XiaomiLensTrainer._distributed_metric_means`：按固定顺序汇总所有进程记录的文字与区域损失，返回用于日志的平均值，避免只报告某一张卡的情况。
- `XiaomiLensTrainer.log`：接收当前记录，加入分项损失、时间、进度与显存占用，再交给父类输出并清空临时统计。
- `XiaomiLensTrainer._format_duration`：把秒数转换成时分秒或分秒的可读文字并返回。

[下一页](src--xiaomi_lens--train--trainer-2.md)
