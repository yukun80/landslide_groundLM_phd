# src/xiaomi_lens/train/preflight.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/preflight.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/preflight.py)。搜索下面的名称可定位原文件。

- `_format_count`：把数量格式化成便于阅读的文字，返回展示字符串，不参与训练计算。
- `_format_ratio`：接收分子和分母，返回可阅读的比例文字，处理分母为零的情况。
- `_summarize_numeric`：汇总一组数值的范围等统计，返回报告所需的摘要，不执行模型预测。
- `_normalize_enum`：如果输入是枚举则返回其实际值，否则原样返回，便于判断训练选项。
- `is_eval_enabled`：检查训练设置是否启用了评估，返回真假，决定是否需要准备评估数据。
- `_count_parameters`：统计模型全部参数与允许训练的参数，返回数量摘要，用来核对冻结设置。
- `_build_dataset_summary`：接收数据集与分组名称，抽查图文及区域数量、来源和缺失情况；返回数据报告，并记录是否只检查了部分样本。
- `_build_model_summary`：汇总模型类别、参数数量和可训练部分，返回模型设置说明。
- `_class_name`：接收对象并返回类名，用来显示实际采用的模型或处理器类型。
- `_build_training_summary`：汇总进程数、每批样本量、学习设置和保存设置，返回训练计划说明。
- `build_preflight_report`：组合数据、模型和训练三个摘要，返回完整的训练前检查报告，帮助先核对再运行。
