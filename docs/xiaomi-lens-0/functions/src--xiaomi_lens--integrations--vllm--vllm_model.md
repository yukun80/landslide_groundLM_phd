# src/xiaomi_lens/integrations/vllm/vllm_model.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/integrations/vllm/vllm_model.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/vllm_model.py)。搜索下面的名称可定位原文件。

- `XiaomiLensMaskPooler`：从文字状态中取出区域查询，并把对应图像特征交给区域解码器，返回逐请求输出。
- `XiaomiLensMaskPooler.__init__`：保存所属模型的引用，避免将它再次登记成自己的子模块形成循环。 构造完成后通过对象保存这些状态，不返回预测结果。
- `XiaomiLensMaskPooler.get_supported_tasks`：返回该区域输出器支持的 plugin 任务集合，供 vLLM 判断能否处理请求。
- `XiaomiLensMaskPooler.get_pooling_updates`：要求框架在区域处理时保留文字编号；返回任务设置，因为必须知道哪些位置是区域特殊符号。
- `XiaomiLensMaskPooler.forward`：按每个请求的文字长度拆开状态与编号，配上捕获的图片特征，再返回区域解码结果列表。
- `XiaomiLensForVLLM`：扩展 vLLM 版 Qwen 模型，添加区域解码器与收集图片特征的回调。
- `XiaomiLensForVLLM.__init__`：先创建基础模型，再创建区域解码器、取得区域符号编号并登记图片特征捕获回调。 构造完成后通过对象保存这些状态，不返回预测结果。
- `XiaomiLensForVLLM._grab_grid`：视觉模块开始计算前读取网格参数，保存在模型对象中供区域解码使用；返回 None，不替换原输入。
- `XiaomiLensForVLLM._grab_pre_merge`：在图片小块合并前保存其特征，供区域分支恢复细节；返回 None，不更改主干输入。
- `XiaomiLensForVLLM.load_weights`：按参数名分开基础模型权重与区域解码器权重，分别加载；返回已加载名称集合。
- `XiaomiLensForVLLM._reorder_pre_merge`：将按合并块排列的图片特征恢复为从上到下、从左到右的顺序；返回按空间位置排列的二维特征。
- `XiaomiLensForVLLM.split_captured_vision`：按每个请求的网格切分刚收集的图片特征；返回特征与网格列表，数量或维数不符时返回空占位。
- `XiaomiLensForVLLM.decode_mask_batch`：定位各请求的区域符号，把相同网格的请求分组一次解码；返回每请求带网格和数量头部的扁平输出。
- `XiaomiLensForVLLM.decode_mask`：把单个请求包装成批次接口需要的列表，再取回单请求结果；保持旧调用方式可用。
