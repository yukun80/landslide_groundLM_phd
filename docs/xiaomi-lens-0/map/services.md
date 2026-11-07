# 运行服务

[返回上级](README.md)

步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。

| 文件 | 阅读步骤 | 用途与使用时机 |
|---|---|---|
| [src/xiaomi_lens/integrations/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/__init__.py) | 7 | 标记外部工具适配目录，本文件没有算法计算。 |
| [src/xiaomi_lens/integrations/vllm/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/__init__.py) | 7 | 标记 vLLM 适配包，具体加载、服务和调用逻辑在相邻文件。 |
| [src/xiaomi_lens/integrations/vllm/client.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/client.py) | 7 | 向文字服务请求回答，再把同一图片和生成结果送给区域服务，并保存叠加图。 |
| [src/xiaomi_lens/integrations/vllm/io_processor.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/io_processor.py) | 7 | 把外部区域请求转换成模型输入，再把模型输出转换为可传输的图片或区域编码。 |
| [src/xiaomi_lens/integrations/vllm/mask_service.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/mask_service.py) | 7 | 提供一个使用普通模型加载方式的 HTTP 区域服务，接收图片、问题和已生成文字。 |
| [src/xiaomi_lens/integrations/vllm/plugin.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/plugin.py) | 7 | 向 vLLM 登记本项目的自定义模型类，让服务能找到它。 |
| [src/xiaomi_lens/integrations/vllm/prepare_served.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/prepare_served.py) | 7 | 在独立输出目录准备服务所需的模型文件视图，兼容文字服务与区域服务两种加载方式。 |
| [src/xiaomi_lens/integrations/vllm/run_vllm.sh](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/run_vllm.sh) | 7 | 启动文字生成服务；与区域服务配合使用。 |
| [src/xiaomi_lens/integrations/vllm/run_vllm_mask.sh](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/run_vllm_mask.sh) | 7 | 启动区域解码服务；登记插件并保留区域解码器权重。 |
| [src/xiaomi_lens/integrations/vllm/vllm_model.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/vllm_model.py) | 7 | 在 vLLM 的模型中接入相同的区域解码器，收集图片特征并按请求生成区域。 |
