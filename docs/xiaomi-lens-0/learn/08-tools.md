# 辅助文件与外部代码

[返回上级](../README.md)

[tools/prepare_base_model.py](../../../Xiaomi-Lens-0-raw/tools/prepare_base_model.py) 负责准备框符号，默认只读检查，显式指定才改写。[vLLM 目录](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm) 将同一算法接入服务：先 `client.py`，再 `io_processor.py` 和 `vllm_model.py`，最后看两个启动脚本与模型视图准备。`mask_service.py` 是另一条 HTTP 服务路径，不要与当前客户端默认的 `/pooling` 混为一谈。

[docs/index.html](../../../Xiaomi-Lens-0-raw/docs/index.html) 只是展示已保存的图、动画与表格。点击案例是在切换文件，不是实时运行模型。图片和字体不需要逐像素“加注释”，它们的用途已经逐文件列在代码树里。

本目录包含 Xiaomi-Lens-0 的模型扩展、区域分支、训练数据处理、训练入口、预测、评测与服务适配，但不包含所有第三方底层代码。进一步阅读时，在实际使用环境中定位：

| 外部部分 | 下一步看什么 | 为什么先不读它 |
|---|---|---|
| Transformers 的 Qwen3.5 | `transformers/models/qwen3_5/modeling_qwen3_5.py` | 这是基础图文主干，先弄清本项目怎样调用它 |
| Transformers 的生成循环 | `transformers/generation/utils.py` | `generate` 如何逐词循环是通用功能 |
| 图片与文字处理器 | `AutoProcessor` 加载的实际类及模型目录中的模板 | 具体类别以加载的模型为准，不凭文件名猜 |
| 通用训练循环 | `transformers/trainer.py` | 先看本项目覆盖了哪些步骤，再跟父类 |
| PyTorch | `torch.nn` 中用到的层和自动求导 | 先理解数据尺寸与输入输出，再深入数学 |
| vLLM、DeepSpeed、加速计算 | 实际安装版本的源码 | 属于最后一层，不是入门阅读起点 |

这份指南以当前本地代码为依据。原文档引用了根目录 `LICENSE`，但本次原始文件清单中没有这个文件；没有补造许可文本，已存在的源码版权声明与字体许可保持不变。

[上一节](07-evaluation.md) · [下一节](09-reading-tips.md)
