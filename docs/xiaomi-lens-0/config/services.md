# 安装与服务设置

[返回上级](README.md)

[requirements.txt](../../../Xiaomi-Lens-0-raw/requirements.txt) 的每项依赖都已解释用途。[pyproject.toml](../../../Xiaomi-Lens-0-raw/pyproject.toml) 负责包的名称、安装位置、额外依赖分组，以及登记 vLLM 所需的模型与输入输出处理器。它不是运行入口。

两个 vLLM 启动脚本分别负责文字和区域，默认端口为 8000 和 8001。`SERVED_NAME` 是客户端使用的模型名字；`MAX_PIXELS` 若覆盖，两个服务必须一致；`SERVED_DIR` 是为服务准备的独立文件目录，不要指向原始模型目录。原脚本若接收非本地模型位置，会尝试在线下载。本次检查只用现有本地模型。

[eval/run.sh](../../../Xiaomi-Lens-0-raw/eval/run.sh) 用第一个参数选择 hf 或 vllm。hf 的 `DEVICES` 可指定多张卡分担样本；vllm 的 `TEXT_DEVICE` 和 `MASK_DEVICE` 给两类引擎分别选卡。`OUTPUT_DIR` 存记录与评分，`MAX_NEW_TOKENS` 限制回答长度。大幅截短回答可能使它还没说到最终答案就停止。
