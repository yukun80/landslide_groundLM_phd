# src/xiaomi_lens/integrations/vllm/client.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/integrations/vllm/client.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/client.py)。搜索下面的名称可定位原文件。

- `_post_json`：将字典编码为 JSON 并发送 HTTP POST，请求失败由网络库报错，成功返回解析后的字典。
- `_image_data_uri`：读取图片并转成带图片类型的 Base64 字符串，返回可嵌入文字服务请求的图片内容。
- `call_vllm`：发送图片和问题给文字服务，要求保留特殊符号与真实生成编号；返回回答文字和编号列表。
- `main`：读取图片与查询，先调用文字服务再调用区域服务，解码返回的图片并保存。
