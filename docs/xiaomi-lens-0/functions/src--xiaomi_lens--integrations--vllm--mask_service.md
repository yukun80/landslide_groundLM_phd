# src/xiaomi_lens/integrations/vllm/mask_service.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/integrations/vllm/mask_service.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/mask_service.py)。搜索下面的名称可定位原文件。

- `_png_b64`：将像素数组按指定颜色模式存成内存 PNG，再转成文本；返回用于网络传输的 Base64 字符串。
- `_load_image`：优先读取请求中的本地图片路径，否则解码 Base64 图片；返回 RGB 图片，两者都没有就报错。
- `decode_mask`：读取图片、问题和已生成文字，使用已加载模型重新计算区域；返回区域 PNG、框和可选叠加图，无答案区域时返回空列表。
- `Handler`：接收 HTTP 请求并调用区域解码函数，组织成功或失败响应。
- `Handler._send`：将状态码和字典写成 HTTP 响应，设置长度与内容类型；无返回值。
- `Handler.do_GET`：处理健康检查路径 /health，返回服务状态；其余路径返回不存在。
- `Handler.do_POST`：只处理 /decode_mask 请求，读取 JSON 并执行区域解码；成功发回结果，异常转成错误响应。
- `Handler.log_message`：覆盖服务器默认的访问日志方法，不输出日志，避免重复终端记录。
- `main`：读取模型与监听地址，加载一次模型，启动服务器等待区域解码请求。
