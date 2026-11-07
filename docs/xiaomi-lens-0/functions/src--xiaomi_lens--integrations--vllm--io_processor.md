# src/xiaomi_lens/integrations/vllm/io_processor.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/integrations/vllm/io_processor.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/io_processor.py)。搜索下面的名称可定位原文件。

- `_load_image`：优先读取请求中的本地图片路径，否则解码 Base64 图片；返回 RGB 图片，两者都没有就报错。
- `_png_b64`：将像素数组按指定颜色模式存成内存 PNG，再转成文本；返回用于网络传输的 Base64 字符串。
- `_encode_rle`：把二值区域按列优先压缩成 COCO RLE，并将字节串改成可写入 JSON 的文字；返回 size/counts 字典。
- `XiaomiLensMaskIOProcessor`：在外部请求和 vLLM 的区域模型之间转换输入输出，暂存每个请求的展示选项。
- `XiaomiLensMaskIOProcessor.__init__`：保存服务提供的分词器，并创建按请求号记录输出选项的字典。 构造完成后通过对象保存这些状态，不返回预测结果。
- `XiaomiLensMaskIOProcessor.parse_data`：检查请求内容必须是字典，合法就返回原对象；不是字典则报错。
- `XiaomiLensMaskIOProcessor.pre_process`：根据图片与问题构造前缀，再拼接实际生成的编号；返回服务模型输入，并按请求号暂存展示选项。
- `XiaomiLensMaskIOProcessor.post_process`：读取网格与区域数量，重建像素分数并移除思考区域；按请求缩放、二值化和编码，返回区域与可选叠加图。
- `make_io_processor`：返回输入输出处理器类的完整名称字符串；框架会再实例化该类，这里不是直接返回对象。
