# src/xiaomi_lens/integrations/vllm/prepare_served.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/integrations/vllm/prepare_served.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/integrations/vllm/prepare_served.py)。搜索下面的名称可定位原文件。

- `keep_key`：判断某个权重是否应保留给文字服务；区域权重被排除，只有共享词嵌入时才去除重复输出层权重。
- `_corrected_tokenizer_config`：读取分词器设置并修正服务兼容项，返回新字典；不直接覆盖来源文件。
- `_rewrite_shard`：从一份权重分片挑出文字服务需要的参数并写到目标位置；返回保留与删除的参数项数。
- `_tie_word_embeddings`：从模型设置读取输入词表与输出层是否共用权重，返回真假，用于避免错误删除非共享输出层。
- `prepare`：接收源模型和独立输出目录，生成适合服务加载的文件；返回输出目录，文字模式可重写分片，区域模式保留所有权重。
- `main`：读取来源与独立输出目录，按文字或区域模式准备服务视图，返回成功退出状态。
