# src/xiaomi_lens/train/data/render.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/data/render.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/data/render.py)。搜索下面的名称可定位原文件。

- `_count_image_placeholders`：只统计用户文字中的 <image> 次数，返回占位符数量，用于与真实图片数核对。
- `ensure_user_message`：样本只有助手答案时补一条用户消息，并将图片标记放到用户侧；返回整理后的对话，已有用户消息则原样返回。
- `align_image_placeholders`：接收对话和真实图片数，缺标记时向首条用户消息补，多标记时从后向前删；返回整理后的消息列表。
- `normalize_qwen_message_content`：将通用 <image> 占位符换成该模型认识的图片起止标记；返回替换后的文字，非文字内容原样通过。
- `last_assistant_index`：从后往前找最后一条助手消息，返回其位置；没有助手答案就无法建立训练目标，因此报错。
- `normalize_messages`：逐条复制消息并替换图片标记，返回标准化消息列表，不改传入对象。
- `render_prompt_and_completion`：用同一个对话模板分别排出问题前缀与完整对话，再取差得到目标答案；返回两段文字。
- `load_sample_images`：按路径打开并真正读入图片像素，统一成 RGB；返回图片列表，避免以后才触发延迟读取。
