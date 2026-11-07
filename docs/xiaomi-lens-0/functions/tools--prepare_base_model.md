# tools/prepare_base_model.py

[返回上级](README.md)

对应源码：[tools/prepare_base_model.py](../../../Xiaomi-Lens-0-raw/tools/prepare_base_model.py)。搜索下面的名称可定位原文件。

- `parse_args`：读取命令行提供的文件位置、运行选项和默认值，返回解析后的选项对象；本函数不执行模型预测。
- `find_model_dirs`：接收模型目录或其父目录，查找同时有两个分词器设置文件的目录并返回排序列表。
- `token_entries`：读取模型目录里的 tokenizer.json 和 tokenizer_config.json，返回两个字典供一致性检查。
- `inspect_model`：检查旧框符号或新框符号是否成对、编号一致且标为特殊符号；返回迁移状态和编号，混杂或冲突则报错。
- `atomic_write`：先将新内容写到同目录临时文件并落盘，再替换目标文件；不返回值，降低中断时只写一半的风险。
- `rewrite_model`：只替换分词器中的旧框符号拼写，先检查并备份，再写回；编号和权重不变，已有备份不覆盖。
- `main`：查找模型目录，先检查全部目标；指定写入时备份并重命名框符号，最后重新检查并打印报告。
