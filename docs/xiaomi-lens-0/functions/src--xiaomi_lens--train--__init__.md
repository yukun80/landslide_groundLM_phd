# src/xiaomi_lens/train/__init__.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/train/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/train/__init__.py)。搜索下面的名称可定位原文件。

- `__getattr__`：当访问包里尚未导入的名字时查找其所属模块，返回相应对象；未知名字抛出 AttributeError，避免提前加载所有依赖。
