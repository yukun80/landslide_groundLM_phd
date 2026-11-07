# 入口与安装

[返回上级](README.md)

步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。

| 文件 | 阅读步骤 | 用途与使用时机 |
|---|---|---|
| [.gitignore](../../../Xiaomi-Lens-0-raw/.gitignore) | 7 | 版本管理忽略规则；管理缓存、环境和输出文件时查看。 |
| [README.md](../../../Xiaomi-Lens-0-raw/README.md) | 1 | 原项目英文总说明；查完整安装命令与发布结果时查看。 |
| [README_CN.md](../../../Xiaomi-Lens-0-raw/README_CN.md) | 1 | 原项目中文总说明；先看方法介绍，再跟学习指南进入代码。 |
| [infer.py](../../../Xiaomi-Lens-0-raw/infer.py) | 2 | 单张图片的完整使用入口。把图片和问题交给模型，先获得文字，再把回答中的区域画到图片上。 |
| [pyproject.toml](../../../Xiaomi-Lens-0-raw/pyproject.toml) | 7 | 安装与插件登记设置；了解软件包如何被导入和服务如何发现模型时查看。 |
| [requirements.txt](../../../Xiaomi-Lens-0-raw/requirements.txt) | 7 | 普通推理所需依赖清单；准备环境时查看，中文逐项说明已加在原文件。 |
| [src/xiaomi_lens/__init__.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/__init__.py) | 2 | 提供软件包入口，只有访问具体名字时才加载对应模块。 |
| [src/xiaomi_lens/constants.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/constants.py) | 2 | 集中存放特殊符号和默认生成长度，供推理与训练共用。 |
| [tools/prepare_base_model.py](../../../Xiaomi-Lens-0-raw/tools/prepare_base_model.py) | 7 | 检查并可选地统一基础模型的边界框特殊符号名称，不重新训练或改写模型权重。 |
