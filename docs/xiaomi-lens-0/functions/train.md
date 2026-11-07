# train.py

[返回上级](README.md)

对应源码：[train.py](../../../Xiaomi-Lens-0-raw/train.py)。搜索下面的名称可定位原文件。

- `_normalize_resume_from_checkpoint`：统一恢复训练选项中的真假值和文字写法；返回 None、True 或指定路径，消除命令行歧义。
- `_cli_arg_passed`：检查用户是否显式传入某个选项，兼容连字符、下划线和等号写法；返回真假，供自动恢复规则判断优先级。
- `_resolve_auto_resume`：显式恢复选项优先，否则按自动恢复设置寻找最新 checkpoint；返回恢复目标或 None，并调整跳过数据的设置。
- `_validate_resume_target`：检查指定恢复路径是否存在；自动恢复时检查输出目录下是否有 checkpoint-*，这里只做存在性检查，不验证所有状态文件完整。
- `main`：解析三类选项、设定随机种子、创建训练器，打印检查报告后启动或恢复训练，最后保存模型。
