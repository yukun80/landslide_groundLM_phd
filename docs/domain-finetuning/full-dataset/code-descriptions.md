# 文字生成与采用来源

[返回代码地图](code.md)

- [本地接续、生成与检查](code-local-resume.md)
- [来源保护、旧答案筛查与接续清单](code-resume-records.md)
- [历史Codex实际看图流程](code-session-history.md)

[captions.py](../../../src/xiaomi_lens/full_dataset/captions.py) 保留历史批量实现。
新流程复用其中的真实生成重试、显存不足时拆分和答案组装，不修改旧描述文件。
[guards.py](../../../src/xiaomi_lens/full_dataset/guards.py) 仅用真实目标颜色发现矛盾，不能据统计量造句。
[views.py](../../../src/xiaomi_lens/full_dataset/views.py) 的 `images_for/view_paths` 只准备独立图片，
排列和尺寸与训练一致；导入它不会加载看图模型。
