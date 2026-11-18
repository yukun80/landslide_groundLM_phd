# 文件与函数地图

[返回总览](README.md) · [上一节：运行](usage.md) · [下一节：检查](validation.md)

新代码位于 [full_dataset](../../../src/xiaomi_lens/full_dataset/)。
复用项目已有数据处理和文字规则，不修改 `Xiaomi-Lens-0-raw/`。

追加文字质量复查见[代码说明](code-quality-review.md)。

## 按步骤查阅

|专题|主要代码|职责|
|---|---|---|
|[清洗和复制](code-preparation.md)|common.py、prepare.py|来源身份、实际标注清洗、保护划分、完整复制|
|[文字和证据](code-descriptions.md)|local_resume.py、resume_records.py、views.py|本地两遍看图、旧答案筛查与逐条来源|
|[检查和接续](code-checks.md)|check.py 与 tools/ 入口|完整副本、真实训练读取、完成标记与进度|

## 与已有代码的衔接

训练与评估增加显式 `--dataset-dir`，默认目录保持原1000条。
训练读取增加可选的普通处理器设备参数，原训练与预测默认行为不变；模型只在真正加载时导入，避免图片检查初始化显卡。
已有 [visual_review.py](../../../src/xiaomi_lens/description_v2/visual_review.py) 的核对函数允许显式传入图片与数值，
共用原有保守删句规则；新流程不调用其写入旧复核目录的入口。
原有文档检查增加下级目录遍历，覆盖这些短文。

[全量定稿合并入口](../../../tools/merge_description_final.py)核对现用原文、按编号合并定稿，经两种图像输入读取后提供三份现用清单。使用说明见[本地操作](usage.md)。
