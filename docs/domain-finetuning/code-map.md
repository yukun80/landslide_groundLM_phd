# 新增代码地图

[返回总览](README.md)

| 位置 | 职责 |
|---|---|
| tools/prepare_domain_model.py | 复制或下载模型，逐文件核对 |
| tools/prepare_domain_data.py | 数据准备入口 |
| tools/check_domain_data.py | 全样本输入、目标和原始代码检查 |
| src/xiaomi_lens/train/data/domain.py | 来源读取、划分、去重、辅助图与答案生成 |
| src/xiaomi_lens/description_checks.py | 对照文字与预测区域，列出可核对的不一致 |
| src/xiaomi_lens/domain_runtime.py | 共用模型加载、输入、预测和保存 |
| src/xiaomi_lens/train/domain_train.py | 两阶段训练与恢复 |
| src/xiaomi_lens/domain_evaluation.py | 自由预测评分与汇总 |
| eval/domain.py | 评测入口及最终评估限制 |
| train.py | 与原项目习惯一致的训练入口 |
| infer.py | 新图片预测入口 |
| scripts/run_domain_experiment.sh | 顺序执行完整对照实验 |
| tools/check_qwen_environment.py | 三类样本训练、保存重载与预测兼容性检查 |
| tools/check_domain_reload.py | 比较保存前后自由预测 |
| tools/lock_domain_models.py | 锁定正式模型和最终样本 |
| tools/domain_auxiliary_check.py | 在val检查辅助图片作用 |
| tools/report_domain_results.py | 汇总结果及审阅图片页 |
| examples/domain_predict.sh | 真实多模态样本调用演示 |
| tools/review_description_samples.py | 限定533条描述的逐图复核、写回和证据检查 |
| src/xiaomi_lens/description_v2/visual_review.py | 本地视觉推理、逐句复核、保守删句和范围核对 |
| scripts/train_stable.sh | 本轮单版本多轮训练及恢复入口 |
| tools/run_stable_finetuning.py | 独立运行目录、阶段检查、模型锁定及最终对照 |
| tools/test_domain_stopping.py | 停止与恢复规则测试 |
| src/xiaomi_lens/train/stopping.py | 明显进步计数、最低轮次及停止原因 |
| src/xiaomi_lens/train/run_review.py | 每轮固定30条图文对照，初始标为未查看 |
| tools/build_domain_review.py | 固定验证样本的图文审阅页 |
| scripts/domain.sh | 使用 qwen3vl 与本项目代码启动命令 |

`src/xiaomi_lens`中其他文件复制自原项目，保留原版权和中文注释。
本项目只新增领域数据和训练流程，不宣称实现了外部基础模型的全部底层代码。

本地页面与500条独立测试代码见[推理代码说明](../local-inference/implementation.md)。
