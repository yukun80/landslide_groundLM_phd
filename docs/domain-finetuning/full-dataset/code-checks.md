# 检查、进度与接续代码

[返回代码地图](code.md)

[check.py](../../../src/xiaomi_lens/full_dataset/check.py) 负责交付检查。

- `structural`：全部来源都有去向，旧划分不变，无已知跨划分内容/范围冲突。
- `inspect_export`、`raw`：完整源文件副本、全部显示像素、真实标注和两版答案。
- `encode`：真正调用训练读取流程，分别检查两版文字、长度与区域对应；不更新权重。
- `final`：按当前看图来源核对实际记录、原代码、旧数据全部核对后才标记完成。

检查入口：

- [check_full_dataset.py](../../../tools/check_full_dataset.py)：完整数据或本轮12张试跑的普通处理器读取。
- [check_local_resume_pilot.py](../../../tools/check_local_resume_pilot.py)：实际12张的图片身份、模型原始返回和句子对应。
- [check_local_resume.py](../../../tools/check_local_resume.py)：来源优先、缺项拒绝、试跑变更失效和真实重试。
- [check_codex_session.py](../../../tools/check_codex_session.py)：历史Codex阶段的隔离检查，不能在本地模式冒充当前检查。
- [check_full_dataset_runtime.py](../../../tools/check_full_dataset_runtime.py)：可选运算与真实模型检查。
- [check_full_dataset_logic.py](../../../tools/check_full_dataset_logic.py)：实际小文件与空间边界。
- [check_full_dataset_recovery.py](../../../tools/check_full_dataset_recovery.py)：复现真实格式错误与重试边界。
- [check_full_dataset_docs.py](../../../tools/check_full_dataset_docs.py)：页面、链接和原始代码。

接续入口：

- [complete_full_dataset.py](../../../tools/complete_full_dataset.py)：按当前模式接续；本地模式通过试跑后生成、发布和检查，Codex模式仍只等待实际会话。
- [full_dataset_status.py](../../../tools/full_dataset_status.py)：只读汇总保存进度。
- [write_full_dataset_validation.py](../../../tools/write_full_dataset_validation.py)：最终检查通过后按真实结果更新说明。
- [full_dataset.sh](../../../scripts/full_dataset.sh)：限定新文字任务的可选加速路径。

接续程序的 `wait_process` 核对进程身份，`main` 组织各步；失败和待用户决定都保留未完成状态。
保存进度不能代表文字准确率，12张试跑不能替代全量检查。
