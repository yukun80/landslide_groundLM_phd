# 历史Codex直接看图速查

[返回文字代码](code-descriptions.md)

[session_review.py](../../../src/xiaomi_lens/full_dataset/session_review.py) 保留历史实际看图记录，
不调用生成模型；当前本地接续模式下禁止通过此入口继续提交或发布。

- `activate`：检查Codex模式和旧记录；当前模式不匹配时拒绝切换。
- `record_path`：Codex记录的位置，与本地记录分离。
- `ordered_rows`：固定试跑和按来源处理的顺序。
- `next_batch`：准备真实图片，不代写外观，不增加完成数量。
- `validate_entry`：检查实际查看角色、两版文字、长度和区域标记。
- `commit`：保存当前会话观察；再次修改须给原因并保留旧版。
- `status`：统计Codex记录，不能代替本地接续进度。
- `publish`：历史Codex完整模式的导出；当前模式下拒绝执行。
- `validate_all`：历史Codex全量检查，不能冒充当前混合来源全量检查。

历史入口：[review_full_dataset_session.py](../../../tools/review_full_dataset_session.py)。
27条已完成答案保持原文，实际查看声明仍未经人工逐图确认。
