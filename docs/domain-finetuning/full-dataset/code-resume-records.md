# 答案来源与旧结果筛查速查

[返回文字代码](code-descriptions.md)

[resume_records.py](../../../src/xiaomi_lens/full_dataset/resume_records.py)：

- `rules_hash`：固定两阶段读图提示与图片版本。
- `manifest_path`：本轮固定规则和模型内容清单的位置。
- `record_path`：本轮单个新增或修订答案的位置。
- `load_manifest`：检查规则与选样清单未变化。
- `activate`：固定官方模型、27条Codex答案及历史本地记录，切换到本地接续模式。
- `check_history`：核对全部历史文字的内容摘要，禁止悄悄覆盖。
- `audit_record`：检查已采用句的格式、长度、可见程度和明显颜色矛盾，给出重新查看理由。
- `_audit_job`：读取一条历史记录并筛查，供多个普通处理器任务使用。
- `audit_existing`：筛查14,185条旧答案，固定缺失、可疑和试跑样本的必修清单。
- `work_plan`：读取并检查必修清单，缓存以避免重复读取大文件。
- `resolve_record`：按优先顺序选择来源；缺少应修订的新结果时报错，不退回旧答案。
- `progress`：分别统计旧答案、Codex保留和本轮保存，未全部检查前不称完成。

筛查是结构检查和像素矛盾提示，不等于本轮重新看图。
没有外观句且目标很小时允许保留；看得见但没有外观句时重新交给模型看图核查。
