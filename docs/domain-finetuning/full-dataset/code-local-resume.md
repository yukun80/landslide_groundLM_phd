# 本地接续生成速查

[返回文字代码](code-descriptions.md)

[local_resume.py](../../../src/xiaomi_lens/full_dataset/local_resume.py)：

- `check_gate`：核对12张试跑的实际查看记录及答案，变化后拒绝全量运行。
- `save_attempt`：保存每次真实返回，包括格式错误的返回。
- `candidate_issues`：检查整句是否过长、小目标纹理和近全图目标的外侧关系；不造替代句。
- `checked_draft`：检查数组和句长，失败后要求模型重新读图生成。
- `read_phase`：光学阶段只看光学，辅助阶段再看全部图，各自真实生成并再次核查。
- `combine_phases`：内部合并用于筛选，最终保留两阶段真实返回，不伪造组合模型返回。
- `run`：加载官方初始模型，分批生成与再次核对；逐条保存，不修改模型参数。
- `approve_pilot`：读取当前会话实际看图后编写的说明，核对12张图片与答案摘要。
- `revise_pilot`：试跑未放行时归档失败轮次，更新规则后真正重做；已冻结规则不得更改。
- `publish`：必修项全部完成后，统一导出两版答案和逐条采用来源。
- `validate_all`：核对历史保护、试跑放行与全部采用记录。
- `main`：命令入口和单任务锁；查看状态不加载模型。

[resume_validation.py](../../../src/xiaomi_lens/full_dataset/resume_validation.py)：

- `validate_records`：逐样本核对实际采用记录、图片、区域后缀与文字长度，汇总前后对照。

入口：[describe_full_dataset.py](../../../tools/describe_full_dataset.py)。
接续并完成全部检查：[complete_full_dataset.py](../../../tools/complete_full_dataset.py)。
关键边界检查：[check_local_resume.py](../../../tools/check_local_resume.py)。
