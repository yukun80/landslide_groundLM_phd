# src/xiaomi_lens/model/lm_loss.py

[返回上级](README.md)

对应源码：[src/xiaomi_lens/model/lm_loss.py](../../../Xiaomi-Lens-0-raw/src/xiaomi_lens/model/lm_loss.py)。搜索下面的名称可定位原文件。

- `_get_fused_linear_ce`：第一次需要时尝试导入融合版文字评分工具并缓存，返回求平均与求和两个版本；不可用时返回 None。
- `_get_flash_ce`：尝试取得 flash-attn 的交叉熵计算工具并缓存；返回均值和总和版本，失败后由调用方使用普通计算。
- `_dist_world_size`：读取当前参与训练的进程总数；没有初始化多进程训练时返回 1。
- `shift_labels_flat`：将正确答案左移一位并展平，末尾用 -100 补齐；例如 [10,20,30] 变成 [20,30,-100]，因为当前位置预测下一个词。
- `CausalLMLossMixin`：给主模型补充文字评分功能的辅助类；依赖主模型已有的词表输出层与设置。
- `CausalLMLossMixin._use_fused_linear_lm_loss`：检查训练模式、设备、选项和加速工具是否满足条件，返回是否可跳过完整词表分数的显式保存。
- `CausalLMLossMixin._token_sum_scale`：根据全局有效文字数量与进程数返回缩放系数，让不同进程参与平均后仍按有效文字数计分。
- `CausalLMLossMixin._fused_linear_lm_loss`：从隐藏状态和标签中只取要学习的位置，融合文字预测与评分，返回文字损失，减少巨大词表带来的临时存储。
- `CausalLMLossMixin._causal_lm_loss`：接收文字预测分数和正确标签，优先使用可用加速计算，否则调用基础模型的评分方法；返回文字损失。
