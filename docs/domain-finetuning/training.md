# 训练

[返回总览](README.md) · [上一步](data.md) · [下一步](inference.md)

先用16个训练样本验证三类有目标输入、参数更新、保存和恢复。
短训练结果不参与正式模型比较；正式训练从初始模型重新开始。

```bash
scripts/domain.sh -m xiaomi_lens.train.domain_train --smoke --epochs 1 --output outputs/smoke
scripts/domain.sh -m xiaomi_lens.train.domain_train --smoke --epochs 2 --output outputs/smoke --resume outputs/smoke/last.pt
```

正式训练每次1个样本、每8个样本更新一次。
高分辨率光学最长边512，其余及辅助图片最长边256；放大不会创造地面细节。
第1轮只更新区域输出部分；后3轮同时更新少量语言参数。
大部分原模型权重保持冻结，训练参数与优化器状态保存到独立文件。
每轮自由预测调参检查集，以三个来源的平均区域分数选择模型。
区域损失无法直接证明自由生成的文字或区域有效。

```bash
scripts/domain.sh -m xiaomi_lens.train.domain_train --output outputs/multiview
scripts/domain.sh -m xiaomi_lens.train.domain_train --rgb-only --output outputs/rgb
```

训练详情在 `train.jsonl`，最好模型在 `best.pt`，最新模型在 `last.pt`。
恢复以最后保存的完整轮次为起点；轮次中断时该轮重新执行。
恢复前验证图片模式一致，并恢复优化器和随机状态。

若显存不足，不允许静默跳过样本或某个来源。先排查输入尺寸并重新检查小目标。
改变设置必须同时更新文档与运行记录，不将不同设置冒充公平对照。

短训练及恢复完成后，可按顺序运行整套对照：

```bash
scripts/run_domain_experiment.sh
```

依次检查恢复、原始模型、两组正式训练、辅助图诊断、模型锁定及最终评估。
当前步骤保存在 `outputs/current-stage.txt`，任一步失败都会停止并保留记录。
已有训练目录禁止直接重跑覆盖；应明确恢复或使用新目录。
训练数值设置集中在 `src/xiaomi_lens/train/configs/domain.json`。
