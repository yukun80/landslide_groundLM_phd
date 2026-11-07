# 模型与环境

[返回总览](README.md) · [下一步：数据](data.md)

模型优先来源：`/home/yukun80/codes/Xiaomi-Lens-0/checkpoints/Xiaomi-Lens-0/`。
本项目复制到 `checkpoints/Xiaomi-Lens-0/`，不是软链接。
复制每个文件后核对内容，保留 LICENSE 和来源记录。
若目标已有不同内容则停止，不覆盖；只有本地模型缺失时才下载固定官方版本。
初始模型与微调结果分别保存，不向来源目录写入。

```bash
python tools/prepare_domain_model.py
```

当前运行环境为已有的 `qwen3vl`，无需激活即可使用统一启动脚本。
不升级该环境原有基础库；没有安装可选加速组件时使用已有计算方式。
实际通过三类样本的训练更新、保存与重新加载对照以及自由预测。
验证记录见 [环境检查](../../outputs/qwen-env-probe/passed.json)。

```bash
scripts/domain.sh infer.py --help
```

所有启动命令通过 `scripts/domain.sh` 使用 qwen3vl、本项目的代码和模型副本。
不使用原始代码目录作为运行时导入来源。
直接依赖版本见根目录 `requirements-domain.txt`，完整列表在 `outputs/environment.txt`。
旧环境与旧数据实验记录保存在 `outputs/history/mixed-samples-v1/`，不能当作本版结果。

通过实际检查后，项目 `.venv` 已删除；原有 Conda 环境保留。
