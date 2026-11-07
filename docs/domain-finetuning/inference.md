# 预测

[返回总览](README.md) · [上一步](training.md) · [下一步](evaluation.md)

省略 `--checkpoint` 使用初始模型；要使用微调结果必须指定该文件。

使用顺序：辅助图片在前，光学图片在最后。
外部图片必须已经对齐，不能把不同地点的图片作为一组输入。
当前入口接收准备好的显示图片，不直接把任意原始多波段文件当成彩色图。

```bash
scripts/domain.sh infer.py --images /绝对路径/光学.png --output outputs/my-prediction
scripts/domain.sh infer.py --images /绝对路径/InSAR.png /绝对路径/高程.png /绝对路径/光学.png --checkpoint outputs/multiview/best.pt --output outputs/my-multiview-prediction
```

输出：

- `prediction.json`：原始回答、回答中区域数、区域计算说明和耗时。
- `mask.png`：在原光学图片大小下的预测区域。
- `overlay.png`：预测区域叠加到光学图。

回答中没有区域标记时输出空区域，不拿真实标注补预测。
文字与区域计算说明分别保存，冲突会记录；不会悄悄改写回答。
推理时区域计算的面积以整幅图为分母；评测则单独排除标注无效区。

训练完成后可用 `examples/domain_predict.sh` 运行一个真实多模态样例。
它使用训练样本演示调用，不作为泛化效果证据。

结果会列出可解析的面积、位置和连续/多块分布矛盾，保留原话供核对。
这类检查不能覆盖所有自然语言表达，不等于专家认可描述。

加载只用光学图片训练的结果时，预测入口自动只使用最后一张光学图片。
