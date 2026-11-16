# 本地滑坡推理

[返回项目入口](../../README.md)

用本轮第10轮最佳模型测试500条独立样本，不继续训练。
页面参考[小米官方演示](https://huggingface.co/spaces/SeerRay-Lab/xiaomi-lens-0)，增加中文样本浏览、辅助图、上传及批量结果。

- [启动与使用](usage.md)：浏览样本、上传图片、查看和下载结果。
- [独立样本](data.md)：来源、排重、有效区与限制。
- [代码与接口](implementation.md)：工具分类、输入输出和恢复。
- [检查记录](validation.md)：实际运行情况、证据及结果限制。

```bash
scripts/start_demo.sh
```

访问 http://127.0.0.1:7860 。使用现有 qwen3vl，默认仅本机可访问。
