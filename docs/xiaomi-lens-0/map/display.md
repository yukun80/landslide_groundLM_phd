# 展示页面与图表

[返回上级](README.md)

步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。

| 文件 | 阅读步骤 | 用途与使用时机 |
|---|---|---|
| [docs/.nojekyll](../../../Xiaomi-Lens-0-raw/docs/.nojekyll) | 7 | 空的静态网站部署标记；不参与模型计算，保持原样。 |
| [docs/assets/figures/architecture.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/architecture.png) | 1 | 模型结构图，理解共享图像特征与区域分支时看；静态素材，不参与模型计算。 |
| [docs/assets/figures/hero_lens.svg](../../../Xiaomi-Lens-0-raw/docs/assets/figures/hero_lens.svg) | 1 | 页面概览装饰图，用于项目展示；静态素材，不参与模型计算。 |
| [docs/assets/figures/lensbench_data_pipeline.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/lensbench_data_pipeline.png) | 1 | LensBench 数据制作流程图，阅读评测数据介绍时看；静态素材，不参与模型计算。 |
| [docs/assets/figures/lensbench_evaluation.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/lensbench_evaluation.png) | 1 | LensBench 定性对照图，理解发布案例时看；静态素材，不参与模型计算。 |
| [docs/assets/figures/radar_comparison.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/radar_comparison.png) | 1 | 发布结果雷达图，比较原项目所报告各项分数；静态素材，不参与模型计算。 |
| [docs/assets/figures/segmentation_data_pipeline.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/segmentation_data_pipeline.png) | 1 | 区域训练数据准备流程图，理解标注来源时看；静态素材，不参与模型计算。 |
| [docs/assets/figures/training_recipe.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/training_recipe.png) | 1 | 两阶段训练配方示意图，阅读训练主线时看；静态素材，不参与模型计算。 |
| [docs/assets/figures/vqa_data_pipeline.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/vqa_data_pipeline.png) | 1 | 视觉问答数据准备图，理解文字任务数据时看；静态素材，不参与模型计算。 |
| [docs/assets/figures/xiaomi-logo.png](../../../Xiaomi-Lens-0-raw/docs/assets/figures/xiaomi-logo.png) | 1 | 项目署名使用的小米标识图；静态素材，不参与模型计算。 |
| [docs/assets/fonts/OFL.txt](../../../Xiaomi-Lens-0-raw/docs/assets/fonts/OFL.txt) | 7 | 随字体提供的许可原文；了解字体使用条件时查看，保持原样。 |
| [docs/assets/fonts/fredoka.woff2](../../../Xiaomi-Lens-0-raw/docs/assets/fonts/fredoka.woff2) | 7 | 标题显示字体；查看页面排版时使用，不参与算法。 |
| [docs/assets/fonts/jetbrains-mono.woff2](../../../Xiaomi-Lens-0-raw/docs/assets/fonts/jetbrains-mono.woff2) | 7 | 代码与等宽文字显示字体；查看页面排版时使用，不参与算法。 |
| [docs/assets/fonts/nunito.woff2](../../../Xiaomi-Lens-0-raw/docs/assets/fonts/nunito.woff2) | 7 | 正文显示字体；查看页面排版时使用，不参与算法。 |
| [docs/assets/video/overview_poster.jpg](../../../Xiaomi-Lens-0-raw/docs/assets/video/overview_poster.jpg) | 1 | 总览视频封面；页面在视频播放前展示，非训练样本。 |
| [docs/assets/video/xiaomi_lens_0_overview.mp4](../../../Xiaomi-Lens-0-raw/docs/assets/video/xiaomi_lens_0_overview.mp4) | 1 | 项目总览视频；用于直观看能力，非实时运行记录。 |
| [docs/index.html](../../../Xiaomi-Lens-0-raw/docs/index.html) | 7 | 项目展示页，含外观、内容与交互；只播放已有案例，不实时预测。 |
