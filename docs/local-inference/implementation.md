# 代码与接口

[返回总览](README.md)

|位置|职责|
|---|---|
|tools/prepare_benchmark500.py|按固定规则抽取、排重、复制与显示转换|
|tools/check_benchmark_resume.py|禁用显卡重读500条结果，核对原回答和区域不变|
|tools/check_domain_docs.py|核对领域和本地页面文档链接、篇幅及原始代码|
|tools/check_local_inputs.py|三类真实输入、问题覆盖与标注隔离检查|
|tools/check_local_demo_browser.py|浏览器操作、下载、上传和重复点击检查|
|tools/check_benchmark500.py|独立核对500份实际数据、旧样本和分组|
|tools/run_benchmark500.py|锁定模型批量预测及断点续跑|
|tools/build_benchmark_review.py|固定每来源10条图文对照页|
|src/xiaomi_lens/local_demo/service.py|模型复用、结果保存、评分、打包|
|src/xiaomi_lens/local_demo/app.py|样本浏览、上传与批量结果界面|
|src/xiaomi_lens/local_demo/boxes.py|解析模型框，新增展示与坐标包，不覆盖历史结果|
|src/xiaomi_lens/local_demo/geoio.py|彩色TIF窗口读取、有效区、四角与日期线转换|
|src/xiaomi_lens/local_demo/geojob.py|固定任务、停止恢复、分块拼接和地理下载|
|src/xiaomi_lens/local_demo/geopage.py|TIF预览、进度、逐块回答、地理坐标页面|
|tools/prepare_localization_views.py|为历史500条新增定位展示，不调用模型|
|tools/check_geospatial_browser.py|定位开关、TIF真实运行、停止继续和下载的浏览器检查|
|tools/check_geospatial_demo.py|已知坐标、边缘、停止重试检查与真实TIF预测|
|scripts/start_demo.sh|使用qwen3vl启动本地页面|

共享domain_runtime中的build和predict新增可选prompt；未传入时行为保持不变。
不改动原始开源目录。共享run_review支持读取本次result.json及既有score.json。
预测入口只读取图片和问题，标准区域在预测返回之后才用于评分。
固定批量与交互结果分别在outputs/benchmark500/batch和interactive下。

## 函数职责

- prepare/main：检查来源并复制样本；不足500条报错，不放宽排重。
- check/main：读取实际文件，比对有效区域、标注、内容、分组和旧样本。
- service/validate、verify_model：核对完整数据、最佳模型及配套文件；失败时阻止执行。
- service/write_json：完整写入后替换，避免页面读到半份结果。
- app/cached、remember：同一会话同输入的结果保留15秒，连续点击复用。
- app/result_table：只读滚动表展示500条结果，避免大表格编辑器卡顿。
- app/result_notes：显示文字区域矛盾和回答截断提示。
- Predictor/run：加载一次模型，串行预测，保存回答与区域；有标注时评分。
- service/batch：先处理三类代表样本，再完成500条并汇总；失败单列。
- service/bundle：打包单条结果，不包含模型权重。
- app/make_app：组织三个页面与事件；批量忙时暂不执行新的交互预测。
- app/display、choose、move：加载样本或已有结果、筛选、前后导航。
- app/upload：检查图片格式、数量、尺寸，复制上传图并预测。
- app/infer_sample：保存自定义问题的独立推理结果。
- app/batch_results：读取汇总和明细，提供下载。

模型来源固定为outputs/finetune-20261115/multiview/best.pt，内容必须符合锁定记录。
服务只监听127.0.0.1:7860，单个模型请求串行执行；批量程序设进程锁。
批量失败后继续记录其他样本，最终非零退出；交互错误直接在页面提示。

TIF任务存于outputs/benchmark500/geospatial，独立于500条固定预测。
窗口数据按需读取；整图分割写盘，网页预览最长边1536，不把预览坐标作为原图坐标。
定位框接口新增valid/raw/pixel_xyxy及地理四角；旧result.json不改变。
新任务保存输入摘要、问题、模型摘要和分块规则，恢复时逐项核验。
模型文件首次使用核对完整内容；同进程中文件大小、修改时间、变更时间或文件身份改变会重新核对。

浏览器回归检查可用--output-dir指定本次证据目录，避免覆盖历史页面检查记录。
