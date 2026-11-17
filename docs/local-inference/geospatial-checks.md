# 定位框与TIF检查记录

[返回总览](README.md) · [操作说明](localization.md)

日期：2026-11-16。上述功能已完成，以下检查通过。
本页记录本次新增功能的检查，不替代原500条模型效果报告。
[完整核对记录](../../outputs/geospatial-check/verification.json)。
完成状态以[逐项检查](../../outputs/geospatial-check/checks.json)为准。

## 坐标和分块

检查模型框的非方形比例、偏移、多个框、无框、错误框和思考内容排除。
检查已知四角、旋转、经纬度转换失败、日期线拆分、投影坐标单位。
检查透明无效区域；纯黑RGB仍为有效图片，不因颜色被剔除。
检查16位、多页TIF拒绝，辅助图片地理位置不一致拒绝。
[额外格式与投影检查](../../outputs/geospatial-check/additional-checks.json)。

检查1像素、小图、恰好512像素、窄边缘与长图的分块覆盖：每像素唯一归属且不越界。
使用人为制造的一次失败检查：失败部分保留254，重试后完成，已完成块不重跑。
修改任务问题后继续会拒绝，避免结果混用。

## 实际模型与页面

真实输入来自GDCLD的1024×1024彩色地理TIF，共9块；没有以测试拼图代替真实影像。
检查中先完成一块，停止，再继续处理余下部分。
地理坐标来自原影像的位置记录，定位范围仍由模型预测，不等于真实滑坡边界。
本次不训练、不调模型，不把坐标正确当作识别正确。

[浏览器操作记录](../../outputs/geospatial-check/browser/checks.json)、
[最终页面复查](../../outputs/geospatial-check/browser/final-layout-check.json)、
[普通图片及三图输入复查](../../outputs/geospatial-check/legacy-browser/live-check.json)。

页面检查包括定位框开关、分割区域开关、下载坐标、TIF准备预览、开始、停止、继续、逐块回答和恢复已保存任务。
下载的TIF已重新读取，尺寸、坐标关系和数值含义核对通过。
原500条相关2515份文件与修改前逐文件一致，见[历史文件核对](../../outputs/geospatial-check/history-check.json)。
原始开源代码检查通过；真实TIF来源文件内容未改变。
没有重新训练，也没有重跑原500条；网页交互检查使用单独结果。
