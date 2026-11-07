# 真实示例与区域编码

[返回上级](README.md)

[examples/train/sample/sample.jsonl](../../../Xiaomi-Lens-0-raw/examples/train/sample/sample.jsonl) 有两行，每行都可以独立解析成一个对象。图片位置由旁边的 [sample.yaml](../../../Xiaomi-Lens-0-raw/examples/train/sample/sample.yaml) 指定。

| 字段 | 第一条样本里是什么 | 如何理解 |
|---|---|---|
| `images` | `COCO_train2014_000000581739.jpg` 的列表 | 图片文件名，可扩展为多张；相对 `image_folder` 定位 |
| `messages` | 两条消息的列表 | 保留用户提问和助手正确回答的次序 |
| `messages[0].role` | `user` | 这是用户输入 |
| `messages[0].content` | 图片占位符与寻找 `RIGHT VASE` 的请求 | `<image>` 声明图片放在这里，处理器随后将它换成模型认识的格式 |
| `messages[1].role` | `assistant` | 这是训练参考答案，不是运行本次代码新产生的回答 |
| `messages[1].content` | 思考起止标记、框与一个区域标记 | 空思考段也属于答案格式；最终框为 `[586, 308, 756, 927]` |
| `segmentation` | 仅含一个区域的列表 | 第一个元素对应助手答案中的第一个 `<|mask|>` |
| `segmentation[0].size` | `[480, 640]` | 顺序是高、宽；注意 Pillow 的图片 size 通常是宽、高 |
| `segmentation[0].counts` | 一段压缩字符串 | 存连续背景/前景像素的长度信息，不能直接当文字答案，也不是模型权重 |

第二条图片是 `COCO_train2014_000000580562.jpg`，问题找 `gray dump truck`，框为 `[0, 429, 292, 775]`，区域尺寸也为高 480、宽 640。两条都有一个助手区域标记和一个正确区域。

### 区域编码小例子

把 2×2 区域 `[[1,1],[0,0]]` 按列读，得到 `[1,0,1,0]`。从背景开始数连续段，计数是 `[0,1,1,1,1]`：先有零个背景，再有一个前景、一个背景、一个前景、一个背景。

未压缩形式可以写成 `{"size":[2,2],"counts":[0,1,1,1,1]}`。COCO 也允许把计数压缩成字符串；本项目的 `decode_coco_rle` 两种都能读。压缩存储与普通图像缩放是不同步骤。
