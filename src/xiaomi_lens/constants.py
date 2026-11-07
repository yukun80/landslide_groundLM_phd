# Copyright 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 【中文导读｜阅读第 2 步】
# 本文件：集中存放特殊符号和默认生成长度，供推理与训练共用。
# 怎么读：MASK_TOKEN 是文字中的区域标记；它不是一张图片，模型需要另一步把它解释为区域。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Project-wide constants for Xiaomi-Lens-0."""

# Keep generation defaults aligned across inference entry points.
# 新生成部分的长度上限，不包含已有问题；16384 是较长默认值，小检查可另传较短上限。
DEFAULT_MAX_NEW_TOKENS = 16_384

# 真正用于定位区域状态的特殊符号；它的编号由模型分词器决定，不能随意手写一个数字代替。
MASK_TOKEN = "<|mask|>"

# Grounding delimiters. The model emits, per instance,
# `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`. The bbox delimiters
# reuse native Qwen token IDs 248049/248050; the mask delimiters are ordinary
# text. Only the `<|mask|>` slot is an added tokenizer row.
# 以下是答案格式的边界文字。框符号沿用 Qwen 编号，<mask> 和 </mask> 是普通文本边界。
BOX_START_TOKEN = "<bbox>"
BOX_END_TOKEN = "</bbox>"
MASK_START_TOKEN = "<mask>"
MASK_END_TOKEN = "</mask>"
