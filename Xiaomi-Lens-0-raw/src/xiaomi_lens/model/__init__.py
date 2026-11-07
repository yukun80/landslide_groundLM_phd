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

# 【中文导读｜阅读第 3 步】
# 本文件：对外提供模型、解码器和损失类，并延迟导入较重的依赖。
# 怎么读：先了解这个目录提供什么，随后去具体实现文件阅读；__init__.py 不是训练起点。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

from .config import build_mask_decoder_config, get_qwen_vision_config
from .pixel_mask_decoder import PixelMaskDecoder

__all__ = [
    "build_mask_decoder_config",
    "get_qwen_vision_config",
    "PixelMaskDecoder",
    "XiaomiLensQwen3_5ForConditionalGeneration",
]


# Load the Transformers model on demand so the decoder and vLLM integration
# can use this package without importing the HF model implementation.
# 【步骤说明】当访问包里尚未导入的名字时查找其所属模块，返回相应对象；未知名字抛出 AttributeError，避免提前加载所有依赖。
# 对照参数：name；下面保留原始英文说明与类型，方便对应调用处。
def __getattr__(name):
    if name == "XiaomiLensQwen3_5ForConditionalGeneration":
        from .modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
        return XiaomiLensQwen3_5ForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
