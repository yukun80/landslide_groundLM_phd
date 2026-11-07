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
# 本文件：从基础模型的设置中取出文字和图像宽度，构造区域解码器需要的尺寸设置。
# 怎么读：先看 get_qwen_vision_config，再看 build_mask_decoder_config；不同宽度必须通过投影对齐。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Model-configuration helpers for Xiaomi-Lens-0."""


# 【步骤说明】从整体设置中查找 vision_config 或兼容名称 visual_config；返回图像部分设置，缺失时明确报错。
# 对照参数：config；下面保留原始英文说明与类型，方便对应调用处。
def get_qwen_vision_config(config):
    for attr_name in ("vision_config", "visual_config"):
        vision_config = getattr(config, attr_name, None)
        if vision_config is not None:
            return vision_config
    raise ValueError("Qwen vision config (vision_config/visual_config) is required to build PixelMaskDecoder.")


# 【步骤说明】读取文字宽度、注意力头数与图像宽度，返回构造区域解码器的设置字典；默认中间通道 256、像素表示宽度 32。
# 对照参数：config；下面保留原始英文说明与类型，方便对应调用处。
def build_mask_decoder_config(config) -> dict:
    vision_config = get_qwen_vision_config(config)
    text_config = config.text_config
    # Project pre-merge vision features to the LLM width for patch attention,
    # then reduce channels before the decoder's transposed convolutions.
    return {
        "llm_hidden_size": text_config.hidden_size,
        "llm_intermediate_size": text_config.intermediate_size,
        "llm_num_heads": text_config.num_attention_heads,
        "vision_hidden_size": vision_config.hidden_size,
        "mask_decoder_channels": int(getattr(config, "mask_decoder_channels", 256)),
        "mask_decoder_embed_dim": int(getattr(config, "mask_decoder_embed_dim", 32)),
    }
