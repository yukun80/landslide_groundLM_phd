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

# 【中文导读｜阅读第 5 步】
# 本文件：定义训练、数据和模型三个方面可调整的选项及默认值。
# 怎么读：与两个训练脚本对照：脚本传入的值可以覆盖这里的默认值。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Command-line argument dataclasses for Xiaomi-Lens-0 training."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


# 【类的用途】保存损失、数据加载、保存方式等训练选项，继承通用训练器的其他选项。
@dataclass
class XiaomiLensTrainingConfig(TrainingArguments):
    # 设置含义：加载模型时额外传入的设置；没有提供则采用空设置。
    model_init_kwargs: Optional[dict] = field(default=None)
    # 设置含义：是否删掉通用训练器不认识的数据字段；这里默认关闭，保留自定义区域信息。
    remove_unused_columns: Optional[bool] = field(default=False)
    # 设置含义：用反向时重新计算换取较少的中间存储；不是保存训练进度。
    gradient_checkpointing: bool = field(default=True)
    # 设置含义：每次学习调整参数的步幅；过大或过小都会影响训练，脚本会覆盖默认值。
    learning_rate: float = field(default=1e-5)
    # 设置含义：用 safetensors 方式保存模型参数。
    save_safetensors: bool = field(default=True)
    # 设置含义：单份权重文件大小上限，超过时拆成多份。
    save_max_shard_size: str = field(default="5GB")

    # 设置含义：文字预测差距的权重；负数会关闭这项计算，零与负数在执行开销上不同。
    lm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of the causal LM loss. A negative value disables it."},
    )
    # 设置含义：区域预测差距的权重；负数关闭区域损失项。
    mask_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of the mask loss. A negative value disables it."},
    )
    # 设置含义：整体区域重叠差距在区域损失中的权重。
    mask_dice_weight: float = field(default=1.0)
    # 设置含义：逐像素前景背景差距在区域损失中的权重。
    mask_bce_weight: float = field(default=2.0)
    # 设置含义：每次转换成较高精度评分的区域数；小分块减少临时显存，非正数表示不分块。
    mask_loss_chunk_size: int = field(
        default=8,
        metadata={
            "help": "Maximum flattened (sample, mask-slot) rows per FP32 dice+BCE "
                    "chunk. With gradient checkpointing, each chunk is "
                    "checkpointed independently, bounding loss temporaries by "
                    "chunk_size*H*W instead of B*K*H*W. Values <= 0 use one "
                    "unchunked reference path."
        },
    )

    # 设置含义：文字评分的计算方式；加速方式不可用时会回退，目标仍是预测下一个词。
    lm_loss_impl: str = field(
        default="fused_linear_ce",
        metadata={
            "help": "Causal-LM loss kernel. `fused_linear_ce` uses liger-kernel's "
                    "fused linear+cross-entropy and never materializes the "
                    "(B, S, vocab) logits. `flash_ce` uses flash-attn cross "
                    "entropy over BF16 logits. `hf` forces the FP32 reference "
                    "loss. Unavailable kernels warn once and fall back in that order."
        },
    )

    # 设置含义：是否将图文类型与长度相近的样本放到相近批次，减少补齐浪费。
    group_by_modality_length: bool = field(
        default=False,
        metadata={
            "help": "Group samples by (modality, text length) so each global batch contains "
                    "similar-length sequences from the same modality, reducing padding waste. "
                    "Applies only to the train_dataset."
        },
    )

    # 设置含义：是否自动查找输出目录中最新训练进度；用户显式指定恢复位置时优先使用显式设置。
    auto_resume: bool = field(
        default=False,
        metadata={
            "help": "On launch, resume from the highest-step `checkpoint-*` under "
                    "`output_dir`, or start fresh when none exists. When a "
                    "checkpoint is found, `ignore_data_skip` is also set to True "
                    "(unless passed explicitly) so skipped batches are not "
                    "re-encoded. An explicit `--resume_from_checkpoint` takes precedence."
        },
    )


# 【类的用途】保存数据路径、图片大小等脚本级选项，供数据构建阶段读取。
@dataclass
class XiaomiLensScriptArguments:
    # 设置含义：直接提供多个样本文件路径，用冒号分隔；也可改用 data_config 清单。
    data_file_paths: Optional[str] = field(
        default=None,
        metadata={"help": "Colon-separated list of JSONL dataset files. Mutually exclusive with --data_config."},
    )
    # 设置含义：与样本文件一一对应的图片根目录，使用冒号分隔。
    image_folders: Optional[str] = field(
        default=None,
        metadata={"help": "Colon-separated image folders aligned with --data_file_paths."},
    )
    # 设置含义：记录样本文件、图片目录及抽样规则的 YAML 清单；相对路径以清单目录为起点。
    data_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a YAML data manifest. Train-only manifests "
                          "use top-level `datasets:`; training-time evaluation "
                          "uses `train:` and `eval:` sections. Mutually exclusive "
                          "with --data_file_paths."},
    )
    # 设置含义：每张处理后图片的像素数量上限，不是宽度上限；多图时还受总文字长度预算限制。
    max_pixels: Optional[int] = field(default=12845056)
    # 设置含义：图片处理的像素数量下限，避免图像被缩得过小。
    min_pixels: Optional[int] = field(default=3136)
    # 设置含义：整条输入允许的最大数字序列长度，包括图片占位位置与文字。
    model_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Override tokenizer.model_max_length; None keeps the value loaded from the processor."},
    )


# 【类的用途】保存模型位置、计算精度与哪些部分允许训练等模型选项。
@dataclass
class XiaomiLensModelConfig:
    # 设置含义：待加载的模型目录或模型名称；第一阶段通常是基础模型，第二阶段是第一阶段结果。
    model_name_or_path: Optional[str] = field(default=None)
    # 设置含义：基础模型注意力的计算实现；可用性取决于环境，独立于区域解码器的环境变量设置。
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # `unfreeze_*` flags follow a freeze-everything-then-opt-in convention: the
    # trainer first sets `requires_grad=False` on every parameter, then each
    # True flag re-enables its module. See
    # `XiaomiLensTrainer._apply_param_trainability`.
    # 设置含义：为真则全部参数参与学习，并忽略其余逐模块开关。
    unfreeze_all_parameters: bool = field(
        default=False,
        metadata={
            "help": "Train the whole model. When True, all other unfreeze_* flags are ignored."
        },
    )
    # 设置含义：是否学习视觉主干的小块特征提取部分，不含后面的合并投影。
    unfreeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Unfreeze the ViT blocks (`model.visual.*` except `merger.*`)."},
    )
    # 设置含义：是否学习把图片小块合并并映射到语言模型的部分。
    unfreeze_vision_projection: bool = field(
        default=False,
        metadata={"help": "Unfreeze the visual-to-LLM projection (`model.visual.merger.*`)."},
    )
    # 设置含义：是否学习语言模型内部层，不包含输入词表表示。
    unfreeze_language_model_backbone: bool = field(
        default=False,
        metadata={
            "help": "Unfreeze `language_model.*` except `embed_tokens.*` "
                    "(decoder layers and final norm)."
        },
    )
    # 设置含义：是否学习预测词表分数的输出层；若与输入词表共享权重，两者一起受影响。
    unfreeze_language_model_head: bool = field(
        default=False,
        metadata={
            "help": "Unfreeze `lm_head.*`. With tied embeddings (e.g. Qwen3.5-4B), "
                    "`lm_head.weight` is `embed_tokens.weight`, so this coincides "
                    "with `unfreeze_embed_tokens`."
        },
    )
    # 设置含义：是否学习整个输入词表表示，优先级高于只学习新符号。
    unfreeze_embed_tokens: bool = field(
        default=False,
        metadata={
            "help": "Unfreeze the entire `embed_tokens.weight`. Takes precedence "
                    "over `unfreeze_new_tokens`."
        },
    )
    # 设置含义：只允许 <|mask|> 对应词表行更新，其余行梯度被清零。
    unfreeze_new_tokens: bool = field(
        default=False,
        metadata={
            "help": "Train only the `<|mask|>` row of `embed_tokens.weight`. A "
                    "backward hook zeros every other row's gradient. Supported "
                    "with DDP and DeepSpeed ZeRO-1/2."
        },
    )
    # 设置含义：是否学习区域解码器，第一阶段需要开启。
    unfreeze_mask_decoder: bool = field(
        default=False,
        metadata={"help": "Unfreeze the PixelMaskDecoder, which is trained from random init."},
    )
    # 设置含义：模型计算和存储所用浮点精度，默认 bfloat16 以减少显存占用。
    torch_dtype: Optional[str] = field(default="bfloat16")
