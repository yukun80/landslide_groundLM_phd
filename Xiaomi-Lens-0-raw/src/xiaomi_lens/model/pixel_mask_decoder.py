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
# 本文件：把目标的文字状态与图片各位置的特征结合，输出每个像素属于目标的原始分数。
# 怎么读：先看 PixelMaskDecoder.forward，再看构造函数；注意力加速和分块反向传播最后读。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""MaskFormer-style dynamic classifier that decodes mask tokens at canvas resolution.

Each mask-token hidden state passes through `mask_embed` to produce a dynamic
classifier, which is dotted with the pixel embeddings. Patch self-attention
provides spatial context; the mask queries enter at the classification step.

Data flow:
    mask_hidden    → mask_embed MLP                          → (B, K, C_emb)
    image_features → patch_proj → patch_self_attn            → (B, P, llm_hidden)
                     reshape + channel_reduce                → (B, C, h, w)
                     2 × ConvTranspose2d(stride=2)           → (B, C_emb, 4h, 4w)
    einsum(mask_embedding, pixel_embedding) / √C_emb         → (B, K, 4h, 4w)
    bilinear interpolate(scale_factor=4)                     → (B, K, 16h, 16w)

Learned 4× upsampling followed by fixed 4× bilinear interpolation gives exactly
`MASK_CANVAS_SCALE=16`. The `(16h, 16w)` output therefore matches the resolution
of the ViT-processed input canvas.

Batching constraints:
- `_decode_pixel_masks` packs `(sample, image)` groups into a batch. Differently
  sized images within one sample also require padding, so batch size 1 does not
  guarantee a padding-free input.
- Smaller grids are zero-padded to the batch maximum and accompanied by
  `patch_key_padding_mask`, where True means padding. Fully padded placeholder
  rows must pass all False to prevent NaNs from attention softmax.
- The transposed convolutions use non-overlapping 2×2 kernels. The final bilinear
  interpolation mixes neighboring outputs, so boundary pixels of smaller grids
  can depend on padding. Callers must crop to each image's valid region.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

logger = logging.getLogger(__name__)

AttentionFunction: TypeAlias = Callable[..., torch.Tensor]
FlashFunctions: TypeAlias = tuple[AttentionFunction, AttentionFunction]

_SUPPORTED_ATTENTION_BACKENDS = ("mha", "sdpa", "flash")
_FLASH_DTYPES = (torch.float16, torch.bfloat16)

# CUDA's bilinear-upsample backward kernel fails once one launch's output
# exceeds INT32_MAX elements. Normal tensors use native F.interpolate; only that
# case is split, with each chunk capped well below the limit.
_BILINEAR_NATIVE_MAX_OUTPUT_ELEMENTS = (1 << 31) - 1
_BILINEAR_CHUNK_MAX_OUTPUT_ELEMENTS = 1 << 28


# 【步骤说明】读取环境变量指定的注意力计算方式并检查取值；返回选用的方式名称，供各条计算路径共用。
def _resolve_attention_backend() -> str:
    """Resolve the decoder attention backend from the environment (default: flash)."""
    backend = os.environ.get("XIAOMI_LENS_DECODER_ATTN", "").lower() or "flash"
    if backend not in _SUPPORTED_ATTENTION_BACKENDS:
        choices = "|".join(_SUPPORTED_ATTENTION_BACKENDS)
        raise ValueError(
            f"XIAOMI_LENS_DECODER_ATTN must be one of {choices}, got {backend!r}"
        )
    return backend


# All three implementations share MultiheadAttention projection weights, so
# switching kernels does not change checkpoints. With padding, the flash path
# packs only k/v and leaves q at full length to preserve MHA semantics for padded
# queries.
_DECODER_ATTN = _resolve_attention_backend()

# flash-attn is optional; import it lazily so CPU environments are unaffected.
_FLASH_FUNCS: FlashFunctions | None = None
_FLASH_IMPORT_ATTEMPTED = False
_FLASH_WARNED: set[str] = set()


# 【步骤说明】接收警告标识和文字，每种原因只记录一次；不返回计算结果，避免相同回退提示刷屏。
# 对照参数：key, message；下面保留原始英文说明与类型，方便对应调用处。
def _flash_warn_once(key: str, message: str) -> None:
    """Log each fallback reason only once."""
    if key in _FLASH_WARNED:
        return
    _FLASH_WARNED.add(key)
    logger.warning(
        "PatchSelfAttention flash path disabled (%s); falling back to SDPA",
        message,
    )


# 【步骤说明】按需导入并缓存 FlashAttention 函数；返回函数组合或 None，让上层可以回退到可用方式。
def _get_flash_funcs() -> FlashFunctions | None:
    """Import and cache flash-attn functions on demand."""
    global _FLASH_FUNCS, _FLASH_IMPORT_ATTEMPTED
    if _FLASH_FUNCS is None and not _FLASH_IMPORT_ATTEMPTED:
        _FLASH_IMPORT_ATTEMPTED = True
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func

            _FLASH_FUNCS = (flash_attn_func, flash_attn_varlen_func)
        except Exception:
            # An incompatible binary extension may raise exceptions other than ImportError.
            _FLASH_FUNCS = None
    return _FLASH_FUNCS


# 【类的用途】把输入表示经过两层线性变换与中间非线性变换，转换成所需宽度；用于生成区域查询。
class MLP(nn.Module):
    """Two-layer MLP: Linear → GELU → Linear."""

    # 【步骤说明】接收输入、中间和输出宽度，创建两层线性变换与激活函数。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：in_dim, hidden_dim, out_dim；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    # 【步骤说明】接收最后一维为输入宽度的数组，依次做线性变换、GELU 和线性变换，返回目标宽度的表示。
    # 对照参数：x；下面保留原始英文说明与类型，方便对应调用处。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 【类的用途】在每个图片位置对通道做归一化，让不同特征数值尺度较稳定，保持高宽不变。
class LayerNorm2d(nn.Module):
    """Apply LayerNorm over the channel dimension of `(B, C, H, W)`."""

    # 【步骤说明】接收通道数和防止除零的小量，创建每通道可学习的缩放与偏置。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：num_channels, eps；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    # 【步骤说明】接收 B×C×H×W 特征，按通道计算均值与方差并归一化；返回同尺寸特征。
    # 对照参数：x；下面保留原始英文说明与类型，方便对应调用处。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize channels at each spatial position with PyTorch's LayerNorm.
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (x.shape[1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


# 【类的用途】让图片的小块彼此交流信息的一层模块；没有区域查询到图片的交叉注意力，也不负责放大图片。
class PatchSelfAttention(nn.Module):
    """Single-layer pre-norm Transformer block over patch tokens.

    This module performs only patch self-attention. It has no `<|mask|>`-to-patch
    cross-attention and does not upsample. Checkpoint keys from the former
    `PatchDecoder` name are converted at load time by
    `PixelMaskDecoder._remap_legacy_keys`.
    """

    # 【步骤说明】接收宽度、头数和中间宽度，创建归一化、注意力与前馈层。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：hidden_dim, num_heads, intermediate_size；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, intermediate_size, hidden_dim)

    # 【步骤说明】接收 B×P×D 小块序列和可选补齐标记，执行归一化、注意力与前馈变换并加回原输入；返回同尺寸序列。
    # 对照参数：x, key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform patch self-attention.

        Args:
            x: A `(B, P, hidden_dim)` patch sequence.
            key_padding_mask: Optional `(B, P)` bool mask where True denotes
                padding. By convention, fully padded placeholder rows must pass
                all False to avoid NaNs from an all-`-inf` softmax.
        """
        attention_output = self._self_attention(self.norm1(x), key_padding_mask)
        x = x + attention_output
        return x + self.mlp(self.norm2(x))

    # 【步骤说明】根据设置选择图片小块之间的注意力实现；返回更新后的特征，加速不可用时回退到普通实现。
    # 对照参数：normed, key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    def _self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Select the configured attention kernel, falling back to SDPA when flash is unavailable.

        MultiheadAttention with a key padding mask materializes the (P, P) attention
        weights of every head, which does not fit in memory for large images; SDPA
        does not.
        """
        if _DECODER_ATTN == "mha":
            return self._mha_self_attention(normed, key_padding_mask)
        if _DECODER_ATTN == "flash":
            output = self._flash_self_attention(normed, key_padding_mask)
            if output is not None:
                return output
        return self._sdpa_self_attention(normed, key_padding_mask)

    # 【步骤说明】使用 PyTorch 的多头注意力处理图片小块，传入补齐位置标记；返回每个位置融合邻近和远处信息后的特征。
    # 对照参数：normed, key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    def _mha_self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.attn(
            normed,
            normed,
            normed,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]

    # 【步骤说明】用已有权重从每个小块特征生成查询、键和值；返回三组按批次、小块、头、头内宽度排列的数组。
    # 对照参数：x；下面保留原始英文说明与类型，方便对应调用处。
    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse MultiheadAttention parameters and return `(B, P, H, D)` q/k/v."""
        batch_size, num_patches, embed_dim = x.shape
        num_heads = self.attn.num_heads
        head_dim = embed_dim // num_heads
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (batch_size, num_patches, num_heads, head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    # 【步骤说明】把各注意力头的结果合回模型宽度，并应用输出投影；返回与后续残差相加兼容的特征。
    # 对照参数：x；下面保留原始英文说明与类型，方便对应调用处。
    def _project_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.attn.out_proj.weight, self.attn.out_proj.bias)

    # 【步骤说明】使用 PyTorch 的缩放点积注意力计算小块关系；正确转换补齐标记的真假含义，返回投影后的特征。
    # 对照参数：normed, key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    def _sdpa_self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run PyTorch SDPA using the MHA projection weights."""
        batch_size, num_patches, embed_dim = normed.shape
        q, k, v = self._project_qkv(normed)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_mask = None
        if key_padding_mask is not None:
            # Convert True padding to additive -inf; all-False rows get a zero mask.
            attention_mask = torch.zeros(
                (batch_size, 1, 1, num_patches),
                dtype=normed.dtype,
                device=normed.device,
            ).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).reshape(
            batch_size,
            num_patches,
            embed_dim,
        )
        return self._project_output(output)

    # 【步骤说明】检查设备、类型等条件后使用快速注意力；条件不支持时返回 None，通知上层回退。
    # 对照参数：normed, key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    def _flash_self_attention(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Run flash-attn FA2, returning None when the environment is unsupported.

        With a mask, use asymmetric varlen: retain all P q positions and pack only
        valid k/v positions. Padded queries can still attend to valid keys,
        matching MHA key-padding-mask semantics. Packing q and padding it back
        with zeros would change boundary values after final bilinear interpolation.
        """
        if not normed.is_cuda:
            _flash_warn_once("cuda", "input not on CUDA")
            return None

        # FA2 requires sm80+. Older GPUs such as V100/T4 can import it but fail at launch.
        major, minor = torch.cuda.get_device_capability(normed.device)
        if (major, minor) < (8, 0):
            _flash_warn_once(
                "arch",
                f"compute capability {major}.{minor} < 8.0 unsupported by FA2",
            )
            return None

        head_dim = self.attn.head_dim
        if head_dim % 8 != 0 or not 8 <= head_dim <= 256:
            _flash_warn_once("headdim", f"head_dim={head_dim} unsupported by FA2")
            return None

        flash_functions = _get_flash_funcs()
        if flash_functions is None:
            _flash_warn_once("import", "flash_attn not importable")
            return None

        q, k, v = self._project_qkv(normed)
        # LayerNorm may emit fp32 under autocast; the projected dtype controls the kernel.
        if q.dtype not in _FLASH_DTYPES:
            _flash_warn_once("dtype", f"projected dtype {q.dtype} not fp16/bf16")
            return None

        flash_attn_func, flash_attn_varlen_func = flash_functions
        batch_size, num_patches, _, _ = q.shape
        # Model forward always passes a mask tensor. For all-False masks without
        # padding, use the fixed-length kernel to avoid nonzero plus two full K/V copies.
        if key_padding_mask is None or not key_padding_mask.any():
            output = flash_attn_func(q, k, v)
            output = output.reshape(batch_size, num_patches, -1)
        else:
            output = self._flash_varlen_attention(
                q,
                k,
                v,
                key_padding_mask,
                flash_attn_varlen_func,
            )
        return self._project_output(output)

    # 【步骤说明】保留所有查询位置，只打包有效的键和值，执行不同有效长度的注意力；返回恢复成规则排列后的特征。
    # 对照参数：q, k, v, key_padding_mask, flash_attn_varlen_func；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def _flash_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,
        flash_attn_varlen_func: AttentionFunction,
    ) -> torch.Tensor:
        """Pack valid k/v and run FA2 with fixed-length q and variable-length k/v."""
        batch_size, num_patches, num_heads, head_dim = q.shape
        valid_mask = ~key_padding_mask
        valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=True)[0]

        flat_shape = (batch_size * num_patches, num_heads, head_dim)
        unpadded_k = k.reshape(flat_shape).index_select(0, valid_indices)
        unpadded_v = v.reshape(flat_shape).index_select(0, valid_indices)

        # Fully padded placeholders pass all False, so every row has valid keys.
        key_lengths = valid_mask.sum(dim=1, dtype=torch.int32)
        cumulative_k = F.pad(key_lengths.cumsum(0), (1, 0)).to(torch.int32)
        cumulative_q = torch.arange(
            0,
            (batch_size + 1) * num_patches,
            num_patches,
            device=q.device,
            dtype=torch.int32,
        )
        output = flash_attn_varlen_func(
            q.reshape(flat_shape),
            unpadded_k,
            unpadded_v,
            cu_seqlens_q=cumulative_q,
            cu_seqlens_k=cumulative_k,
            # Using P as the upper bound avoids a host sync from lengths.max().item().
            max_seqlen_q=num_patches,
            max_seqlen_k=num_patches,
        )
        return output.view(batch_size, num_patches, num_heads * head_dim)


# 【类的用途】提供高宽四倍插值的自定义反向计算；分块避免 CUDA 单次启动规模过大。
class _ChunkedBilinearInterpolate4x(torch.autograd.Function):
    """Bilinear 4× interpolation with a row-chunked CUDA backward.

    Bilinear interpolation treats every `(batch, channel)` plane independently,
    so flattening those dimensions and splitting rows preserves its semantics
    while avoiding the single >INT32_MAX-element launch that CUDA rejects.
    """

    # 【步骤说明】接收区域分数并放大四倍，同时保存输入尺寸与设备信息，供反向传播还原对应关系。
    # 对照参数：ctx, inputs；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, input_h, input_w = inputs.shape
        num_rows = batch_size * channels
        output_h = input_h * 4
        output_w = input_w * 4
        output_area = output_h * output_w
        rows_per_chunk = max(
            1,
            min(
                num_rows,
                _BILINEAR_CHUNK_MAX_OUTPUT_ELEMENTS // output_area,
            ),
        )

        flat_inputs = inputs.reshape(num_rows, 1, input_h, input_w)
        flat_output = inputs.new_empty((num_rows, 1, output_h, output_w))
        for start in range(0, num_rows, rows_per_chunk):
            end = min(start + rows_per_chunk, num_rows)
            chunk = torch.ops.aten.upsample_bilinear2d.default(
                flat_inputs[start:end],
                [output_h, output_w],
                False,
                4.0,
                4.0,
            )
            flat_output[start:end].copy_(chunk)

        ctx.input_shape = (batch_size, channels, input_h, input_w)
        ctx.output_size = (output_h, output_w)
        ctx.rows_per_chunk = rows_per_chunk
        return flat_output.view(batch_size, channels, output_h, output_w)

    # 【步骤说明】接收放大后的梯度，按批次与通道分块计算输入梯度；返回原始尺寸的梯度。
    # 对照参数：ctx, grad_output；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        batch_size, channels, input_h, input_w = ctx.input_shape
        output_h, output_w = ctx.output_size
        num_rows = batch_size * channels
        flat_grad_output = grad_output.reshape(num_rows, 1, output_h, output_w)
        flat_grad_input = grad_output.new_empty((num_rows, 1, input_h, input_w))

        for start in range(0, num_rows, ctx.rows_per_chunk):
            end = min(start + ctx.rows_per_chunk, num_rows)
            chunk = torch.ops.aten.upsample_bilinear2d_backward.default(
                flat_grad_output[start:end],
                [output_h, output_w],
                [end - start, 1, input_h, input_w],
                False,
                4.0,
                4.0,
            )
            flat_grad_input[start:end].copy_(chunk)

        return flat_grad_input.view(ctx.input_shape)


# 【步骤说明】接收区域分数，将高宽分别放大四倍；需要反向传播且超过 CUDA 限制时走分块版本，返回放大后的分数。
# 对照参数：mask_logits；下面保留原始英文说明与类型，方便对应调用处。
def _bilinear_interpolate_4x(mask_logits: torch.Tensor) -> torch.Tensor:
    """Upsample logits 4×, chunking only gradient-bearing launches over the CUDA limit."""
    output_h = mask_logits.shape[-2] * 4
    output_w = mask_logits.shape[-1] * 4
    output_area = output_h * output_w
    num_rows = mask_logits.shape[0] * mask_logits.shape[1]
    output_elements = num_rows * output_area

    needs_chunking = (
        torch.is_grad_enabled()
        and mask_logits.requires_grad
        and output_elements > _BILINEAR_NATIVE_MAX_OUTPUT_ELEMENTS
    )
    if not needs_chunking:
        return F.interpolate(
            mask_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
    if output_area > _BILINEAR_NATIVE_MAX_OUTPUT_ELEMENTS:
        raise RuntimeError(
            "A single decoder mask plane exceeds the CUDA bilinear-backward "
            f"limit: output size ({output_h}, {output_w}) has "
            f"{output_area} elements. Reduce --max_pixels."
        )
    return _ChunkedBilinearInterpolate4x.apply(mask_logits)


# 【类的用途】把文字中的目标表示与图片特征变成区域分数图；每个区域标记对应一张分数图。
class PixelMaskDecoder(nn.Module):
    """Decode `<|mask|>` hidden states into full-resolution mask logits.

    Configuration fields:
        llm_hidden_size, llm_intermediate_size, llm_num_heads: Text-model parameters.
        vision_hidden_size: Pre-merge ViT hidden size.
        mask_decoder_channels: Channel count C after 1×1 reduction.
        mask_decoder_embed_dim: Pixel-embedding channel count C_emb, which must
            match the mask_embed output dimension.
    """

    # 【步骤说明】读取文字和视觉宽度，创建查询变换、小块注意力、降通道和两次放大层。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：config；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, config: Mapping[str, int]):
        super().__init__()
        llm_hidden = int(config["llm_hidden_size"])
        llm_intermediate = int(config["llm_intermediate_size"])
        llm_num_heads = int(config["llm_num_heads"])
        vision_hidden = int(config["vision_hidden_size"])
        channels = int(config["mask_decoder_channels"])
        embed_dim = int(config["mask_decoder_embed_dim"])

        # Upsampling channels are C → C/2 → C_emb. If C/2 has only one channel,
        # LayerNorm2d always emits zero and makes the decoder image-independent,
        # so require at least two intermediate channels.
        if channels < 4 or channels % 2 != 0:
            raise ValueError(
                "mask_decoder_channels must be even and >= 4 "
                f"(so the mid layer C/2 has >= 2 channels); got {channels}"
            )

        self.embed_dim = embed_dim
        # 两条支路最后必须有相同宽度 embed_dim：文字目标变成一串“判断标准”，
        # 图片每个位置也变成同样长的一串特征，才能通过逐项相乘求和给每个位置打分。
        self.mask_embed = MLP(llm_hidden, llm_hidden, embed_dim)
        self.patch_proj = nn.Linear(vision_hidden, llm_hidden)
        self.patch_self_attn = PatchSelfAttention(
            llm_hidden,
            llm_num_heads,
            llm_intermediate,
        )
        self.channel_reduce = nn.Conv2d(llm_hidden, channels, kernel_size=1)

        # Two stride-2 deconvolutions learn 4× upsampling, followed by fixed 4×
        # interpolation in forward to produce a 16× patch-grid canvas.
        intermediate_channels = channels // 2
        # 这里两次带学习参数的放大都让高宽乘 2，总计乘 4。
        # forward 最后还会做固定的四倍插值，因此输出高宽相对原始小块网格各乘 16。
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                intermediate_channels,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(intermediate_channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                intermediate_channels,
                embed_dim,
                kernel_size=2,
                stride=2,
            ),
        )

        # Transformers' `gradient_checkpointing_enable` flips this attribute on
        # every module that declares it, which opts the decoder in to recompute.
        self.gradient_checkpointing = False

        # Support the legacy `patch_decoder.*` checkpoint parameter names.
        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)

    # 【步骤说明】加载旧权重时改写历史参数名称，使它们对应当前层名；修改传入的权重映射，不改变训练出来的数值。
    # 对照参数：state_dict, prefix；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def _remap_legacy_keys(
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *_args,
        **_kwargs,
    ) -> None:
        old_prefix = prefix + "patch_decoder."
        new_prefix = prefix + "patch_self_attn."
        old_keys = [key for key in state_dict if key.startswith(old_prefix)]
        for old_key in old_keys:
            suffix = old_key[len(old_prefix) :]
            state_dict[new_prefix + suffix] = state_dict.pop(old_key)

    # 【步骤说明】接收要执行的模块和输入，训练需要节省显存时通过重算保留必要信息；返回模块结果。
    # 对照参数：module；下面保留原始英文说明与类型，方便对应调用处。
    def _maybe_checkpoint(self, module: nn.Module, *args):
        """Run `module` under non-reentrant activation checkpointing while training.

        Only `patch_self_attn` is wrapped. It holds the decoder's largest
        activations, and wrapping a submodule keeps exactly one decoder call
        per model forward on every rank.
        """
        if not (self.gradient_checkpointing and self.training):
            return module(*args)
        return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)

    # 【步骤说明】接收 B×K×D 的区域查询和 B×(h·w)×V 的图片特征；输出 B×K×(16h)×(16w) 原始像素分数。
    # 对照参数：mask_hidden, image_features, grid_h, grid_w, patch_key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    def forward(
        self,
        mask_hidden: torch.Tensor,
        image_features: torch.Tensor,
        grid_h: int,
        grid_w: int,
        patch_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate mask logits in a batch.

        Args:
            mask_hidden: `(B, K, llm_hidden)` with K mask queries per group.
            image_features: `(B, grid_h * grid_w, vision_hidden)`, with smaller
                grids zero-padded to the batch maximum.
            grid_h: Shared patch-grid height for the batch.
            grid_w: Shared patch-grid width for the batch.
            patch_key_padding_mask: Optional `(B, grid_h * grid_w)` bool mask
                where True denotes padding. Fully padded placeholder rows must
                pass all False.

        Returns:
            `(B, K, grid_h * 16, grid_w * 16)` mask logits.
        """
        batch_size = self._validate_inputs(
            mask_hidden,
            image_features,
            grid_h,
            grid_w,
            patch_key_padding_mask,
        )

        # 步骤 1：把区域标记处的文字状态转换成区域查询。
        # 形状 B×K×D → B×K×C：B 为本次分组数，K 为每组区域数，C 为共享表示宽度。
        mask_embedding = self.mask_embed(mask_hidden)
        # 步骤 2：将图片特征宽度 V 转到文字模型宽度 D；小块数 P=h×w 不变。
        # 随后小块之间互相交换信息，得到更能理解整张图的表示。
        patch_features = self.patch_proj(image_features)
        patch_embedding = self._maybe_checkpoint(
            self.patch_self_attn,
            patch_features,
            patch_key_padding_mask,
        )
        # 步骤 3：把小块的一维列表放回二维网格：B×(h·w)×D → B×D×h×w。
        # reshape 改排列形状，permute 调整轴顺序；必须先确认小块顺序已恢复成逐行空间顺序。
        patch_embedding = patch_embedding.reshape(
            batch_size,
            grid_h,
            grid_w,
            -1,
        ).permute(0, 3, 1, 2)

        # 步骤 4：先减少通道数，再逐步增加高宽。通道是每个位置的信息种类，不是 RGB 颜色数。
        # 例如 h=2,w=3，放大后的像素表示高 8、宽 12，接下来每个区域都给这些位置评分。
        reduced_features = self.channel_reduce(patch_embedding)
        pixel_embedding = self.upscaler(reduced_features)

        # Divide by √C_emb to prevent high initial dot-product variance from saturating sigmoid.
        # 步骤 5：每个区域查询与每个像素表示做点积，沿 C 求和。
        # "bkc,bchw->bkhw" 表示保留批次 b、区域 k、高 h、宽 w，只把共同宽度 c 相乘再累加。
        # 除以 √C 控制初始分数幅度，避免一开始 sigmoid 就接近 0 或 1 而难以学习。
        mask_logits = torch.einsum(
            "bkc,bchw->bkhw",
            mask_embedding,
            pixel_embedding,
        ) / (self.embed_dim**0.5)

        # Interpolate K independently as channels; a fixed integer scale stabilizes
        # sampling offsets.
        # 步骤 6：高宽再乘 4，得到模型画布大小的原始分数。
        # 上例最终为 B×K×32×48。这里只输出分数，不在这里用阈值切断，训练才能继续传递梯度。
        return _bilinear_interpolate_4x(mask_logits)

    # 【步骤说明】检查区域查询和图片特征的维数、批次及网格数量是否匹配；合法时返回批次大小，不匹配则报错。
    # 对照参数：mask_hidden, image_features, grid_h, grid_w, patch_key_padding_mask；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def _validate_inputs(
        mask_hidden: torch.Tensor,
        image_features: torch.Tensor,
        grid_h: int,
        grid_w: int,
        patch_key_padding_mask: torch.Tensor | None,
    ) -> int:
        """Validate decoder inputs and return the batch size."""
        if mask_hidden.dim() != 3:
            raise ValueError(
                "mask_hidden must be (B, K, llm_hidden); "
                f"got {tuple(mask_hidden.shape)}"
            )
        if image_features.dim() != 3:
            raise ValueError(
                "image_features must be (B, P, vision_hidden); "
                f"got {tuple(image_features.shape)}"
            )

        batch_size = mask_hidden.shape[0]
        image_batch_size, num_patches = image_features.shape[:2]
        if image_batch_size != batch_size:
            raise ValueError(
                "mask_hidden and image_features must share batch dim; "
                f"got {batch_size} vs {image_batch_size}"
            )

        expected_patches = grid_h * grid_w
        if num_patches != expected_patches:
            raise ValueError(
                f"Patch feature count {num_patches} does not match grid "
                f"({grid_h}, {grid_w})"
            )

        if patch_key_padding_mask is not None:
            expected_shape = (batch_size, num_patches)
            if tuple(patch_key_padding_mask.shape) != expected_shape:
                raise ValueError(
                    "patch_key_padding_mask must have shape "
                    f"{expected_shape}; got {tuple(patch_key_padding_mask.shape)}"
                )
            if patch_key_padding_mask.dtype != torch.bool:
                raise TypeError(
                    "patch_key_padding_mask must use torch.bool; "
                    f"got {patch_key_padding_mask.dtype}"
                )

        return batch_size
