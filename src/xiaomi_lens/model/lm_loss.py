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
# 本文件：衡量下一个文字单元是否预测正确，同时提供节省显存的等价计算路径。
# 怎么读：先理解 shift_labels_flat 的错一位对齐，再看 _causal_lm_loss；加速路径不改变学习目标。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Causal LM loss implementations used during training.

`config.lm_loss_impl` selects one of `LM_LOSS_IMPLS`, falling back in order:

- `fused_linear_ce`: liger-kernel fused `lm_head` + cross entropy. The
  `(B, S, vocab)` logits are never materialized.
- `flash_ce`: flash-attn cross entropy on BF16/FP16 logits with in-place backward.
- `hf`: Transformers' FP32 `ForCausalLMLoss`, used on CPU and for debugging.

liger-kernel and flash-attn are optional; failed imports log a warning once and
fall back to the next implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LM_LOSS_IMPLS = ("fused_linear_ce", "flash_ce", "hf")

_FUSED_LINEAR_CE: Optional[tuple] = None
_FUSED_LINEAR_CE_IMPORT_ATTEMPTED = False
_FLASH_CE: Optional[tuple] = None
_FLASH_CE_IMPORT_ATTEMPTED = False


# 【步骤说明】第一次需要时尝试导入融合版文字评分工具并缓存，返回求平均与求和两个版本；不可用时返回 None。
def _get_fused_linear_ce() -> Optional[tuple]:
    """Return liger's `(mean, sum)` fused linear+CE modules, or None if unavailable.

    The public `LigerFusedLinearCrossEntropyLoss` module is used rather than
    the raw autograd Function, whose positional argument order differs across
    liger releases.
    """
    global _FUSED_LINEAR_CE, _FUSED_LINEAR_CE_IMPORT_ATTEMPTED
    if _FUSED_LINEAR_CE is None and not _FUSED_LINEAR_CE_IMPORT_ATTEMPTED:
        _FUSED_LINEAR_CE_IMPORT_ATTEMPTED = True
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            _FUSED_LINEAR_CE = (
                LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean"),
                LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum"),
            )
        except Exception as exc:
            logger.warning(
                "[lm-loss] liger-kernel fused linear cross entropy unavailable; "
                "falling back to materialized logits: %s",
                exc,
            )
    return _FUSED_LINEAR_CE


# 【步骤说明】尝试取得 flash-attn 的交叉熵计算工具并缓存；返回均值和总和版本，失败后由调用方使用普通计算。
def _get_flash_ce() -> Optional[tuple]:
    """Return flash-attn's `(mean, sum)` cross-entropy modules, or None if unavailable."""
    global _FLASH_CE, _FLASH_CE_IMPORT_ATTEMPTED
    if _FLASH_CE is None and not _FLASH_CE_IMPORT_ATTEMPTED:
        _FLASH_CE_IMPORT_ATTEMPTED = True
        try:
            from flash_attn.losses.cross_entropy import CrossEntropyLoss

            # In-place backward reuses the logits storage for dlogits; training
            # never reads logits after the loss.
            _FLASH_CE = (
                CrossEntropyLoss(ignore_index=-100, reduction="mean", inplace_backward=True),
                CrossEntropyLoss(ignore_index=-100, reduction="sum", inplace_backward=True),
            )
        except Exception as exc:
            logger.warning(
                "[lm-loss] flash-attn fused cross entropy unavailable; "
                "falling back to HF FP32 causal LM loss: %s",
                exc,
            )
    return _FLASH_CE


# 【步骤说明】读取当前参与训练的进程总数；没有初始化多进程训练时返回 1。
def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# 【步骤说明】将正确答案左移一位并展平，末尾用 -100 补齐；例如 [10,20,30] 变成 [20,30,-100]，因为当前位置预测下一个词。
# 对照参数：labels；下面保留原始英文说明与类型，方便对应调用处。
def shift_labels_flat(labels: torch.Tensor) -> torch.Tensor:
    """Shift labels one position left and flatten, padding with `-100`.

    Matches the shift/ignore semantics of HF's `ForCausalLMLoss`.
    """
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    return shift_labels.contiguous().view(-1)


# 【类的用途】给主模型补充文字评分功能的辅助类；依赖主模型已有的词表输出层与设置。
class CausalLMLossMixin:
    """LM-loss methods for a `*ForConditionalGeneration` model with `lm_head`.

    With `num_items_in_batch=None` every implementation returns the plain mean
    over supervised tokens. Otherwise it returns the token-sum scaled by
    `_token_sum_scale`, which HF Trainer expects when the model sets
    `accepts_loss_kwargs = True`.
    """

    # 【步骤说明】检查训练模式、设备、选项和加速工具是否满足条件，返回是否可跳过完整词表分数的显式保存。
    # 对照参数：hidden_states, logits_to_keep；下面保留原始英文说明与类型，方便对应调用处。
    def _use_fused_linear_lm_loss(self, hidden_states: torch.Tensor, logits_to_keep) -> bool:
        """Whether this forward can skip `lm_head` and fuse the projection into the loss.

        Only training forwards qualify: evaluation and generation still need
        real logits. liger kernels are Triton-only, so CPU tensors take the
        reference path. `logits_to_keep` slicing is a generation feature and
        also falls back.
        """
        if not self.training:
            return False
        if getattr(self.config, "lm_loss_impl", "fused_linear_ce") != "fused_linear_ce":
            return False
        if not hidden_states.is_cuda:
            return False
        if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
            return False
        return _get_fused_linear_ce() is not None

    # 【步骤说明】根据全局有效文字数量与进程数返回缩放系数，让不同进程参与平均后仍按有效文字数计分。
    # 对照参数：num_items_in_batch, like；下面保留原始英文说明与类型，方便对应调用处。
    def _token_sum_scale(self, num_items_in_batch, like: torch.Tensor) -> torch.Tensor:
        """Return `world_size / num_items_in_batch` for HF's token-mean contract.

        With `average_tokens_across_devices` (HF default), Trainer counts
        supervised tokens across the accumulation window and all ranks. Each
        rank emits `token_sum * W / global_count`; DDP/ZeRO gradient averaging
        then yields the exact global per-token mean.
        """
        denom = torch.as_tensor(
            num_items_in_batch, dtype=like.dtype, device=like.device
        ).reshape(-1)[0].clamp(min=1)
        world = (
            _dist_world_size()
            if getattr(self.config, "average_tokens_across_devices", True)
            else 1
        )
        return like.new_tensor(float(world)) / denom

    # 【步骤说明】从隐藏状态和标签中只取要学习的位置，融合文字预测与评分，返回文字损失，减少巨大词表带来的临时存储。
    # 对照参数：hidden_states, labels, num_items_in_batch；下面保留原始英文说明与类型，方便对应调用处。
    def _fused_linear_lm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """Compute causal LM loss without materializing logits.

        Only rows whose shifted label is supervised reach the `lm_head`
        projection, so prompt, image and padding tokens skip the vocabulary
        GEMM. The boolean gather has a scatter-add backward, so gradients still
        reach the full `hidden_states`.
        """
        mean_ce, sum_ce = _get_fused_linear_ce()
        shift_labels = shift_labels_flat(labels)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        supervised = shift_labels != -100
        sel_hidden = flat_hidden[supervised]
        sel_labels = shift_labels[supervised]
        bias = getattr(self.lm_head, "bias", None)
        if num_items_in_batch is None:
            return mean_ce(self.lm_head.weight, sel_hidden, sel_labels, bias)
        token_sum = sum_ce(self.lm_head.weight, sel_hidden, sel_labels, bias)
        return token_sum * self._token_sum_scale(num_items_in_batch, token_sum)

    # 【步骤说明】接收文字预测分数和正确标签，优先使用可用加速计算，否则调用基础模型的评分方法；返回文字损失。
    # 对照参数：logits, labels, num_items_in_batch；下面保留原始英文说明与类型，方便对应调用处。
    def _causal_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """Compute causal LM loss from materialized logits.

        flash-attn CE reads BF16/FP16 logits directly and reduces in FP32
        internally, avoiding the full FP32 logits copy that HF's loss makes.
        """
        shift_labels = shift_labels_flat(labels)
        want_flash = getattr(self.config, "lm_loss_impl", "fused_linear_ce") != "hf"
        if want_flash and logits.is_cuda and logits.dtype in (torch.float16, torch.bfloat16):
            flash_ce = _get_flash_ce()
            if flash_ce is not None:
                mean_ce, sum_ce = flash_ce
                flat_logits = logits.view(-1, logits.shape[-1])
                if num_items_in_batch is None:
                    return mean_ce(flat_logits, shift_labels)
                token_sum = sum_ce(flat_logits, shift_labels)
                return token_sum * self._token_sum_scale(num_items_in_batch, token_sum)

        vocab_size = self.config.text_config.vocab_size
        if num_items_in_batch is None:
            return self.loss_function(logits=logits, labels=labels, vocab_size=vocab_size)
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
            num_items_in_batch=num_items_in_batch,
        )
        if getattr(self.config, "average_tokens_across_devices", True):
            loss = loss * _dist_world_size()
        return loss
