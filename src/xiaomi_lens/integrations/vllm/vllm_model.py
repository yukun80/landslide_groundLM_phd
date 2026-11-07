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

# 【中文导读｜阅读第 7 步】
# 本文件：在 vLLM 的模型中接入相同的区域解码器，收集图片特征并按请求生成区域。
# 怎么读：先看 decode_mask_batch，再看收集特征的两个小函数；这里预期每个请求只有一张图片。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Extend vLLM's Qwen3.5 model with PixelMaskDecoder for pooling requests.

`XiaomiLensForVLLM` subclasses `Qwen3_5ForConditionalGeneration` from vLLM and adds:

  1. `self.mask_decoder = PixelMaskDecoder(...)` — the pure-torch segmentation head
     implemented with PyTorch and shared with HF inference.
  2. Two forward-pre-hooks on the vision tower: one on `self.visual` captures the
     `grid_thw` argument (PoolingMetadata does not carry it), one on
     `self.visual.merger` captures the pre-merge ViT patch features used by the
     mask decoder.
  3. `self.pooler` (a `Pooler` for the "plugin" task): on a pooling request over
     `[prompt ++ completion]`, gathers the `<|mask|>` hidden states + the stashed ViT
     features and runs the decoder → a flat mask-logits tensor. The io_processor
     reshapes + serializes it.

Text generation and mask pooling run in separate vLLM processes. The pooling
request replays the image, instruction, and generated completion.

Weights: the checkpoint uses the base Qwen3.5 layout (`model.language_model.*`,
`model.visual.*`) plus `mask_decoder.*`; `load_weights` passes the backbone to
vLLM's loader and routes mask_decoder.* to the head.
"""

from __future__ import annotations

from collections.abc import Iterable, Set

import torch

from xiaomi_lens.constants import MASK_TOKEN

from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.tasks import PoolingTask


# 【类的用途】从文字状态中取出区域查询，并把对应图像特征交给区域解码器，返回逐请求输出。
class XiaomiLensMaskPooler(Pooler):
    """Pooler that decodes the `<|mask|>` into a dense mask via PixelMaskDecoder.

    Registered under the "plugin" pooling task; the io_processor drives it and
    reshapes the flat output. Reads the model's per-forward captures (pre-merge ViT
    features + grid_thw) that the vision hooks stash.
    """

    # 【步骤说明】保存所属模型的引用，避免将它再次登记成自己的子模块形成循环。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：model；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, model: "XiaomiLensForVLLM"):
        super().__init__()
        # Bypass nn.Module.__setattr__ so the model is NOT registered as a child
        # module of this pooler — the pooler is itself a child of the model, and a
        # back-reference child would make model<->pooler a cycle (infinite recursion
        # in .train()/.to()/parameters()).
        object.__setattr__(self, "_model", model)

    # 【步骤说明】返回该区域输出器支持的 plugin 任务集合，供 vLLM 判断能否处理请求。
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"plugin"}

    # 【步骤说明】要求框架在区域处理时保留文字编号；返回任务设置，因为必须知道哪些位置是区域特殊符号。
    # 对照参数：task；下面保留原始英文说明与类型，方便对应调用处。
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        # We need the prompt token ids to locate the `<|mask|>` position; without
        # this flag vLLM leaves `pooling_metadata.prompt_token_ids` as None.
        return PoolingParamsUpdate(requires_token_ids=True)

    # 【步骤说明】按每个请求的文字长度拆开状态与编号，配上捕获的图片特征，再返回区域解码结果列表。
    # 对照参数：hidden_states, pooling_metadata；下面保留原始英文说明与类型，方便对应调用处。
    def forward(self, hidden_states, pooling_metadata):
        model = self._model
        # hidden_states is (num_tokens, llm_hidden) for the batch. Split it back
        # into requests by prompt_lens, then batch requests with identical patch
        # grids for PixelMaskDecoder.
        prompt_lens = pooling_metadata.prompt_lens.tolist()
        ids_all = pooling_metadata.prompt_token_ids
        hidden_groups: list[torch.Tensor] = []
        id_groups: list = []
        vision_groups = model.split_captured_vision(len(prompt_lens))
        start = 0
        for req_idx, plen in enumerate(prompt_lens):
            hidden_groups.append(hidden_states[start:start + plen])
            id_groups.append(ids_all[req_idx][:plen] if ids_all is not None else None)
            start += plen
        return model.decode_mask_batch(hidden_groups, id_groups, vision_groups)


# 【类的用途】扩展 vLLM 版 Qwen 模型，添加区域解码器与收集图片特征的回调。
class XiaomiLensForVLLM(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True

    # 【步骤说明】先创建基础模型，再创建区域解码器、取得区域符号编号并登记图片特征捕获回调。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：vllm_config, prefix；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        from xiaomi_lens.model.config import build_mask_decoder_config
        from xiaomi_lens.model.pixel_mask_decoder import PixelMaskDecoder

        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        self.mask_decoder = PixelMaskDecoder(build_mask_decoder_config(hf_config))

        vis = hf_config.vision_config
        self._merge_size = int(getattr(vis, "spatial_merge_size", 2) or 2)

        from vllm.transformers_utils.tokenizer import get_tokenizer
        mc = vllm_config.model_config
        tok = get_tokenizer(mc.tokenizer, trust_remote_code=mc.trust_remote_code)
        self._mask_token_id = tok.convert_tokens_to_ids(MASK_TOKEN)
        self._pre_merge: torch.Tensor | None = None
        self._grid_thw: torch.Tensor | None = None

        # Vision tower captures. `self.visual(x, grid_thw)` — grab grid_thw (not in
        # PoolingMetadata). `self.visual.merger(hidden)` — grab the pre-merge patch
        # features (its input) that PixelMaskDecoder consumes.
        # 【步骤说明】视觉模块开始计算前读取网格参数，保存在模型对象中供区域解码使用；返回 None，不替换原输入。
        # 对照参数：_m, args, kwargs；下面保留原始英文说明与类型，方便对应调用处。
        def _grab_grid(_m, args, kwargs):
            # vLLM calls `self.visual(pixel_values, grid_thw=grid_thw)` — grid_thw
            # is a KEYWORD arg, so read kwargs first, fall back to positional.
            g = kwargs.get("grid_thw")
            if g is None and len(args) >= 2:
                g = args[1]
            if g is not None:
                self._grid_thw = g
            return None

        # 【步骤说明】在图片小块合并前保存其特征，供区域分支恢复细节；返回 None，不更改主干输入。
        # 对照参数：_m, args；下面保留原始英文说明与类型，方便对应调用处。
        def _grab_pre_merge(_m, args):
            self._pre_merge = args[0].detach()
            return None

        self.visual.register_forward_pre_hook(_grab_grid, with_kwargs=True)
        self.visual.merger.register_forward_pre_hook(_grab_pre_merge)

        self.pooler = XiaomiLensMaskPooler(self)

    # ---- weight loading -------------------------------------------------------
    # 【步骤说明】按参数名分开基础模型权重与区域解码器权重，分别加载；返回已加载名称集合。
    # 对照参数：weights；下面保留原始英文说明与类型，方便对应调用处。
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        base_weights: list[tuple[str, torch.Tensor]] = []
        mask_sd: dict[str, torch.Tensor] = {}
        for name, w in weights:
            if name.startswith("mask_decoder."):
                mask_sd[name[len("mask_decoder."):]] = w
            else:
                base_weights.append((name, w))

        loaded = super().load_weights(base_weights)
        if mask_sd:
            missing, unexpected = self.mask_decoder.load_state_dict(mask_sd, strict=False)
            if unexpected:
                raise ValueError(f"unexpected mask_decoder keys: {list(unexpected)[:4]}")
            loaded |= {f"mask_decoder.{k}" for k in mask_sd}
        return loaded

    # ---- mask decode ----------------------------------------------------------
    # 【步骤说明】将按合并块排列的图片特征恢复为从上到下、从左到右的顺序；返回按空间位置排列的二维特征。
    # 对照参数：tokens, grid_h, grid_w；下面保留原始英文说明与类型，方便对应调用处。
    def _reorder_pre_merge(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """ViT stores tokens grouped by merge blocks; permute to row-major (P, hidden).

        Mirror of `modeling_xiaomi_lens._extract_pre_merge_features`.
        """
        m = self._merge_size
        merged_h, merged_w = grid_h // m, grid_w // m
        hidden = tokens.shape[-1]
        return (
            tokens.view(merged_h, merged_w, m, m, hidden)
            .permute(0, 2, 1, 3, 4)
            .reshape(grid_h * grid_w, hidden)
            .contiguous()
        )

    # 【步骤说明】按每个请求的网格切分刚收集的图片特征；返回特征与网格列表，数量或维数不符时返回空占位。
    # 对照参数：request_count；下面保留原始英文说明与类型，方便对应调用处。
    def split_captured_vision(
        self, request_count: int
    ) -> list[tuple[torch.Tensor, tuple[int, int]] | None]:
        """Split captured ViT features, expecting one image per pooling request."""
        if self._grid_thw is None or self._pre_merge is None:
            return [None] * request_count
        grid = self._grid_thw
        rows = grid.reshape(-1, 3).tolist() if isinstance(grid, torch.Tensor) else list(grid)
        if len(rows) != request_count:
            return [None] * request_count

        pre = self._pre_merge.reshape(-1, self._pre_merge.shape[-1])
        groups: list[tuple[torch.Tensor, tuple[int, int]] | None] = []
        start = 0
        for row in rows:
            temporal, grid_h, grid_w = (int(value) for value in row)
            patch_count = temporal * grid_h * grid_w
            end = start + patch_count
            if temporal != 1 or end > pre.shape[0]:
                return [None] * request_count
            groups.append((pre[start:end], (grid_h, grid_w)))
            start = end
        if start != pre.shape[0]:
            return [None] * request_count
        return groups

    # 【步骤说明】定位各请求的区域符号，把相同网格的请求分组一次解码；返回每请求带网格和数量头部的扁平输出。
    # 对照参数：hidden_groups, prompt_id_groups, vision_groups；下面保留原始英文说明与类型，方便对应调用处。
    def decode_mask_batch(
        self,
        hidden_groups: list[torch.Tensor],
        prompt_id_groups: list,
        vision_groups: list[tuple[torch.Tensor, tuple[int, int]] | None],
    ) -> list[torch.Tensor]:
        """Group identical patch grids and run PixelMaskDecoder in batches."""
        outputs = [hidden.new_zeros(1, dtype=torch.float32) for hidden in hidden_groups]
        grouped: dict[
            tuple[int, int],
            list[tuple[int, torch.Tensor, torch.Tensor]],
        ] = {}
        for index, (hidden, prompt_ids, vision) in enumerate(
            zip(hidden_groups, prompt_id_groups, vision_groups, strict=True)
        ):
            if prompt_ids is None or vision is None:
                continue
            pre, grid = vision
            grid_h, grid_w = grid
            # Preserve the image canvas even when generation emitted no mask.
            # The IO processor uses that canvas with K=0 to serialize an empty
            # mask list.
            outputs[index] = torch.tensor(
                [grid_h, grid_w, 0], device=hidden.device, dtype=torch.float32
            )
            ids = (
                prompt_ids
                if isinstance(prompt_ids, torch.Tensor)
                else torch.as_tensor(prompt_ids)
            )
            ids = ids.reshape(-1).to(hidden.device)
            mask_pos = (ids == self._mask_token_id).nonzero(as_tuple=True)[0]
            if mask_pos.numel() == 0 or int(mask_pos.max()) >= hidden.shape[0]:
                continue
            mask_hidden = hidden.index_select(0, mask_pos)
            # 只把网格大小完全相同的请求放一起计算，避免把不同图片的空间排列混用。
            grouped.setdefault(grid, []).append((index, mask_hidden, pre))

        for (grid_h, grid_w), entries in grouped.items():
            max_masks = max(mask_hidden.shape[0] for _, mask_hidden, _ in entries)
            hidden_size = entries[0][1].shape[-1]
            vision_hidden = entries[0][2].shape[-1]
            mask_batch = entries[0][1].new_zeros(
                (len(entries), max_masks, hidden_size)
            )
            image_batch = entries[0][2].new_zeros(
                (len(entries), grid_h * grid_w, vision_hidden)
            )
            mask_counts: list[int] = []
            for batch_index, (_, mask_hidden, pre) in enumerate(entries):
                count = int(mask_hidden.shape[0])
                mask_counts.append(count)
                mask_batch[batch_index, :count] = mask_hidden
                image_batch[batch_index] = self._reorder_pre_merge(pre, grid_h, grid_w)

            logits = self.mask_decoder(
                mask_batch.to(image_batch.dtype), image_batch, grid_h, grid_w
            )
            for batch_index, (output_index, _, _) in enumerate(entries):
                count = mask_counts[batch_index]
                flat = logits[batch_index, :count].reshape(-1).float()
                header = torch.tensor(
                    [grid_h, grid_w, count],
                    device=flat.device,
                    dtype=torch.float32,
                )
                outputs[output_index] = torch.cat([header, flat])
        return outputs

    # 【步骤说明】把单个请求包装成批次接口需要的列表，再取回单请求结果；保持旧调用方式可用。
    # 对照参数：hidden_states, prompt_ids, vision；下面保留原始英文说明与类型，方便对应调用处。
    def decode_mask(
        self,
        hidden_states: torch.Tensor,
        prompt_ids,
        *,
        vision: tuple[torch.Tensor, tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Compatibility entry point for one request, reusing batched decoding."""
        if vision is None:
            fallback = self.split_captured_vision(1)
            vision = fallback[0] if fallback else None
        return self.decode_mask_batch(
            [hidden_states], [prompt_ids], [vision]
        )[0]
