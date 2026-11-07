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
# 本文件：把基础图文模型与区域解码器接起来，是理解算法的核心文件。
# 怎么读：先看 forward 的主线，再看 _prepare_multimodal_inputs 和 _decode_pixel_masks。基础 Qwen 的内部层在 transformers 中。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Extend Transformers' Qwen3.5 model with a <|mask|> token and pixel-mask decoder.

Core design:
1. Add the <|mask|> special token to the vocabulary.
2. PixelMaskDecoder applies self-attention to pre-merge ViT patch features and
   upsamples them 4× with two stride-2 transposed convolutions.
3. Each <|mask|> hidden state produces a dynamic classifier over the resulting
   pixel embeddings. A further 4× bilinear interpolation brings the mask logits
   to `(grid_h*16, grid_w*16)`, matching the vision encoder's input resolution.
4. When `labels` / `mask_targets` are passed, forward also returns the training
   loss `lm_loss_weight * lm_loss + mask_loss_weight * mask_loss`, where the
   mask loss is dice + BCE at canvas resolution.
"""

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from .config import build_mask_decoder_config, get_qwen_vision_config
from .lm_loss import CausalLMLossMixin
from .losses import MaskLoss
from .pixel_mask_decoder import PixelMaskDecoder

# Each ViT patch spans 16 input pixels. Learned 4× upsampling followed by 4×
# bilinear interpolation restores masks to the vision encoder's input resolution.
MASK_CANVAS_SCALE = 16


# 【类的用途】整理一次模型计算的返回值，包括文字分数、区域分数、有效像素和两类训练损失。
@dataclass
class XiaomiLensOutput(ModelOutput):
    """Output type for XiaomiLensQwen3_5ForConditionalGeneration."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.Tensor] = None
    pred_masks: Optional[torch.Tensor] = None
    pred_mask_valid: Optional[torch.Tensor] = None
    # Weighted per-microbatch training contributions; `loss == lm_loss + mask_loss`.
    lm_loss: Optional[torch.Tensor] = None
    mask_loss: Optional[torch.Tensor] = None


# 【类的用途】继承基础图文模型并接入区域分支，统一文字生成、像素预测与联合训练。
class XiaomiLensQwen3_5ForConditionalGeneration(CausalLMLossMixin, Qwen3_5ForConditionalGeneration):
    # Lets HF Trainer pass `num_items_in_batch` into forward; see
    # `CausalLMLossMixin._token_sum_scale` for how the LM loss consumes it.
    accepts_loss_kwargs = True

    # 【步骤说明】先构造基础 Qwen 图文模型，再创建区域解码器和区域损失函数。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：config；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, config):
        super().__init__(config)

        decoder_config = getattr(config, 'mask_decoder', None)
        if decoder_config is None:
            decoder_config = build_mask_decoder_config(config)
        self.mask_decoder = PixelMaskDecoder(decoder_config)

        # Parameter-free, so it does not change the checkpoint's state dict.
        self.mask_loss_fn = MaskLoss(
            dice_weight=getattr(config, "mask_dice_weight", 1.0),
            bce_weight=getattr(config, "mask_bce_weight", 2.0),
            chunk_size=getattr(config, "mask_loss_chunk_size", 8),
        )

    # 【步骤说明】转交给统一设置读取函数，返回图像部分设置，避免在模型类中重复处理兼容名称。
    # 对照参数：config；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def _get_vision_config(config):
        return get_qwen_vision_config(config)

    # 【步骤说明】读取图像小块合并的边长倍数；返回整数，用于恢复合并前的空间排列。
    def _get_spatial_merge_size(self) -> int:
        vision_config = self._get_vision_config(self.config)
        return int(getattr(vision_config, "spatial_merge_size", 1) or 1)

    # 【步骤说明】从视觉输出提取供语言模型使用的合并后特征，统一返回列表；缺失必要特征时报错。
    # 对照参数：visual_outputs；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def _flatten_image_features(visual_outputs) -> list[torch.Tensor]:
        image_features = visual_outputs.pooler_output
        if image_features is None:
            raise ValueError(
                "visual_outputs.pooler_output is None; ensure get_image_features returns patch features."
            )
        if isinstance(image_features, torch.Tensor):
            return [image_features]
        return list(image_features)

    # 【步骤说明】按每张图片的网格尺寸切分视觉特征，再将合并块顺序还原成逐行顺序；返回每张图片的细网格特征列表。
    # 对照参数：visual_outputs, image_grid_thw；下面保留原始英文说明与类型，方便对应调用处。
    def _extract_pre_merge_features(
        self,
        visual_outputs,
        image_grid_thw: torch.LongTensor,
    ) -> list[torch.Tensor]:
        """Split ViT `last_hidden_state` per-image and reorder to row-major spatial layout.

        Qwen3.5 ViT stores tokens grouped by merge blocks:
            (merged_h, merged_w, merge_size, merge_size, hidden)
        We permute back to row-major (grid_h, grid_w, hidden) at patch_size=16.
        """
        raw = visual_outputs.last_hidden_state
        if raw is None:
            raise ValueError(
                "visual_outputs.last_hidden_state is None; required for pre-merge patch features."
            )
        merge_size = self._get_spatial_merge_size()
        # One host copy covers all images, replacing three `.item()` syncs per image.
        thw_l = image_grid_thw.tolist()
        token_counts = [int(t) * int(h) * int(w) for t, h, w in thw_l]
        per_image = list(torch.split(raw, token_counts, dim=0))

        reordered: list[torch.Tensor] = []
        for batch_idx, tokens in enumerate(per_image):
            grid_t = int(thw_l[batch_idx][0])
            grid_h = int(thw_l[batch_idx][1])
            grid_w = int(thw_l[batch_idx][2])
            if grid_t != 1:
                raise ValueError(f"Xiaomi-Lens-0 expects image_grid_thw[:, 0] == 1 for images, got {grid_t}.")
            if grid_h % merge_size != 0 or grid_w % merge_size != 0:
                raise ValueError(
                    f"Image grid ({grid_h}, {grid_w}) must be divisible by merge_size={merge_size}."
                )
            merged_h = grid_h // merge_size
            merged_w = grid_w // merge_size
            hidden = tokens.shape[-1]
            # (merged_h, merged_w, m, m, hidden) -> (merged_h, m, merged_w, m, hidden) -> (grid_h, grid_w, hidden)
            # 恢复空间顺序：视觉主干为了合并相邻小块，会按小组存储特征。
            # 这里先展开“组行、组列、组内行、组内列”，再交换轴，使展平后的顺序对应真实图片行列。
            spatial = (
                tokens.view(merged_h, merged_w, merge_size, merge_size, hidden)
                .permute(0, 2, 1, 3, 4)
                .reshape(grid_h * grid_w, hidden)
                .contiguous()
            )
            reordered.append(spatial)
        return reordered

    # 【步骤说明】把文字编号转换成表示，把图片特征填入图片占位符，并准备空间位置信息；返回混合输入、视觉输出和位置编号。
    # 对照参数：input_ids, attention_mask, position_ids, past_key_values, pixel_values, image_grid_thw, mm_token_type_ids, inputs_embeds；下面保留原始英文说明与类型，方便对应调用处。
    def _prepare_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        mm_token_type_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.FloatTensor],
    ):
        visual_outputs = None
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        has_images = (
            pixel_values is not None
            and image_grid_thw is not None
            and pixel_values.numel() > 0
            and image_grid_thw.numel() > 0
        )

        if has_images:
            visual_outputs = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_features = self._flatten_image_features(visual_outputs)
            image_embeds = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            # 图片与文字会合：把占位符处的表示替换成图像编码器产生的合并后特征。
            # 区域解码器随后使用的是合并前的较细网格，两条支路共享视觉主干，但取用的位置不同。
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            # Insert spatially merged visual features into the image-token slots.

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if has_images else None,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        return inputs_embeds, visual_outputs, position_ids

    # 【步骤说明】按样本及所属图片收集区域查询，补齐不同大小的网格后统一解码；返回区域分数和有效像素范围，补齐区域不参与评分。
    # 对照参数：hidden_states, mask_positions, image_grid_thw, visual_outputs, mask_valid_mask, images_per_sample, mask_image_index, mask_canvas_grid；下面保留原始英文说明与类型，方便对应调用处。
    def _decode_pixel_masks(
        self,
        hidden_states: torch.Tensor,
        mask_positions: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        visual_outputs,
        mask_valid_mask: Optional[torch.Tensor] = None,
        images_per_sample: Optional[torch.Tensor] = None,
        mask_image_index: Optional[torch.Tensor] = None,
        mask_canvas_grid: Optional[tuple] = None,
    ):
        """Decode pixel masks via a single batched mask_decoder call.

        Every rank calls `self.mask_decoder` exactly once per forward, so
        DeepSpeed collectives on the decoder parameters stay in step even when
        ranks hold different numbers of masks.

        **The decoder's batch dim is a `(sample, image)` GROUP, not a sample.**
        `mask_image_index[b, k]` (sample-local) says which of row `b`'s images
        the k-th `<|mask|>` binds to — the nearest image preceding it in the
        token stream. Every `<|mask|>` bound to the same image of the same
        sample shares one group, so that image's patch features are lifted and
        attended over exactly once no matter how many queries hit it.

        `images_per_sample[i]` is the number of images owned by batch row `i`
        (0 / 1 / ≥2); a row with 0 images produces no group and its outputs
        stay zero. Padded slots (`mask_valid_mask[b, k] == 0`) also get no
        group and zero outputs.

        `mask_canvas_grid` is the `(grid_h, grid_w)` the trainer sized its
        targets at, so predictions and targets cannot disagree on H/W. When it
        is absent (inference), the canvas covers every image a `<|mask|>`
        binds to.

        Returns `(pred_masks, pred_mask_valid)`, both
        `(B, K, max_grid_h*16, max_grid_w*16)`: raw mask logits, and a 0/1 map
        of each mask's real (unpadded) area.
        """
        if mask_positions.dim() == 1:
            mask_positions = mask_positions.unsqueeze(1)
        mask_positions = mask_positions.to(hidden_states.device)

        batch_size = hidden_states.shape[0]
        K = mask_positions.shape[1]
        num_images = int(image_grid_thw.shape[0])

        if images_per_sample is None:
            # Back-compat: assume every sample contributes exactly one image.
            images_per_sample = torch.ones(batch_size, dtype=torch.long, device=hidden_states.device)
        else:
            images_per_sample = images_per_sample.to(hidden_states.device).long()

        # One host sync up front rather than an `.item()` per element in the loops.
        counts = images_per_sample.tolist()
        if sum(counts) != num_images:
            raise ValueError(
                f"images_per_sample sums to {sum(counts)} but "
                f"image_grid_thw has {num_images} rows."
            )

        positions_l = mask_positions.tolist()
        valid_l = (
            mask_valid_mask.tolist() if mask_valid_mask is not None
            else [[1.0] * K for _ in range(batch_size)]
        )
        if mask_image_index is None:
            img_idx_l = [[0] * K for _ in range(batch_size)]
        else:
            img_idx_l = mask_image_index.to(hidden_states.device).long().tolist()

        # Global image row of each sample's image 0.
        offsets, _cursor = [], 0
        for count in counts:
            offsets.append(_cursor)
            _cursor += count

        # Group the valid (b, k) slots by the IMAGE they bind to. Insertion order
        # makes the group index deterministic.
        # 分组不是按区域各复制一次图片：同一样本中属于同一图片的所有区域共用一组图片特征。
        # 例如一张图片有三只猫，只需要做一次这张图片的小块注意力，再用三个目标查询分别评分。
        groups: dict = {}
        for b in range(batch_size):
            n_img = counts[b]
            if n_img == 0:
                continue  # zero-image row: no group, its outputs stay zero
            for k in range(K):
                if not valid_l[b][k] > 0:
                    continue
                local = min(max(int(img_idx_l[b][k]), 0), n_img - 1)
                groups.setdefault((b, offsets[b] + local), []).append(k)

        group_keys = list(groups.keys())
        # Clamp to ≥1: attention on a zero-size batch is a landmine, and an
        # all-invalid batch must still make its one rank-symmetric call.
        num_groups = max(1, len(group_keys))
        max_queries = max(1, max((len(v) for v in groups.values()), default=1))

        if mask_canvas_grid is not None:
            max_grid_h, max_grid_w = int(mask_canvas_grid[0]), int(mask_canvas_grid[1])
        elif group_keys:
            # The canvas covers every image a <|mask|> binds to.
            _rows = torch.tensor(
                [row for _, row in group_keys], device=image_grid_thw.device, dtype=torch.long,
            )
            max_grid_h = int(image_grid_thw[_rows, 1].max().item())
            max_grid_w = int(image_grid_thw[_rows, 2].max().item())
        else:
            max_grid_h = int(image_grid_thw[:, 1].max().item())
            max_grid_w = int(image_grid_thw[:, 2].max().item())
        max_patches = max_grid_h * max_grid_w
        vision_hidden = int(self._get_vision_config(self.config).hidden_size)
        llm_hidden = hidden_states.shape[-1]

        image_patch_features = self._extract_pre_merge_features(visual_outputs, image_grid_thw)
        if len(image_patch_features) != num_images:
            raise ValueError(
                f"Expected {num_images} image feature groups, but got {len(image_patch_features)}."
            )

        # Decoder inputs are per-GROUP; outputs are scattered back per-(sample, k).
        # 为不同大小的图片与不同区域数准备统一容器。
        # 补零只是让数据能放成规则数组，后面的有效位置标记必须阻止这些补齐像素参与注意力和损失。
        grouped_mask_hidden = hidden_states.new_zeros((num_groups, max_queries, llm_hidden))
        grouped_image_features = hidden_states.new_zeros((num_groups, max_patches, vision_hidden))
        # Mark each mask's unpadded area at the decoder's output resolution,
        # which is 16× the patch grid along each spatial axis.
        mask_h = max_grid_h * MASK_CANVAS_SCALE
        mask_w = max_grid_w * MASK_CANVAS_SCALE
        pred_mask_valid = hidden_states.new_zeros((batch_size, K, mask_h, mask_w))
        # True marks padded keys ignored by self-attention. Only groups with
        # real image features receive padding masks; an empty placeholder stays
        # all-False and avoids the all-`-inf` softmax NaN.
        patch_key_padding_mask = torch.zeros(
            (num_groups, max_patches), dtype=torch.bool, device=hidden_states.device
        )

        # Scatter coordinates: grouped[g_idx, j_idx] → pred_masks[b_idx, k_idx].
        b_idx: list = []
        k_idx: list = []
        g_idx: list = []
        j_idx: list = []

        # Share one host copy across the scatter loop to avoid per-group `.item()` syncs.
        thw_l = image_grid_thw.tolist()
        for g, (b, image_row) in enumerate(group_keys):
            grid_h = int(thw_l[image_row][1])
            grid_w = int(thw_l[image_row][2])
            feats = image_patch_features[image_row]
            expected = grid_h * grid_w
            if feats.shape[0] != expected:
                raise ValueError(
                    f"Expected {expected} pre-merge ViT patch features, got {feats.shape[0]}."
                )
            if grid_h > max_grid_h or grid_w > max_grid_w:
                raise ValueError(
                    f"Image grid ({grid_h}, {grid_w}) exceeds the mask canvas grid "
                    f"({max_grid_h}, {max_grid_w}); `mask_canvas_grid` must cover every "
                    "image that a valid <|mask|> binds to."
                )

            # This image's features, shared by every query in the group via einsum.
            feat_2d = feats.reshape(grid_h, grid_w, vision_hidden)
            padded = F.pad(
                feat_2d,
                (0, 0, 0, max_grid_w - grid_w, 0, max_grid_h - grid_h),
            )
            grouped_image_features[g] = padded.reshape(max_patches, vision_hidden)

            # Mark the padded patches (outside this image's real grid) so the
            # decoder's self-attention ignores them. Layout matches the F.pad
            # above: position (r, c) in the (max_grid_h, max_grid_w) row-major
            # flatten is real iff r < grid_h and c < grid_w.
            pad_mask_2d = torch.ones(
                (max_grid_h, max_grid_w), dtype=torch.bool, device=hidden_states.device
            )
            pad_mask_2d[:grid_h, :grid_w] = False
            patch_key_padding_mask[g] = pad_mask_2d.reshape(max_patches)

            for j, k in enumerate(groups[(b, image_row)]):
                pos = int(positions_l[b][k])
                # 关键连接点：从语言模型中取的是 <|mask|> 所在位置的上下文状态，
                # 它已经结合了图片、问题与前文，不是单独读取一个永远不变的特殊符号词向量。
                grouped_mask_hidden[g, j] = hidden_states[b, pos]
                pred_mask_valid[
                    b, k,
                    : grid_h * MASK_CANVAS_SCALE,
                    : grid_w * MASK_CANVAS_SCALE,
                ] = 1
                b_idx.append(b)
                k_idx.append(k)
                g_idx.append(g)
                j_idx.append(j)

        # 整批只调用一次区域解码器，便于多个训练进程保持相同的执行次序。
        # 即使某个进程没有有效区域，也需要配合其他进程，不能独自提前跳过所有相关计算。
        grouped = self.mask_decoder(
            grouped_mask_hidden,         # (N, Kg, llm_hidden)
            grouped_image_features,      # (N, max_patches, vision_hidden)
            grid_h=max_grid_h,
            grid_w=max_grid_w,
            patch_key_padding_mask=patch_key_padding_mask,  # (N, max_patches) True=pad
        )  # (N, Kg, max_grid_h*16, max_grid_w*16) — already canvas resolution

        # Scatter the groups back into the (sample, k) layout. The index-assign
        # lowers to `index_put_`, so gradients flow back to `grouped`.
        # 解码完后把“图片分组”中的结果放回“样本编号、区域编号”的原顺序，
        # 这样第 k 个输出仍与第 k 个区域标记、训练标注一一对应。
        pred_masks = grouped.new_zeros((batch_size, K, mask_h, mask_w))
        if b_idx:
            pred_masks[b_idx, k_idx] = grouped[g_idx, j_idx]
        del grouped
        return pred_masks, pred_mask_valid

    # 【步骤说明】接收图文输入及可选正确答案；先计算共享图文状态，再生成文字和区域分数，训练时返回两类加权损失。
    # 对照参数：input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids, logits_to_keep, mask_positions, mask_valid_mask, mask_image_index, mask_canvas_grid, mask_targets, mask_pixel_valid, num_items_in_batch, gradient_accumulation_steps；下面保留原始英文说明与类型，方便对应调用处。
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        mask_positions: Optional[torch.Tensor] = None,
        mask_valid_mask: Optional[torch.Tensor] = None,
        mask_image_index: Optional[torch.Tensor] = None,
        mask_canvas_grid: Optional[tuple] = None,
        mask_targets: Optional[torch.Tensor] = None,
        mask_pixel_valid: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        gradient_accumulation_steps: Optional[int] = None,
        **kwargs,
    ):
        """Run the LM and, when `mask_positions` is given, decode one mask per `<|mask|>`.

        `mask_positions` is `(B, K)` token indices of `<|mask|>` slots in
        `input_ids`. `mask_image_index` optionally binds each slot to one of the
        sample's images; by default each slot binds to the nearest preceding
        image.

        Training passes `labels` for the LM loss and `mask_targets` /
        `mask_pixel_valid` / `mask_valid_mask` / `mask_canvas_grid` for the
        mask loss (see `XiaomiLensTrainer.compute_loss`). Without them, forward
        is the inference path.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise ValueError("Xiaomi-Lens-0 currently supports image inputs only.")

        inputs_embeds, visual_outputs, position_ids = self._prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # 共享主干步骤：基础语言模型处理混合后的图文序列。
        # 其内部的注意力、状态更新和缓存实现不在本仓库，而来自 transformers 的 Qwen3.5 实现。
        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        loss = None
        logits = None
        lm_loss = None
        mask_loss = None

        # A negative weight disables that loss term.
        lm_loss_weight = getattr(self.config, "lm_loss_weight", 1.0)
        mask_loss_weight = getattr(self.config, "mask_loss_weight", 1.0)
        compute_lm_loss = labels is not None and lm_loss_weight >= 0
        is_training_forward = labels is not None or mask_targets is not None

        # The fused kernel folds `lm_head` into the loss, so no logits are
        # materialized. Mask-only training (`lm_loss_weight < 0`) skips the
        # full-vocab projection entirely.
        use_fused_lm_loss = compute_lm_loss and self._use_fused_linear_lm_loss(
            hidden_states, logits_to_keep
        )
        if (labels is None or compute_lm_loss) and not use_fused_lm_loss:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        pred_masks = None
        pred_mask_valid = None
        mask_targets_for_loss = None
        mask_valid_for_loss = None
        real_mask_loss_contribution = False
        # 并非每次计算都有区域任务：生成文字的过程中可能只需要下一个词的分数。
        # 只有给出区域位置、图片网格与有效图片特征时，才进入真实区域解码路径。
        has_mask_work = bool(
            mask_positions is not None
            and mask_positions.numel() > 0
            and image_grid_thw is not None
            and image_grid_thw.numel() > 0
            and visual_outputs is not None
        )
        if has_mask_work:
            images_per_sample = None
            vision_start_id = getattr(self.config, "vision_start_token_id", None)
            if input_ids is not None and vision_start_id is not None:
                is_vision_start = (input_ids == vision_start_id)
                images_per_sample = is_vision_start.sum(dim=1)
                if mask_image_index is None:
                    # Each `<|mask|>` binds to the nearest image preceding it.
                    seq_len = input_ids.shape[1]
                    pos = mask_positions
                    if pos.dim() == 1:
                        pos = pos.unsqueeze(1)
                    pos = pos.to(input_ids.device).clamp(0, seq_len - 1)
                    # 多图归属示例：某区域标记前出现两次图片开始符号，则它归属编号 1 的图片。
                    # 编号从 0 开始，所以累计数量需要减 1；调用者也可以显式传入 mask_image_index。
                    seen = is_vision_start.long().cumsum(dim=1)
                    mask_image_index = (seen.gather(1, pos) - 1).clamp_min(0)
            pred_masks, pred_mask_valid = self._decode_pixel_masks(
                hidden_states=hidden_states,
                mask_positions=mask_positions,
                image_grid_thw=image_grid_thw,
                visual_outputs=visual_outputs,
                mask_valid_mask=mask_valid_mask,
                images_per_sample=images_per_sample,
                mask_image_index=mask_image_index,
                mask_canvas_grid=mask_canvas_grid,
            )
            if mask_targets is not None:
                if pred_masks.shape[-2:] != mask_targets.shape[-2:]:
                    raise ValueError(
                        f"pred_masks {tuple(pred_masks.shape[-2:])} and mask_targets "
                        f"{tuple(mask_targets.shape[-2:])} disagree on canvas size; pass "
                        "`mask_canvas_grid` matching the grid the targets were sized at."
                    )
                mask_targets_for_loss = mask_targets
                mask_valid_for_loss = (
                    mask_pixel_valid if mask_pixel_valid is not None else pred_mask_valid
                )
                real_mask_loss_contribution = True
        elif is_training_forward:
            # Rank-collective safety: every rank must call `mask_decoder` and
            # `mask_loss_fn` once per forward, or DeepSpeed's gradient
            # reduction desyncs. Ranks without masks run a zero-weighted
            # surrogate through the same modules.
            llm_hidden = hidden_states.shape[-1]
            vision_hidden = int(self._get_vision_config(self.config).hidden_size)
            # 这是多进程同步所需的占位计算，不是有效的预测结果。
            # 对应有效像素和权重为零，让该进程保持计算连接，但不从虚构目标中学习。
            dummy_mask = hidden_states.new_zeros((1, 1, llm_hidden))
            dummy_patch = hidden_states.new_zeros((1, 1, vision_hidden))
            pred_masks = self.mask_decoder(dummy_mask, dummy_patch, grid_h=1, grid_w=1)
            pred_mask_valid = torch.zeros_like(pred_masks)
            mask_targets_for_loss = torch.zeros_like(pred_masks)
            mask_valid_for_loss = pred_mask_valid

        if compute_lm_loss:
            # When truncation drops every supervised token, the mean CE is
            # 0/0 = NaN. Emit a graph-connected zero instead so `lm_head` stays
            # in the backward graph on every rank.
            # -100 表示“不考这个位置”。如果截断后所有标签都被忽略，普通平均会变成 0/0。
            # 这里返回仍连接计算图的零，既避免数值异常，也保证训练进程参与同样的反向过程。
            if (labels[..., 1:] != -100).any():
                if use_fused_lm_loss:
                    lm_loss_value = self._fused_linear_lm_loss(
                        hidden_states, labels, num_items_in_batch
                    )
                else:
                    lm_loss_value = self._causal_lm_loss(
                        logits, labels, num_items_in_batch
                    )
            elif use_fused_lm_loss:
                lm_loss_value = (hidden_states.sum() + self.lm_head.weight.sum()) * 0.0
            else:
                lm_loss_value = logits.sum() * 0.0
            lm_loss = lm_loss_weight * lm_loss_value
            loss = lm_loss

        if mask_targets_for_loss is not None and mask_loss_weight >= 0:
            # (B, K, H, W) → (B*K, 1, H, W): each (sample, slot) is an
            # independent mask. Padded slots have an all-zero valid mask and
            # contribute nothing.
            B_, K_, H_, W_ = pred_masks.shape
            # 将样本与区域两个轴合起来：B×K×H×W → (B·K)×1×H×W。
            # 每张目标区域各算一份差距，补齐区域由有效像素标记排除，再与文字损失相加。
            mask_loss_value = self.mask_loss_fn(
                pred_masks.reshape(B_ * K_, 1, H_, W_),
                mask_targets_for_loss.reshape(B_ * K_, 1, H_, W_),
                valid_mask=mask_valid_for_loss.reshape(B_ * K_, 1, H_, W_),
            )
            effective_weight = mask_loss_weight if real_mask_loss_contribution else 0.0
            contribution = effective_weight * mask_loss_value
            if num_items_in_batch is not None:
                # HF Trainer skips its own `/ gradient_accumulation_steps` when
                # the model accepts loss kwargs. The LM term folds that into
                # its token-count scale; the per-microbatch mask mean must
                # divide explicitly.
                contribution = contribution / max(gradient_accumulation_steps or 1, 1)
            if not real_mask_loss_contribution:
                # Keep the LM forward in the backward graph on mask-free ranks
                # (e.g. a text-only batch under mask-only training).
                contribution = contribution + hidden_states.sum() * 0.0
            loss = contribution if loss is None else loss + contribution
            mask_loss = contribution

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XiaomiLensOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(self.model, "rope_deltas", None),
            pred_masks=pred_masks,
            pred_mask_valid=pred_mask_valid,
            lm_loss=lm_loss,
            mask_loss=mask_loss,
        )
