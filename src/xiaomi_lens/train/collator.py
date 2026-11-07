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

# 【中文导读｜阅读第 4 步】
# 本文件：把多条长短不同的样本临时加工并拼成一批；失败样本会记录并跳过。
# 怎么读：先看 __call__；补齐文字长度不等于添加真实内容，labels=-100 的位置不计文字损失。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""Runtime collator for the lazy JSONL pipeline.

This mirrors ms-swift's large-dataset behavior: keep startup cheap and do
tokenization/image preprocessing inside DataLoader workers while training
is running. It reuses `encode_sample` so prompt rendering, truncation, and
mask target construction have one implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .data.preprocess import encode_sample

logger = logging.getLogger(__name__)


# 【类的用途】在取到一批样本后才处理图片与文字，并补齐长度；不在启动时一次处理完整数据集。
class LazyCollator:
    # 【步骤说明】保存处理器、区域符号编号和长度像素限制，初始化错误提示计数。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：processor, mask_token_id, model_max_length, max_pixels, min_pixels；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(
        self,
        processor,
        *,
        mask_token_id: int,
        model_max_length: int,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ) -> None:
        self.processor = processor
        self.mask_token_id = mask_token_id
        self.model_max_length = model_max_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._drop_warnings = 0
        self._dummy_warnings = 0

    # 【步骤说明】为整批失败的占位输入寻找可用文字编号，优先补齐符号，再尝试结束或起始符号，最后返回 0。
    def _fallback_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return int(value)
        return 0

    # 【步骤说明】从样本中提取来源名称、标签和图文数量，返回简短文字用于错误记录。
    # 对照参数：row；下面保留原始英文说明与类型，方便对应调用处。
    @staticmethod
    def _sample_tag(row: dict) -> str:
        return (
            f"dataset={row.get('_dataset_name', '?')!r} "
            f"tag={row.get('_dataset_tag', '')!r} "
            f"images={len(row.get('images') or [])} "
            f"messages={len(row.get('messages') or [])}"
        )

    # 【步骤说明】接收原批次大小，构造不含图片且所有标签均忽略的占位批次；返回字典，让多进程训练保持同步而不学习坏样本。
    # 对照参数：batch_size；下面保留原始英文说明与类型，方便对应调用处。
    def _dummy_batch(self, batch_size: int) -> dict:
        """Return a graph-safe zero-loss batch when every sampled row fails.

        The trainer/model handle K=0 and no images by running the LM path
        with all labels ignored plus the rank-symmetric dummy mask path.
        """
        n = max(1, int(batch_size))
        input_ids = torch.full((n, 1), self._fallback_token_id(), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((n, 1), dtype=torch.long),
            "labels": torch.full((n, 1), -100, dtype=torch.long),
            "mm_token_type_ids": torch.zeros((n, 1), dtype=torch.long),
            "_per_sample_mask_positions": [[] for _ in range(n)],
            "_per_sample_mask_valid_mask": [[] for _ in range(n)],
            "_per_sample_mask_pixel_masks": [[] for _ in range(n)],
            "_per_sample_mask_image_index": [[] for _ in range(n)],
            "_per_sample_mask_grid_hw": [[] for _ in range(n)],
        }

    # 【步骤说明】接收原始样本列表，逐条编码、跳过失败记录并补齐成功样本；返回批次字典，全失败时返回零损失占位批次。
    # 对照参数：features；下面保留原始英文说明与类型，方便对应调用处。
    def __call__(self, features: list[dict]) -> dict:
        encoded: list[dict] = []
        for row in features:
            tag = self._sample_tag(row)
            reported_drop = False
            try:
                out = encode_sample(
                    row,
                    processor=self.processor,
                    mask_token_id=self.mask_token_id,
                    model_max_length=self.model_max_length,
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                )
            except Exception as exc:
                out = None
                if self._drop_warnings < 10:
                    logger.warning(
                        "[lazy-collator] dropped sample during encode (%s): %s",
                        tag, exc,
                    )
                    self._drop_warnings += 1
                    reported_drop = True
            if out is None and not reported_drop and self._drop_warnings < 10:
                logger.warning("[lazy-collator] dropped sample during encode (%s)", tag)
                self._drop_warnings += 1
            if out is not None:
                encoded.append(out)

        if not encoded:
            if self._dummy_warnings < 10:
                logger.warning(
                    "[lazy-collator] every sample in a batch failed; "
                    "returning a zero-loss dummy batch of size %d.",
                    len(features),
                )
                self._dummy_warnings += 1
            return self._dummy_batch(len(features))

        # 同一批次需要规则的长方形数组，因此以最长样本为准在右侧补齐。
        # attention_mask 的 0 表示补齐位置；labels 的 -100 表示这里不计文字错误。
        max_len = max(len(r["input_ids"]) for r in encoded)

        # 【步骤说明】接收短序列、填充值和目标长度，在右侧补齐并返回新列表；文字补 0，学习标签补 -100。
        # 对照参数：seq, value, length；下面保留原始英文说明与类型，方便对应调用处。
        def _pad(seq, value, length):
            return list(seq) + [value] * (length - len(seq))

        input_ids = torch.tensor(
            [_pad(r["input_ids"], 0, max_len) for r in encoded],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [_pad(r["attention_mask"], 0, max_len) for r in encoded],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [_pad(r["labels"], -100, max_len) for r in encoded],
            dtype=torch.long,
        )
        mm_token_type_ids = torch.tensor(
            [_pad(r["mm_token_type_ids"], 0, max_len) for r in encoded],
            dtype=torch.long,
        )

        # 图片小块直接按样本顺序连接，并单独保留每张图片网格。
        # 文字轴的补齐与图片小块轴的连接不同，不能把两个维度混为一谈。
        pixel_chunks = [
            torch.as_tensor(r["pixel_values"])
            for r in encoded
            if "pixel_values" in r and np.asarray(r["pixel_values"]).size > 0
        ]
        if pixel_chunks:
            pixel_values = torch.cat(pixel_chunks, dim=0)
            image_grid_thw = torch.cat(
                [
                    torch.as_tensor(r["image_grid_thw"], dtype=torch.long)
                    for r in encoded
                    if len(r["image_grid_thw"]) > 0
                ],
                dim=0,
            )
        else:
            pixel_values = None
            image_grid_thw = None

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
            "_per_sample_mask_positions": [
                list(r["mask_positions"]) for r in encoded
            ],
            "_per_sample_mask_valid_mask": [
                list(r["mask_valid_mask"]) for r in encoded
            ],
            "_per_sample_mask_pixel_masks": [
                r["mask_pixel_masks"] for r in encoded
            ],
            # Per-MASK, not per-sample: each `<|mask|>` binds to its nearest
            # preceding image, so its target canvas is that image's grid.
            "_per_sample_mask_image_index": [
                list(r["mask_image_index"]) for r in encoded
            ],
            "_per_sample_mask_grid_hw": [
                list(r["mask_grid_hw"]) for r in encoded
            ],
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
        return batch
