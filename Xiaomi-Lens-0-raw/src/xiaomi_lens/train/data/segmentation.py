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
# 本文件：把压缩存储的区域标注还原成黑白像素图，并缩放到模型处理图片的大小。
# 怎么读：先看 decode_coco_rle，再看 segmentation_to_pixel_mask；标注缩放用最近邻，避免产生不存在的中间类别。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""COCO RLE helpers for segmentation targets.

Every ``<|mask|>`` in an assistant turn is supervised by one COCO RLE object
``{"size": [height, width], "counts": ...}``. Compressed ``counts`` strings are
decoded without requiring pycocotools.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image


# 【步骤说明】检查对象是否具有区域游程编码所需的 size 和 counts，返回真假，不执行像素解码。
# 对照参数：segmentation；下面保留原始英文说明与类型，方便对应调用处。
def is_coco_rle(segmentation) -> bool:
    """Return whether *segmentation* is a JSON-serializable COCO RLE object."""
    return (
        isinstance(segmentation, Mapping)
        and "size" in segmentation
        and "counts" in segmentation
    )


# 【步骤说明】把空值、单个区域或区域列表统一成按答案顺序排列的列表；不接受无法解释的标注类型。
# 对照参数：segmentation；下面保留原始英文说明与类型，方便对应调用处。
def split_segmentation_masks(segmentation) -> list:
    """Return one RLE per assistant ``<|mask|>``, in reading order.

    Accepts ``None`` (no masks), a single RLE, or a list of RLEs.
    """
    if segmentation is None:
        return []
    if is_coco_rle(segmentation):
        return [segmentation]
    if isinstance(segmentation, (list, tuple)) and all(
        is_coco_rle(value) for value in segmentation
    ):
        return list(segmentation)
    raise ValueError(
        "`segmentation` must be a COCO RLE object or a list of COCO RLE objects "
        "with `size` and `counts`."
    )


# 【步骤说明】读取 COCO 压缩字符串中的分段整数与差分，返回连续黑白段的长度列表，供解码器还原像素。
# 对照参数：encoded；下面保留原始英文说明与类型，方便对应调用处。
def _decode_compressed_rle_counts(encoded: str | bytes) -> list[int]:
    """Decode the compact COCO ``counts`` string without requiring pycocotools."""
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("COCO RLE counts bytes must be ASCII") from exc
    if not isinstance(encoded, str):
        raise TypeError("compressed COCO RLE counts must be str or bytes")

    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        while True:
            code = ord(encoded[position]) - 48
            position += 1
            if code < 0 or code > 0x3F:
                raise ValueError("invalid character in compressed COCO RLE counts")
            # 每个字符贡献 5 位整数信息，额外位表示是否继续读取和符号信息。
            # 这是 COCO 的存储格式，不是神经网络计算，初读可先把函数当作解压器。
            value |= (code & 0x1F) << shift
            shift += 5
            more = bool(code & 0x20)
            if not more:
                if code & 0x10:
                    value |= -1 << shift
                break
            if position >= len(encoded):
                raise ValueError("truncated compressed COCO RLE counts")

        # COCO stores a second-order delta from the fourth run onward.
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("compressed COCO RLE contains a negative run")
        counts.append(value)
    return counts


# 【步骤说明】把连续黑白段长度按列优先还原成高×宽的 0/1 数组；检查大小和计数，不要求安装 pycocotools。
# 对照参数：rle；下面保留原始英文说明与类型，方便对应调用处。
def decode_coco_rle(rle: Mapping) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE into an ``(H, W)`` uint8 mask."""
    if not is_coco_rle(rle):
        raise TypeError("COCO RLE must be an object containing 'size' and 'counts'")

    size = rle["size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("COCO RLE size must be [height, width]")
    try:
        height, width = (int(size[0]), int(size[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("COCO RLE size values must be integers") from exc
    if height <= 0 or width <= 0:
        raise ValueError("COCO RLE size values must be positive")

    raw_counts = rle["counts"]
    if isinstance(raw_counts, (str, bytes)):
        counts = _decode_compressed_rle_counts(raw_counts)
    elif isinstance(raw_counts, (list, tuple)):
        counts = []
        for run in raw_counts:
            if isinstance(run, (bool, np.bool_)) or not isinstance(run, (int, np.integer)):
                raise ValueError("uncompressed COCO RLE counts must be non-negative integers")
            run = int(run)
            if run < 0:
                raise ValueError("uncompressed COCO RLE counts must be non-negative integers")
            counts.append(run)
    else:
        raise TypeError("COCO RLE counts must be a compressed string or integer list")

    total = height * width
    if sum(counts) != total:
        raise ValueError(
            f"COCO RLE run total {sum(counts)} does not match size product {total}"
        )

    # 从全背景开始，只把第 1、3、5…段填为前景。
    # 计数序列从背景段开始，即使第一个像素就是前景，前面也要有一个长度为 0 的背景段。
    flat = np.zeros(total, dtype=np.uint8)
    offset = 0
    for index, run in enumerate(counts):
        end = offset + run
        if index % 2:
            flat[offset:end] = 1
        offset = end
    return flat.reshape((height, width), order="F")


# 【步骤说明】接收标注与目标画布大小，先解码，再按最近邻缩放；多个标注取并集，返回画布上的 0/1 区域。
# 对照参数：segmentation, image_size；下面保留原始英文说明与类型，方便对应调用处。
def segmentation_to_pixel_mask(segmentation, image_size: tuple[int, int]) -> np.ndarray:
    """Convert COCO RLE to a ``(height, width)`` uint8 mask at *image_size*.

    *image_size* is ``(width, height)`` of the processed-image canvas. RLE is
    decoded at its declared source ``size`` and nearest-neighbor resized. A
    list of RLE objects is decoded as their union.
    """
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    pixel_mask = np.zeros((height, width), dtype=np.uint8)
    for rle in split_segmentation_masks(segmentation):
        decoded = decode_coco_rle(rle)
        if decoded.shape != (height, width):
            decoded = np.asarray(
                Image.fromarray(decoded, mode="L").resize(
                    (width, height), resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
        pixel_mask |= decoded
    return pixel_mask
