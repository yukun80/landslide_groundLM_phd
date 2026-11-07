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
# 本文件：把外部区域请求转换成模型输入，再把模型输出转换为可传输的图片或区域编码。
# 怎么读：pre_process 负责进入，post_process 负责返回；原图大小与模型画布大小是两个概念。
# 完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。

"""vLLM IO-processor plugin — turns a mask request into a pooling forward and the
pooler's flat output back into a dense mask.

Registered via `pyproject.toml` under `[project.entry-points."vllm.io_processor_plugins"]`.
The pooling model (`vllm_model.XiaomiLensForVLLM.pooler`) returns a flat tensor
`[grid_h, grid_w, K, *mask_logits]`; `post_process` reshapes it to (K, grid_h*16,
grid_w*16) and serializes to base64 PNG(s) + optional overlay, reusing the shared
visualization in `infer`.

Example request (passed as the plugin's `data` object):
    {
      "image_path": "/path/to/image.jpg",
      "text": "Locate all instances in the image that match the description \\"the cat on the left\\". Report bbox coordinates and masks.",
      "completion": "</think><bbox>[100, 200, 400, 800]</bbox><mask><|mask|></mask>",
      "threshold": 0.5, "return_overlay": true,
      "output_format": "png",
      "mask_space": "canvas"
    }

Send the actual generated completion and the same instruction used for text
generation. Include `completion_token_ids` from the text server when available
to replay exact IDs; otherwise the completion is tokenized again. Use `image_b64`
instead of `image_path` to send base64-encoded PNG/JPEG bytes.

`output_format` accepts `png`, `rle`, or `both`. `mask_space` accepts `canvas`
or `original`; the overlay always uses the vision encoder's input canvas.
Reasoning masks before `</think>` are excluded from the response.

Response: {"num_masks": K, "canvas": [w, h], "mask_size": [w, h],
           "mask_space": "canvas", "bboxes": [...], "masks_b64": [...],
           "masks_rle": [...], "overlay_b64": ...}
"""

from __future__ import annotations

import base64
import io as _io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vllm.plugins.io_processors.interface import IOProcessor


# 【步骤说明】优先读取请求中的本地图片路径，否则解码 Base64 图片；返回 RGB 图片，两者都没有就报错。
# 对照参数：payload；下面保留原始英文说明与类型，方便对应调用处。
def _load_image(payload: dict) -> Image.Image:
    if payload.get("image_path"):
        return Image.open(payload["image_path"]).convert("RGB")
    if payload.get("image_b64"):
        return Image.open(_io.BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB")
    raise ValueError("request must provide 'image_path' or 'image_b64'")


# 【步骤说明】将像素数组按指定颜色模式存成内存 PNG，再转成文本；返回用于网络传输的 Base64 字符串。
# 对照参数：arr, mode；下面保留原始英文说明与类型，方便对应调用处。
def _png_b64(arr: np.ndarray, mode: str) -> str:
    buf = _io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# 【步骤说明】把二值区域按列优先压缩成 COCO RLE，并将字节串改成可写入 JSON 的文字；返回 size/counts 字典。
# 对照参数：binary_mask；下面保留原始英文说明与类型，方便对应调用处。
def _encode_rle(binary_mask: np.ndarray) -> dict:
    """Encode as JSON-serializable COCO compressed RLE."""
    from pycocotools import mask as coco_mask

    encoded = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": counts,
    }


# 【类的用途】在外部请求和 vLLM 的区域模型之间转换输入输出，暂存每个请求的展示选项。
class XiaomiLensMaskIOProcessor(IOProcessor):
    """Bridges a mask request <-> vLLM pooling forward + dense-mask serialization."""

    # 【步骤说明】保存服务提供的分词器，并创建按请求号记录输出选项的字典。 构造完成后通过对象保存这些状态，不返回预测结果。
    # 对照参数：vllm_config, renderer；下面保留原始英文说明与类型，方便对应调用处。
    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        self._tokenizer = renderer.get_tokenizer()
        # One IO processor serves concurrent requests, so it cannot use one self._payload.
        self._payloads: dict[str, dict] = {}

    # 【步骤说明】检查请求内容必须是字典，合法就返回原对象；不是字典则报错。
    # 对照参数：data；下面保留原始英文说明与类型，方便对应调用处。
    def parse_data(self, data: object) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Xiaomi-Lens-0 mask request must be a JSON object")
        return data

    # 【步骤说明】根据图片与问题构造前缀，再拼接实际生成的编号；返回服务模型输入，并按请求号暂存展示选项。
    # 对照参数：prompt, request_id；下面保留原始英文说明与类型，方便对应调用处。
    def pre_process(self, prompt, request_id=None, **kwargs):
        """Render the image and instruction, then append the generated token IDs.

        The checkpoint's chat template supplies the generation prefix. Completion
        IDs are appended separately to preserve the text generation boundary;
        vLLM receives the combined prompt IDs and the image for its forward pass.
        """
        payload = prompt if isinstance(prompt, dict) else self.parse_data(prompt)
        payload_key = request_id or "__default__"
        image = _load_image(payload)

        # Match HF build_inputs: the template supplies image tokens and the
        # assistant prefix, including any opening thought block.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": payload["text"]},
            ],
        }]
        prompt_prefix = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        completion = payload["completion"]
        prompt_text = prompt_prefix + completion

        # Keep the generation boundary explicit. Tokenizing the concatenated
        # string can merge whitespace tokens across ``prefix ++ completion``;
        # HF run_forward concatenates the prefix and generated token IDs.
        # Callers without completion IDs encode the completion separately.
        prefix_ids = self._tokenizer.encode(prompt_prefix, add_special_tokens=False)
        completion_ids = payload.get("completion_token_ids")
        if completion_ids is None:
            completion_ids = self._tokenizer.encode(
                completion, add_special_tokens=False
            )
        elif not isinstance(completion_ids, list) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in completion_ids
        ):
            raise ValueError("completion_token_ids must be a list of integer token IDs")
        # 保留生成边界：问题前缀与生成结果分别提供编号后拼接，
        # 避免把两个文本重新合并分词而改变区域标记附近的对应位置。
        prompt_token_ids = list(prefix_ids) + list(completion_ids)

        self._payloads[payload_key] = payload
        return {
            "prompt": prompt_text,
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"image": image},
        }

    # 【步骤说明】读取网格与区域数量，重建像素分数并移除思考区域；按请求缩放、二值化和编码，返回区域与可选叠加图。
    # 对照参数：model_output, request_id；下面保留原始英文说明与类型，方便对应调用处。
    def post_process(self, model_output, request_id=None, **kwargs):
        out = model_output[0]
        data = out.outputs.data if hasattr(out.outputs, "data") else out.outputs
        flat = torch.as_tensor(data).float().reshape(-1)
        # 扁平结果的前三个数字是网格高、网格宽与区域数，其余才是像素分数。
        # 这个头部约定要与 vllm_model.py 的 decode_mask_batch 保持一致。
        grid_h, grid_w, k = int(flat[0]), int(flat[1]), int(flat[2])
        canvas_h, canvas_w = grid_h * 16, grid_w * 16
        logits = flat[3:].reshape(k, canvas_h, canvas_w)

        payload_key = request_id or "__default__"
        payload = self._payloads.pop(payload_key, {})
        from infer import parse_answer_bboxes, split_answer_content

        completion = payload.get("completion", "")
        _, thought_mask_count = split_answer_content(completion)
        if thought_mask_count > k:
            raise ValueError(
                f"completion reports {thought_mask_count} thought masks but pooler returned K={k}"
            )
        logits = logits[thought_mask_count:]
        k = int(logits.shape[0])
        threshold = float(payload.get("threshold", 0.5))
        return_overlay = bool(payload.get("return_overlay", True))
        output_format = str(payload.get("output_format", "png")).lower()
        if output_format not in {"png", "rle", "both"}:
            raise ValueError("output_format must be one of: png, rle, both")
        mask_space = str(payload.get("mask_space", "canvas")).lower()
        if mask_space not in {"canvas", "original"}:
            raise ValueError("mask_space must be one of: canvas, original")

        image = None
        output_logits = logits
        if mask_space == "original":
            image = _load_image(payload)
            output_logits = F.interpolate(
                logits.unsqueeze(0),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0]
        mask_h, mask_w = int(output_logits.shape[-2]), int(output_logits.shape[-1])
        binary_masks = (
            torch.sigmoid(output_logits) >= threshold
        ).to(torch.uint8).cpu().numpy()

        masks_b64 = []
        if output_format in {"png", "both"}:
            masks_b64 = [_png_b64(binary * 255, mode="L") for binary in binary_masks]

        masks_rle = []
        if output_format in {"rle", "both"}:
            masks_rle = [_encode_rle(binary) for binary in binary_masks]

        bboxes = parse_answer_bboxes(completion)
        overlay_b64 = None
        if return_overlay and (payload.get("image_path") or payload.get("image_b64")):
            from infer import visualize_pixel_mask

            image = image or _load_image(payload)
            grid_thw = torch.tensor([[1, grid_h, grid_w]])
            overlay = visualize_pixel_mask(
                image=image,
                pred_masks=logits.unsqueeze(0),
                image_grid_thw=grid_thw,
                threshold=threshold,
                bboxes=bboxes or None,
                draw_bbox=bool(bboxes),
            )
            overlay_b64 = _png_b64(np.asarray(overlay), mode="RGB")

        return {
            "num_masks": k,
            "canvas": [canvas_w, canvas_h],
            "mask_size": [mask_w, mask_h],
            "mask_space": mask_space,
            "bboxes": bboxes,
            "masks_b64": masks_b64,
            "masks_rle": masks_rle,
            "overlay_b64": overlay_b64,
        }


# 【步骤说明】返回输入输出处理器类的完整名称字符串；框架会再实例化该类，这里不是直接返回对象。
def make_io_processor():
    """Entry-point hook. vLLM calls this with NO args and expects the qualified
    name of the IOProcessor class; it then resolves + instantiates it itself with
    `(vllm_config, renderer)` (see vllm.plugins.io_processors.get_io_processor)."""
    return "xiaomi_lens.integrations.vllm.io_processor.XiaomiLensMaskIOProcessor"
