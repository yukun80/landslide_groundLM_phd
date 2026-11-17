"""训练和预测共享的图文输入、少量参数调整、区域输出与保存逻辑。"""

import os

os.environ.setdefault("XIAOMI_LENS_DECODER_ATTN", "sdpa")
from pathlib import Path
import json, time, random, re
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
# 图片/训练格式检查不需要导入模型及其加速组件，避免它们初始化显卡。
# 官方区域分支恢复倍数为16；真正加载模型时再次核对，防止两处发生偏差。
MASK_CANVAS_SCALE = 16
from .train.data.domain import ROOT, describe, write_json
from .description_checks import consistency_issues, answer_presence

BASE = ROOT / "checkpoints/Xiaomi-Lens-0"
CONFIG = json.loads((Path(__file__).parent / "train/configs/domain.json").read_text())
PROMPT = "请识别最后一张光学遥感图片中的滑坡区域，用中文简短说明位置、面积比例以及看得清的颜色、形状或纹理，并输出区域标记。前面的图片如有则是同一区域的辅助观测，仅补充能可靠对应目标或邻近区域的局部差异；看不清时不要推测。没有滑坡时请明确回答未检测到滑坡区域。不要推测灾害成因、风险、活动性或位移速度，不把亮度直接解释为形变大小。"


def seed_all(seed=CONFIG["seed"]):
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_adapters(model):
    """仅语言层的线性变换增加可学习参数，视觉主干保持冻结。"""
    from peft import LoraConfig, inject_adapter_in_model

    names = [
        n
        for n, m in model.named_modules()
        if n.startswith("model.language_model.layers.")
        and isinstance(m, torch.nn.Linear)
    ]
    model = inject_adapter_in_model(
        LoraConfig(
            r=CONFIG["adapter_rank"],
            lora_alpha=CONFIG["adapter_alpha"],
            lora_dropout=0.0,
            target_modules=names,
        ),
        model,
    )
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n or n.startswith("mask_decoder."))
        if p.requires_grad:
            p.data = p.data.float()
    return model


def configure_processor(processor):
    """训练、预测和检查共用同一图片尺寸设置。"""
    minimum = CONFIG["auxiliary_edge"] ** 2
    maximum = CONFIG["high_resolution_edge"] ** 2
    processor.image_processor.size = {"shortest_edge": minimum, "longest_edge": maximum}
    processor.image_processor.min_pixels = minimum
    processor.image_processor.max_pixels = maximum
    return processor


def load(checkpoint=None, training=False):
    from .model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
    from .model.modeling_xiaomi_lens import MASK_CANVAS_SCALE as official_scale
    assert MASK_CANVAS_SCALE == official_scale
    processor = configure_processor(
        AutoProcessor.from_pretrained(BASE, local_files_only=True, use_fast=False)
    )
    model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to("cuda")
    model.requires_grad_(False)
    if training or checkpoint:
        model = add_adapters(model)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["trainable"], strict=False)
        model._domain_rgb_only = bool(state["rgb_only"])
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(state["trainable"]) != expected or unexpected:
            raise ValueError("checkpoint parameter coverage mismatch")
    if training:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
    model.eval()
    return model, processor


def images_for(row, rgb_only=False):
    paths = row["images"][-1:] if rgb_only else row["images"]
    images = []
    native_sizes = []
    for i, p in enumerate(paths):
        with Image.open(ROOT / p) as opened:
            im = opened.convert("RGB")
        native_sizes.append(im.size)
        bound = (
            CONFIG["high_resolution_edge"]
            if i == len(paths) - 1 and max(im.size) > CONFIG["auxiliary_edge"]
            else CONFIG["auxiliary_edge"]
        )
        ratio = bound / max(im.size)
        size = tuple(max(32, round(v * ratio / 32) * 32) for v in im.size)
        images.append(im.resize(size, Image.Resampling.BICUBIC))
    if len(set(native_sizes)) != 1:
        raise ValueError("辅助图与光学图尺寸不一致，请先对齐")
    return images


def build(processor, row, rgb_only=False, training=False, prompt=None, device="cuda"):
    images = images_for(row, rgb_only)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": PROMPT if prompt is None else prompt}],
        }
    ]
    prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text = prefix + (row["answer"] + "<|im_end|>" if training else "")
    batch = processor(
        text=[text], images=images, return_tensors="pt", add_special_tokens=False
    )
    if batch["input_ids"].shape[1] > CONFIG["maximum_sequence_length"]:
        raise ValueError("样本过长，不允许静默截断")
    if training:
        prefix_ids = processor(
            text=[prefix], images=images, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        n = prefix_ids.shape[1]
        if not torch.equal(batch["input_ids"][:, :n], prefix_ids):
            raise ValueError("assistant prefix mismatch")
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        batch["labels"] = labels
        token = processor.tokenizer.convert_tokens_to_ids("<|mask|>")
        positions = (batch["input_ids"][0] == token).nonzero().flatten()
        if len(positions):
            gh, gw = map(int, batch["image_grid_thw"][-1, 1:])
            h, w = gh * MASK_CANVAS_SCALE, gw * MASK_CANVAS_SCALE
            m = (
                np.array(
                    Image.open(ROOT / row["mask"]).resize(
                        (w, h), Image.Resampling.NEAREST
                    )
                )
                > 0
            )
            v = (
                np.array(
                    Image.open(ROOT / row["valid"]).resize(
                        (w, h), Image.Resampling.NEAREST
                    )
                )
                > 0
            )
            if not (m & v).any():
                # 极小目标可能被最近邻采样漏过。仅此时把原有效目标覆盖到对应输出格，
                # 不修改原标注；最终评分仍在原尺寸进行。
                original_mask = np.array(Image.open(ROOT / row["mask"])) > 0
                original_valid = np.array(Image.open(ROOT / row["valid"])) > 0
                source = torch.tensor(
                    (original_mask & original_valid)[None, None], dtype=torch.float32
                )
                m = (
                    torch.nn.functional.adaptive_max_pool2d(source, (h, w))[
                        0, 0
                    ].numpy()
                    > 0
                )
                v = v | m
                if not (m & v).any():
                    raise ValueError("没有有效目标，不能作为正样本")
            batch.update(
                mask_positions=positions[None],
                mask_image_index=torch.full((1, len(positions)), len(images) - 1),
                mask_valid_mask=torch.ones(1, len(positions)),
                mask_canvas_grid=(gh, gw),
                mask_targets=torch.tensor(m[None, None], dtype=torch.uint8),
                mask_pixel_valid=torch.tensor(v[None, None], dtype=torch.bool),
            )
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def save_checkpoint(path, model, optimizer, epoch, step, rgb_only, extra=None):
    state = {
        "trainable": {
            n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad
        },
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "step": step,
        "rgb_only": rgb_only,
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    torch.save(state, temp)
    temp.replace(path)


@torch.inference_mode()
def predict(model, processor, row, outdir, rgb_only=False, max_tokens=None, prompt=None):
    """只在实际预测时进入显卡环境，CPU图片检查导入本文件不会占用显卡。"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _predict_impl(model, processor, row, outdir, rgb_only, max_tokens, prompt)


def _predict_impl(model, processor, row, outdir, rgb_only=False, max_tokens=None, prompt=None):
    """自由生成后只解码回答里的区域标记；不借用标准答案或真实区域。"""
    max_tokens = CONFIG["maximum_answer_tokens"] if max_tokens is None else max_tokens
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    t = time.monotonic()
    inputs = build(processor, row, rgb_only, prompt=prompt)
    ids = model.generate(
        **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True
    )
    n = inputs["input_ids"].shape[1]
    completion = ids[:, n:]
    token = processor.tokenizer.convert_tokens_to_ids("<|mask|>")
    raw = processor.tokenizer.decode(completion[0], skip_special_tokens=False)
    text = raw.split("</think>")[-1]
    thought = raw.split("</think>")[0].count("<|mask|>") if "</think>" in raw else 0
    positions = (completion[0] == token).nonzero().flatten()[thought:] + n
    with Image.open(ROOT / row["images"][-1]) as im:
        original = im.convert("RGB")
    union = np.zeros((original.height, original.width), bool)
    if len(positions):
        replay = dict(inputs)
        replay["input_ids"] = ids
        replay["attention_mask"] = torch.ones_like(ids)
        if "mm_token_type_ids" in replay:
            replay["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(completion)], 1
            )
        gh, gw = map(int, inputs["image_grid_thw"][-1, 1:])
        replay.update(
            mask_positions=positions[None],
            mask_image_index=torch.full(
                (1, len(positions)), len(inputs["image_grid_thw"]) - 1, device="cuda"
            ),
            mask_canvas_grid=(gh, gw),
            logits_to_keep=1,
            use_cache=False,
        )
        pred = model(**replay).pred_masks[0].sigmoid().amax(0)
        resized = torch.nn.functional.interpolate(
            pred[None, None].float(),
            size=union.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        union = resized.cpu().numpy() >= 0.5
    Image.fromarray(union.astype("uint8") * 255).save(outdir / "mask.png")
    overlay = np.array(original).copy()
    overlay[union] = (overlay[union] * 0.5 + np.array([255, 40, 30]) * 0.5).astype(
        "uint8"
    )
    Image.fromarray(overlay).save(outdir / "overlay.png")
    issues = consistency_issues(text, union)
    result = {
        "id": row.get("id", "input"),
        "answer": re.sub(
            r"<bbox>.*?</bbox>|<\|[^>]+\|>|</?think>|</?mask>", "", text, flags=re.S
        ).strip(),
        "raw_answer": raw,
        "mask_count": len(positions),
        "answer_target_present": answer_presence(text, len(positions)),
        "derived_description": describe(union),
        "seconds": time.monotonic() - t,
        "generated_tokens": len(completion[0]),
        "truncated": len(completion[0]) >= max_tokens,
        "prediction_positive": bool(union.any()),
        "text_mask_conflict": bool(issues),
        "consistency_issues": issues,
        "consistency_scope": "规则检查不覆盖所有地质语义，需要结合图片审阅",
    }
    write_json(outdir / "prediction.json", result)
    return union, result
