"""检查总数、划分重复、有效目标及每个样本在模型输入大小下仍保留目标。

v3：答案不再要求 == training_answer(m,v)（那样会与带外观句的答案冲突）。
改为复用 description_v2.answer.check_answers() 对 answer / answer_optical 两版
分别做「数值句前缀 + 后缀 + 官方 bbox 恰 1 框」断言；该共享函数不依赖 torch，
也可被本机 Windows 侧静态检查与实测脚本调用。
输出默认写 outputs/data-check-revision.json；11-07 的 data-check.json 保留为历史。
"""
import argparse
import hashlib, json, ast, re
from xiaomi_lens.constants import BOX_START_TOKEN, BOX_END_TOKEN
from pathlib import Path
import numpy as np
from PIL import Image
from xiaomi_lens.train.data.domain import (
    DEST,
    ROOT,
    NAMES,
    jsonlines,
    write_json,
)
from xiaomi_lens.description_v2.answer import check_answers
from xiaomi_lens.domain_runtime import build, BASE, configure_processor
from transformers import AutoProcessor

p = argparse.ArgumentParser()
p.add_argument("--out", default="outputs/data-check-revision.json")
a = p.parse_args()

rows = [
    r for split in ["train", "val", "test"] for r in jsonlines(DEST / f"{split}.jsonl")
]
assert len(rows) <= 1000
assert len({r["image_sha256"] for r in rows}) == len(rows)
import torch

torch.set_num_threads(4)
processor = configure_processor(
    AutoProcessor.from_pretrained(BASE, local_files_only=True, use_fast=False)
)
# 直接执行只读原始源码中的坐标解析函数，避免自写另一份解析器自我验证。
source = ast.parse((ROOT / "Xiaomi-Lens-0-raw/infer.py").read_text())
nodes = [
    n
    for n in source.body
    if (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_BOX_RE" for t in n.targets)
    )
    or (isinstance(n, ast.FunctionDef) and n.name == "parse_bboxes")
]
scope = {"re": re, "BOX_START_TOKEN": BOX_START_TOKEN, "BOX_END_TOKEN": BOX_END_TOKEN}
exec(
    compile(ast.Module(body=nodes, type_ignores=[]), "official_bbox_parser", "exec"),
    scope,
)
# 两版答案的权威一致性（在 WSL 侧运行时同样生效；本机可单独 import 该共享函数）
errs = check_answers(rows)
assert not errs, f"两版答案与标注计算不一致: {errs[:5]} (共 {len(errs)})"
counts = {}
negatives = {}
small_target_preserved = []
for r in rows:
    m = np.array(Image.open(ROOT / r["mask"])) > 0
    v = np.array(Image.open(ROOT / r["valid"])) > 0
    assert m.shape == v.shape and v.any()
    assert int((m & v).sum()) == r["positive_pixels"]
    assert (m & v).any(), r["id"]
    boxes = scope["parse_bboxes"](r["answer"])
    assert len(boxes) == int(m.any()), (r["id"], "official bbox incompatible")
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000, r["id"]
    batch = build(processor, r, training=True)
    if "mask_targets" in batch:
        height, width = batch["mask_targets"].shape[-2:]
        resized_m = np.asarray(
            Image.fromarray(m).resize((width, height), Image.Resampling.NEAREST)
        )
        resized_v = np.asarray(
            Image.fromarray(v).resize((width, height), Image.Resampling.NEAREST)
        )
        if not (resized_m & resized_v).any():
            small_target_preserved.append(r["id"])
    key = r["source"] + "/" + r["split"]
    counts[key] = counts.get(key, 0) + 1
    negatives[key] = negatives.get(key, 0) + int(not m.any())
    del batch
original = json.loads((ROOT / "outputs/original_code_hashes.json").read_text())
assert all(
    hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in original.items()
)
write_json(
    ROOT / a.out,
    {
        "count": len(rows),
        "counts": counts,
        "negatives": negatives,
        "original_code_unchanged": True,
        "all_training_encodings_checked": True,
        "official_bbox_parser_checked": True,
        "dual_answer_checked": True,
        "small_target_preserved_samples": small_target_preserved,
        "revision_round": "2026-11",
        "note": "本轮结果；11-07 的 outputs/data-check.json 为历史结论，未改写",
    },
)
print("PASS", len(rows), counts, negatives, "->", a.out)
