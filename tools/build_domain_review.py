"""为固定30个val样本导出原图、标注、预测和文字，便于逐项审阅。

v3：rgb（--rgb-only）模型的参考答案取 answer_optical，其余取 answer；
缺光学版显式报错，不静默回退到多图版。纯函数 reference_for 可单测。
"""

from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from xiaomi_lens.train.data.domain import ROOT, DEST, NAMES, jsonlines, write_json


def reference_for(name, row):
    """按模型给出应展示的参考答案：rgb 用光学版，baseline/multiview 用多图版。"""
    if name == "rgb":
        if "answer_optical" not in row or not row["answer_optical"]:
            raise ValueError("rgb 模型缺光学版答案（answer_optical），禁止回退到多图版")
        return row["answer_optical"]
    return row["answer"]


rows = jsonlines(DEST / "val.jsonl")
chosen = [r for name in NAMES for r in [x for x in rows if x["source"] == name][:10]]
review = ROOT / "outputs/review/validation"
review.mkdir(parents=True, exist_ok=True)
texts = []
for name in ["baseline", "rgb", "multiview"]:
    if name == "baseline":
        folder = ROOT / "outputs/baseline-val"
    else:
        state = torch.load(
            ROOT / "outputs" / name / "best.pt", map_location="cpu", weights_only=False
        )
        epoch = state["epoch"]
        del state
        folder = ROOT / "outputs" / name / f"val-epoch-{epoch}"
    for page in range(3):
        canvas = Image.new("RGB", (768, 2800), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(chosen[page * 10 : (page + 1) * 10]):
            image = Image.open(ROOT / row["images"][-1]).convert("RGB")
            truth = np.asarray(Image.open(ROOT / row["mask"])) > 0
            gt = np.array(image).copy()
            gt[truth] = (gt[truth] * 0.5 + np.array([30, 255, 60]) * 0.5).astype(
                "uint8"
            )
            prediction = Image.open(folder / row["id"] / "overlay.png").convert("RGB")
            for column, picture in enumerate([image, Image.fromarray(gt), prediction]):
                canvas.paste(picture.resize((256, 256)), (256 * column, 280 * j + 24))
            draw.text(
                (2, j * 280 + 3),
                row["id"] + " | IMAGE / LABEL / PREDICTION",
                fill="black",
            )
            answer = json.loads((folder / row["id"] / "prediction.json").read_text())
            texts.append(
                {
                    "model": name,
                    "id": row["id"],
                    "reference": reference_for(name, row),
                    "answer": answer["answer"],
                    "derived": answer["derived_description"],
                }
            )
        canvas.save(review / f"{name}-{page}.jpg")
write_json(review / "answers.json", texts)
write_json(review / "sample_ids.json", [r["id"] for r in chosen])
print("Prepared fixed validation review:", len(chosen), "samples x 3 models")
