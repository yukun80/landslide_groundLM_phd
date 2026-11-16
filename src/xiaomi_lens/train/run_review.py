"""每轮固定验证样本的图文对照。仅在预测结束后读取标注，不反馈给模型。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from .data.domain import ROOT, NAMES, write_json


def prepare_review(rows, folder):
    folder = Path(folder)
    chosen = [r for source in NAMES for r in [x for x in rows if x["source"] == source][:10]]
    target = folder / "review"
    target.mkdir(exist_ok=True)
    answers = []
    for start in range(0, len(chosen), 5):
        canvas = Image.new("RGB", (768, 280 * min(5, len(chosen) - start)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(chosen[start:start + 5]):
            with Image.open(ROOT / row["images"][-1]) as im:
                optical = im.convert("RGB")
            truth = np.array(Image.open(ROOT / row["mask"])) > 0
            label = np.array(optical).copy()
            label[truth] = (label[truth] * .5 + np.array([30, 255, 60]) * .5).astype("uint8")
            with Image.open(folder / row["id"] / "overlay.png") as im:
                predicted = im.convert("RGB")
            for col, im in enumerate([optical, Image.fromarray(label), predicted]):
                canvas.paste(ImageOps.pad(im, (256, 256)), (col * 256, j * 280 + 24))
            draw.text((2, j * 280 + 4), row["id"] + " | IMAGE / LABEL / PREDICTION", fill="black")
            result_path = folder / row["id"] / "score.json"
            if not result_path.exists():
                result_path = folder / row["id"] / "result.json"
            result = json.loads(result_path.read_text())
            answers.append({"id": row["id"], "source": row["source"],
                            "answer": result["answer"], "derived": result["derived_description"],
                            "iou": result["iou"], "issues": result["consistency_issues"],
                            "reference": row["answer"], "auxiliary_images": row["images"][:-1]})
        canvas.save(target / f"page-{start // 5}.jpg", quality=95)
    write_json(target / "answers.json", answers)
    write_json(target / "status.json", {"prepared": len(chosen), "visually_reviewed": False,
                                       "note": "生成对照页不代表已看图，更不代表人工确认"})
