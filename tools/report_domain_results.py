"""汇总三版本的最终结果与按场景重采样的波动范围，生成供审阅的图片页。"""

import json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from xiaomi_lens.train.data.domain import ROOT, DEST, jsonlines, write_json

rows = jsonlines(DEST / "test.jsonl")
byid = {r["id"]: r for r in rows}
report = {}
for name in ["baseline", "rgb", "multiview"]:
    folder = ROOT / "outputs" / f"{name}-test"
    summary = json.loads((folder / "summary.json").read_text())
    scores = [json.loads((folder / r["id"] / "score.json").read_text()) for r in rows]
    intervals = {}
    rng = np.random.default_rng(20261107)
    for source in sorted({r["source"] for r in rows}):
        groups = {}
        for s in scores:
            if s["source"] == source:
                groups.setdefault(byid[s["id"]]["group"], []).append(s["iou"])
        keys = list(groups)
        means = []
        for _ in range(2000):
            sampled = rng.choice(keys, len(keys), replace=True)
            means.append(np.mean([v for k in sampled for v in groups[k]]))
        intervals[source] = {
            "interval_95": np.percentile(means, [2.5, 97.5]).tolist(),
            "groups": len(keys),
            "limitation": "分组来源不完整，区间不证明跨地区推广能力",
        }
    report[name] = {"summary": summary, "sampling_intervals": intervals}
    # 图片页同时显示光学原图、真实标注和自由预测，不将真值传回模型。
    for start in range(0, len(rows), 10):
        canvas = Image.new("RGB", (768, 10 * 280), "white")
        draw = ImageDraw.Draw(canvas)
        for j, r in enumerate(rows[start : start + 10]):
            im = Image.open(ROOT / r["images"][-1]).convert("RGB")
            gt = np.asarray(Image.open(ROOT / r["mask"])) > 0
            overlay = np.asarray(im).copy()
            overlay[gt] = (overlay[gt] * 0.5 + np.array([30, 255, 60]) * 0.5).astype(
                "uint8"
            )
            for col, picture in enumerate(
                [
                    im,
                    Image.fromarray(overlay),
                    Image.open(folder / r["id"] / "overlay.png").convert("RGB"),
                ]
            ):
                canvas.paste(picture.resize((256, 256)), (col * 256, j * 280 + 24))
            draw.text(
                (2, j * 280 + 3), r["id"] + " | RGB / LABEL / PREDICTION", fill="black"
            )
        target = ROOT / "outputs/review" / name
        target.mkdir(parents=True, exist_ok=True)
        canvas.save(target / f"page-{start//10:02}.jpg")
write_json(ROOT / "outputs/final-comparison.json", report)
print(
    json.dumps(
        {k: v["summary"]["selection_score"] for k, v in report.items()}, indent=2
    )
)
