"""加载短训练保存文件，与保存前的自由预测逐像素及逐字对照。"""

import json
import numpy as np
from PIL import Image
from xiaomi_lens.domain_runtime import load, predict, seed_all
from xiaomi_lens.train.data.domain import ROOT, DEST, jsonlines, write_json, NAMES

seed_all()
model, processor = load(ROOT / "outputs/smoke/last.pt")
row = next(r for r in jsonlines(DEST / "val.jsonl") if r["source"] == NAMES[2])
mask, result = predict(model, processor, row, ROOT / "outputs/smoke-reload" / row["id"])
old = ROOT / "outputs/smoke/val-epoch-1" / row["id"]
previous = json.loads((old / "prediction.json").read_text())
truth = np.asarray(Image.open(old / "mask.png")) > 0
assert np.array_equal(mask, truth)
assert result["raw_answer"] == previous["raw_answer"]
changed = json.loads((ROOT / "outputs/smoke/epoch-1-update.json").read_text())[
    "changed_parameters"
]
assert any("lora_" in n for n in changed)
assert any("mask_decoder." in n for n in changed)
write_json(
    ROOT / "outputs/reload-check.json",
    {
        "same_answer": True,
        "same_mask": True,
        "language_and_mask_parameters_updated": True,
        "sample": row["id"],
    },
)
print("PASS saved/reloaded predictions and both parameter groups updated")
