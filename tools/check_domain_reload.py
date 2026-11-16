"""加载短训练保存文件，与保存前的自由预测逐像素及逐字对照。"""

import argparse, json
import numpy as np
import torch
from PIL import Image
from xiaomi_lens.domain_runtime import load, predict, seed_all, build
from xiaomi_lens.train.data.domain import ROOT, DEST, jsonlines, write_json, NAMES

p = argparse.ArgumentParser()
p.add_argument("--training-dir", default="outputs/smoke")
p.add_argument("--output", default="outputs/smoke-reload")
p.add_argument("--report", default="outputs/reload-check.json")
a = p.parse_args()
training = ROOT / a.training_dir
seed_all()
model, processor = load(training / "last.pt")
row = next(r for r in jsonlines(DEST / "val.jsonl") if r["source"] == NAMES[2])
inputs = build(processor, row, training=False)
changed_row = dict(row, answer="这是不应该进入预测的标准答案", mask="不存在的标注.png", valid="不存在的有效区.png")
check_inputs = build(processor, changed_row, training=False)
assert inputs.keys() == check_inputs.keys()
assert "labels" not in inputs and "mask_targets" not in inputs
assert all(torch.equal(value, check_inputs[key]) if isinstance(value, torch.Tensor)
           else value == check_inputs[key] for key, value in inputs.items())
del inputs, check_inputs
state = torch.load(training / "last.pt", map_location="cpu", weights_only=False)
assert state["epoch"] == 1 and state["step"] == 4
assert state["extra"]["stopping"]["joint_epochs"] == 1
assert state["optimizer"]["state"]
del state
mask, result = predict(model, processor, row, ROOT / a.output / row["id"])
old = training / "val-epoch-1" / row["id"]
previous = json.loads((old / "prediction.json").read_text())
truth = np.asarray(Image.open(old / "mask.png")) > 0
assert np.array_equal(mask, truth)
assert result["raw_answer"] == previous["raw_answer"]
changed = json.loads((training / "epoch-1-update.json").read_text())[
    "changed_parameters"
]
assert any("lora_" in n for n in changed)
assert any("mask_decoder." in n for n in changed)
write_json(
    ROOT / a.report,
    {
        "same_answer": True,
        "free_prediction_excludes_groundtruth": True,
        "resume_steps_and_stopping_state_checked": True,
        "same_mask": True,
        "language_and_mask_parameters_updated": True,
        "sample": row["id"],
    },
)
print("PASS saved/reloaded predictions and both parameter groups updated")
