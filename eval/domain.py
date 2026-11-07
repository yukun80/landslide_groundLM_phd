"""领域评测入口；测试集须在两个正式训练版本完成并锁定后使用。"""

import argparse, json, hashlib
from pathlib import Path
from xiaomi_lens.domain_runtime import load, seed_all
from xiaomi_lens.domain_evaluation import evaluate
from xiaomi_lens.train.data.domain import DEST, ROOT, jsonlines

p = argparse.ArgumentParser()
p.add_argument("--checkpoint")
p.add_argument("--split", choices=["val", "test"], default="val")
p.add_argument("--rgb-only", action="store_true")
p.add_argument("--output", required=True)
a = p.parse_args()
if a.split == "test" and not (ROOT / "outputs/final-models-locked.json").exists():
    raise RuntimeError("先完成对照训练并锁定最终模型，不能提前查看最终评估")
if a.split == "test":
    lock = json.loads((ROOT / "outputs/final-models-locked.json").read_text())
    if (
        hashlib.sha256((DEST / "test.jsonl").read_bytes()).hexdigest()
        != lock["test_sha256"]
    ):
        raise RuntimeError("test split changed")
    if a.checkpoint:
        h = hashlib.sha256()
        with open(a.checkpoint, "rb") as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(b)
        expected = lock["models"]["rgb" if a.rgb_only else "multiview"]["sha256"]
        if h.hexdigest() != expected:
            raise RuntimeError("模型与锁定记录不一致")
seed_all()
m, processor = load(a.checkpoint)
evaluate(m, processor, jsonlines(DEST / f"{a.split}.jsonl"), a.output, a.rgb_only)
