"""只在固定val样本检查辅助图缺失/错配；不读测试集、不用于新的参数搜索。"""

from copy import deepcopy
from xiaomi_lens.domain_runtime import load, seed_all
from xiaomi_lens.domain_evaluation import evaluate
from xiaomi_lens.train.data.domain import ROOT, DEST, NAMES, jsonlines, write_json

seed_all()
rows = jsonlines(DEST / "val.jsonl")
chosen = [r for n in NAMES[1:] for r in [x for x in rows if x["source"] == n][:10]]
model, processor = load(ROOT / "outputs/multiview/best.pt")
evaluate(model, processor, chosen, ROOT / "outputs/aux-check/correct")
evaluate(model, processor, chosen, ROOT / "outputs/aux-check/missing", True)
wrong = deepcopy(chosen)
for n in NAMES[1:]:
    group = [r for r in wrong if r["source"] == n]
    originals = [list(r["images"]) for r in group]
    for i, r in enumerate(group):
        r["images"] = originals[(i + 1) % len(group)][:-1] + originals[i][-1:]
evaluate(model, processor, wrong, ROOT / "outputs/aux-check/wrong")
write_json(
    ROOT / "outputs/aux-check/note.json",
    {"scope": "固定val样本诊断，不是最终测试或物理意义证明", "samples": len(chosen)},
)
