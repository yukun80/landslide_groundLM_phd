"""仅在两个正式训练版本完成后锁定最终模型，之后不得用测试集调参。"""

import hashlib, json
from pathlib import Path
from xiaomi_lens.train.data.domain import ROOT, DEST, write_json
from prepare_domain_model import digest

models = {}
for name in ["rgb", "multiview"]:
    folder = ROOT / "outputs" / name
    if not (folder / "completed.json").exists():
        raise RuntimeError(f"{name} training unfinished")
    models[name] = {
        "path": str((folder / "best.pt").relative_to(ROOT)),
        "sha256": digest(folder / "best.pt"),
    }
result = {
    "models": models,
    "test_sha256": digest(DEST / "test.jsonl"),
    "rule": "Final evaluation only; no further tuning using test results",
}
target = ROOT / "outputs/final-models-locked.json"
if target.exists() and json.loads(target.read_text()) != result:
    raise RuntimeError("已有不同锁定记录，禁止覆盖")
write_json(target, result)
print("Locked models and final evaluation samples")
