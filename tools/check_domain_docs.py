"""检查领域短文长度、本地链接和原始文件未改动。"""

import hashlib, json, re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fail = []
for p in [*(root / "docs/domain-finetuning").glob("*.md"),
          *(root / "docs/local-inference").glob("*.md"), root / "README.md", root / "docs/README.md"]:
    text = p.read_text()
    limit = 60 if p.name == "README.md" else 120
    if len(text.splitlines()) > limit:
        fail.append(str(p) + " too long")
    for link in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in link and not (p.parent / link.split("#")[0]).exists():
            fail.append(str(p) + ": " + link)
original = json.loads((root / "outputs/original_code_hashes.json").read_text())
for p, h in original.items():
    if hashlib.sha256((root / p).read_bytes()).hexdigest() != h:
        fail.append("original changed " + p)
assert not fail, fail
print("PASS documentation and original-source preservation")
