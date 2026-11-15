"""检查 docs/description-v2 短文的行数上限与本地链接可达性。"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "description-v2"


def main():
    fail = []
    if not DOCS.exists():
        print("SKIP: docs/description-v2 尚未建立")
        return 0
    files = sorted(DOCS.glob("*.md"))
    for p in files:
        text = p.read_text(encoding="utf-8")
        limit = 60 if p.name == "README.md" else 120
        if len(text.splitlines()) > limit:
            fail.append(f"{p.name} 超过 {limit} 行")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in link or link.startswith("#"):
                continue
            target = (p.parent / link.split("#")[0]).resolve()
            if not target.exists():
                fail.append(f"{p.name} 链接不可达: {link}")
        if p.name != "README.md" and "](" not in text:
            fail.append(f"{p.name} 缺少返回上级链接")
    # 根入口
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "docs/description-v2/README.md" not in readme:
        fail.append("根 README 缺少描述改进 v2 入口")
    docs_readme = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    if "description-v2/README.md" not in docs_readme:
        fail.append("docs/README.md 缺少描述改进 v2 入口")
    if fail:
        print("FAIL check_desc_v2_docs")
        for f in fail:
            print("  ", f)
        return 1
    print(f"PASS check_desc_v2_docs: {len(files)} 篇短文，行数与链接通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
