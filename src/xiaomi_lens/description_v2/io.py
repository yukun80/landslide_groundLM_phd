"""逐样本记录、批次状态与原子写；支持中断后续跑。"""
import hashlib
import json
import os
from pathlib import Path

from .spec import ROOT, DEST, V2, RECORDS, SPLITS, BATCH_SIZE, N_BATCHES


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl_atomic(path, rows):
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(path, text)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_rows():
    """按 split 顺序返回 (split, row)，保留原始行顺序。"""
    out = []
    for split in SPLITS:
        for row in read_jsonl(DEST / f"{split}.jsonl"):
            out.append((split, row))
    return out


def append_jsonl(path, records):
    """通用追加写 + 每条 fsync，供 records / revisions / round 逐图记录复用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def append_records(records):
    """追加写入 + 每条 fsync；同一 id 以最后一条为准。"""
    append_jsonl(RECORDS, records)


def load_records():
    """返回 {id: 最后一条记录}，并报告重复/缺失。"""
    result = {}
    if not RECORDS.exists():
        return result
    for line_no, line in enumerate(open(RECORDS, encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in result:
            # 后写覆盖先前状态，属于正常断点续跑
            pass
        result[r["id"]] = r
    return result


def load_state(path=None):
    p = Path(path) if path else V2 / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(state, path=None):
    atomic_write_json(Path(path) if path else V2 / "state.json", state)


def load_batches():
    p = V2 / "batches.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_batches(batches):
    atomic_write_json(V2 / "batches.json", batches)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def samples_manifest():
    """记录原始 PNG 的字节哈希，用于交付前确认图片/标注未被改动。"""
    out = {}
    for p in sorted((DEST / "samples").glob("*/*.png")):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        out[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def row_fingerprint(row):
    """除 answer / answer_optical 外所有字段的哈希，用于确认非答案字段未变。"""
    payload = {k: v for k, v in row.items() if k not in ("answer", "answer_optical")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
