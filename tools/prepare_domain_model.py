"""检查并完整复制已有模型；仅在本地模型缺失时下载固定版本。"""

import hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/yukun80/codes/Xiaomi-Lens-0/checkpoints/Xiaomi-Lens-0")
TARGET = ROOT / "checkpoints/Xiaomi-Lens-0"
REVISION = "68f57cdde1a19f751e78b18af04efa8ca8bd7fce"


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    source = SOURCE
    required = [
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    ]
    local_ok = all((source / n).is_file() for n in required)
    if local_ok:
        try:
            required += sorted(
                set(
                    json.loads((source / "model.safetensors.index.json").read_text())[
                        "weight_map"
                    ].values()
                )
            )
            local_ok = all((source / n).is_file() for n in required)
            if local_ok and (source / "LOCAL_PROVENANCE.json").exists():
                local_ok = all(
                    (source / r["path"]).is_file()
                    and digest(source / r["path"]) == r["sha256"]
                    for r in json.loads((source / "LOCAL_PROVENANCE.json").read_text())[
                        "files"
                    ]
                )
        except (ValueError, KeyError, OSError):
            local_ok = False
    if not local_ok:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download("SeerRay-Lab/Xiaomi-Lens-0", revision=REVISION))
    files = set(
        json.loads((source / "model.safetensors.index.json").read_text())[
            "weight_map"
        ].values()
    )
    files.update(
        [
            "model.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "LICENSE",
        ]
    )
    if (source / "LOCAL_PROVENANCE.json").exists():
        files.add("LOCAL_PROVENANCE.json")
    TARGET.mkdir(parents=True, exist_ok=True)
    results = []
    for name in sorted(files):
        s = source / name
        t = TARGET / name
        expected = digest(s)
        if t.exists():
            if digest(t) != expected:
                raise RuntimeError(f"目标已存在但内容不同，未覆盖: {t}")
        else:
            temp = t.with_suffix(t.suffix + ".partial")
            shutil.copy2(s, temp)
            if digest(temp) != expected:
                raise RuntimeError(f"复制校验失败: {name}")
            temp.rename(t)
        results.append({"file": name, "sha256": expected, "bytes": s.stat().st_size})
    (ROOT / "outputs/model-copy.json").write_text(
        json.dumps(
            {"source": str(source), "target": str(TARGET), "files": results}, indent=2
        )
    )
    print("Model copied and verified", len(results), flush=True)


if __name__ == "__main__":
    main()
