"""单版本多轮训练：新目录、固定数据、阶段恢复、模型锁定后最终评估。

运行时一直保留进程锁；任何子步骤失败立即记录，不自动调整科学设置。
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xiaomi_lens.train.data.domain import DEST, NAMES, jsonlines, write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = (ROOT / args.output).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock_handle = (run / "process.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    write_json(run / "process.json", {"pid": os.getpid(), "python": sys.executable,
                                      "started": time.time()})

    def command(stage, command_args):
        done = run / f"{stage}.done.json"
        if done.exists():
            return
        write_json(run / "status.json", {"stage": stage, "state": "running", "time": time.time()})
        print(stage, flush=True)
        with (run / f"{stage}.log").open("a") as log:
            result = subprocess.run([str(ROOT / "scripts/domain.sh"), *map(str, command_args)],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(run / "status.json", {"stage": stage, "state": "failed",
                                              "returncode": result.returncode, "time": time.time()})
            raise RuntimeError(f"{stage}失败，请检查{run / (stage + '.log')}")
        write_json(done, {"command": list(map(str, command_args)), "finished": time.time()})

    write_json(run / "status.json", {"stage": "input-verification", "state": "running", "time": time.time()})

    # 数据、图片、原始模型逐文件校验，恢复时也再核对，不能把不同输入混作同一实验。
    rows = [r for split in ("train", "val", "test") for r in jsonlines(DEST / f"{split}.jsonl")]
    assert len(rows) == 1000 and len({r["id"] for r in rows}) == 1000
    assert all(r["positive_pixels"] > 0 for r in rows)
    files = {DEST / f"{s}.jsonl" for s in ("train", "val", "test")}
    for row in rows:
        files.update(ROOT / p for p in row["images"] + [row["mask"], row["valid"]])
    model_record = json.loads((ROOT / "outputs/model-copy.json").read_text())
    for item in model_record["files"]:
        path = ROOT / "checkpoints/Xiaomi-Lens-0" / item["file"]
        assert digest(path) == item["sha256"], path
        files.add(path)
    original = json.loads((ROOT / "outputs/original_code_hashes.json").read_text())
    for p, expected in original.items():
        assert digest(p) == expected, p
    snapshot = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    frozen = run / "input_hashes.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == snapshot, "输入发生变化，停止恢复"
    else:
        write_json(frozen, snapshot)
    command("stopping-tests", ["tools/test_domain_stopping.py"])
    command("data-check", ["tools/check_domain_data.py", "--out", run / "data-check.json"])
    command("encoding-check", ["tools/smoke_desc_v2_loader.py", "--mode", "encode",
                               "--out", run / "encoding-check.json"])
    smoke = run / "smoke"
    smoke_args = ["-m", "xiaomi_lens.train.domain_train", "--smoke", "--epochs", "1", "--output", smoke]
    if not (run / "smoke-first.done.json").exists():
        if (smoke / "last.pt").exists():
            smoke_args += ["--resume", smoke / "last.pt"]
        elif (smoke / "train.jsonl").exists():
            smoke_args += ["--restart-incomplete"]
    command("smoke-first", smoke_args)
    command("smoke-resume", ["-m", "xiaomi_lens.train.domain_train", "--smoke", "--epochs", "2", "--output", smoke,
                             "--resume", smoke / "last.pt"])
    command("reload-check", ["tools/check_domain_reload.py", "--training-dir", smoke,
                             "--output", run / "smoke-reload", "--report", run / "reload-check.json"])
    command("baseline-val", ["eval/domain.py", "--output", run / "baseline-val"])
    training = run / "multiview"
    train_args = ["-m", "xiaomi_lens.train.domain_train", "--epochs", "20", "--early-stop", "--output", training]
    if (training / "last.pt").exists():
        train_args += ["--resume", training / "last.pt"]
    elif (training / "train.jsonl").exists():
        train_args += ["--restart-incomplete"]
    command("training", train_args)
    completion = json.loads((training / "completed.json").read_text())
    assert completion["formal_training"] and completion["epochs"] >= 6
    model_lock = {"models": {"multiview": {"path": str(training / "best.pt"),
                                           "sha256": digest(training / "best.pt")}},
                  "test_sha256": digest(DEST / "test.jsonl"),
                  "base_model": {r["file"]: r["sha256"] for r in model_record["files"]},
                  "training_completion": completion,
                  "rule": "Final evaluation only; no further tuning using test results"}
    locked = run / "final-models-locked.json"
    if locked.exists():
        assert json.loads(locked.read_text()) == model_lock, "禁止更换已锁定模型"
    write_json(locked, model_lock)
    command("baseline-test", ["eval/domain.py", "--split", "test", "--lock", locked, "--output", run / "baseline-test"])
    command("finetuned-test", ["eval/domain.py", "--split", "test", "--lock", locked,
                               "--checkpoint", training / "best.pt", "--output", run / "finetuned-test"])
    results = {name: json.loads((run / name / "summary.json").read_text())
               for name in ("baseline-val", "baseline-test", "finetuned-test")}
    results["training"] = completion
    results["test_change_by_source"] = {
        source: results["finetuned-test"][source]["mean_iou"] - results["baseline-test"][source]["mean_iou"]
        for source in NAMES + ["all"]}
    results["remaining"] = "需要实际查看逐轮和最终图文；无背景样本；文字并未经过人工逐图确认"
    write_json(run / "comparison.json", results)
    assert all(digest(ROOT / name) == sha for name, sha in snapshot.items()), "输入文件意外改变"
    write_json(run / "status.json", {"stage": "numerical_evaluation_complete", "state": "awaiting_visual_review",
                                      "time": time.time()})
    print("数值训练与评估完成，等待图文复核", flush=True)


if __name__ == "__main__":
    main()
