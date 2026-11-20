"""单卡两阶段训练，逐轮自由预测选模型，保存优化器及随机状态用于恢复。"""

import argparse, json, random, time, shutil, hashlib
from pathlib import Path
import numpy as np
import torch
from ..domain_runtime import load, save_checkpoint, seed_all, CONFIG, PROMPT
from ..domain_evaluation import evaluate
from .stopping import StopState
from .data.domain import DEST, jsonlines, write_json, NAMES
from .domain_batch import training_batches, merged_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-dir", type=Path, default=DEST, help="显式指定数据目录；默认保留原1000条")
    p.add_argument("--rgb-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--restart-incomplete", action="store_true", help="仅在首轮尚无保存点时保留失败记录并重跑")
    p.add_argument("--micro-batch", type=int, choices=(1,2,3,4,5,6,7,8), default=1)
    p.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=0,
                   help="在 CPU 上提前准备下一批训练图片")
    p.add_argument("--use-merged-descriptions", action="store_true")
    p.add_argument("--vram-probe", action="store_true", help="仅用训练集进行两阶段显存试跑")
    p.add_argument("--probe-mixed", action="store_true", help="显存试跑中打乱三个来源，核对混合输入")
    p.add_argument("--keep-activations", action="store_true",
                   help="保留前向计算结果，减少反向重算；需要更多显存")
    p.add_argument("--retain-layers", type=int, default=0, choices=range(33),
                   help="仅保留前若干语言层的中间结果，其余继续重算")
    a = p.parse_args()
    if a.keep_activations and a.retain_layers:
        p.error("choose either full or partial activation retention")
    if a.probe_mixed and not a.vram_probe:
        p.error("--probe-mixed requires --vram-probe")
    if a.epochs < 1 or (a.early_stop and a.epochs < 6):
        p.error("轮次不足；自动停止至少需要1轮区域训练和5轮联合训练")
    dataset_dir = a.dataset_dir.resolve()
    receipt = merged_snapshot(dataset_dir) if a.use_merged_descriptions else None
    if receipt is None and (dataset_dir / "build.json").exists():
        summary_path = dataset_dir / "summary.json"
        if not summary_path.exists() or not json.loads(summary_path.read_text()).get("complete", False):
            raise ValueError("全量数据尚未完成文字与检查，禁止启动训练")
    seed_all(CONFIG["seed"])
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "train.jsonl").exists() and not a.resume:
        if not a.restart_incomplete or (folder / "last.pt").exists():
            raise RuntimeError("已有训练记录；请用 --resume 恢复，或指定新的输出目录")
        (folder / "train.jsonl").rename(folder / f"train-interrupted-{time.time_ns()}.jsonl")
    rows = jsonlines(dataset_dir / "train.jsonl")
    val = jsonlines(dataset_dir / "val.jsonl")
    if a.vram_probe:
        # Largest display per source and long answers, in contiguous groups of eight.
        groups = [sorted((r for r in rows if r['source'] == name),
                         key=lambda r: (min(r['height'],r['width']),len(r['answer'])),
                         reverse=True)[:10] for name in NAMES]
        val = [g[9] for g in groups]
        rows = [r for g in groups for r in g[:8]] + [groups[0][8]]
    elif a.smoke:
        rows = [
            r
            for n, quota in zip(NAMES, [6, 5, 5])
            for r in [x for x in rows if x["source"] == n][:quota]
        ]
        val = [next(r for r in val if r["source"] == n) for n in NAMES]
    if not all(r["positive_pixels"] > 0 for r in rows + val):
        raise ValueError("当前方案仅允许有有效目标的样本")
    contract = {
        "dataset": {s: hashlib.sha256((dataset_dir / f"{s}.jsonl").read_bytes()).hexdigest()
                    for s in ("train", "val", "test")},
        "settings": CONFIG, "prompt": PROMPT, "rgb_only": a.rgb_only,
        "smoke": a.smoke, "early_stop": a.early_stop,
    }
    if a.micro_batch != 1 or receipt or a.vram_probe:
        contract.update(micro_batch=a.micro_batch, merged_descriptions=receipt, vram_probe=a.vram_probe)
    if a.prefetch_batches:
        contract["prefetch_batches"] = a.prefetch_batches
    if a.retain_layers:
        contract["retain_layers"] = a.retain_layers
    if a.keep_activations:
        contract["keep_activations"] = True
    if a.probe_mixed:
        contract["probe_mixed"] = True
    if dataset_dir != DEST.resolve():
        contract["dataset_dir"] = str(dataset_dir)
    contract_path = folder / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("训练数据、提问或设置发生变化，禁止混合恢复")
    write_json(contract_path, contract)
    model, processor = load(a.resume, training=True)
    if a.keep_activations:
        model.gradient_checkpointing_disable()
    for layer in model.model.language_model.layers[:a.retain_layers]:
        if not layer.gradient_checkpointing:
            raise ValueError("语言层的重算设置与预期不一致")
        layer.gradient_checkpointing = False
    mask = [
        p
        for n, p in model.named_parameters()
        if n.startswith("mask_decoder.") and p.requires_grad
    ]
    adapter = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": mask, "lr": CONFIG["mask_learning_rate"]},
            {"params": adapter, "lr": CONFIG["adapter_learning_rate"]},
        ],
        weight_decay=0.0,
    )
    start = 0
    step = 0
    best = -1.0
    stopping = StopState()
    stopped = False
    resume_state = None
    if a.resume:
        resume_state = torch.load(a.resume, map_location="cpu", weights_only=False)
        if resume_state["extra"].get("contract") != contract:
            raise ValueError("保存文件与当前训练约定不一致")
        optimizer.load_state_dict(resume_state["optimizer"])
        stopping = StopState(**resume_state["extra"]["stopping"])
        stopped = bool(resume_state["extra"].get("stopped", False))
        # 若在保存last之后、复制best之前中断，恢复最佳副本。
        if stopping.best_epoch == resume_state["epoch"] and Path(a.resume).resolve() != (folder / "best.pt").resolve():
            shutil.copy2(a.resume, folder / "best.pt")
        start = resume_state["epoch"] + 1
        step = resume_state["step"]
        best = resume_state["extra"].get("best_score", -1.0)
        if resume_state["rgb_only"] != a.rgb_only:
            raise ValueError("resume modality mismatch")
        random.setstate(resume_state["random"])
        np.random.set_state(resume_state["numpy"])
        torch.set_rng_state(resume_state["torch"])
        torch.cuda.set_rng_state_all(resume_state["cuda"])
    write_json(
        folder / "config.json",
        {
            "args": vars(a),
            "train_count": len(rows),
            "val_count": len(val),
            "settings": CONFIG,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "source_dataset": json.loads((dataset_dir / "summary.json").read_text()),
        },
    )
    torch.cuda.reset_peak_memory_stats()
    initial = (
        {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        if a.smoke or a.vram_probe
        else None
    )
    if a.resume and not (folder / "best.pt").exists():
        raise ValueError("恢复目录缺少历史最佳模型，请在原运行目录恢复")
    run_peak = 0
    for epoch in range(start, a.epochs):
        if stopped:
            break
        previous_validation = folder / f"val-epoch-{epoch}"
        if previous_validation.exists():
            previous_validation.rename(folder / f"val-epoch-{epoch}-interrupted-{time.time_ns()}")
        order = list(rows)
        if not a.vram_probe or a.probe_mixed:
            random.Random(CONFIG["seed"] + epoch).shuffle(order)
        model.train()
        model.config.lm_loss_weight = -1.0 if epoch == 0 else 1.0
        for param in adapter:
            param.requires_grad_(epoch != 0)
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        t = time.monotonic()
        group = CONFIG["accumulation"]
        batches = training_batches(processor, order, a.rgb_only, group, a.micro_batch, a.prefetch_batches)
        torch.cuda.reset_peak_memory_stats()
        ready_at = time.monotonic()
        for i, chunk, inputs in batches:
            input_wait = time.monotonic() - ready_at
            end = i + len(chunk)
            row = chunk[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inputs, use_cache=False)
                loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss {row["id"]}')
            accumulation = CONFIG["accumulation"]
            group_size = min(
                accumulation, len(order) - (i // accumulation) * accumulation
            )
            (loss * (len(chunk) / group_size)).backward()
            total += float(loss.detach()) * len(chunk)
            if end % group == 0 or end == len(order):
                if epoch == 0:
                    for param in adapter:
                        param.grad = None
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            record = {
                "epoch": epoch,
                "sample": end, "ids": [r["id"] for r in chunk],
                "id": row["id"],
                "loss": float(loss.detach()),
                "language_loss": float(out.lm_loss.detach()) if out.lm_loss is not None else None,
                "mask_loss": float(out.mask_loss.detach()) if out.mask_loss is not None else None,
                "seconds": time.monotonic() - t, "input_wait_seconds": input_wait,
                "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            with (folder / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if i % 10 == 0:
                print(record, flush=True)
            del out, loss, inputs
            ready_at = time.monotonic()
        for param in adapter:
            param.requires_grad_(True)
        model.eval()
        model.config.lm_loss_weight = 1.0
        train_seconds = time.monotonic() - t
        train_peak = torch.cuda.max_memory_allocated()
        train_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        if a.smoke or a.vram_probe:
            changed = [
                n
                for n, p in model.named_parameters()
                if p.requires_grad and not torch.equal(p.detach().cpu(), initial[n])
            ]
            write_json(
                folder / f"epoch-{epoch}-update.json",
                {
                    "changed_parameters": changed,
                    "mean_loss": total / len(order),
                    "peak_gb": train_peak / 1024**3,
                },
            )
        validation_start = time.monotonic()
        summary = evaluate(
            model, processor, val, folder / f"val-epoch-{epoch}", a.rgb_only
        )
        validation_seconds = time.monotonic() - validation_start
        run_peak = max(run_peak, train_peak, torch.cuda.max_memory_allocated())
        stop_signal = stopping.update(epoch, summary["selection_score"])
        stopped = a.early_stop and stop_signal
        improved = summary["selection_score"] > best
        if improved:
            best = summary["selection_score"]
        save_checkpoint(
            folder / "last.pt",
            model,
            optimizer,
            epoch,
            step,
            a.rgb_only,
            {"best_score": best, "stopping": stopping.to_dict(),
             "stopped": stopped, "contract": contract},
        )
        if improved:
            shutil.copy2(folder / "last.pt", folder / "best.pt")
        write_json(folder / f"epoch-{epoch}.json", {
            "epoch": epoch, "stage": "mask" if epoch == 0 else "joint",
            "mean_loss": total / len(order), "seconds": time.monotonic() - t,
            "validation_seconds": validation_seconds, "train_seconds": train_seconds, "train_peak_bytes": train_peak, "train_reserved_bytes": train_reserved,
            "validation_peak_bytes": torch.cuda.max_memory_allocated(), "validation": summary, "stopping": stopping.to_dict(), "stopped": stopped,
        })
        print("epoch complete", epoch, summary["selection_score"], flush=True)
    write_json(
        folder / "completed.json",
        {
            "epochs": stopping.last_epoch + 1,
            "epoch_limit": a.epochs,
            "stop_reason": stopping.outcome(stopped),
            "stopping": stopping.to_dict(),
            "formal_training": not (a.smoke or a.vram_probe),
            "description_accuracy_confirmed": False,
            "step": step,
            "best_score": best,
            "peak_gb": run_peak / 1024**3,
        },
    )


if __name__ == "__main__":
    main()
