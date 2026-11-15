"""单卡两阶段训练，逐轮自由预测选模型，保存优化器及随机状态用于恢复。"""

import argparse, json, random, time, math, shutil
from pathlib import Path
import numpy as np
import torch
from ..domain_runtime import load, build, save_checkpoint, seed_all, CONFIG
from ..domain_evaluation import evaluate
from ..description_v2 import select_answer_row
from .data.domain import DEST, ROOT, jsonlines, write_json, NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--rgb-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    a = p.parse_args()
    seed_all(CONFIG["seed"])
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "train.jsonl").exists() and not a.resume:
        raise RuntimeError("已有训练记录；请用 --resume 恢复，或指定新的输出目录")
    rows = jsonlines(DEST / "train.jsonl")
    val = jsonlines(DEST / "val.jsonl")
    if a.smoke:
        rows = [
            r
            for n, quota in zip(NAMES, [6, 5, 5])
            for r in [x for x in rows if x["source"] == n][:quota]
        ]
        val = [next(r for r in val if r["source"] == n) for n in NAMES]
    if not all(r["positive_pixels"] > 0 for r in rows + val):
        raise ValueError("当前方案仅允许有有效目标的样本")
    model, processor = load(a.resume, training=True)
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
    resume_state = None
    if a.resume:
        resume_state = torch.load(a.resume, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer"])
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
            "source_dataset": json.loads((DEST / "summary.json").read_text()),
        },
    )
    torch.cuda.reset_peak_memory_stats()
    initial = (
        {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        if a.smoke
        else None
    )
    for epoch in range(start, a.epochs):
        order = list(rows)
        random.Random(CONFIG["seed"] + epoch).shuffle(order)
        model.train()
        model.config.lm_loss_weight = -1.0 if epoch == 0 else 1.0
        for param in adapter:
            param.requires_grad_(epoch != 0)
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        t = time.monotonic()
        for i, row in enumerate(order):
            inputs = build(processor, select_answer_row(row, a.rgb_only), a.rgb_only, training=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inputs, use_cache=False)
                loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss {row["id"]}')
            accumulation = CONFIG["accumulation"]
            group_size = min(
                accumulation, len(order) - (i // accumulation) * accumulation
            )
            (loss / group_size).backward()
            total += float(loss.detach())
            if (i + 1) % CONFIG["accumulation"] == 0 or i + 1 == len(order):
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
                "sample": i + 1,
                "id": row["id"],
                "loss": float(loss.detach()),
                "seconds": time.monotonic() - t,
                "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            with (folder / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if i % 10 == 0:
                print(record, flush=True)
            del out, loss, inputs
        for param in adapter:
            param.requires_grad_(True)
        model.eval()
        model.config.lm_loss_weight = 1.0
        if a.smoke:
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
                    "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
                },
            )
        summary = evaluate(
            model, processor, val, folder / f"val-epoch-{epoch}", a.rgb_only
        )
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
            {"best_score": best},
        )
        if improved:
            shutil.copy2(folder / "last.pt", folder / "best.pt")
        print("epoch complete", epoch, summary["selection_score"], flush=True)
    write_json(
        folder / "completed.json",
        {
            "epochs": a.epochs,
            "step": step,
            "best_score": best,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        },
    )


if __name__ == "__main__":
    main()
