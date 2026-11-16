"""统一评测自由预测；忽略标注无效区，单列空图误报与文字限制。"""

from pathlib import Path
import json, hashlib
import torch
import numpy as np
from PIL import Image
from .domain_runtime import predict, PROMPT, CONFIG
from .train.data.domain import ROOT, write_json


def score(mask, truth, valid):
    p = mask & valid
    t = truth & valid
    intersection = int((p & t).sum())
    union = int((p | t).sum())
    return {
        "intersection": intersection,
        "union": union,
        "predicted_pixels": int(p.sum()),
        "true_pixels": int(t.sum()),
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (p.sum() + t.sum()) if p.any() or t.any() else 1.0,
        "background": not bool(t.any()),
        "false_positive_image": not t.any() and bool(p.any()),
        "missed_pixels": int((t & ~p).sum()),
        "extra_pixels": int((p & ~t).sum()),
    }


def evaluate(model, processor, rows, outdir, rgb_only=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            digest.update(name.encode())
            digest.update(
                param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
    contract = {
        "model": digest.hexdigest(),
        "ids": [r["id"] for r in rows],
        "rgb_only": rgb_only,
        "prompt": PROMPT, "settings": CONFIG,
        "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    }
    if (outdir / "contract.json").exists() and json.loads(
        (outdir / "contract.json").read_text()
    ) != contract:
        raise ValueError("评估目录属于不同模型或样本，请使用新目录")
    write_json(outdir / "contract.json", contract)
    for row in rows:
        folder = outdir / row["id"]
        # 只复用当前这一次运行目录中已完成且输入一致的结果。
        identity = {"images": row["images"], "rgb_only": rgb_only}
        if (folder / "score.json").exists():
            item = json.loads((folder / "score.json").read_text())
            if item["input"] != identity:
                raise ValueError("evaluation input changed")
        else:
            prediction, result = predict(model, processor, row, folder, rgb_only)
            truth = np.array(Image.open(ROOT / row["mask"])) > 0
            valid = np.array(Image.open(ROOT / row["valid"])) > 0
            item = {
                "id": row["id"],
                "source": row["source"],
                "input": identity,
                **score(prediction, truth, valid),
                **result,
            }
            # 自动检查只能找出明显越界表达，不能替代逐图审阅。
            item["unsupported_terms"] = [
                s
                for s in ["降雨导致", "地震导致", "毫米", "高风险", "危险等级", "活动滑坡", "形变较大"]
                if s in result["answer"]
            ]
            write_json(folder / "score.json", item)
        records.append(item)
        print(
            "eval",
            len(records),
            len(rows),
            row["id"],
            round(item["iou"], 4),
            flush=True,
        )
    summary = {}
    for source in sorted({r["source"] for r in records}) + ["all"]:
        items = [r for r in records if source == "all" or r["source"] == source]
        bg = [r for r in items if r["background"]]
        positive = [r for r in items if not r["background"]]
        summary[source] = {
            "n": len(items),
            "mean_dice": float(np.mean([r["dice"] for r in items])),
            "pixel_recall": sum(r["intersection"] for r in items)
            / max(1, sum(r["true_pixels"] for r in items)),
            "pixel_precision": sum(r["intersection"] for r in items)
            / max(1, sum(r["predicted_pixels"] for r in items)),
            "target_image_recall": (
                sum(r["predicted_pixels"] > 0 for r in positive) / len(positive)
                if positive
                else None
            ),
            "mean_iou": float(np.mean([r["iou"] for r in items])),
            "positive_mean_iou": (
                float(np.mean([r["iou"] for r in positive])) if positive else None
            ),
            "background_count": len(bg),
            "background_false_positive_rate": (
                sum(r["false_positive_image"] for r in bg) / len(bg) if bg else None
            ),
            "answer_background_false_positive_rate": (
                sum(r.get("answer_target_present") is True for r in bg) / len(bg)
                if bg
                else None
            ),
            "answer_presence_unknown": sum(
                r.get("answer_target_present") is None for r in items
            ),
            "text_mask_conflicts": sum(bool(r["text_mask_conflict"]) for r in items),
            "empty_answers": sum(not r["answer"].strip() for r in items),
            "missing_region_markers": sum(r["mask_count"] == 0 for r in items),
            "unsupported_answers": sum(bool(r["unsupported_terms"]) for r in items),
            "truncated_answers": sum(r["truncated"] for r in items),
            "mean_seconds": float(np.mean([r["seconds"] for r in items])),
        }
    summary["selection_score"] = float(
        np.mean([summary[s]["mean_iou"] for s in {r["source"] for r in records}])
    )
    summary["description_review"] = (
        "自动检查及区域评分不等于人工认可描述正确，需查看图片与回答"
    )
    write_json(outdir / "summary.json", summary)
    if len(rows) >= 30:
        from .train.run_review import prepare_review
        prepare_review(rows, outdir)
    return summary
