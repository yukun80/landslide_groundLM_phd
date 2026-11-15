"""WSL 侧 loader 冒烟脚本（本机 Windows 不运行）。

用法（在项目的 qwen3vl 环境里）：
  scripts/domain.sh tools/smoke_desc_v2_loader.py --mode data
  scripts/domain.sh tools/smoke_desc_v2_loader.py --mode encode --limit 20

--mode data   纯数据：两版答案字段、select_answer 分支、后缀一致性。
--mode encode 只加载 processor 并调用 build(training=True) 检查编码长度；
              不加载模型权重、不做前向/反向、不写 checkpoint、不启动正式训练。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xiaomi_lens.description_v2 import io, spec  # noqa: E402
from xiaomi_lens.description_v2.answer import (  # noqa: E402
    select_answer, select_answer_row, split_original, suffix_from_mask,
    load_mask_valid, official_parse_bboxes,
)
from xiaomi_lens.train.data.domain import DEST, jsonlines, training_answer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("data", "encode"), default="data")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="loader_smoke.json")  # 本轮传 loader_smoke_revision.json，不覆盖历史
    a = p.parse_args()

    rows = [(s, r) for s in ("train", "val", "test") for r in jsonlines(DEST / f"{s}.jsonl")]
    if a.limit:
        rows = rows[: a.limit]
    records = io.load_records()
    result = {"mode": a.mode, "n": len(rows), "passed": False,
              "formal_training_run": False, "revision_round": "2026-11",
              "note": "读取 apply 之后的最终 jsonl；--mode encode 需 WSL qwen3vl 环境（本机无 torch/transformers/processor）",
              "loader_encode_smoke_run": False}

    errors = []
    for split, r in rows:
        if "answer_optical" not in r or not r["answer_optical"]:
            errors.append(f"{r['id']}: 缺或空 answer_optical")
            continue
        if select_answer(r, True) != r["answer_optical"]:
            errors.append(f"{r['id']}: rgb_only 未选中光学版")
        if select_answer(r, False) != r["answer"]:
            errors.append(f"{r['id']}: multiview 未选中多图版")
        # 用原函数权威复核数值与后缀（WSL 有 scipy/h5py）
        m, v = load_mask_valid(r)
        expected_suffix = suffix_from_mask(m, v)
        for key in ("answer", "answer_optical"):
            text = r[key]
            if not text.endswith(expected_suffix):
                errors.append(f"{r['id']}: {key} 后缀与 training_answer 不一致")
            boxes = official_parse_bboxes(text)
            if len(boxes) != 1:
                errors.append(f"{r['id']}: {key} 框数量异常")
        # apply 之后数值句后面还会接外观句，split_original(answer) 得到的是整段前缀；
        # 因此用权威 training_answer() 拆出「数值句 + 后缀」，分别做前缀 / 后缀断言。
        authoritative = training_answer(m, v)
        t_num, t_suf = split_original(authoritative)
        for key in ("answer", "answer_optical"):
            if not r[key].startswith(t_num):
                errors.append(f"{r['id']}: {key} 数值句与 training_answer 不一致")
        if t_suf != expected_suffix:
            errors.append(f"{r['id']}: training_answer 后缀与 description_v2 计算不一致")
        # 阶段 B 用 v2 实现复核过原答案；这里用 domain.py 权威函数逐字复核唯一副本。
        if records.get(r["id"], {}).get("original_answer") != authoritative:
            errors.append(f"{r['id']}: 原答案唯一副本与 training_answer 不一致")
        if select_answer_row(r, True)["answer"] != r["answer_optical"]:
            errors.append(f"{r['id']}: select_answer_row 注入失败")
    # 旧数据回退 + v3 空值不得静默回退
    if select_answer({"answer": "A"}, True) != "A":
        errors.append("旧行无 answer_optical 时回退失败")
    for empty in ("", None):
        try:
            select_answer({"answer": "A", "answer_optical": empty}, True)
            errors.append(f"新版 answer_optical={empty!r} 未报错，存在静默回退风险")
        except ValueError:
            pass

    if a.mode == "encode":
        result["loader_encode_smoke_run"] = True
        import torch
        from transformers import AutoProcessor
        from xiaomi_lens.domain_runtime import BASE, CONFIG, build, configure_processor
        torch.set_num_threads(4)
        processor = configure_processor(
            AutoProcessor.from_pretrained(BASE, local_files_only=True, use_fast=False)
        )
        max_ids = 0
        for split, r in rows:
            for rgb_only in (True, False):
                batch = build(processor, select_answer_row(r, rgb_only), rgb_only, training=True)
                n = int(batch["input_ids"].shape[1])
                max_ids = max(max_ids, n)
                if n > CONFIG["maximum_sequence_length"]:
                    errors.append(f"{r['id']}: 编码 {n} 超过 {CONFIG['maximum_sequence_length']}")
                if "mask_targets" not in batch:
                    errors.append(f"{r['id']}: 缺 mask_targets")
                del batch
        result["max_input_ids"] = max_ids
        result["maximum_sequence_length"] = CONFIG["maximum_sequence_length"]
        result["maximum_answer_tokens"] = CONFIG["maximum_answer_tokens"]
        # B 口径描述变长：额外检查答案 token 是否超过推理生成上限，避免推理时被静默截断。
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            max_ans = 0
            for split, r in rows:
                for key in ("answer", "answer_optical"):
                    n_ans = len(tokenizer(r[key], add_special_tokens=False)["input_ids"])
                    max_ans = max(max_ans, n_ans)
                    if n_ans > CONFIG["maximum_answer_tokens"]:
                        errors.append(
                            f"{r['id']}: {key} 答案 {n_ans} tokens 超过 "
                            f"maximum_answer_tokens={CONFIG['maximum_answer_tokens']}（推理生成会被截断）"
                        )
            result["max_answer_tokens_observed"] = max_ans
            result["answer_token_check"] = "tokenizer 直接编码答案文本的保守估算，未计入特殊 token 合并"
        else:
            result["answer_token_check"] = "processor 未暴露 tokenizer，跳过答案 token 估算"
        result["note"] += "；--mode encode 只加载 processor 做编码，未加载权重、未前向，不构成正式训练"

    result["errors"] = errors[:20]
    result["passed"] = not errors
    out = spec.V2 / a.out
    io.atomic_write_json(out, result)
    print(("PASS" if not errors else "FAIL") + f" smoke_desc_v2_loader [{a.mode}]: {len(rows)} 行")
    for e in errors[:10]:
        print("  ", e)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
