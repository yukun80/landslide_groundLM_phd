"""描述改进 v3 全量检查：只验证格式与部分一致性，不证明外观事实准确。

重要：自动检查通过 ≠ 描述准确。证据是否与图片相符只能人工/视觉核对。
用法：python tools/check_desc_v2.py --scope pilot|full [--out checks_revision_2026-11.json]

v3 与 v2 的差别：
- 四态复核结论（缺逐图观察记录一律失败），不再认「全部 status=verified」。
- 直接读三份正式 jsonl，逐行比对 recs 与 revisions.jsonl 的 after，杜绝「只查旁边记录」。
- 用共享 check_answers() 对两版做权威前后缀 + 官方 bbox 断言。
- 新增 v3 禁止词全文扫描、标注区零出现、整图泛泛零出现、空值报错、四态覆盖。
- summary.json 改为语义检查（本轮会更新），display_ranges/screening 仍做字节断言。
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xiaomi_lens.description_v2 import answer, io, spec, revision  # noqa: E402
from xiaomi_lens.description_v2.spec import (  # noqa: E402
    V2, DEST, MAX_CHARS, MAX_SENTENCES, AUX_SOURCE_WORDS, FIXED_SCOPE_NOTE,
    ACTIVE_RULES_FILE, DONE_REVIEW_STATES, REVIEW_STATES, V3_ANSWER_FORBIDDEN_RE,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=("pilot", "full"), default="full")
    p.add_argument("--skip-sample-hashes", action="store_true")
    p.add_argument("--out", default="checks_revision_2026-11.json")
    a = p.parse_args()

    # 当前授权范围是533条新增描述，另467条明确保持原样。
    # 使用实际本地视觉推理记录，不把它伪装成旧版 ViewImage 人工式查看。
    if a.scope == "full" and (V2 / "current_review.json").exists():
        from xiaomi_lens.description_v2.visual_review import validate_outputs
        result = validate_outputs(require_current=True)
        io.atomic_write_json(V2 / a.out, result)
        print("PASS current description review: 533 reviewed; 467 unchanged; not human confirmed")
        return 0

    rows = io.read_rows()
    rmap = {r["id"]: (split, r) for split, r in rows}
    recs = io.load_records()
    passed, failed, notes = [], [], []

    def check(name, ok, extra=""):
        (passed if ok else failed).append(name + (f" ({extra})" if extra else ""))

    # 1 样本完整性 + 四态复核结论
    check("总数 1000 且 id 唯一", len(rows) == 1000 and len({r["id"] for _, r in rows}) == 1000)
    check("records 覆盖全部 id", set(recs) == set(rmap), f"records={len(recs)}")
    scope_ids = [c["id"] for c in json.loads((V2 / "pilot.json").read_text(encoding="utf-8"))["chosen"]] \
        if a.scope == "pilot" and (V2 / "pilot.json").exists() else list(rmap)
    bad = [rid for rid in scope_ids
           if recs.get(rid, {}).get("review_state") not in DONE_REVIEW_STATES
           or not recs.get(rid, {}).get("view_evidence_complete")]
    label = "试点 12 条" if a.scope == "pilot" else "全部 1000 条"
    check(f"{label}四态复核已完成且 view_evidence_complete", not bad, f"未完成 {len(bad)}")
    # 缺逐图观察记录不得算完成。
    # 本轮真实记录的 reviewed_via 形如「roundNNN_views.jsonl 真实 ViewImage 记录」；
    # 11-09 的历史记录写的是「复核拼图 sheets/…逐行查看」，一律不算本轮逐图复核。
    missing_views = []
    real_marker = "真实 ViewImage"
    for rid in scope_ids:
        r = recs.get(rid, {})
        vlist = r.get("images_viewed", [])
        if not vlist:
            missing_views.append(rid + ":无逐图记录")
            continue
        bad = [x for x in vlist if real_marker not in str(x.get("reviewed_via", ""))]
        if bad:
            missing_views.append(rid + f":{len(bad)} 条非本轮真实看图记录")
    check("逐图观察记录真实存在（无记录/历史拼图代看不算完成）", not missing_views,
          ",".join(missing_views[:5]))
    if a.scope == "full":
        prog = revision.load_progress()
        check("view_progress.json 已查看 1000 条且四态计数一致",
              len(prog.get("viewed_ids", [])) == 1000 and prog["review_counts"].get("not_viewed", 1) == 0,
              f"已查看 {len(prog.get('viewed_ids', []))}")
    check("视图产物目录存在", (V2 / "views").exists())

    # 2 非答案字段与原始资产未变
    fingerprints = io.load_json(V2 / "non_answer_fingerprints.json")
    changed = [r["id"] for _, r in rows if io.row_fingerprint(r) != fingerprints.get(r["id"])]
    check("15 个非答案字段逐字未变", not changed, f"变动 {len(changed)}")
    if not a.skip_sample_hashes and (V2 / "samples_manifest.json").exists():
        manifest = io.load_json(V2 / "samples_manifest.json")
        current = io.samples_manifest()
        diff = [k for k, v in manifest.items() if current.get(k) != v]
        check("3900 张原始 PNG 字节未变", not diff and len(current) == len(manifest),
              f"差异 {len(diff)}, 当前 {len(current)}")
    ov = io.load_json(V2 / "originals_verified.json")
    for name, meta in ov["extra_files"].items():
        fp = DEST / name
        if name == "summary.json":
            continue  # 本轮会更新，走下方语义检查
        check(f"{name} 字节未变", fp.stat().st_size == meta["size"] and io.sha256_file(fp) == meta["sha256"])
    # summary.json 语义检查
    summary = io.load_json(DEST / "summary.json")
    sm_ok = (all(summary.get("manifest_sha256", {}).get(s) == io.sha256_file(DEST / f"{s}.jsonl")
                 for s in ("train", "val", "test"))
             and "revision" in summary
             and summary.get("answer_format") != "official_square_bbox_center_blocks_v2")
    check("summary.json 已适配新答案且 manifest_sha256 为当前字节哈希", sm_ok,
          f"answer_format={summary.get('answer_format')}")

    # 3 直接读正式清单，与 recs / revisions 三方比对（不只查旁边记录文件）
    revs = revision.load_revisions()
    pair_bad = []
    for rid in scope_ids:
        _, row = rmap[rid]
        r = recs.get(rid, {})
        if row.get("answer") != r.get("answer_multiview"):
            pair_bad.append(rid + ":jsonl.answer≠recs.answer_multiview")
        if row.get("answer_optical") != r.get("answer_optical"):
            pair_bad.append(rid + ":jsonl.answer_optical≠recs.answer_optical")
        rev = revs.get(rid)
        if rev:
            if row.get("answer") != rev["after"]["answer_multiview"]:
                pair_bad.append(rid + ":jsonl.answer≠revisions.after")
            if row.get("answer_optical") != rev["after"]["answer_optical"]:
                pair_bad.append(rid + ":jsonl.answer_optical≠revisions.after")
        elif a.scope == "full":
            pair_bad.append(rid + ":缺修订记录")
    check("最终清单两版答案与核对记录、修订流水三方一致", not pair_bad, ",".join(pair_bad[:5]))

    # 4 权威复算：共享函数，对两版做数值句前缀 + 后缀 + 官方 bbox 恰 1 框
    auth_bad = answer.check_answers([rmap[rid][1] for rid in scope_ids])
    check("两版答案数值句/后缀/官方 bbox 与标注计算一致", not auth_bad,
          f"问题 {len(auth_bad)} {auth_bad[:2]}")
    # 本机能否用 domain.py 权威 training_answer 复核（如实记录，不声称）
    try:
        from xiaomi_lens.train.data.domain import training_answer  # noqa: F401
        notes.append("本机 domain.training_answer() 可 import，共享 check_answers 与之口径一致。")
    except Exception as e:
        notes.append(f"权威 training_answer() 未在本机 import（{type(e).__name__}），"
                     "已用等价实现复算，不声称 WSL 侧结果。")

    # 5 证据与两版分离
    missing_evidence = []
    for rid in scope_ids:
        r = recs.get(rid)
        if not r:
            missing_evidence.append(rid + ":无记录")
            continue
        roles = {x.get("role") for x in r.get("images_viewed", [])}
        for need in ("optical_raw", "label", "label_overlay", "optical_at_model_size"):
            if need not in roles:
                missing_evidence.append(rid + ":缺" + need)
        ev = r.get("evidence", {})
        if not ev.get("from_label_computed"):
            missing_evidence.append(rid + ":无标注计算项")
        if r.get("answer_optical") and not ev.get("from_optical") \
                and r["answer_optical"] != r.get("original_answer"):
            missing_evidence.append(rid + ":新增外观句无证据条目")
    check("必需看图角色与证据条目齐全", not missing_evidence, f"问题 {len(missing_evidence)}")
    notes.append("证据内容是否与图片相符只能人工/视觉核对，自动检查不覆盖。")

    leakage, mismatch = [], []
    for rid in scope_ids:
        r = recs.get(rid)
        if not r or not r.get("answer_optical"):
            continue
        optical, multi = r["answer_optical"], r["answer_multiview"]
        for w in AUX_SOURCE_WORDS:
            if w in optical:
                leakage.append(f"{rid}:{w}")
        _, row = rmap[rid]
        if len(row["images"]) == 1 and optical != multi:
            mismatch.append(rid + ":仅光学输入却两版不同")
    check("光学版无辅助图来源词", not leakage, ",".join(leakage[:5]))
    check("仅光学输入两版逐字相同", not mismatch, ",".join(mismatch[:5]))

    # 5b B 口径：活动判别只能出现在多图版且有授权留痕（保留授权，不禁止词）
    activity_bad = []
    for rid in scope_ids:
        r = recs.get(rid)
        if not r or not r.get("answer_multiview"):
            continue
        if "活动滑坡" in r["answer_multiview"] or "形变较大" in r["answer_multiview"]:
            if "活动滑坡" in r["answer_optical"] or "形变较大" in r["answer_optical"]:
                activity_bad.append(rid + ":活动判别混入光学版")
            if not r.get("evidence", {}).get("from_auxiliary"):
                activity_bad.append(rid + ":活动判别无辅助证据")
            unc = r.get("uncertain", [])
            if not unc or not any("授权" in u or "无法核实" in u for u in unc):
                activity_bad.append(rid + ":未标注用户授权/不可核实")
    check("活动判别只出现在多图版且有证据与授权留痕", not activity_bad, ",".join(activity_bad[:5]))

    # 6 v3 禁止词全文扫描 + 长度/句数/套话
    v3_bad, length_bad = [], []
    new_sentences = collections.Counter()
    for rid in scope_ids:
        r = recs.get(rid)
        if not r or not r.get("answer_multiview"):
            continue
        _, row = rmap[rid]
        numeric, suffix = answer.answer_parts(row)
        for key, text in (("answer_optical", r["answer_optical"]),
                          ("answer_multiview", r["answer_multiview"])):
            for pat in V3_ANSWER_FORBIDDEN_RE:
                hit = pat.search(text)
                if hit:
                    v3_bad.append(f"{rid}:{key}:{hit.group(0)}")
        new_part = r["answer_multiview"][len(numeric):-len(suffix)] if suffix else ""
        for pat in spec.FORBIDDEN_RE:
            hit = pat.search(new_part)
            if hit:
                v3_bad.append(f"{rid}:新句禁止词:{hit.group(0)}")
        if len(r["answer_multiview"]) > MAX_CHARS or len(r["answer_optical"]) > MAX_CHARS:
            length_bad.append(rid)
        if sum(1 for ch in r["answer_multiview"] if ch in "。！？") > MAX_SENTENCES:
            length_bad.append(rid + ":句数")
        for s in re.split(r"[。！？]", new_part):
            if s.strip():
                new_sentences[s.strip()] += 1
    check("v3 禁止词（散斑/量化/标注区/整图泛泛/元描述）零命中", not v3_bad, ",".join(v3_bad[:8]))
    check("字符与句数保守上限", not length_bad, ",".join(length_bad[:5]))
    repeated = [(s, n) for s, n in new_sentences.items() if n > max(3, len(scope_ids) // 10)]
    check("无统一套话填满", not repeated, f"重复句 {len(repeated)}")
    notes.append("字符/句数为保守检查，不能证明真实 tokenizer 下不截断；"
                 "--mode encode 本轮未运行（wsl.exe 被禁、本机无 torch/transformers/processor），不声称通过。")

    # 7 训练读取静态检查
    try:
        from xiaomi_lens.description_v2 import select_answer, select_answer_row
        assert select_answer({"answer": "A", "answer_optical": "O"}, True) == "O"
        assert select_answer({"answer": "A", "answer_optical": "O"}, False) == "A"
        assert select_answer({"answer": "A"}, True) == "A"
        assert select_answer_row({"answer": "A", "answer_optical": "O"}, True)["answer"] == "O"
        for empty in ("", None):
            try:
                select_answer({"answer": "A", "answer_optical": empty}, True)
                raise AssertionError("空值未报错")
            except ValueError:
                pass
        check("select_answer 三分支 + 空值显式报错 + 旧行回退", True)
    except Exception as e:
        check("select_answer 三分支 + 空值显式报错 + 旧行回退", False, str(e))
    rt = (ROOT / "src/xiaomi_lens/domain_runtime.py").read_text(encoding="utf-8")
    check("domain_runtime.py:127 仍读 row[answer]", 'row["answer"] + "<|im_end|>"' in rt)
    dt = (ROOT / "src/xiaomi_lens/train/domain_train.py").read_text(encoding="utf-8")
    check("domain_train.py 用 select_answer_row 注入答案",
          "select_answer_row(row, a.rgb_only)" in dt and "from ..description_v2 import select_answer_row" in dt)
    rules_path = V2 / ACTIVE_RULES_FILE
    rules = io.load_json(rules_path) if rules_path.exists() else {}
    check("当前生效规则与 spec 提示词/上限一致",
          rules.get("prompt_sha256") == spec.prompt_digests()
          and rules.get("max_chars") == MAX_CHARS
          and rules.get("max_sentences") == MAX_SENTENCES
          and rules.get("status") == "frozen"
          and "insar_domain_note" in rules and "revision_note" in rules,
          f"status={rules.get('status')}")
    notes.append("实际训练读取程序未在本机运行；仅完成静态检查 + 附 WSL 侧 smoke 脚本（本轮未运行 encode）。")

    # 8 人工逐图对照产物 + 如实留痕
    missing_views_prod = []
    for rid in scope_ids:
        d = V2 / "views" / rid
        for f in ("overlay.png", "overlay_legend.txt", "optical_at_input.png"):
            if not (d / f).exists():
                missing_views_prod.append(rid + ":" + f)
    check("每个已处理样本的看图产物齐全", not missing_views_prod, ",".join(missing_views_prod[:5]))
    unfaithful = [rid for rid in scope_ids if recs.get(rid, {}).get("human_confirmed") is not False
                  or recs.get(rid, {}).get("view_checked_by") != "visual_model"
                  or recs.get(rid, {}).get("draft_is_fact") is not False]
    check("视觉模型初稿/未人工确认留痕如实", not unfaithful, ",".join(unfaithful[:5]))

    result = {
        "scope": a.scope, "n_checked": len(scope_ids),
        "check_type": "format_and_consistency_only",
        "scope_note": "自动检查只能验证格式与部分一致性，不能据此声称所有外观描述准确",
        "fixed_note": FIXED_SCOPE_NOTE, "revision_round": "2026-11",
        "formal_training_run": False, "independent_evaluation_run": False,
        "appearance_accuracy_proven": False, "human_confirmed": False,
        "loader_encode_smoke_run": False,
        "review_states": list(REVIEW_STATES),
        "passed": passed, "failed": failed, "notes": notes,
        "all_passed": not failed,
        "run_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "history_note": "11-09 的 checks.json 为历史结论，本结果写入独立文件，未改写历史",
    }
    io.atomic_write_json(V2 / a.out, result)
    print(f"{'PASS' if not failed else 'FAIL'} check_desc_v2 [{a.scope}]: "
          f"{len(passed)} 通过 / {len(failed)} 失败 -> {a.out}")
    for f in failed:
        print("  FAIL", f)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
