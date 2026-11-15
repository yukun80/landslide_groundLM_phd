"""描述改进 v2 的分阶段工具。

用法（Windows 侧 Bash 中的 python）：
  python tools/desc_v2.py init
  python tools/desc_v2.py sync-rules     # 改 spec 后重算 rules.json，不触碰记录
  python tools/desc_v2.py verify
  python tools/desc_v2.py aux-spec
  python tools/desc_v2.py select-pilot
  python tools/desc_v2.py render --pilot
  python tools/desc_v2.py write-drafts --input V2/drafts_pilot.json
  python tools/desc_v2.py pilot-review
  python tools/desc_v2.py apply          # 阶段 F，须先冻结规则并全部 verified

本机不启动训练；apply 之前三份 jsonl 保持原样。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xiaomi_lens.description_v2 import answer, io, render, spec, revision  # noqa: E402
from xiaomi_lens.description_v2.spec import (  # noqa: E402
    V2, DEST, ROOT as _ROOT, VIEW_SPEC, OPTICAL_INDEX, PROMPTS, prompt_digests,
    VIEW_MODEL, VIEW_MODEL_NOTE, FIXED_SCOPE_NOTE, OPTICAL_ONLY_NOTE,
    MAX_CHARS, MAX_SENTENCES, BATCH_SIZE, N_BATCHES, input_size,
    INSAR_DOMAIN_NOTE, AUX_RELATIVE_RULES, ACTIVITY_JUDGEMENT_NOTE,
    ACTIVE_RULES_FILE, REVIEW_STATES, DONE_REVIEW_STATES, REVISION_NOTE,
    V3_ANSWER_FORBIDDEN_RE,
)


def now():
    return datetime.now().isoformat(timespec="seconds")


def rows_by_id():
    return {r["id"]: (split, r) for split, r in io.read_rows()}


PILOT_ADJUSTMENTS_V1 = [
    "后缀实测 38-44 字符，120 字符上限无法容纳光学句+辅助句，故调整为 130 字符保守上限；仍不能证明真实 tokenizer 下不截断",
    "证据不足（极小目标、全黑光学图、低对比度）时不补写外观句，两版答案与原答案相同",
    "InSAR / DEM 单位未确认，辅助图无可可靠对应关系时统一记录『未提取可靠补充信息』",
]
PILOT_ADJUSTMENTS_V2 = [
    "按用户反馈删除『看不出差异』类否定式外观句（05811 退回原答案），并固化为禁止词",
    "辅助图从『一律记录未提取可靠补充信息』放宽为可核实的相对明暗/渐变/空间对应，由视觉模型逐样本认知生成",
    "光学外观在图片支持时允许最多 2 句具体内容，不得套模板或编造；证据不足仍留空",
    "InSAR 改为 0-255 编码值相对渐变认知；m/yr 单位与 20 mm/yr 活动阈值记为领域背景，不进训练答案",
    "MAX_CHARS 130→180、MAX_SENTENCES 4→5；真实 tokenizer 与 maximum_answer_tokens=160 / maximum_sequence_length=2048 的关系需 WSL --mode encode 验证，本机不能证明不截断",
]
PILOT_ADJUSTMENTS_V3 = [
    "用户明确选择 B 口径，授权在多图版答案写『形变较大 / 活动滑坡』定性判别；同步放开对应禁止词，但仍禁具体速率数值与单位换算",
    "只在 InSAR 有相对偏亮连片渐变区 + 光学轮廓可辨认 + DEM 有地形过渡时写活动判别；散斑/块状噪声或极小目标不写",
    "活动判别是用户授权的领域解释，本地 channel_schema.json 无法核实 |v|>20 mm/yr，已写入 ACTIVITY_JUDGEMENT_NOTE 与逐样本 uncertain",
    "MAX_CHARS 180→200；答案变长后更可能触碰 maximum_answer_tokens=160 与 maximum_sequence_length=2048，WSL --mode encode 增加答案 token 检查",
]
# v3 本轮规则变更（新增，不改 V1/V2/V3 历史）
PILOT_ADJUSTMENTS_V4 = [
    "目标从『增加外观句』改为『文字准确描述标注覆盖区』：全量逐张独立原图复核，不使用复核拼图宣称细节已核实",
    "保留 B 口径授权，去掉『禁止写活动性/未确认形变大小』两条要求；但数据不足时直接不写，不写『数据支持不足』类否定式元描述",
    "禁止把邻近/旁边区域外观写成目标外观；禁止整图泛泛介绍；辅助句必须对应目标所在位置",
    "禁止把散斑/量化/压缩块等图像处理痕迹当成地表特征或成因判断",
    "统一用『滑坡区域/目标区域』，禁止反复使用『标注区』",
    "新增 v3 答案全文禁止词（存量也查）；生成端与检查端共用 rules_current.json",
    "看图记录只能由真实 ViewImage 返回产生，四态结论，缺记录一律 not_viewed 不算完成",
    "先快照 + revisions.jsonl 追加留痕，全部检查通过后才统一写回三份 jsonl",
    "真实 tokenizer 输入长度检查（--mode encode）本轮未运行，只做代码静态审查",
]


PILOT_ADJUSTMENTS_V5 = [
    "2026-11-12：只复核533条新增描述，467条保留原答案且不冒充重新看图",
    "实际使用项目内初始视觉模型两遍读取独立图片，保留原始返回和图片身份；不记成ViewImage调用",
    "另用Codex ViewImage复查部分样本，如实区分两种观察来源",
    "InSAR只描述局部明暗、形态与目标内外对应，可写疑似形变异常线索",
    "不写活动滑坡、形变较大、具体速率或阈值；单位换算猜测留在核对记录",
    "颜色与目标像素明显矛盾、句子核对不对应或存在不确定推测时删句",
    "完整范围核对、两版编码检查和保存续接检查通过后再交付",
]


def build_rules():
    """规则单一来源：spec.py。init 与 sync-rules 都调用它，避免手写第二份。"""
    return {
        "status": "draft",
        "prompts": PROMPTS,
        "prompt_sha256": prompt_digests(),
        "max_chars": MAX_CHARS, "max_sentences": MAX_SENTENCES,
        "forbidden_patterns": spec.FORBIDDEN_PATTERNS,
        "v3_answer_forbidden_patterns": [p.pattern for p in V3_ANSWER_FORBIDDEN_RE],
        "aux_source_words": list(spec.AUX_SOURCE_WORDS),
        "review_states": list(REVIEW_STATES),
        "done_review_states": list(DONE_REVIEW_STATES),
        "active_rules_file": ACTIVE_RULES_FILE,
        "revision_note": REVISION_NOTE,
        "two_answer_rule": "answer=多图版, answer_optical=光学版, 共用程序生成数值句与后缀",
        "overlay_note": spec.OVERLAY_LEGEND,
        "input_rules": spec.INPUT_RULES,
        "insar_domain_note": INSAR_DOMAIN_NOTE,
        "aux_relative_rules": AUX_RELATIVE_RULES,
        "activity_judgement_note": ACTIVITY_JUDGEMENT_NOTE,
        "pilot_adjustments_v1": PILOT_ADJUSTMENTS_V1,
        "pilot_adjustments_v2": PILOT_ADJUSTMENTS_V2,
        "pilot_adjustments_v3": PILOT_ADJUSTMENTS_V3,
        "pilot_adjustments_v4": PILOT_ADJUSTMENTS_V4,
        "pilot_adjustments_v5": PILOT_ADJUSTMENTS_V5,
        "pilot_adjustments": PILOT_ADJUSTMENTS_V5,
    }


def cmd_sync_rules(args):
    """把当前 spec 规则同步到指定文件；默认本轮写 rules_current.json，不触碰历史 rules.json。"""
    target = V2 / (args.out or ACTIVE_RULES_FILE)
    rules = build_rules()
    if args.freeze:
        rules["status"] = "frozen"
        rules["frozen_at"] = now()
        rules["frozen_after"] = "2026-11-12 533条修订（用户已确认规则口径）"
        rules["frozen_prompt_sha256"] = prompt_digests()
    io.atomic_write_json(target, rules)
    print(f"PASS sync-rules: {target.name} status={rules['status']}, "
          f"max_chars={MAX_CHARS}, max_sentences={MAX_SENTENCES}, "
          f"forbidden_patterns={len(spec.FORBIDDEN_PATTERNS)}, prompt_sha256 已重算")


def cmd_snapshot(args):
    """阶段 0：生成完整原答案与上一轮两版答案快照（只写 snapshots/ 新文件）。"""
    m = revision.build_snapshots()
    revision.verify_snapshots()
    for name, meta in m["files"].items():
        print(f"  {name}: {meta['lines']} 行, sha256={meta['sha256'][:16]}...")
    print("PASS snapshot: 两份快照 + manifest 已生成并复核一致")


def cmd_freeze_rules(args):
    """试点检查全绿后冻结规则；冻结后不再用 val/test 成绩回调规则。"""
    checks = io.load_json(V2 / "checks.json")
    assert checks.get("all_passed"), "试点自动检查未全绿，禁止冻结"
    recs = io.load_records()
    pilot = json.loads((V2 / "pilot.json").read_text(encoding="utf-8"))["chosen"]
    bad = [c["id"] for c in pilot if recs[c["id"]]["status"] != "verified"]
    assert not bad, f"试点未全部 verified: {bad}"
    rules = io.load_json(V2 / "rules.json")
    assert rules.get("prompt_sha256") == prompt_digests(), "规则文件与 spec 不一致，请先 sync-rules"
    rules["status"] = "frozen"
    rules["frozen_at"] = now()
    rules["frozen_after"] = "pilot round v3（用户确认 B 口径）"
    rules["frozen_prompt_sha256"] = prompt_digests()
    io.atomic_write_json(V2 / "rules.json", rules)
    state = io.load_state()
    state["rules_frozen"] = True
    state["pilot_confirmed"] = True
    state["phase"] = "rules_frozen"
    io.save_state(state)
    print("PASS freeze-rules: rules.json status=frozen, "
          "state.rules_frozen=true, pilot_confirmed=true")


def cmd_init(args):
    rows = io.read_rows()
    ids = [r["id"] for _, r in rows]
    assert len(ids) == 1000, f"样本数不是 1000: {len(ids)}"
    assert len(set(ids)) == 1000, "id 有重复"
    records = []
    for idx, (split, r) in enumerate(rows):
        records.append({
            "id": r["id"], "source": r["source"], "split": split,
            "batch": idx // BATCH_SIZE, "pilot": False, "status": "pending",
            "view_checked_by": "visual_model", "view_model": VIEW_MODEL,
            "view_model_note": VIEW_MODEL_NOTE, "human_confirmed": False,
            "draft_model": None, "draft_prompt": None, "draft_is_fact": False,
            "view_completed": False, "images_viewed": [], "crops_viewed": [],
            "original_answer": r["answer"], "answer_optical": None, "answer_multiview": None,
            "evidence": {"from_label_computed": [], "from_optical": [], "from_auxiliary": [],
                         "auxiliary_reliable": False, "auxiliary_note": None},
            "view_spec": None, "uncertain": [], "needs_recheck": False,
            "length": None, "suffix_verified": False, "timestamps": {}, "error": None,
        })
    io.write_jsonl_atomic(io.RECORDS, records)

    batches = {
        "version": "desc_v2", "total": len(ids), "batch_size": BATCH_SIZE,
        "batches": [
            {"index": b, "ids": ids[b * BATCH_SIZE:(b + 1) * BATCH_SIZE],
             "status": "pending", "completed": 0, "failed": 0}
            for b in range(N_BATCHES)
        ],
    }
    io.save_batches(batches)
    io.save_state({
        "phase": "init", "rules_frozen": False, "pilot_confirmed": False,
        "last_batch": -1, "formal_training_run": False,
        "independent_evaluation_run": False, "appearance_accuracy_proven": False,
        "created_at": now(),
    })
    io.atomic_write_json(V2 / "rules.json", build_rules())
    io.atomic_write_json(V2 / "run_meta.json", {
        "python": sys.version.split()[0], "pillow": __import__("PIL").__version__,
        "numpy": __import__("numpy").__version__,
        "view_model": VIEW_MODEL,
        "view_model_note": "底层视觉模型标识未由工具暴露，标待核实",
        "generation": "内置视觉读取工具逐样本看图生成初稿，未经人工确认",
        "display_rules": "各通道 2%-98% 线性拉伸，无效像素置黑；叠加色品红仅标注显示",
        "input_size_rule": spec.INPUT_RULES["rule"],
        "scope_note": FIXED_SCOPE_NOTE,
        "created_at": now(),
    })
    for d in ("views", "logs"):
        (V2 / d).mkdir(parents=True, exist_ok=True)
    print(f"PASS init: {len(records)} 条骨架, {N_BATCHES} 批")


def cmd_verify(args):
    """只读式复核：不重写任何历史文件（originals_verified.json / fingerprints / samples_manifest）。

    v2 缺陷：rec['original_answer'] != r['answer'] 在 apply 之后必然失败（jsonl 已含外观句）。
    v3：把记录的 original_answer 与快照逐字比对；当前两版答案用前后缀断言，不做全等。
    """
    rows = io.read_rows()
    recs = io.load_records()
    assert len(rows) == 1000 and len(recs) == 1000, (len(rows), len(recs))
    snap = {x["id"]: x["original_answer"] for x in io.read_jsonl(revision.ORIGINAL_FULL)}
    failures = []
    for split, r in rows:
        rec = recs.get(r["id"])
        if not rec:
            failures.append((r["id"], "缺记录"))
            continue
        if rec.get("original_answer") != snap.get(r["id"]):
            failures.append((r["id"], "记录中的原答案与完整快照不一致"))
        vr = answer.verify_answer_pair(r)
        for e in vr["errors"]:
            failures.append((r["id"], e))
    assert not failures, f"原答案/两版答案校验失败: {failures[:10]} (共 {len(failures)})"

    # 历史指纹：存在则校验，不重写
    fp_path = V2 / "non_answer_fingerprints.json"
    if fp_path.exists():
        expected = io.load_json(fp_path)
        changed = [r["id"] for _, r in rows if io.row_fingerprint(r) != expected.get(r["id"])]
        assert not changed, f"非答案字段指纹变动: {changed[:5]}"
    # 明确不重写 originals_verified.json（11-07 历史）与 samples_manifest.json
    print(f"PASS verify: {len(rows)} 条原答案与快照逐字一致、两版答案前后缀与标注计算一致；"
          f"历史文件未改写")


def cmd_aux_spec(args):
    ranges = json.loads((DEST / "display_ranges.json").read_text(encoding="utf-8"))
    rows = io.read_rows()
    observed = {}
    for _, r in rows:
        observed.setdefault(r["source"], set()).add(len(r["images"]))
    for src, spec_views in VIEW_SPEC.items():
        assert observed[src] == {len(spec_views["views"])}, (src, observed[src])
        used_channels = {c for v in spec_views["views"] for c in v["channels"]}
        assert len(ranges[src]) > max(used_channels), (
            f"{src} 的显示范围条目不足，无法覆盖使用到的通道 {sorted(used_channels)}"
        )
    io.atomic_write_json(V2 / "aux_spec_verified.json", {
        "passed": True, "view_spec": VIEW_SPEC,
        "image_order": "辅助图在前，光学图最后；mask/valid 与最后一张光学图对齐",
        "display_ranges": ranges, "observed_view_counts": {k: sorted(v) for k, v in observed.items()},
        "not_inferable": [
            "InSAR 通道为 0-255 编码值，physical_unit / scale / offset / 零点 / 符号未确认：不写具体速率数值、单位换算、形变方向或位移量",
            "灰度偏亮不等于形变绝对值大；用户授权的 B 口径允许定性活动判别，但该结论不是由本地数据计算得出",
            "DEM 绝对高程基准未确认：不推断坡度、绝对高程",
            "landslide4sense view_0 是假彩：颜色不能当作地表真实颜色",
        ],
        "insar_domain_note": INSAR_DOMAIN_NOTE,
        "aux_relative_rules": AUX_RELATIVE_RULES,
        "activity_judgement_note": ACTIVITY_JUDGEMENT_NOTE,
        "verified_at": now(),
    })
    print("PASS aux-spec: 三来源视图构成与显示范围核实")


def _metrics(split, r):
    m, v = answer.load_mask_valid(r)
    target = m & v
    coverage = float(r["positive_pixels"] / r["valid_pixels"])
    blocks = answer.count_components8(target)
    optical = render._read_rgb(_ROOT / r["images"][-1])
    size = input_size(optical.size, True)
    gray = np.asarray(optical.resize(size, __import__("PIL").Image.Resampling.BICUBIC).convert("L"), dtype=float)
    from PIL import Image as _I
    t_in = np.array(_I.fromarray((target * 255).astype("uint8")).resize(size, _I.Resampling.NEAREST)) > 0
    v_in = np.array(_I.fromarray((v * 255).astype("uint8")).resize(size, _I.Resampling.NEAREST)) > 0
    inner, outer = gray[t_in], gray[v_in & ~t_in]
    delta = float(abs(inner.mean() - outer.mean())) if len(inner) and len(outer) else 999.0
    aux_std, black_frac = [], []
    for p in r["images"][:-1]:
        a = np.asarray(render._read_rgb(_ROOT / p).convert("L"), dtype=float)
        aux_std.append(float(a.std()))
        black_frac.append(float((np.asarray(render._read_rgb(_ROOT / p)).sum(axis=2) == 0).mean()))
    return {
        "id": r["id"], "source": r["source"], "coverage": coverage, "blocks": blocks,
        "contrast_delta": delta, "aux_std_mean": float(np.mean(aux_std)) if aux_std else None,
        "aux_black_frac_mean": float(np.mean(black_frac)) if black_frac else None,
        "optical_input_size": list(size), "n_aux": len(r["images"]) - 1,
    }


def cmd_select_pilot(args):
    import numpy as np
    globals()["np"] = np
    rows = [(s, r) for s, r in io.read_rows() if s == "train"]
    metrics = [_metrics(s, r) for s, r in rows]
    chosen, detail = [], []
    slots = ["small_target", "dispersed", "low_contrast", "aux_info_deficit"]
    for src in ("GDCLD", "landslide4sense", "multimodal-landslide-dataset"):
        pool = [x for x in metrics if x["source"] == src]
        for slot in slots:
            if slot == "small_target":
                key = lambda x: x["coverage"]
            elif slot == "dispersed":
                key = lambda x: -x["blocks"]
            elif slot == "low_contrast":
                key = lambda x: x["contrast_delta"]
            else:
                key = lambda x: (x["aux_std_mean"] if x["aux_std_mean"] is not None else 999.0)
            cand = sorted((x for x in pool if x["id"] not in {c["id"] for c in chosen}), key=key)
            if not cand:
                continue
            pick = dict(cand[0])
            pick["slot"] = slot
            if src == "GDCLD" and slot == "aux_info_deficit":
                pick["slot_note"] = "GDCLD 仅光学输入，天然属于辅助图信息不足代表"
            chosen.append(pick)
            detail.append(pick)
    assert len(chosen) == 12, len(chosen)
    io.atomic_write_json(V2 / "pilot.json", {
        "selection_rule": "只从 train 选；每来源 4 个槽位；指标由掩膜与图像只读计算；未使用 val/test 成绩、未训练、未预测",
        "chosen": chosen, "all_metrics_count": len(metrics), "created_at": now(),
    })
    recs = io.load_records()
    ids = {c["id"] for c in chosen}
    updated = []
    for rid in ids:
        rec = dict(recs[rid])
        rec["pilot"] = True
        rec["pilot_slot"] = next(c["slot"] for c in chosen if c["id"] == rid)
        updated.append(rec)
    io.append_records(updated)
    print("PASS select-pilot: 12 个试点")
    for c in chosen:
        print(f"  {c['source']:<32} {c['slot']:<20} {c['id']}")


def cmd_render(args):
    import numpy as np
    globals()["np"] = np
    rmap = rows_by_id()
    if args.pilot:
        ids = [c["id"] for c in json.loads((V2 / "pilot.json").read_text(encoding="utf-8"))["chosen"]]
    elif args.batch is not None:
        ids = io.load_batches()["batches"][args.batch]["ids"]
    elif args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        raise SystemExit("需指定 --pilot / --batch / --ids")
    for i, rid in enumerate(ids, 1):
        _, r = rmap[rid]
        result = render.render_sample(r)
        (V2 / "logs").mkdir(exist_ok=True)
        with open(V2 / "logs" / "render.log", "a", encoding="utf-8") as f:
            f.write(f"{now()} {rid} {result['view_spec']['native']} aux={len(result['aux_raw'])}\n")
        print(f"[{i}/{len(ids)}] {rid} -> {len(result['produced'])} 个产物")
    print(f"PASS render: {len(ids)} 个样本")


REQUIRED_ROLES = ("optical_raw", "label", "label_overlay", "optical_at_model_size")


def cmd_write_drafts(args):
    """v3：按四态结论写入 records.jsonl（追加），并同步追加 revisions.jsonl 修订流水。

    与 v2 的关键差别：
    - 数值句/后缀由 answer.answer_parts(row) 从 mask/valid 权威复算，不看当前 jsonl 的旧答案；
      不再用 split_original(当前答案) 求全等（那个缺陷会让二次修订必然失败）。
    - status 不再是闸门；review_state 四态 + view_evidence_complete 才是。
    - 缺逐图记录、review_state=not_viewed 的样本一律标 failed，绝不自动升级为已完成。
    """
    drafts = json.loads(Path(args.input).read_text(encoding="utf-8"))
    recs = io.load_records()
    rmap = rows_by_id()
    before = {x["id"]: x for x in io.read_jsonl(revision.V2_BEFORE)}
    updated, revisions = [], []
    for d in drafts:
        rid = d["id"]
        assert rid in recs and rid in rmap, f"未知 id: {rid}"
        rec = dict(recs[rid])
        _, row = rmap[rid]
        n_aux = len(row["images"]) - 1
        roles = {x.get("role") for x in d.get("viewed", [])}
        errors = []
        for role in REQUIRED_ROLES:
            if role not in roles:
                errors.append(f"未完成逐图检查，缺角色: {role}")
        for i in range(n_aux):
            if f"auxiliary_raw_{i}" not in roles:
                errors.append(f"未完成逐图检查，缺角色: auxiliary_raw_{i}")
        state = d.get("review_state", "not_viewed")
        if state not in DONE_REVIEW_STATES:
            errors.append(f"review_state={state}，缺逐图观察记录不得完成")
        if not d.get("view_evidence_complete"):
            errors.append("view_evidence_complete=false")

        # 权威复算：数值句与后缀来自标注，不来自旧答案文本
        numeric, suffix = answer.answer_parts(row)
        optical_sentences = d.get("optical_sentences", [])
        aux_sentences = d.get("aux_sentences", [])
        text_errors, optical, multiview = answer.check_new_text(
            numeric, suffix, optical_sentences, aux_sentences, n_aux > 0
        )
        errors.extend(text_errors)
        # 最终答案全文也必须不含 v3 新增禁止词（存量清理）
        for text, key in ((optical, "answer_optical"), (multiview, "answer_multiview")):
            for pat in V3_ANSWER_FORBIDDEN_RE:
                hit = pat.search(text)
                if hit:
                    errors.append(f"{key} 命中 v3 禁止词: {hit.group(0)}")
        pair = answer.verify_answer_pair({**row, "answer": multiview, "answer_optical": optical})
        if not pair["ok"]:
            errors.extend(pair["errors"])

        evidence = d.get("evidence", {})
        evidence.setdefault("from_label_computed", [
            "整体中心位置（九宫格质心）", "占有效图像面积比例", "连续或多块分布",
            "整体范围（外接框）", "bbox 坐标（归一化 0-1000）", "<|mask|> 区域标记",
        ])
        evidence.setdefault("from_optical", [])
        evidence.setdefault("from_auxiliary", [])
        evidence.setdefault("auxiliary_reliable", False)
        evidence.setdefault("auxiliary_note", OPTICAL_ONLY_NOTE)

        ok = not errors
        view_evidence_complete = bool(d.get("view_evidence_complete")) and ok
        final_state = state if ok and state in DONE_REVIEW_STATES else "not_viewed"
        rec.update({
            "status": "verified" if ok else "failed",   # 兼容字段，不作闸门
            "review_state": final_state,
            "view_evidence_complete": view_evidence_complete,
            "view_completed": view_evidence_complete,
            "view_round": d.get("round"),
            "review_conclusion": d.get("review_conclusion"),
            "view_checked_by": "visual_model", "view_model": VIEW_MODEL,
            "view_model_note": VIEW_MODEL_NOTE, "human_confirmed": False,
            "draft_model": d.get("draft_model", "tool:ViewImage"),
            "draft_prompt": {"optical": "PROMPT_OPTICAL", "auxiliary": "PROMPT_AUX",
                             "review": "PROMPT_REVIEW", "sha256": prompt_digests()},
            "draft_is_fact": False,
            "images_viewed": d.get("viewed", []), "crops_viewed": d.get("crops_viewed", []),
            "answer_optical": optical, "answer_multiview": multiview,
            "evidence": evidence,
            "view_spec": {"source": row["source"], "optical_index": OPTICAL_INDEX,
                          "views": VIEW_SPEC[row["source"]]["views"],
                          "input_size_rule": spec.INPUT_RULES["rule"]},
            "uncertain": d.get("uncertain", []),
            "needs_recheck": bool(d.get("needs_recheck", True)),
            "doubts": d.get("problems_found", []),
            "length": {"optical_chars": len(optical), "multiview_chars": len(multiview),
                       "suffix": suffix},
            "suffix_verified": pair["suffix_matches_mask"] and pair["mask_token"] and ok,
            "error": "; ".join(errors) if errors else None,
        })
        rec["timestamps"] = {**rec.get("timestamps", {}), "viewed": now(), "drafted": now(),
                             "verified": now() if ok else None}
        updated.append(rec)

        b = before.get(rid, {})
        b_opt, b_mul = b.get("answer_optical", rec.get("answer_optical")), b.get("answer", rec.get("answer_multiview"))
        revisions.append({
            "revision_id": f"v3-r{d.get('round')}-{rid}", "round": d.get("round"),
            "recorded_at": now(), "sample_id": rid, "split": rec.get("split"),
            "source": rec.get("source"), "batch": rec.get("batch"),
            "review_state": final_state,
            "before": {"answer_optical": b_opt, "answer_multiview": b_mul},
            "after": {"answer_optical": optical, "answer_multiview": multiview},
            "changed": (b_opt != optical) or (b_mul != multiview),
            "problems_found": d.get("problems_found", []),
            "images_actually_viewed": [
                {"path_abs": x.get("path") or x.get("path_abs"), "role": x.get("role"),
                 "step": x.get("step"), "tool": "tool:ViewImage"}
                for x in d.get("viewed", [])
            ],
            "sentence_basis": d.get("sentence_basis", []),
            "uncertain": d.get("uncertain", []),
            "review_conclusion": d.get("review_conclusion"),
            "view_tool": "tool:ViewImage", "human_confirmed": False, "draft_is_fact": False,
            "view_round": d.get("round"), "ok": ok,
        })
        print(f"  {rid}: {final_state}" + (f" | {rec['error']}" if errors else ""))
    io.append_records(updated)
    revision.append_revisions(revisions)
    print(f"PASS write-drafts: {len(updated)} 条（records + revisions 各追加），"
          f"完成三态={sum(r['review_state'] in DONE_REVIEW_STATES for r in updated)}, "
          f"not_viewed={sum(r['review_state']=='not_viewed' for r in updated)}")


def cmd_pilot_review(args):
    pilot = json.loads((V2 / "pilot.json").read_text(encoding="utf-8"))
    recs = io.load_records()
    lines = ["# 12 个试点样本新旧描述对照", "",
             "> 初稿由内置视觉读取工具（视觉模型）逐样本看图生成，未经人工逐图确认。", "",
             "| 来源 | 槽位 | 样本 id | 原答案 | 光学版新答案 | 多图版新答案 |",
             "|---|---|---|---|---|---|"]
    for c in pilot["chosen"]:
        r = recs[c["id"]]
        def esc(s):
            return (s or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {c['source']} | {c['slot']} | `{c['id']}` | {esc(r['original_answer'])} | "
                     f"{esc(r['answer_optical'])} | {esc(r['answer_multiview'])} |")
    lines += ["", "## 逐项复核清单（需人工/视觉核对）", "",
              "1. 是否真的增加了可确认的外观信息（对照 evidence.from_optical）。",
              "2. 是否把标注叠加色（品红）当成地表颜色。",
              "3. 是否凭空添加道路、河流、成因、时间、风险判断、地理方向或物理量。",
              "4. 是否与实际输入图片（接收尺寸 512 或 256）匹配。",
              "5. 是否保留正确的坐标与区域标记（suffix 逐字一致、恰 1 个框）。",
              "6. 光学版是否混入只由辅助图支持的信息。",
              "7. 是否套用统一套话填满、或出现大量重复新增句。",
              "8. 句数与字符上限、禁止词、否定式元描述、辅助信息不足记录是否正确。",
              "9. 辅助图认知是否只写相对明暗/渐变与空间对应，没有绝对物理量；活动判别是否只在多图版且有证据与授权留痕。",
              "10. view_checked_by / human_confirmed / view_model_note 是否如实。", "",
              "## 已冻结规则（v2，rules.status=frozen，用户已确认）", "",
              "- 删除『看不出差异』类否定式外观句，证据不足一律两版＝原答案。",
              "- 辅助图从『一律记录未提取可靠补充信息』放宽为可核实的相对明暗/渐变/空间对应。",
              "- 光学外观在图片支持时最多 2 句具体内容，不套模板、不编造。",
              "- InSAR 按 0-255 编码值相对渐变图解读，由视觉模型逐样本认知生成。",
              "- 用户明确选择 B 口径：授权在三图依据充分时写『形变较大 / 活动滑坡』定性判别；仍不写具体速率数值或单位换算。",
              "- 该活动判别是用户授权的领域解释，本地编码值文件无法核实 |v|>20 mm/yr，已逐样本记为不确定项。",
              "- 长度上限 130→200 字符、4→5 句；真实 tokenizer 与 160/2048 上限是否冲突需 WSL `--mode encode` 验证。", "",
              "## 自动检查结论", ""]
    ok = [c["id"] for c in pilot["chosen"] if recs[c["id"]]["status"] == "verified"]
    bad = [c["id"] for c in pilot["chosen"] if recs[c["id"]]["status"] != "verified"]
    lines += [f"- verified: {len(ok)}；failed/未完成: {len(bad)}",
              "- 自动检查只验证格式与部分一致性，不能据此声称所有外观描述准确。", ""]
    if bad:
        lines += ["未完成或失败样本：" + ", ".join(f"`{x}`" for x in bad), ""]
    io.atomic_write_text(V2 / "pilot_review.md", "\n".join(lines))
    print(f"PASS pilot-review: verified={len(ok)} failed={len(bad)} -> {V2 / 'pilot_review.md'}")


def apply_rows(originals, recs):
    """纯函数：从 recs 取两版答案组装新行，返回 new_rows（不写盘）。

    缺陷 1 修复：v2 在 k=='answer' 时写入新 answer_optical，随后循环到 jsonl 里
    原有的 'answer_optical' 键时又用 else 分支把旧值覆盖回去。v3 在遇到
    'answer' 时一次写入两版，并显式跳过原有的 'answer_optical' 键。
    """
    new_rows = []
    for r in originals:
        rec = recs[r["id"]]
        new = {}
        for k, val in r.items():
            if k == "answer":
                new["answer"] = rec["answer_multiview"]
                new["answer_optical"] = rec["answer_optical"]
            elif k == "answer_optical":
                continue  # 跳过旧值，防止覆盖新光学版
            else:
                new[k] = val
        assert "answer_optical" in new and new["answer_optical"], f"{r['id']}: 缺光学版"
        new_rows.append(new)
    assert [x["id"] for x in new_rows] == [x["id"] for x in originals]
    return new_rows


def validate_before_apply(rows, recs, fingerprints):
    """纯函数：写盘前置校验，返回错误列表（不写盘）。

    校验：1000 条 / id 唯一 / recs 覆盖 / 四态完成 / view_evidence_complete /
    suffix_verified / 两版存在且非空 / 权威复算前后缀与官方 bbox / 非答案指纹不变。
    """
    errors = []
    ids = [r["id"] for _, r in rows]
    if len(rows) != 1000:
        errors.append(("GLOBAL", "样本数不是 1000"))
    if len(set(ids)) != 1000:
        errors.append(("GLOBAL", "id 有重复"))
    if set(recs) != set(ids):
        errors.append(("GLOBAL", "records 与清单 id 集合不一致"))
    for split, r in rows:
        rid = r["id"]
        rec = recs.get(rid)
        if not rec:
            errors.append((rid, "缺记录"))
            continue
        if rec.get("review_state") not in DONE_REVIEW_STATES:
            errors.append((rid, f"review_state={rec.get('review_state')} 未完成"))
        if not rec.get("view_evidence_complete"):
            errors.append((rid, "view_evidence_complete=false"))
        if not rec.get("suffix_verified"):
            errors.append((rid, "suffix_verified=false"))
        opt, mul = rec.get("answer_optical"), rec.get("answer_multiview")
        if not opt or not mul:
            errors.append((rid, "两版答案为空或缺失"))
        else:
            for e in answer.assert_dual_row({**r, "answer": mul, "answer_optical": opt}):
                errors.append((rid, e))
        if io.row_fingerprint(r) != fingerprints.get(rid):
            errors.append((rid, "非答案字段已变"))
    return errors


def cmd_apply(args):
    """阶段 4：全量校验通过后才原子覆盖三份 jsonl。未满足任一闸门则一行不写。"""
    rules_path = V2 / (args.rules or ACTIVE_RULES_FILE)
    rules = io.load_json(rules_path)
    assert rules.get("status") == "frozen", "本轮规则未冻结"
    assert rules.get("prompt_sha256") == prompt_digests(), "本轮规则与 spec 不一致，请先 sync-rules"
    hist_state = io.load_state()
    assert hist_state.get("rules_frozen") and hist_state.get("pilot_confirmed"), "历史闸门未满足"
    revision.verify_snapshots()

    recs = io.load_records()
    rows = io.read_rows()
    fingerprints = io.load_json(V2 / "non_answer_fingerprints.json")
    problems = validate_before_apply(rows, recs, fingerprints)
    assert not problems, f"存在未完成/不一致样本，禁止覆盖: {problems[:5]} (共 {len(problems)})"

    report = {"applied_at": now(), "revision_round": "2026-11", "dry_run": bool(args.dry_run),
              "splits": {}, "old_sha256": {}, "new_sha256": {}}
    for split in ("train", "val", "test"):
        path = DEST / f"{split}.jsonl"
        report["old_sha256"][split] = io.sha256_file(path)
        originals = io.read_jsonl(path)
        new_rows = apply_rows(originals, recs)
        report["splits"][split] = len(new_rows)
        if args.dry_run:
            changed = sum(1 for o, n in zip(originals, new_rows)
                          if o["answer"] != n["answer"] or o.get("answer_optical") != n["answer_optical"])
            print(f"  [dry-run] {split}: {len(new_rows)} 行, 将改动 {changed} 行")
            continue
        io.write_jsonl_atomic(path, new_rows)
        report["new_sha256"][split] = io.sha256_file(path)
    report["invalidated"] = [
        "summary.json 的 manifest_sha256 记录旧字节哈希，本轮写回后由 stage4 更新",
        "check_domain_data.py:55 旧逐字 answer 断言已由本轮改为两版前后缀断言",
        "eval/domain.py 与 lock_domain_models.py 的 test.jsonl 旧哈希不可复现",
        "build_domain_review.py:48 的 reference 语义已按模型分版修复",
        "outputs/data-check.json 旧结论为 11-07 历史，不适用于新答案",
    ]
    report["fixed_this_round"] = [
        "apply_rows 显式跳过旧 answer_optical（缺陷1）",
        "verify_answer_pair 前后缀断言替代 split_original 全等（缺陷2）",
        "check_domain_data.py 两版断言（缺陷4）",
        "build_domain_review.py rgb 取 answer_optical（缺陷6）",
        "select_answer 三分支空值报错（缺陷7）",
    ]
    out_report = V2 / (args.out_report or "apply_report_revision_2026-11.json")
    if args.dry_run:
        print(f"PASS apply --dry-run: 未写回任何 jsonl；三份清单仍为基线")
    else:
        io.atomic_write_json(out_report, report)
        io.save_state({
            "phase": "applied_revision_2026_11", "revision_round": "2026-11",
            "rules_file": rules_path.name, "checks_file": "checks_revision_2026-11.json",
            "review_states": list(REVIEW_STATES),
            "loader_encode_smoke_run": False, "appearance_accuracy_proven": False,
            "formal_training_run": False, "independent_evaluation_run": False,
            "human_confirmed": False, "applied_at": now(),
        }, path=revision.STATE_REVISION)
        print(f"PASS apply: 三份 jsonl 已原子覆盖 {report['splits']} -> {out_report.name}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sr = sub.add_parser("sync-rules")
    sr.add_argument("--out")            # 默认写 rules_current.json，不触碰历史 rules.json
    sr.add_argument("--freeze", action="store_true")
    sub.add_parser("freeze-rules")
    sub.add_parser("snapshot")          # 阶段 0：完整原答案 + v2 两版快照
    v = sub.add_parser("verify")
    v.add_argument("--skip-sample-hashes", action="store_true")
    sub.add_parser("aux-spec")
    sub.add_parser("select-pilot")
    r = sub.add_parser("render")
    r.add_argument("--pilot", action="store_true")
    r.add_argument("--batch", type=int)
    r.add_argument("--ids")
    w = sub.add_parser("write-drafts")
    w.add_argument("--input", required=True)
    sub.add_parser("pilot-review")
    ap = sub.add_parser("apply")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rules")
    ap.add_argument("--state-out")
    ap.add_argument("--out-report")
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
