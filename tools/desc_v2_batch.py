"""描述改进 v2：全量分批看图流水线（断点续跑）。

设计要点：
- 每个样本一张「复核行」，把必须实际查看的面板按固定列序拼在一起；
  每张 sheet 放若干行，行号与 id 的映射写入 index.json，避免靠文件顺序对应。
- 每批的客观指标（掩膜像素、光学均值/方差、辅助图方差/黑像素比例）由程序计算，
  作为逐样本记录的证据，不是套话。
- 视觉模型只在 observation 文件里给出「确实看得见」的句子；看不清的样本不写外观句。
- 生成的 drafts 文件交给 desc_v2.py write-drafts 走同一套校验，不绕过任何闸门。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xiaomi_lens.description_v2 import io, revision  # noqa: E402
from xiaomi_lens.description_v2.spec import (  # noqa: E402
    V2, DEST, ROOT as PROJECT_ROOT, AUX_SOURCE_WORDS, REVIEW_STATES, DONE_REVIEW_STATES,
)

PANEL = 200
LABEL_W = 190
HEADER_H = 30
COLUMNS = 8
COL_NAMES = ("opt raw", "mask", "overlay", "opt@input",
             "aux0 raw", "aux0@in", "aux1 raw", "aux1@in")
# 按来源分组：GDCLD 无辅助图，只需 4 列；来源一致时列数更少、下采样更轻。
SOURCE_ROWS = {"GDCLD": 9, "landslide4sense": 8, "multimodal-landslide-dataset": 7}
SOURCE_COLS = {"GDCLD": 4, "landslide4sense": 6, "multimodal-landslide-dataset": 8}
SHEETS = V2 / "sheets"
OBS = V2 / "obs"


def _rows():
    return {r["id"]: (split, r) for split, r in io.read_rows()}


def _pil():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    return np, Image, ImageDraw, ImageFont


def _font(ImageFont, bold=False):
    for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, 13 if not bold else 17)
            except Exception:
                pass
    return ImageFont.load_default()


def _panel(path, Image, size, nearest=False):
    if path is None or not Path(path).exists():
        im = Image.new("RGB", (size, size), (38, 38, 42))
        from PIL import ImageDraw
        ImageDraw.Draw(im).text((size // 2 - 22, size // 2 - 6), "NO AUX", fill=(150, 150, 155))
        return im
    im = Image.open(path).convert("RGB")
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    return im.resize((size, size), resample)


def _view_paths(rid, row):
    d = V2 / "views" / rid
    n_aux = len(row["images"]) - 1
    cols = [
        PROJECT_ROOT / row["images"][-1],   # 无覆盖光学原图
        PROJECT_ROOT / row["mask"],         # 滑坡标注
        d / "overlay.png",                  # 标注叠加对照
        d / "optical_at_input.png",         # 模型接收尺寸光学图
    ]
    nearest = [False, True, False, False]
    for i in range(n_aux):
        cols += [PROJECT_ROOT / row["images"][i], d / f"aux_{i}_at_input.png"]
        nearest += [False, False]
    while len(cols) < COLUMNS:
        cols.append(None)
        nearest.append(False)
    return cols[:COLUMNS], nearest[:COLUMNS], n_aux


def cmd_sheets(a):
    np, Image, ImageDraw, ImageFont = _pil()
    rmap = _rows()
    batch = io.load_batches()["batches"][a.batch]
    ids = list(batch["ids"])
    out = SHEETS / f"batch{a.batch}"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("sheet_*.png"):
        stale.unlink()
    index = {"batch": a.batch, "grouped_by_source": True, "sheets": []}
    k = 0
    for source in SOURCE_ROWS:
        sids = [x for x in ids if rmap[x][1]["source"] == source]
        per, ncol = SOURCE_ROWS[source], SOURCE_COLS[source]
        for start in range(0, len(sids), per):
            group = sids[start:start + per]
            w = LABEL_W + ncol * PANEL
            h = HEADER_H + len(group) * PANEL
            sheet = Image.new("RGB", (w, h), (24, 24, 28))
            draw = ImageDraw.Draw(sheet)
            f, fb = _font(ImageFont), _font(ImageFont, True)
            for c, name in enumerate(COL_NAMES[:ncol]):
                draw.text((LABEL_W + c * PANEL + 6, 8), name, fill=(235, 235, 240), font=f)
            for r, rid in enumerate(group):
                _, row = rmap[rid]
                cols, nearest, _ = _view_paths(rid, row)
                y0 = HEADER_H + r * PANEL
                draw.rectangle((0, y0, w - 1, y0 + PANEL - 1), outline=(70, 70, 78), width=1)
                draw.text((6, y0 + 6), f"ROW {r + 1}", fill=(255, 214, 92), font=fb)
                draw.text((6, y0 + 28), rid[:24], fill=(190, 195, 205), font=f)
                draw.text((6, y0 + 46), rid[24:48], fill=(190, 195, 205), font=f)
                draw.text((6, y0 + 64), source[:14], fill=(140, 200, 255), font=f)
                for c, (p, near) in enumerate(zip(cols[:ncol], nearest[:ncol])):
                    sheet.paste(_panel(p, Image, PANEL, near), (LABEL_W + c * PANEL, y0))
            name = f"sheet_{k:03d}.png"
            sheet.save(out / name)
            index["sheets"].append({"file": name, "rows": group, "source": source,
                                    "columns": ncol})
            k += 1
    io.atomic_write_json(out / "index.json", index)
    print(f"PASS sheets: batch{a.batch} {len(ids)} 个样本 -> {k} 张 sheet")


def _metrics(rid, row):
    import numpy as np
    from PIL import Image
    opt = np.array(Image.open(PROJECT_ROOT / row["images"][-1]).convert("L"))
    m = np.array(Image.open(PROJECT_ROOT / row["mask"]).convert("L")) > 127
    v = np.array(Image.open(PROJECT_ROOT / row["valid"]).convert("L")) > 127
    aux = []
    for i in range(len(row["images"]) - 1):
        p = V2 / "views" / rid / f"aux_{i}_at_input.png"
        if p.exists():
            z = np.array(Image.open(p).convert("L"))
            aux.append({"index": i, "mean": round(float(z.mean()), 1),
                        "std": round(float(z.std()), 1),
                        "black_ratio": round(float((z == 0).mean()), 3)})
    return {
        "positive_pixels": int(m.sum()), "valid_pixels": int(v.sum()),
        "coverage_pct": round(float(m.sum() / max(1, v.sum()) * 100), 2),
        "optical_mean": round(float(opt.mean()), 1), "optical_std": round(float(opt.std()), 1),
        "aux": aux,
    }


def cmd_metrics(a):
    rmap = _rows()
    OBS.mkdir(exist_ok=True)
    ids = io.load_batches()["batches"][a.batch]["ids"]
    data = {rid: _metrics(rid, rmap[rid][1]) for rid in ids}
    io.atomic_write_json(OBS / f"batch{a.batch}_metrics.json", data)
    print(f"PASS metrics: batch{a.batch} {len(data)} 条客观指标")


AUX_DISPLAY = {
    "landslide4sense": ["近红外+双短波红外假彩合成（颜色非地表真实色）"],
    "multimodal-landslide-dataset": ["InSAR 编码值灰度渐变（非物理量）", "DEM 灰度（非地表真实色）"],
}
AUTH_NOTE = ("该活动滑坡判别由用户明确授权（B 口径）；本地 channel_schema.json 无法核实 |v|>20 mm/yr，"
             "灰度偏亮不等于形变绝对值大，结论不是由本地数据计算得出，也未经人工确认。")

# 5 步复核的角色 → (文件位置, 说明)。s=SAMPLES, v=VIEWS
# step 1 无覆盖光学原图；step 2 掩膜+叠加确认位置；step 3 回到原图只核对目标区；
# step 4 接收尺寸图；step 5 全部辅助图原图 + 接收尺寸图。


def _sample_images(rid, row):
    """返回该样本 5 步复核所需的独立原图清单（绝对路径 + 角色 + step）。"""
    s = DEST / "samples" / rid
    v = V2 / "views" / rid
    n = len(row["images"])
    items = [
        {"path_abs": str(s / f"view_{n - 1}.png"), "role": "optical_raw", "step": 1,
         "display": "无覆盖真彩光学原图（理解整图）"},
        {"path_abs": str(s / "mask.png"), "role": "label", "step": 2,
         "display": "二值掩膜（确认目标位置）"},
        {"path_abs": str(v / "overlay.png"), "role": "label_overlay", "step": 2,
         "display": "GT 叠加，品红仅标注显示色"},
        {"path_abs": str(s / f"view_{n - 1}.png"), "role": "optical_raw_recheck", "step": 3,
         "display": "回到无覆盖原图，只核对掩膜覆盖区外观"},
        {"path_abs": str(v / "optical_at_input.png"), "role": "optical_at_model_size", "step": 4,
         "display": "模型接收尺寸光学图（确认细节仍可见）"},
    ]
    labels = AUX_DISPLAY.get(row["source"], ["辅助图"] * (n - 1))
    for i in range(n - 1):
        items.append({"path_abs": str(s / f"view_{i}.png"), "role": f"auxiliary_raw_{i}", "step": 5,
                      "display": labels[i] + " 原图"})
        items.append({"path_abs": str(v / f"aux_{i}_at_input.png"),
                      "role": f"auxiliary_at_model_size_{i}", "step": 5,
                      "display": labels[i] + " 接收尺寸"})
    for it in items:
        it["sample_id"] = rid
        p = Path(it["path_abs"])
        it["exists"] = p.exists()
        it["image_sha256"] = io.sha256_file(p) if p.exists() else None
        it["est_bytes"] = p.stat().st_size if p.exists() else 0
    return items


def cmd_plan_rounds(a):
    """生成本轮待看样本的独立原图绝对路径清单（不使用任何拼图）。"""
    rmap = _rows()
    prog = revision.load_progress()
    viewed = set(prog.get("viewed_ids", []))
    ids_all = [r["id"] for _, r in io.read_rows()]
    remaining = [x for x in ids_all if x not in viewed]
    # 按来源混排（GDCLD 1024² 与 128² 来源交替），避免单轮全是大图
    buckets = {}
    for rid in remaining:
        buckets.setdefault(rmap[rid][1]["source"], []).append(rid)
    src_order = ["GDCLD", "landslide4sense", "multimodal-landslide-dataset"]
    picked, pools = [], {k: list(v) for k, v in buckets.items()}
    while len(picked) < a.size and any(pools.values()):
        for src in src_order:
            if pools.get(src) and len(picked) < a.size:
                picked.append(pools[src].pop(0))
    images = []
    for rid in picked:
        images.extend(_sample_images(rid, rmap[rid][1]))
    plan = {"round": a.round, "ids": picked, "images": images,
            "est_total_bytes": sum(x["est_bytes"] for x in images),
            "view_mode": "per_sample_original_images_only",
            "note": "逐张独立原图，禁止用复核拼图宣称细节已核实；缺真实 ViewImage 记录不算完成"}
    io.atomic_write_json(revision.round_plan_path(a.round), plan)
    print(f"PASS plan-rounds: round{a.round:03d} {len(picked)} 条样本 / {len(images)} 张独立原图 "
          f"/ 约 {plan['est_total_bytes']/1024/1024:.1f} MB")
    for rid in picked:
        print("  ", rid, rmap[rid][1]["source"])


def cmd_record_views(a):
    """把本轮【真实 ViewImage 返回】的图片登记进 roundNNN_views.jsonl。

    调用方必须逐张实际看图后再调用；本函数只登记传入的条目，绝不批量补写「已查看」。
    输入 JSON 为 [{sample_id, path_abs, role, step}] 列表，哈希由磁盘实算。
    """
    entries = json.loads(Path(a.input).read_text(encoding="utf-8"))
    plan = io.load_json(revision.round_plan_path(a.round))
    planned = {(x["sample_id"], x["role"]): x for x in plan["images"]}
    checked = []
    for e in entries:
        key = (e["sample_id"], e["role"])
        assert key in planned, f"本轮计划外的看图条目: {key}"
        p = Path(e["path_abs"])
        assert p.exists() and p.resolve() == Path(planned[key]["path_abs"]).resolve(), \
            f"路径与计划不符: {e['path_abs']}"
        checked.append({"sample_id": e["sample_id"], "image_path_abs": str(p),
                        "image_sha256": io.sha256_file(p), "role": e["role"],
                        "step": planned[key]["step"], "note": e.get("note")})
    revision.append_views(a.round, checked)
    print(f"PASS record-views: round{a.round:03d} 登记 {len(checked)} 张真实看图记录")


def cmd_drafts(a):
    """把本轮【真实看图记录 + 视觉模型观察结论】展开成 write-drafts 输入。

    缺陷 3 修复：viewed 只来自 roundNNN_views.jsonl；无记录的样本一律跳过
    （保持 not_viewed，绝不生成『已逐行查看/证据不足』元描述）。
    """
    rmap = _rows()
    recs = io.load_records()
    plan = io.load_json(revision.round_plan_path(a.round))
    views = revision.read_views(a.round)
    obs = json.loads(Path(a.obs).read_text(encoding="utf-8")) if a.obs and Path(a.obs).exists() else {}
    by_sample = {}
    for v in views:
        by_sample.setdefault(v["sample_id"], []).append(v)
    out, skipped = [], []
    for rid in plan["ids"]:
        vlist = by_sample.get(rid, [])
        if not vlist:
            skipped.append(rid)   # 缺真实看图记录 -> 保持 not_viewed，不生成草稿
            continue
        _, row = rmap[rid]
        n_aux = len(row["images"]) - 1
        o = obs.get(rid, {})
        state = o.get("review_state", "viewed_keep")
        assert state in REVIEW_STATES, f"{rid}: 非法 review_state {state}"
        # 证据完整性：5 步必需角色齐全 + 全部 returned + 哈希与磁盘一致
        roles = {v["role"] for v in vlist}
        need = {"optical_raw", "label", "label_overlay", "optical_at_model_size",
                "optical_raw_recheck"} | {f"auxiliary_raw_{i}" for i in range(n_aux)}
        evidence_complete = need.issubset(roles) and all(v.get("returned") for v in vlist) and all(
            Path(v["image_path_abs"]).exists()
            and io.sha256_file(Path(v["image_path_abs"])) == v["image_sha256"] for v in vlist)
        viewed = [{"path": v["image_path_abs"], "path_abs": v["image_path_abs"],
                   "role": v["role"], "step": v["step"],
                   "display": v.get("note") or v["role"],
                   "reviewed_via": f"round{a.round:03d}_views.jsonl 真实 ViewImage 记录"}
                  for v in vlist]
        opt, aux = o.get("optical_sentences", []), o.get("aux_sentences", [])
        m = o.get("objective_metrics", {})
        out.append({
            "id": rid, "round": a.round, "view_mode": "per_sample_original_images_only",
            "review_state": state, "view_evidence_complete": evidence_complete,
            "review_conclusion": o.get("review_conclusion"),
            "viewed": viewed, "crops_viewed": [],
            "optical_sentences": opt, "aux_sentences": aux,
            "problems_found": o.get("problems_found", []),
            "sentence_basis": o.get("sentence_basis", []),
            "evidence": {
                "from_label_computed": [
                    "整体中心位置（九宫格质心）", "占有效图像面积比例", "连续或多块分布",
                    "整体范围（外接框）", "bbox 坐标（归一化 0-1000）", "<|mask|> 区域标记",
                ],
                "from_optical": o.get("from_optical", []),
                "from_auxiliary": o.get("from_auxiliary", []),
                "auxiliary_reliable": bool(aux),
                "auxiliary_note": o.get("auxiliary_note", "该位置无可可靠对应则不写辅助句"),
                "objective_metrics": m,
                "reviewed_via": f"round{a.round:03d}_views.jsonl 真实逐图记录",
            },
            "uncertain": o.get("uncertain", ["视觉模型初稿，未经人工逐图确认"]),
            "needs_recheck": True,
        })
    dest = revision.round_obs_path(a.round).with_name(f"round{a.round:03d}_drafts.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS drafts: round{a.round:03d} -> {len(out)} 条草稿，跳过（缺看图记录）{len(skipped)} 条")
    if skipped:
        print("  未查看（仍 not_viewed）:", skipped)


def cmd_check_round(a):
    """校验本轮：记录完整 + 四态完成 + 更新 view_progress.json（断点续跑）。"""
    plan = io.load_json(revision.round_plan_path(a.round))
    ok, problems = revision.verify_round_evidence(a.round, plan)
    recs = io.load_records()
    not_done = [rid for rid in plan["ids"]
                if recs.get(rid, {}).get("review_state") not in DONE_REVIEW_STATES
                or not recs.get(rid, {}).get("view_evidence_complete")]
    print(f"round{a.round:03d}: 计划 {len(plan['ids'])} 条 | 证据问题 {len(problems)} | 未完成 {len(not_done)}")
    for p in problems[:5]:
        print("  证据:", p)
    if not_done:
        print("  未完成前 10:", not_done[:10])
    assert ok and not not_done, "本轮存在未完成样本，round_completed=false，整轮需重看"
    prog = revision.load_progress()
    for rid in plan["ids"]:
        if rid not in prog["viewed_ids"]:
            prog["viewed_ids"].append(rid)
    prog["completed_rounds"] = prog.get("completed_rounds", 0) + 1
    prog["next_round"] = a.round + 1
    prog["rounds"].append({"round": a.round, "ids": plan["ids"],
                           "planned_images": len(plan["images"]),
                           "viewed_images": len(revision.read_views(a.round)),
                           "round_completed": True, "recorded_at": revision.now()})
    counts = {k: 0 for k in REVIEW_STATES}
    for rid in prog["viewed_ids"]:
        counts[recs.get(rid, {}).get("review_state", "not_viewed")] += 1
    prog["review_counts"] = counts
    prog["revisions_lines"] = sum(1 for _ in open(revision.REVISIONS, encoding="utf-8")) \
        if revision.REVISIONS.exists() else 0
    revision.save_progress(prog)
    print(f"PASS check-round: round{a.round:03d} 完成 | 累计已查看 {len(prog['viewed_ids'])}/1000"
          f" | 四态 {counts}")


def cmd_check(a):
    """兼容旧调用：打印全部 1000 条四态进度，缺记录不得通过。"""
    recs = io.load_records()
    prog = revision.load_progress()
    not_done = [rid for rid, r in recs.items()
                if r.get("review_state") not in DONE_REVIEW_STATES
                or not r.get("view_evidence_complete")]
    print(f"全量: 记录 {len(recs)} | 已完成三态 {len(recs) - len(not_done)} | 未完成 {len(not_done)}"
          f" | 已查看 id {len(prog.get('viewed_ids', []))}/1000")
    if not_done:
        print("  未完成前 10:", not_done[:10])
    assert len(prog.get("viewed_ids", [])) == 1000, "未看完 1000 条，禁止判为完成"
    assert not not_done, "存在未完成样本"
    print("PASS check: 1000 条全部完成四态复核")


def cmd_render_missing(a):
    import subprocess
    rmap = _rows()
    missing = [rid for rid, _ in rmap.items()
               if not (V2 / "views" / rid / "contact_sheet.png").exists()]
    print(f"待渲染 {len(missing)} 个样本")
    for i in range(0, len(missing), 100):
        chunk = ",".join(missing[i:i + 100])
        subprocess.run([sys.executable, "tools/desc_v2.py", "render", "--ids", chunk],
                       cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL)
        print(f"  已渲染 {min(i + 100, len(missing))}/{len(missing)}")
    print("PASS render-missing")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    # 历史流水线（本轮禁用 sheets，但保留代码）
    for name in ("sheets", "metrics", "render-missing"):
        s = sub.add_parser(name)
        if name != "render-missing":
            s.add_argument("--batch", type=int, required=True)
        if name == "sheets":
            s.add_argument("--only-pending", action="store_true")
    # v3 逐图复核流水线
    pr = sub.add_parser("plan-rounds")
    pr.add_argument("--round", type=int, required=True)
    pr.add_argument("--size", type=int, default=8)
    rv = sub.add_parser("record-views")
    rv.add_argument("--round", type=int, required=True)
    rv.add_argument("--input", required=True)
    dr = sub.add_parser("drafts")
    dr.add_argument("--round", type=int, required=True)
    dr.add_argument("--obs")
    cr = sub.add_parser("check-round")
    cr.add_argument("--round", type=int, required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--batch", type=int, default=0)
    a = p.parse_args()
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
