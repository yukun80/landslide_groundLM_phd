"""v3 修订流水线的共享状态与记录工具。

被现有 tools/desc_v2.py、tools/desc_v2_batch.py 复用，不是独立工具链：
只负责快照、四态、view_progress、round 逐图记录、revisions.jsonl 的读写。
看图事实本身由调用方（视觉模型复核）提供，本模块不臆造任何「已查看」。
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from . import io
from .spec import V2, DEST, ROOT, SPLITS

# ---- 本轮文件名（一律新建，不覆盖 11-07~11-09 的历史文件） ----
SNAP_DIR = V2 / "snapshots"
ORIGINAL_FULL = SNAP_DIR / "original_answers_full.jsonl"
V2_BEFORE = SNAP_DIR / "v2_answers_before_revision.jsonl"
SNAP_MANIFEST = SNAP_DIR / "snapshots_manifest.json"

REVISIONS = V2 / "revisions.jsonl"
VIEW_PROGRESS = V2 / "view_progress.json"
RULES_CURRENT = V2 / "rules_current.json"
STATE_REVISION = V2 / "state_revision.json"
PILOT_REVISION = V2 / "pilot_revision_2026-11.json"
TOOL_FIX_TESTS = V2 / "tool_fix_tests.json"
SELFTEST_DIR = V2 / "selftest"

OBS = V2 / "obs"


def now():
    return datetime.now().isoformat(timespec="seconds")


# 四态复核结论。缺记录一律 not_viewed，不得自动升级。
REVIEW_STATES = ("not_viewed", "viewed_no_new_info", "viewed_needs_change", "viewed_keep")
DONE_STATES = ("viewed_no_new_info", "viewed_needs_change", "viewed_keep")


def build_snapshots():
    """阶段 0：在任何写操作之前生成完整原答案与上一轮两版答案快照。

    完整原答案取自 records.jsonl 中每条 id 最早一条带 original_answer 的记录。
    只读现有文件，只写 snapshots/ 下三个新文件。
    """
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    earliest = {}
    for line_no, line in enumerate(open(io.RECORDS, encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r["id"]
        if rid not in earliest and r.get("original_answer"):
            earliest[rid] = {
                "id": rid, "source": r.get("source"), "split": r.get("split"),
                "original_answer": r["original_answer"],
                "original_answer_sha256": hashlib.sha256(
                    r["original_answer"].encode("utf-8")
                ).hexdigest(),
                "source_records_line": line_no, "generated_at": now(),
            }
    assert len(earliest) == 1000, f"完整原答案副本不足 1000: {len(earliest)}"
    # 按 train/val/test 原始行顺序输出，便于逐行对照
    orig_rows = [earliest[r["id"]] for _, r in io.read_rows()]
    io.write_jsonl_atomic(ORIGINAL_FULL, orig_rows)

    v2_rows = []
    for split, r in io.read_rows():
        v2_rows.append({
            "id": r["id"], "split": split, "source": r["source"],
            "answer": r["answer"], "answer_optical": r["answer_optical"],
            "answer_sha256": hashlib.sha256(r["answer"].encode("utf-8")).hexdigest(),
            "answer_optical_sha256": hashlib.sha256(r["answer_optical"].encode("utf-8")).hexdigest(),
        })
    io.write_jsonl_atomic(V2_BEFORE, v2_rows)

    def meta(p):
        return {"size": p.stat().st_size, "sha256": io.sha256_file(p),
                "lines": len(io.read_jsonl(p))}

    manifest = {
        "generated_at": now(), "read_only": True,
        "note": "完整原答案唯一副本来自 records.jsonl.original_answer；"
                "originals_verified.json 只有检查结论，不是备份。本目录只读留存。",
        "files": {"original_answers_full.jsonl": meta(ORIGINAL_FULL),
                  "v2_answers_before_revision.jsonl": meta(V2_BEFORE)},
        "sources": {
            "records.jsonl": {"sha256": io.sha256_file(io.RECORDS),
                              "lines": sum(1 for _ in open(io.RECORDS, encoding="utf-8"))},
            **{f"{s}.jsonl": {"sha256": io.sha256_file(DEST / f"{s}.jsonl")} for s in SPLITS},
        },
    }
    io.atomic_write_json(SNAP_MANIFEST, manifest)
    return manifest


def verify_snapshots():
    """复核两份快照仍与 manifest 一致（阶段 4 前置）。"""
    m = io.load_json(SNAP_MANIFEST) if SNAP_MANIFEST.exists() else None
    assert m, "缺少 snapshots_manifest.json"
    for name, meta in m["files"].items():
        p = SNAP_DIR / name
        assert p.stat().st_size == meta["size"] and io.sha256_file(p) == meta["sha256"], \
            f"快照被改动: {name}"
    return m


# ---- 断点续跑进度 ----
def new_progress(total_rounds_estimate=None):
    return {
        "created_at": now(), "revision_round": "2026-11",
        "view_mode": "per_sample_original_images_only",
        "montage_used_for_detail_verification": False,
        "total_samples": 1000, "completed_rounds": 0,
        "total_rounds_estimate": total_rounds_estimate,
        "viewed_ids": [], "next_round": 0,
        "rounds": [],
        "review_counts": {k: 0 for k in REVIEW_STATES},
        "revisions_lines": 0, "records_lines_appended": 0,
    }


def load_progress():
    return io.load_json(VIEW_PROGRESS) if VIEW_PROGRESS.exists() else new_progress()


def save_progress(p):
    io.atomic_write_json(VIEW_PROGRESS, p)


# ---- 每轮逐图查看记录（只能由真实 ViewImage 返回产生） ----
def round_views_path(round_no):
    return OBS / f"round{round_no:03d}_views.jsonl"


def round_plan_path(round_no):
    return OBS / f"round{round_no:03d}_plan.json"


def round_obs_path(round_no):
    return OBS / f"round{round_no:03d}_observations.json"


def append_views(round_no, entries):
    """entries: [{sample_id, image_path_abs, image_sha256, role, step}]。

    调用方必须只在 ViewImage 真实返回后调用；本函数不补写任何「已查看」。
    """
    records = []
    for seq, e in enumerate(entries, 1):
        records.append({
            "round": round_no, "seq": seq,
            "sample_id": e["sample_id"], "image_path_abs": e["image_path_abs"],
            "image_sha256": e["image_sha256"], "role": e["role"], "step": e["step"],
            "tool": "tool:ViewImage", "observed_at": now(), "returned": True,
            "note": e.get("note"),
        })
    io.append_jsonl(round_views_path(round_no), records)


def read_views(round_no):
    p = round_views_path(round_no)
    return io.read_jsonl(p) if p.exists() else []


def verify_round_evidence(round_no, plan):
    """校验本轮逐图记录：图片 sha256 与磁盘一致、returned、角色与计划相符。

    返回 (ok, problems)。任一不符 -> 整轮不计完成。
    """
    views = read_views(round_no)
    problems = []
    planned = {}
    for item in plan.get("images", []):
        planned.setdefault(item["sample_id"], []).append(item)
    seen = {}
    for v in views:
        if not v.get("returned") or v.get("tool") != "tool:ViewImage":
            problems.append(f"round{round_no}: 非真实看图记录 {v.get('seq')}")
            continue
        p = Path(v["image_path_abs"])
        if not p.exists():
            problems.append(f"{v['sample_id']}: 图片不存在 {v['image_path_abs']}")
            continue
        if io.sha256_file(p) != v["image_sha256"]:
            problems.append(f"{v['sample_id']}: 图片哈希不符 {p.name}")
        seen.setdefault(v["sample_id"], set()).add(v["role"])
    for sid, items in planned.items():
        need = {i["role"] for i in items}
        got = seen.get(sid, set())
        missing = need - got
        if missing:
            problems.append(f"{sid}: 缺看图角色 {sorted(missing)}")
    return (not problems), problems


# ---- revisions.jsonl（追加式修订流水） ----
def append_revisions(records):
    io.append_jsonl(REVISIONS, records)


def load_revisions():
    out = {}
    if REVISIONS.exists():
        for line in open(REVISIONS, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out
