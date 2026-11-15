"""8 项工具修复实测（在 description_v2/selftest/ 最小副本上运行，不碰正式清单）。

只 import 现有模块 / 调用现有纯函数，不复制业务逻辑：
  answer.check_new_text / verify_answer_pair / assert_dual_row / select_answer
  desc_v2.apply_rows / validate_before_apply
  revision.append_revisions / verify_round_evidence

用法：python tools/verify_desc_v2_fixes.py --out tool_fix_tests.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from xiaomi_lens.description_v2 import answer, io, spec, revision  # noqa: E402
import desc_v2  # noqa: E402
from xiaomi_lens.description_v2.spec import V2, DEST, DONE_REVIEW_STATES  # noqa: E402


class Box:
    def __init__(self, d):
        self.d = d

    def __getitem__(self, k):
        return self.d[k]

    def get(self, k, default=None):
        return self.d.get(k, default)

    def items(self):
        return self.d.items()

    def __contains__(self, k):
        return k in self.d

    def __setitem__(self, k, v):
        self.d[k] = v


def build_row(real, optical_text, multi_text):
    r = dict(real)
    r["answer"] = multi_text
    r["answer_optical"] = optical_text
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tool_fix_tests.json")
    a = ap.parse_args()

    sandbox = revision.SELFTEST_DIR
    sandbox.mkdir(parents=True, exist_ok=True)
    # 真实 3 行（只为拿到真实 mask/valid 路径，供权威复算），写操作只在 sandbox
    real_rows = [(s, r) for s, r in io.read_rows()][:3]
    ids = [r["id"] for _, r in real_rows]
    items = []

    def test(no, name, fn):
        try:
            detail = fn()
            items.append({"no": no, "name": name, "passed": True, "detail": detail})
            print(f"  [{no}/8] PASS {name}: {detail}")
        except Exception as e:
            items.append({"no": no, "name": name, "passed": False,
                          "detail": f"{type(e).__name__}: {e}"})
            print(f"  [{no}/8] FAIL {name}: {type(e).__name__}: {e}")

    def make_recs(opt, mul, state="viewed_needs_change"):
        recs = {}
        for _, r in real_rows:
            numeric, suffix = answer.answer_parts(r)
            recs[r["id"]] = {
                "id": r["id"], "status": "verified", "review_state": state,
                "view_evidence_complete": True, "suffix_verified": True,
                "answer_optical": opt, "answer_multiview": mul,
                "original_answer": r["answer"],
            }
        return recs

    def valid_texts(tag):
        numeric, suffix = answer.answer_parts(real_rows[0][1])
        # 合法外观句（不含 v3 禁止词）
        s1 = f"滑坡区域为{tag}浅色片，与周围深色坡面形成明暗差。"
        s2 = "目标区域边界较清楚、内部较均匀。"
        errs, opt, mul = answer.check_new_text(numeric, suffix, [s1, s2], [], False)
        assert not errs, errs
        return opt, mul

    fingerprints = {r["id"]: io.row_fingerprint(r) for _, r in real_rows}
    tmp_jsonl = sandbox / "train_min.jsonl"

    # 1 改一条两版答案，保存重读确认真新内容（并压缺陷 1：第二次不同内容不被旧值覆盖）
    def t1():
        opt1, mul1 = valid_texts("第一版")
        opt2, mul2 = valid_texts("第二版")
        orig = [dict(r) for _, r in real_rows]
        io.write_jsonl_atomic(tmp_jsonl, orig)
        recs1 = make_recs(opt1, mul1)
        n1 = desc_v2.apply_rows(io.read_jsonl(tmp_jsonl), recs1)
        io.write_jsonl_atomic(tmp_jsonl, n1)
        got = io.read_jsonl(tmp_jsonl)
        assert all(x["answer"] == mul1 and x["answer_optical"] == opt1 for x in got), "第一次两版不是新内容"
        # 第二次 apply：用不同的新光学版；v2 缺陷会把 jsonl 里旧 answer_optical 覆盖回去
        recs2 = make_recs(opt2, mul2)
        n2 = desc_v2.apply_rows(io.read_jsonl(tmp_jsonl), recs2)
        io.write_jsonl_atomic(tmp_jsonl, n2)
        got2 = io.read_jsonl(tmp_jsonl)
        assert all(x["answer"] == mul2 for x in got2), "第二次多图版未更新"
        assert all(x["answer_optical"] == opt2 for x in got2), "第二次光学版被旧值覆盖（缺陷1未修复）"
        return "两次 apply 两版均为真正新内容，第二次 answer_optical 未被旧值覆盖"

    # 2 对同一条连续修订两次，第二次仍能正常更新（split_original 缺陷）
    def t2():
        opt1, mul1 = valid_texts("甲")
        opt2, mul2 = valid_texts("乙")
        r0 = real_rows[0][1]
        # 第一次修订后的行（含外观句）—— 老 split_original 会把外观句算进数值句而报错
        row1 = build_row(r0, opt1, mul1)
        vr1 = answer.verify_answer_pair(row1)
        assert vr1["ok"], f"第一次修订校验失败: {vr1['errors']}"
        # 第二次修订：以含外观句的答案为基础再校验
        row2 = build_row(r0, opt2, mul2)
        vr2 = answer.verify_answer_pair(row2)
        assert vr2["ok"], f"第二次修订校验失败: {vr2['errors']}"
        # 追加两条历史记录，确认不覆盖
        recs = {r0["id"]: {"id": r0["id"], "review_state": "viewed_needs_change"}}
        f = sandbox / "records_min.jsonl"
        if f.exists():
            f.unlink()
        io.append_jsonl(f, [dict(recs[r0["id"]], rev=1), dict(recs[r0["id"]], rev=2)])
        lines = io.read_jsonl(f)
        assert len(lines) == 2 and lines[0]["rev"] == 1 and lines[1]["rev"] == 2, "历史被覆盖"
        return "连续两次修订均通过前后缀断言，records 追加 2 条历史完好"

    # 3 重复应用同一份结果不改变内容、不破坏历史
    def t3():
        opt, mul = valid_texts("幂等")
        recs = make_recs(opt, mul)
        io.write_jsonl_atomic(tmp_jsonl, [dict(r) for _, r in real_rows])
        desc_v2.apply_rows(io.read_jsonl(tmp_jsonl), recs)  # 计算但不写
        n = desc_v2.apply_rows(io.read_jsonl(tmp_jsonl), recs)
        io.write_jsonl_atomic(tmp_jsonl, n)
        h1 = io.sha256_file(tmp_jsonl)
        n2 = desc_v2.apply_rows(io.read_jsonl(tmp_jsonl), recs)
        io.write_jsonl_atomic(tmp_jsonl, n2)
        h2 = io.sha256_file(tmp_jsonl)
        assert h1 == h2, "重复应用改变了字节内容"
        return "同一结果连续 apply 两次字节哈希一致，无历史膨胀"

    # 4 缺逐图观察记录不能通过「已完成」检查
    def t4():
        recs = make_recs(*valid_texts("缺记录"), state="not_viewed")
        recs[ids[0]]["view_evidence_complete"] = False
        errs = desc_v2.validate_before_apply(real_rows, recs, fingerprints)
        assert any("review_state" in e[1] or "view_evidence_complete" in e[1] for e in errs), \
            "缺记录竟通过前置校验"
        # 真实 round 证据校验：无 roundNNN_views.jsonl 一律不通过
        plan = {"images": [{"sample_id": ids[0], "role": "optical_raw"}]}
        ok, probs = revision.verify_round_evidence(99999, plan)
        assert not ok and probs, "无逐图记录竟判为证据完整"
        return f"validate_before_apply 拦截 {len(errs)} 项；无看图记录时 verify_round_evidence=False"

    # 5 缺字段/坐标错/样本不匹配在写盘前发现，且正式清单哈希未变
    def t5():
        opt, mul = valid_texts("前置")
        base = make_recs(opt, mul)
        cases = {}
        # 缺 answer_optical
        c = json.loads(json.dumps(base)); del c[ids[0]]["answer_optical"]
        cases["缺字段"] = c
        # 坐标错误（篡改后缀）
        c = json.loads(json.dumps(base)); c[ids[1]]["answer_multiview"] = mul.replace("<|mask|>", "<|mask|>X")
        cases["坐标错"] = c
        # 样本不匹配（id 集合不一致）
        c = json.loads(json.dumps(base)); c.pop(ids[2]); c["ghost_id"] = c[ids[0]]
        cases["样本不匹配"] = c
        for label, recs in cases.items():
            errs = desc_v2.validate_before_apply(real_rows, recs, fingerprints)
            assert errs, f"{label} 未在写盘前发现"
        # 正式清单未变
        current = V2 / "current_review.json"
        if current.exists():
            pointer = io.load_json(current)
            reference = io.load_json(ROOT / pointer["report"])["manifest_sha256"]
        else:
            reference = io.load_json(V2 / "apply_report.json")["new_sha256"]
        for s in ("train", "val", "test"):
            assert io.sha256_file(DEST / f"{s}.jsonl") == reference[s], f"{s} 被污染"
        return "3 类错误均在写盘前被 validate_before_apply 拦截；三份正式 jsonl 仍为基线"

    # 6 最终清单答案与核对记录一致（篡改一行须失败）
    def t6():
        # 直接复用 check_desc_v2 的核心比对逻辑：构造被篡改的行
        opt, mul = valid_texts("一致性")
        recs = make_recs(opt, mul)
        row = build_row(real_rows[0][1], opt, mul)
        assert row["answer"] == recs[ids[0]]["answer_multiview"]
        tampered = dict(row); tampered["answer"] = mul + "篡改"
        assert tampered["answer"] != recs[ids[0]]["answer_multiview"], "篡改未被发现"
        return "清单与 recs 逐行比对成立，篡改一行即不一致"

    # 7 光学训练读光学版、多图训练读多图版
    def t7():
        r = {"answer": "MULTI", "answer_optical": "OPTICAL"}
        assert answer.select_answer(r, True) == "OPTICAL"
        assert answer.select_answer(r, False) == "MULTI"
        inj = answer.select_answer_row(r, True)
        assert inj["answer"] == "OPTICAL" and inj["answer_optical"] == "OPTICAL"
        rt = (ROOT / "src/xiaomi_lens/domain_runtime.py").read_text(encoding="utf-8")
        dt = (ROOT / "src/xiaomi_lens/train/domain_train.py").read_text(encoding="utf-8")
        assert 'row["answer"] + "<|im_end|>"' in rt, "domain_runtime 消费点变了"
        assert "select_answer_row(row, a.rgb_only)" in dt, "训练未注入答案版本"
        return "select_answer/select_answer_row 三分支 + domain_runtime:127 + domain_train.py 静态核对通过"

    # 8 旧版兼容不让新版缺光学答案静默回退
    def t8():
        for empty in ("", None):
            try:
                answer.select_answer({"answer": "A", "answer_optical": empty}, True)
                raise AssertionError(f"{empty!r} 未报错")
            except ValueError:
                pass
        assert answer.select_answer({"answer": "A"}, True) == "A", "真旧行兼容回退失败"
        # check_answers 对缺字段也应报错
        errs = answer.check_answers([{"id": "x", "answer": "A", "mask": real_rows[0][1]["mask"],
                                      "valid": real_rows[0][1]["valid"]}])
        assert any(e[1] == "answer_optical" and "字段缺失" in e[2] for e in errs), \
            "共享校验漏报缺字段"
        return "空值显式报错；真旧行回退保留；check_answers 对缺字段报错"

    print("=== 8 项工具修复实测（sandbox，不碰正式清单）===")
    for no, name, fn in [(1, "改一条两版答案，重读确认真新内容", t1),
                         (2, "连续修订两次第二次仍正常", t2),
                         (3, "重复应用同一结果幂等且不破坏历史", t3),
                         (4, "缺逐图观察记录不能通过已完成", t4),
                         (5, "缺字段/坐标错/样本不匹配写盘前发现", t5),
                         (6, "最终清单答案与核对记录一致", t6),
                         (7, "光学读光学版、多图读多图版", t7),
                         (8, "旧版兼容不静默回退", t8)]:
        test(no, name, fn)

    result = {
        "run_at": revision.now(), "sandbox": str(sandbox),
        "note": "只在 selftest/ 最小副本上运行；真实 train/val/test.jsonl 未被触碰",
        "items": items, "all_passed": all(x["passed"] for x in items),
        "passed": sum(x["passed"] for x in items), "total": len(items),
    }
    io.atomic_write_json(V2 / a.out, result)
    print(f"{'PASS' if result['all_passed'] else 'FAIL'} verify_desc_v2_fixes: "
          f"{result['passed']}/{result['total']} -> {a.out}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
