"""答案组装：数值与坐标沿用程序按标注生成的原文，看图模型只提供外观句。

本机不 import domain.py（缺 scipy/h5py）；这里用 numpy 等价实现逐条复核，
并在 WSL 侧 smoke 脚本中用原函数做权威复核。
"""
import ast
import functools
import json
import re

import numpy as np
from PIL import Image

from .spec import (
    ROOT, MAX_CHARS, MAX_SENTENCES, FORBIDDEN_RE, AUX_SOURCE_WORDS,
)

BOX_MARK = " <bbox>"


def split_original(answer):
    """拆成 (数值句, 后缀)。后缀含前导空格，逐字保留 json.dumps 的空格与取整规则。"""
    i = answer.find(BOX_MARK)
    if i < 0:
        raise ValueError("原答案缺少 <bbox> 后缀")
    return answer[:i], answer[i:]


def load_mask_valid(row):
    m = np.array(Image.open(ROOT / row["mask"]).convert("L")) > 127
    v = np.array(Image.open(ROOT / row["valid"]).convert("L")) > 127
    if m.shape != v.shape:
        raise ValueError(f"mask/valid 尺寸不一致: {row['id']}")
    return m, v


def count_components8(mask):
    """8 连通块数：按行行程 + 并查集，等价 scipy.ndimage.label(structure=ones((3,3)))。"""
    h, w = mask.shape
    parent = []

    def new_label():
        parent.append(len(parent))
        return len(parent) - 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    prev = []
    for y in range(h):
        row = mask[y]
        runs = []
        x = 0
        while x < w:
            if row[x]:
                start = x
                while x < w and row[x]:
                    x += 1
                runs.append((start, x, new_label()))
            else:
                x += 1
        for s, e, lab in runs:
            for ps, pe, plab in prev:
                # 相邻两行、列差 <=1 即连通
                if s <= pe and ps <= e:
                    union(lab, plab)
        prev = runs
    if not parent:
        return 0
    return len({find(i) for i in range(len(parent))})


def numeric_from_mask(mask, valid):
    """等价复现 domain.describe() L62-80。"""
    m = mask.astype(bool) & valid.astype(bool)
    if not m.any():
        return "未检测到滑坡区域。"
    yy, xx = np.where(m)
    h, w = m.shape
    vertical = ["上", "中", "下"][min(2, int(yy.mean() / h * 3))]
    horizontal = ["左", "中", "右"][min(2, int(xx.mean() / w * 3))]
    position = "中央" if vertical == horizontal == "中" else horizontal + vertical + "部"
    fraction = m.sum() / max(1, valid.sum()) * 100
    area = "小于0.1%" if fraction < 0.05 else f"约{fraction:.1f}%"
    pattern = "呈多块分布" if count_components8(m) > 1 else "呈连续分布"
    return f"检测到滑坡区域，区域整体中心位于图像{position}，{pattern}，占有效图像面积{area}。"


def suffix_from_mask(mask, valid):
    """等价复现 domain.training_answer() L83-97 的 bbox 与区域标记。"""
    target = mask.astype(bool) & valid.astype(bool)
    if not target.any():
        return ""
    yy, xx = np.where(target)
    h, w = target.shape
    box = [
        int(xx.min() / w * 1000),
        int(yy.min() / h * 1000),
        int(np.ceil((xx.max() + 1) / w * 1000)),
        int(np.ceil((yy.max() + 1) / h * 1000)),
    ]
    return BOX_MARK + json.dumps(box) + "</bbox><|mask|>"


@functools.lru_cache(maxsize=1)
def _official_parser():
    """复用 Xiaomi-Lens-0-raw/infer.py 的官方解析器（ast 提取，不自写第二份）。"""
    source = (ROOT / "Xiaomi-Lens-0-raw" / "infer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        n for n in tree.body
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BOX_RE" for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "parse_bboxes")
    ]
    scope = {"re": re, "BOX_START_TOKEN": "<bbox>", "BOX_END_TOKEN": "</bbox>"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "official_bbox_parser", "exec"), scope)
    return scope["parse_bboxes"]


def official_parse_bboxes(text):
    return _official_parser()(text)


def _check_one_answer(text, numeric, suffix):
    """校验单版答案：以标注计算的数值句开头、后缀逐字收尾、官方解析器恰 1 个有效框。

    不要求 <bbox> 之前的文本恰好等于数值句 —— apply 之后那里还会接外观句。
    """
    errors = []
    if not isinstance(text, str) or not text:
        return ["答案为空或非字符串"]
    if not text.startswith(numeric):
        errors.append("数值句与标注计算不一致")
    if not text.endswith(suffix):
        errors.append("区域坐标/标记后缀与标注计算不一致")
    boxes = official_parse_bboxes(text)
    ok_box = (
        len(boxes) == 1
        and 0 <= boxes[0][0] < boxes[0][2] <= 1000
        and 0 <= boxes[0][1] < boxes[0][3] <= 1000
    )
    if not ok_box:
        errors.append("官方解析器未得到恰好 1 个有效框")
    if not suffix.endswith("<|mask|>"):
        errors.append("缺少 <|mask|> 区域标记")
    return errors


def answer_parts(row):
    """由 mask/valid 权威复算 (数值句, 后缀)；等价 domain.training_answer() 的拆分。"""
    m, v = load_mask_valid(row)
    return numeric_from_mask(m, v), suffix_from_mask(m, v)


def verify_original(row):
    """校验 answer 由标注算出；返回逐项布尔结果，兼容含外观句的新答案。

    v2 缺陷：split_original(row['answer']) 把「数值句+外观句」整段当数值句，
    再要求与 numeric_from_mask 全等，导致 apply 之后二次修订必然报错。
    v3 改为前缀 + 后缀断言。
    """
    numeric, suffix = answer_parts(row)
    errors = _check_one_answer(row["answer"], numeric, suffix)
    return {
        "numeric_matches_mask": not any("数值句" in e for e in errors),
        "suffix_matches_mask": not any("后缀" in e for e in errors),
        "one_valid_bbox": not any("有效框" in e for e in errors),
        "mask_token": suffix.endswith("<|mask|>") and row["answer"].endswith("<|mask|>"),
        "numeric": numeric,
        "suffix": suffix,
        "errors": errors,
    }


def verify_answer_pair(row):
    """校验两版答案都与同一份标注计算的前缀/后缀一致，并给出两版错误明细。"""
    numeric, suffix = answer_parts(row)
    err_answer = _check_one_answer(row.get("answer"), numeric, suffix)
    err_optical = _check_one_answer(row.get("answer_optical"), numeric, suffix)
    errors = [f"answer: {e}" for e in err_answer] + [f"answer_optical: {e}" for e in err_optical]
    return {
        "ok": not errors,
        "numeric_matches_mask": not any("数值句" in e for e in err_answer + err_optical),
        "suffix_matches_mask": not any("后缀" in e for e in err_answer + err_optical),
        "one_valid_bbox": not any("有效框" in e for e in err_answer + err_optical),
        "mask_token": suffix.endswith("<|mask|>"),
        "numeric": numeric, "suffix": suffix, "errors": errors,
    }


def check_answers(rows):
    """共享校验函数（不依赖 torch）：批量检查每行两版答案的前缀/后缀/坐标。

    供 tools/check_domain_data.py 与实测脚本复用，避免重复实现。
    返回错误列表 [(id, key, reason)]。
    """
    errors = []
    for r in rows:
        numeric, suffix = answer_parts(r)
        for key in ("answer", "answer_optical"):
            if key not in r:
                errors.append((r["id"], key, "字段缺失"))
                continue
            for e in _check_one_answer(r[key], numeric, suffix):
                errors.append((r["id"], key, e))
    return errors


def assert_dual_row(row):
    """写盘前置断言：两版字段存在、非空、前后缀各自与标注计算一致。"""
    errors = []
    for key in ("answer", "answer_optical"):
        if key not in row:
            errors.append(f"{key} 字段缺失")
        elif not isinstance(row[key], str) or not row[key].strip():
            errors.append(f"{key} 为空或非字符串")
    if errors:
        return errors
    numeric, suffix = answer_parts(row)
    for key in ("answer", "answer_optical"):
        errors.extend(f"{key}: {e}" for e in _check_one_answer(row[key], numeric, suffix))
    return errors


def _sentence(s):
    s = (s or "").strip()
    if not s:
        return ""
    return s if s.endswith("。") else s + "。"


def compose(numeric, optical_sentences, aux_sentences, suffix):
    """两版答案共用同一数值句与同一后缀；外观句由看图证据提供。"""
    optical_part = "".join(_sentence(s) for s in optical_sentences)
    aux_part = "".join(_sentence(s) for s in aux_sentences)
    return numeric + optical_part + suffix, numeric + optical_part + aux_part + suffix


def check_new_text(numeric, suffix, optical_sentences, aux_sentences, has_aux_views):
    """返回 (错误列表, 光学版, 多图版)。只对新写入的句子做禁止词检查。"""
    errors = []
    new_text = "".join(_sentence(s) for s in list(optical_sentences) + list(aux_sentences))
    for pat in FORBIDDEN_RE:
        hit = pat.search(new_text)
        if hit:
            errors.append(f"禁止词命中: {pat.pattern} -> {hit.group(0)}")
    optical, multiview = compose(numeric, optical_sentences, aux_sentences, suffix)

    for word in AUX_SOURCE_WORDS:
        if word in optical_part_only(optical, numeric, suffix):
            errors.append(f"光学版混入辅助图来源词: {word}")
    if not has_aux_views and optical != multiview:
        errors.append("仅光学输入的样本两版答案必须逐字相同")
    if not optical.startswith(numeric) or not multiview.startswith(numeric):
        errors.append("数值句被改动")
    if not optical.endswith(suffix) or not multiview.endswith(suffix):
        errors.append("区域坐标/标记后缀被改动")
    if not suffix.endswith("<|mask|>"):
        errors.append("缺少 <|mask|>")
    if len(multiview) > MAX_CHARS:
        errors.append(f"多图版 {len(multiview)} 字符超过保守上限 {MAX_CHARS}")
    if len(optical) > MAX_CHARS:
        errors.append(f"光学版 {len(optical)} 字符超过保守上限 {MAX_CHARS}")
    sentences = sum(1 for ch in multiview if ch in "。！？")
    if sentences > MAX_SENTENCES:
        errors.append(f"{sentences} 句超过上限 {MAX_SENTENCES}")
    return errors, optical, multiview


def optical_part_only(optical, numeric, suffix):
    return optical[len(numeric):-len(suffix)] if suffix else optical[len(numeric):]


def select_answer(row, rgb_only=False):
    """--rgb-only 读光学版，其余读多图版。

    v3 三分支（修复「新版缺失的光学答案悄悄退回多图答案」）：
    - 非 rgb_only：恒返回 answer。
    - rgb_only 且完全无 answer_optical 键：旧格式行，兼容回退到 answer。
    - rgb_only 且键存在、值非空：返回光学版。
    - rgb_only 且键存在但值为 ""/None：显式报错，不静默回退。
    """
    if not rgb_only:
        return row["answer"]
    if "answer_optical" not in row:
        return row["answer"]
    val = row["answer_optical"]
    if val is None or val == "":
        raise ValueError("新版数据缺失光学版答案（answer_optical 为空），禁止静默回退到多图版")
    return val


def select_answer_row(row, rgb_only=False):
    """返回浅拷贝，使 domain_runtime.build() 无需改动即可读选定版本。"""
    return {**row, "answer": select_answer(row, rgb_only)}
