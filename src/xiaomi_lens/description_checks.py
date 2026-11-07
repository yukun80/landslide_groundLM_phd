"""对照模型原话与其预测区域；规则只标记可核对矛盾，不冒充专家审核。"""

import re
import numpy as np
from .train.data.domain import describe


def consistency_issues(text, mask):
    """返回有无目标、面积及明确位置/分布表述的不一致；无法解析时不猜测。"""
    issues = []
    present = bool(mask.any())
    denies = "未检测到" in text or "没有滑坡" in text or "无滑坡" in text
    asserts = ("检测到滑坡" in text or "滑坡区域位于" in text) and not denies
    if (denies and present) or (asserts and not present):
        issues.append("文字中的有无目标与预测区域不一致")
    percentages = list(re.finditer(r"(\d+(?:\.\d+)?)\s*%", text))
    if len(percentages) == 1:
        match = percentages[0]
        stated = float(match.group(1))
        measured = float(mask.mean() * 100)
        lead = text[max(0, match.start() - 6) : match.start()]
        bounded = any(word in lead for word in ["小于", "不足", "不到", "低于", "<"])
        different = (
            measured >= stated
            if bounded
            else abs(stated - measured) > max(0.1, 0.25 * measured)
        )
        if different:
            issues.append(f"文字面积与预测区域计算值不同（预测区域约{measured:.1f}%）")
    if present:
        derived = describe(mask)
        # 明确的九宫格词可以核对；“附近”等模糊地理描述留给审阅者。
        aliases = [
            ("中部偏右", "右中部"),
            ("中部偏左", "左中部"),
            ("右上", "右上部"),
            ("右下", "右下部"),
            ("左上", "左上部"),
            ("左下", "左下部"),
            ("中上", "中上部"),
            ("中下", "中下部"),
            ("左中", "左中部"),
            ("右中", "右中部"),
            ("中央", "中央"),
        ]
        recognized = next(
            (canonical for phrase, canonical in aliases if phrase in text), None
        )
        if recognized and f"图像{recognized}" not in derived:
            issues.append("文字位置与预测区域整体中心不同，需结合图片核对")
        for pattern, opposite in [
            ("呈多块分布", "呈连续分布"),
            ("呈连续分布", "呈多块分布"),
        ]:
            if pattern in text and opposite in derived:
                issues.append("文字中的连续或多块分布与预测区域不同")
    return issues


def answer_presence(text, mask_count=0):
    """单独判断回答是否声称有目标；无法辨认返回None，不能拿空区域代替文字判断。"""
    if any(
        term in text
        for term in [
            "未检测到滑坡",
            "未发现滑坡",
            "没有滑坡",
            "无滑坡",
            "未发现明显滑坡",
        ]
    ):
        return False
    if mask_count or any(
        term in text
        for term in ["检测到滑坡", "滑坡区域位于", "可以识别出", "存在滑坡"]
    ):
        return True
    return None
