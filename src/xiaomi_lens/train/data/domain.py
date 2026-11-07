"""滑坡样本准备：源文件只读、先划分后拟合显示范围、逐像素检查目标。"""

from pathlib import Path
import hashlib, json, random
import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_labels

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path("/home/yukun80/codes/datasets")
DEST = ROOT / "datasets/landslide_lens_positive_v1"
NAMES = ["GDCLD", "landslide4sense", "multimodal-landslide-dataset"]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )


def jsonlines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def read_source(name, row, image=True):
    """返回原始波段、有效像素、二值标签和标签有效区；非法标签立即拒绝。"""
    base = SOURCE / name / ("hdf5" if name == "landslide4sense" else "")
    with h5py.File(base / row["mask_hdf5"], "r") as f:
        mask = f["mask"][:]
        valid = (
            f["valid_mask"][:].astype(bool)
            if "valid_mask" in f
            else np.ones(mask.shape, bool)
        )
    if not set(np.unique(mask[valid])).issubset({0, 1}):
        raise ValueError("unknown_label")
    mask = (mask == 1) & valid
    if not image:
        return None, None, mask, valid
    with h5py.File(base / row["image_hdf5"], "r") as f:
        x = f["image"][:]
        pv = (
            f["pixel_valid"][:].astype(bool)
            if "pixel_valid" in f
            else np.ones(x.shape, bool)
        )
        if "channel_valid" in f:
            pv &= f["channel_valid"][:].astype(bool)[:, None, None]
    pv &= np.isfinite(x)
    rgb = [3, 2, 1] if name == "landslide4sense" else [0, 1, 2]
    valid &= pv[rgb].all(0)
    return x, pv, mask & valid, valid


def describe(mask, valid=None):
    """只说明由区域直接计算的位置、有效面积比例，不猜测灾害原因。"""
    if valid is None:
        valid = np.ones(mask.shape, bool)
    mask = mask.astype(bool) & valid
    if not mask.any():
        return "未检测到滑坡区域。"
    yy, xx = np.where(mask)
    h, w = mask.shape
    vertical = ["上", "中", "下"][min(2, int(yy.mean() / h * 3))]
    horizontal = ["左", "中", "右"][min(2, int(xx.mean() / w * 3))]
    position = (
        "中央" if vertical == horizontal == "中" else horizontal + vertical + "部"
    )
    fraction = mask.sum() / max(1, valid.sum()) * 100
    area = "小于0.1%" if fraction < 0.05 else f"约{fraction:.1f}%"
    _, blocks = connected_labels(mask, structure=np.ones((3, 3), dtype=np.uint8))
    pattern = "呈多块分布" if blocks > 1 else "呈连续分布"
    return f"检测到滑坡区域，区域整体中心位于图像{position}，{pattern}，占有效图像面积{area}。"


def training_answer(mask, valid):
    """生成官方格式答案；坐标上边界向外取整，防止极小区域变成零宽框。"""
    answer = describe(mask, valid)
    target = mask.astype(bool) & valid.astype(bool)
    if target.any():
        yy, xx = np.where(target)
        h, w = target.shape
        box = [
            int(xx.min() / w * 1000),
            int(yy.min() / h * 1000),
            int(np.ceil((xx.max() + 1) / w * 1000)),
            int(np.ceil((yy.max() + 1) / h * 1000)),
        ]
        answer += " <bbox>" + json.dumps(box) + "</bbox><|mask|>"
    return answer


def dhash(rgb):
    a = np.asarray(Image.fromarray(rgb).convert("L").resize((9, 8)))
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), "big")


def footprint(row):
    g = row.get("geotransform")
    if not g:
        return None
    # 转换清单采用 Affine 顺序 a,b,c,d,e,f，不是 GDAL 顺序。
    a, b, c, d, e, f = g
    h = w = 128
    xs = [c, c + a * w + b * h]
    ys = [f, f + d * w + e * h]
    return [min(xs), min(ys), max(xs), max(ys)]


def overlaps(a, b):
    return (
        a
        and b
        and min(a[2], b[2]) > max(a[0], b[0])
        and min(a[3], b[3]) > max(a[1], b[1])
    )


def prepare():
    if (DEST / "summary.json").exists():
        raise RuntimeError("已存在完成的数据集，避免覆盖；请先运行检查工具。")
    DEST.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261107)
    audit = []
    selected = []
    used = {}
    hashes = []
    for name in NAMES:
        rows = jsonlines(SOURCE / name / "hdf5/conversion_manifest.jsonl")
        rng.shuffle(rows)
        if name == NAMES[2]:
            pool = []
            for r in rows:
                try:
                    _, _, m, v = read_source(name, r, False)
                    reason = "candidate" if m.any() else "no_target_in_valid_label"
                except Exception as e:
                    reason = str(e)
                audit.append({"source": name, "id": r["sample_key"], "screen": reason})
                if reason == "candidate":
                    pool.append(r)
            rows = pool
        for split, total, negative in [
            ("train", 320 if name == NAMES[0] else 240, 0),
            ("val", 40 if name == NAMES[0] else 30, 0),
            ("test", 40 if name == NAMES[0] else 30, 0),
        ]:
            count = 0
            neg = 0
            for r in rows:
                key = r["sample_key"]
                if any(z["id"] == key and z["source"] == name for z in selected):
                    continue
                native = r.get("split")
                if name == NAMES[0] and native != split:
                    continue
                if name == NAMES[2] and native != (
                    "train" if split == "train" else "val"
                ):
                    continue
                group = (
                    name,
                    str(r.get("source_origin", "")) + ":" + str(r.get("scene") or key),
                )
                if group in used and used[group] != split:
                    continue
                declared = r.get("positive_pixel_count", 0) > 0
                if name == NAMES[1] and not declared:
                    audit.append(
                        {
                            "source": name,
                            "id": key,
                            "screen": "metadata_zero_not_sampled",
                        }
                    )
                    continue
                try:
                    x, pv, m, v = read_source(name, r)
                    if not v.any():
                        raise ValueError("all_invalid")
                    if not m.any():
                        raise ValueError("no_target_after_image_validity")
                    if name == NAMES[1] and bool(m.any()) != declared:
                        raise ValueError("presence_mismatch")
                    rgb = np.moveaxis(
                        x[[3, 2, 1] if name == NAMES[1] else [0, 1, 2]], 0, -1
                    )
                    preview = (
                        (rgb - rgb.min())
                        / max(1e-6, float(rgb.max() - rgb.min()))
                        * 255
                    ).astype("uint8")
                    image_hash = hashlib.sha256(x.tobytes()).hexdigest()
                    perceptual = dhash(preview)
                    box = footprint(r)
                    for old in selected:
                        if image_hash == old["image_sha256"]:
                            raise ValueError("exact_duplicate")
                        if old["split"] != split and (
                            (perceptual ^ old["dhash"]).bit_count() <= 2
                            or (
                                old["source"] == name
                                and overlaps(box, old["footprint"])
                            )
                        ):
                            raise ValueError("cross_split_near_duplicate_or_overlap")
                    selected.append(
                        {
                            "id": key,
                            "source": name,
                            "split": split,
                            "group": group[1],
                            "image_sha256": image_hash,
                            "dhash": perceptual,
                            "footprint": box,
                            "positive_pixels": int(m.sum()),
                            "valid_pixels": int(v.sum()),
                            "source_record": r,
                        }
                    )
                    used[group] = split
                    count += 1
                    neg += not m.any()
                except Exception as e:
                    audit.append({"source": name, "id": key, "screen": str(e)})
                    continue
                if count >= total:
                    break
            print(name, split, count, "negative", neg, flush=True)
            if count < total:
                raise RuntimeError(
                    f"分组后候选不足: {name}/{split} {count}/{total}; inspect before changing quotas"
                )
    # 仅训练图片贡献显示范围；每通道确定性采样，避免大图支配统计。
    ranges = {}
    for name in NAMES:
        chunks = []
        for r in selected:
            if r["source"] != name or r["split"] != "train":
                continue
            x, pv, _, _ = read_source(name, r["source_record"])
            chunks.append(
                [
                    x[c][pv[c]][:: max(1, int(pv[c].sum()) // 2048)]
                    for c in range(len(x))
                ]
            )
        ranges[name] = [
            np.percentile(np.concatenate([a[c] for a in chunks]), [2, 98]).tolist()
            for c in range(len(chunks[0]))
        ]
    write_json(DEST / "display_ranges.json", ranges)
    for r in selected:
        name = r["source"]
        x, pv, m, v = read_source(name, r["source_record"])
        folder = DEST / "samples" / r["id"]
        folder.mkdir(parents=True, exist_ok=True)
        views = (
            [[7, 10, 11], [3, 2, 1]]
            if name == NAMES[1]
            else [[4, 4, 4], [3, 3, 3], [0, 1, 2]] if name == NAMES[2] else [[0, 1, 2]]
        )
        files = []
        for i, channels in enumerate(views):
            out = []
            for c in channels:
                lo, hi = ranges[name][c]
                a = np.clip((x[c] - lo) / max(hi - lo, 1e-6), 0, 1)
                a = np.where(pv[c], a, 0)
                out.append((a * 255).astype("uint8"))
            file = folder / f"view_{i}.png"
            Image.fromarray(np.stack(out, -1)).save(file)
            files.append(str(file.relative_to(ROOT)))
        Image.fromarray(m.astype("uint8") * 255).save(folder / "mask.png")
        Image.fromarray(v.astype("uint8") * 255).save(folder / "valid.png")
        r.update(
            images=files,
            mask=str((folder / "mask.png").relative_to(ROOT)),
            valid=str((folder / "valid.png").relative_to(ROOT)),
            answer=training_answer(m, v),
            height=m.shape[0],
            width=m.shape[1],
        )
    for split in ["train", "val", "test"]:
        (DEST / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in selected
                if r["split"] == split
            )
        )
    (DEST / "screening.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit)
    )
    write_json(
        DEST / "summary.json",
        {
            "total": len(selected),
            "answer_format": "official_square_bbox_center_blocks_v2",
            "seed": 20261107,
            "counts": {
                s: sum(r["split"] == s for r in selected)
                for s in ["train", "val", "test"]
            },
            "positive_only": True,
            "empty_samples": sum(not r["positive_pixels"] for r in selected),
            "multimodal_empty": sum(
                r["source"] == NAMES[2] and not r["positive_pixels"] for r in selected
            ),
            "limitations": [
                "Landslide4Sense无地点信息，近似图筛查不证明地理独立",
                "全部来源只含正样本，无法评估无目标图片的误报",
            ],
            "manifest_sha256": {
                s: hashlib.sha256((DEST / f"{s}.jsonl").read_bytes()).hexdigest()
                for s in ["train", "val", "test"]
            },
        },
    )
    print("Prepared", len(selected), flush=True)
