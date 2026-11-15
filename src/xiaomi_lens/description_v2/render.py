"""生成看图产物：GT 标注叠加图、模型接收尺寸图、拼接对照图、局部放大。

只用 numpy + Pillow，不依赖 scipy / h5py / torch。只读 samples/，只写 V2/views/。
"""
import numpy as np
from PIL import Image, ImageDraw

from .spec import (
    ROOT, V2, OVERLAY_COLOR, OVERLAY_ALPHA, OVERLAY_LEGEND, VIEW_SPEC, input_size,
)


def _read_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def _mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def _label(panel, label, cell_w=300):
    img = panel.copy()
    img.thumbnail((cell_w, 280), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell_w, 310), (245, 245, 245))
    canvas.paste(img, ((cell_w - img.width) // 2, 24 + (280 - img.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), label, fill=(20, 20, 20))
    return canvas


def _contact_sheet(panels, out_path, sample_id):
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 300, rows * 310), (255, 255, 255))
    for i, (label, img) in enumerate(panels):
        sheet.paste(_label(img, label), ((i % cols) * 300, (i // cols) * 310))
    sheet.save(out_path)
    with open(out_path.with_name("contact_sheet_legend.txt"), "w", encoding="utf-8") as f:
        f.write(f"sample_id: {sample_id}\n")
        f.write(OVERLAY_LEGEND + "\n")
        f.write("panel labels: " + " | ".join(label for label, _ in panels) + "\n")
        f.write("局部放大仅用于定位，放大不会产生新的地面细节。\n")
        f.write("InSAR/DEM 灰度是 0-255 编码值经 2%-98% 拉伸的显示图，只作相对明暗/渐变解读，"
                "不代表物理量或地表真实颜色；灰度偏亮不等于形变绝对值大。\n")
        f.write("用户授权 B 口径：三图依据充分时可写定性活动滑坡判别；但本地文件无法核实 |v|>20 mm/yr，"
                "答案中不写速率数值，该结论不是由本地数据计算得出。\n")


def render_sample(row):
    """为一条样本生成全部看图产物，返回产物路径清单。"""
    sid = row["id"]
    out = V2 / "views" / sid
    out.mkdir(parents=True, exist_ok=True)

    optical_path = ROOT / row["images"][-1]
    optical = _read_rgb(optical_path)
    m = _mask(ROOT / row["mask"])
    v = _mask(ROOT / row["valid"])
    target = m & v

    # GT 叠加：品红只是标注显示色，不代表地表颜色
    base = np.array(optical).astype(np.float32)
    color = np.array(OVERLAY_COLOR, dtype=np.float32)
    base[target] = base[target] * (1 - OVERLAY_ALPHA) + color * OVERLAY_ALPHA
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    overlay.save(out / "overlay.png")
    (out / "overlay_legend.txt").write_text(OVERLAY_LEGEND + "\n", encoding="utf-8")

    # 模型实际接收尺寸：严格复现 images_for() L102-109
    optical_input = optical.resize(input_size(optical.size, True), Image.Resampling.BICUBIC)
    optical_input.save(out / "optical_at_input.png")

    aux_views = []
    for i, p in enumerate(row["images"][:-1]):
        aux = _read_rgb(ROOT / p)
        aux_at_input = aux.resize(input_size(aux.size, False), Image.Resampling.BICUBIC)
        aux_at_input.save(out / f"aux_{i}_at_input.png")
        aux_views.append((i, aux, aux_at_input))

    panels = [
        ("optical raw (no overlay)", optical),
        ("mask", Image.fromarray((m * 255).astype(np.uint8)).convert("RGB")),
        ("GT overlay (label color only)", overlay),
        (f"optical @ input {optical_input.width}", optical_input),
    ]
    for i, aux, aux_at_input in aux_views:
        panels.append((f"aux{i} raw", aux))
        panels.append((f"aux{i} @ input {aux_at_input.width}", aux_at_input))
    _contact_sheet(panels, out / "contact_sheet.png", sid)

    # 目标局部放大：只用于定位；放大不产生新细节
    crops = out / "crops"
    crops.mkdir(exist_ok=True)
    yy, xx = np.where(target)
    if len(xx):
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        pad = max(16, (x1 - x0) // 4, (y1 - y0) // 4)
        box = (max(0, x0 - pad), max(0, y0 - pad), min(optical.width, x1 + pad), min(optical.height, y1 + pad))
        crop = optical.crop(box)
        zoom = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
        zoom.save(crops / "target_zoom.png")
        (crops / "README.txt").write_text(
            "局部放大只帮助定位，放大不会产生新的地面细节；答案只写接收尺寸图上可分辨的特征。\n",
            encoding="utf-8",
        )

    produced = [
        {"path": str(out / "overlay.png"), "role": "label_overlay"},
        {"path": str(out / "optical_at_input.png"), "role": "optical_at_model_size"},
        {"path": str(out / "contact_sheet.png"), "role": "contact_sheet"},
        {"path": str(crops / "target_zoom.png"), "role": "crop"},
    ]
    for i, _, _ in aux_views:
        produced.append({"path": str(out / f"aux_{i}_at_input.png"), "role": f"aux_{i}_at_model_size"})
    spec = VIEW_SPEC[row["source"]]
    return {"sample_id": sid, "optical_raw": str(optical_path), "aux_raw": [str(ROOT / p) for p in row["images"][:-1]],
            "mask": str(ROOT / row["mask"]), "valid": str(ROOT / row["valid"]),
            "view_spec": spec, "produced": produced}
