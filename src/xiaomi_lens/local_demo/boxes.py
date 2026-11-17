"""定位框只取模型最终回答；展示产物与原预测隔离，绝不从mask冒充模型框。"""
import json, re, math, zipfile, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def parse_boxes(raw, width, height, offset=(0, 0)):
    """将0—1000相对边界换成原图像素边界；错误框保留原文而不修正。"""
    text = raw.split('</think>')[-1]
    chunks = re.findall(r'<bbox>(.*?)</bbox>', text, re.S)
    items, warnings = [], []
    if text.count('<bbox>') != len(chunks):
        warnings.append('定位框标记不完整')
    if not chunks:
        warnings.append('模型没有输出可读取的定位框')
    for i, chunk in enumerate(chunks):
        item = {'number': i + 1, 'raw': chunk, 'valid': False}
        try:
            box = json.loads(chunk)
            if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
                raise ValueError('必须是四个有限数字')
            if any(v < 0 or v > 1000 for v in box):
                raise ValueError('坐标超出0—1000范围')
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('边界顺序错误或框没有面积')
            x, y = offset
            item.update(valid=True, normalized=box, pixel_xyxy=[x+box[0]*width/1000, y+box[1]*height/1000, x+box[2]*width/1000, y+box[3]*height/1000])
        except (ValueError, TypeError) as e:
            item['error'] = str(e)
            warnings.append(f'框{i+1}：{e}')
        items.append(item)
    return {'coordinate_convention': 'x向右,y向下;像素外边界,左上为(0,0),右下为(width,height)', 'boxes': items, 'warnings': warnings}


def draw_view(image, mask, boxes, show_mask=True, show_boxes=True, scale=(1, 1)):
    """预览比例仅影响绘画；导出坐标始终保持原图单位。"""
    a = np.array(image.convert('RGB')).copy()
    if show_mask:
        m = np.asarray(mask) == 1
        a[m] = (a[m]*.5 + np.array([255,40,30])*.5).astype('uint8')
    im = Image.fromarray(a); pen = ImageDraw.Draw(im)
    if show_boxes:
        for b in boxes:
            if not b.get('valid'): continue
            x0,y0,x1,y1 = b['pixel_xyxy'];sx,sy=scale
            xy=[x0*sx,y0*sy,min(im.width-1,x1*sx),min(im.height-1,y1*sy)]
            pen.rectangle(xy,outline='yellow',width=2)
            pen.text((xy[0]+2,xy[1]+2),str(b['number']),fill='yellow',stroke_width=1,stroke_fill='black')
    return im


def presentation(image_path, result_folder, destination):
    """为已有预测新增四种展示图及坐标下载包，不覆盖原结果。"""
    src=Path(result_folder);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    result=json.loads((src/'result.json').read_text())
    with Image.open(image_path) as im: image=im.convert('RGB')
    with Image.open(src/'mask.png') as im: mask=(np.array(im)>0).astype('uint8')
    record=parse_boxes(result.get('raw_answer',''),image.width,image.height)
    record['source_result_sha256']=hashlib.sha256((src/'result.json').read_bytes()).hexdigest()
    (dest/'boxes.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    for m in (False,True):
        for b in (False,True):draw_view(image,mask,record['boxes'],m,b).save(dest/f'view_{int(m)}_{int(b)}.png')
    with zipfile.ZipFile(dest/'localization.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(dest/'boxes.json','boxes.json');z.write(dest/'view_1_1.png','localization.png')
        for name in ['result.json','prediction.json','mask.png','overlay.png']:
            if (src/name).exists():z.write(src/name,name)
    return record


def view_path(folder, show_mask, show_boxes):
    return str(Path(folder)/f'view_{int(show_mask)}_{int(show_boxes)}.png') if folder else None
