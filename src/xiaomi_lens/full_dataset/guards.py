"""只拦截可由实际目标颜色明显反证的句子，不按统计量制造外观文字。"""
import re

import numpy as np
from PIL import Image
from .common import ROOT,NAMES


def unsupported_color(sentence,image_path,mask_path):
    """背景颜色不算目标声明；颜色必须在目标原始显示像素中占有可辨比例。"""
    clause=re.split(r'与周围|和周围|周围|背景',sentence,maxsplit=1)[0]
    rgb=np.asarray(Image.open(ROOT/image_path).convert('RGB')).astype(float)
    mask=np.asarray(Image.open(ROOT/mask_path))>0
    pixels=rgb[mask];r,g,b=pixels.T
    maximum=pixels.max(-1);minimum=pixels.min(-1);saturation=(maximum-minimum)/np.maximum(maximum,1)
    fractions={
        '绿':(g>r+5)&(g>b+2),
        '紫':(r>g+8)&(b>g+8),
        '蓝':(b>r+8)&(b>g+4),
        '红':(r>g+12)&(r>b+5),
        '粉':(r>g+12)&(r>b)&(b>g-10)&(maximum>130),
        '橙':(r>g+15)&(g>b+10),
        '黄':(r>b+15)&(g>b+12),
    }
    # 仅拒绝少于一成像素支持的色相；不能据此证明保留的句子一定正确。
    for word,seen in fractions.items():
        if word in clause and float(seen.mean())<.1:
            return {'color':word,'support_fraction':float(seen.mean())}
    if re.search(r'白色|灰白|浅白',clause) and float(((maximum>175)&(saturation<.35)).mean())<.05 and float(pixels.mean())<140:
        return {'color':'白','reason':'very few bright near-neutral target pixels'}
    return None


def color_guard(row,optical,auxiliary):
    """实际图对应实际颜色；InSAR/DEM只有灰度，不据灰度描述形变大小。"""
    optical=list(optical);auxiliary=list(auxiliary);rejected=[];accepted=[]
    for sentence in optical:
        contradiction=unsupported_color(sentence,row['images'][-1],row['mask'])
        if contradiction:rejected.append({'kind':'optical','sentence':sentence,'reason':'target color contradicts claim','evidence':contradiction})
        else:accepted.append(sentence)
    optical=accepted;accepted=[]
    for sentence in auxiliary:
        contradiction=None
        if row['source']==NAMES[1]:contradiction=unsupported_color(sentence,row['images'][0],row['mask'])
        elif re.search(r'绿|紫|蓝|红|粉|橙|黄',sentence):contradiction={'reason':'only grayscale auxiliary views provided'}
        if contradiction:rejected.append({'kind':'auxiliary','sentence':sentence,'reason':'auxiliary display color contradicts claim','evidence':contradiction})
        else:accepted.append(sentence)
    return optical,accepted,rejected
