"""只用普通处理器准备独立核对图；不导入或加载任何看图模型。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT,atomic_write_json
GUIDE_VERSION=2

def images_for(row):
    config=json.loads((ROOT/'src/xiaomi_lens/train/configs/domain.json').read_text())
    result=[];sizes=[]
    for i,path in enumerate(row['images']):
        with Image.open(ROOT/path) as opened:im=opened.convert('RGB')
        sizes.append(im.size)
        bound=config['high_resolution_edge'] if i==len(row['images'])-1 and max(im.size)>config['auxiliary_edge'] else config['auxiliary_edge']
        ratio=bound/max(im.size);size=tuple(max(32,round(v*ratio/32)*32) for v in im.size)
        result.append(im.resize(size,Image.Resampling.BICUBIC))
    assert len(set(sizes))==1,'unaligned input views'
    return result


def view_paths(row,dest,subdir="review/inputs"):
    """独立原图、标注、叠加、输入尺寸与目标原像素核对图，不拼成缩略总图。"""
    folder=Path(dest)/subdir/row['id'];folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'ready.json'
    if not marker.exists() or json.loads(marker.read_text()).get('guide_version')!=GUIDE_VERSION:
        raw=np.asarray(Image.open(ROOT/row['images'][-1]).convert('RGB'))
        mask=np.asarray(Image.open(ROOT/row['mask']))>0;valid=np.asarray(Image.open(ROOT/row['valid']))>0;t=mask&valid
        overlay=raw.astype(float);overlay[t]=overlay[t]*.5+np.array([255,0,255])*.5
        Image.fromarray(overlay.astype('uint8')).save(folder/'overlay.png',compress_level=1)
        Image.fromarray(np.where(t[...,None],raw,128).astype('uint8')).save(folder/'target.png',compress_level=1)
        images=images_for(row)
        images[-1].save(folder/'optical_at_input.png',compress_level=1)
        from scipy.ndimage import label,find_objects
        im=images[-1];small=np.asarray(Image.fromarray(t).resize(im.size,Image.Resampling.NEAREST))
        components,count=label(small,structure=np.ones((3,3),dtype='uint8'))
        focus=None;elongation=None
        if count:
            areas=np.bincount(components.ravel());areas[0]=0;largest=int(areas.argmax());yy,xx=np.where(components==largest)
            if len(xx)>=3:
                eigen=np.linalg.eigvalsh(np.cov(np.stack([xx,yy])))
                elongation=float(np.sqrt(eigen[-1]/max(float(eigen[0]),.01)))
            box=[max(0,int(xx.min())-16),max(0,int(yy.min())-16),min(im.width,int(xx.max())+17),min(im.height,int(yy.max())+17)]
            if small.sum()>=16:
                im.crop(box).save(folder/'focus_optical.png',compress_level=1)
                Image.fromarray((components==largest).astype('uint8')*255).crop(box).save(folder/'focus_label.png',compress_level=1)
                focus=box
        for i,im in enumerate(images[:-1]):im.save(folder/f'aux_{i}.png',compress_level=1)
        atomic_write_json(marker,{'source_files':row['files'],'input_sizes':[list(im.size) for im in images],
                                 'guide_version':GUIDE_VERSION,'visible_target_pixels':int(small.sum()),'focus':focus,'elongation':elongation})
    else:assert json.loads(marker.read_text())['source_files']==row['files']
    paths=[('optical_raw',ROOT/row['images'][-1]),('label',ROOT/row['mask']),('label_overlay',folder/'overlay.png'),
           ('optical_at_model_size',folder/'optical_at_input.png'),('target_original_pixels',folder/'target.png')]
    if json.loads(marker.read_text())['focus'] is not None:
        paths.extend([('focus_optical',folder/'focus_optical.png'),('focus_label',folder/'focus_label.png')])
    for i,p in enumerate(row['images'][:-1]):paths.extend([(f'auxiliary_raw_{i}',ROOT/p),(f'auxiliary_at_model_size_{i}',folder/f'aux_{i}.png')])
    return paths

