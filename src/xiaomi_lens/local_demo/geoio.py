"""彩色TIF的窗口读取、有效区域与定位框地理坐标转换。无模型依赖。"""
import math,re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.windows import Window
from rasterio.warp import transform as project
from affine import Affine
from PIL import Image
from shapely.geometry import Polygon, box, mapping
from shapely.affinity import translate


def inspect_image(path):
    """只接收已定义RGB的uint8 TIF；普通显示图供辅助观测使用。"""
    path=Path(path)
    if path.suffix.lower() not in ('.tif','.tiff'):
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):raise ValueError('仅支持彩色TIF或普通显示图片')
        with Image.open(path) as im:
            if im.mode not in ('RGB','RGBA','L'):raise ValueError('请先制作普通彩色图片')
            im.verify()
        with Image.open(path) as im:w,h=im.size
        return {'width':w,'height':h,'tif':False,'crs':None,'transform':None,'units':None}
    with rasterio.open(path) as d:
        if d.subdatasets:raise ValueError('不支持包含多个影像页面的TIF，请先导出单幅彩色TIF')
        if d.gcps[0] or d.rpcs:raise ValueError('含控制点或卫星定位参数，请先完成地理校正')
        colors=list(d.colorinterp)
        if d.count not in (3,4) or any(c not in colors for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)) or any(t!='uint8' for t in d.dtypes):
            raise ValueError('仅支持具有明确红绿蓝通道的8位彩色TIF，请先配色导出')
        if d.count==4 and ColorInterp.alpha not in colors:raise ValueError('第四通道必须明确为透明通道')
        if not all(math.isfinite(v) for v in d.transform) or abs(d.transform.determinant)<1e-15:raise ValueError('图片与地图的对应关系无效')
        with Image.open(path) as im:
            if getattr(im,'n_frames',1)>1:raise ValueError('不支持多页TIF')
        units=(re.findall(r'UNIT\["([^"]+)',d.crs.to_wkt())[-1] if d.crs.is_geographic else d.crs.linear_units) if d.crs else None
        return {'width':d.width,'height':d.height,'tif':True,'crs':d.crs.to_wkt() if d.crs else None,'crs_name':d.crs.to_string() if d.crs else None,'transform':list(d.transform)[:6],'units':units,'rgb_indexes':[colors.index(c)+1 for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)]}


def read_window(path, meta, window=None, preview_edge=None):
    """读取RGB和有效区；透明/无数据无效，黑色不自动无效。"""
    x,y,w,h=window or (0,0,meta['width'],meta['height'])
    ratio=min(1,preview_edge/max(w,h)) if preview_edge else 1
    outw,outh=max(1,round(w*ratio)),max(1,round(h*ratio))
    if meta['tif']:
        with rasterio.open(path) as d:
            win=Window(x,y,w,h)
            a=d.read(meta['rgb_indexes'],window=win,out_shape=(3,outh,outw),resampling=Resampling.bilinear).transpose(1,2,0)
            valid=(d.read_masks(meta['rgb_indexes'],window=win,out_shape=(3,outh,outw),resampling=Resampling.nearest)>0).all(0)
            if ColorInterp.alpha in d.colorinterp:
                valid &= d.read(list(d.colorinterp).index(ColorInterp.alpha)+1,window=win,out_shape=(outh,outw),resampling=Resampling.nearest)>0
    else:
        with Image.open(path) as im:
            crop=im.crop((x,y,x+w,y+h)).convert('RGBA').resize((outw,outh),Image.Resampling.BILINEAR)
            a=np.array(crop);valid=a[:,:,3]>0;a=a[:,:,:3]
    a[~valid]=0
    return Image.fromarray(a),valid


def same_grid(a,b):
    if (a['width'],a['height'])!=(b['width'],b['height']):raise ValueError('辅助图与光学图尺寸不一致')
    if a['crs'] and b['crs']:
        if rasterio.crs.CRS.from_wkt(a['crs'])!=rasterio.crs.CRS.from_wkt(b['crs']) or not np.allclose(a['transform'],b['transform'],atol=1e-9,rtol=0):
            raise ValueError('辅助图与光学图的地理位置不一致，须先对齐')


def geographic_box(pixel_xyxy, meta):
    """使用像素外边界，保存原坐标四角；经纬度沿边加密且在日期线拆分。"""
    if not meta.get('crs'):return {'geographic_status':'无地理信息，仅提供像素坐标'}
    x0,y0,x1,y1=pixel_xyxy
    pixel=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    t=Affine(*meta['transform']);corners=[t*p for p in pixel]
    out={'native_corners_xy':corners,'native_bounds_xyxy':[min(p[0] for p in corners),min(p[1] for p in corners),max(p[0] for p in corners),max(p[1] for p in corners)],'native_crs':meta['crs'],'native_units':meta['units'],'corner_order':'左上,右上,右下,左下; x/y 或 经度/纬度'}
    try:
        lon,lat=project(meta['crs'],'EPSG:4326',*[list(v) for v in zip(*corners)])
        if not all(math.isfinite(v) for v in lon+lat):raise ValueError('坐标转换结果不是有限值')
        out['wgs84_corners_lonlat']=[[((x+180)%360)-180,y] for x,y in zip(lon,lat)]
        dense=[]
        for a,b in zip(corners,corners[1:]+corners[:1]):
            dense.extend([(a[0]+(b[0]-a[0])*k/16,a[1]+(b[1]-a[1])*k/16) for k in range(16)])
        xx,yy=project(meta['crs'],'EPSG:4326',*[list(v) for v in zip(*dense)])
        unwrapped=[]
        for x,y in zip(xx,yy):
            if not math.isfinite(x+y) or not -90<=y<=90:raise ValueError('经纬度无效')
            if unwrapped:x+=360*round((unwrapped[-1][0]-x)/360)
            unwrapped.append((x,y))
        poly=Polygon(unwrapped)
        if not poly.is_valid or poly.is_empty:raise ValueError('转换后的范围无效')
        pieces=[]
        for k in range(math.floor((poly.bounds[0]+180)/360),math.floor((poly.bounds[2]+180)/360)+1):
            part=poly.intersection(box(-180+360*k,-90,180+360*k,90))
            if part.area>0:
                shifted=translate(part,xoff=-360*k)
                pieces.extend(list(shifted.geoms) if shifted.geom_type=='MultiPolygon' else [shifted])
        out.update(geographic_status='成功',wgs84_geometry={'type':'MultiPolygon','coordinates':[mapping(p)['coordinates'] for p in pieces]},wgs84_bounds_parts=[list(p.bounds) for p in pieces])
    except Exception as e:out.update(geographic_status='经纬度转换失败',geographic_error=str(e))
    return out
