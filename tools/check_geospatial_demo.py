"""坐标/窗口的已知答案检查；--live另用真实带地理信息TIF实际预测。"""
import argparse,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from affine import Affine
from PIL import Image
from xiaomi_lens.local_demo import geojob,geoio
from xiaomi_lens.local_demo.boxes import parse_boxes
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/geospatial-check'


def fixture(path,w=660,h=520,transform=None,crs='EPSG:4326',rgba=True):
    a=np.zeros((4 if rgba else 3,h,w),dtype='uint8');a[0]=80;a[1]=120;a[2]=30
    if rgba:a[3]=255;a[3,:20,:]=0
    with rasterio.open(path,'w',driver='GTiff',width=w,height=h,count=len(a),dtype='uint8',transform=transform or from_origin(119,30,.0001,.0001),crs=crs) as d:
        d.write(a);d.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)+( (ColorInterp.alpha,) if rgba else ())


def main():
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True)
    checks={}
    d=parse_boxes('<think><bbox>[0,0,1,1]</bbox></think><bbox>[100,200,900,800]</bbox><bbox>[0,0,0,1]</bbox><bbox>[-1,0,1,2]</bbox><bbox>oops</bbox>',200,100,(10,20))
    assert d['boxes'][0]['pixel_xyxy']==[30,40,190,100] and sum(b['valid'] for b in d['boxes'])==1
    assert parse_boxes('none',20,10)['warnings'];assert parse_boxes('<bbox>[0,0,1000,1000]',20,10)['warnings'];checks['boxes_validation']=True
    f=OUT/'known.tif';fixture(f);meta=geoio.inspect_image(f)
    geo=geoio.geographic_box([0,0,100,200],meta);assert np.allclose(geo['native_corners_xy'],[(119,30),(119.01,30),(119.01,29.98),(119,29.98)])
    rot=dict(meta,transform=[2,.3,100,.4,-2,200]);g=geoio.geographic_box([0,0,10,20],rot);assert np.allclose(g['native_corners_xy'],[(100,200),(120,204),(126,164),(106,160)])
    bad=dict(meta,crs='INVALID');assert geoio.geographic_box([0,0,1,1],bad)['geographic_status']=='经纬度转换失败'
    date=dict(meta,transform=[.02,0,179,0,-.01,10]);g=geoio.geographic_box([0,0,100,100],date);assert len(g['wgs84_geometry']['coordinates'])==2
    assert all(b[2]-b[0]<5 for b in g['wgs84_bounds_parts']);checks['corners_rotation_failure_dateline']=True
    fixture(OUT/'plain.tif',100,80,crs=None);plain=geoio.inspect_image(OUT/'plain.tif');assert not plain['crs'];assert '仅提供像素' in geoio.geographic_box([0,0,10,10],plain)['geographic_status']
    im,v=geoio.read_window(f,meta);assert not v[:20].any() and v[20:].all()
    fixture(OUT/'black.tif',32,32,rgba=False)
    with rasterio.open(OUT/'black.tif','r+') as ds:ds.write(np.zeros((3,32,32),dtype='uint8'))
    assert geoio.read_window(OUT/'black.tif',geoio.inspect_image(OUT/'black.tif'))[1].all();checks['nodata_not_black']=True
    for w,h in [(1,1),(512,512),(513,600),(1024,1024),(2000,80)]:
        owner=np.zeros((h,w),dtype='uint8');tiles=geojob.tiles_for(w,h)
        for t in tiles:
            x,y,ww,hh=t['window'];cx,cy,cw,ch=t['owner'];assert x<=cx and y<=cy and cx+cw<=x+ww and cy+ch<=y+hh
            owner[cy:cy+ch,cx:cx+cw]+=1
        assert (owner==1).all()
    checks['tile_coverage_edges_small_images']=True
    wrong=dict(meta,transform=[1,0,0,0,-1,0])
    try:geoio.same_grid(meta,wrong);raise AssertionError('accepted wrong grid')
    except ValueError:pass
    checks['aux_grid_rejection']=True
    job=Path(geojob.prepare([str(f)],'test'));calls=[];fail={'000001'}
    def fake(row,folder,prompt):
        name=row['id'];calls.append(name)
        if name in fail:fail.remove(name);raise RuntimeError('test injected failure')
        with Image.open(row['images'][-1]) as im:size=im.size
        mask=np.zeros((size[1],size[0]),dtype='uint8');mask[10:40,10:40]=255;Image.fromarray(mask).save(folder/'mask.png')
        d={'answer':'test','raw_answer':'<bbox>[10,10,100,100]</bbox><|mask|>'};geojob.save(folder/'result.json',d);return d
    it=geojob.run(job,predictor=fake);next(it);next(it);geojob.request_stop(job);list(it)
    assert json.loads((job/'status.json').read_text())['state']=='stopped'
    with rasterio.open(job/'segmentation.tif') as ds:assert (ds.read(1)==254).any()
    list(geojob.run(job,predictor=fake));assert json.loads((job/'status.json').read_text())['failed']
    list(geojob.run(job,predictor=fake));assert calls.count('000000')==1
    with rasterio.open(job/'segmentation.tif') as ds:
        a=ds.read(1);assert not (a==254).any() and (a[:20]==255).all();assert ds.crs.to_epsg()==4326 and ds.transform==from_origin(119,30,.0001,.0001)
    checks['stop_resume_failure_retry_invalid_export']=True
    with zipfile.ZipFile(job/'geographic_results.zip') as z:assert 'segmentation.tif' in z.namelist() and 'boxes.geojson' in z.namelist()
    cfg=json.loads((job/'job.json').read_text());cfg['prompt']='changed';geojob.save(job/'job.json',cfg)
    try:list(geojob.run(job,predictor=fake));raise AssertionError('accepted changed task')
    except ValueError:pass
    checks['resume_contract_rejection']=True
    if args.live:
        src=Path('/home/yukun80/codes/datasets/GDCLD/cleaned/test/image/gdcld_test_test_palu_palu_1_r06144_c03072.tif')
        before=geojob.sha(src);from xiaomi_lens.local_demo.service import PROMPT
        live=Path(geojob.prepare([str(src)],PROMPT));it=geojob.run(live);next(it);next(it);geojob.request_stop(live);list(it)
        first=json.loads((live/'status.json').read_text());assert first['state']=='stopped' and len(first['completed'])==1
        for update in geojob.run(live):print(update['state'],update['completed'],update['total'],flush=True)
        status=json.loads((live/'status.json').read_text());assert status['state']=='complete' and len(status['completed'])==9 and not status['failed'];assert before==geojob.sha(src)
        checks['real_tif_9_tiles_stop_resume']={'job':str(live),'source':str(src),'sha256':before,'source_unchanged':True}
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print(checks)
if __name__=='__main__':main()
