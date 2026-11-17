"""可停止/恢复的大图扫描；按窗口读写，结果与历史预测分离。"""
import json,uuid,shutil,fcntl,hashlib,zipfile,os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from affine import Affine
from PIL import Image
from .geoio import inspect_image,read_window,same_grid,geographic_box
from .boxes import parse_boxes,draw_view

ROOT=Path(__file__).resolve().parents[3]
JOBS=ROOT/'outputs/benchmark500/geospatial'


def save(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def axis_windows(n,edge=512,overlap=128):
    """返回窗口及最近中心的归属范围；相同距离归较早窗口。"""
    starts=[0]
    while starts[-1]+edge<n:starts.append(starts[-1]+edge-overlap)
    ends=[min(n,x+edge) for x in starts];centers=[(a+b)/2 for a,b in zip(starts,ends)]
    bounds=[0]+[max(starts[i+1],min(ends[i],int(np.floor((a+b)/2+.5)))) for i,(a,b) in enumerate(zip(centers,centers[1:]))]+[n]
    return [(s,e-s,bounds[i],bounds[i+1]-bounds[i]) for i,(s,e) in enumerate(zip(starts,ends))]


def tiles_for(w,h):
    return [{'id':f'{i:06d}','window':[x,y,ww,hh],'owner':[cx,cy,cw,ch]} for i,(x,y,ww,hh,cx,cy,cw,ch) in enumerate((x,y,ww,hh,cx,cy,cw,ch) for y,hh,cy,ch in axis_windows(h) for x,ww,cx,cw in axis_windows(w))]


def resolve_job(job):
    p=Path(job).resolve()
    if p.parent!=JOBS.resolve() or not (p/'job.json').exists():raise ValueError('无效任务，请先准备图片')
    return p


def prepare(paths,prompt,aligned=False):
    """复制上传输入并固定问题、模型、分块和有效区；不运行模型。"""
    if not prompt.strip():raise ValueError('请填写问题')
    metas=[inspect_image(p) for p in paths]
    if not metas[-1]['tif']:raise ValueError('此入口的光学图须为已配好颜色的TIF')
    if len(paths)>1 and not aligned:raise ValueError('请确认辅助图与光学图已对齐')
    for meta in metas[:-1]:same_grid(metas[-1],meta)
    from .service import verify_model
    model=verify_model()
    job=JOBS/uuid.uuid4().hex;job.mkdir(parents=True);copied=[]
    for i,p in enumerate(paths):
        dest=job/f'source_{i}{Path(p).suffix.lower()}';shutil.copyfile(p,dest)
        copied.append({'path':dest.name,'sha256':sha(dest),'meta':metas[i]})
    meta=metas[-1];tiles=tiles_for(meta['width'],meta['height'])
    record={'schema':1,'inputs':copied,'prompt':prompt,'model_sha256':model['checkpoint_sha256'],'tile_edge':512,'overlap':128,'merge':'nearest-center; equal distances use earlier tile','tiles':tiles,'codes':{'background':0,'target':1,'pending_or_failed':254,'invalid':255}}
    save(job/'job.json',record)
    save(job/'contract.json',{'job_sha256':sha(job/'job.json')})
    profile=dict(driver='GTiff',width=meta['width'],height=meta['height'],count=1,dtype='uint8',nodata=255,compress='deflate',tiled=True,BIGTIFF='IF_SAFER')
    if meta['transform']:profile['transform']=Affine(*meta['transform'])
    if meta['crs']:profile['crs']=meta['crs']
    initial_counts={254:0,255:0}
    with rasterio.open(job/'segmentation.tif','w',**profile) as dst:
        for tile in tiles:
            _,v=read_window(job/copied[-1]['path'],meta,tile['owner'])
            dst.write(np.where(v,254,255).astype('uint8'),1,window=Window(*tile['owner']))
            initial_counts[254]+=int(v.sum());initial_counts[255]+=int((~v).sum())
        dst.update_tags(value_0='background',value_1='predicted target',value_254='pending or failed',value_255='invalid')
    save(job/'status.json',{'state':'prepared','completed':{},'failed':{},'total':len(tiles),'initial_counts':initial_counts})
    refresh(job)
    return str(job)


def request_stop(job):
    p=resolve_job(job);(p/'stop').touch()
    return '已请求停止；正在处理的一块完成后停止，已完成结果会保留。'


def run(job,predictor=None):
    """逐块产生进度；锁防重复；已完成块核验后复用，失败块可重试。"""
    job=resolve_job(job)
    with (job/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('此任务正在运行，请勿重复开始')
        config=json.loads((job/'job.json').read_text());status=json.loads((job/'status.json').read_text())
        if sha(job/'job.json')!=json.loads((job/'contract.json').read_text())['job_sha256']:raise ValueError('任务设置改变，不能继续旧任务')
        for inp in config['inputs']:
            if sha(job/inp['path'])!=inp['sha256']:raise ValueError('输入图片内容改变，不能混用原任务')
        if config['tile_edge']!=512 or config['overlap']!=128:raise ValueError('分块设置不匹配')
        if predictor is None:
            from .service import SERVICE,verify_model
            if verify_model()['checkpoint_sha256']!=config['model_sha256']:raise ValueError('模型改变，不能继续旧任务')
            predictor=SERVICE.run
        (job/'stop').unlink(missing_ok=True)
        status['state']='running';save(job/'status.json',status);yield refresh(job)
        meta=config['inputs'][-1]['meta']
        for tile in config['tiles']:
            if (job/'stop').exists():break
            name=tile['id'];folder=job/'tiles'/name
            if name in status['completed']:
                for file,h in status['completed'][name]['hashes'].items():
                    if sha(folder/file)!=h:raise ValueError('已完成结果被修改，不能静默复用')
                continue
            try:
                folder.mkdir(parents=True,exist_ok=True);images=[];valid=None
                for i,inp in enumerate(config['inputs']):
                    image,v=read_window(job/inp['path'],inp['meta'],tile['window'])
                    path=folder/f'input_{i}.png';image.save(path);images.append(str(path))
                    if i==len(config['inputs'])-1:valid=v
                if not valid.any():
                    result={'raw_answer':'','answer':'该块全部为无效区域，未调用模型','skipped_invalid':True}
                    mask=np.zeros(valid.shape,dtype='uint8');save(folder/'result.json',result);Image.fromarray(mask).save(folder/'mask.png')
                else:
                    result=predictor({'id':name,'images':images},folder,config['prompt'])
                    mask=(np.array(Image.open(folder/'mask.png'))>0).astype('uint8')
                    if mask.shape!=valid.shape:raise ValueError('预测区域尺寸不匹配')
                coords=parse_boxes(result.get('raw_answer',''),tile['window'][2],tile['window'][3],tile['window'][:2])
                if result.get('skipped_invalid'):coords['warnings']=[]
                for b in coords['boxes']:
                    b['tile']=name
                    if b['valid']:b.update(geographic_box(b['pixel_xyxy'],meta))
                save(folder/'boxes.json',coords)
                x,y,_,_=tile['window'];cx,cy,cw,ch=tile['owner']
                values=np.where(valid,mask,255).astype('uint8')[cy-y:cy-y+ch,cx-x:cx-x+cw]
                with rasterio.open(job/'segmentation.tif','r+') as dst:dst.write(values,1,window=Window(*tile['owner']))
                status['completed'][name]={'pixel_counts':{str(k):int((values==k).sum()) for k in [0,1,255]},'skipped_invalid':bool(result.get('skipped_invalid')),'hashes':{f:sha(folder/f) for f in ['result.json','mask.png','boxes.json']}}
                status['failed'].pop(name,None)
            except Exception as e:
                status['failed'][name]=str(e)
            save(job/'status.json',status);yield refresh(job)
        status['state']='stopped' if (job/'stop').exists() else ('completed_with_failures' if status['failed'] else 'complete')
        save(job/'status.json',status);yield refresh(job,archive=True)


def refresh(job,archive=False):
    """生成有限尺寸的整图预览与完整坐标；254保持灰蓝，255保持暗灰。"""
    job=resolve_job(job);config=json.loads((job/'job.json').read_text());status=json.loads((job/'status.json').read_text())
    meta=config['inputs'][-1]['meta'];image,_=read_window(job/config['inputs'][-1]['path'],meta,preview_edge=1536)
    if not (job/'source_preview.png').exists():image.save(job/'source_preview.png')
    with rasterio.open(job/'segmentation.tif') as d:
        mask=d.read(1,out_shape=(image.height,image.width),resampling=Resampling.nearest)
        counts={0:0,1:0,254:status['initial_counts']['254'],255:status['initial_counts']['255']}
        for record in status['completed'].values():
            for k in (0,1):counts[k]+=record['pixel_counts'][str(k)];counts[254]-=record['pixel_counts'][str(k)]
    boxes=[];answers=[];warnings=[]
    for tile in config['tiles']:
        name=tile['id'];folder=job/'tiles'/name
        if name not in status['completed']:continue
        data=json.loads((folder/'boxes.json').read_text());warnings.extend([f'{name}: {w}' for w in data['warnings']]);boxes.extend(data['boxes'])
        result=json.loads((folder/'result.json').read_text());answers.append({'tile':name,'window':tile['window'],'answer':result['answer'],'raw_answer':result['raw_answer']})
    # 只去掉完全相同的框；交叠不代表同一处滑坡。
    unique={};invalid=[]
    for b in boxes:
        if not b['valid']:invalid.append(b);continue
        key=tuple(b['pixel_xyxy'])
        if key in unique:unique[key]['tiles'].append(b['tile'])
        else:unique[key]=dict(b,tiles=[b['tile']])
    boxes=list(unique.values())+invalid
    for i,b in enumerate(boxes):b['number']=i+1
    save(job/'boxes.json',{'boxes':boxes,'warnings':warnings,'source':'model final answers','pixel_convention':'outer boundaries; x right,y down','meta':meta})
    save(job/'answers.json',answers)
    features=[{'type':'Feature','geometry':b['wgs84_geometry'],'properties':{'number':b['number'],'tiles':b['tiles'],'pixel_xyxy':b['pixel_xyxy']}} for b in boxes if b.get('wgs84_geometry')]
    save(job/'boxes.geojson',{'type':'FeatureCollection','features':features})
    for m in (False,True):
        for b in (False,True):
            im=draw_view(image,mask,boxes,m,b,(image.width/meta['width'],image.height/meta['height']))
            a=np.array(im);a[mask==254]=(a[mask==254]*.55+np.array([110,130,160])*.45).astype('uint8');a[mask==255]=[45,45,45]
            tmp=job/f'view_{int(m)}_{int(b)}.partial.png';Image.fromarray(a).save(tmp);tmp.replace(job/f'view_{int(m)}_{int(b)}.png')
    Image.fromarray(np.select([mask==1,mask==254,mask==255],[255,128,64],default=0).astype('uint8')).save(job/'mask_preview.png')
    summary={'state':status['state'],'completed':len(status['completed']),'total':status['total'],'failed':status['failed'],'pixel_counts':counts,'box_warnings':warnings,'geographic_warnings':[b['geographic_status'] for b in boxes if b.get('valid') and b.get('geographic_status')!='成功'],'description':'仅汇总预测覆盖；逐块回答见answers.json，不合成整图地质结论'}
    save(job/'summary.json',summary)
    if archive:
        tmp=job/'download.partial'
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for f in ['job.json','contract.json','status.json','summary.json','boxes.json','boxes.geojson','answers.json','segmentation.tif','source_preview.png','mask_preview.png','view_1_1.png']:z.write(job/f,f)
            for name in status['completed']:
                for f in ['result.json','boxes.json','mask.png','overlay.png']:
                    path=job/'tiles'/name/f
                    if path.exists():z.write(path,str(path.relative_to(job)))
        tmp.replace(job/'geographic_results.zip')
    return summary
