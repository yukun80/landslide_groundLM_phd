"""逐文件实际清洗、索引分组、固定样本保护与物理复制。"""
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from ..train.data.domain import read_source, training_answer, dhash
from .common import *


def group_key(source, row):
    """GDCLD original的scene只是各子集内序号；不能用同名序号合并不同原图。"""
    if source==NAMES[0] and row.get('source_origin')!='original' and row.get('scene'):
        return source+':'+str(row.get('source_origin',''))+':'+row['scene']
    return source+':'+row['sample_key']


def bounds(row, shape):
    g=row.get('geotransform')
    if not g:return None
    a,b,c,d,e,f=map(float,g);h,w=shape
    corners=[(c+a*x+b*y,f+d*x+e*y) for x,y in [(0,0),(w,0),(0,h),(w,h)]]
    return [min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners)]


def inspect_one(item, dest):
    source,row=item;rid=row['sample_key'];out=Path(dest)/'audit/scan'/f'{rid}.json'
    if out.exists():
        cached=json.loads(out.read_text())
        meta=cached.get('label_identity') or cached.get('source_files',{}).get('mask_hdf5')
        if meta:
            p=source_paths(source,row)['mask_hdf5'];st=p.stat()
            assert st.st_size==meta['size'] and st.st_mtime_ns==meta['mtime_ns'],'cached source changed'
            for key,identity in cached.get('source_files',{}).items():
                stat=source_paths(source,row)[key].stat()
                assert stat.st_size==identity['size'] and stat.st_mtime_ns==identity['mtime_ns'],'cached image changed'
            return cached
    result={'id':rid,'source':source,'source_record':row}
    try:
        paths=source_paths(source,row)
        result['label_identity']=file_meta(paths['mask_hdf5'])
        # 必须实际读标注；清单的positive_pixel_count不替代本次检查。
        with h5py.File(paths['mask_hdf5'],'r') as f:
            label=f['mask'][:];valid=f['valid_mask'][:].astype(bool) if 'valid_mask' in f else np.ones(label.shape,bool)
        if label.ndim!=2 or label.shape!=valid.shape:raise ValueError('label_dimensions')
        if not set(np.unique(label[valid])) <= {0,1}:raise ValueError('illegal_labels')
        if not valid.any():raise ValueError('all_invalid_label')
        if not ((label==1)&valid).any():raise ValueError('no_positive_in_valid_label')
        before={k:file_meta(p) for k,p in paths.items()}
        x,pv,m,v=read_source(source,row)
        if x.ndim!=3 or x.shape[1:]!=m.shape or m.shape!=v.shape:raise ValueError('image_mask_dimensions')
        if not v.any():raise ValueError('no_valid_optical_pixels')
        if not m.any():raise ValueError('no_positive_after_optical_validity')
        for channels in VIEWS[source]:
            if not pv[channels].all(0).any():raise ValueError('required_view_has_no_valid_pixels')
        rgb=x[VIEWS[source][-1]]
        if float(np.ptp(rgb[:,v]))<1e-8:raise ValueError('constant_optical_image')
        preview=np.moveaxis(rgb,0,-1);lo,hi=float(preview.min()),float(preview.max())
        thumb=((preview-lo)/max(hi-lo,1e-6)*255).astype('uint8')
        result.update(status='eligible',height=m.shape[0],width=m.shape[1],
                      positive_pixels=int(m.sum()),valid_pixels=int(v.sum()),
                      image_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                      optical_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                      label_sha256=hashlib.sha256(m.tobytes()+v.tobytes()).hexdigest(),
                      dhash=dhash(thumb),group=group_key(source,row),footprint=bounds(row,m.shape),
                      source_files={k:{'path':str(p),**before[k]} for k,p in paths.items()})
        for k,p in paths.items():
            s=p.stat();assert s.st_size==before[k]['size'] and s.st_mtime_ns==before[k]['mtime_ns'],'source_changed_during_read'
    except (ValueError,OSError,KeyError,AssertionError) as e:
        result.update(status='excluded',reason=f'{type(e).__name__}:{e}')
    atomic_write_json(out,result);return result


def scan(dest=DEFAULT,workers=6):
    initialize(dest);items=[(n,r) for n in NAMES for r in read_jsonl(SOURCE/n/'hdf5/conversion_manifest.jsonl')]
    assert len({r['sample_key'] for n,r in items})==len(items)
    start=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
        for i,r in enumerate(pool.map(partial(inspect_one,dest=dest),items,chunksize=4),1):
            results.append(r)
            if i%200==0:print('SCAN',i,len(items),round(time.monotonic()-start),flush=True)
    write_jsonl_atomic(Path(dest)/'audit/inventory.jsonl',results)
    report={'total':len(results),'eligible':sum(r['status']=='eligible' for r in results),
            'by_source':dict(Counter((r['source']+':'+r['status']) for r in results)),
            'exclusions':dict(Counter(r['reason'] for r in results if r['status']=='excluded'))}
    atomic_write_json(Path(dest)/'audit/scan.json',report);print('SCAN_DONE',report,flush=True)


class SpatialIndex:
    """WGS84切片范围的网格索引；四角按真实尺寸计算，忽略浮点边界误差。"""
    def __init__(self):self.cells=defaultdict(list)
    def keys(self,b):
        return [(i,j) for i in range(math.floor(b[0]/.02),math.floor(b[2]/.02)+1) for j in range(math.floor(b[1]/.02),math.floor(b[3]/.02)+1)]
    def query(self,b):
        if b is None:return []
        found={}
        for key in self.keys(b):
            for rid,split,old in self.cells[key]:
                if min(b[2],old[2])-max(b[0],old[0])>1e-10 and min(b[3],old[3])-max(b[1],old[1])>1e-10:found[rid]=(rid,split,old)
        return list(found.values())
    def add(self,r,split):
        if r['footprint'] is not None:
            for key in self.keys(r['footprint']):self.cells[key].append((r['id'],split,r['footprint']))


def verify_near_identities(rows,union,dest):
    """相似哈希只定位候选；再比较真实RGB结构，只有近乎逐像素一致才合组。"""
    audit_path=Path(dest)/'audit/near_identity_checks.jsonl';meta_path=audit_path.with_suffix('.meta.json')
    fingerprint=digest([(r['id'],r['image_sha256'],r['optical_sha256']) for r in rows])
    if meta_path.exists() and audit_path.exists() and json.loads(meta_path.read_text())['fingerprint']==fingerprint:
        byid={r['id']:r for r in rows}
        for check in read_jsonl(audit_path):
            if check['same_image_group']:union(byid[check['id']]['group'],byid[check['other']]['group'])
        return
    buckets=defaultdict(list);cache={};audit=[]
    def thumbnail(r):
        if r['id'] in cache:return cache[r['id']]
        with h5py.File(r['source_files']['image_hdf5']['path'],'r') as f:x=f['image'][:]
        rgb=x[VIEWS[r['source']][-1]];planes=[]
        for plane in rgb:
            vals=plane[np.isfinite(plane)];lo,hi=np.percentile(vals,[2,98])
            z=np.clip((plane-lo)/max(float(hi-lo),1e-6),0,1);z=np.nan_to_num(z)
            planes.append(np.asarray(Image.fromarray(z.astype('float32')).resize((64,64),Image.Resampling.BILINEAR)))
        out=np.stack(planes);cache[r['id']]=out;return out
    for r in rows:
        candidates={o['id']:o for j in range(4) for o in buckets[(j,(r['dhash']>>(16*j))&65535)]}
        for other in candidates.values():
            if (r['dhash']^other['dhash']).bit_count()>2 or r['optical_sha256']==other['optical_sha256']:continue
            a,b=thumbnail(r),thumbnail(other);delta=float(np.sqrt(np.mean((a-b)**2)))
            textured=min(float(a.std()),float(b.std()))>.08
            corr=float(np.corrcoef(a.flatten(),b.flatten())[0,1]) if textured else 0.0
            identical=textured and delta<.002 and corr>.9999
            if identical:union(r['group'],other['group'])
            audit.append({'id':r['id'],'other':other['id'],'normalized_rgb_rmse':delta,'correlation':corr,'same_image_group':identical})
        for j in range(4):buckets[(j,(r['dhash']>>(16*j))&65535)].append(r)
    write_jsonl_atomic(audit_path,audit)
    atomic_write_json(meta_path,{'fingerprint':fingerprint,'checks':len(audit),'confirmed_same_image':sum(a['same_image_group'] for a in audit)})


def partition(dest=DEFAULT):
    dest=Path(dest)
    if (dest/'review/manifest.json').exists():
        raise RuntimeError('文字任务已经固定样本划分，禁止重新划分；继续已有任务即可')
    initialize(dest)
    allrows=read_jsonl(dest/'audit/inventory.jsonl');good=[r for r in allrows if r['status']=='eligible']
    old={r['id']:r for r in old_rows()};bench={r['id']:r for r in read_jsonl(BENCHMARK/'samples.jsonl')}
    pins={k:r['split'] for k,r in old.items()};pins.update({k:'test' for k in bench})
    byid={r['id']:r for r in good};missing=set(pins)-set(byid)
    conflicts=[{**r,'requested_split':pins[r['id']],'decision':'pending_user_choice'} for r in allrows if r['id'] in missing]
    atomic_write_json(dest/'audit/protected_quality_conflicts.json',{'samples':conflicts,'resolved':not conflicts})
    # 其余样本可独立处理；质量冲突不悄悄保留，也不将未解决数据标成完成。
    assert all(r.get('reason')=='ValueError:constant_optical_image' for r in conflicts),f'protected samples not eligible: {missing}'
    pins={rid:split for rid,split in pins.items() if rid not in missing}
    # 内容一致的图片必须同组；完全相同图片和标注只保留一个代表，优先保留固定样本。
    parent={r['group']:r['group'] for r in good}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    seen={};duplicates=[];selected=[]
    for r in sorted(good,key=lambda r:(r['id'] not in pins,r['id'])):
        key=(r['image_sha256'],r['label_sha256'])
        if key in seen:
            other=seen[key]
            if r['id'] in pins and other['id'] in pins:
                assert pins[r['id']]==pins[other['id']],('fixed_duplicate_cross_split',r['id'],other['id'])
                union(r['group'],other['group'])
            else:
                duplicates.append({'id':r['id'],'status':'excluded','reason':'exact_duplicate','representative':other['id']});continue
        else:seen[key]=r
        selected.append(r)
    optical={}
    for r in selected:
        if r['optical_sha256'] in optical:union(r['group'],optical[r['optical_sha256']]['group'])
        else:optical[r['optical_sha256']]=r
    verify_near_identities(selected,union,dest)
    groups=defaultdict(list)
    for r in selected:groups[find(r['group'])].append(r)
    counts={n:Counter() for n in NAMES};totals=Counter(r['source'] for r in selected);indices={n:SpatialIndex() for n in NAMES}
    output=[];excluded=duplicates;fixed=[];free=[]
    for key,items in groups.items():
        folds={pins[r['id']] for r in items if r['id'] in pins}
        assert len(folds)<=1,('fixed_parent_conflict',key,folds)
        (fixed if folds else free).append((key,items,next(iter(folds)) if folds else None))
    def assign(items,split,is_fixed):
        for r in sorted(items,key=lambda r:(r['id'] not in pins,r['id'])):
            opposite=[hit for hit in indices[r['source']].query(r['footprint']) if hit[1]!=split]
            if opposite:
                if r['id'] in pins:raise RuntimeError(('protected_geographic_overlap',r['id'],opposite[:3]))
                excluded.append({'id':r['id'],'status':'excluded','reason':'cross_split_overlap','conflicts':[x[0] for x in opposite]});continue
            r={**r,'split':split,'group':find(r['group']),'protected':r['id'] in pins,'reuse_text':r['id'] in old,'independent_benchmark':r['id'] in bench}
            output.append(r);counts[r['source']][split]+=1;indices[r['source']].add(r,split)
    # 先安排所有固定样本，再扩展其同组样本，避免未固定邻片挤占固定测试位置。
    for key,items,split in fixed:assign([r for r in items if r['id'] in pins],split,True)
    for key,items,split in fixed:assign([r for r in items if r['id'] not in pins],split,False)
    for key,items,_ in sorted(free,key=lambda g:digest([SEED,g[0]])):
        neighbors={hit[1] for r in items for hit in indices[r['source']].query(r['footprint'])}
        if len(neighbors)>1:
            excluded.extend({'id':r['id'],'status':'excluded','reason':'bridges_reserved_splits'} for r in items);continue
        if neighbors:split=next(iter(neighbors))
        else:
            n=items[0]['source'];split=max(SPLITS,key=lambda s:totals[n]*{'train':.8,'val':.1,'test':.1}[s]-counts[n][s])
        assign(items,split,False)
    assert set(pins)<={r['id'] for r in output}
    assert all(r['split']==pins[r['id']] for r in output if r['id'] in pins)
    write_jsonl_atomic(dest/'audit/selected.jsonl',sorted(output,key=lambda r:(SPLITS.index(r['split']),r['source'],r['id'])))
    write_jsonl_atomic(dest/'audit/partition_exclusions.jsonl',excluded)
    # 近似哈希只报告候选，不凭相似删除低对比度真样本；像素同一已在上面合组。
    buckets=defaultdict(list);near=[]
    for r in output:
        candidates={k['id']:k for j in range(4) for k in buckets[(j,(r['dhash']>>(16*j))&65535)]}
        for other in candidates.values():
            if other['split']!=r['split'] and (r['dhash']^other['dhash']).bit_count()<=2:
                near.append({'id':r['id'],'other':other['id'],'same_optical_content':r['optical_sha256']==other['optical_sha256'],'status':'similarity_only_no_identity_evidence'})
        for j in range(4):buckets[(j,(r['dhash']>>(16*j))&65535)].append(r)
    write_jsonl_atomic(dest/'audit/near_candidates.jsonl',near)
    report={'selected':len(output),'excluded_after_cleaning':len(excluded),'counts':{n:dict(c) for n,c in counts.items()},
            'fold_counts':dict(Counter(r['split'] for r in output)),'pinned':len(pins),'old_text_reuse':len(set(old)&set(byid)),'benchmark_test':len(bench),'unresolved_protected_quality':len(conflicts),'approximate_target':[.8,.1,.1],
            'similarity_candidates':len(near),'limitations':['Landslide4Sense lacks location metadata; no claim of cross-region independence','positive-only data cannot measure background false alarms','fixed folds and true groups override exact percentages']}
    atomic_write_json(dest/'audit/partition.json',report);print('PARTITION_DONE',report,flush=True)


def export_one(r,dest,ranges,old):
    dest=Path(dest);folder=dest/'samples'/r['id'];done=folder/'prepared.json'
    if done.exists():return json.loads(done.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    raw=dest/'raw'/r['source']/r['id'];raw.mkdir(parents=True,exist_ok=True)
    local={}
    for key,meta in r['source_files'].items():
        source=Path(meta['path']);target=raw/('image.h5' if key=='image_hdf5' else 'mask.h5')
        if not target.exists():
            temp=target.with_suffix('.partial');shutil.copy2(source,temp);assert sha256_file(temp)==meta['sha256'];temp.replace(target)
        assert sha256_file(target)==meta['sha256']
        s=source.stat();assert s.st_size==meta['size'] and s.st_mtime_ns==meta['mtime_ns'],'source changed'
        local[key]=str(target.relative_to(ROOT))
    # 用本项目副本实际读取全部波段；源文件只保留为来源证据。
    from ..train.data import domain
    rr={**r['source_record'],**{k:str(ROOT/p) for k,p in local.items()}}
    x,pv,m,v=read_source(r['source'],rr)
    assert m.any() and int(m.sum())==r['positive_pixels']
    files=[]
    for i,channels in enumerate(VIEWS[r['source']]):
        f=folder/f'view_{i}.png';planes=[]
        for c in channels:
            lo,hi=ranges[r['source']][c];a=np.clip((x[c]-lo)/max(hi-lo,1e-6),0,1)
            planes.append((np.where(pv[c],a,0)*255).astype('uint8'))
        img=np.stack(planes,-1)
        if r['id'] in old:
            with Image.open(ROOT/old[r['id']]['images'][i]) as oldimg:assert np.array_equal(img,np.asarray(oldimg)),('old_view_changed',r['id'])
            shutil.copy2(ROOT/old[r['id']]['images'][i],f)
        else:Image.fromarray(img).save(f,compress_level=1)
        files.append(str(f.relative_to(ROOT)))
    for key,arr in [('mask',m),('valid',v)]:
        f=folder/f'{key}.png'
        if r['id'] in old:
            assert np.array_equal(arr,np.asarray(Image.open(ROOT/old[r['id']][key]))>0)
            shutil.copy2(ROOT/old[r['id']][key],f)
        else:Image.fromarray(arr.astype('uint8')*255).save(f,compress_level=1)
    base_answer=training_answer(m,v)
    if r['id'] in old:
        o=old[r['id']];assert o['split']==r['split']
        answer=o['answer'];opt=o['answer_optical']
        assert answer.startswith(base_answer.split(' <bbox>')[0]) and opt.startswith(base_answer.split(' <bbox>')[0])
    else:answer=opt=base_answer
    row={**r,'images':files,'mask':str((folder/'mask.png').relative_to(ROOT)),'valid':str((folder/'valid.png').relative_to(ROOT)),
         'raw_files':local,'answer':answer,'answer_optical':opt,'description_status':'reused' if r['id'] in old else 'pending'}
    row['files']={str(f.relative_to(ROOT)):sha256_file(f) for f in [folder/f'view_{i}.png' for i in range(len(files))]+[folder/'mask.png',folder/'valid.png']}
    atomic_write_json(done,row);return row


def _export_init(dest,ranges,old):
    global _EXPORT_CONTEXT
    _EXPORT_CONTEXT=(dest,ranges,old)


def _export_worker(row):
    return export_one(row,*_EXPORT_CONTEXT)


def export(dest=DEFAULT,workers=6):
    dest=Path(dest);build=initialize(dest);rows=read_jsonl(dest/'audit/selected.jsonl');old={r['id']:r for r in old_rows()}
    atomic_write_json(dest/'display_ranges.json',build['display_ranges'])
    priority=pilot_rows(rows);priority_ids={r['id'] for r in priority}
    bysource=[[r for r in rows if r['source']==n and r['id'] not in priority_ids] for n in NAMES]
    scheduled=priority+[pool[i] for i in range(max(map(len,bysource))) for pool in bysource if i<len(pool)]
    outputs=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"),initializer=_export_init,initargs=(dest,build["display_ranges"],old)) as pool:
        for i,r in enumerate(pool.map(_export_worker,scheduled,chunksize=2),1):
            outputs.append(r)
            if i%100==0:
                print('EXPORT',i,len(rows),round(time.monotonic()-start),flush=True)
                atomic_write_json(dest/'audit/export_progress.json',{'completed':i,'total':len(rows),'complete':False})
    byid={r['id']:r for r in outputs};outputs=[byid[r['id']] for r in rows]
    write_jsonl_atomic(dest/'prepared.jsonl',outputs)
    for split in SPLITS:write_jsonl_atomic(dest/f'{split}.pending.jsonl',[r for r in outputs if r['split']==split])
    atomic_write_json(dest/'display_ranges.json',build['display_ranges'])
    atomic_write_json(dest/'audit/export_progress.json',{'completed':len(rows),'total':len(rows),'complete':True})
    print('EXPORT_DONE',len(outputs),flush=True)
