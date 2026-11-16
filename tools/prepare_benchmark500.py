"""只读筛选benchmark，复制独立样本；来源标签只用于恢复有效区和核对。"""
import hashlib,json,random
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
from xiaomi_lens.train.data.domain import ROOT,DEST,SOURCE,NAMES,jsonlines,write_json,dhash,overlaps
B=Path('/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/full')
OUT=ROOT/'datasets/landslide_benchmark_500'
MAP=dict(zip(['gdcld','landslide4sense','multimodal_landslide'],NAMES))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resize(a):
 return torch.nn.functional.interpolate(torch.tensor(a.astype('float32'))[None,None],size=(224,224),mode='nearest')[0,0].numpy().astype(bool)
def hashes(r):
 # 标签和有效区相同不能证明图像重复（全有效图尤其常见）。
 found={r[k] for k in ['source_image_sha256','source_image_slice_sha256','source_record_sha256','record_sha256'] if isinstance(r.get(k),str)}
 if isinstance(r.get('source_sha256'),dict) and r['source_sha256'].get('rgb'):
  found.add(r['source_sha256']['rgb'])
 return found

def main():
 torch.set_num_threads(4)
 if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text()).get('complete'): raise RuntimeError('已有完整数据，禁止覆盖')
 OUT.mkdir(parents=True,exist_ok=True)
 old=[r for s in ['train','val','test'] for r in jsonlines(DEST/f'{s}.jsonl')]
 index=jsonlines(B/'index.jsonl'); bykey={r['source_sample_key']:r for r in index}
 excluded_ids={r['id'] for r in old}
 groups={(bykey[r['id']]['source'],bykey[r['id']]['source_group_id']) for r in old if r['id'] in bykey and bykey[r['id']].get('source_group_id')}
 excluded_hashes=set().union(*(hashes(r['source_record']) for r in old))
 fps=[(r['source'],r['footprint']) for r in old if r.get('footprint')]
 perceptual=[r['dhash'] for r in old]; contents=set(); ranges=json.loads((DEST/'display_ranges.json').read_text())
 selected=[];audit=[];rng=random.Random(20261116)
 for source,quota in zip(MAP,[200,150,150]):
  pool=[r for r in index if r['source']==source]; rng.shuffle(pool); pool.sort(key=lambda r:['test','val','train'].index(r['split']))
  count=0
  for r in pool:
   key=r['source_sample_key'];reason=None
   if key in excluded_ids: reason='previous_1000_id'
   elif r.get('source_group_id') and (source,r['source_group_id']) in groups: reason='known_same_original_group'
   elif hashes(r['provenance']) & excluded_hashes: reason='source_content_duplicate'
   if reason: audit.append({'id':key,'reason':reason});continue
   try:
    box=None;g=r['provenance'].get('geotransform')
    if g:
     h,w=r['resize']['original_size'];a,b,c,d,e,f=g
     corners=[(c+a*x+b*y,f+d*x+e*y) for x,y in [(0,0),(w,0),(0,h),(w,h)]]
     box=[min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners)]
     if any(n==MAP[source] and overlaps(box,p) for n,p in fps):raise ValueError('geographic_overlap')
    with h5py.File(B/r['storage']['shard']) as f:
     i=r['storage']['row']; arrays={}
     f.visititems(lambda name,obj: arrays.update({name:obj[i]}) if isinstance(obj,h5py.Dataset) else None)
    rgb_idx=[3,2,1] if source=='landslide4sense' else [0,1,2]
    x=arrays['optical'];pv=arrays['optical_pixel_valid'].astype(bool)&arrays['optical_channel_valid'][:,None,None].astype(bool)&np.isfinite(x)
    m=arrays['mask'][0];assert set(np.unique(m))<= {0,1}
    labelpath=SOURCE/MAP[source]/('hdf5' if source=='landslide4sense' else '')/r['provenance']['mask_hdf5']
    with h5py.File(labelpath) as f:
     native=f['mask'][:].squeeze(); valid=f['valid_mask'][:].squeeze().astype(bool) if 'valid_mask' in f else np.ones(native.shape,bool)
    assert np.array_equal(resize((native==1)&valid),m>0),'source_label_mismatch'
    v=resize(valid)&pv[rgb_idx].all(0);m=(m>0)&v
    if not m.any():raise ValueError('no_target_in_valid_region')
    rgb=np.moveaxis(x[rgb_idx],0,-1); preview=((rgb-np.nanmin(rgb))/max(1e-6,float(np.nanmax(rgb)-np.nanmin(rgb)))*255).astype('uint8')
    ph=dhash(preview); ih=hashlib.sha256(x.tobytes()).hexdigest()
    if ih in contents or any((ph^p).bit_count()<=2 for p in perceptual):raise ValueError('exact_or_near_duplicate')
    folder=OUT/'samples'/key;folder.mkdir(parents=True,exist_ok=True)
    with h5py.File(folder/'data.h5','w') as f:
     for name,arr in arrays.items(): f.create_dataset(name,data=arr,compression='gzip')
     f.create_dataset('label_valid',data=resize(valid).astype('uint8'))
    views=[]
    if source=='landslide4sense': views=[(x,pv,[7,10,11],[7,10,11]),(x,pv,rgb_idx,rgb_idx)]
    elif source=='multimodal_landslide':
     for name,c in [('insar_velocity',4),('dem',3)]:
      values=arrays[f'auxiliary/{name}/values']; good=arrays[f'auxiliary/{name}/pixel_valid'].astype(bool)&arrays[f'auxiliary/{name}/channel_valid'][:,None,None].astype(bool)&np.isfinite(values)
      if not good.any():raise ValueError('missing_auxiliary')
      views.append((values,good,[0,0,0],[c,c,c]))
     views.append((x,pv,rgb_idx,rgb_idx))
    else:views=[(x,pv,rgb_idx,rgb_idx)]
    paths=[]
    for j,(values,good,channels,rc) in enumerate(views):
     rendered=[]
     for c,k in zip(channels,rc):
      lo,hi=ranges[MAP[source]][k]; a=np.where(good[c],np.clip((values[c]-lo)/max(hi-lo,1e-6),0,1),0)
      rendered.append((a*255).astype('uint8'))
     p=folder/f'view_{j}.png';Image.fromarray(np.stack(rendered,-1)).save(p);paths.append(str(p.relative_to(ROOT)))
    for name,arr in [('mask',m),('valid',v)]:Image.fromarray(arr.astype('uint8')*255).save(folder/f'{name}.png')
    entry={'id':key,'source':MAP[source],'split':'independent_test','benchmark_split':r['split'],'images':paths,'mask':str((folder/'mask.png').relative_to(ROOT)),'valid':str((folder/'valid.png').relative_to(ROOT)),'positive_pixels':int(m.sum()),'valid_pixels':int(v.sum()),'benchmark_record':r,'source_label_sha256':sha(labelpath),'source_label_path':str(labelpath),'footprint':box,'dhash':ph,'image_sha256':ih,'answer':'仅用于测试，不提供训练答案'}
    entry['files']={str(p.relative_to(ROOT)):sha(p) for p in folder.iterdir()}
    selected.append(entry);count+=1;contents.add(ih);perceptual.append(ph)
    if box:fps.append((MAP[source],box))
    if r.get('source_group_id'):groups.add((source,r['source_group_id']))
    audit.append({'id':key,'reason':'selected','split':r['split']})
   except Exception as e:audit.append({'id':key,'reason':str(e)});continue
   if count==quota:break
  print(source,count,flush=True)
 (OUT/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 (OUT/'screening.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
 write_json(OUT/'manifest.json',{'count':len(selected),'counts':{n:sum(r['source']==n for r in selected) for n in NAMES},'seed':20261116,'previous_1000_excluded':True,'benchmark_index_sha256':sha(B/'index.jsonl'),'old_manifests':{s:sha(DEST/f'{s}.jsonl') for s in ['train','val','test']},'display_ranges':ranges,'limitations':['benchmark已统一为224x224','地点缺失来源无法证明跨地区独立','只含正样本，不评估背景误报'],'complete':len(selected)==500})
 assert len(selected)==500,'严格筛选后数量不足，见筛选记录'
if __name__=='__main__':main()
