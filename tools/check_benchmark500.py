"""独立核对500条的实际复制内容、旧样本排除、标注和分组。"""
import json,hashlib
from collections import Counter
from pathlib import Path
import h5py,numpy as np
from PIL import Image
from xiaomi_lens.train.data.domain import ROOT,DEST,jsonlines,dhash,overlaps,NAMES
from xiaomi_lens.local_demo.service import DATA,OUT

def main():
 rows=jsonlines(DATA/'samples.jsonl');old=[r for s in ['train','val','test'] for r in jsonlines(DEST/f'{s}.jsonl')]
 idx=jsonlines(Path('/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/full/index.jsonl'));byid={r['source_sample_key']:r for r in idx}
 assert all(r['id'] in byid for r in old),'旧样本无法对应benchmark，需补充排重'
 groups={(byid[r['id']]['source'],byid[r['id']]['source_group_id']) for r in old if byid[r['id']].get('source_group_id')}
 assert len(rows)==len({r['id'] for r in rows})==500
 assert not {r['id'] for r in rows}&{r['id'] for r in old}
 seen=[];exact=set();fp=[(r['source'],r['footprint']) for r in old if r.get('footprint')]
 for r in rows:
  br=r['benchmark_record'];g=(br['source'],br.get('source_group_id'))
  if g[1]:assert g not in groups;groups.add(g)
  for p,h in r['files'].items():assert (ROOT/p).is_file() and not (ROOT/p).is_symlink() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
  with h5py.File(DATA/'samples'/r['id']/'data.h5') as f:
   x=f['optical'][:];pv=f['optical_pixel_valid'][:].astype(bool)&f['optical_channel_valid'][:][:,None,None].astype(bool)&np.isfinite(x)
   channels=[3,2,1] if r['source']==NAMES[1] else [0,1,2]
   valid=f['label_valid'][:].astype(bool)&pv[channels].all(0);mask=f['mask'][0].astype(bool)&valid
   assert np.array_equal(valid,np.array(Image.open(ROOT/r['valid']))>0)
   assert np.array_equal(mask,np.array(Image.open(ROOT/r['mask']))>0) and mask.any()
   ih=hashlib.sha256(x.tobytes()).hexdigest();assert ih not in exact;exact.add(ih)
   rgb=np.moveaxis(x[channels],0,-1);preview=((rgb-rgb.min())/max(1e-6,float(rgb.max()-rgb.min()))*255).astype('uint8');ph=dhash(preview)
   assert all((ph^p).bit_count()>2 for p in seen+[o['dhash'] for o in old]);seen.append(ph)
   if r['footprint']:
    assert all(n!=r['source'] or not overlaps(r['footprint'],box) for n,box in fp)
    fp.append((r['source'],r['footprint']))
 report={'passed':True,'count':len(rows),'previous_samples_mapped':len(old),'by_source':dict(Counter(r['source'] for r in rows)),'native_data_copies_checked':500,'positive_masks_checked':500,'known_groups_and_geographic_overlap_checked':True,'unknown_locations_remain':True,'source_code_unchanged':True}
 orig=json.loads((ROOT/'outputs/original_code_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in orig.items())
 OUT.mkdir(exist_ok=True);(OUT/'data-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report)
if __name__=='__main__':main()
