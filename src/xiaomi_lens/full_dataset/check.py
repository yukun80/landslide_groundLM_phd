"""全量交付检查：源记录去向、真实副本、两版训练读取和逐图标注证据。"""
import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

import numpy as np
from PIL import Image

from .common import *
from .prepare import SpatialIndex
from ..train.data.domain import read_source, training_answer
from ..description_v2.answer import split_original, _check_one_answer


def structural(dest,rows):
    """每条来源只能有一个最终去向；保护既有划分及真实同图/地理分组。"""
    build=initialize(dest)
    for n,meta in build['source_manifests'].items():
        assert file_meta(SOURCE/n/'hdf5/conversion_manifest.jsonl')==meta,n
    inventory=read_jsonl(dest/'audit/inventory.jsonl')
    exclusions=read_jsonl(dest/'audit/partition_exclusions.jsonl')
    ids=[r['id'] for r in rows];assert len(ids)==len(set(ids))
    outcomes=[r['id'] for r in inventory if r['status']=='excluded']+[r['id'] for r in exclusions]+ids
    assert len(outcomes)==len(set(outcomes))==len(inventory)
    assert set(outcomes)=={r['id'] for r in inventory}
    old={r['id']:r for r in old_rows()};bench={r['id'] for r in read_jsonl(BENCHMARK/'samples.jsonl')}
    now={r['id']:r for r in rows}
    quality=json.loads((dest/'audit/protected_quality_conflicts.json').read_text())
    conflicts={r['id'] for r in quality['samples']}
    if quality['resolved'] and conflicts:
        assert quality['user_decision']=='exclude_from_new_dataset_keep_original_1000'
        assert all(r['decision']=='excluded_by_user_choice' for r in quality['samples'])
        assert not conflicts&set(now),'excluded all-black sample returned to the new data'
    assert ((set(old)|bench)-conflicts)<=set(now)
    quality_baseline=None
    if any(r.get('description_provider')=='quality_review' for r in rows):
        from .quality_review import REVIEW, load
        load(dest)
        quality_baseline={e['id']:e for e in read_jsonl(dest/REVIEW/'baseline.jsonl')}
    from .quality_review import merged_entries
    adopted = merged_entries(dest)
    for rid,r in old.items():
        if rid not in now:assert rid in conflicts;continue
        assert now[rid]['split']==r['split'] and now[rid]['reuse_text']
        if rid in adopted:
            entry = adopted[rid]
            assert entry['before'] == {k:r[k] for k in ('answer','answer_optical')}
            assert all(entry[k] == now[rid][k] for k in ('answer','answer_optical'))
        elif now[rid].get('description_provider')=='quality_review':
            # Only the new dataset copy may adopt a separately reviewed answer.
            # Original 1000-row files and all image/split protections stay intact.
            assert quality_baseline[rid]['before']=={k:r[k] for k in ('answer','answer_optical')}
            revised=json.loads((dest/REVIEW/'samples'/f'{rid}.json').read_text())
            assert revised['after']=={k:now[rid][k] for k in ('answer','answer_optical')}
        else:
            for k in ('answer','answer_optical'):assert now[rid][k]==r[k]
        for key,paths in [('images',r['images']),('mask',[r['mask']]),('valid',[r['valid']])]:
            newpaths=now[rid][key] if key=='images' else [now[rid][key]]
            assert [sha256_file(ROOT/p) for p in newpaths]==[sha256_file(ROOT/p) for p in paths]
    assert all(now[rid]['split']=='test' for rid in bench)
    groups=defaultdict(set);content=defaultdict(set);indices={n:SpatialIndex() for n in NAMES}
    for r in rows:
        groups[r['group']].add(r['split']);content[r['optical_sha256']].add(r['split'])
        assert not [h for h in indices[r['source']].query(r['footprint']) if h[1]!=r['split']],r['id']
        indices[r['source']].add(r,r['split'])
    assert all(len(s)==1 for s in groups.values()) and all(len(s)==1 for s in content.values())
    report={'passed':True,'source_records':len(inventory),'included':len(rows),
            'quality_excluded':sum(r['status']=='excluded' for r in inventory),'partition_excluded':len(exclusions),
            'old_1000_unchanged':True,'independent_500_test_only':True,'known_cross_split_leakage':False,
            'counts':dict(Counter(r['source']+'/'+r['split'] for r in rows))}
    atomic_write_json(dest/'audit/structure_check.json',report);return report


def inspect_export(row,dest):
    """核对每份原始副本、重建显示图、真实标注和两版答案。"""
    dest=Path(dest);rid=row['id'];report_path=dest/'audit/raw_checks'/f'{rid}.json'
    assert len(row['images'])==len(VIEWS[row['source']]) and row['source_record']['sample_key']==rid
    fingerprint=digest({'raw':row['source_files'],'files':row['files']})
    if report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous['fingerprint']==fingerprint:
            for p,stat in previous['checked_stats'].items():
                s=Path(p).stat();assert [s.st_size,s.st_mtime_ns]==stat,('checked file changed',p)
            for key in ('answer','answer_optical'):
                assert not _check_one_answer(row[key],previous['numeric'],previous['suffix']),(rid,key)
            return previous
    files=[]
    for key,meta in row['source_files'].items():
        src=Path(meta['path']);local=ROOT/row['raw_files'][key]
        assert src.resolve()!=local.resolve() and not local.is_symlink()
        assert sha256_file(src)==sha256_file(local)==meta['sha256'],(rid,key)
        files.extend([src,local])
    source_row={**row['source_record'],**{k:str(ROOT/p) for k,p in row['raw_files'].items()}}
    x,pv,m,v=read_source(row['source'],source_row)
    assert m.shape==v.shape==(row['height'],row['width']) and v.any() and m.any()
    assert int(m.sum())==row['positive_pixels'] and int(v.sum())==row['valid_pixels']
    for key,arr in [('mask',m),('valid',v)]:
        p=ROOT/row[key];assert np.array_equal(np.asarray(Image.open(p))>0,arr);files.append(p)
    ranges=json.loads((dest/'display_ranges.json').read_text())[row['source']]
    for p,channels in zip(row['images'],VIEWS[row['source']]):
        planes=[]
        for c in channels:
            lo,hi=ranges[c];a=np.clip((x[c]-lo)/max(hi-lo,1e-6),0,1)
            planes.append((np.where(pv[c],a,0)*255).astype('uint8'))
        p=ROOT/p;assert np.array_equal(np.stack(planes,-1),np.asarray(Image.open(p))),(rid,'display')
        files.append(p)
    for p,h in row['files'].items():assert sha256_file(ROOT/p)==h,(rid,p)
    numeric,suffix=split_original(training_answer(m,v))
    for key in ('answer','answer_optical'):assert not _check_one_answer(row[key],numeric,suffix),(rid,key)
    if len(row['images'])==1:assert row['answer']==row['answer_optical']
    result={'id':rid,'passed':True,'fingerprint':fingerprint,'positive_pixels':int(m.sum()),'numeric':numeric,'suffix':suffix,
            'checked_stats':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in files}}
    atomic_write_json(report_path,result);return result


def raw(dest,prepared=False,workers=6):
    rows=read_jsonl(dest/'prepared.jsonl') if prepared else [r for s in SPLITS for r in read_jsonl(dest/f'{s}.jsonl')]
    report=structural(dest,rows);start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(partial(inspect_export,dest=dest),rows,chunksize=2),1):
            assert result['passed']
            if i%200==0:print('RAW_CHECK',i,len(rows),round(time.monotonic()-start),flush=True)
    report.update(raw_copies_and_all_display_pixels_verified=True,seconds=time.monotonic()-start,prepared_only=prepared)
    atomic_write_json(dest/'audit/raw_check.json',report);print('RAW_PASS',report,flush=True)


def encode(dest,pilot=False,device="cpu"):
    """真实调用训练的图片处理与文字读取；不加载权重，不进行训练。"""
    if device=='cpu':
        import os
        os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from transformers import AutoProcessor
    from ..domain_runtime import BASE,CONFIG,build,configure_processor,seed_all
    from ..description_v2 import select_answer_row
    seed_all();processor=configure_processor(AutoProcessor.from_pretrained(BASE,local_files_only=True,use_fast=False))
    provider=json.loads((dest/'build.json').read_text()).get('annotation_provider')
    if pilot and provider=='local_resume':
        from .resume_records import load_manifest, record_path
        ids=set(load_manifest(dest)['pilot_ids'])
        rows=[r for r in read_jsonl(dest/'prepared.jsonl') if r['id'] in ids]
        rows=[{**r,**json.loads(record_path(dest,r['id']).read_text())['after']} for r in rows]
        assert len(rows)==12
    elif pilot and provider=='codex_session':
        from .session_review import REVIEW,record_path
        ids=set(json.loads((dest/REVIEW/'manifest.json').read_text())['pilot_ids'])
        rows=[r for r in read_jsonl(dest/'prepared.jsonl') if r['id'] in ids]
        rows=[{**r,**json.loads(record_path(dest,r['id']).read_text())['after']} for r in rows]
        assert len(rows)==6
    elif pilot:
        ids={r['id'] for r in json.loads((dest/'review/pilot_check.json').read_text())['results']}
        rows=[r for r in read_jsonl(dest/'prepared.jsonl') if r['id'] in ids]
        rows=[{**r,**json.loads((dest/f"review/samples/{r['id']}.json").read_text())['after']} for r in rows]
        assert len(rows)==12
    else:rows=[r for s in SPLITS for r in read_jsonl(dest/f'{s}.jsonl')]
    completed=[];start=time.monotonic()
    for i,r in enumerate(rows,1):
        marker=dest/'audit/encoding_checks'/f"{r['id']}.json";fingerprint=digest({'files':r['files'],'answers':[r['answer'],r['answer_optical']],'config':CONFIG,'device':device,'provider':json.loads((dest/'build.json').read_text()).get('annotation_provider')})
        if marker.exists() and json.loads(marker.read_text())['fingerprint']==fingerprint:
            completed.append(json.loads(marker.read_text()));continue
        modes=[]
        for rgb in (False,True):
            batch=build(processor,select_answer_row(r,rgb),rgb,training=True,device=device)
            assert batch['mask_targets'].any() and batch['mask_pixel_valid'].any()
            assert batch['mask_positions'].numel()==1
            assert int(batch['mask_image_index'].item())==(0 if rgb else len(r['images'])-1)
            answer_length=int((batch['labels']!=-100).sum())
            assert answer_length<=CONFIG['maximum_answer_tokens'],(r['id'],rgb,answer_length)
            modes.append({'rgb_only':rgb,'input_tokens':batch['input_ids'].shape[1],
                          'answer_tokens_including_end':answer_length,'mask_target_shape':list(batch['mask_targets'].shape),
                          'mask_image_index':int(batch['mask_image_index'].item())})
            del batch
        result={'id':r['id'],'fingerprint':fingerprint,'modes':modes,'passed':True}
        atomic_write_json(marker,result);completed.append(result)
        if i%100==0:print('ENCODE',i,len(rows),round(time.monotonic()-start),flush=True)
    report={'passed':True,'count':len(completed),'dual_encodings':2*len(completed),'formal_training_run':False,
            'seconds':time.monotonic()-start,'maximum_answer_tokens':max(m['answer_tokens_including_end'] for r in completed for m in r['modes'])}
    report['device']=device
    report['cuda_initialized']=torch.cuda.is_initialized()
    if device=='cpu':assert not report['cuda_initialized'],'CPU check unexpectedly initialized CUDA'
    report['scope']='training_pilot_only' if pilot else 'all_final_rows'
    if pilot:report['answer_text_sha256']={r['id']:digest([r['answer'],r['answer_optical']]) for r in rows}
    else:report['manifest_sha256']={s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS}
    atomic_write_json(dest/('audit/pilot_encoding_check.json' if pilot else 'audit/encoding_check.json'),report);print('ENCODE_PASS',report,flush=True)


def final(dest):
    from ..description_v2 import visual_review as visual
    from .views import GUIDE_VERSION,view_paths
    rows=[r for s in SPLITS for r in read_jsonl(dest/f'{s}.jsonl')];structural(dest,rows)
    quality=json.loads((dest/'audit/protected_quality_conflicts.json').read_text())
    assert quality['resolved'],'protected sample quality choice remains unresolved'
    for name in ('raw_check','encoding_check'):
        check=json.loads((dest/f'audit/{name}.json').read_text());assert check['passed'] and not check.get('prepared_only',False)
        assert check.get('included',check.get('count'))==len(rows)
    provider=json.loads((dest/'build.json').read_text()).get('annotation_provider')
    if provider=='local_resume':
        from .local_resume import validate_all
        stats=validate_all(dest,rows);uncertain=stats.pop('samples_with_notes')
    elif provider=='codex_session':
        from .session_review import validate_all
        stats=validate_all(dest,rows);uncertain=stats.pop('samples_with_notes')
    else:
        from .captions import EXTRA_POLICY
        manifest=json.loads((dest/'review/manifest.json').read_text());assert manifest['prompt_sha256']==visual.digest([visual.POLICY,visual.DRAFT,visual.VERIFY,EXTRA_POLICY,GUIDE_VERSION])
        stats=Counter();uncertain=0
        for r in rows:
            if r['reuse_text']:
                assert r['description_status']=='reused';stats['reused']+=1;continue
            review=json.loads((dest/f"review/samples/{r['id']}.json").read_text())
            assert review['status']=='reviewed' and review['prompt_sha256']==manifest['prompt_sha256']
            assert review['human_confirmed'] is False and review['after']=={k:r[k] for k in ('answer','answer_optical')}
            paths=view_paths(r,dest)
            expected=[(role,str(p.relative_to(ROOT)),sha256_file(p)) for role,p in paths]
            for phase in ('draft','verification'):
                receipt=review[phase];assert isinstance(receipt['parsed'],dict) and receipt['raw']
                assert [(x['role'],x['path'],x['sha256']) for x in receipt['images']]==expected
                assert all(x['consumed_by']=='Xiaomi-Lens-0.generate' and x['input_grid_thw'] for x in receipt['images'])
            stats['reviewed']+=1
            stats['with_optical_appearance']+=bool(review['optical_sentences'])
            stats['with_auxiliary_detail']+=bool(review['aux_sentences'])
            uncertain+=bool(review['uncertain'])
    expected_reuse=sum(r['reuse_text'] for r in rows)
    assert stats['reused']==expected_reuse and stats['reviewed']+stats['reused']==len(rows)
    original=json.loads((ROOT/'outputs/original_code_hashes.json').read_text())
    assert all(sha256_file(ROOT/p)==h for p,h in original.items())
    snapshot=dest/'audit/descriptions_before_resume_20261117.json'
    if snapshot.exists():
        before=json.loads(snapshot.read_text())
        assert all(sha256_file(dest/p)==h for p,h in before['sha256'].items()),'completed descriptions changed during resume'
    build=initialize(dest);summary=json.loads((dest/'summary.json').read_text())
    assert summary['manifest_sha256']=={s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS}
    assert json.loads((dest/'audit/encoding_check.json').read_text())['manifest_sha256']==summary['manifest_sha256'], 'encoding report belongs to different final manifests'
    report={'passed':True,'count':len(rows),'description_coverage':dict(stats),'samples_with_review_notes':uncertain,
            'old_data_and_original_code_unchanged':True,'human_confirmed':False,'formal_training_run':False,
            'annotation_provider':provider,'checked_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'limitations':summary['limitations']}
    atomic_write_json(dest/'audit/final_check.json',report)
    summary.update(complete=True,validation='audit/final_check.json')
    if (dest/'review/quality_review/manifest.json').exists():summary['quality_review']='complete'
    atomic_write_json(dest/'summary.json',summary)
    build.update(complete=True);atomic_write_json(dest/'build.json',build)
    print('FINAL_PASS',report,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['raw','encode','final'])
    p.add_argument('--dataset-dir',type=Path,default=DEFAULT);p.add_argument('--prepared',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();dest=a.dataset_dir.resolve()
    if a.stage=='raw':raw(dest,a.prepared,a.workers)
    elif a.stage=='encode':encode(dest,a.pilot,a.device)
    else:final(dest)
