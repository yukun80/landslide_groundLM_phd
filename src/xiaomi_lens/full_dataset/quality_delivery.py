"""质量复查的追加修订、来源核对与发布；未完成时不写训练清单。"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT, SPLITS, read_jsonl, atomic_write_json, write_jsonl_atomic, sha256_file
from .quality_review import REVIEW, load
from .quality_generate import protocol
from ..description_v2 import answer
from ..train.data.domain import training_answer


def apply_visual(dest, submission):
    """保存本次实际看图后的修订，原模型返回与旧答案不覆盖。"""
    dest=Path(dest);folder=dest/REVIEW;load(dest)
    rows={r['id']:r for r in read_jsonl(dest/'prepared.jsonl')}
    candidates={r['id']:r for r in read_jsonl(folder/'candidates.jsonl')}
    for e in submission['entries']:
        row=rows[e['id']];old=candidates[e['id']]
        assert e['note'].strip() and e['reviewer']=='Codex direct image review'
        assert e['viewed_images'] and all(sha256_file(Path(p))==h for p,h in e['viewed_images'].items())
        assert str(ROOT/row['images'][-1]) in e['viewed_images']
        assert str(dest/'review/inputs'/row['id']/'target.png') in e['viewed_images']
        assert not e['aux_sentences'],'auxiliary correction requires a separate auxiliary review'
        mask=np.asarray(Image.open(ROOT/row['mask']))>0;valid=np.asarray(Image.open(ROOT/row['valid']))>0
        numeric,suffix=answer.split_original(training_answer(mask,valid))
        errors,opt,multi=answer.check_new_text(numeric,suffix,e['optical_sentences'],[],len(row['images'])>1)
        assert not errors,errors
        result={'id':row['id'],'source':row['source'],'split':row['split'],'status':'reviewed',
                'before':old['before'],'after':{'answer':multi,'answer_optical':opt},
                'optical_sentences':e['optical_sentences'],'aux_sentences':[],
                'quality_protocol_sha256':protocol(),'quality_manifest_sha256':sha256_file(folder/'manifest.json'),
                'previous_path':old['previous_path'],'previous_sha256':old['previous_sha256'],
                'agent_visual_review':e,'human_confirmed':False,'review_reasons':old['reasons'],
                'quality_decision':'rewritten' if e['optical_sentences'] else 'confirmed_no_appearance'}
        target=folder/'samples'/f"{row['id']}.json"
        assert not target.exists(),'do not overwrite an existing quality record'
        atomic_write_json(target,result)


def validate_record(dest, row, e, record, hash_file=sha256_file):
    folder=Path(dest)/REVIEW
    assert record['id']==row['id'] and record['status']=='reviewed' and record['human_confirmed'] is False
    assert record['before']==e['before']
    assert record['quality_manifest_sha256']==hash_file(folder/'manifest.json')
    if 'agent_visual_review' not in record:assert record['quality_protocol_sha256']==protocol()
    if e['previous_path']:assert hash_file(Path(dest)/e['previous_path'])==e['previous_sha256']
    mask=np.asarray(Image.open(ROOT/row['mask']))>0;valid=np.asarray(Image.open(ROOT/row['valid']))>0
    numeric,suffix=answer.split_original(training_answer(mask,valid))
    errors,opt,multi=answer.check_new_text(numeric,suffix,record['optical_sentences'],record['aux_sentences'],len(row['images'])>1)
    assert not errors and record['after']=={'answer':multi,'answer_optical':opt}
    if 'agent_visual_review' in record:
        review=record['agent_visual_review']
        assert review['reviewer']=='Codex direct image review' and review['note'].strip()
        assert record['optical_sentences']==review['optical_sentences'] and not record['aux_sentences']
        assert str(ROOT/row['images'][-1]) in review['viewed_images']
        assert str(Path(dest)/'review/inputs'/row['id']/'target.png') in review['viewed_images']
        for path,h in review['viewed_images'].items():assert hash_file(Path(path))==h
        return
    from .quality_generate import quality_paths
    expected={(role,str(path.relative_to(ROOT))):hash_file(path) for role,path in quality_paths(row,dest)}
    for name,key in [('optical','optical_sentences'),('auxiliary','aux_sentences')]:
        phase=record[name+'_review']
        if phase is None:
            assert name=='auxiliary' and len(row['images'])==1 and not record[key]
            continue
        candidate=phase['candidate'][key]
        from ..description_v2.visual_review import sentences
        originals=sentences(phase['draft']['parsed'],key,2 if key=='optical_sentences' else 1)
        assert all(s in originals for s in candidate),'candidate not in actual draft'
        checks=phase['checks']
        assert [c['sentence'] for c in checks]==candidate
        accepted=[]
        receipts=[phase['draft']]
        for check in checks:
            receipt=check['receipt'];assert type(receipt['parsed'].get('supported')) is bool
            if receipt['parsed']['supported']:accepted.append(check['sentence'])
            receipts.append(receipt)
        if not candidate:
            assert phase['empty_check']['parsed'].get('no_candidate') is True
            receipts.append(phase['empty_check'])
        from .quality_generate import presentation
        allowed=[presentation(s) for s in accepted] if key=='optical_sentences' else accepted
        assert all(s in allowed for s in record[key])
        required={k:h for k,h in expected.items() if name=='auxiliary' or not k[0].startswith('auxiliary_')}
        for receipt in receipts:
            assert receipt['raw'] and isinstance(receipt['parsed'],dict) and receipt['request_prompt']
            seen={(im['role'],im['path']):im['sha256'] for im in receipt['images']}
            assert seen==required
            assert all(im['consumed_by']=='Xiaomi-Lens-0.generate' and im['input_grid_thw'] for im in receipt['images'])


def final_sample_ids(records):
    groups={}
    for record in records:
        groups.setdefault((record['source'],record['quality_decision']),[]).append(record['id'])
    return sorted(rid for group in groups.values() for rid in sorted(group)[:3])


def publish(dest):
    dest=Path(dest);folder=dest/REVIEW;m=load(dest)
    # This publisher only reconciles its original candidate list. After the
    # conversation merge that list omits pending rows and contains older drafts.
    summary = json.loads((dest/'summary.json').read_text())
    if summary.get('description_merge') and summary['description_merge']['awaiting_review']:
        raise ValueError('现用全量清单已合并逐图定稿。旧候选清单未覆盖全部待确认项，不能据此发布全量终审结果。')
    baseline={r['id']:r for r in read_jsonl(folder/'baseline.jsonl')}
    candidates={r['id']:r for r in read_jsonl(folder/'candidates.jsonl')}
    paths={p.stem:p for p in (folder/'samples').glob('*.json')}
    assert set(paths)==set(candidates),'quality review incomplete'
    assert (folder/'pilot_approval.json').exists(),'pilot visual review missing'
    assert (folder/'final_visual_review.json').exists(),'final stratified visual review missing'
    visual=json.loads((folder/'final_visual_review.json').read_text())
    assert visual['passed'] and visual['entries']
    sample_ids=final_sample_ids([json.loads(p.read_text()) for p in paths.values()])
    assert len(visual['entries'])==len(sample_ids) and {e['id'] for e in visual['entries']}==set(sample_ids)
    for e in visual['entries']:
        assert e['passed'] and e['record_sha256']==sha256_file(paths[e['id']])
        assert e['note'].strip() and e['viewed_images']
        for p,h in e['viewed_images'].items():assert sha256_file(Path(p))==h
    rows=read_jsonl(dest/'prepared.jsonl');out=[];adopted=[];counts=Counter();cache={}
    def hashed(p):
        p=Path(p)
        if p not in cache:cache[p]=sha256_file(p)
        return cache[p]
    for row in rows:
        e=baseline[row['id']]
        if e['previous_path']:assert hashed(dest/e['previous_path'])==e['previous_sha256']
        record=None
        if row['id'] in candidates:
            path=paths[row['id']];record=json.loads(path.read_text())
            validate_record(dest,row,candidates[row['id']],record,hashed)
            after=record['after'];source='quality_review';counts[record['quality_decision']]+=1
        else:
            path=dest/e['previous_path'] if e['previous_path'] else None
            after=e['before'];source=e['previous_provider'];counts['not_flagged']+=1
        out.append({**row,**after,'description_provider':source,'description_status':'reviewed' if record else ('reused' if source=='reused' else 'reviewed')})
        adopted.append({'id':row['id'],'source':source,'path':str(path.relative_to(dest)) if path else None,'sha256':hashed(path) if path else None})
    for split in SPLITS:write_jsonl_atomic(dest/f'{split}.jsonl',[r for r in out if r['split']==split])
    write_jsonl_atomic(folder/'adopted.jsonl',adopted)
    summary=json.loads((dest/'summary.json').read_text())
    summary.update(complete=False,quality_review='awaiting_full_data_checks',quality_counts=dict(counts),
                   manifest_sha256={s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS},human_confirmed=False)
    atomic_write_json(dest/'summary.json',summary)
    atomic_write_json(folder/'records_check.json',{'passed':True,'count':len(rows),'counts':dict(counts),
        'adopted_sha256':sha256_file(folder/'adopted.jsonl'),'manifest_sha256':summary['manifest_sha256']})
    return summary


def validate_final(dest, rows):
    """最终清单逐条对应新旧原始记录，并核对实际图片与真实返回。"""
    dest=Path(dest);folder=dest/REVIEW;m=load(dest)
    report=json.loads((folder/'records_check.json').read_text())
    assert report['passed'] and report['adopted_sha256']==sha256_file(folder/'adopted.jsonl')
    assert report['manifest_sha256']=={s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS}
    adopted={r['id']:r for r in read_jsonl(folder/'adopted.jsonl')}
    baseline={r['id']:r for r in read_jsonl(folder/'baseline.jsonl')}
    candidates={r['id']:r for r in read_jsonl(folder/'candidates.jsonl')}
    prepared={r['id']:r for r in read_jsonl(dest/'prepared.jsonl')}
    assert len(rows)==len(adopted)==len(baseline)==m['selected']
    assert {r['id'] for r in rows}==set(baseline)
    stats=Counter(samples_with_notes=0);cache={}
    def hashed(p):
        p=Path(p)
        if p not in cache:cache[p]=sha256_file(p)
        return cache[p]
    for i,row in enumerate(rows,1):
        rid=row['id'];e=baseline[rid];a=adopted[rid]
        if e['previous_path']:assert hashed(dest/e['previous_path'])==e['previous_sha256']
        assert row['description_provider']==a['source']
        if a['path']:assert hashed(dest/a['path'])==a['sha256']
        if rid in candidates:
            assert a['source']=='quality_review' and a['path']==str(Path(REVIEW)/'samples'/f'{rid}.json')
            record=json.loads((dest/a['path']).read_text())
            validate_record(dest,prepared[rid],candidates[rid],record,hashed)
            expected=record['after'];stats[record['quality_decision']]+=1
            stats['samples_with_notes']+=bool(record.get('uncertain'))
        else:
            assert a['source']==e['previous_provider'] and a['path']==e['previous_path']
            expected=e['before']
        assert {k:row[k] for k in ('answer','answer_optical')}==expected
        for key in prepared[rid]:
            if key not in ('answer','answer_optical','description_status','description_provider'):
                assert row[key]==prepared[rid][key],('non_answer_changed',rid,key)
        stats[a['source']]+=1
        if i%500==0:print('QUALITY_RECORDS_CHECK',i,len(rows),flush=True)
    assert stats['quality_review']==m['required']
    return stats
