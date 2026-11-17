"""当前Codex会话逐图标注：程序只准备、检查、保存，不代写文字、不调用模型。"""
import argparse
import json
import os
import time
from itertools import zip_longest
from pathlib import Path

from .common import *
from .views import view_paths
from ..train.data.domain import training_answer
from ..description_v2.answer import split_original,check_new_text,_check_one_answer

PROVIDER='codex_session'
REVIEW='review/codex_session'
RULES='当前Codex会话直接查看全部独立图片后重新撰写、再次对照。不参考本地候选文字。不确定外观留空，数值和区域标记由标注计算。光学版不混入辅助依据。不写成因、风险、活动性、具体形变速率，不按亮度判断形变大小。保存记录是会话看图声明，不冒充人工或独立模型复核。'


def record_path(dest,rid):return Path(dest)/REVIEW/'samples'/f'{rid}.json'


def activate(dest=DEFAULT):
    """锁定在线会话来源，保留停机后的本地历史，未完成前禁止训练。"""
    dest=Path(dest);build=initialize(dest);p=dest/REVIEW/'manifest.json'
    if p.exists():
        assert build.get('annotation_provider')==PROVIDER,'annotation provider lock changed'
        manifest=json.loads(p.read_text());assert manifest['rules_sha256']==digest(RULES)
        assert manifest['selected_sha256']==sha256_file(dest/'audit/selected.jsonl')
        assert manifest['prepared_sha256']==sha256_file(dest/'prepared.jsonl'),'prepared inputs changed'
        return manifest
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=process.read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if len(args)>1 and args[1] in ('tools/describe_full_dataset.py','tools/complete_full_dataset.py'):
            raise RuntimeError('先停止本地生成与自动接续进程，再切换')
    rows=read_jsonl(dest/'prepared.jsonl');assert len(rows)==18987 and sum(r['reuse_text'] for r in rows)==999
    history={str(f.relative_to(dest)):sha256_file(f) for f in (dest/'review/samples').glob('*.json')}
    atomic_write_json(dest/'audit/local_descriptions_at_switch.json',{'count':len(history),'sha256':history,'preserve_only':True})
    for name in ('summary.json','audit/final_check.json','review/manifest.json','audit/pipeline_state.json'):
        source=dest/name
        if source.exists():
            target=dest/'review/local_history'/name
            if not target.exists():atomic_write_json(target,json.loads(source.read_text()))
    pilot=pilot_rows(rows);pilot=[pilot[i] for i in (0,2,4,6,8,10)]
    manifest={'provider':PROVIDER,'rules':RULES,'rules_sha256':digest(RULES),'selected_sha256':sha256_file(dest/'audit/selected.jsonl'),
              'prepared_sha256':sha256_file(dest/'prepared.jsonl'),'target_count':17988,'reused_count':999,
              'pilot_ids':[r['id'] for r in pilot],'batch_size':3,'local_history_count':len(history),
              'human_confirmed':False,'view_record_type':'current_session_attestation','created_at':time.strftime('%Y-%m-%dT%H:%M:%S')}
    build.update(annotation_provider=PROVIDER,complete=False);atomic_write_json(dest/'build.json',build)
    atomic_write_json(p,manifest)
    summary={'total':len(rows),'counts':dict(__import__('collections').Counter(r['split'] for r in rows)),
             'annotation_provider':PROVIDER,'complete':False,'reused':999,'codex_reviewed':0,'required':17988,
             'limitations':json.loads((dest/'audit/partition.json').read_text())['limitations']}
    atomic_write_json(dest/'summary.json',summary);status(dest)
    return manifest


def ordered_rows(dest,manifest):
    rows=[r for r in read_jsonl(Path(dest)/'prepared.jsonl') if not r['reuse_text']]
    byid={r['id']:r for r in rows};pilot=[byid[i] for i in manifest['pilot_ids']]
    groups=[sorted([r for r in rows if r['source']==n and r['id'] not in manifest['pilot_ids']],key=lambda r:(SPLITS.index(r['split']),r['id'])) for n in NAMES]
    return pilot+[r for group in zip_longest(*groups) for r in group if r is not None]


def status(dest=DEFAULT):
    dest=Path(dest);manifest=json.loads((dest/REVIEW/'manifest.json').read_text())
    folder=dest/REVIEW/'samples';done=len(list(folder.glob('*.json')))
    active=json.loads((dest/'build.json').read_text()).get('annotation_provider')
    if active!=PROVIDER:
        return {'provider':PROVIDER,'historical_only':True,'active_provider':active,
                'reviewed':done,'complete':False,
                'note':'历史查看数量；未改写当前进度或完成状态'}
    summary=json.loads((dest/'summary.json').read_text())
    complete=bool(summary.get('complete')) and (dest/'audit/final_check.json').exists()
    result={'provider':PROVIDER,'complete':complete,'step':'complete' if complete else ('awaiting_final_checks' if done==manifest['target_count'] else 'awaiting_codex_session'),'reviewed':done,
            'required':manifest['target_count'],'remaining':manifest['target_count']-done,'reused':999,
            'local_history_count':manifest['local_history_count'],'background_image_review':False,
            'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S')}
    atomic_write_json(dest/REVIEW/'progress.json',result);atomic_write_json(dest/'audit/pipeline_state.json',result)
    if not complete:
        summary.update(codex_reviewed=done,complete=False);atomic_write_json(dest/'summary.json',summary)
    return result


def next_batch(dest=DEFAULT,pilot=False):
    """准备待查看路径和基础数值；准备文件本身绝不计为已经看图。"""
    dest=Path(dest);manifest=activate(dest);active=dest/REVIEW/'active_batch.json'
    if active.exists():
        batch=json.loads(active.read_text())
        if any(not record_path(dest,r['id']).exists() for r in batch['samples']):return batch
    rows=[r for r in ordered_rows(dest,manifest) if not record_path(dest,r['id']).exists()]
    if pilot:rows=[r for r in rows if r['id'] in manifest['pilot_ids']]
    rows=rows[:6 if pilot else manifest['batch_size']]
    samples=[]
    for r in rows:
        from PIL import Image
        import numpy as np
        mask=np.asarray(Image.open(ROOT/r['mask']))>0;valid=np.asarray(Image.open(ROOT/r['valid']))>0
        numeric,suffix=split_original(training_answer(mask,valid))
        views=[{'role':role,'path':str(p.relative_to(ROOT)),'sha256':sha256_file(p)} for role,p in view_paths(r,dest,REVIEW+'/inputs')]
        samples.append({'id':r['id'],'source':r['source'],'split':r['split'],'numeric':numeric,'suffix':suffix,'views':views})
    batch={'batch_id':digest([r['id'] for r in rows])[:16],'provider':PROVIDER,'samples':samples,'prepared_only':True}
    atomic_write_json(active,batch);atomic_write_json(dest/REVIEW/'batches'/f"{batch['batch_id']}.json",batch)
    return batch


def validate_entry(entry,sample,tokenizer):
    """只验证明确约束；颜色和形状仍由Codex实际看图判断，不自动制造外观。"""
    assert entry['id']==sample['id']
    assert entry['viewed_roles']==[v['role'] for v in sample['views']],('unviewed_roles',entry['id'])
    for key,maximum in [('optical_sentences',2),('aux_sentences',1)]:
        assert isinstance(entry[key],list) and len(entry[key])<=maximum and all(isinstance(s,str) and s.strip() for s in entry[key])
    assert isinstance(entry['uncertain'],list) and all(isinstance(s,str) for s in entry['uncertain'])
    assert entry['observation_note'].strip() and entry['self_check_note'].strip()
    errors,optical,multi=check_new_text(sample['numeric'],sample['suffix'],entry['optical_sentences'],entry['aux_sentences'],sample['source']!=NAMES[0])
    assert not errors,(entry['id'],errors)
    for text in (optical,multi):
        assert not _check_one_answer(text,sample['numeric'],sample['suffix'])
        assert len(tokenizer(text,add_special_tokens=False)['input_ids'])+1<=160,('answer_too_long',entry['id'])
    if sample['source']==NAMES[1]:assert all('InSAR' not in s and 'DEM' not in s for s in entry['aux_sentences'])
    return {'answer':multi,'answer_optical':optical}


def commit(dest,input_path,session_id,revise=False):
    """仅当前会话提交实际观测；失败不写完成文件，再次提交可核对恢复。"""
    dest=Path(dest);manifest=activate(dest);batch=json.loads((dest/REVIEW/'active_batch.json').read_text())
    entries=json.loads(Path(input_path).read_text());byid={s['id']:s for s in batch['samples']}
    assert isinstance(entries,list) and entries and len({e['id'] for e in entries})==len(entries)
    assert session_id and session_id!='unavailable','requires actual current session identity'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/Xiaomi-Lens-0',local_files_only=True,use_fast=False)
    outputs=[]
    for entry in entries:
        if revise and entry['id'] not in byid:
            old=json.loads(record_path(dest,entry['id']).read_text())
            row=next(r for r in read_jsonl(dest/'prepared.jsonl') if r['id']==entry['id'])
            from PIL import Image
            import numpy as np
            numeric,suffix=split_original(training_answer(np.asarray(Image.open(ROOT/row['mask']))>0,np.asarray(Image.open(ROOT/row['valid']))>0))
            byid[entry['id']]={'id':row['id'],'source':row['source'],'numeric':numeric,'suffix':suffix,'views':old['views']}
        sample=byid[entry['id']]
        for view in sample['views']:assert sha256_file(ROOT/view['path'])==view['sha256']
        after=validate_entry(entry,sample,tokenizer)
        record={**entry,'provider':PROVIDER,'status':'reviewed','after':after,'views':sample['views'],
                'rules_sha256':manifest['rules_sha256'],'session_id':session_id,'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%S'),
                'view_record_type':'current_session_attestation','tool':'view_image','tool_call_ids':None,
                'human_confirmed':False,'independent_reviewer':False,'local_model_used':False}
        path=record_path(dest,entry['id'])
        if path.exists():
            old=json.loads(path.read_text())
            assert old['views']==sample['views'],'source images changed'
            changed=any(old.get(k)!=record.get(k) for k in ('after','optical_sentences','aux_sentences','viewed_roles','observation_note','self_check_note','uncertain'))
            if changed:
                assert revise and entry.get('revision_reason','').strip(),'explicit revision and reason required'
                atomic_write_json(dest/REVIEW/'revisions'/entry['id']/f"{digest(old)}.json",old)
                record.update(revision_reason=entry['revision_reason'],previous_sha256=digest(old))
                atomic_write_json(path,record)
        else:atomic_write_json(path,record)
        outputs.append(entry['id'])
    status(dest);return outputs


def publish(dest=DEFAULT):
    """只接收本轮Codex记录；本地历史文字不能补缺。"""
    dest=Path(dest);manifest=activate(dest);rows=read_jsonl(dest/'prepared.jsonl');counts={}
    for r in rows:
        if r['reuse_text']:r['description_status']='reused'
        else:
            p=record_path(dest,r['id']);assert p.exists(),('missing_codex_review',r['id']);record=json.loads(p.read_text())
            assert record['provider']==PROVIDER and record['status']=='reviewed'
            r.update(record['after']);r.update(description_status='codex_reviewed',annotation_provider=PROVIDER)
        counts[r['split']]=counts.get(r['split'],0)+1
    for split in SPLITS:write_jsonl_atomic(dest/f'{split}.jsonl',[r for r in rows if r['split']==split])
    summary=json.loads((dest/'summary.json').read_text());summary.update(complete=False,counts=counts,
        codex_reviewed=manifest['target_count'],description_counts={'reused':999,'codex_reviewed':manifest['target_count']},
        sources=dict(__import__('collections').Counter(r['source'] for r in rows)),positive_only=True,empty_samples=0,
        display_ranges_sha256=sha256_file(dest/'display_ranges.json'),
        manifest_sha256={s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS})
    atomic_write_json(dest/'summary.json',summary)


def validate_all(dest,rows):
    manifest=activate(dest);stats={'reused':0,'reviewed':0,'with_optical_appearance':0,'with_auxiliary_detail':0,'samples_with_notes':0}
    for row in rows:
        if row['reuse_text']:stats['reused']+=1;continue
        record=json.loads(record_path(dest,row['id']).read_text())
        assert record['provider']==PROVIDER and record['rules_sha256']==manifest['rules_sha256']
        assert record['status']=='reviewed' and record['local_model_used'] is False and record['human_confirmed'] is False
        assert record['session_id'] and record['observation_note'] and record['self_check_note']
        expected=[{'role':role,'path':str(p.relative_to(ROOT)),'sha256':sha256_file(p)} for role,p in view_paths(row,dest,REVIEW+'/inputs')]
        assert record['views']==expected and record['viewed_roles']==[v['role'] for v in expected]
        assert record['after']=={k:row[k] for k in ('answer','answer_optical')}
        stats['reviewed']+=1;stats['with_optical_appearance']+=bool(record['optical_sentences']);stats['with_auxiliary_detail']+=bool(record['aux_sentences'])
        stats['samples_with_notes']+=bool(record['uncertain'])
    assert stats['reused']==999 and stats['reviewed']==17988
    history=json.loads((Path(dest)/'audit/local_descriptions_at_switch.json').read_text())
    assert all(sha256_file(Path(dest)/p)==h for p,h in history['sha256'].items()),'local historical records changed'
    return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','next','commit','status','publish']);p.add_argument('--dataset-dir',type=Path,default=DEFAULT)
    p.add_argument('--pilot',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--session-id',default=os.environ.get('CODEX_THREAD_ID'))
    p.add_argument('--revise',action='store_true',help='显式修订须再次看图并给出revision_reason，旧记录保留')
    args=p.parse_args();dest=args.dataset_dir
    import fcntl
    lock=dest/'codex-session.lock'
    with lock.open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='init':result=activate(dest)
        elif args.stage=='next':result=next_batch(dest,args.pilot)
        elif args.stage=='commit':result=commit(dest,args.input,args.session_id,args.revise)
        elif args.stage=='publish':result=publish(dest)
        else:result=status(dest)
    print(json.dumps(result,ensure_ascii=False,indent=2))
