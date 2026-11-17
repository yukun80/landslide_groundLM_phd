"""逐样本两遍视觉标注；复用旧文字，保留实际返回，支持任意完整数据目录。"""
import argparse
import json
import time
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from ..description_v2 import visual_review as visual, answer, spec
from ..domain_runtime import images_for, load, seed_all, CONFIG
from .common import *
from . import guards

from .views import view_paths,GUIDE_VERSION
_SAFE_CAP={}
EXTRA_POLICY='''仅写目标本身可辨的颜色、轮廓或纹理。光学颜色只从光学原图和光学局部核对，不从InSAR、DEM或假彩图读取。
新增的局部光学图及其白色目标图来自实际输入尺寸，帮助排除相邻浅色条带；局部以外的背景不是目标。
禁止在训练描述中提及二值图、核对图、标注、像素、图号或任何制作过程。不要把阴影、树冠或森林当成已确认地物。
不写从哪里向哪里延伸的方向句，不复述位置数值句。最多两句光学信息，两句不能换一种说法重复。
单个或极少像素的小目标无法确认形状或纹理时请留空，不凭上采样的方块边界描述目标形状。'''


def source_prompt(row):
    if row['source']==NAMES[0]:return '\n本样本只有光学图片，完全没有辅助观测，aux_sentences必须为空。'
    if row['source']==NAMES[1]:return '\n本样本辅助观测仅为近红外/双短波红外假彩，没有InSAR和DEM，禁止提及InSAR或DEM。'
    return '\n本样本辅助观测是InSAR相对灰度和DEM相对灰度；两者只表达相对显示，不代表活动性或形变大小。'




def annotation_path(dest,rid):return Path(dest)/'review/samples'/f'{rid}.json'


def context(dest):
    dest=Path(dest);build=initialize(dest);p=dest/'review/manifest.json'
    prompt_hash=visual.digest([visual.POLICY,visual.DRAFT,visual.VERIFY,EXTRA_POLICY,GUIDE_VERSION])
    if p.exists():
        meta=json.loads(p.read_text());assert meta['prompt_sha256']==prompt_hash
        source_prompts={n:source_prompt({'source':n}) for n in NAMES}
        if 'source_prompts' not in meta:
            meta['source_prompts']=source_prompts;atomic_write_json(p,meta)
        assert meta['source_prompts']==source_prompts,'source prompts changed'
        guard_hash=sha256_file(Path(guards.__file__))
        if 'color_guard_sha256' not in meta:
            meta['color_guard_sha256']=guard_hash;atomic_write_json(p,meta)
        assert meta['color_guard_sha256']==guard_hash,'color guard changed'
        return meta
    meta={'model':visual.MODEL_NAME,'prompt_sha256':prompt_hash,'policy':visual.POLICY,'draft_prompt':visual.DRAFT,'verify_prompt':visual.VERIFY,
          'source_manifest_sha256':sha256_file(dest/'audit/selected.jsonl'),'human_confirmed':False,'formal_training_run':False,
          'images_are_independent':True,'groundtruth_used_for_annotation_only':True,'extra_policy':EXTRA_POLICY,'guide_version':GUIDE_VERSION,
          'source_prompts':{n:source_prompt({'source':n}) for n in NAMES},'color_guard_sha256':sha256_file(Path(guards.__file__))}
    atomic_write_json(p,meta);return meta


def assemble(row,dest,meta,draft,check,processor):
    """复用已核查的生成、坐标和筛选逻辑，新增真实编码长度检查。"""
    base=answer.split_original(row['answer'])
    candidate=draft_candidate(draft,len(row['images'])>1)
    decision=check['parsed']
    if not isinstance(decision,dict):raise ValueError('verification JSON missing')
    for key,target in [('keep_optical',candidate['optical_sentences']),('keep_aux',candidate['aux_sentences'])]:
        flags=decision.get(key)
        if not isinstance(flags,list) or len(flags)!=len(target) or any(type(v) is not bool for v in flags):
            raise ValueError(('verification_alignment',key,flags,len(target)))
    result=visual.review_one(None,None,row,meta,draft,check,paths=view_paths(row,dest),computed_parts=base)
    opt,aux,rejected=visual.conservative_sentences(row,result['optical_sentences'],result['aux_sentences'])
    guide=json.loads((Path(dest)/'review/inputs'/row['id']/'ready.json').read_text())
    visible=guide['visible_target_pixels']
    forbidden=re.compile(r'二值|核对|标注|像素|图[一二三四五六七八九0-9]|树冠|森林|从.{0,16}向|从.{0,16}延伸')
    clean=[]
    for sentence in opt:
        weak_shape=bool(re.search(r'细长|长条',sentence)) and (guide['elongation'] or 0)<2.5
        duplicate=any(len(set(sentence)&set(s))/max(1,len(set(sentence)|set(s)))>.72 for s in clean)
        if visible<16 or forbidden.search(sentence) or weak_shape or duplicate:
            rejected.append({'kind':'optical','sentence':sentence,'reason':'insufficient visible target, annotation leakage, geometry or repetition'})
        else:clean.append(sentence)
    opt=clean
    clean=[]
    for sentence in aux:
        if visible<16 or forbidden.search(sentence):
            rejected.append({'kind':'auxiliary','sentence':sentence,'reason':'insufficient visible target or annotation leakage'})
        else:clean.append(sentence)
    aux=clean
    opt,aux,color_rejections=guards.color_guard(row,opt,aux);rejected.extend(color_rejections)
    numeric,suffix=base
    while True:
        errs,o,m=answer.check_new_text(numeric,suffix,opt,aux,len(row['images'])>1)
        lengths=[len(processor.tokenizer(s,add_special_tokens=False)['input_ids'])+1 for s in (o,m)]
        if not errs and max(lengths)<=CONFIG['maximum_answer_tokens']:break
        if aux:rejected.append({'sentence':aux.pop(),'reason':'format or answer length'})
        elif opt:rejected.append({'sentence':opt.pop(),'reason':'format or answer length'})
        else:raise ValueError(('base_answer_not_encodable',row['id'],errs,lengths))
    result.update(after={'answer':m,'answer_optical':o},optical_sentences=opt,aux_sentences=aux,
                  additional_rejections=rejected,description_status='reviewed',answer_token_lengths=lengths)
    return result


def draft_candidate(draft,has_aux):
    obj=draft['parsed']
    if not isinstance(obj,dict):raise ValueError('draft JSON missing')
    for key in ('optical_sentences','aux_sentences'):
        if not isinstance(obj.get(key),list) or any(not isinstance(s,str) for s in obj[key]):
            raise ValueError(('draft_schema',key))
    return {'optical_sentences':visual.sentences(obj,'optical_sentences',2),
            'aux_sentences':visual.sentences(obj,'aux_sentences',1) if has_aux else []}


def recover_draft(model,processor,row,path,draft,dest):
    """格式失败时重新真实读图；一次失败只影响该样本，不丢弃同批其他返回。"""
    instruction='\n仅修正输出格式：optical_sentences与aux_sentences必须都是字符串数组，单句也必须用方括号包围；没有可靠信息用空数组[]，不能使用字符串、null或省略字段。只返回完整JSON对象。'
    for retry in range(4):
        try:
            candidate=draft_candidate(draft,len(row['images'])>1)
            if retry:draft['format_recovery']={'attempt':retry,'instruction':instruction}
            return draft,candidate
        except (ValueError,TypeError) as error:
            atomic_write_json(Path(dest)/'review/errors'/f"{row['id']}_draft_{time.time_ns()}.json",{'error':repr(error),'draft':draft,'retry':retry})
            if retry==3:raise
            prompt=visual.DRAFT+EXTRA_POLICY+source_prompt(row)+instruction
            draft=generate_safely(model,processor,[(path,prompt)],384)[0]


def generate_safely(model,processor,requests,max_new_tokens=210):
    """显存不足时缩小同批样本数，绝不删某个样本的辅助输入。"""
    import torch
    key=max(len(paths) for paths,prompt in requests);cap=_SAFE_CAP.get(key,len(requests))
    if len(requests)>cap:
        return [result for i in range(0,len(requests),cap) for result in generate_safely(model,processor,requests[i:i+cap],max_new_tokens)]
    try:return visual.generate_many(model,processor,requests,max_new_tokens=max_new_tokens)
    except torch.OutOfMemoryError:
        if len(requests)==1:raise
    # 离开except后再重试，避免异常栈继续持有刚才失败批次的显存。
    import gc
    gc.collect();torch.cuda.empty_cache();middle=len(requests)//2;_SAFE_CAP[key]=middle
    return generate_safely(model,processor,requests[:middle],max_new_tokens)+generate_safely(model,processor,requests[middle:],max_new_tokens)


def run(dest=DEFAULT,limit=0,follow=False,pilot=False):
    dest=Path(dest)
    if json.loads((dest/'build.json').read_text()).get('annotation_provider') in ('codex_session','local_resume'):raise RuntimeError('use the active provider entrypoint')
    meta=context(dest);selection=read_jsonl(dest/'audit/selected.jsonl')
    if not pilot:
        pilot_check=json.loads((dest/'review/pilot_check.json').read_text())
        assert pilot_check['passed'] and pilot_check['count']==12,'finish the training pilot before full generation'
        assert pilot_check['prompt_sha256']==meta['prompt_sha256'] and pilot_check['color_guard_sha256']==meta['color_guard_sha256']
    ordered=sorted(selection,key=lambda r:(r['source'],r['id']))
    if pilot:
        # 代表性试跑固定在训练部分，覆盖小、大目标；不会依据测试表现改规则。
        ordered=pilot_rows(ordered)
    pending=[r for r in ordered if not r['reuse_text'] and not annotation_path(dest,r['id']).exists()]
    if limit:pending=pending[:limit]
    if not pending:print('CAPTION_PENDING',0,flush=True);return
    total=len(pending);seed_all();model,processor=load();completed=0;failures=[];start=time.monotonic()
    while pending:
        source=pending[0]['source'];cap=16 if source==NAMES[0] else 24
        group=[r for r in pending if r['source']==source][:cap];group_ids={r['id'] for r in group}
        ready=[]
        candidates_to_read=[r for r in pending if r['source']==source] if follow else group
        for r in candidates_to_read:
            p=dest/'samples'/r['id']/'prepared.json'
            if p.exists():ready.append(json.loads(p.read_text()))
            if len(ready)==cap:break
        if not ready:
            if not follow:raise RuntimeError('Run export first, or use --follow while exporting')
            time.sleep(1);continue
        ready_ids={r['id'] for r in ready};paths=[view_paths(r,dest) for r in ready]
        tick=time.monotonic()
        draft_results=generate_safely(model,processor,[(p,visual.DRAFT+EXTRA_POLICY+source_prompt(r)) for p,r in zip(paths,ready)])
        candidates=[];accepted=[]
        for row,path,draft in zip(ready,paths,draft_results):
            try:draft,candidate=recover_draft(model,processor,row,path,draft,dest)
            except (ValueError,TypeError) as error:
                failures.append({'id':row['id'],'error':repr(error),'phase':'draft'})
                completed+=1;continue
            candidates.append(candidate)
            accepted.append((row,path,draft))
        checks=generate_safely(model,processor,[(p,visual.VERIFY+EXTRA_POLICY+source_prompt(r)+json.dumps(c,ensure_ascii=False)) for (r,p,d),c in zip(accepted,candidates)]) if accepted else []
        for index,((row,path,draft),check) in enumerate(zip(accepted,checks)):
            attempt=dest/'review/attempts'/f"{row['id']}.json"
            atomic_write_json(attempt,{'draft':draft,'verification':check})
            for retry in range(3):
                try:
                    result=assemble(row,dest,meta,draft,check,processor)
                    atomic_write_json(annotation_path(dest,row['id']),result);break
                except (ValueError,TypeError) as e:
                    atomic_write_json(dest/'review/errors'/f"{row['id']}_{retry}.json",{'error':repr(e),'draft':draft,'verification':check})
                    if retry==2:
                        failures.append({'id':row['id'],'error':repr(e)});break
                    candidate=candidates[index]
                    prompt=visual.VERIFY+EXTRA_POLICY+source_prompt(row)+json.dumps(candidate,ensure_ascii=False)+f"\n必须逐项给出布尔值：keep_optical恰好{len(candidate['optical_sentences'])}项，keep_aux恰好{len(candidate['aux_sentences'])}项。"
                    check=generate_safely(model,processor,[(path,prompt)],320)[0]
            completed+=1
            if completed%12==0:print('CAPTION',completed,total,'last',row['id'],'seconds',round(time.monotonic()-start),'failed',len(failures),flush=True)
        pending=[r for r in pending if r['id'] not in ready_ids]
        atomic_write_json(dest/'review/progress.json',{'remaining_in_run':len(pending),'completed_in_run':completed,'failed_in_run':len(failures),'model':visual.MODEL_NAME,'human_confirmed':False})
        print('BATCH',source,len(ready),'seconds',round(time.monotonic()-tick,1),'safe_caps',_SAFE_CAP,flush=True)
    atomic_write_json(dest/'review/failures.json',failures)
    if failures:raise RuntimeError(('caption_rows_still_incomplete',failures))
    print('CAPTIONS_DONE',completed,round(time.monotonic()-start),flush=True)


def publish(dest=DEFAULT):
    """仅在每个新增样本都有真实模型返回后发布最终清单，完整性仍待全量验证。"""
    dest=Path(dest)
    if json.loads((dest/'build.json').read_text()).get('annotation_provider') in ('codex_session','local_resume'):raise RuntimeError('use the active provider entrypoint')
    meta=context(dest);rows=read_jsonl(dest/'prepared.jsonl');outputs=[];status=Counter();old={r['id']:r for r in old_rows()}
    for r in rows:
        if r['reuse_text']:
            assert r['answer']==old[r['id']]['answer'] and r['answer_optical']==old[r['id']]['answer_optical']
            r={**r,'description_status':'reused'}
        else:
            p=annotation_path(dest,r['id']);assert p.exists(),('missing_review',r['id']);a=json.loads(p.read_text())
            assert a['status']=='reviewed' and a['human_confirmed'] is False
            assert a['prompt_sha256']==meta['prompt_sha256']
            r={**r,**a['after'],'description_status':'reviewed'}
        outputs.append(r);status[r['description_status']]+=1
    for s in SPLITS:write_jsonl_atomic(dest/f'{s}.jsonl',[r for r in outputs if r['split']==s])
    info=initialize(dest);summary={'total':len(outputs),'counts':dict(Counter(r['split'] for r in outputs)),
        'sources':dict(Counter(r['source'] for r in outputs)),'description_counts':dict(status),
        'positive_only':True,'empty_samples':0,'complete':False,'seed':SEED,'human_confirmed':False,
        'display_ranges_sha256':sha256_file(dest/'display_ranges.json'),
        'manifest_sha256':{s:sha256_file(dest/f'{s}.jsonl') for s in SPLITS},
        'limitations':json.loads((dest/'audit/partition.json').read_text())['limitations']}
    atomic_write_json(dest/'summary.json',summary);print('PUBLISHED_FOR_CHECKING',summary,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','publish']);p.add_argument('--dataset-dir',type=Path,default=DEFAULT)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--follow',action='store_true');p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    import fcntl
    lock=a.dataset_dir/'review/process.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=='run':run(a.dataset_dir,a.limit,a.follow,a.pilot)
        else:publish(a.dataset_dir)
