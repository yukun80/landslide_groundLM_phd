"""逐图质量复查，逐句真实判断；新记录与历史记录分开保存。"""
import json
import re
import time
from pathlib import Path
from .common import ROOT, NAMES, read_jsonl, atomic_write_json, sha256_file, digest
from .quality_review import REVIEW, load, status
from .views import view_paths
from .captions import generate_safely as generate_with_policy, draft_candidate, assemble
from .local_resume import candidate_issues
from ..description_v2 import answer, visual_review as visual

OPTICAL = '''依次提供光学原图、全图白色目标、目标原像素局部图、局部白色目标。
只回答第三张图中目标本身的实际颜色。灰色填充和白色标注不算目标颜色。
只写一条简短颜色陈述，不写形状、边缘、纹理、地物名称、位置或比较。
无法可靠确认颜色时留空。仅输出JSON：
{"optical_sentences":[],"aux_sentences":[],"uncertain":[]}。
本阶段完全没有辅助观测，aux_sentences必须是空数组[]，不能把位置或背景放入该字段。'''
AUXILIARY = '''查看辅助原图并对照目标位置，只写一条有可靠局部对应关系的辅助信息。
必须写明“InSAR图”或“假彩辅助图”及相对目标的位置；不写形变大小、活动性、速度、成因、风险。
假彩不是地表真实色。不能可靠对应就留空，不写泛泛“存在差异”。
只返回JSON：{"optical_sentences":[],"aux_sentences":[],"uncertain":[]}。optical_sentences必须为空。'''
VERIFY = '对照图片核查下面这一句话。只要有部分不支持或看不清就判false。只返回JSON：{"supported":true或false,"reason":"简短理由"}。候选句：'
EMPTY = '本轮没有候选描述。再次查看图片，不补写句子，只返回JSON：{"no_candidate":true,"reason":"无可核查候选句"}。'
POLICY = '按每张图片的角色定位目标。白色二值标注和灰色遮挡不是地表颜色。只依据实际目标，不猜测成因、风险、活动性或速度。遵守当前问题要求，只返回指定JSON。'


def generate_safely(model, processor, requests, max_new_tokens=160):
    responses=generate_with_policy(model,processor,requests,max_new_tokens,policy=POLICY)
    for response in responses:response['generation_policy']=POLICY
    return responses


def protocol():
    return digest([POLICY, OPTICAL, AUXILIARY, VERIFY, EMPTY, 'quality_review_v5_verified_color_presentation'])


def presentation(sentence):
    """只替换制作术语前缀；颜色原文、判断和原始返回均保留。"""
    return re.sub(r'^目标(?:原像素局部图显示为|原像素局部图呈现|像素呈现|像素为)', '目标呈', sentence)


def quality_paths(row, dest):
    """原图和原像素局部；只生成派生核对图，不改输入图片。"""
    import numpy as np
    from PIL import Image
    allpaths=view_paths(row,dest)
    folder=Path(dest)/REVIEW/'inputs'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'ready.json').exists():
        rgb=np.asarray(Image.open(ROOT/row['images'][-1]).convert('RGB'))
        mask=(np.asarray(Image.open(ROOT/row['mask']))>0)&(np.asarray(Image.open(ROOT/row['valid']))>0)
        yy,xx=np.where(mask);assert len(xx)
        box=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
        target=np.where(mask[...,None],rgb,128).astype('uint8')
        Image.fromarray(target).crop(box).save(folder/'target_crop.png')
        Image.fromarray(mask.astype('uint8')*255).crop(box).save(folder/'label_crop.png')
        atomic_write_json(folder/'ready.json',{'files':row['files'],'box':box})
    else:assert json.loads((folder/'ready.json').read_text())['files']==row['files']
    return [('optical_raw',ROOT/row['images'][-1]),('label',ROOT/row['mask']),('quality_target_crop',folder/'target_crop.png'),
            ('quality_label_crop',folder/'label_crop.png')]+[(r,p) for r,p in allpaths if r.startswith('auxiliary_')]


def quality_issues(row, candidate, kind):
    issues=candidate_issues(row,candidate,kind)
    if kind=='optical':
        for i,s in enumerate(candidate['optical_sentences']):
            if re.search(r'周围|背景|植被|纹理|锯齿|凹凸|粗糙|颗粒|裂纹|轮廓|边缘|块状|条状|局部图|无法|不清|难以|标注|裸露|地表',presentation(s)):
                issues.append({'field':'optical_sentences','index':i,'sentence':s,'reason':'unresolved_texture_or_generic_background'})
    return issues


def receipt(model, processor, paths, prompt, folder, stem, valid, generator=None):
    for attempt in range(3):
        response=(generator or generate_safely)(model,processor,[(paths,prompt)],160)[0]
        response['request_prompt']=prompt
        atomic_write_json(folder/'attempts'/f'{stem}-{time.time_ns()}.json',response)
        try:
            valid(response)
            return response
        except (ValueError,TypeError,AssertionError,KeyError) as error:
            if attempt==2:raise
            prompt += '\n上次返回不符合要求：'+str(error)+'。仅返回要求的完整JSON对象，不重复说明。'


def phase(model, processor, row, paths, folder, auxiliary=False, generator=None):
    key='aux_sentences' if auxiliary else 'optical_sentences'
    kind='auxiliary' if auxiliary else 'optical'
    if not auxiliary:paths=[(r,p) for r,p in paths if not r.startswith('auxiliary_')]
    prompt=AUXILIARY if auxiliary else OPTICAL
    if not auxiliary and row['positive_pixels']<256:prompt+='目标原始像素较少，禁止纹理、颗粒、裂纹描述。'
    def valid_draft(response):
        obj=response['parsed'];assert isinstance(obj,dict)
        for field in ('optical_sentences','aux_sentences'):
            assert isinstance(obj.get(field),list) and all(isinstance(s,str) for s in obj[field])
        assert not obj['optical_sentences' if auxiliary else 'aux_sentences'], ('本轮只检查辅助信息，optical_sentences必须为空[]' if auxiliary else '本轮没有辅助观测，aux_sentences必须为空[]，不能写位置或背景')
        assert len(obj[key])<=(1 if auxiliary else 2)
    draft=receipt(model,processor,paths,prompt,folder,row['id']+'-'+kind+'-draft',valid_draft,generator)
    candidate=draft_candidate(draft,len(row['images'])>1)
    issues=quality_issues(row,candidate,kind)
    bad={(e['field'],e['index']) for e in issues}
    candidate={k:[s for i,s in enumerate(v) if (k,i) not in bad] for k,v in candidate.items()}
    checks=[];kept=[]
    def valid_check(r):
        assert isinstance(r['parsed'],dict) and type(r['parsed'].get('supported')) is bool
    if candidate[key]:
        for i,sentence in enumerate(candidate[key]):
            check=receipt(model,processor,paths,VERIFY+sentence,folder,
                          row['id']+'-'+kind+f'-sentence-{i}',valid_check,generator)
            checks.append({'sentence':sentence,'receipt':check})
            if check['parsed']['supported']:kept.append(sentence)
        empty=None
    else:
        def valid_empty(r):
            assert isinstance(r['parsed'],dict) and r['parsed'].get('no_candidate') is True
        empty=receipt(model,processor,paths,EMPTY,folder,row['id']+'-'+kind+'-empty',valid_empty,generator)
    return {'draft':draft,'candidate':candidate,'checks':checks,'empty_check':empty,
            'kept':kept,'preverification_rejections':issues}


def phases(model, processor, rows, allpaths, folder, auxiliary=False):
    """同类图片成批生成；每条每句的真实返回仍独立保存，失败单条重试。"""
    # Reuse the strict single-sample implementation with a request cache: only
    # actual matching requests are served; retries always invoke the model.
    paths = [[(r,p) for r,p in ps if auxiliary or not r.startswith('auxiliary_')] for ps in allpaths]
    prompts=[]
    for row in rows:
        prompt=AUXILIARY if auxiliary else OPTICAL
        if not auxiliary and row['positive_pixels']<256:prompt+='目标原始像素较少，禁止纹理、颗粒、裂纹描述。'
        prompts.append(prompt)
    drafts=generate_safely(model,processor,list(zip(paths,prompts)),160)
    cache={}
    def key(ps,prompt):return (tuple((r,str(p)) for r,p in ps),prompt)
    requests=[]
    for row,ps,prompt,draft in zip(rows,paths,prompts,drafts):
        cache[key(ps,prompt)]=draft
        try:
            obj=draft['parsed'];assert isinstance(obj,dict)
            assert all(isinstance(obj.get(k),list) and all(isinstance(s,str) for s in obj[k]) for k in ('optical_sentences','aux_sentences'))
            field='aux_sentences' if auxiliary else 'optical_sentences'
            assert not obj['optical_sentences' if auxiliary else 'aux_sentences']
            assert len(obj[field])<=(1 if auxiliary else 2)
            candidate=draft_candidate(draft,len(row['images'])>1)
            issues=quality_issues(row,candidate,'auxiliary' if auxiliary else 'optical')
            bad={(e['field'],e['index']) for e in issues}
            sentences=[s for i,s in enumerate(candidate[field]) if (field,i) not in bad]
            requests.extend((ps,VERIFY+s) for s in sentences)
            if not sentences:requests.append((ps,EMPTY))
        except (ValueError,TypeError,AssertionError,KeyError):pass
    if requests:
        for (ps,prompt),response in zip(requests,generate_safely(model,processor,requests,160)):
            cache[key(ps,prompt)]=response
    original=generate_safely
    def cached(m,p,requests,max_new_tokens=160):
        results=[]
        for ps,prompt in requests:
            k=key(ps,prompt)
            results.append(cache.pop(k) if k in cache else original(m,p,[(ps,prompt)],max_new_tokens)[0])
        return results
    results={};failures=[]
    for row,ps in zip(rows,allpaths):
        try:results[row['id']]=phase(model,processor,row,ps,folder,auxiliary,generator=cached)
        except (ValueError,TypeError,AssertionError,KeyError) as e:failures.append({'id':row['id'],'error':repr(e)})
    return results,failures


def run(dest, pilot=False, limit=0):
    dest=Path(dest);m=load(dest);folder=dest/REVIEW
    rule_file=folder/'protocols'/f'{protocol()}.json'
    if not rule_file.exists():
        atomic_write_json(rule_file,{'policy':POLICY,'optical':OPTICAL,'auxiliary':AUXILIARY,
            'verify':VERIFY,'empty':EMPTY,'protocol_sha256':protocol(),
            'presentation_rule':'only exact target pixel/crop reference prefixes; color words unchanged',
            'formal_training_run':False})
    candidates={e['id']:e for e in read_jsonl(folder/'candidates.jsonl')}
    wanted=set(m['pilot_ids']) if pilot else set(candidates)
    from .quality_review import merged_entries
    wanted.difference_update(merged_entries(dest))
    if not pilot:
        approval=json.loads((folder/'pilot_approval.json').read_text())
        assert approval['passed'] and approval['protocol_sha256']==protocol()
        for rid,h in approval['records_sha256'].items():assert sha256_file(folder/'samples'/f'{rid}.json')==h
        assert set(approval['records_sha256'])==set(m['pilot_ids'])
    rows=[r for r in read_jsonl(dest/'prepared.jsonl') if r['id'] in wanted and not (folder/'samples'/f"{r['id']}.json").exists()]
    rows.sort(key=lambda r:(r['source'],r['id']))
    if limit:rows=rows[:limit]
    if not rows:return
    from ..domain_runtime import load as load_model,seed_all,BASE
    from .resume_records import load_manifest
    base_record=load_manifest(dest)
    assert str(BASE)==base_record['base_path']
    if not pilot:
        for name,h in base_record['model_files_sha256'].items():assert sha256_file(BASE/name)==h
    from ..train.data.domain import training_answer
    from PIL import Image
    import numpy as np
    seed_all();model,processor=load_model()
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    failures=[]
    phase_results={};aux_results={}
    for i,row in enumerate(rows,1):
        e=candidates[row['id']]
        try:
            if e['previous_path']:assert sha256_file(dest/e['previous_path'])==e['previous_sha256']
            if row['id'] not in phase_results:
                group=[r for r in rows[i-1:] if r['source']==row['source']][:8]
                allpaths=[quality_paths(r,dest) for r in group]
                found,errors=phases(model,processor,group,allpaths,folder)
                phase_results.update({r['id']:found.get(r['id']) for r in group});failures.extend(errors)
                if len(row['images'])>1:
                    found,errors=phases(model,processor,group,allpaths,folder,True)
                    aux_results.update(found);failures.extend(errors)
            optical=phase_results.pop(row['id'])
            aux=aux_results.pop(row['id'],None) if len(row['images'])>1 else None
            assert optical is not None and (len(row['images'])==1 or aux is not None),'phase_review_failed'
            opt=[presentation(s) for s in optical['kept']];auxs=aux['kept'] if aux else []
            mask=np.asarray(Image.open(ROOT/row['mask']))>0;valid=np.asarray(Image.open(ROOT/row['valid']))>0
            clean_row={**row,'answer':training_answer(mask,valid)}
            composed={'parsed':{'optical_sentences':opt,'aux_sentences':auxs,'uncertain':[]},'composition_only':True}
            verified={'parsed':{'keep_optical':[True]*len(opt),'keep_aux':[True]*len(auxs),'reasons':[]},'composition_only':True}
            result=assemble(clean_row,dest,{'prompt_sha256':protocol()},composed,verified,processor)
            decision=('retained' if result['after']==e['before'] else 'rewritten')
            if not result['optical_sentences']:
                decision='removed_unsupported' if e['optical_sentences'] else 'confirmed_no_appearance'
            result.update(before=e['before'],quality_protocol_sha256=protocol(),
                          quality_manifest_sha256=sha256_file(folder/'manifest.json'),
                          optical_review=optical,auxiliary_review=aux,quality_decision=decision,
                          presentation_changes=[{'original':s,'final':presentation(s)} for s in optical['kept'] if presentation(s)!=s],
                          review_reasons=e['reasons'],previous_path=e['previous_path'],
                          previous_sha256=e['previous_sha256'],human_confirmed=False)
            atomic_write_json(folder/'samples'/f"{row['id']}.json",result)
        except (ValueError,AssertionError,TypeError,KeyError) as error:
            failure={'id':row['id'],'error':repr(error)};failures.append(failure)
            atomic_write_json(folder/'failures'/f"{row['id']}-{time.time_ns()}.json",failure)
        print('QUALITY_REVIEW',i,len(rows),row['id'],'failures',len(failures),flush=True)
        atomic_write_json(folder/'progress.json',status(dest))
    if failures:raise RuntimeError(('quality_review_incomplete',failures))
