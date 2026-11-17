"""官方初始模型接续标注：试跑须经实际看图放行，历史文字只读。"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .common import (ROOT, DEFAULT, SPLITS, NAMES, read_jsonl, atomic_write_json,
                     write_jsonl_atomic, sha256_file, initialize)
from . import resume_records as records
from .captions import (assemble, draft_candidate, generate_safely, source_prompt, EXTRA_POLICY)
from .views import view_paths
from ..description_v2 import visual_review as visual, answer


def check_gate(dest, m, require_encoding=False):
    """试跑的实际图片核查记录与12条当前答案必须一一对应。"""
    gate = json.loads((Path(dest) / records.REVIEW / 'pilot_check.json').read_text())
    assert gate['passed'] and gate['prompt_sha256'] == m['prompt_sha256']
    assert set(gate['records_sha256']) == set(m['pilot_ids'])
    for rid, h in gate['records_sha256'].items():
        assert sha256_file(records.record_path(dest, rid)) == h, ('pilot_changed', rid)
    if require_encoding:
        encoding=json.loads((Path(dest)/'audit/pilot_encoding_check.json').read_text())
        assert encoding['passed'] and encoding['count']==12 and encoding['device']=='cpu'
        assert encoding['scope']=='training_pilot_only' and not encoding['cuda_initialized']
        for rid in m['pilot_ids']:
            from .common import digest
            row=json.loads(records.record_path(dest,rid).read_text())['after']
            assert encoding['answer_text_sha256'][rid]==digest([row['answer'],row['answer_optical']])
        receipts=json.loads((Path(dest)/records.REVIEW/'pilot_records_check.json').read_text())
        assert receipts['passed'] and receipts['records_sha256']==gate['records_sha256']
    return gate


def save_attempt(dest, rid, phase, receipt):
    p = Path(dest) / records.REVIEW / 'attempts' / f'{rid}_{time.time_ns()}_{phase}.json'
    atomic_write_json(p, receipt)


def candidate_issues(row, candidate, phase):
    """仅删除不合规则的整句；不凭统计信息生成替代外观。"""
    import re
    issues=[]
    for key,values in candidate.items():
        for index,s in enumerate(values):
            reason=None
            if len(s)>36:reason='sentence_too_long'
            elif any(p.search(s) for p in answer.FORBIDDEN_RE):reason='forbidden_training_word'
            elif phase=='optical' and key=='optical_sentences':
                if visual.BANNED.search(s):reason='unsupported_or_forbidden_claim'
                elif row['positive_pixels']<records.NATIVE_TEXTURE_LIMIT and re.search(r'纹理|斑驳|颗粒|裂纹',s):
                    reason='native_target_too_small_for_texture'
                elif row['positive_pixels']/row['valid_pixels']>=.95 and re.search(r'周围|背景',s):
                    reason='outside_relation_unreliable_for_nearly_full_target'
            if reason:issues.append({'field':key,'index':index,'sentence':s,'reason':reason})
    return issues


def checked_draft(model, processor, row, paths, receipt, dest, prompt=None, phase="draft"):
    """格式失败必须真实重试；合规格式里的存疑句先要求改写，仍无依据则整句删除并留痕。"""
    for attempt in range(4):
        if prompt:receipt.setdefault('request_prompt',receipt.get('retry_request_prompt',prompt))
        save_attempt(dest,row['id'],phase+'_draft',receipt)
        try:
            candidate=draft_candidate(receipt,len(row['images'])>1)
            assert len(receipt['parsed']['optical_sentences'])<=2
            assert len(receipt['parsed']['aux_sentences'])<=1
            if phase=='optical':assert not candidate['aux_sentences']
            elif phase=='auxiliary':assert not candidate['optical_sentences']
        except (ValueError,TypeError,AssertionError):
            if attempt==3:raise ValueError(('invalid_draft_after_retries',row['id']))
            candidate=None
        issues=candidate_issues(row,candidate,phase) if candidate is not None else []
        if candidate is not None and not issues:return receipt,candidate
        if candidate is not None and attempt>=2:
            # 原始JSON和原句仍在receipt中，只选择规则允许的候选进入第二遍核查。
            receipt['preverification_rejections']=issues
            bad={(e['field'],e['index']) for e in issues}
            candidate={key:[text for i,text in enumerate(values) if (key,i) not in bad]
                       for key,values in candidate.items()}
            return receipt,candidate
        instruction=('只输出完整JSON，optical_sentences与aux_sentences始终为字符串数组。'
                     '每句尽量少于20个汉字。颜色、形状分别用短句写，不写地物类型或植被覆盖。'
                     '没有可靠信息用[]，不复制被指出的存疑内容。')
        if phase=='optical':instruction+='只写目标本身，不写周围或背景。原始目标很小时不写纹理。aux_sentences必须是[]。'
        if phase=='auxiliary':instruction+='optical_sentences必须是[]；只写一条有可靠局部对应关系的辅助句，不能判断形变强弱。'
        if issues:instruction+='需要纠正的问题：'+json.dumps(issues,ensure_ascii=False)
        request=(prompt or (visual.DRAFT+EXTRA_POLICY+records.DETAIL_PROMPT+source_prompt(row)))+instruction
        receipt=generate_safely(model,processor,[(paths,request)],384)[0]
        receipt['retry_request_prompt']=request


def read_phase(model, processor, group, paths, dest, auxiliary=False):
    """光学阶段不传辅助图；辅助阶段真实再读全部图片，分别保留两遍返回。"""
    phase = 'auxiliary' if auxiliary else 'optical'
    if not auxiliary:
        paths=[[(role,p) for role,p in ps if not role.startswith('auxiliary_')] for ps in paths]
    def prompt_for(row):
        if auxiliary:
            return records.AUX_DRAFT + EXTRA_POLICY + records.DETAIL_PROMPT + source_prompt(row)
        small = ('原始目标不足256个像素，不足以支持纹理描述；只写可靠颜色或整体轮廓，'
                 '禁止使用纹理、斑驳、颗粒、裂纹等细节词。' if row['positive_pixels'] < records.NATIVE_TEXTURE_LIMIT else '')
        return visual.DRAFT + EXTRA_POLICY + records.DETAIL_PROMPT + source_prompt({'source':NAMES[0]}) + small
    drafts = generate_safely(model, processor, [(ps,prompt_for(r)) for r,ps in zip(group,paths)])
    candidates = []; failures = []
    for row,ps,draft in zip(group,paths,drafts):
        try:
            draft,candidate = checked_draft(model,processor,row,ps,draft,dest,
                prompt=prompt_for(row),phase=phase)
            candidates.append((row,ps,draft,candidate))
        except (ValueError,TypeError) as error:
            failures.append({'id':row['id'],'phase':phase+'_draft','error':repr(error)})
    def verification_prompt(candidate,instruction=''):
        sentences=[(key,s) for key in ('optical_sentences','aux_sentences') for s in candidate[key]]
        if len(sentences)==1:
            return (EXTRA_POLICY+records.DETAIL_PROMPT+
                '\n'+records.SINGLE_VERIFY+json.dumps(sentences[0][1],ensure_ascii=False)+instruction)
        return visual.VERIFY+EXTRA_POLICY+records.DETAIL_PROMPT+json.dumps(candidate,ensure_ascii=False)+instruction
    prompts=[verification_prompt(c) for r,ps,d,c in candidates]
    checks=generate_safely(model,processor,[(ps,prompt) for (r,ps,d,c),prompt in zip(candidates,prompts)]) if candidates else []
    result={}
    for (row,ps,draft,candidate),check in zip(candidates,checks):
        for retry in range(3):
            check.setdefault('request_prompt',check.get('retry_request_prompt',verification_prompt(candidate)))
            save_attempt(dest,row['id'],phase+'_verification',check)
            try:
                obj=check['parsed'];assert isinstance(obj,dict)
                single=[(key,s) for key in ('optical_sentences','aux_sentences') for s in candidate[key]]
                if len(single)==1:
                    assert type(obj.get('supported')) is bool
                    check['single_sentence_evaluation']={'field':single[0][0],'sentence':single[0][1],
                                                        'supported':obj['supported']}
                    result[row['id']]={'draft':draft,'verification':check,'candidate':candidate}
                    break
                if not candidate['optical_sentences'] and not candidate['aux_sentences']:
                    # 草稿明确为空时没有可判断的句子。仍保留第二次真实读图返回，
                    # 但不把它的多余布尔值当作任何句子通过，也不伪造空数组响应。
                    check['candidate_evaluation']='not_applicable_explicit_empty_draft'
                    result[row['id']]={'draft':draft,'verification':check,'candidate':candidate}
                    break
                for key,target in [('keep_optical',candidate['optical_sentences']),('keep_aux',candidate['aux_sentences'])]:
                    flags=obj.get(key)
                    assert isinstance(flags,list) and all(type(v) is bool for v in flags)
                    if not target and flags and retry>=1 and len(flags)<=2 and not any(flags):
                        # 真正重新核查后仍为不存在的候选多给false：保留原返回，忽略无对应句的字段。
                        # 不删截任何实际候选的true，也不将未知判断改成通过。
                        check.setdefault('unused_zero_candidate_fields',{})[key]=list(flags)
                    else:assert len(flags)==len(target)
                result[row['id']]={'draft':draft,'verification':check,'candidate':candidate}
                break
            except (TypeError,AssertionError,AttributeError) as error:
                if retry==2:
                    failures.append({'id':row['id'],'phase':phase+'_verification','error':repr(error)});break
                instruction=f"keep_optical必须有{len(candidate['optical_sentences'])}个布尔值，keep_aux必须有{len(candidate['aux_sentences'])}个布尔值；与候选句逐一对应。没有候选的字段必须是[]，不能写[false]。"
                request=verification_prompt(candidate,instruction)
                check=generate_safely(model,processor,[(ps,request)],320)[0]
                check['retry_request_prompt']=request
    return result,failures


def combine_phases(row, dest, manifest, optical, auxiliary, processor):
    """内部合并只用于共用筛选；保存的模型原始返回始终按真实读取阶段分开。"""
    opt=optical['candidate']['optical_sentences']
    aux=auxiliary['candidate']['aux_sentences'] if auxiliary else []
    draft={'parsed':{'optical_sentences':opt,'aux_sentences':aux,'uncertain':
        visual.notes_list(optical['draft']['parsed'].get('uncertain',[]))+
        (visual.notes_list(auxiliary['draft']['parsed'].get('uncertain',[])) if auxiliary else [])},
        'composition_only':True}
    def flags(phase,key):
        if not phase or not phase['candidate'][key]:return []
        single=phase['verification'].get('single_sentence_evaluation')
        if single:
            assert single['field']==key and single['sentence']==phase['candidate'][key][0]
            return [single['supported']]
        return phase['verification']['parsed']['keep_optical' if key=='optical_sentences' else 'keep_aux']
    check={'parsed':{'keep_optical':flags(optical,'optical_sentences'),
        'keep_aux':flags(auxiliary,'aux_sentences'),
        'reasons':visual.notes_list(optical['verification']['parsed'].get('reasons',optical['verification']['parsed'].get('reason',[])))+
        (visual.notes_list(auxiliary['verification']['parsed'].get('reasons',auxiliary['verification']['parsed'].get('reason',[]))) if auxiliary else [])},
        'composition_only':True}
    result=assemble(row,dest,manifest,draft,check,processor)
    result['draft']=optical['draft'];result['verification']=optical['verification']
    if auxiliary:
        result['auxiliary_draft']=auxiliary['draft'];result['auxiliary_verification']=auxiliary['verification']
    result['generation_layout']='separate_optical_auxiliary_v2'
    result['optical_candidate']=optical['candidate']
    result['auxiliary_candidate']=auxiliary['candidate'] if auxiliary else None
    return result


def run(dest, pilot=False, limit=0):
    dest=Path(dest);m=records.load_manifest(dest)
    plan=records.work_plan(dest) if not pilot else None
    if not pilot:check_gate(dest,m,require_encoding=True)
    wanted=set(m['pilot_ids'] if pilot else plan['required_ids'])
    rows=[r for r in read_jsonl(dest/'prepared.jsonl') if r['id'] in wanted and not records.record_path(dest,r['id']).exists()]
    rows.sort(key=lambda r:(SPLITS.index(r['split']),NAMES.index(r['source']),r['id']))
    if limit:rows=rows[:limit]
    if not rows:print('NO_PENDING_DESCRIPTIONS',flush=True);return
    from ..domain_runtime import load,seed_all,BASE
    assert str(BASE)==m['base_path']
    seed_all();model,processor=load()
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    failures=[];start=time.monotonic();completed=0;pending=list(rows)
    while pending:
        first=pending[0];cap=16 if first['source']==NAMES[0] else 24
        group=[r for r in pending if (r['source'],r['split'])==(first['source'],first['split'])][:cap]
        group_ids={r['id'] for r in group};allpaths=[view_paths(r,dest) for r in group]
        optical_paths=[[(role,p) for role,p in ps if not role.startswith('auxiliary_')] for ps in allpaths]
        tick=time.monotonic()
        optics,errors=read_phase(model,processor,group,optical_paths,dest);failures.extend(errors)
        auxiliary={}
        if len(first['images'])>1:
            auxrows=[r for r in group if r['id'] in optics]
            auxpaths=[ps for r,ps in zip(group,allpaths) if r['id'] in optics]
            auxiliary,errors=read_phase(model,processor,auxrows,auxpaths,dest,auxiliary=True);failures.extend(errors)
        for row in group:
            if row['id'] not in optics or (len(row['images'])>1 and row['id'] not in auxiliary):continue
            try:
                result=combine_phases(row,dest,m,optics[row['id']],auxiliary.get(row['id']),processor)
                assert all(len(s)<=36 for s in result['optical_sentences']+result['aux_sentences'])
                previous=dest/'review/samples'/f"{row['id']}.json"
                before=json.loads(previous.read_text())['after'] if previous.exists() else {k:row[k] for k in ('answer','answer_optical')}
                guide=json.loads((dest/'review/inputs'/row['id']/'ready.json').read_text())
                result.update(provider=records.PROVIDER,before=before,
                    revision_reason='pilot_or_previous_audit_flag' if previous.exists() else 'missing_description',
                    previous_path=str(previous.relative_to(dest)) if previous.exists() else None,
                    previous_sha256=sha256_file(previous) if previous.exists() else None,
                    base_only_reason=('target_unresolved_at_input_size' if guide['visible_target_pixels']<16 else
                        'no_supported_appearance_after_two_pass_review') if not result['optical_sentences'] and not result['aux_sentences'] else None,
                    changed_from_previous=before!=result['after'],model_files_manifest_sha256=sha256_file(records.manifest_path(dest)))
                atomic_write_json(records.record_path(dest,row['id']),result);completed+=1
            except (ValueError,TypeError,AssertionError) as error:
                failures.append({'id':row['id'],'phase':'assembly','error':repr(error)})
        pending=[r for r in pending if r['id'] not in group_ids]
        plan_file=dest/records.REVIEW/'work_plan.json'
        plan_ready=plan_file.exists() and 'additional_audit_sha256' in json.loads(plan_file.read_text())
        state=records.progress(dest) if plan_ready else {'remaining':'audit_running'}
        print('LOCAL_RESUME_BATCH',first['source'],first['split'],len(group),'saved',completed,'failed',len(failures),
              'remaining',state['remaining'],'seconds',round(time.monotonic()-tick,1),flush=True)
    atomic_write_json(dest/records.REVIEW/'last_run.json',{'saved':completed,'failures':failures,'pilot':pilot,
        'seconds':time.monotonic()-start,'formal_training_run':False})
    if failures:raise RuntimeError(('descriptions_still_missing',failures))


def approve_pilot(dest, submission):
    """只保存实际看图后的人工编写核查说明；脚本本身不产生看图声明。"""
    dest = Path(dest); m = records.load_manifest(dest)
    entries = json.loads(Path(submission).read_text())
    assert len(entries) == 12 and {e['id'] for e in entries} == set(m['pilot_ids'])
    hashes = {}
    for entry in entries:
        assert entry['passed'] is True and entry['note'].strip()
        p = records.record_path(dest, entry['id']); r = json.loads(p.read_text())
        assert r['prompt_sha256'] == m['prompt_sha256']
        receipt=r.get('auxiliary_draft',r['draft'])
        expected = [{'role': x['role'], 'path': x['path'], 'sha256': x['sha256']} for x in receipt['images']]
        assert entry['viewed_images'] == expected
        assert entry['record_sha256'] == sha256_file(p)
        hashes[entry['id']] = sha256_file(p)
    gate = {'passed': True, 'count': 12, 'entries': entries, 'records_sha256': hashes,
            'prompt_sha256': m['prompt_sha256'], 'checked_at': visual.now(),
            'review_type': 'current_codex_image_view_not_human_or_independent_model',
            'formal_training_run': False}
    atomic_write_json(dest / records.REVIEW / 'pilot_check.json', gate)
    print('PILOT_VISUAL_CHECK_SAVED', flush=True)


def revise_pilot(dest, submission):
    """试跑未放行时保留失败轮次，更新读图规则并真正重做，不能改已冻结全量规则。"""
    dest=Path(dest);folder=dest/records.REVIEW
    assert not (folder/'pilot_check.json').exists(),'approved rules are frozen'
    previous=json.loads(records.manifest_path(dest).read_text())
    entries=json.loads(Path(submission).read_text())
    assert {e['id'] for e in entries}==set(previous['pilot_ids'])
    assert any(e['passed'] is False for e in entries)
    paths=list((folder/'samples').glob('*.json'))
    assert paths and {p.stem for p in paths}<=set(previous['pilot_ids'])
    assert all(e['note'].strip() and e['viewed_images'] for e in entries)
    round_folder=folder/'pilot_rounds'/str(sum(p.is_dir() for p in (folder/'pilot_rounds').glob('*'))+1)
    assert not round_folder.exists();round_folder.mkdir(parents=True)
    atomic_write_json(round_folder/'manifest.json',previous)
    atomic_write_json(round_folder/'visual_check.json',{'passed':False,'entries':entries,'checked_at':visual.now()})
    (round_folder/'samples').mkdir()
    for p in paths:p.rename(round_folder/'samples'/p.name)
    if (folder/'last_run.json').exists():(folder/'last_run.json').rename(round_folder/'last_run.json')
    previous.update(prompt_sha256=records.rules_hash(),detail_prompt=records.DETAIL_PROMPT,
        auxiliary_draft_prompt=records.AUX_DRAFT,generation_layout='separate_optical_auxiliary_v2',
        native_texture_limit=records.NATIVE_TEXTURE_LIMIT,
        single_sentence_verify_prompt=records.SINGLE_VERIFY,
        previous_pilot_manifest_path=str((round_folder/'manifest.json').relative_to(dest)),
        previous_pilot_manifest_sha256=sha256_file(round_folder/'manifest.json'),
        rule_revision_reason='; '.join(e['note'] for e in entries if not e['passed']))
    atomic_write_json(records.manifest_path(dest),previous)
    print('FAILED_PILOT_PRESERVED_REPEAT_REQUIRED',round_folder,flush=True)


def publish(dest):
    """全部必修项完成才导出；每条答案附采用来源和记录摘要。"""
    dest = Path(dest); m = records.load_manifest(dest); check_gate(dest, m, require_encoding=True)
    records.check_history(dest, m)
    assert records.progress(dest)['remaining'] == 0
    out = []; adopted = []; counts = Counter()
    for row in read_jsonl(dest / 'prepared.jsonl'):
        source, path, record = records.resolve_record(dest, row, m)
        r = {**row, **(record['after'] if record is not None else {}),
             'description_status': 'reused' if source == 'reused' else 'reviewed',
             'description_provider': source}
        counts[source] += 1; out.append(r)
        adopted.append({'id': r['id'], 'source': source,
                        'path': str(path.relative_to(dest)) if path else None,
                        'sha256': sha256_file(path) if path else None})
    for split in SPLITS:
        write_jsonl_atomic(dest / f'{split}.jsonl', [r for r in out if r['split'] == split])
    write_jsonl_atomic(dest / records.REVIEW / 'adopted.jsonl', adopted)
    summary = {'annotation_provider': records.PROVIDER, 'total': len(out), 'complete': False,
               'counts': dict(Counter(r['split'] for r in out)), 'sources': dict(Counter(r['source'] for r in out)),
               'answer_sources': dict(counts), 'positive_only': True, 'empty_samples': 0,
               'human_confirmed': False, 'formal_training_run': False,
               'display_ranges_sha256': sha256_file(dest / 'display_ranges.json'),
               'manifest_sha256': {s: sha256_file(dest / f'{s}.jsonl') for s in SPLITS},
               'limitations': json.loads((dest / 'audit/partition.json').read_text())['limitations']}
    atomic_write_json(dest / 'summary.json', summary)
    print('LOCAL_RESUME_PUBLISHED_FOR_CHECKING', summary, flush=True)


def validate_all(dest, rows):
    """最终检查实际采用的每一种来源；不把历史核对冒充本轮重新看图。"""
    dest = Path(dest); m = records.load_manifest(dest); check_gate(dest, m)
    records.check_history(dest, m)
    from .resume_validation import validate_records
    return validate_records(dest, rows, m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['init', 'audit', 'pilot', 'revise-pilot', 'approve-pilot', 'run', 'status', 'publish'])
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT)
    p.add_argument('--input', type=Path); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(); dest = a.dataset_dir.resolve()
    if a.stage == 'status': print(json.dumps(records.progress(dest), ensure_ascii=False), flush=True); return
    import fcntl
    lock = dest / ('review/local_resume/audit.lock' if a.stage=='audit' else 'review/local_resume/generation.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'init': print('LOCAL_RESUME_INITIALIZED', records.activate(dest)['model'], flush=True)
        elif a.stage == 'audit': print('AUDIT_DONE', {k:v for k,v in records.audit_existing(dest,a.workers).items() if not k.endswith('_ids')}, flush=True)
        elif a.stage in ('pilot', 'run'): run(dest, pilot=a.stage=='pilot', limit=a.limit)
        elif a.stage == 'approve-pilot':
            assert a.input; approve_pilot(dest, a.input)
        elif a.stage == 'revise-pilot':
            assert a.input; revise_pilot(dest,a.input)
        elif a.stage == 'publish': publish(dest)


if __name__ == '__main__': main()
