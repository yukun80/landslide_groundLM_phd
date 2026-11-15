"""限定范围的逐图视觉复核：保留原始模型输出，第二遍核对后才形成候选答案。

模型实际接收独立图片，不用拼图代替看图。不把文件存在、推理成功或格式正确
当作外观已经证实；不确定句删除，所有结果均为未获人工确认的模型标注。
"""
import argparse
import copy
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import answer, io, revision, spec

RUN = spec.V2 / 'review_20261112'
MODEL_NAME = 'project-checkpoints/Xiaomi-Lens-0 (initial weights, no domain finetuning)'
POLICY = '''仅描述滑坡标注白色区域对应的目标自身。目标颜色、明暗与材质没有预设，必须回到目标原始像素核对，禁止照抄提示词中的例子。
品红色只是定位标记，不是地表颜色。不要把旁边的浅色带、河道、背景说成目标外观。
只写模型输入尺寸中能看见的外观；不根据文件名、常识、面积、位置模板补写。不得把二值图的黑色背景当成目标；二值图只有白色为目标，原图第五张核对图的中灰背景也不是目标。
辅助图只写目标所在位置及邻近位置的相对差异和空间关系；假彩颜色不是真实地表色。
InSAR仅在存在可辨认局部差异时写出明暗、形状、位置以及与目标重合或在其外侧的关系。
可以写“疑似形变异常线索”，但亮度不代表形变大小，不得写活动滑坡、形变较大、具体速率、单位、阈值、成因、风险、时间或地理方向。
均匀图、细碎孤立点、模糊目标不强行补写；不确定内容留空，不用套话凑字数。只描述颜色、明暗、轮廓、边界、纹理；不要推测地物类型，不写河道、道路、沟谷、土质、土体或无植被。'''
DRAFT = '''请按给出的独立图片逐张查看，核对目标的位置后，以中文输出JSON：
{"optical_sentences":["最多两句目标本身的颜色、形状或纹理"],"aux_sentences":["最多一句有明确目标位置关系的辅助信息"],"uncertain":["具体疑问"]}。
无可靠信息的数组留空，禁止输出数字、坐标、区域标记。每句不超过35个汉字，不使用“标注区”“明显”“显著”等词。颜色和形状分开成句，不写河道、道路、土质或土体。辅助句明确写“InSAR图”或“假彩辅助图”，不得写辅助图编号。
'''
VERIFY = '''再次对照这些独立图片核查下面每一句候选描述。特别检查目标本身与邻近区域是否混淆，位置、颜色、形状是否确实对应白色标注，辅助图描述是否只在多图版出现。
只要有部分不支持或不确定，该句判false；不要为了保留描述而放宽。
输出JSON：{"keep_optical":[true或false，与候选句逐一对应],"keep_aux":[true或false，与候选句逐一对应],"reasons":["至多一条、20字以内的核对依据或问题"]}。
'''
BANNED = re.compile(r'活动滑坡|形变较大|判断为活动|形变(速率|速度|程度)(较|很|为|大|小)|毫米|厘米|米/|mm/|m/|标注区|品红|掩膜|量化|噪声|散斑|[0-9]|东南|东北|西南|西北|风险|导致|诱发|河道|道路|沟谷|土质|土体|无植被|明显|显著|褶皱')


def now():
    return datetime.now().isoformat(timespec='seconds')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def initialize():
    """固定范围与运行前清单；已有计划必须严格复用，禁止重新抽取。"""
    RUN.mkdir(exist_ok=True)
    p = RUN / 'manifest.json'
    if p.exists():
        existing = io.load_json(p)
        assert existing['prompt_sha256'] == digest([POLICY, DRAFT, VERIFY]), 'prompts changed; use a separately identified run'
        return existing
    revision.verify_snapshots()
    baseline = {r['id']: r for r in io.read_jsonl(revision.V2_BEFORE)}
    originals = {r['id']: r['original_answer'] for r in io.read_jsonl(revision.ORIGINAL_FULL)}
    rows = io.read_rows()
    for _, r in rows:
        b = baseline[r['id']]
        assert r['answer'] == b['answer'] and r['answer_optical'] == b['answer_optical'], r['id']
    ids = [r['id'] for _, r in rows if baseline[r['id']]['answer'] != originals[r['id']] or baseline[r['id']]['answer_optical'] != originals[r['id']]]
    assert len(ids) == 533
    m = {'created_at': now(), 'target_ids': ids, 'target_count': len(ids), 'unchanged_count': 467,
         'source_counts': dict(Counter(r['source'] for _, r in rows if r['id'] in ids)),
         'before_manifest_sha256': {s: io.sha256_file(spec.DEST / f'{s}.jsonl') for s in spec.SPLITS},
         'before_nonanswer_fingerprints': {r['id']: io.row_fingerprint(r) for _, r in rows},
         'before_records_sha256': io.sha256_file(io.RECORDS), 'before_records_size': io.RECORDS.stat().st_size,
         'previous_revision_state': io.load_state(revision.STATE_REVISION),
         'historical_files': {str(p.relative_to(spec.ROOT)): io.sha256_file(p) for p in [spec.V2/'rules.json', spec.V2/'checks.json', spec.V2/'originals_verified.json', spec.V2/'apply_report.json']},
         'baseline_snapshot_sha256': io.sha256_file(revision.V2_BEFORE),
         'model': MODEL_NAME, 'policy': POLICY, 'draft_prompt': DRAFT, 'verify_prompt': VERIFY,
         'prompt_sha256': digest([POLICY, DRAFT, VERIFY]), 'human_confirmed': False}
    io.atomic_write_json(p, m)
    return m


def paths_for(row, create=True):
    d = spec.V2 / 'views' / row['id']
    isolated_path = RUN / 'target_pixels' / f'{row["id"]}.png'
    if not isolated_path.exists():
        if not create:
            raise FileNotFoundError(isolated_path)
        with Image.open(spec.ROOT/row['images'][-1]) as opened:
            rgb=np.asarray(opened.convert('RGB'))
        mask,valid=answer.load_mask_valid(row)
        shown=np.where((mask & valid)[...,None],rgb,128).astype('uint8')
        isolated_path.parent.mkdir(exist_ok=True)
        Image.fromarray(shown).save(isolated_path)
    ps = [('optical_raw', spec.ROOT / row['images'][-1]), ('label', spec.ROOT / row['mask']),
          ('label_overlay', d/'overlay.png'), ('optical_at_model_size', d/'optical_at_input.png'),
          ('target_original_pixels', isolated_path)]
    for i, p in enumerate(row['images'][:-1]):
        ps += [(f'auxiliary_raw_{i}', spec.ROOT/p), (f'auxiliary_at_model_size_{i}', d/f'aux_{i}_at_input.png')]
    return ps


def parse_json(text):
    """只接受完整对象；解析失败不得算已完成。"""
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('visual model did not return JSON')
    obj = json.loads(text[start:end+1])
    if not isinstance(obj, dict):
        raise ValueError('expected object')
    return obj


def generate_many(model, processor, requests):
    """分批计算但每条样本独立输入、独立输出，不拼接成复核图。"""
    import torch
    texts, images, identities, counts, guides = [], [], [], [], []
    chinese = {'optical_raw':'无覆盖光学原图', 'label':'白色为目标的二值标注',
               'label_overlay':'品红仅用于定位的叠加对照', 'optical_at_model_size':'模型输入尺寸光学图',
               'target_original_pixels':'仅保留目标原始颜色像素的核对图，目标外中灰只是占位背景，不是地表'}
    for paths, prompt in requests:
        local_images, identity = [], []
        for role, path in paths:
            with Image.open(path) as f:
                im = f.convert('RGB')
            local_images.append(im)
            identity.append({'path':str(path.relative_to(spec.ROOT)), 'role':role,
                             'native_size':list(im.size), 'sha256':io.sha256_file(path)})
        modality_names = dict(chinese)
        is_multispectral = 'landslide4sense' in str(paths[0][1])
        for role, _ in paths:
            if role.startswith('auxiliary_'):
                index = int(role.rsplit('_', 1)[1])
                modality = '近红外与双短波红外假彩辅助图（非真实地表颜色）' if is_multispectral else ('InSAR相对灰度辅助图' if index == 0 else 'DEM相对灰度辅助图')
                modality_names[role] = modality + ('，模型输入尺寸' if 'model_size' in role else '，原图')
        labels = '\n'.join(f'第{i+1}张：{modality_names.get(role, role)}' for i,(role,_) in enumerate(paths))
        original=np.asarray(local_images[0]).astype(float)
        target=np.asarray(local_images[1])[:,:,0]>127
        pixels=original[target];lum=pixels.mean(-1)
        guide={'target_rgb_median':np.median(pixels,axis=0).round(1).tolist(), 'target_bright_fraction':round(float((lum>180).mean()),3), 'target_dark_fraction':round(float((lum<100).mean()),3)}
        guidance='原图目标像素的程序核对参考（不是新增观测）：'+json.dumps(guide,ensure_ascii=False)+'。这些统计仅用于避免把背景当目标，颜色仍须看原图；不得把统计数字写进答案。'
        guides.append(guide)
        msg=[{'role':'user','content':[{'type':'image','image':im} for im in local_images]+
              [{'type':'text','text':POLICY+'\n'+labels+'\n'+guidance+'\n'+prompt}]}]
        texts.append(processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False))
        images.extend(local_images);identities.append(identity);counts.append(len(local_images))
    processor.tokenizer.padding_side='left'
    batch=processor(text=texts,images=images,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model.generate(**batch,max_new_tokens=210,do_sample=False)
    results=[];cursor=0
    for index,identity in enumerate(identities):
        text=processor.decode(output[index,batch.input_ids.shape[1]:],skip_special_tokens=True)
        for j,item in enumerate(identity):
            item.update(consumed_by='Xiaomi-Lens-0.generate',returned_at=now(),input_grid_thw=batch.image_grid_thw[cursor+j].tolist())
        cursor+=counts[index]
        # 非JSON响应留存并重试，不能默认为“已查看但无信息”。
        try:
            obj=parse_json(text)
        except (ValueError,json.JSONDecodeError):
            obj=None
        results.append({'raw':text,'parsed':obj,'images':identity,
                        'input_tokens':int(batch.attention_mask[index].sum()),'pixel_reference':guides[index],'completed_at':now()})
    return results


def generate(model,processor,paths,prompt):
    result=generate_many(model,processor,[(paths,prompt)])[0]
    if result['parsed'] is None:
        raise ValueError('visual model did not return complete JSON: '+result['raw'])
    return result


def sentences(obj, key, maximum):
    xs = obj.get(key)
    if not isinstance(xs, list) or any(not isinstance(s, str) for s in xs):
        obj.setdefault('_schema_issues',[]).append(f'invalid {key}; no sentence accepted')
        return []
    return [s.strip().rstrip('。')+'。' for s in xs[:maximum] if s.strip()]


def notes_list(value):
    """模型的说明字段允许字符串或字符串数组，不改变描述内容。"""
    if isinstance(value, str):
        return [value]
    return value if isinstance(value, list) else []


def color_guard(row, texts):
    """仅拦截最明显的目标明暗反转，不根据统计值自动编写外观句。"""
    with Image.open(spec.ROOT / row['images'][-1]) as f:
        rgb = np.asarray(f.convert('RGB')).astype(float)
    m, v = answer.load_mask_valid(row)
    t = m & v
    gray = rgb.mean(-1)
    issues = []
    if t.any() and (v & ~t).any():
        dark_target = gray[t].mean() + 20 < gray[v & ~t].mean()
        bright_fraction = float((gray[t] > 180).mean())
        light_target = gray[t].mean() > 160 and gray[t].mean() > gray[v & ~t].mean() + 20
        dark_fraction = float((gray[t] < 100).mean())
        for i, s in enumerate(texts):
            if (dark_target and bright_fraction < .15 or bright_fraction < .05 and gray[t].mean() < 140) and re.search(r'浅白|浅灰白|白色横带|浅色.*比.*亮', s):
                issues.append(i)
            target_clause = re.split(r'与周围|和周围|周围|背景', s, maxsplit=1)[0]
            if light_target and dark_fraction < .15 and re.search(r'深色|偏暗|暗褐色|深褐色', target_clause):
                issues.append(i)
    return issues


def result_path(rid):
    return RUN / 'samples' / f'{rid}.json'


def review_one(model, processor, row, manifest, draft=None, verified=None):
    paths = paths_for(row)
    draft = draft if draft is not None else generate(model, processor, paths, DRAFT)
    opt = sentences(draft['parsed'], 'optical_sentences', 2)
    aux = sentences(draft['parsed'], 'aux_sentences', 1) if len(row['images']) > 1 else []
    candidate = {'optical_sentences': opt, 'aux_sentences': aux}
    verified = verified if verified is not None else generate(model, processor, paths, VERIFY + json.dumps(candidate, ensure_ascii=False))
    decision = verified['parsed']
    if not isinstance(decision, dict):
        raise ValueError('verification JSON incomplete: '+verified['raw'])
    ko, ka = decision.get('keep_optical'), decision.get('keep_aux')
    # 模型有时按辅助图片数而非候选句数给判断。对应关系不明确时整组删句，
    # 绝不把多余的 true 截成“通过”；这是未确认删除，不是格式自动修复为通过。
    alignment_issues = []
    if not isinstance(ko,list) or len(ko) != len(opt) or any(type(x) is not bool for x in ko):
        ko = [False] * len(opt)
        alignment_issues.append('光学核对条数不对应，未确认句全部删除')
    if not isinstance(ka,list) or len(ka) != len(aux) or any(type(x) is not bool for x in ka):
        ka = [False] * len(aux)
        if aux:
            alignment_issues.append('辅助核对条数不对应，未确认句全部删除')
    rejected = []
    bad_color = color_guard(row, opt)
    for kind, textlist, keeps in [('optical', opt, ko), ('auxiliary', aux, ka)]:
        for i, s in enumerate(textlist):
            if BANNED.search(s) or len(s) > 65 or (kind == 'optical' and any(w in s for w in spec.AUX_SOURCE_WORDS)):
                keeps[i] = False
                rejected.append({'kind': kind, 'sentence': s, 'reason': 'policy or length'})
            if kind == 'optical' and i in bad_color:
                keeps[i] = False
                rejected.append({'kind': kind, 'sentence': s, 'reason': 'target brightness contradicts white appearance'})
    optical = [s for s, k in zip(opt, ko) if k]
    auxiliary = [s for s, k in zip(aux, ka) if k]
    numeric, suffix = answer.answer_parts(row)
    # 不确定时删句；不自动补成通用形容句。先删除辅助句，再删除最长光学句，防止截断。
    while True:
        errors, o, m = answer.check_new_text(numeric, suffix, optical, auxiliary, len(row['images']) > 1)
        if not errors:
            break
        if auxiliary:
            rejected.append({'kind': 'auxiliary', 'sentence': auxiliary.pop(), 'reason': str(errors)})
        elif optical:
            rejected.append({'kind': 'optical', 'sentence': optical.pop(), 'reason': str(errors)})
        else:
            raise ValueError(errors)
    before = {'answer': row['answer'], 'answer_optical': row['answer_optical']}
    after = {'answer': m, 'answer_optical': o}
    state = 'modified' if before != after else 'kept'
    if not optical and not auxiliary:
        state = 'removed_unconfirmed' if before != after else 'kept'
    return {'id': row['id'], 'source': row['source'], 'split': row['split'], 'status': 'reviewed',
            'decision': state, 'before': before, 'after': after, 'optical_sentences': optical,
            'aux_sentences': auxiliary, 'draft': draft, 'verification': verified, 'rejected': rejected,
            'uncertain': notes_list(draft['parsed'].get('uncertain', [])) + notes_list(decision.get('reasons', [])) + alignment_issues,
            'prompt_sha256': manifest['prompt_sha256'], 'model': MODEL_NAME, 'human_confirmed': False,
            'note': 'local visual model reviewed every independent image twice; not Codex ViewImage or human confirmation',
            'completed_at': now()}


def run(limit=0):
    from ..domain_runtime import load, seed_all
    manifest = initialize()
    wanted = set(manifest['target_ids'])
    todo = [r for _, r in io.read_rows() if r['id'] in wanted and not result_path(r['id']).exists()]
    # 已指出的错误优先实际处理，便于先检验方法是否能发现目标混淆。
    priority = ['multimodal_landslide_xiz_005573', 'multimodal_landslide_xiz_001834']
    todo.sort(key=lambda r: priority.index(r['id']) if r['id'] in priority else 2)
    if limit:
        todo = todo[:limit]
    if not todo:
        print('No pending samples', flush=True)
        return
    seed_all(); model, processor = load()
    # 同来源成组，避免混合尺寸的填充；每个样本仍独立图文输入。
    pending=list(todo)
    while pending:
        source=pending[0]['source']
        cap=8 if source=='GDCLD' else 12
        group=[r for r in pending if r['source']==source][:cap]
        group_ids={r['id'] for r in group}
        pending=[r for r in pending if r['id'] not in group_ids]
        tick=time.monotonic()
        drafts=generate_many(model,processor,[(paths_for(row),DRAFT) for row in group])
        candidates=[]
        for row,draft in zip(group,drafts):
            if draft['parsed'] is None:
                draft.update(generate(model,processor,paths_for(row),DRAFT))
            opt=sentences(draft['parsed'],'optical_sentences',2)
            aux=sentences(draft['parsed'],'aux_sentences',1) if len(row['images'])>1 else []
            candidates.append({'optical_sentences':opt,'aux_sentences':aux})
        checks=generate_many(model,processor,[(paths_for(row),VERIFY+json.dumps(c,ensure_ascii=False)) for row,c in zip(group,candidates)])
        # 先落盘真实推理返回，后处理失败时可以从证据继续，不丢弃已做的视觉读取。
        for row,draft,check in zip(group,drafts,checks):
            io.atomic_write_json(RUN/'attempts'/f'{row["id"]}.json', {'draft':draft,'verification':check})
        for row,draft,check in zip(group,drafts,checks):
            try:
                try:
                    result=review_one(model,processor,row,manifest,draft,check)
                except (ValueError,TypeError):
                    # 单条格式/对齐失败再看一次；再次失败保持未完成并报错。
                    retry=generate(model,processor,paths_for(row),VERIFY+json.dumps(candidates[group.index(row)],ensure_ascii=False))
                    result=review_one(model,processor,row,manifest,draft,retry)
                    result['first_verification_attempt']=check
                io.atomic_write_json(result_path(row['id']),result)
                print('REVIEWED',row['id'],result['decision'],round(time.monotonic()-tick,1),flush=True)
            except Exception as e:
                io.append_jsonl(RUN/'errors.jsonl',[{'id':row['id'],'error':repr(e),'at':now(),'draft':draft,'verification':check}])
                raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    if args.apply:
        apply_review()
    elif args.check:
        print(json.dumps(validate_outputs(require_current=True),ensure_ascii=False))
    else:
        run(args.limit)


def completed():
    return {p.stem: io.load_json(p) for p in (RUN/'samples').glob('*.json')}


def validate_outputs(require_current=False):
    """复核范围、逐次独立图片身份、原始输出和正式数据；不覆盖历史报告。"""
    manifest = io.load_json(RUN/'manifest.json')
    outputs = completed()
    wanted = set(manifest['target_ids'])
    assert set(outputs) == wanted, f'review incomplete: {len(outputs)}/{len(wanted)}'
    assert io.sha256_file(revision.V2_BEFORE) == manifest['baseline_snapshot_sha256']
    baseline = {r['id']:r for r in io.read_jsonl(revision.V2_BEFORE)}
    rows = io.read_rows()
    assert len(rows) == 1000 and len({r['id'] for _,r in rows}) == 1000
    source_records = io.load_records()
    errors = []
    for _, row in rows:
        rid = row['id']
        assert io.row_fingerprint(row) == manifest['before_nonanswer_fingerprints'][rid], rid
        if rid not in wanted:
            assert row['answer'] == baseline[rid]['answer'] and row['answer_optical'] == baseline[rid]['answer_optical'], rid
            continue
        rec = outputs[rid]
        assert rec['status'] == 'reviewed' and rec['human_confirmed'] is False, rid
        assert rec['prompt_sha256'] == manifest['prompt_sha256'], rid
        expected = {(role,str(path.relative_to(spec.ROOT))):io.sha256_file(path) for role,path in paths_for(row,create=False)}
        for phase in ['draft','verification']:
            real = {(i['role'],i['path']): i['sha256'] for i in rec[phase]['images']}
            assert real == expected and rec[phase]['raw'] and rec[phase]['completed_at'], (rid,phase)
        candidate = {**row, **rec['after']}
        errors += answer.assert_dual_row(candidate)
        numeric,suffix = answer.answer_parts(row)
        new = candidate['answer'][len(numeric):-len(suffix)]
        assert not BANNED.search(new), (rid,new)
        optical_new=candidate['answer_optical'][len(numeric):-len(suffix)]
        assert not re.search(r'形变|变形',optical_new),rid
        for sentence in rec['aux_sentences']:
            if re.search(r'形变|变形',sentence):
                assert row['source']=='multimodal-landslide-dataset' and 'InSAR' in sentence and '疑似' in sentence and '线索' in sentence,rid
        assert len(candidate['answer']) <= spec.MAX_CHARS and len(candidate['answer_optical']) <= spec.MAX_CHARS
        if require_current:
            assert row['answer'] == rec['after']['answer'] and row['answer_optical'] == rec['after']['answer_optical'], rid
            assert source_records[rid]['answer_multiview'] == row['answer'] and source_records[rid]['answer_optical'] == row['answer_optical'], rid
    assert not errors, errors[:5]
    for p,h in manifest['historical_files'].items():
        assert io.sha256_file(spec.ROOT/p) == h,p
    # 历史 records 前缀必须按字节保留，不能把原始答案或旧观察覆盖。
    with open(io.RECORDS,'rb') as f:
        assert hashlib.sha256(f.read(manifest['before_records_size'])).hexdigest() == manifest['before_records_sha256']
    image_originals=io.load_json(spec.V2/'samples_manifest.json')
    assert io.samples_manifest() == image_originals
    originals_verified=io.load_json(spec.V2/'originals_verified.json')
    for name in ['display_ranges.json','screening.jsonl']:
        assert io.sha256_file(spec.DEST/name) == originals_verified['extra_files'][name]['sha256']
    if require_current:
        summary=io.load_json(spec.DEST/'summary.json')
        assert summary['manifest_sha256'] == {s:io.sha256_file(spec.DEST/f'{s}.jsonl') for s in spec.SPLITS}
    stats={'reviewed':len(outputs),'unchanged_not_reviewed':467,
           'decisions':dict(Counter(r['decision'] for r in outputs.values())),
           'sources':manifest['source_counts'],'auxiliary_sentences':sum(bool(r['aux_sentences']) for r in outputs.values()),
           'human_confirmed':False,'all_files_and_bindings_checked':True,'formal_training_run':False,
           'review_method':MODEL_NAME,'viewimage_tool_reviewed_all':False,
           'codex_spot_reviewed':sum(bool(r.get('codex_spot_review')) for r in outputs.values()),
           'inference_data_contains_groundtruth':True,
           'scope':'description curation only; not free prediction evaluation','checked_at':now()}
    return stats


def record_updates(recs, outputs):
    """纯函数：只返回真正变化的追加记录，便于实测连续修订与重复应用。"""
    append=[]
    for rid, result in outputs.items():
        # 候选再次修订时追加新记录；重复应用相同内容则不膨胀历史。
        previous=recs[rid]
        if (previous.get('answer_optical') != result['after']['answer_optical']
                or previous.get('answer_multiview') != result['after']['answer']
                or previous.get('review_20261112') != str(result_path(rid).relative_to(spec.ROOT))):
            rec=copy.deepcopy(recs[rid])
            rec.update(answer_optical=result['after']['answer_optical'], answer_multiview=result['after']['answer'],
                       review_state='viewed_keep' if result['decision']=='kept' else 'viewed_needs_change',
                       status='verified', view_evidence_complete=True, human_confirmed=False,
                       view_checked_by='local_visual_model', view_model=MODEL_NAME,
                       draft_is_fact=False, needs_recheck=True, suffix_verified=True,
                       images_viewed=result['draft']['images'], review_20261112=str(result_path(rid).relative_to(spec.ROOT)))
            append.append(rec)
    return append


def apply_review():
    """仅写回范围内533条；全部候选和证据通过后才写盘。重复应用保持清单字节相同。"""
    polish_completed()
    apply_spot_proposals()
    stats=validate_outputs()
    outputs=completed()
    recs=io.load_records()
    report_path=RUN/'apply.json'
    append=record_updates(recs,outputs)
    if append:
        io.append_records(append)
    # 三个文件先在内存通过检查；原子替换以文件为单位，若中断可重复执行收敛。
    for split in spec.SPLITS:
        path=spec.DEST/f'{split}.jsonl'
        rows=io.read_jsonl(path)
        updated=[{**r,**outputs[r['id']]['after']} if r['id'] in outputs else r for r in rows]
        assert not answer.check_answers(updated)
        io.write_jsonl_atomic(path,updated)
    summary=io.load_json(spec.DEST/'summary.json')
    summary.update(answer_format='mask_numeric_plus_reviewed_appearance_20261112',
                   manifest_sha256={s:io.sha256_file(spec.DEST/f'{s}.jsonl') for s in spec.SPLITS},
                   description_review={'count':533,'unchanged':467,'report':str(RUN.relative_to(spec.ROOT)),
                                       'model':MODEL_NAME,'human_confirmed':False})
    io.atomic_write_json(spec.DEST/'summary.json',summary)
    stats=validate_outputs(require_current=True)
    stats['manifest_sha256']={s:io.sha256_file(spec.DEST/f'{s}.jsonl') for s in spec.SPLITS}
    io.atomic_write_json(report_path,stats)
    io.atomic_write_json(spec.V2/'current_review.json',{'review_dir':str(RUN.relative_to(spec.ROOT)),
        'report':str(report_path.relative_to(spec.ROOT)),'target_count':533,'untouched_count':467,'human_confirmed':False})
    io.save_state({'phase':'applied_review_20261112','reviewed':533,'untouched':467,
                   'model':MODEL_NAME,'human_confirmed':False,'appearance_accuracy_proven':False,
                   'formal_training_run':False,'independent_evaluation_run':False,
                   'loader_encode_smoke_run':False,'report':str(report_path.relative_to(spec.ROOT))},
                  path=revision.STATE_REVISION)
    print(json.dumps(stats,ensure_ascii=False),flush=True)


def polish_completed():
    """对已看图结果追加保守拦截，只删句，不凭统计值或模板制造新外观。"""
    rows={r['id']:r for _,r in io.read_rows()}
    extra=re.compile(r'隆起|塌陷|厚度|纹理模糊|无清晰|无浅色裸土|目标(?:位于|在)(?:图像|图片).{0,3}(?:左|右|上|下)')
    for rid,result in completed().items():
        if result.get('codex_spot_review'):
            continue
        row=rows[rid]
        old=copy.deepcopy(result['after'])
        opt=list(result['optical_sentences']); aux=list(result['aux_sentences'])
        rejected=[];bad=color_guard(row,opt)
        clean=[]
        for i,s in enumerate(opt):
            if i in bad or BANNED.search(s) or extra.search(s) or re.search(r'形变|变形',s):
                rejected.append({'kind':'optical','sentence':s,'reason':'conservative target appearance or uncertainty guard'})
            else:clean.append(s)
        opt=clean;clean=[]
        for s in aux:
            source_ok=('假彩' in s and 'InSAR' not in s and 'DEM' not in s) if row['source']=='landslide4sense' else ('InSAR' in s or 'DEM' in s)
            relation_ok=bool(re.search(r'上|下|左|右|边缘|内部|中央|中部|内侧|外侧|条|带|斑|块|片|连片',s))
            # 疑似形变线索只允许明确来自InSAR，不能来自假彩或DEM明暗。
            deformation_ok=('形变' not in s and '变形' not in s) or (
                'InSAR' in s and row['source']=='multimodal-landslide-dataset'
                and '疑似' in s and '线索' in s
                and not re.search(r'较强|较弱|强烈|剧烈|强形变|弱形变',s))
            if BANNED.search(s) or extra.search(s) or not source_ok or not relation_ok or not deformation_ok:
                rejected.append({'kind':'auxiliary','sentence':s,'reason':'modality or specific spatial evidence not established'})
            else:clean.append(s)
        aux=clean
        numeric,suffix=answer.answer_parts(row)
        errs,optical,multi=answer.check_new_text(numeric,suffix,opt,aux,len(row['images'])>1)
        assert not errs,(rid,errs)
        result.update(optical_sentences=opt,aux_sentences=aux,after={'answer':multi,'answer_optical':optical})
        result.setdefault('additional_rejections',[]).extend(rejected)
        result['decision']='kept' if result['before']==result['after'] else ('modified' if opt or aux else 'removed_unconfirmed')
        if old!=result['after']:
            result['polished_at']=now()
            io.atomic_write_json(result_path(rid),result)


def apply_spot_proposals():
    """合入本轮实际独立看图的Codex补核，保留被替代的模型候选，不冒充人工。"""
    rows={r['id']:r for _,r in io.read_rows()}
    for p in (RUN/'spot_proposals').glob('*.json'):
        proposal=io.load_json(p);rid=proposal['id'];row=rows[rid]
        expected={(role,str(path.relative_to(spec.ROOT))):io.sha256_file(path)
                  for role,path in paths_for(row,create=False) if role!='target_original_pixels'}
        actual={(i['role'],i['path']):i['sha256'] for i in proposal['evidence']}
        assert actual==expected and proposal['human_confirmed'] is False,rid
        result=io.load_json(result_path(rid))
        if result.get('codex_spot_review') and result['after']==proposal['after']:
            continue
        io.append_jsonl(RUN/'spot_revision_history.jsonl',[{
            'id':rid,'at':now(),'before':result['after'],
            'previous_spot_review':result.get('codex_spot_review'),
            'proposal':proposal}])
        before=result['after']
        for key in ['after','optical_sentences','aux_sentences']:
            result[key]=proposal[key]
        result['decision']='modified'
        result['codex_spot_review']={**proposal,'before_spot_review':before}
        io.atomic_write_json(result_path(rid),result)
