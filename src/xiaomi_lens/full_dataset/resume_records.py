"""本地接续的答案来源与审计：旧文件只读，新结果另存，未完成不得发布。"""
import json
import re
from functools import lru_cache
from collections import Counter
from pathlib import Path

from .common import (ROOT, SPLITS, NAMES, atomic_write_json, write_jsonl_atomic,
                     read_jsonl, sha256_file, digest, initialize, pilot_rows)
from ..description_v2 import answer, visual_review as visual
from .captions import EXTRA_POLICY
from .views import GUIDE_VERSION

PROVIDER = 'local_resume'
REVIEW = 'review/local_resume'
DETAIL_PROMPT = '''每句只陈述一项可核对的事实，颜色、形状分别写，不要把可靠颜色与猜测的地物名称合在一句。
每句不超过35个汉字。最多两句光学句和一句辅助句，两字段始终为字符串数组。
可观察但不确定的特征不要写入答案，放在uncertain中。可以留空，但不能用套话替代看图。
不复述已有位置、面积，不写标注、像素、核对过程。不要从纯色背景或放大方格推断目标纹理。
InSAR允许描述目标处或邻近处的局部明暗、形态、对应关系和疑似形变异常线索；
不能从亮暗直接判断形变大小，不写速度、单位、活动性、成因或风险。
光学外观与辅助信息分开读图生成；光学阶段没有任何辅助图片，不得想象辅助信息。'''
AUX_DRAFT = '''本轮只生成辅助补充，optical_sentences必须是空数组[]，不要重写光学外观。
对照标注定位目标，在辅助图中找到可靠局部差异时才写一句，明确它在目标内部、上缘、下缘、左侧、右侧或外侧的哪个位置，及其明暗、形态。
不只写“存在局部差异”，不只列颜色；如没有可靠局部对应关系则aux_sentences留空。
仅返回JSON：{"optical_sentences":[],"aux_sentences":[],"uncertain":[]}。'''
NATIVE_TEXTURE_LIMIT=256
SINGLE_VERIFY='''现在只有一条候选句。逐张对照目标，判断这句话的每个部分是否有图像依据；不确定则false。
只返回JSON对象，supported必须是真正的布尔值true或false，reason用简短中文。
不按图片数输出数组。候选句：'''


def rules_hash():
    return digest([visual.POLICY, visual.DRAFT, visual.VERIFY, EXTRA_POLICY,
                   DETAIL_PROMPT, AUX_DRAFT, GUIDE_VERSION, 'separate_optical_auxiliary_v2',
                   NATIVE_TEXTURE_LIMIT,SINGLE_VERIFY,'reject_unresolved_texture_and_near_full_target_background'])


def manifest_path(dest):
    return Path(dest) / REVIEW / 'manifest.json'


def record_path(dest, rid):
    return Path(dest) / REVIEW / 'samples' / f'{rid}.json'


def load_manifest(dest):
    m = json.loads(manifest_path(dest).read_text())
    assert m['provider'] == PROVIDER
    assert m['prompt_sha256'] == rules_hash()
    assert m['selection_sha256'] == sha256_file(Path(dest) / 'audit/selected.jsonl')
    assert m['training_config'] == json.loads((ROOT/'src/xiaomi_lens/train/configs/domain.json').read_text())
    return m


def activate(dest):
    """只允许一次可核对的来源切换；重复运行复用同一规则，不覆盖历史。"""
    dest = Path(dest)
    build = initialize(dest)
    if manifest_path(dest).exists():
        m = load_manifest(dest)
        assert build['annotation_provider'] == PROVIDER
        return m
    # 锁在入口处取得；禁止与仍在生成或提交的旧任务并行切换。
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = proc.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(a.decode(errors='replace')).name in
               {'complete_full_dataset.py', 'review_full_dataset_session.py'} for a in args if a):
            raise RuntimeError(f'wait for existing dataset task: {proc.parent.name}')
    local = json.loads((dest / 'audit/local_descriptions_at_switch.json').read_text())
    assert local['count'] == 14212
    cloud = {str(p.relative_to(dest)): sha256_file(p)
             for p in sorted((dest / 'review/codex_session/samples').glob('*.json'))}
    assert len(cloud) == 27
    selected = read_jsonl(dest / 'audit/selected.jsonl')
    assert len(selected) == 18987 and sum(r['reuse_text'] for r in selected) == 999
    cloud_ids = [Path(p).stem for p in cloud]
    pilot = [r['id'] for r in pilot_rows([r for r in selected if r['id'] not in cloud_ids])]
    assert len(pilot) == 12
    from ..domain_runtime import BASE, CONFIG
    model_files = {str(p.relative_to(BASE)): sha256_file(p) for p in sorted(BASE.iterdir()) if p.is_file()}
    assert 'config.json' in model_files and 'model.safetensors.index.json' in model_files
    m = {'provider': PROVIDER, 'model': visual.MODEL_NAME, 'base_path': str(BASE),
         'model_files_sha256': model_files, 'prompt_sha256': rules_hash(),
         'policy': visual.POLICY, 'draft_prompt': visual.DRAFT, 'verify_prompt': visual.VERIFY,
         'extra_policy': EXTRA_POLICY, 'detail_prompt': DETAIL_PROMPT, 'auxiliary_draft_prompt': AUX_DRAFT,
         'generation_layout': 'separate_optical_auxiliary_v2',
         'native_texture_limit':NATIVE_TEXTURE_LIMIT,
         'single_sentence_verify_prompt':SINGLE_VERIFY,
         'selection_sha256': sha256_file(dest / 'audit/selected.jsonl'),
         'prepared_sha256': sha256_file(dest / 'prepared.jsonl'),
         'legacy_manifest_sha256': sha256_file(dest / 'review/manifest.json'),
         'legacy_snapshot_sha256': sha256_file(dest / 'audit/local_descriptions_at_switch.json'),
         'codex_records_sha256': cloud, 'codex_ids': cloud_ids, 'pilot_ids': pilot,
         'training_config': CONFIG, 'created_at': visual.now(), 'formal_training_run': False,
         'human_confirmed': False, 'previous_provider': build.get('annotation_provider')}
    atomic_write_json(manifest_path(dest), m)
    build.update(annotation_provider=PROVIDER, complete=False)
    atomic_write_json(dest / 'build.json', build)
    summary = json.loads((dest / 'summary.json').read_text())
    summary.update(annotation_provider=PROVIDER, complete=False)
    atomic_write_json(dest / 'summary.json', summary)
    return m


def check_history(dest, m):
    """两种历史文字均逐文件核对；缺失或变化都报错，不静默改用其他来源。"""
    dest = Path(dest)
    assert sha256_file(dest / 'prepared.jsonl') == m['prepared_sha256']
    assert sha256_file(dest / 'review/manifest.json') == m['legacy_manifest_sha256']
    assert sha256_file(dest / 'audit/local_descriptions_at_switch.json') == m['legacy_snapshot_sha256']
    snapshot = json.loads((dest / 'audit/local_descriptions_at_switch.json').read_text())
    for rel, h in {**snapshot['sha256'], **m['codex_records_sha256']}.items():
        assert sha256_file(dest / rel) == h, ('history_changed', rel)


def audit_record(dest, row, record):
    """检查已保留句，不把历史被删掉的候选句直接认定为当前答案错误。"""
    from .guards import color_guard
    reasons = []
    opt, aux = record.get('optical_sentences', []), record.get('aux_sentences', [])
    base, suffix = answer.split_original(row['answer'])
    errors, optical, multiview = answer.check_new_text(base, suffix, opt, aux, len(row['images']) > 1)
    if errors or record.get('after') != {'answer': multiview, 'answer_optical': optical}:
        reasons.append('answer_format_or_composition')
    if record.get('status') != 'reviewed' or record.get('human_confirmed') is not False:
        reasons.append('invalid_review_status')
    guide = json.loads((Path(dest) / 'review/inputs' / row['id'] / 'ready.json').read_text())
    visible = guide['visible_target_pixels']
    if not opt and visible >= 16:
        reasons.append('no_optical_detail_for_visible_target')
    if any(visual.BANNED.search(s) or len(s) > 36 for s in opt + aux):
        reasons.append('unsupported_or_overlong_sentence')
    if any(any(w in s for w in ('InSAR', 'DEM', '假彩', '高程', '形变')) for s in opt):
        reasons.append('auxiliary_leaked_into_optical')
    if row['source'] == NAMES[1] and any(re.search(r'InSAR|DEM|高程|形变', s) for s in aux):
        reasons.append('wrong_auxiliary_modality')
    _, _, rejected = color_guard(row, opt, aux)
    if rejected or visual.color_guard(row, opt):
        reasons.append('retained_color_or_brightness_conflict')
    if visible < 16 and (opt or aux):
        reasons.append('appearance_claim_on_unresolved_tiny_target')
    return {'id': row['id'], 'source': row['source'], 'split': row['split'],
            'visible_target_pixels': visible, 'needs_review': bool(reasons), 'reasons': reasons,
            'base_only': not opt and not aux, 'previous_sha256': sha256_file(
                Path(dest) / 'review/samples' / f"{row['id']}.json"),
            'check_type': 'structural_and_pixel_contradiction_screen_not_new_visual_review'}


def _audit_job(job):
    dest, row = job
    record = json.loads((Path(dest) / 'review/samples' / f"{row['id']}.json").read_text())
    return audit_record(dest, row, record)


def audit_existing(dest, workers=4):
    """全量筛查旧本地答案；只读旧文件，输出逐条理由与真正待处理清单。"""
    dest = Path(dest); m = load_manifest(dest)
    check_history(dest, m)
    rows = read_jsonl(dest / 'prepared.jsonl')
    existing = [r for r in rows if not r['reuse_text'] and r['id'] not in m['codex_ids']
                and (dest / 'review/samples' / f"{r['id']}.json").exists()]
    path = dest / REVIEW / 'existing_audit.jsonl'
    if path.exists():
        results = read_jsonl(path)
        assert {r['id'] for r in results} == {r['id'] for r in existing}
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = []
            for r in pool.map(_audit_job, [(str(dest), r) for r in existing], chunksize=24):
                results.append(r)
                if len(results) % 500 == 0: print('AUDIT', len(results), len(existing), flush=True)
        write_jsonl_atomic(path, results)
    additional_path=dest/REVIEW/'additional_audit.jsonl'
    if additional_path.exists():
        additional=read_jsonl(additional_path)
    else:
        from PIL import Image
        import numpy as np
        additional=[]
        for i,row in enumerate(existing,1):
            if row['positive_pixels']>=NATIVE_TEXTURE_LIMIT and row['source']!=NAMES[2] and row['positive_pixels']/row['valid_pixels']<.95:continue
            record=json.loads((dest/'review/samples'/f"{row['id']}.json").read_text())
            opt=record['optical_sentences'];reasons=[]
            if row['positive_pixels']<NATIVE_TEXTURE_LIMIT and any(re.search(r'纹理|斑驳|颗粒|裂纹',s) for s in opt):
                reasons.append('texture_unresolved_in_few_native_target_pixels')
            if row['positive_pixels']/row['valid_pixels']>=.95 and any(re.search(r'周围|背景',s) for s in opt):
                reasons.append('unsupported_outside_relation_on_nearly_full_target')
            if row['source']==NAMES[2] and any(re.search(r'灰|黑',s) for s in opt):
                rgb=np.asarray(Image.open(ROOT/row['images'][-1]).convert('RGB')).astype(float)
                target=np.asarray(Image.open(ROOT/row['mask']))>0
                pixels=rgb[target];median=np.median(pixels,axis=0)
                # 只标记潜在矛盾，不能据此自动改成“褐色”。重新读光学图后决定。
                if median[0]>median[2]+12 and (median.max()-median.min())/max(1,median.max())>.25:
                    reasons.append('possible_auxiliary_gray_leakage_into_optical')
            additional.append({'id':row['id'],'needs_review':bool(reasons),'reasons':reasons,
                               'screen_only_not_visual_confirmation':True})
            if len(additional)%500==0:print('ADDITIONAL_AUDIT',len(additional),flush=True)
        write_jsonl_atomic(additional_path,additional)
    flagged = {r['id']: {**r,'reasons':list(r['reasons'])} for r in results if r['needs_review']}
    for item in additional:
        if item['needs_review']:
            if item['id'] not in flagged:flagged[item['id']]={**item,'reasons':[]}
            flagged[item['id']]['reasons'].extend(item['reasons'])
    missing = [r['id'] for r in rows if not r['reuse_text'] and r['id'] not in m['codex_ids']
               and not (dest / 'review/samples' / f"{r['id']}.json").exists()]
    required = sorted(set(flagged) | set(missing) | set(m['pilot_ids']))
    plan = {'existing_checked': len(results), 'codex_preserved': len(m['codex_ids']),
            'reused': sum(r['reuse_text'] for r in rows), 'missing_count': len(missing),
            'flagged_count': len(flagged), 'required_count': len(required), 'required_ids': required,
            'missing_ids': missing, 'flagged_reasons': dict(Counter(
                reason for r in flagged.values() for reason in r['reasons'])),
            'audit_sha256': sha256_file(path),'additional_audit_sha256':sha256_file(additional_path),
            'created_at': visual.now()}
    atomic_write_json(dest / REVIEW / 'work_plan.json', plan)
    work_plan.cache_clear()
    return plan


@lru_cache(maxsize=2)
def work_plan(dest):
    plan = json.loads((Path(dest) / REVIEW / 'work_plan.json').read_text())
    assert plan['audit_sha256'] == sha256_file(Path(dest) / REVIEW / 'existing_audit.jsonl')
    assert plan['additional_audit_sha256']==sha256_file(Path(dest)/REVIEW/'additional_audit.jsonl')
    return plan


def resolve_record(dest, row, m=None):
    """严格选择已指定的来源；待修订项缺失时拒绝用旧答案冒充完成。"""
    dest = Path(dest); m = m or load_manifest(dest)
    if row['reuse_text']:
        return 'reused', None, None
    if row['id'] in m['codex_ids']:
        p = dest / 'review/codex_session/samples' / f"{row['id']}.json"
        assert sha256_file(p) == m['codex_records_sha256'][str(p.relative_to(dest))]
        return 'codex_preserved', p, json.loads(p.read_text())
    p = record_path(dest, row['id'])
    if p.exists():
        record = json.loads(p.read_text())
        assert record['prompt_sha256'] == m['prompt_sha256']
        return 'local_current', p, record
    plan = work_plan(dest)
    if row['id'] in plan['required_ids']:
        raise ValueError(('required_description_missing', row['id']))
    p = dest / 'review/samples' / f"{row['id']}.json"
    return 'local_preserved', p, json.loads(p.read_text())


def progress(dest):
    dest = Path(dest); m = load_manifest(dest); plan = work_plan(dest)
    done = {p.stem for p in (dest / REVIEW / 'samples').glob('*.json')}
    assert done <= set(plan['required_ids']), 'unexpected current-run sample'
    remaining = set(plan['required_ids']) - done
    summary=json.loads((dest/'summary.json').read_text())
    complete=summary.get('complete',False) and not remaining
    report = {'provider': PROVIDER, 'complete': False, 'step': 'generating_descriptions',
              'selected': 18987, 'reused': 999, 'codex_preserved': 27,
              'existing_checked': plan['existing_checked'], 'flagged': plan['flagged_count'],
              'initial_missing': plan['missing_count'], 'required_this_run': plan['required_count'],
              'saved_this_run': len(done & set(plan['required_ids'])), 'remaining': len(remaining),
              'formal_training_run': False, 'updated_at': visual.now()}
    report.update(complete=complete,step='complete' if complete else ('awaiting_final_checks' if not remaining else 'generating_descriptions'))
    atomic_write_json(dest / REVIEW / 'progress.json', report)
    atomic_write_json(dest / 'audit/pipeline_state.json', report)
    return report
