"""追加质量复查：固定采用来源与范围，旧答案只读，空白和重复不是自动错误。"""
import json
import re
from collections import Counter
from pathlib import Path

from .common import ROOT, NAMES, read_jsonl, atomic_write_json, write_jsonl_atomic, sha256_file
from . import resume_records as records
from ..description_v2 import answer

REVIEW = 'review/quality_review'
RISK = re.compile(r'锯齿|与周围.{0,12}植被.{0,10}(?:不同|差异|对比)|与周围.{0,12}纹理')


def normalized(sentence):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', sentence)


def current_text(row, record):
    if record is not None:
        return record['after'], record['optical_sentences'], record['aux_sentences']
    numeric = re.match(r'^检测到滑坡区域，.*?占有效图像面积.*?%。', row['answer'])
    if numeric is None:
        raise ValueError(('numeric_prefix_missing', row['id']))
    base = numeric[0]
    def extra(text):
        body, _ = answer.split_original(text)
        assert body.startswith(base)
        return [s+'。' for s in body[len(base):].split('。') if s.strip()]
    optical = extra(row['answer_optical'])
    multi = extra(row['answer'])
    assert multi[:len(optical)] == optical, ('optical_prefix_mismatch', row['id'])
    return {k: row[k] for k in ('answer', 'answer_optical')}, optical, multi[len(optical):]


def reasons_for(row, optical, repeated):
    reasons = []
    if not optical:
        reasons.append('no_optical_appearance')
    if any(normalized(s) in repeated for s in optical):
        reasons.append('frequent_sentence')
    if any(RISK.search(s) for s in optical):
        reasons.append('generic_shape_or_background_claim')
    if row['positive_pixels'] < 256 and any(re.search(r'纹理|斑驳|颗粒|裂纹', s) for s in optical):
        reasons.append('small_target_texture')
    if row['positive_pixels']/row['valid_pixels'] >= .95 and any(re.search(r'周围|背景', s) for s in optical):
        reasons.append('nearly_full_target_background')
    return reasons


def inventory(dest):
    dest = Path(dest); folder = dest/REVIEW
    if (folder/'manifest.json').exists():
        return load(dest)
    m = records.load_manifest(dest)
    rows = read_jsonl(dest/'prepared.jsonl')
    entries = []; counts = Counter()
    for row in rows:
        source, path, record = records.resolve_record(dest, row, m)
        before, optical, auxiliary = current_text(row, record)
        entries.append({'id':row['id'], 'source':row['source'], 'split':row['split'],
                        'previous_provider':source, 'previous_path':str(path.relative_to(dest)) if path else None,
                        'previous_sha256':sha256_file(path) if path else None,
                        'before':before, 'optical_sentences':optical, 'aux_sentences':auxiliary})
        counts.update(set(normalized(s) for s in optical))
    repeated = {s for s, n in counts.items() if n >= 20}
    candidates = []
    repair=dest/'review/failure_repair_20261118/before.json'
    repair_ids=set(json.loads(repair.read_text())['missing_ids']) if repair.exists() else set()
    for row, e in zip(rows, entries):
        reasons = reasons_for(row, e['optical_sentences'], repeated)
        if row['id'] in repair_ids:reasons.append('repaired_failure_visual_followup')
        # Existing contradiction flags remain review triggers even after an earlier pass.
        if e['previous_path']:
            r = json.loads((dest/e['previous_path']).read_text())
            if any('color' in str(x).lower() for x in r.get('additional_rejections', [])):
                reasons.append('previous_color_contradiction')
        if reasons:
            candidates.append({**e, 'reasons':reasons, 'positive_pixels':row['positive_pixels']})
    pilot = []
    for source in NAMES:
        pool = sorted([e for e in candidates if e['source']==source], key=lambda e:e['id'])
        for predicate in [lambda e:'no_optical_appearance' in e['reasons'],
                          lambda e:'frequent_sentence' in e['reasons'],
                          lambda e:e['positive_pixels']<256,
                          lambda e:'generic_shape_or_background_claim' in e['reasons'],
                          lambda e:'previous_color_contradiction' in e['reasons'], lambda e:True]:
            choices = [e for e in pool if e['id'] not in pilot and predicate(e)]
            if not choices: choices = [e for e in pool if e['id'] not in pilot]
            if choices: pilot.append(choices[0]['id'])
    assert len(pilot)==18
    write_jsonl_atomic(folder/'baseline.jsonl', entries)
    write_jsonl_atomic(folder/'candidates.jsonl', candidates)
    meta = {'schema':1, 'selected':len(rows), 'required':len(candidates), 'pilot_ids':pilot,
            'prepared_sha256':sha256_file(dest/'prepared.jsonl'),
            'baseline_sha256':sha256_file(folder/'baseline.jsonl'),
            'candidates_sha256':sha256_file(folder/'candidates.jsonl'),
            'reasons':dict(Counter(r for e in candidates for r in e['reasons'])),
            'repeat_threshold':20, 'human_confirmed':False, 'formal_training_run':False}
    atomic_write_json(folder/'manifest.json', meta)
    return meta


def load(dest):
    dest=Path(dest); folder=dest/REVIEW
    m=json.loads((folder/'manifest.json').read_text())
    for name, key in [('prepared.jsonl','prepared_sha256'),
                      (REVIEW+'/baseline.jsonl','baseline_sha256'),
                      (REVIEW+'/candidates.jsonl','candidates_sha256')]:
        assert sha256_file(dest/name)==m[key], ('quality_input_changed',name)
    return m


def merged_entries(dest):
    """Return the adopted conversation decisions, checking the immutable delivery."""
    summary = json.loads((Path(dest) / 'summary.json').read_text()) if (Path(dest) / 'summary.json').exists() else {}
    receipt = summary.get('description_merge')
    if not receipt:
        return {}
    path = ROOT / 'outputs/description-final/定稿.jsonl'
    assert sha256_file(path) == receipt['final_sha256'], 'adopted delivery changed'
    return {r['id']: r for r in read_jsonl(path)}


def status(dest):
    dest=Path(dest);m=load(dest)
    wanted={r['id'] for r in read_jsonl(dest/REVIEW/'candidates.jsonl')}
    done={p.stem for p in (dest/REVIEW/'samples').glob('*.json')}
    assert done<=wanted
    adopted = wanted.intersection(merged_entries(dest))
    return {'required':len(wanted),'saved':len(done),'already_adopted':len(adopted),'remaining':len(wanted-done-adopted),
            'complete':False,'meaning':'saved is not visually approved or published'}
