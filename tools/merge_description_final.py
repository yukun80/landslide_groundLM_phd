"""Merge the verified 1,314-row delivery into a local, review-pending dataset."""
import csv
import fcntl
import json
import os
from collections import Counter
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path

from xiaomi_lens.full_dataset.common import (
    DEFAULT, ROOT, SPLITS, atomic_write_json, read_jsonl, sha256_file, write_jsonl_atomic,
)
from xiaomi_lens.full_dataset.quality_review import load as check_baseline

DATA = DEFAULT
FINAL = ROOT / 'outputs/description-final/定稿.jsonl'
AUDIT = ROOT / 'outputs/description-audit/逐图清单.csv'
VERSION = DATA / 'review/merged_descriptions'
KEYS = ('answer', 'answer_optical')


def prepare():
    check_baseline(DATA)
    baseline_rows = read_jsonl(DATA / 'review/quality_review/baseline.jsonl')
    baseline = {r['id']: r for r in baseline_rows}
    final_rows = read_jsonl(FINAL)
    final = {r['id']: r for r in final_rows}
    with AUDIT.open(encoding='utf-8-sig') as f:
        decisions = {r['图像编号']: r['结论'] for r in csv.DictReader(f)}
    assert set(final) == {i for i, decision in decisions.items() if decision == '建议修改'}
    assert len(final) == len(final_rows) == 1314
    assert sum(v == '证据不足待确认' for v in decisions.values()) == 12392
    rows = read_jsonl(DATA / 'prepared.jsonl')
    assert set(baseline) == set(decisions) == {r['id'] for r in rows}
    assert len(baseline) == len(baseline_rows) == len(rows) == 18987
    local = DATA / 'review/local_resume'
    saved = {p.stem for p in (local/'samples').glob('*.json')}
    prescribed = set(json.loads((local/'work_plan.json').read_text())['required_ids'])
    codex = set(json.loads((local/'manifest.json').read_text())['codex_ids'])
    changed = 0
    for r in rows:
        i = r['id']
        b = baseline[i]
        if r['reuse_text']:
            provider = 'reused'
        elif i in codex:
            provider = 'codex_preserved'
        elif i in saved:
            provider = 'local_current'
        else:
            assert i not in prescribed, f'Missing required revision: {i}'
            provider = 'local_preserved'
        assert provider == b['previous_provider'], f'Current description source changed: {i}'
        old = b['before']
        if b['previous_path']:
            path = DATA / b['previous_path']
            assert sha256_file(path) == b['previous_sha256'] and json.loads(path.read_text())['after'] == old, i
        else:
            assert old == {k: r[k] for k in KEYS}, i
        entry = final.get(i)
        if entry:
            assert entry['before'] == old and all(entry[k] == r[k] for k in ('split','source','images','valid','mask','files')), i
        after = {k: entry[k] for k in KEYS} if entry else old
        for key in KEYS:
            assert after[key].partition('。')[0] == old[key].partition('。')[0]
            assert after[key].partition(' <bbox>')[2] == old[key].partition(' <bbox>')[2]
        if entry:
            assert entry['decision'] == ('已修改' if after != old else '保留原文'), i
        changed += after != old
        for path, h in r['files'].items():
            assert sha256_file(ROOT / path) == h, path
        r.update(after)
        r.update(description_provider='conversation_final' if entry else b['previous_provider'],
                 description_status=entry['decision'] if entry else decisions[i])
    assert changed == 1174
    assert Counter(r['split'] for r in rows) == {'train':15347,'val':1512,'test':2128}
    return rows


def read_both(row):
    from xiaomi_lens.description_v2 import select_answer_row
    from xiaomi_lens.domain_runtime import CONFIG, build
    for optical in (False, True):
        batch = build(processor, select_answer_row(row,optical), optical, training=True, device='cpu')
        assert batch['mask_positions'].numel() == 1 and batch['mask_targets'].any() and batch['mask_pixel_valid'].any()
        assert batch['mask_image_index'].item() == (0 if optical else len(row['images'])-1)
        assert 0 < (batch['labels'] != -100).sum().item() <= CONFIG['maximum_answer_tokens']
    return row['id']


def check_reading(rows):
    global processor
    import torch
    from transformers import AutoProcessor
    from xiaomi_lens.domain_runtime import BASE, configure_processor
    torch.set_num_threads(1)
    processor = configure_processor(AutoProcessor.from_pretrained(BASE, local_files_only=True, use_fast=False))
    seen = set()
    with get_context('fork').Pool(12) as pool:
        for i in pool.imap_unordered(read_both, rows, chunksize=4):
            seen.add(i)
            if len(seen) % 1000 == 0:
                print('checked', len(seen), flush=True)
    assert seen == {r['id'] for r in rows}


def merge():
    stage = VERSION if VERSION.exists() else VERSION.with_name('.merged_descriptions.staging')
    stage.mkdir(exist_ok=True)
    rows = prepare()
    hashes = {}
    for split in SPLITS:
        name = split + '.jsonl'
        target = stage / name
        link = DATA / name
        if os.path.lexists(link) and (not link.is_symlink() or link.resolve() != VERSION / name):
            raise RuntimeError(f'Existing active manifest requires reconciliation: {link}')
        expected = [r for r in rows if r['split'] == split]
        if target.exists():
            assert read_jsonl(target) == expected, target
        else:
            if stage == VERSION:
                raise ValueError(f'Active manifest missing: {target}')
            write_jsonl_atomic(target, expected)
        assert read_jsonl(target) == expected
        hashes[split] = sha256_file(target)
    decisions = Counter(r['description_status'] for r in rows)
    report = {'total':len(rows),'changed':decisions['已修改'],'reviewed_unchanged':decisions['保留原文'],
              'awaiting_review':decisions['证据不足待确认'],
              'split_sha256':hashes,'final_sha256':sha256_file(FINAL),'actual_reads':2*len(rows)}
    receipt = stage / 'verified.json'
    if not receipt.exists() or json.loads(receipt.read_text()) != report:
        check_reading(rows)
        atomic_write_json(receipt, report)
    for split in SPLITS:
        link = DATA / (split + '.jsonl')
        if not os.path.lexists(link):
            link.symlink_to(Path('review/merged_descriptions') / link.name)
    if stage != VERSION:
        stage.rename(VERSION)
    for split in SPLITS:
        assert sha256_file(DATA / (split + '.jsonl')) == hashes[split]
    summary_path = DATA / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(complete=False, quality_review='pending', description_merge=report, manifest_sha256=hashes)
    atomic_write_json(summary_path, summary)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with ExitStack() as stack:
        for relative in ('review/generation.lock','review/local_resume/generation.lock'):
            lock = stack.enter_context((DATA / relative).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        merge()
