"""混合来源的最终核对；保存逐样本采用记录，不篡改历史查看声明。"""
import json
from pathlib import Path
from collections import Counter
from .common import ROOT, atomic_write_json, read_jsonl, sha256_file
from . import resume_records as records
from .views import view_paths
from ..description_v2 import answer


def validate_records(dest, rows, manifest):
    adopted = read_jsonl(dest / records.REVIEW / 'adopted.jsonl')
    assert len(adopted) == len(rows) == 18987
    adopted = {r['id']: r for r in adopted}
    assert len(adopted) == len(rows)
    prepared = {r['id']: r for r in read_jsonl(dest / 'prepared.jsonl')}
    legacy = json.loads((dest / 'review/manifest.json').read_text())
    plan = records.work_plan(dest)
    initial_missing=set(plan['missing_ids'])
    assert records.progress(dest)['remaining'] == 0
    audit = {r['id']: r for r in read_jsonl(dest / records.REVIEW / 'existing_audit.jsonl')}
    stats = Counter(); reasons = Counter(); sources = {}; examples = {}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['base_path'], local_files_only=True, use_fast=False)
    for index, row in enumerate(rows, 1):
        source, path, record = records.resolve_record(dest, row, manifest)
        a = adopted[row['id']]
        assert a == {'id': row['id'], 'source': source,
                     'path': str(path.relative_to(dest)) if path else None,
                     'sha256': sha256_file(path) if path else None}
        assert row['description_provider'] == source
        if source == 'reused':
            assert all(row[k] == prepared[row['id']][k] for k in ('answer', 'answer_optical'))
            stats['reused'] += 1
            continue
        assert record['id'] == row['id'] and record['status'] == 'reviewed'
        assert record['human_confirmed'] is False
        assert record['after'] == {k: row[k] for k in ('answer', 'answer_optical')}
        base, suffix = answer.split_original(prepared[row['id']]['answer'])
        errors, optical, multi = answer.check_new_text(base, suffix, record['optical_sentences'],
                record['aux_sentences'], len(row['images']) > 1)
        assert not errors and record['after'] == {'answer': multi, 'answer_optical': optical}
        assert all(len(tokenizer(t, add_special_tokens=False)['input_ids'])+1 <= 160 for t in (optical, multi))
        if source == 'codex_preserved':
            assert record['local_model_used'] is False and record['session_id']
            expected = [{'role': role, 'path': str(p.relative_to(ROOT)), 'sha256': sha256_file(p)}
                        for role, p in view_paths(row, dest, 'review/codex_session/inputs')]
            assert record['views'] == expected
            assert record['viewed_roles'] == [v['role'] for v in expected]
        else:
            assert record['prompt_sha256'] == (manifest if source == 'local_current' else legacy)['prompt_sha256']
            expected = [(role, str(p.relative_to(ROOT)), sha256_file(p)) for role, p in view_paths(row, dest)]
            phases=['draft','verification']
            if source=='local_current' and len(row['images'])>1:
                phases+=['auxiliary_draft','auxiliary_verification']
            for phase in phases:
                receipt = record[phase]
                assert isinstance(receipt['parsed'], dict) and receipt['raw']
                required=expected
                if source=='local_current' and phase in ('draft','verification'):
                    required=[x for x in expected if not x[0].startswith('auxiliary_')]
                assert [(x['role'], x['path'], x['sha256']) for x in receipt['images']] == required
                assert all(x['consumed_by'] == 'Xiaomi-Lens-0.generate' and x['input_grid_thw'] for x in receipt['images'])
            if source=='local_current':
                from ..description_v2 import visual_review as visual
                for key,phase in [('optical_sentences','optical'),('aux_sentences','auxiliary')]:
                    candidate=record[phase+'_candidate']
                    if candidate is None:
                        assert not record[key];continue
                    draft=record['draft' if phase=='optical' else 'auxiliary_draft']
                    verification=record['verification' if phase=='optical' else 'auxiliary_verification']
                    originals=visual.sentences(draft['parsed'],key,2 if phase=='optical' else 1)
                    assert all(s in originals for s in candidate[key])
                    flags=verification['parsed'].get('keep_optical' if phase=='optical' else 'keep_aux',[])
                    single=verification.get('single_sentence_evaluation')
                    if single:
                        assert len(candidate[key])==1 and single['sentence']==candidate[key][0] and single['field']==key
                        assert single['supported'] is verification['parsed']['supported'] and type(single['supported']) is bool
                        flags=[single['supported']]
                    if candidate[key]:
                        assert len(flags)==len(candidate[key]) and all(type(x) is bool for x in flags)
                        supported=[s for s,keep in zip(candidate[key],flags) if keep]
                        assert all(s in supported for s in record[key])
                    else:assert not record[key]
            if source == 'local_preserved': assert not audit[row['id']]['needs_review']
            else:
                previous = record.get('previous_path')
                if previous: assert sha256_file(dest / previous) == record['previous_sha256']
                assert record['model_files_manifest_sha256'] == sha256_file(records.manifest_path(dest))
                stats['changed_from_previous'] += record['changed_from_previous']
                stats['new_samples_generated'] += row['id'] in initial_missing
                stats['legacy_revised'] += row['id'] not in initial_missing and record['changed_from_previous']
                stats['legacy_unchanged_after_reread'] += row['id'] not in initial_missing and not record['changed_from_previous']
                reasons[record.get('base_only_reason') or 'has_detail'] += 1
        stats['reviewed'] += 1; stats[source] += 1
        stats['with_optical_appearance'] += bool(record['optical_sentences'])
        stats['with_auxiliary_detail'] += bool(record['aux_sentences'])
        stats['base_only'] += not record['optical_sentences'] and not record['aux_sentences']
        stats['samples_with_notes'] += bool(record['uncertain'])
        c = sources.setdefault(row['source'], Counter()); c[source] += 1
        if source == 'local_current' and record['changed_from_previous'] and row['source'] not in examples:
            examples[row['source']] = {'id': row['id'], 'before': record['before'], 'after': record['after'],
                                      'kind':'new_description' if row['id'] in initial_missing else 'legacy_revision'}
        if index % 200 == 0: print('RECORDS_CHECK', index, len(rows), flush=True)
    assert stats['reused'] == 999 and stats['codex_preserved'] == 27
    assert stats['local_current'] == plan['required_count']
    assert stats['new_samples_generated']==plan['missing_count']
    assert all(sha256_file(Path(manifest['base_path']) / p) == h
               for p, h in manifest['model_files_sha256'].items())
    atomic_write_json(dest / records.REVIEW / 'validation.json', {
        'passed': True, 'counts': dict(stats), 'by_source': {s: dict(v) for s,v in sources.items()},
        'base_only_reasons_this_run': dict(reasons), 'examples': examples,
        'history_unchanged': True, 'human_confirmed': False, 'formal_training_run': False})
    return stats
