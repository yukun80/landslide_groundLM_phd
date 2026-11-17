"""实际检查12条试跑的两阶段图片身份、模型原始返回及最终句子的来源。"""
import json
from collections import Counter

from xiaomi_lens.full_dataset.common import ROOT,DEFAULT,read_jsonl,sha256_file,atomic_write_json
from xiaomi_lens.full_dataset import resume_records as records
from xiaomi_lens.full_dataset.views import view_paths
from xiaomi_lens.description_v2 import visual_review as visual,answer


def main():
    manifest=records.load_manifest(DEFAULT);ids=set(manifest['pilot_ids'])
    rows=[r for r in read_jsonl(DEFAULT/'prepared.jsonl') if r['id'] in ids]
    assert len(rows)==12 and all(r['split']=='train' for r in rows)
    counts=Counter();hashes={}
    for row in rows:
        r=json.loads(records.record_path(DEFAULT,row['id']).read_text())
        hashes[row['id']]=sha256_file(records.record_path(DEFAULT,row['id']))
        assert r['prompt_sha256']==manifest['prompt_sha256']
        assert r['model_files_manifest_sha256']==sha256_file(records.manifest_path(DEFAULT))
        views=[(role,str(p.relative_to(ROOT)),sha256_file(p)) for role,p in view_paths(row,DEFAULT)]
        for phase,key in [('optical','optical_sentences'),('auxiliary','aux_sentences')]:
            candidate=r[phase+'_candidate']
            if candidate is None:
                assert not r[key];continue
            prefix='' if phase=='optical' else 'auxiliary_'
            expected=[v for v in views if not v[0].startswith('auxiliary_')] if phase=='optical' else views
            for name in ('draft','verification'):
                receipt=r[prefix+name]
                assert receipt['raw'] and isinstance(receipt['parsed'],dict)
                assert [(x['role'],x['path'],x['sha256']) for x in receipt['images']]==expected
                assert all(x['consumed_by']=='Xiaomi-Lens-0.generate' and x['input_grid_thw'] for x in receipt['images'])
                counts['real_primary_receipts']+=1
            originals=visual.sentences(r[prefix+'draft']['parsed'],key,2 if phase=='optical' else 1)
            assert all(s in originals for s in candidate[key])
            verification=r[prefix+'verification']
            if not candidate[key]:assert not r[key];continue
            single=verification.get('single_sentence_evaluation')
            if single:
                assert single['field']==key and single['sentence']==candidate[key][0]
                assert type(single['supported']) is bool and single['supported'] is verification['parsed']['supported']
                flags=[single['supported']]
            else:flags=verification['parsed']['keep_optical' if phase=='optical' else 'keep_aux']
            assert len(flags)==len(candidate[key]) and all(type(x) is bool for x in flags)
            supported=[s for s,keep in zip(candidate[key],flags) if keep]
            assert all(s in supported for s in r[key])
        base,suffix=answer.split_original(row['answer'])
        errors,opt,multi=answer.check_new_text(base,suffix,r['optical_sentences'],r['aux_sentences'],len(row['images'])>1)
        assert not errors and r['after']=={'answer':multi,'answer_optical':opt}
        counts['rows']+=1;counts['with_optical']+=bool(r['optical_sentences']);counts['with_auxiliary']+=bool(r['aux_sentences'])
    report={'passed':True,'counts':dict(counts),'records_sha256':hashes,'scope':'12_training_pilot_records_only',
            'model':'official_initial_Xiaomi_Lens_0','human_confirmed':False,'formal_training_run':False}
    atomic_write_json(DEFAULT/records.REVIEW/'pilot_records_check.json',report)
    print('PILOT_RECORDS_PASS',report,flush=True)


if __name__=='__main__':main()
