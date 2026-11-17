"""验证来源隔离、真实长度、失败不计完成、重复保存及纯CPU图片准备。"""
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from xiaomi_lens.full_dataset.common import *
from xiaomi_lens.full_dataset import session_review as session
from xiaomi_lens.full_dataset.views import images_for
from xiaomi_lens.description_v2.io import atomic_write_json


def main():
    build=initialize(DEFAULT);assert build['annotation_provider']==session.PROVIDER and not build['complete']
    manifest=session.activate(DEFAULT);assert manifest['target_count']==17988 and manifest['reused_count']==999
    pilot_batch=digest(manifest['pilot_ids'])[:16]
    batch=json.loads((DEFAULT/session.REVIEW/'batches'/f'{pilot_batch}.json').read_text())
    assert len(batch['samples'])==6
    submission=DEFAULT/session.REVIEW/'submissions/pilot.json';entries=json.loads(submission.read_text())
    before={r['id']:sha256_file(session.record_path(DEFAULT,r['id'])) for r in entries}
    session.commit(DEFAULT,submission,__import__('os').environ['CODEX_THREAD_ID'],revise=True)
    assert all(sha256_file(session.record_path(DEFAULT,rid))==h for rid,h in before.items())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/Xiaomi-Lens-0',local_files_only=True,use_fast=False)
    invalid=copy.deepcopy(entries[0]);invalid['viewed_roles']=invalid['viewed_roles'][:-1]
    try:session.validate_entry(invalid,batch['samples'][0],tokenizer)
    except AssertionError:pass
    else:raise AssertionError('missing image was accepted')
    invalid=copy.deepcopy(entries[1]);invalid['optical_sentences']=['目标纹理。'*100]
    try:session.validate_entry(invalid,batch['samples'][1],tokenizer)
    except AssertionError:pass
    else:raise AssertionError('overlong answer accepted')
    try:session.publish(DEFAULT)
    except AssertionError:pass
    else:raise AssertionError('local history filled missing Codex records')
    assert not (DEFAULT/'train.jsonl').exists(),'incomplete session was published'
    blocked=subprocess.run([str(ROOT/'scripts/domain.sh'),'tools/describe_full_dataset.py','run'],capture_output=True,text=True)
    assert blocked.returncode!=0 and '已关闭' in blocked.stderr
    waiting=subprocess.run([str(ROOT/'scripts/domain.sh'),'tools/complete_full_dataset.py'],capture_output=True,text=True)
    assert waiting.returncode==0 and '不会启动本地模型' in waiting.stdout
    from xiaomi_lens.domain_runtime import images_for as training_images
    prepared={r['id']:r for r in read_jsonl(DEFAULT/'prepared.jsonl')}
    for rid in manifest['pilot_ids']:
        a=images_for(prepared[rid]);b=training_images(prepared[rid])
        assert len(a)==len(b) and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(a,b))
    # 样本并未改变；此文件只与实际固定的三套来源和批次身份关联。
    history=json.loads((DEFAULT/'audit/local_descriptions_at_switch.json').read_text())
    assert all(sha256_file(DEFAULT/p)==h for p,h in history['sha256'].items())
    report={'passed':True,'pilot_records':6,'duplicate_commit_preserved':True,'missing_view_rejected':True,
        'overlong_answer_rejected':True,'local_history_cannot_publish':True,'local_generation_disabled':True,
        'no_automatic_online_or_local_worker':True,'cpu_views_equal_training_views':True,'local_history_unchanged':history['count']}
    atomic_write_json(DEFAULT/'audit/codex_session_check.json',report);print('CODEX_SESSION_PASS',report,flush=True)


if __name__=='__main__':main()
