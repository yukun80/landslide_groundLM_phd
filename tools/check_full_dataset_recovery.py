"""重现实际出错返回，验证重新读图修正格式及持续失败的明确报错。"""
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from xiaomi_lens.full_dataset import captions as c
from xiaomi_lens.full_dataset.common import DEFAULT,ROOT,atomic_write_json


def main():
    path=DEFAULT/'review/errors/multimodal_landslide_sic_000235_draft.json'
    original=json.loads(path.read_text());before=copy.deepcopy(original)
    fixed=copy.deepcopy(original);fixed['parsed']['aux_sentences']=[fixed['parsed']['aux_sentences']]
    row={'id':'recovery_test','images':['a','b','c'],'source':'multimodal-landslide-dataset'}
    with tempfile.TemporaryDirectory(prefix='caption-recovery-') as tmp:
        with patch.object(c,'generate_safely',return_value=[fixed]) as generate:
            recovered,candidate=c.recover_draft(None,None,row,[],original,Path(tmp))
            assert generate.call_count==1 and '字符串数组' in generate.call_args.args[2][0][1]
            assert candidate['aux_sentences'] and recovered['format_recovery']['attempt']==1
            assert original==before,'original returned evidence was changed'
        with patch.object(c,'generate_safely',return_value=[original]) as generate:
            try:c.recover_draft(None,None,row,[],original,Path(tmp))
            except ValueError:pass
            else:raise AssertionError('invalid response was marked complete')
            assert generate.call_count==3
        assert len(list((Path(tmp)/'review/errors').glob('*.json')))>=4
    atomic_write_json(ROOT/'outputs/full-dataset-recovery-logic-check.json',{'passed':True,
        'actual_failed_response_reproduced':True,'original_evidence_preserved':True,
        'new_model_call_required':True,'persistent_failure_remains_incomplete':True})
    print('RECOVERY_LOGIC_PASS',flush=True)


if __name__=='__main__':main()
