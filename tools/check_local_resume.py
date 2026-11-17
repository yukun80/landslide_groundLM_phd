"""接续的关键边界：来源优先、缺项拒绝、试跑变更失效和真实重试。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xiaomi_lens.full_dataset import resume_records as records, local_resume
from xiaomi_lens.full_dataset.common import ROOT, atomic_write_json, sha256_file


class ResumeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'outputs')
        self.dest = Path(self.temp.name)
        self.row = {'id':'sample', 'source':'GDCLD', 'reuse_text':False, 'images':['one.png'],
                    'positive_pixels':500,'valid_pixels':10000}
        self.manifest = {'prompt_sha256':'fixed', 'codex_ids':[], 'codex_records_sha256':{}}

    def tearDown(self): self.temp.cleanup()

    def test_missing_required_cannot_fall_back_to_old_answer(self):
        old=self.dest/'review/samples/sample.json';atomic_write_json(old,{'after':{'answer':'old'}})
        with patch.object(records,'work_plan',return_value={'required_ids':['sample']}):
            with self.assertRaises(ValueError): records.resolve_record(self.dest,self.row,self.manifest)

    def test_codex_has_priority_and_is_hash_locked(self):
        p=self.dest/'review/codex_session/samples/sample.json';atomic_write_json(p,{'after':{'answer':'codex'}})
        atomic_write_json(records.record_path(self.dest,'sample'),{'after':{'answer':'local'},'prompt_sha256':'fixed'})
        m={**self.manifest,'codex_ids':['sample'],'codex_records_sha256':{str(p.relative_to(self.dest)):sha256_file(p)}}
        self.assertEqual(records.resolve_record(self.dest,self.row,m)[2]['after']['answer'],'codex')
        atomic_write_json(p,{'after':{'answer':'changed'}})
        with self.assertRaises(AssertionError): records.resolve_record(self.dest,self.row,m)

    def test_new_record_must_match_current_rules(self):
        atomic_write_json(records.record_path(self.dest,'sample'),{'prompt_sha256':'wrong'})
        with self.assertRaises(AssertionError): records.resolve_record(self.dest,self.row,self.manifest)

    def test_repeated_resolution_does_not_rewrite_records(self):
        p=records.record_path(self.dest,'sample');atomic_write_json(p,{'prompt_sha256':'fixed','after':{'answer':'new'}})
        before=p.read_bytes();stat=p.stat().st_mtime_ns
        for _ in range(2):records.resolve_record(self.dest,self.row,self.manifest)
        self.assertEqual(p.read_bytes(),before);self.assertEqual(p.stat().st_mtime_ns,stat)

    def test_pilot_modified_after_check_is_rejected(self):
        p=records.record_path(self.dest,'sample');atomic_write_json(p,{'answer':'first'})
        atomic_write_json(self.dest/records.REVIEW/'pilot_check.json',{
            'passed':True,'prompt_sha256':'fixed','records_sha256':{'sample':sha256_file(p)}})
        m={**self.manifest,'pilot_ids':['sample']}
        local_resume.check_gate(self.dest,m)
        atomic_write_json(p,{'answer':'second'})
        with self.assertRaises(AssertionError):local_resume.check_gate(self.dest,m)

    def test_schema_failure_requires_another_generation(self):
        invalid={'parsed':{'optical_sentences':[],'aux_sentences':'not an array'},'raw':'bad'}
        valid={'parsed':{'optical_sentences':['目标呈暗褐色。'],'aux_sentences':[]},'raw':'real retry'}
        with patch.object(local_resume,'generate_safely',return_value=[valid]) as generate:
            receipt,candidate=local_resume.checked_draft(None,None,self.row,[],invalid,self.dest)
            self.assertEqual(generate.call_count,1)
            self.assertEqual(receipt['raw'],'real retry')
            self.assertEqual(candidate['aux_sentences'],[])
        self.assertEqual(len(list((self.dest/records.REVIEW/'attempts').glob('*.json'))),2)

    def test_persistent_generation_failure_stays_unfinished(self):
        invalid={'parsed':None,'raw':'broken'}
        with patch.object(local_resume,'generate_safely',return_value=[invalid]) as generate:
            with self.assertRaises(ValueError):local_resume.checked_draft(None,None,self.row,[],invalid,self.dest)
            self.assertEqual(generate.call_count,3)
        self.assertFalse(records.record_path(self.dest,'sample').exists())

    def test_optical_pass_rejects_auxiliary_or_tiny_texture_claim(self):
        row={**self.row,'source':'multimodal-landslide-dataset','positive_pixels':24,'images':['insar','dem','rgb']}
        bad={'parsed':{'optical_sentences':['目标内部纹理细碎。'],'aux_sentences':[]},'raw':'unsupported texture'}
        good={'parsed':{'optical_sentences':['目标呈浅灰白色。'],'aux_sentences':[]},'raw':'retry optical only'}
        with patch.object(local_resume,'generate_safely',return_value=[good]) as generate:
            receipt,candidate=local_resume.checked_draft(None,None,row,[],bad,self.dest,prompt='optical only',phase='optical')
            self.assertEqual(generate.call_count,1)
            self.assertEqual(receipt['raw'],'retry optical only')

    def test_full_run_needs_cpu_pilot_encoding(self):
        p=records.record_path(self.dest,'sample');atomic_write_json(p,{'answer':'one'})
        atomic_write_json(self.dest/records.REVIEW/'pilot_check.json',{
            'passed':True,'prompt_sha256':'fixed','records_sha256':{'sample':sha256_file(p)}})
        m={**self.manifest,'pilot_ids':['sample']}
        with self.assertRaises(FileNotFoundError):local_resume.check_gate(self.dest,m,require_encoding=True)

    def test_spurious_false_for_no_candidate_needs_real_retry_and_keeps_raw(self):
        row={**self.row,'positive_pixels':500}
        draft={'raw':'draft','parsed':{'optical_sentences':['目标呈浅灰白色。','目标轮廓不规则。'],'aux_sentences':[]}}
        check={'raw':'unchanged model reply','parsed':{'keep_optical':[True,True],'keep_aux':[False],'reasons':[]}}
        import copy
        with patch.object(local_resume,'generate_safely',side_effect=[[copy.deepcopy(draft)],[copy.deepcopy(check)],[copy.deepcopy(check)]]) as generate:
            result,failures=local_resume.read_phase(None,None,[row],[[]],self.dest)
            self.assertFalse(failures);self.assertEqual(generate.call_count,3)
            receipt=result['sample']['verification']
            self.assertEqual(receipt['raw'],'unchanged model reply')
            self.assertEqual(receipt['parsed']['keep_aux'],[False])
            self.assertEqual(receipt['unused_zero_candidate_fields'],{'keep_aux':[False]})

    def test_wrong_active_candidate_count_is_never_trimmed_to_pass(self):
        row={**self.row,'positive_pixels':500}
        draft={'raw':'draft','parsed':{'optical_sentences':['目标呈浅灰白色。','目标轮廓不规则。'],'aux_sentences':[]}}
        check={'raw':'bad alignment','parsed':{'keep_optical':[True,True,True],'keep_aux':[],'reasons':[]}}
        import copy
        with patch.object(local_resume,'generate_safely',side_effect=[[draft],[copy.deepcopy(check)],[copy.deepcopy(check)],[copy.deepcopy(check)]]) as generate:
            result,failures=local_resume.read_phase(None,None,[row],[[]],self.dest)
            self.assertFalse(result);self.assertEqual(len(failures),1);self.assertEqual(generate.call_count,4)

    def test_explicit_empty_draft_does_not_create_approved_sentences(self):
        row={**self.row,'positive_pixels':500}
        draft={'raw':'explicit empty draft','parsed':{'optical_sentences':[],'aux_sentences':[]}}
        check={'raw':'model returned image-level flags','parsed':{'keep_optical':[True]*7,'keep_aux':[False]*7,'reasons':[]}}
        with patch.object(local_resume,'generate_safely',side_effect=[[draft],[check]]):
            result,failures=local_resume.read_phase(None,None,[row],[[]],self.dest)
            self.assertFalse(failures)
            self.assertEqual(result['sample']['candidate'],{'optical_sentences':[],'aux_sentences':[]})
            self.assertEqual(result['sample']['verification']['candidate_evaluation'],'not_applicable_explicit_empty_draft')
            self.assertEqual(result['sample']['verification']['parsed']['keep_optical'],[True]*7)

    def test_single_sentence_uses_real_boolean_and_keeps_model_receipt(self):
        row={**self.row,'positive_pixels':500}
        draft={'raw':'draft','parsed':{'optical_sentences':['目标呈浅灰白色。'],'aux_sentences':[]}}
        check={'raw':'{"supported":false,"reason":"颜色不支持"}',
               'parsed':{'supported':False,'reason':'颜色不支持'}}
        with patch.object(local_resume,'generate_safely',side_effect=[[draft],[check]]):
            result,failures=local_resume.read_phase(None,None,[row],[[]],self.dest)
            self.assertFalse(failures)
            self.assertFalse(result['sample']['verification']['single_sentence_evaluation']['supported'])
            self.assertEqual(result['sample']['verification']['raw'],check['raw'])

    def test_historical_codex_status_cannot_reset_local_progress(self):
        from xiaomi_lens.full_dataset.session_review import status
        atomic_write_json(self.dest/'build.json',{'annotation_provider':'local_resume'})
        atomic_write_json(self.dest/'review/codex_session/manifest.json',{'provider':'codex_session'})
        p=self.dest/'summary.json';atomic_write_json(p,{'complete':False,'marker':'local'})
        state=self.dest/'audit/pipeline_state.json';atomic_write_json(state,{'marker':'local'})
        before=[p.read_bytes(),state.read_bytes()]
        self.assertTrue(status(self.dest)['historical_only'])
        self.assertEqual([p.read_bytes(),state.read_bytes()],before)


if __name__=='__main__':unittest.main()
