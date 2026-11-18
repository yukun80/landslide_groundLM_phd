"""复查名单、逐句判断与失败重试的边界测试，不加载模型。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xiaomi_lens.full_dataset import quality_review as q, quality_generate as g


class QualityChecks(unittest.TestCase):
    def test_missing_optical_is_selected_even_with_auxiliary(self):
        row={'positive_pixels':10,'valid_pixels':100}
        self.assertEqual(q.reasons_for(row,[],set()),['no_optical_appearance'])

    def test_punctuation_does_not_hide_repetition(self):
        self.assertEqual(q.normalized('目标为白色。'),q.normalized('目标 为 白色！'))

    def test_small_texture_and_full_background_are_selected(self):
        self.assertIn('small_target_texture',q.reasons_for({'positive_pixels':10,'valid_pixels':100},['纹理粗糙。'],set()))
        self.assertIn('nearly_full_target_background',q.reasons_for({'positive_pixels':990,'valid_pixels':1000},['与周围背景不同。'],set()))

    def test_unparseable_empty_check_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            response={'parsed':None,'raw':'broken'}
            with patch.object(g,'generate_safely',return_value=[response]) as generate:
                with self.assertRaises(AssertionError):
                    g.receipt(None,None,[],'prompt',Path(t),'sample',
                              lambda r:self.assertIsInstance(r['parsed'],dict))
                self.assertEqual(generate.call_count,3)
                self.assertEqual(len(list((Path(t)/'attempts').glob('*.json'))),3)

    def test_false_sentence_is_never_adopted(self):
        row={'id':'x','images':['a'],'positive_pixels':500,'valid_pixels':1000}
        draft={'parsed':{'optical_sentences':['目标呈白色。'],'aux_sentences':[]},'raw':'draft'}
        denied={'parsed':{'supported':False,'reason':'颜色不符'},'raw':'false'}
        with tempfile.TemporaryDirectory() as t, patch.object(g,'generate_safely',side_effect=[[draft],[denied]]):
            out=g.phase(None,None,row,[],Path(t))
            self.assertEqual(out['kept'],[])
            self.assertEqual(out['checks'][0]['receipt']['raw'],'false')

    def test_explicit_empty_has_real_second_response(self):
        row={'id':'x','images':['a'],'positive_pixels':500,'valid_pixels':1000}
        draft={'parsed':{'optical_sentences':[],'aux_sentences':[]},'raw':'empty draft'}
        checked={'parsed':{'no_candidate':True,'reason':'无候选句'},'raw':'empty check'}
        with tempfile.TemporaryDirectory() as t, patch.object(g,'generate_safely',side_effect=[[draft],[checked]]) as gen:
            out=g.phase(None,None,row,[],Path(t))
            self.assertEqual(gen.call_count,2);self.assertEqual(out['kept'],[])
            self.assertEqual(out['empty_check']['raw'],'empty check')

    def test_old_publish_cannot_bypass_quality_review(self):
        from xiaomi_lens.full_dataset.local_resume import publish
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/q.REVIEW;p.mkdir(parents=True);(p/'manifest.json').write_text('{}')
            with self.assertRaises(RuntimeError):publish(Path(t))

    def test_final_sampling_covers_each_source_and_decision(self):
        from xiaomi_lens.full_dataset.quality_delivery import final_sample_ids
        rows=[{'id':f'{source}-{decision}-{i}','source':source,'quality_decision':decision}
              for source in ['a','b'] for decision in ['retained','rewritten'] for i in range(5)]
        ids=final_sample_ids(rows)
        self.assertEqual(len(ids),12)
        self.assertEqual(ids,final_sample_ids(list(reversed(rows))))

    def test_batched_generation_keeps_each_sentence_receipt(self):
        rows=[{'id':str(i),'images':['a'],'positive_pixels':500,'valid_pixels':1000} for i in range(2)]
        drafts=[{'parsed':{'optical_sentences':['目标呈白色。'],'aux_sentences':[]},'raw':f'draft{i}'} for i in range(2)]
        checks=[{'parsed':{'supported':v},'raw':str(v)} for v in [True,False]]
        with tempfile.TemporaryDirectory() as t, patch.object(g,'generate_safely',side_effect=[drafts,checks]) as gen:
            out,failed=g.phases(None,None,rows,[[('optical_raw',Path('a'))],[('optical_raw',Path('b'))]],Path(t))
            self.assertFalse(failed);self.assertEqual(gen.call_count,2)
            self.assertEqual(out['0']['kept'],['目标呈白色。']);self.assertEqual(out['1']['kept'],[])
            self.assertEqual(out['1']['checks'][0]['receipt']['raw'],'False')

    def test_policy_survives_memory_safe_batch_splitting(self):
        from xiaomi_lens.full_dataset import captions
        requests=[([('label',Path('a'))],'q1'),([('label',Path('b'))],'q2')]
        with patch.object(captions,'_SAFE_CAP',{1:1}), patch.object(captions.visual,'generate_many',return_value=[{'raw':'response'}]) as gen:
            out=captions.generate_safely(None,None,requests,160,policy='isolated')
            self.assertEqual(len(out),2)
            self.assertTrue(all(call.kwargs['policy']=='isolated' for call in gen.call_args_list))

    def test_historical_calls_keep_default_policy(self):
        from xiaomi_lens.full_dataset import captions
        with patch.object(captions.visual,'generate_many',return_value=[{'raw':'response'}]) as gen:
            captions.generate_safely(None,None,[([('label',Path('a'))],'q')])
            self.assertIsNone(gen.call_args.kwargs['policy'])

    def test_presentation_does_not_invent_or_change_color(self):
        self.assertEqual(g.presentation('目标原像素局部图显示为浅黄白色。'),'目标呈浅黄白色。')
        self.assertEqual(g.presentation('目标像素呈现深褐色。'),'目标呈深褐色。')
        self.assertEqual(g.presentation('目标为白色。'),'目标为白色。')
        self.assertEqual(g.presentation('无法确认目标颜色。'),'无法确认目标颜色。')


if __name__=='__main__':unittest.main()
