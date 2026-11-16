"""核对三类真实输入：默认行为不变、自定义问题生效、标准标注不泄漏。"""
import json
import torch
from transformers import AutoProcessor
from xiaomi_lens.domain_runtime import BASE,PROMPT,build,configure_processor
from xiaomi_lens.local_demo.service import DATA,OUT
from xiaomi_lens.train.data.domain import jsonlines,NAMES,write_json
from prepare_benchmark500 import hashes

def main():
 torch.set_num_threads(4)
 processor=configure_processor(AutoProcessor.from_pretrained(BASE,local_files_only=True,use_fast=False))
 rows=jsonlines(DATA/'samples.jsonl');checks=[]
 for name in NAMES:
  r=next(r for r in rows if r['source']==name)
  original=build(processor,r)
  explicit=build(processor,dict(r,mask='/not-used',valid='/not-used',answer='不应进入预测'),prompt=PROMPT)
  assert original.keys()==explicit.keys() and all(torch.equal(v,explicit[k]) if isinstance(v,torch.Tensor) else v==explicit[k] for k,v in original.items())
  assert 'labels' not in original and 'mask_targets' not in original
  custom=build(processor,r,prompt='请描述最后一张图像中的滑坡。')
  assert not torch.equal(original['input_ids'],custom['input_ids'])
  checks.append({'source':name,'default_unchanged':True,'custom_question_used':True,'groundtruth_excluded':True})
 # 回归：全有效标签相同绝不应使两张不同的图像被判为重复。
 a={'source_image_sha256':'image-a','source_valid_mask_sha256':'all-valid'}
 b={'source_image_sha256':'image-b','source_valid_mask_sha256':'all-valid'}
 assert not hashes(a)&hashes(b)
 write_json(OUT/'input-check.json',{'passed':True,'sources':checks,'same_validity_is_not_image_duplicate':True})
 print('PASS three real input modes and duplicate regression')
if __name__=='__main__':main()
