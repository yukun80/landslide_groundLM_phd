"""在指定环境实际检查加载、三类正样本更新、保存恢复与自由预测。"""
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from xiaomi_lens.domain_runtime import load,build,predict,save_checkpoint,seed_all
from xiaomi_lens.train.data.domain import ROOT,NAMES,jsonlines,write_json

folder=ROOT/'outputs/qwen-env-probe';folder.mkdir(parents=True,exist_ok=True)
archived=ROOT/'outputs/history/mixed-samples-v1/prepared-data'
rows=jsonlines(archived/'train.jsonl');chosen=[]
for source in NAMES:
 row=next(r for r in rows if r['source']==source and r['positive_pixels']>0).copy()
 for key in ['mask','valid']:
  row[key]=row[key].replace('datasets/landslide_lens_v1',str(archived.relative_to(ROOT)))
 row['images']=[p.replace('datasets/landslide_lens_v1',str(archived.relative_to(ROOT))) for p in row['images']]
 assert (np.asarray(Image.open(ROOT/row['mask']))>0).any()
 chosen.append(row)


def run_training_probe():
 seed_all();model,processor=load(training=True);model.train();model.config.lm_loss_weight=1.0
 parameters=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(parameters,lr=1e-5)
 losses=[];torch.cuda.reset_peak_memory_stats()
 for row in chosen:
  batch=build(processor,row,training=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**batch,use_cache=False);loss=out.loss
  assert torch.isfinite(loss)
  (loss/len(chosen)).backward();losses.append(float(loss.detach()))
  print('BACKWARD_OK',row['source'],losses[-1],flush=True)
 torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
 model.eval();batch=build(processor,chosen[-1],training=True)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):expected=model(**batch,use_cache=False).pred_masks.float().cpu()
 save_checkpoint(folder/'probe.pt',model,optimizer,0,1,False)
 write_json(folder/'training.json',{'positive_samples':[r['id'] for r in chosen],'losses':losses,'peak_gb':torch.cuda.max_memory_allocated()/1024**3,'scope':'compatibility probe only; this checkpoint will not initialize formal training'})
 return expected

expected=run_training_probe();gc.collect();torch.cuda.empty_cache()
model,processor=load(folder/'probe.pt');batch=build(processor,chosen[-1],training=True)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(**batch,use_cache=False).pred_masks.float().cpu()
assert torch.allclose(expected,actual,rtol=1e-4,atol=1e-5),(expected-actual).abs().max().item()
_,result=predict(model,processor,chosen[-1],folder/'prediction')
write_json(folder/'passed.json',{'model_load':True,'three_source_backward':True,'finite_gradients':True,'optimizer_update':True,'saved_mask_logits_match':True,'free_prediction':True,'python':__import__('sys').executable,'raw_answer':result['raw_answer']})
print('QWEN_ENVIRONMENT_PASSED',flush=True)
