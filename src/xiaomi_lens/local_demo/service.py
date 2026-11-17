"""本地预测服务：共享一个模型，所有请求串行，独立结果和自定义结果分开。"""
import json,hashlib,threading,time,uuid,zipfile,os,fcntl
from pathlib import Path
import numpy as np
from PIL import Image
from ..domain_runtime import load,predict,seed_all,PROMPT,BASE
from ..domain_evaluation import score
from ..train.data.domain import ROOT,jsonlines,NAMES
DATA=ROOT/'datasets/landslide_benchmark_500'
OUT=ROOT/'outputs/benchmark500'
CHECKPOINT=ROOT/'outputs/finetune-20261115/multiview/best.pt'
LOCK=ROOT/'outputs/finetune-20261115/final-models-locked.json'

def write_json(path, value):
 """提交完整文件后再替换，避免页面读到只写了一半的进度或结果。"""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
 temp.replace(path)

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

_verified_model=None

def verify_model():
 global _verified_model
 record=json.loads(LOCK.read_text())
 files=[CHECKPOINT,LOCK]+[BASE/name for name in record['base_model']]
 signature=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns,p.stat().st_ino) for p in files)
 if _verified_model and _verified_model[0]==signature:return dict(_verified_model[1])
 assert digest(CHECKPOINT)==record['models']['multiview']['sha256']
 for name,expected in record['base_model'].items():assert digest(BASE/name)==expected,name
 result={'checkpoint_sha256':record['models']['multiview']['sha256'],'base_files_checked':len(record['base_model']),'best_epoch_number':record['training_completion']['stopping']['best_epoch']+1}
 _verified_model=(signature,result)
 return dict(result)

def validate():
 manifest=json.loads((DATA/'manifest.json').read_text());rows=jsonlines(DATA/'samples.jsonl')
 assert manifest['complete'] and len(rows)==500 and len({r['id'] for r in rows})==500
 for s,h in manifest['old_manifests'].items():assert digest(ROOT/f'datasets/landslide_lens_positive_v1/{s}.jsonl')==h
 verify_model()
 for r in rows:
  for p,h in r['files'].items():assert not (ROOT/p).is_symlink() and digest(ROOT/p)==h,p
  m=np.array(Image.open(ROOT/r['mask']))>0;v=np.array(Image.open(ROOT/r['valid']))>0
  assert int((m&v).sum())==r['positive_pixels']>0
 return rows

class Predictor:
 def __init__(self):self.model=None;self.processor=None;self.lock=threading.Lock()
 def ensure_loaded(self):
  if self.model is None:
   verify_model()
   seed_all();self.model,self.processor=load(CHECKPOINT)
 def run(self,row,folder,prompt=PROMPT):
  with self.lock:
   self.ensure_loaded()
   mask,result=predict(self.model,self.processor,row,folder,prompt=prompt)
   result['prompt']=prompt
   if 'mask' in row:
    result.update(score(mask,np.array(Image.open(ROOT/row['mask']))>0,np.array(Image.open(ROOT/row['valid']))>0))
   write_json(Path(folder)/'result.json',result)
   return result

SERVICE=Predictor()
def bundle(folder):
 folder=Path(folder);archive=folder/'result.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for name in ['result.json','prediction.json','mask.png','overlay.png']:
   if (folder/name).exists():z.write(folder/name,name)
 return str(archive)

def batch():
 OUT.mkdir(parents=True,exist_ok=True)
 pidfile=OUT/'batch-process.json'
 if pidfile.exists():
  oldpid=json.loads(pidfile.read_text())['pid']
  cmdline=Path(f'/proc/{oldpid}/cmdline')
  if oldpid!=os.getpid() and cmdline.exists() and b'run_benchmark500.py' in cmdline.read_bytes():
   raise RuntimeError('已有批量预测正在运行，不重复启动')
 handle=(OUT/'batch.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write_json(OUT/'batch-process.json',{'pid':os.getpid()})
 rows=validate()
 contract={'checkpoint':digest(CHECKPOINT),'samples':digest(DATA/'samples.jsonl'),'prompt':PROMPT}
 path=OUT/'contract.json'
 if path.exists():assert json.loads(path.read_text())==contract,'结果属于不同输入或模型'
 write_json(path,contract)
 # 每来源先跑1条，作为真实冒烟检查，之后完成其他样本。
 first=[next(r for r in rows if r['source']==n) for n in NAMES]
 order=first+[r for r in rows if r['id'] not in {x['id'] for x in first}]
 records=[];failures=[]
 for i,r in enumerate(order):
  folder=OUT/'batch'/r['id']
  try:
   result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else SERVICE.run(r,folder)
   bundle(folder);records.append(dict(result,source=r['source']))
  except Exception as e:
   failures.append({'id':r['id'],'error':repr(e)});write_json(OUT/'failures.json',failures)
  write_json(OUT/'status.json',{'processed':i+1,'total':len(rows),'success':len(records),'failed':len(failures),'state':'running'})
  print(i+1,len(rows),r['id'],flush=True)
 summary={}
 for source in NAMES+['all']:
  part=[r for r in records if source=='all' or r['source']==source]
  summary[source]={'count':len(part),'mean_iou':float(np.mean([r['iou'] for r in part])) if part else None,'mean_dice':float(np.mean([r['dice'] for r in part])) if part else None,'missed_pixels':sum(r['missed_pixels'] for r in part),'extra_pixels':sum(r['extra_pixels'] for r in part),'truncated':sum(r['truncated'] for r in part),'text_region_conflicts':sum(r['text_mask_conflict'] for r in part),'mean_seconds':float(np.mean([r['seconds'] for r in part])) if part else None}
 write_json(OUT/'summary.json',summary);write_json(OUT/'results.json',records);write_json(OUT/'failures.json',failures)
 with zipfile.ZipFile(OUT/'all_results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for row in rows:
   for p in (OUT/'batch'/row['id']).glob('*'):
    if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))
  for name in ['summary.json','results.json','failures.json','contract.json']:z.write(OUT/name,name)
 write_json(OUT/'status.json',{'processed':len(rows),'total':len(rows),'success':len(records),'failed':len(failures),'state':'complete' if not failures else 'completed_with_failures'})
 if failures:raise RuntimeError('存在失败样本，见failures.json')
