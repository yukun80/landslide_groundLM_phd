"""在无显卡可用的子进程续跑完整结果，确认不会再次预测或改变已有回答/区域。"""
import os,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/benchmark500'
def main():
 status=json.loads((OUT/'status.json').read_text());assert status['state']=='complete' and status['success']==500
 files=[p for p in (OUT/'batch').glob('*/*') if p.name in ['result.json','prediction.json','mask.png','overlay.png']]
 assert len(files)==2000
 before={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 with (OUT/'resume-check.log').open('w') as log:
  subprocess.run([str(ROOT/'scripts/domain.sh'),'tools/run_benchmark500.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert all(hashlib.sha256((OUT/p).read_bytes()).hexdigest()==h for p,h in before.items())
 (OUT/'resume-check.json').write_text(json.dumps({'passed':True,'completed_samples':500,'unchanged_prediction_files':len(files),'gpu_disabled_in_resume':True},indent=2))
 print('PASS reuse all 500 results without GPU; all 2000 prediction files unchanged')
if __name__=='__main__':main()
