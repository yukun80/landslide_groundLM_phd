"""按当前锁定的标注方式分派入口；保留历史文件不等于永久禁用本地模式。"""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
dest=root/'datasets/landslide_lens_full_positive_v1'
if '--dataset-dir' in sys.argv:dest=Path(sys.argv[sys.argv.index('--dataset-dir')+1])
meta=dest/'build.json'
provider=json.loads(meta.read_text()).get('annotation_provider') if meta.exists() else None
if provider=='codex_session' and (len(sys.argv)<2 or sys.argv[1]!='init'):
    raise SystemExit('当前仅允许Codex会话直接看图；本地模型入口已关闭，使用 tools/review_full_dataset_session.py')
if provider=='local_resume' or (len(sys.argv)>1 and sys.argv[1]=='init'):
    from xiaomi_lens.full_dataset.local_resume import main
else:
    from xiaomi_lens.full_dataset.captions import main
if __name__=='__main__':main()
