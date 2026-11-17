"""只读取500条历史结果，新增定位展示和坐标，不运行模型。"""
import json
from pathlib import Path
from xiaomi_lens.local_demo.boxes import presentation
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[json.loads(l) for l in (ROOT/'datasets/landslide_benchmark_500/samples.jsonl').open()]
    counts=[]
    for row in rows:
        d=presentation(ROOT/row['images'][-1],ROOT/'outputs/benchmark500/batch'/row['id'],ROOT/'outputs/benchmark500/localization'/row['id'])
        counts.append({'id':row['id'],'valid':sum(b['valid'] for b in d['boxes']),'warnings':d['warnings']})
    (ROOT/'outputs/geospatial-check/localization.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2));print('Prepared',len(counts))
if __name__=='__main__': main()
