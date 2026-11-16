"""固定每来源10条图文复核页；仅生成，不自动标为已查看。"""
from xiaomi_lens.train.run_review import prepare_review
from xiaomi_lens.train.data.domain import jsonlines,NAMES
from xiaomi_lens.local_demo.service import DATA,OUT
if __name__=='__main__':
 rows=jsonlines(DATA/'samples.jsonl')
 chosen=[r for n in NAMES for r in [x for x in rows if x['source']==n][:10]]
 assert all((OUT/'batch'/r['id']/'result.json').exists() for r in chosen),'固定样本尚未预测完毕'
 prepare_review(chosen,OUT/'batch')
 print('Prepared 30 visual comparisons; awaiting actual viewing')
