"""只读汇总新数据制作进度；文件数量只代表保存进度，不代表描述准确率。"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'datasets/landslide_lens_full_positive_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',type=Path,default=DEFAULT);args=parser.parse_args()
    dest=args.dataset_dir;rows=[json.loads(line) for line in (dest/'audit/selected.jsonl').read_text().splitlines()]
    copied=0;progress=dest/'audit/export_progress.json'
    if (dest/'prepared.jsonl').exists():copied=sum(1 for line in (dest/'prepared.jsonl').open() if line.strip())
    elif progress.exists():copied=json.loads(progress.read_text())['completed']
    else:
        log=ROOT/'outputs/full-dataset-export-resume.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16384));lines=f.read().decode(errors='replace')
            matches=re.findall(r'EXPORT (\d+) (\d+) \d+',lines)
            if matches:copied=int(matches[-1][0])
    provider=json.loads((dest/'build.json').read_text()).get('annotation_provider','local_model')
    if provider=='local_resume':
        summary=json.loads((dest/'summary.json').read_text())
        state=dest/'review/local_resume/progress.json'
        report={'当前看图方式':'本地小米官方原始模型','暂选样本':len(rows),
                '已复制并制作图片':copied,'已有文字保留':999,'Codex文字保留':27,
                '全部完成':summary.get('complete',False),
                '说明':'本地生成文字，不是微调；生成与再次核对不等于人工确认'}
        if state.exists(): report['本轮进度']=json.loads(state.read_text())
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    folder=dest/('review/codex_session/samples' if provider=='codex_session' else 'review/samples');described={p[:-5] for p in os.listdir(folder) if p.endswith('.json')} if folder.exists() else set()
    report={'当前看图方式':provider,'暂选样本':len(rows),'已确认复制并制作图片':copied,
            '已有文字可复用':sum(r['reuse_text'] for r in rows),
            '已保存新增文字':sum(not r['reuse_text'] and r['id'] in described for r in rows),
            '尚缺新增文字':sum(not r['reuse_text'] and r['id'] not in described for r in rows),
            '全部完成':False,'说明':'复制进度按最近一次保存的数量汇报；文字保存进度不代表已经获得人工确认'}
    for path,key in [('audit/pipeline_state.json','当前阶段'),('audit/protected_quality_conflicts.json','待决定的图片')]:
        if (dest/path).exists():report[key]=json.loads((dest/path).read_text())
    if (dest/'summary.json').exists() and (dest/'audit/final_check.json').exists():
        report['全部完成']=json.loads((dest/'summary.json').read_text())['complete'] and json.loads((dest/'audit/final_check.json').read_text())['passed']
    if '待决定的图片' in report:
        quality=report['待决定的图片'];report['待决定的图片']={'已解决':quality['resolved'],'编号':[r['id'] for r in quality['samples']]}
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
