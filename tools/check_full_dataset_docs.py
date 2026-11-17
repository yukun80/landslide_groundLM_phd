"""独立核对本轮说明可达、链接有效及原版源码内容不变，不覆盖旧检查记录。"""
import json
import re
from pathlib import Path

from xiaomi_lens.full_dataset.common import ROOT,atomic_write_json,sha256_file


def main():
    pages=[ROOT/'README.md',ROOT/'AGENTS.md',*sorted((ROOT/'docs').rglob('*.md'))]
    graph={p.resolve():[] for p in pages};links=0
    for p in pages:
        text=p.read_text();limit=60 if p.name=='README.md' else 120
        assert len(text.splitlines())<=limit,('long_document',p)
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',link):continue
            destination=(p.parent/link.split('#')[0]).resolve()
            assert destination.exists(),('broken_link',p,link)
            if destination in graph:graph[p.resolve()].append(destination)
            links+=1
    seen=set();todo=[(ROOT/'README.md').resolve()]
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p);todo.extend(graph[p])
    new=list((ROOT/'docs/domain-finetuning/full-dataset').glob('*.md'))
    assert all(p.resolve() in seen for p in new),'new page unreachable from root'
    assert all('[返回' in p.read_text() for p in new)
    original=json.loads((ROOT/'outputs/original_code_hashes.json').read_text())
    assert all(sha256_file(ROOT/p)==h for p,h in original.items())
    report={'passed':True,'pages':len(pages),'local_links':links,'new_pages':len(new),'new_pages_reachable':True,
            'original_source_unchanged':True,'historical_migration_checker_note':'old infer.py AST baseline predates current HEAD; left unchanged'}
    atomic_write_json(ROOT/'outputs/full-dataset-documents-check.json',report)
    print('PASS',report,flush=True)


if __name__=='__main__':main()
