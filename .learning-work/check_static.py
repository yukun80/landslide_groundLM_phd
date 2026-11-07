"""Check current learning docs and migration fingerprints; no original backup required.

Run from any directory. Historical static-results.json is never overwritten.
For this migration only, --allow-backup permits pre-deletion validation.
"""
from pathlib import Path
from collections import Counter
import argparse,ast,hashlib,json,os,re,subprocess
from urllib.parse import unquote
W=Path(__file__).resolve().parent.parent;T=W/'.learning-work';R=W/'Xiaomi-Lens-0-raw';D=W/'docs/xiaomi-lens-0'
a=argparse.ArgumentParser();a.add_argument('--allow-backup',action='store_true');args=a.parse_args()
baseline=json.loads((T/'docs-migration-baseline.json').read_text());expected=json.loads((T/'docs-coverage-expected.json').read_text())
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def canonical(p):
 s=p.read_text()
 if p.suffix=='.py':return ast.dump(ast.parse(s))
 if p.suffix=='.sh':return '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
 if p.suffix=='.md':return s.split('\n\n',1)[1]
 return s
for name,record in baseline['files'].items():
 p=R/name;assert p.is_file(),f'missing source {name}'
 if 'content_sha256' in record:assert digest(canonical(p))==record['content_sha256'],f'execution/upstream text changed: {name}'
 else:assert digest(p.read_bytes())==record['sha256'],f'protected file changed: {name}'
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
 if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
for name,h in baseline['historical_evidence'].items():assert digest((T/name).read_bytes())==h,f'historical evidence changed: {name}'
for name in ['学习指南.md','代码树与文件说明.md','函数阅读索引.md','配置与示例详解.md','学习版检查结果.md']:assert not (R/name).exists(),f'old document still present: {name}'
if not args.allow_backup:assert not (T/'original').exists(),'original source backup still exists'
docs=[W/'README.md',W/'AGENTS.md']+sorted((W/'docs').rglob('*.md'))
graph={p.resolve():[] for p in docs};links=0
for p in docs:
 text=p.read_text();limit=60 if p.name=='README.md' else 120
 assert len(text.splitlines())<=limit,f'page too long: {p}'
 if p.parent!=W:assert re.search(r'\[返回[^\]]*\]\(',text),f'missing parent link: {p}'
 for target in re.findall(r'\]\(([^)]+)\)',text):
  if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
  path,_,anchor=target.partition('#');dest=(p.parent/unquote(path)).resolve() if path else p.resolve()
  assert dest.exists(),f'broken link in {p.relative_to(W)}: {target}'
  assert dest!=T/'original',f'link to deleted backup: {p}'
  if dest in graph:graph[p.resolve()].append(dest)
  # Generated pages use page-level links; fail if a future local anchor lacks a matching heading.
  if anchor and dest.suffix=='.md':
   headings=re.findall(r'^#+\s+(.+)$',dest.read_text(),re.M)
   slugs=[re.sub(r'[^\w\-\s]','',h.lower()).replace(' ','-') for h in headings]
   assert unquote(anchor) in slugs,f'broken heading: {p}: {target}'
  links+=1
seen=set();queue=[(W/'README.md').resolve()]
while queue:
 p=queue.pop()
 if p in seen:continue
 seen.add(p);queue.extend(graph[p])
assert set(graph)<=seen,f'unreachable pages: {set(graph)-seen}'
files=[]
for p in (D/'map').glob('*.md'):
 files+=re.findall(r'^\| \[([^\]]+)\]\([^)]*\) \| [1-7] \|',p.read_text(),re.M)
assert Counter(files)==Counter(expected['files']),'file map coverage differs'
symbols=[]
for p in (D/'functions').glob('*.md'):
 if p.name=='README.md':continue
 text=p.read_text();m=re.search(r'对应源码：\[([^\]]+)\]',text);assert m,p
 names=re.findall(r'^- `([^`]+)`：',text,re.M);assert len(names)<=20,p
 symbols += [(m[1],n) for n in names]
assert Counter(symbols)==Counter(map(tuple,expected['symbols'])),'function explanations missing/duplicated'
# Check real source symbols too, so future added functions cannot silently go undocumented.
actual=[]
def walk(node,path,parent=''):
 for n in ast.iter_child_nodes(node):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
   actual.append((path,parent+'.'+n.name if parent else n.name));walk(n,path,n.name if isinstance(n,ast.ClassDef) else parent)
  else:walk(n,path,parent)
for p in R.rglob('*.py'):walk(ast.parse(p.read_text()),p.relative_to(R).as_posix())
assert Counter(actual)==Counter(symbols),'source symbols differ from documentation'
# Inspect updated learning links in original readmes without treating pre-existing upstream links as new ones.
for p in [R/'README.md',R/'README_CN.md',R/'examples/train/README.md',R/'examples/train/README_CN.md']:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text().split('\n\n',1)[0]):assert (p.parent/target).resolve().exists(),(p,target)
for p in list(R.rglob('*.py'))+list(R.rglob('*.sh')):
 text=p.read_text();assert '根目录「学习指南.md」' not in text,p
result={'status':'passed','backup_removed':not (T/'original').exists(),'document_pages':len(docs),'learning_pages':len(list(D.rglob('*.md'))),'local_links_checked':links,'file_explanations':len(files),'class_function_explanations':len(symbols),'all_pages_reachable':True,'execution_content_unchanged':True,'historical_evidence_unchanged':True,'historical_model_tests_rerun':False}
(T/'docs-check-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
