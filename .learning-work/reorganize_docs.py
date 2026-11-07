from pathlib import Path
import ast,hashlib,json,os,re
W=Path.cwd();R=W/'Xiaomi-Lens-0-raw';D=W/'docs/xiaomi-lens-0';T=W/'.learning-work'
oldnames=['学习指南.md','代码树与文件说明.md','函数阅读索引.md','配置与示例详解.md','学习版检查结果.md']
old={n:(R/n).read_text() for n in oldnames}
manifest=json.loads((T/'original-sha256.json').read_text())
def digest(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def canonical(p):
 s=p.read_text()
 if p.suffix=='.py':return ast.dump(ast.parse(s))
 if p.suffix=='.sh':return '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
 if p.suffix=='.md':return s.split('\n\n',1)[1] # original README body
 return s
baseline={}
for n in manifest:
 p=R/n
 baseline[n]={'sha256':digest(p.read_bytes())}
 if p.suffix in ('.py','.sh','.md'):baseline[n]['content_sha256']=digest(canonical(p))
evidence={p.relative_to(T).as_posix():digest(p.read_bytes()) for p in T.rglob('*') if p.is_file() and 'node_modules' not in p.parts and 'original' not in p.relative_to(T).parts and p.suffix in ('.json','.png','.pt','.log')}
(T/'docs-migration-baseline.json').write_text(json.dumps({'files':baseline,'historical_evidence':evidence},indent=2,ensure_ascii=False))
# Link destinations for the five replaced documents.
remap={R/'学习指南.md':D/'README.md',R/'代码树与文件说明.md':D/'map/README.md',R/'函数阅读索引.md':D/'functions/README.md',R/'配置与示例详解.md':D/'config/README.md',R/'学习版检查结果.md':D/'checks/README.md'}
def relative(path,base):return os.path.relpath(path,base).replace(os.sep,'/')
def fixlinks(text,dest,source=R):
 def change(m):
  label,target=m.groups()
  if '://' in target or target.startswith('#'):return m.group(0)
  parts=target.split('#',1);p=(source/parts[0]).resolve();p=remap.get(p,p)
  return '['+label+']('+relative(p,dest.parent)+('#'+parts[1] if len(parts)>1 else '')+')'
 return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',change,text)
def write(name,title,body,parent='README.md',raw=False,tail=''):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True)
 nav=f'[返回上级]({relative(D/parent,p.parent)})'
 p.write_text(f'# {title}\n\n{nav}\n\n'+(body if raw else fixlinks(body,p)).strip()+'\n'+('\n'+tail+'\n' if tail else ''))
def section(text,start,end=None):
 a=text.index(start)+len(start);b=text.index(end,a) if end else len(text)
 return text[a:b].strip()
# Learning chapters preserve the existing explanations and examples.
s=old['学习指南.md'];ch=[
('01-overview','先认识整个过程',section(s,'## 1．先用一张路线图认识整个过程','## 2．')),
('02-inference','跟着一次真实使用走',section(s,'## 2．第一遍：跟着一次真实使用走','## 3．')),
('03-model','理解模型怎样连接',section(s,'### 3.1 先看连接处','### 3.2')),
('04-mask','理解区域怎样算出来',section(s,'### 3.2 再看像素计算','## 4．')),
('05-data','先认识训练数据',section(s,'## 4．第三遍：先认识训练数据，再看训练','## 5．')),
('06-training','模型如何学习',section(s,'## 5．接着读模型如何学习','## 6．')),
('07-evaluation','结果怎样评分',section(s,'## 6．看结果怎样评分','## 7．')),
('08-tools','辅助文件与外部代码',section(s,'## 7．最后看辅助文件与外部代码','## 遇到看不懂的代码时')),
('09-reading-tips','遇到看不懂的代码时',section(s,'## 遇到看不懂的代码时'))]
for i,(slug,title,body) in enumerate(ch):
 tail=' · '.join(([f'[上一节]({ch[i-1][0]}.md)'] if i else [])+([f'[下一节]({ch[i+1][0]}.md)'] if i<len(ch)-1 else []))
 write('learn/'+slug+'.md',title,body,tail=tail)
intro=section(s,'# Xiaomi-Lens-0：从零开始读代码','## 先打开这几份材料')
body=intro+'\n\n## 按顺序学习\n\n'+'\n'.join(f'{i+1}. [{title}](learn/{slug}.md)' for i,(slug,title,_) in enumerate(ch))+'\n\n## 需要时再查\n\n- [代码地图](map/README.md)：完整文件清单和用途。\n- [函数速查](functions/README.md)：按源文件查看类和函数。\n- [配置与示例](config/README.md)：字段、样本与启动选项。\n- [检查记录](checks/README.md)：历史验证及本次整理检查。\n- [维护检查单](../maintenance.md)：以后修改时同步更新说明。'
write('README.md','Xiaomi-Lens-0 学习总览',body,parent='../README.md',raw=True)
# File maps, split into coherent groups, retain every original row.
rows=re.findall(r'^\| \[([^\]]+)\]\(([^)]+)\) \| ([1-7]) \| (.*) \|$',old['代码树与文件说明.md'],re.M);assert len(rows)==120
cats={'entry':('入口与安装',[]),'model':('模型核心',[]),'training':('训练组织',[]),'data':('训练数据与示例',[]),'evaluation':('评测',[]),'services':('运行服务',[]),'display':('展示页面与图表',[]),'animations':('案例动画',[]),'thumbnails':('案例缩略图',[])}
for label,path,step,desc in rows:
 if '/samples/thumbs/' in path:k='thumbnails'
 elif '/samples/' in path:k='animations'
 elif path.startswith('docs/'):k='display'
 elif '/integrations/' in path:k='services'
 elif '/train/data/' in path or path.startswith('examples/'):k='data'
 elif '/model/' in path:k='model'
 elif '/train/' in path or path.startswith('scripts/') or path=='train.py':k='training'
 elif path.startswith('eval/'):k='evaluation'
 else:k='entry'
 cats[k][1].append((label,path,step,desc))
for slug,(title,items) in cats.items():
 body='步骤编号对应原学习路线：1 全貌、2 预测、3 模型、4 数据、5 训练、6 评测、7 辅助。\n\n| 文件 | 阅读步骤 | 用途与使用时机 |\n|---|---|---|\n'+'\n'.join(f'| [{a}]({b}) | {c} | {d} |' for a,b,c,d in items)
 write('map/'+slug+'.md',title,body,parent='map/README.md')
body='覆盖原有 **120 个文件**，其中 **34 个 Python 文件**。这里仅展开主要目录；各分组列出全部文件的相对位置与用途，包括图片、动画和字体。\n\n```text\nXiaomi-Lens-0-raw/\n├── infer.py、train.py     预测与训练入口\n├── src/xiaomi_lens/      模型、数据处理和服务\n├── examples/train/      训练示例\n├── scripts/             训练启动脚本\n├── eval/                评测\n├── tools/               模型准备\n└── docs/                原项目展示网站\n```\n\n## 完整文件清单\n\n'+'\n'.join(f'- [{title}]({slug}.md)：{len(items)} 个文件。' for slug,(title,items) in cats.items())+'\n\n新学习文档位于仓库最外层 docs/xiaomi-lens-0，与源码中的展示网站分开。历史日志和输出图片在 .learning-work 中，原始源码备份已按要求删除。'
write('map/README.md','代码地图',body,raw=True)
# Function reference: per original source file, max 20 entries/page; include empty modules.
parts=re.split(r'^## \[([^\]]+)\]\(([^)]+)\)\n',old['函数阅读索引.md'],flags=re.M)
index=[];expected=[]
for i in range(1,len(parts),3):
 label,path,body=parts[i:i+3];items=[l for l in body.splitlines() if l.startswith('- `')];expected.extend((path,l) for l in items)
 slug=path.removesuffix('.py').replace('/','--')
 links=[]
 for j in range(max(1,(len(items)+19)//20)):
  file=slug+(f'-{j+1}' if j else '')+'.md';chunk=items[j*20:(j+1)*20]
  body=f'对应源码：[{path}]({path})。搜索下面的名称可定位原文件。\n\n'+ ('\n'.join(chunk) if chunk else '本文件没有类或函数定义；用途见代码地图和源文件开头的中文说明。')
  if j:body+='\n\n[上一页]('+relative(D/'functions'/(slug+(f'-{j}' if j>1 else '')+'.md'),R)+')'
  if (j+1)*20<len(items):body+='\n\n[下一页]('+relative(D/'functions'/(slug+f'-{j+2}.md'),R)+')'
  write('functions/'+file,label+(f'（第 {j+1} 页）' if len(items)>20 else ''),body,parent='functions/README.md');links.append(f'[第 {j+1} 页]({file})')
 index.append(f'- `{path}`（{len(items)} 项）：'+ ' · '.join(links))
assert len(expected)==280
write('functions/README.md','函数速查','共 **280 个类和函数**。按源文件查找，每页最多 20 项；这里也列出没有函数定义的入口文件。\n\n'+'\n'.join(index),raw=True)
# Config chapters.
s=old['配置与示例详解.md'];configs=[
('training','两份训练设置',section(s,'## 两份训练设置','## 真实示例逐项看')),
('samples','真实示例与区域编码',section(s,'## 真实示例逐项看','### 三份 YAML 清单')),
('datasets','数据清单',section(s,'### 三份 YAML 清单','## 两个训练脚本的选项')),
('launch','训练启动选项',section(s,'## 两个训练脚本的选项','## 安装与服务相关文件')),
('services','安装与服务设置',section(s,'## 安装与服务相关文件','## 展示素材与许可文件')),
('assets','展示素材与许可文件',section(s,'## 展示素材与许可文件'))]
for slug,title,body in configs:write('config/'+slug+'.md',title,body,parent='config/README.md')
write('config/README.md','配置与示例','需要查字段时再进入具体页面。JSON 和 JSONL 不支持普通注释，原文件保持原样；这里集中解释。\n\n'+'\n'.join(f'- [{title}]({slug}.md)' for slug,title,_ in configs),raw=True)
# Historical verification is not a fresh rerun; original backup link removed.
s=old['学习版检查结果.md'];checks=[
('coverage','原学习版交付范围',section(s,'## 已完成的内容','## 注释有没有改变代码')),
('source','历史源码对照',section(s,'## 注释有没有改变代码','## 实际运行的对照')),
('runtime','历史运行对照',section(s,'## 实际运行的对照','## 展示页面检查')),
('browser','历史页面检查',section(s,'## 展示页面检查','## 验证范围')),
('limits','验证范围与限制',section(s,'## 验证范围'))]
for slug,title,body in checks:
 if slug=='coverage':body=body.replace('- 新增学习指南、目录说明、函数索引、配置示例详解和本报告。','- 当时新增五份学习文档；现已按主题拆分迁入 docs/xiaomi-lens-0。')
 if slug=='source':body=body.replace('检查脚本见 [check_static.py](../.learning-work/check_static.py)，结构化结果见','当时的结构化结果见').replace('。\n\n','。\n\n',1);body+='\n\n原始源码备份现已删除。[当前检查脚本](../.learning-work/check_static.py)已改为检查当前文档和本次整理前后的内容摘要，不再重新执行原版源码对照。'
 if slug=='runtime':body+='\n\n原始源码备份已删除，以上是保留的历史结果；原运行脚本仍可检查现有代码，但不能凭这些记录重建或重新运行已删除的原始版本。'
 write('checks/'+slug+'.md',title,'以下记录来自 2026-11-07 的学习版验证，本次文档整理没有重跑模型或页面。\n\n'+body,parent='checks/README.md')
body='历史检查已验证注释前后的一张真实图片预测及代表性小样例一致；本次仅整理文档，不将历史结果算作新运行。\n\n'+'\n'.join(f'- [{title}]({slug}.md)' for slug,title,_ in checks)+'\n- [本次整理检查](migration.md)\n\n## 保留的依据\n\n- [原文件核对清单](../../../.learning-work/original-sha256.json)\n- [历史源码检查结果](../../../.learning-work/static-results.json)\n- [历史运行比较结果](../../../.learning-work/comparison-results.json)\n\n原始源码备份 `.learning-work/original/` 已按要求删除。运行日志、图片与检查记录保留；不再提供指向已删除目录的链接。'
write('checks/README.md','检查记录',body,raw=True)
# Root/docs entries and maintenance.
(W/'README.md').write_text('''# 项目文档入口

本项目包含 Xiaomi-Lens-0 的中文注释代码及初学者学习材料。

- **[从学习总览开始](docs/xiaomi-lens-0/README.md)**：先看哪里、后看哪里。
- [文档目录](docs/README.md)：学习、查阅和检查记录。
- [代码目录](Xiaomi-Lens-0-raw)：保留中文注释的算法代码。
- [原项目中文说明](Xiaomi-Lens-0-raw/README_CN.md)：原有介绍与使用说明。
- [维护检查单](docs/maintenance.md)：修改功能后同步核对文档。

新增说明统一放在最外层 `docs/`，本页只提供入口。协作规则见 [AGENTS.md](AGENTS.md)。
''')
(W/'docs/README.md').write_text('''# 文档目录

[返回项目入口](../README.md)

## Xiaomi-Lens-0

- [学习总览](xiaomi-lens-0/README.md)：按顺序阅读算法。
- [代码地图](xiaomi-lens-0/map/README.md)：逐文件用途。
- [函数速查](xiaomi-lens-0/functions/README.md)：按名称回查。
- [配置与示例](xiaomi-lens-0/config/README.md)：具体字段与选项。
- [检查记录](xiaomi-lens-0/checks/README.md)：验证依据和限制。

## 维护

- [文档维护检查单](maintenance.md)

最外层 docs 是新增文档的统一位置；算法目录内的 docs 是原项目展示网站。
''')
(W/'docs/maintenance.md').write_text('''# 文档维护检查单

[返回文档目录](README.md)

## 讨论与执行

讨论方案时正常讨论，不必把尚未确认的行为写成已实现。方案确认后，执行功能修改时同步更新受影响的说明和入口。

## 每次修改交付前

- [ ] 图片、文字、区域或训练流程改变，对应学习章节已更新。
- [ ] 新增、删除或重命名文件，对应代码地图和函数速查已更新。
- [ ] 配置、示例、输入输出改变，字段说明与用法已更新。
- [ ] 完成的检查写明实际执行范围；历史结果没有被当作本次结果。
- [ ] 从根 README 能找到新说明；页面返回上级，顺序章节能继续往下读。
- [ ] 入口不超过 60 行；专题原则上不超过 120 行；函数页最多 20 项。
- [ ] 运行 `python .learning-work/check_static.py` 检查文档链接、内容覆盖和代码摘要。

该检查脚本记录本次整理时的代码摘要。以后确实修改功能时，它会提示代码变化；应先验证新功能并同步文档，再有依据地更新摘要，不能为了消除报错直接覆盖。历史对照结果不覆盖。

## 定期人工检查

每次较大功能交付或集中维护时，从根入口按一次真实使用过程阅读文档，再与当前代码和实际输出核对。特别检查默认值、阅读顺序、文件数量、外部依赖边界及尚未验证的部分。没有创建自动定时任务。
''')
(W/'AGENTS.md').write_text('''# 项目协作与文档规则

适用于本仓库。项目入口见 [README.md](README.md)，日常检查见 [文档维护检查单](docs/maintenance.md)。

## 沟通和完成标准

- 用简单直白的中文汇报做了什么、结果怎样；最终回复避免术语、实现细节和代码腔。
- 开始工作前明确完成标准，交付前用合适工具实际检查。能运行的运行；可视化改动查看页面并操作流程。
- 发现问题先修复并复查，只有确实需要用户介入时再说明障碍；不把未经验证的草稿当作完成结果。

## 文档位置与层次

- 新增文档统一放在最外层 docs/；根 README 只作简短目录，AGENTS.md 留在根目录。
- 原项目 README 和展示网站保留，必要时更新指向新学习资料的入口。
- 文档采用渐进式披露：先概要和目录，再链接具体专题，避免一份长文承载全部内容。
- 入口页不超过 60 行，专题页原则上不超过 120 行；函数速查每页最多 20 个类或函数，超出则分页。
- 每页提供返回上级的链接；顺序学习章节提供上一节、下一节入口。

## 修改功能必须同步文档

- 讨论阶段正常讨论方案，不将未执行方案描述成已实现功能。
- 方案确认并执行时，同步更新受影响的学习章节、代码地图、函数说明、配置用法和验证范围。
- 新增、删除、重命名文件或调整输入输出、默认值、执行顺序时，检查并修正相关链接和说明。
- 交付前运行文档检查，确认页面可达、链接有效、内容与代码一致；不能只添加规则而省略实际核对。
- 历史检查结论保持原始时间和范围，不把旧记录当作新运行。模型、数据和已保留的验证证据不得因文档整理改写。
- 较大交付或定期人工维护时按 docs/maintenance.md 核对，防止规则遗漏。
''')
# Update only our learning banner in upstream readmes and guide comments in code.
for p in [R/'README.md',R/'README_CN.md',R/'examples/train/README.md',R/'examples/train/README_CN.md']:
 s=p.read_text();head,rest=s.split('\n\n',1);head=fixlinks(head,p,source=p.parent);p.write_text(head+'\n\n'+rest)
for p in list(R.rglob('*.py'))+list(R.rglob('*.sh')):
 s=p.read_text();s=s.replace('完整路线、目录地图和概念解释见项目根目录「学习指南.md」。','完整路线与概念解释见仓库最外层 docs/xiaomi-lens-0/README.md（不是本算法子目录）。')
 s=s.replace('完整解释见根目录「学习指南.md」与「配置与示例详解.md」。','完整解释见仓库最外层 docs/xiaomi-lens-0/README.md 与 docs/xiaomi-lens-0/config/README.md。')
 p.write_text(s)
# Coverage records contain only path/symbol identifiers, not copies of source.
(T/'docs-coverage-expected.json').write_text(json.dumps({'files':sorted(manifest),'symbols':[[p,re.match(r'- `([^`]+)`',l)[1]] for p,l in expected]},ensure_ascii=False,indent=2))
for n in oldnames:(R/n).unlink()
print('Migrated',len(list(D.rglob('*.md'))),'pages; original backup still present pending validation')
