from pathlib import Path
import json,ast
R=Path('Xiaomi-Lens-0-raw');manifest=json.load(open('.learning-work/original-sha256.json'));files=json.load(open('.learning-work/files.json'))
M={p:(v[0],v[1]) for p,v in files.items()}
extra={
'.gitignore':('7','版本管理忽略规则；管理缓存、环境和输出文件时查看。'),
'README.md':('1','原项目英文总说明；查完整安装命令与发布结果时查看。'),
'README_CN.md':('1','原项目中文总说明；先看方法介绍，再跟学习指南进入代码。'),
'pyproject.toml':('7','安装与插件登记设置；了解软件包如何被导入和服务如何发现模型时查看。'),
'requirements.txt':('7','普通推理所需依赖清单；准备环境时查看，中文逐项说明已加在原文件。'),
'scripts/train_stage1.sh':('5','第一阶段启动配方；只训练区域解码器与特殊符号行。'),
'scripts/train_stage2.sh':('5','第二阶段启动配方；从第一阶段继续学习全部参数。'),
'eval/run.sh':('6','评测启动入口；选择计算方式、设备、数据与结果目录。'),
'examples/train/README.md':('4','训练数据格式英文说明；理解示例、配置与启动方法。'),
'examples/train/README_CN.md':('4','训练数据格式中文说明；配合真实样本逐字段阅读。'),
'examples/train/stage1.yaml':('4','人造数据第一阶段清单；生成 toy 数据后使用。'),
'examples/train/stage2.yaml':('4','人造数据第二阶段清单；评估复用训练前 16 条，仅供流程检查。'),
'examples/train/sample/sample.yaml':('4','附带真实样本的清单；解析同目录样本和 images 中图片。'),
'examples/train/sample/sample.jsonl':('4','两条真实图文训练示例；参考答案与区域按顺序对应，字段详解见配套文档。'),
'examples/train/sample/images/COCO_train2014_000000580562.jpg':('4','灰色自卸卡车示例图片，与 sample.jsonl 第二条对应；查看真实训练输入时用。'),
'examples/train/sample/images/COCO_train2014_000000581739.jpg':('4','右侧花瓶示例图片，与 sample.jsonl 第一条对应；本次也用于预测前后对照。'),
'src/xiaomi_lens/train/configs/zero1.json':('5','第一阶段 DeepSpeed 存储分配设置，逐字段解释见配置与示例详解。'),
'src/xiaomi_lens/train/configs/zero2.json':('5','第二阶段 DeepSpeed 存储分配设置，逐字段解释见配置与示例详解。'),
'src/xiaomi_lens/integrations/vllm/run_vllm.sh':('7','启动文字生成服务；与区域服务配合使用。'),
'src/xiaomi_lens/integrations/vllm/run_vllm_mask.sh':('7','启动区域解码服务；登记插件并保留区域解码器权重。'),
'docs/index.html':('7','项目展示页，含外观、内容与交互；只播放已有案例，不实时预测。'),
'docs/.nojekyll':('7','空的静态网站部署标记；不参与模型计算，保持原样。'),
'docs/assets/fonts/OFL.txt':('7','随字体提供的许可原文；了解字体使用条件时查看，保持原样。'),
'docs/assets/video/overview_poster.jpg':('1','总览视频封面；页面在视频播放前展示，非训练样本。'),
'docs/assets/video/xiaomi_lens_0_overview.mp4':('1','项目总览视频；用于直观看能力，非实时运行记录。'),
}
M.update(extra)
figs={'architecture':'模型结构图，理解共享图像特征与区域分支时看','hero_lens':'页面概览装饰图，用于项目展示','lensbench_data_pipeline':'LensBench 数据制作流程图，阅读评测数据介绍时看','lensbench_evaluation':'LensBench 定性对照图，理解发布案例时看','radar_comparison':'发布结果雷达图，比较原项目所报告各项分数','segmentation_data_pipeline':'区域训练数据准备流程图，理解标注来源时看','training_recipe':'两阶段训练配方示意图，阅读训练主线时看','vqa_data_pipeline':'视觉问答数据准备图，理解文字任务数据时看','xiaomi-logo':'项目署名使用的小米标识图'}
subjects={'coco_bottles':'瓶子多实例','coco_oranges':'橙子多实例','coco_people':'人物多实例','coco_surfboards':'冲浪板多实例','countqa_climbing_stones':'攀岩石计数','countqa_leaves':'树叶计数','hrbench_chairs_bicycles':'高分辨率椅子与自行车','hrbench_trailer':'高分辨率拖车','reasonseg_citrus':'根据用途寻找柑橘','reasonseg_gifts':'礼物推理分割','reasonseg_mower':'割草机推理分割','reasonseg_satellite':'卫星影像推理分割','rec_baseball':'棒球指代定位','rec_bear':'熊指代定位','rec_phone':'手机指代定位','rec_spoon':'勺子指代定位','res_cow':'牛指代分割','res_grapefruit':'西柚指代分割','res_kite':'风筝指代分割','res_laptop':'笔记本电脑指代分割','roborefit_book':'书本具身指代','roborefit_cleanser':'清洁剂具身指代','roborefit_cup':'杯子具身指代','roborefit_pepsi':'饮料具身指代','showcase_all_domains':'六类任务合集'}
for p in manifest:
 if p in M:continue
 path=Path(p)
 if p.startswith('docs/assets/figures/'):M[p]=('1',figs[path.stem]+'；静态素材，不参与模型计算。')
 elif p.startswith('docs/assets/fonts/'):M[p]=('7',{'fredoka':'标题','nunito':'正文','jetbrains-mono':'代码与等宽文字'}[path.stem]+'显示字体；查看页面排版时使用，不参与算法。')
 elif p.startswith('docs/assets/samples/'):
  kind='按钮缩略图' if '/thumbs/' in p else '预先保存的动画'
  M[p]=('1',subjects[path.stem]+kind+'；用于浏览展示案例，不是本次验证结果。')
 else:raise RuntimeError(p)
assert set(M)==set(manifest)
intro='# 完整代码树与逐文件说明\n\n先读[学习指南](学习指南.md)。以下覆盖修改前的全部 **120 个文件**，其中 **34 个 Python 文件**；不把本次检查产生的缓存当作算法源码。\n\n目录树先给位置，后面的表给用途和阅读步骤。步骤编号对应学习指南的 1～7 节；素材只需先浏览，研究网页实现留到第 7 步。\n\n## 完整目录树\n\n```text\nXiaomi-Lens-0-raw/\n'
tree={}
for path in sorted(manifest):
 node=tree
 for part in Path(path).parts:node=node.setdefault(part,{})
def render(node,prefix=''):
 out=[]
 for i,(name,children) in enumerate(sorted(node.items(),key=lambda item:(not bool(item[1]),item[0]))):
  last=i==len(node)-1;out.append(prefix+('└── ' if last else '├── ')+name+('/' if children else ''))
  if children:out+=render(children,prefix+('    ' if last else '│   '))
 return out
text=intro+'\n'.join(render(tree))+'\n```\n\n## 每个文件的用途\n\n| 文件 | 第几步看 | 做什么、何时用 |\n|---|---|---|\n'
for p in sorted(M):stage,desc=M[p];text+=f'| [{p}]({p}) | {stage} | {desc} |\n'
text+='\n## 本次新增的学习材料\n\n| 文件 | 用途 |\n|---|---|\n'
for name,desc in [('学习指南.md','从零开始的阅读顺序、流程图、小例子与自测'),('代码树与文件说明.md','本页：全部原文件位置与用途'),('函数阅读索引.md','全部类和函数的用途速查'),('配置与示例详解.md','设置字段、真实样本、启动选项详解'),('学习版检查结果.md','本次实际检查结果与验证范围')]:text+=f'| [{name}]({name}) | {desc} |\n'
text+='\n原始备份、运行日志、对照输出和检查脚本保存在上一级 `.learning-work/` 中，不混入模型源码目录。\n'
(R/'代码树与文件说明.md').write_text(text)
# 由当前文件提取已写入的实际中文说明，避免文档与后续纠正不同步。
text='# 函数阅读索引\n\n按[学习指南](学习指南.md)的顺序学习；本页供遇到名字时查用途。包括类、方法、内部函数共 280 项，点击文件名再搜索相应名称。\n'
count=0
for p in sorted(R.rglob('*.py')):
 rel=p.relative_to(R).as_posix();source=p.read_text();lines=source.splitlines();text+=f'\n## [{rel}]({rel})\n\n'
 def walk(node,parent=''):
  global text,count
  for n in ast.iter_child_nodes(node):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    index=min([n.lineno]+[d.lineno for d in n.decorator_list])-2
    notes=[]
    while index>=0 and lines[index].lstrip().startswith('#'):
     if '【' in lines[index]:notes.append(lines[index].split('】',1)[1])
     index-=1
    assert notes,(rel,n.name)
    name=parent+'.'+n.name if parent else n.name
    text+=f'- `{name}`：'+notes[-1]+'\n';count+=1
    walk(n,n.name if isinstance(n,ast.ClassDef) else parent)
   else:walk(n,parent)
 walk(ast.parse(source))
assert count==280,count
(R/'函数阅读索引.md').write_text(text)
print('Full map',len(M),'files; function index',count)
