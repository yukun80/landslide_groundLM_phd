from pathlib import Path
import ast,re
R=Path('Xiaomi-Lens-0-raw')
fields=dict(line.split('|',1) for line in '''model_init_kwargs|加载模型时额外传入的设置；没有提供则采用空设置。
remove_unused_columns|是否删掉通用训练器不认识的数据字段；这里默认关闭，保留自定义区域信息。
gradient_checkpointing|用反向时重新计算换取较少的中间存储；不是保存训练进度。
learning_rate|每次学习调整参数的步幅；过大或过小都会影响训练，脚本会覆盖默认值。
save_safetensors|用 safetensors 方式保存模型参数。
save_max_shard_size|单份权重文件大小上限，超过时拆成多份。
lm_loss_weight|文字预测差距的权重；负数会关闭这项计算，零与负数在执行开销上不同。
mask_loss_weight|区域预测差距的权重；负数关闭区域损失项。
mask_dice_weight|整体区域重叠差距在区域损失中的权重。
mask_bce_weight|逐像素前景背景差距在区域损失中的权重。
mask_loss_chunk_size|每次转换成较高精度评分的区域数；小分块减少临时显存，非正数表示不分块。
lm_loss_impl|文字评分的计算方式；加速方式不可用时会回退，目标仍是预测下一个词。
group_by_modality_length|是否将图文类型与长度相近的样本放到相近批次，减少补齐浪费。
auto_resume|是否自动查找输出目录中最新训练进度；用户显式指定恢复位置时优先使用显式设置。
data_file_paths|直接提供多个样本文件路径，用冒号分隔；也可改用 data_config 清单。
image_folders|与样本文件一一对应的图片根目录，使用冒号分隔。
data_config|记录样本文件、图片目录及抽样规则的 YAML 清单；相对路径以清单目录为起点。
max_pixels|每张处理后图片的像素数量上限，不是宽度上限；多图时还受总文字长度预算限制。
min_pixels|图片处理的像素数量下限，避免图像被缩得过小。
model_max_length|整条输入允许的最大数字序列长度，包括图片占位位置与文字。
model_name_or_path|待加载的模型目录或模型名称；第一阶段通常是基础模型，第二阶段是第一阶段结果。
attn_implementation|基础模型注意力的计算实现；可用性取决于环境，独立于区域解码器的环境变量设置。
unfreeze_all_parameters|为真则全部参数参与学习，并忽略其余逐模块开关。
unfreeze_vision_tower|是否学习视觉主干的小块特征提取部分，不含后面的合并投影。
unfreeze_vision_projection|是否学习把图片小块合并并映射到语言模型的部分。
unfreeze_language_model_backbone|是否学习语言模型内部层，不包含输入词表表示。
unfreeze_language_model_head|是否学习预测词表分数的输出层；若与输入词表共享权重，两者一起受影响。
unfreeze_embed_tokens|是否学习整个输入词表表示，优先级高于只学习新符号。
unfreeze_new_tokens|只允许 <|mask|> 对应词表行更新，其余行梯度被清零。
unfreeze_mask_decoder|是否学习区域解码器，第一阶段需要开启。
torch_dtype|模型计算和存储所用浮点精度，默认 bfloat16 以减少显存占用。'''.splitlines())
p=R/'src/xiaomi_lens/train/config.py';s=p.read_text();lines=s.splitlines(keepends=True);ins={}
for n in ast.walk(ast.parse(s)):
 if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):ins[n.lineno-1]=' '*n.col_offset+'# 设置含义：'+fields[n.target.id]+'\n'
p.write_text(''.join(ins.get(i,'')+l for i,l in enumerate(lines)))
headers={
'scripts/train_stage1.sh':'阅读第 5 步：第一阶段只学习区域解码器和区域符号表示，关闭文字损失；这里是启动配方，实际训练在 train.py。',
'scripts/train_stage2.sh':'阅读第 5 步：从第一阶段模型继续，全部参数一起学习文字和区域；恢复进度时需使用匹配的分卡保存方式。',
'eval/run.sh':'阅读第 6 步：选择普通预测或 vLLM 预测，再把模型、数据、设备与输出位置交给 eval/run.py；本文件不计算评分。',
'src/xiaomi_lens/integrations/vllm/run_vllm.sh':'阅读第 7 步：启动负责生成文字的服务，默认端口 8000；区域生成需要另外运行 run_vllm_mask.sh。',
'src/xiaomi_lens/integrations/vllm/run_vllm_mask.sh':'阅读第 7 步：启动负责生成像素区域的服务，默认端口 8001；与文字服务采用相同图片处理大小。',
}
for name,desc in headers.items():
 p=R/name;s=p.read_text();lines=s.splitlines(keepends=True);lines.insert(1,'# 【中文导读】'+desc+'\n# 从路径设置 → 设备设置 → 启动命令阅读。完整解释见根目录「学习指南.md」与「配置与示例详解.md」。\n');s=''.join(lines)
 anchors={
 'set -euo pipefail':'遇到命令失败、未设置变量或管道错误就停止，避免继续使用不完整设置。',
 'REPO_HOME=':'定位代码根目录并进入，后面的相对路径由此解释。',
 'NPROC_PER_NODE=':'设置每台机器的进程数和机器编号；通常一张 GPU 对应一个训练进程。',
 'LAUNCHER_ARGS=(':'将多进程启动选项保存为数组，保证带空格的参数不会被错误拆开。',
 'python -m torch.distributed.run ':'开始训练：下方连续多行实际属于同一条命令，反斜杠表示续行。\n# 不要在这些续行中间插入普通注释，否则可能截断命令。各选项详解见根目录配置说明。',
 'BACKEND=':'第一个位置参数选择 hf 或 vllm，shift 后剩余参数继续交给评测程序。',
 'case "${BACKEND}" in':'按计算方式分支：hf 可按卡拆分样本；vllm 分别给文字和区域引擎指定设备。',
 'CKPT=':'读取模型位置；本地目录不存在时，下方代码会把它当在线模型名称下载。',
 'SERVED_NAME=':'设置对外模型名称和端口；客户端请求使用此名称，不是直接传模型目录。',
 'MAX_PIXELS=':'可选地覆盖图片像素上限；两个服务必须一致，否则区域位置可能无法与文字对应。',
 'export PYTHONPATH=':'让当前进程和后续子进程都能找到本项目模块。',
 'if ! python3 -':'检查框架能否发现自定义模型和输入输出处理器；缺失时在临时目录补登记信息。',
 'CKPT_ABS=':'把模型目录变成明确的真实路径，并用其摘要区分缓存，避免同名模型目录互相混淆。',
 'exec vllm serve ':'启动服务并由服务进程接管当前脚本；下方参数分别控制名称、端口和运行方式。',
 }
 out=[]
 for line in s.splitlines(keepends=True):
  for anchor,note in anchors.items():
   if line.startswith(anchor):out.append('# 步骤：'+note+'\n')
  out.append(line)
 p.write_text(''.join(out))
# 格式允许 # 的文件直接解释；不支持注释的 JSON/JSONL 保持原始字节。
for name in ['examples/train/stage1.yaml','examples/train/stage2.yaml','examples/train/sample/sample.yaml']:
 p=R/name;s=p.read_text();out=['# 【中文导读｜第 4 步】这是一份数据来源清单，告诉训练器到哪里读取样本和图片。\n']
 if name.endswith('stage2.yaml'):out.append('# 此示例的 eval 重用了训练文件前 16 条，仅检查流程，不能作为独立泛化评估。\n')
 meanings={'datasets':'数据来源列表；可以列出多个来源。','train':'训练使用的数据分组。','eval':'训练过程中检查损失的数据分组。','name':'便于日志识别的数据名称。','path':'每行一条样本的文件路径，相对本清单目录。','image_folder':'样本中相对图片文件名所依赖的根目录。','tag':'数据用途标签，便于报告按来源统计。','sampling_strategy':'只读取指定范围；first:16 表示前 16 条。'}
 for l in s.splitlines(keepends=True):
  m=re.match(r'(\s*)(?:- )?(\w+):',l)
  if m and m[2] in meanings:out.append(m[1]+'# '+meanings[m[2]]+'\n')
  out.append(l)
 p.write_text(''.join(out))
p=R/'requirements.txt';s=p.read_text();req={'torch':'计算数组、搭建神经网络和执行自动求导。','torchvision':'提供视觉相关工具，与 torch 配套。','transformers':'提供 Qwen 基础模型、分词器、图片处理器与文字生成流程。','accelerate':'协助将模型放到设备上，并组织训练执行。','flash-attn':'可选计算路径所需的快速注意力组件，安装受设备和环境限制。','numpy':'处理 CPU 数组与区域标注。','pillow':'读取、缩放、绘制和保存图片。'}
out=['# 【中文导读｜第 7 步】列出普通推理依赖；阅读代码不需要立即安装所有项目。\n']
for l in s.splitlines(keepends=True):
 key=re.split(r'[><=]',l.strip())[0]
 if key in req:out.append('# '+req[key]+'\n')
 out.append(l)
p.write_text(''.join(out))
p=R/'pyproject.toml';s=p.read_text();sections={'[build-system]':'打包构建工具：将 src 中的代码安装成可导入的软件包。','[project]':'软件包名称、版本和基本信息；运行依赖另见 requirements.txt。','[project.optional-dependencies]':'按用途选择额外依赖：运行服务、评测、训练或训练加速，不必全部装在同一环境。','[tool.setuptools]':'告诉打包工具 infer 模块与 xiaomi_lens 包分别在哪里。','[tool.setuptools.packages.find]':'从 src 开始寻找可安装的 Python 包。','[tool.setuptools.package-data]':'安装时同时携带服务脚本和训练设置文件。','[project.entry-points."vllm.general_plugins"]':'登记自定义模型，vLLM 的每个工作进程都可据此找到它。','[project.entry-points."vllm.io_processor_plugins"]':'登记区域请求的输入输出处理器，负责请求数据与模型数据互相转换。'}
for a,n in sections.items():assert a in s;s=s.replace(a,'# '+n+'\n'+a,1)
p.write_text('# 【中文导读｜第 7 步】这是安装说明书，不是算法运行入口。\n'+s)
p=R/'.gitignore';s=p.read_text();s='# 【中文导读｜第 7 步】告诉版本管理忽略哪些自动生成文件；不会删除文件，也不控制模型读取。\n# 按下方英文分组阅读：缓存、编译产物、运行环境、日志及本项目模型输出。\n# 路径结尾 / 表示目录，* 表示匹配任意字符，! 表示从忽略规则中重新包含。\n'+s
s=s.replace('# Xiaomi-Lens-0 project files','# 项目专用规则：模型权重、运行输出、人造训练数据不随普通源码提交。\n# Xiaomi-Lens-0 project files');p.write_text(s)
for name in ['README.md','README_CN.md','examples/train/README.md','examples/train/README_CN.md']:
 p=R/name;prefix='../../' if name.startswith('examples/') else ''
 p.write_text('> 中文学习导读：先看[学习指南]('+prefix+'学习指南.md)与[完整代码树]('+prefix+'代码树与文件说明.md)。本页保留原项目说明；'+('本训练示例的字段逐项解释见[配置与示例详解]('+prefix+'配置与示例详解.md)。' if prefix else '初学者可先看方法介绍，再按指南跟读一次预测过程；评测表格是原项目报告，不是本次重新测出的成绩。')+'\n\n'+p.read_text())
# 展示页面只增加源码注释，不改变可见内容或交互。
p=R/'docs/index.html';s=p.read_text();s=s.replace('<html lang="en">','<!-- 中文导读｜第 7 步：静态项目展示页，展示已保存的案例，不在浏览器中运行模型。\n阅读顺序：页面结构 → style 外观 → 底部 script 的切换、筛选与复制操作。 -->\n<html lang="en">',1)
for a,n in {'<style>':'外观部分：字体、颜色、布局和窄屏适配，不参与任何模型计算。','<header class="site-header">':'页头与导航：点击后跳到同页对应章节。','<section id="overview">':'项目概览：介绍目标和总览图片。','<section id="capabilities"':'能力介绍：简述不同任务及案例入口。','<section id="samples"':'案例播放区：显示预先保存的 GIF，通过下方脚本切换。','<section id="method"':'算法介绍：展示共享图像特征与区域解码的结构图。','<section id="data"':'数据流程：通过页签切换不同数据说明。','<section id="results"':'结果展示：原项目发布的表格和图，不是打开网页后新运行的结果。','<section id="cite"':'引用信息：提供论文引用文本与复制按钮。','<footer>':'页尾：项目署名与相关链接。','<script>':'交互部分：只负责浏览已有内容，不加载模型权重。'}.items():
 assert a in s;s=s.replace(a,'<!-- '+n+' -->\n'+a,1)
js={
'  function loadLazy(root) {':'接收页面容器，把其中 data-lazy 图片地址转成真正 src；仅在需要展示时加载。',
'  document.querySelectorAll(\'[role="tablist"]\')':'对每组页签登记点击事件：更新选中状态、显示对应面板，并加载面板图片。',
'  function show(btn, scroll) {':'接收选中的案例按钮，读取它保存的图片与文字信息并更新播放器；可选滚动到播放器。',
'  cases.forEach(function (b) { b.addEventListener':'给所有案例按钮绑定选择操作，点击哪个就播放哪个。',
'  document.getElementById("replay")':'重新设置同一 GIF 地址，从头播放当前案例。',
'  document.querySelectorAll(".chip[data-filter]")':'按任务类别筛选案例，隐藏不匹配条目，并自动展示第一个匹配项。',
'  document.querySelectorAll("[data-jump]")':'从能力介绍跳到指定案例：先恢复显示全部，再选中目标案例。',
'  if ("IntersectionObserver" in window) {':'浏览器支持可见区域观察时，靠近屏幕的图片才加载；否则一次加载所有延迟图片。',
'    var links = {};':'记录导航锚点，滚动页面时高亮当前正在阅读的章节。',
'  document.getElementById("copy")':'将引用文字写入剪贴板，成功后暂时显示 Copied，再恢复按钮文字。',
}
for a,n in js.items():assert a in s;s=s.replace(a,'  // '+n+'\n'+a,1)
s=s.replace(':root {','/* 集中定义颜色与字体，下面通过 var(...) 复用。修改这里影响外观，不改变案例数据。 */\n:root {',1)
s=s.replace('@media','/* 窄屏适配：屏幕宽度满足条件时调整布局。 */\n@media')
p.write_text(s)
print('Annotated scripts, configs, readmes and display page')
