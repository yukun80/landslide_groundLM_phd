from pathlib import Path
import json
R=Path('Xiaomi-Lens-0-raw')
def block(file,anchor,note):
 p=R/file;s=p.read_text();pos=s.index(anchor);start=s.rfind('\n',0,pos)+1;indent=s[start:pos];assert not indent.strip()
 s=s[:start]+''.join(indent+'# '+l+'\n' for l in note.split('\n'))+s[start:];p.write_text(s)
for a,n in [
('prompt_length = prompt_inputs["input_ids"].shape[1]','先记住问题占了多少位置。generate 返回时带有原问题前缀，之后必须切掉，才是模型新写的回答。'),
('generated_text = processor.tokenizer.decode','保留特殊符号：skip_special_tokens=False 很关键。若把区域符号当无用字符删除，就无法按文字核对区域个数。'),
('completion_mask = torch.ones_like(completion_ids)','重新计算时，新生成的每个位置都是真实内容，因此注意力有效标记全为 1，再与问题部分拼起来。'),
('completion_mm = torch.zeros(','新回答是文字位置，因此多模态类型标记补零；图片仍保留在原问题中的位置。'),
('mask_positions = (completion_mask_indices + prompt_length)','区域在“新回答”中的下标，要加上问题长度才是“完整序列”中的下标。\n例如问题有 100 个位置、区域在回答第 8 个位置，则完整序列下标是 108（从 0 开始）。')]:block('infer.py',a,n)
for f,a,n in [
('src/xiaomi_lens/train/data/jsonl_index.py','cursor += line_len','cursor 按原始字节前进，包括换行；只去掉换行后的长度用于读取 JSON。\n不能按中文字符个数当字节偏移，否则含中文样本会读错位置。'),
('src/xiaomi_lens/train/data/jsonl_index.py','sample = json.loads(self._mmap[off:off + ln])','只从当前位置读这一行，不解析整个文件；图片此时仍是路径，实际像素留到拼批时再读。'),
('src/xiaomi_lens/train/data/jsonl_index.py','for attempt in range(self.n_try_fetch):','先试用户要求的样本，失败才取备用编号；有限重试防止一直困在坏数据上。\n这会改变实际送入的样本，所以警告和数据检查不能忽略。'),
('src/xiaomi_lens/train/data/io.py','unknown = set(raw_entry) - allowed','检查拼错的字段名，避免把本应生效的设置悄悄忽略。\n例如 image_folders 与本清单要求的 image_folder 并不是同一个名字。'),
('src/xiaomi_lens/train/data/render.py','completion_text = full_text[len(prompt_text):]','用同一模板得到完整文本与问题前缀，然后切掉前缀得到训练回答。\n这里依赖模板输出的对应关系，不能随意在其中一侧追加格式符号。'),
('src/xiaomi_lens/train/data/segmentation.py','value |= (code & 0x1F) << shift','每个字符贡献 5 位整数信息，额外位表示是否继续读取和符号信息。\n这是 COCO 的存储格式，不是神经网络计算，初读可先把函数当作解压器。'),
('src/xiaomi_lens/train/data/segmentation.py','flat = np.zeros(total, dtype=np.uint8)','从全背景开始，只把第 1、3、5…段填为前景。\n计数序列从背景段开始，即使第一个像素就是前景，前面也要有一个长度为 0 的背景段。'),
('src/xiaomi_lens/train/collator.py','max_len = max(len(r["input_ids"]) for r in encoded)','同一批次需要规则的长方形数组，因此以最长样本为准在右侧补齐。\nattention_mask 的 0 表示补齐位置；labels 的 -100 表示这里不计文字错误。'),
('src/xiaomi_lens/train/collator.py','pixel_chunks = [','图片小块直接按样本顺序连接，并单独保留每张图片网格。\n文字轴的补齐与图片小块轴的连接不同，不能把两个维度混为一谈。'),
('src/xiaomi_lens/train/sampler.py','generator.manual_seed(self.curr_seed)','固定本轮随机种子，使所有进程先得到相同的总顺序，再按进程编号领取不同份额。'),
('src/xiaomi_lens/train/preflight.py','max_samples = int(os.environ.get','默认最多抽查 20000 条，避免启动报告就遍历巨大数据集。\nsummary_truncated 会告诉你是否只看了部分样本，报告不能当作完整质量审计。'),
('src/xiaomi_lens/integrations/vllm/io_processor.py','prompt_token_ids = list(prefix_ids) + list(completion_ids)','保留生成边界：问题前缀与生成结果分别提供编号后拼接，\n避免把两个文本重新合并分词而改变区域标记附近的对应位置。'),
('src/xiaomi_lens/integrations/vllm/io_processor.py','grid_h, grid_w, k =','扁平结果的前三个数字是网格高、网格宽与区域数，其余才是像素分数。\n这个头部约定要与 vllm_model.py 的 decode_mask_batch 保持一致。'),
('src/xiaomi_lens/integrations/vllm/vllm_model.py','grouped.setdefault(grid, []).append','只把网格大小完全相同的请求放一起计算，避免把不同图片的空间排列混用。'),
('src/xiaomi_lens/integrations/vllm/prepare_served.py','drop_lm_head = _tie_word_embeddings(checkpoint)','输入与输出词表共享时，重复输出层可从文字服务视图移除；不共享时不能删除。\n这个判断只针对服务副本，不是修改原模型的算法或学习内容。'),
('tools/prepare_base_model.py','before_by_dir =','先检查所有待处理目录，再进入实际修改，避免前一个目录已写入后才发现另一个格式不合法。'),
('examples/train/make_toy_data.py','mask_image = Image.new','图片和正确区域使用同一组几何坐标绘制；图片有颜色，正确区域只存 0/1。\n因此人造样本无需人工描边，但也不能代表复杂真实场景的难度。'),
]:block(f,a,n)
p=R/'src/xiaomi_lens/constants.py';s=p.read_text().replace('特殊符号、画图颜色和默认生成长度','特殊符号和默认生成长度');s=s.replace('DEFAULT_MAX_NEW_TOKENS =','# 新生成部分的长度上限，不包含已有问题；16384 是较长默认值，小检查可另传较短上限。\nDEFAULT_MAX_NEW_TOKENS =').replace('MASK_TOKEN =','# 真正用于定位区域状态的特殊符号；它的编号由模型分词器决定，不能随意手写一个数字代替。\nMASK_TOKEN =').replace('BOX_START_TOKEN =','# 以下是答案格式的边界文字。框符号沿用 Qwen 编号，<mask> 和 </mask> 是普通文本边界。\nBOX_START_TOKEN =');p.write_text(s)
p=Path('.learning-work/files.json');d=json.loads(p.read_text());d['src/xiaomi_lens/constants.py'][1]=d['src/xiaomi_lens/constants.py'][1].replace('特殊符号、画图颜色和默认生成长度','特殊符号和默认生成长度');p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
p=R/'学习指南.md';s=p.read_text().replace('它放特殊符号、默认长度和颜色。','它放特殊符号和默认生成长度；画图颜色在 infer.py 内。');p.write_text(s)
p=R/'配置与示例详解.md';s=p.read_text().replace('此配方传 0.03；实际转换规则由已安装训练参数实现处理，不要把小数直觉读成“运行 0.03 步”','此配方传 0.03；已核对本机 Transformers 5.4.0：小于 1 时按总训练步数的比例计算并向上取整，即预热约前 3% 的步骤');p.write_text(s)
Path('.learning-work/static-results.json').write_text('{"status":"checking"}\n')
