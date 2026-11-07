from pathlib import Path
R=Path('Xiaomi-Lens-0-raw')
def block(file,anchor,note):
 p=R/file;s=p.read_text();assert anchor in s,(file,anchor)
 pos=s.index(anchor);start=s.rfind('\n',0,pos)+1;indent=s[start:pos];assert not indent.strip(),(file,anchor)
 text=''.join(indent+'# '+line+'\n' for line in note.split('\n'))
 s=s[:start]+text+s[start:];p.write_text(s)
f='infer.py'
for a,n in [
('messages = [{','第 1 步：把图片与问题放进一条用户消息。模型要求的角色和起止符号由其自带模板补上，\n不是简单把任意文字拼在图片文件名后面；训练和使用时的格式需要一致。'),
('processor.tokenizer.padding_side = "left"','第 2 步：处理器把文字换成编号，并把图片缩放、切成小块。\ninput_ids 是文字/图片占位位置；pixel_values 是实际图片内容；image_grid_thw 记住网格尺寸。'),
('completion_ids, generated_text, total_mask_count = generate_completion(','主线 A：先让模型自由生成文字。此时还没有把目标区域画出来。\n回答中若出现 <|mask|>，就相当于留下一个“这里有一个目标区域”的索引。'),
('pred_masks = run_forward(model, prompt_inputs, completion_ids, mask_token_id, device)','主线 B：将刚才生成的整段回答重新送入同一个模型，读出各区域标记的上下文状态。\n这是基于模型自己的回答生成区域，不是把人工正确区域喂给模型。'),
('canvas = image.resize((pixel_w, pixel_h), Image.LANCZOS).convert("RGBA")','显示尺寸：这里把原图缩放到模型处理画布，而不是将区域放回原始照片分辨率。\n例如网格是 20×30，小块边长 16，则画布高 320、宽 480；Image.resize 接收顺序是宽、高。'),
('probs = torch.sigmoid(pred_masks[0, k]).float().cpu().numpy()','原始分数不限制范围，sigmoid 将其变成 0～1。分数为 0 时概率是 0.5。\n通常按阈值决定是否着色；threshold=0 时按概率深浅画热力图，并不是所有像素一律选中。'),
('x0 = bb[0] / 1000.0 * pixel_w','框坐标使用 0～1000 的相对刻度。例如宽 480 时，横坐标 500 表示第 240 像素。\n框来自生成文字，区域来自解码器；它们不是同一段代码的两个画法。')]:block(f,a,n)
f='src/xiaomi_lens/model/pixel_mask_decoder.py'
for a,n in [
('self.mask_embed = MLP(llm_hidden, llm_hidden, embed_dim)','两条支路最后必须有相同宽度 embed_dim：文字目标变成一串“判断标准”，\n图片每个位置也变成同样长的一串特征，才能通过逐项相乘求和给每个位置打分。'),
('self.upscaler = nn.Sequential(','这里两次带学习参数的放大都让高宽乘 2，总计乘 4。\nforward 最后还会做固定的四倍插值，因此输出高宽相对原始小块网格各乘 16。'),
('mask_embedding = self.mask_embed(mask_hidden)','步骤 1：把区域标记处的文字状态转换成区域查询。\n形状 B×K×D → B×K×C：B 为本次分组数，K 为每组区域数，C 为共享表示宽度。'),
('patch_features = self.patch_proj(image_features)','步骤 2：将图片特征宽度 V 转到文字模型宽度 D；小块数 P=h×w 不变。\n随后小块之间互相交换信息，得到更能理解整张图的表示。'),
('patch_embedding = patch_embedding.reshape(','步骤 3：把小块的一维列表放回二维网格：B×(h·w)×D → B×D×h×w。\nreshape 改排列形状，permute 调整轴顺序；必须先确认小块顺序已恢复成逐行空间顺序。'),
('reduced_features = self.channel_reduce(patch_embedding)','步骤 4：先减少通道数，再逐步增加高宽。通道是每个位置的信息种类，不是 RGB 颜色数。\n例如 h=2,w=3，放大后的像素表示高 8、宽 12，接下来每个区域都给这些位置评分。'),
('mask_logits = torch.einsum(','步骤 5：每个区域查询与每个像素表示做点积，沿 C 求和。\n"bkc,bchw->bkhw" 表示保留批次 b、区域 k、高 h、宽 w，只把共同宽度 c 相乘再累加。\n除以 √C 控制初始分数幅度，避免一开始 sigmoid 就接近 0 或 1 而难以学习。'),
('return _bilinear_interpolate_4x(mask_logits)','步骤 6：高宽再乘 4，得到模型画布大小的原始分数。\n上例最终为 B×K×32×48。这里只输出分数，不在这里用阈值切断，训练才能继续传递梯度。')]:block(f,a,n)
f='src/xiaomi_lens/model/modeling_xiaomi_lens.py'
for a,n in [
('spatial = (','恢复空间顺序：视觉主干为了合并相邻小块，会按小组存储特征。\n这里先展开“组行、组列、组内行、组内列”，再交换轴，使展平后的顺序对应真实图片行列。'),
('inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)','图片与文字会合：把占位符处的表示替换成图像编码器产生的合并后特征。\n区域解码器随后使用的是合并前的较细网格，两条支路共享视觉主干，但取用的位置不同。'),
('groups: dict = {}','分组不是按区域各复制一次图片：同一样本中属于同一图片的所有区域共用一组图片特征。\n例如一张图片有三只猫，只需要做一次这张图片的小块注意力，再用三个目标查询分别评分。'),
('grouped_mask_hidden = hidden_states.new_zeros','为不同大小的图片与不同区域数准备统一容器。\n补零只是让数据能放成规则数组，后面的有效位置标记必须阻止这些补齐像素参与注意力和损失。'),
('grouped_mask_hidden[g, j] = hidden_states[b, pos]','关键连接点：从语言模型中取的是 <|mask|> 所在位置的上下文状态，\n它已经结合了图片、问题与前文，不是单独读取一个永远不变的特殊符号词向量。'),
('grouped = self.mask_decoder(','整批只调用一次区域解码器，便于多个训练进程保持相同的执行次序。\n即使某个进程没有有效区域，也需要配合其他进程，不能独自提前跳过所有相关计算。'),
('pred_masks = grouped.new_zeros','解码完后把“图片分组”中的结果放回“样本编号、区域编号”的原顺序，\n这样第 k 个输出仍与第 k 个区域标记、训练标注一一对应。'),
('outputs = self.model.language_model(','共享主干步骤：基础语言模型处理混合后的图文序列。\n其内部的注意力、状态更新和缓存实现不在本仓库，而来自 transformers 的 Qwen3.5 实现。'),
('has_mask_work = bool(','并非每次计算都有区域任务：生成文字的过程中可能只需要下一个词的分数。\n只有给出区域位置、图片网格与有效图片特征时，才进入真实区域解码路径。'),
('seen = is_vision_start.long().cumsum(dim=1)','多图归属示例：某区域标记前出现两次图片开始符号，则它归属编号 1 的图片。\n编号从 0 开始，所以累计数量需要减 1；调用者也可以显式传入 mask_image_index。'),
('dummy_mask = hidden_states.new_zeros','这是多进程同步所需的占位计算，不是有效的预测结果。\n对应有效像素和权重为零，让该进程保持计算连接，但不从虚构目标中学习。'),
('if (labels[..., 1:] != -100).any():','-100 表示“不考这个位置”。如果截断后所有标签都被忽略，普通平均会变成 0/0。\n这里返回仍连接计算图的零，既避免数值异常，也保证训练进程参与同样的反向过程。'),
('mask_loss_value = self.mask_loss_fn(','将样本与区域两个轴合起来：B×K×H×W → (B·K)×1×H×W。\n每张目标区域各算一份差距，补齐区域由有效像素标记排除，再与文字损失相加。')]:block(f,a,n)
f='src/xiaomi_lens/train/data/preprocess.py'
for a,n in [
('image_paths = list(sample.get("images") or [])','步骤 1：先整理对话与图片路径。训练第一阶段可能只有助手答案，辅助函数会补齐用户图片消息。'),
('image_processor_kwargs = {}','步骤 2：分配图片占用的序列预算。图片越大，需要的小块和占位符越多。\n多张图共用总长度，因此这里按图片数量分摊上限，同时给文字预留位置。'),
('input_ids = np.concatenate([prompt_ids, completion_ids])','步骤 3：把已知问题与正确回答连起来。训练时模型能看到前面的正确词，学习预测下一个词。\n这与使用时从自己的预测继续生成不同，不应把训练回答误认为模型自由生成结果。'),
('labels = input_ids.copy()','步骤 4：标出哪些文字位置参与学习。问题部分先设为 -100，再恢复各轮助手回答。\n模板自动提供的生成前缀不应该当作模型需要学会生成的正文。'),
('full_input_ids = input_ids','步骤 5：保留截断前的原序列，确保第 k 个区域仍对应 segmentation[k]。\n被截掉的区域会标为无效，不能直接从列表删除，否则后面的标注就错位了。'),
('if assistant_spans:\n','步骤 6：只在助手回答中寻找区域标记。用户引用同样的符号不代表模型输出了区域。'),
('num_images = int(image_grid_thw.shape[0])','步骤 7：为每个区域找到其前面最近的图片，并记录那张图片自己的网格尺寸。\n多图大小不同，正确区域必须与自己的图片对齐，而不是全都按第一张图缩放。'),
('mask_pixel_masks: list[bytes] = []','步骤 8：把正确区域缩放到对应模型画布，用位打包减少传输体积。\n一个字节装 8 个 0/1 像素；训练器稍后再展开，这种压缩不改变区域含义。')]:block(f,a,n)
f='src/xiaomi_lens/train/trainer.py'
for a,n in [
('K = max((len(p) for p in inputs["_per_sample_mask_positions"]), default=0)','准备区域批次：同批样本可以有不同区域数，K 取最大值，小于 K 的样本补空槽。\n所有空槽都必须有有效标记，不能因补齐而被当成“应该预测全背景”的真实样本。'),
('mask_targets = torch.zeros','为整批建立最大画布，但仍为每个区域标记真实高宽。\n正确区域用 uint8 存 0/1，有效范围用 bool，避免所有画布提前转成较大的浮点数组。'),
('bits = ((packed_t.unsqueeze(1) >> shifts) & 1).flatten()','按位展开：例如一个字节 160 的二进制为 10100000，按从高位到低位还原为 8 个像素。\n每张图不足 8 的尾部补位不属于真实像素，后面按 ch*cw 截取。'),
('model_output = model(','训练的一次计算从这里进入核心模型：输入既有图片问题，也有用于评分的正确回答和区域。\n返回损失后，父类训练循环负责求梯度和更新参数；本函数负责把模型要用的数据组织好。'),
('if model_args.unfreeze_embed_tokens:','新符号行训练与整个词表训练不同：前者只允许指定行更新，后者所有行都能变化。\n输入词表和输出层若共享同一份权重，开放其中一个也影响另一个，因此后面要检查冲突。')]:block(f,a,n)
f='src/xiaomi_lens/model/losses.py'
for a,n in [
('numerator = 2 * (probs * tgt).sum(dim=-1)','整体重叠：预测越接近正确区域，分子越接近分母，1 减比值越接近 0。\n这里用连续概率计算，不能先切成硬 0/1，否则模型难以得到“该向哪个方向改”的信息。'),
('num_masks = _global_num_masks(local_num_masks)','多张卡的有效区域数可能不同，所以统一分母，避免某张卡少量区域被赋予过大的相对权重。\n这里的同步只执行一次，不能放进每个分块或反向重算中，否则各进程次序可能不一致。'),
('for start in range(0, num_rows, chunk_size):','分块只为节省临时存储：各块仍用同一个全局分母，最后相加。\n不要误把各块各自平均再直接相加，那会改变不同块的权重。')]:block(f,a,n)
f='eval/run.py'
for a,n in [
('intersection = int(np.logical_and(prediction, target).sum())','交集是预测与标注都选中的像素，并集是至少一方选中的像素。\n例如交集 6、并集 10，IoU=0.6；这说明像素重叠程度，不代表文字回答正确率。'),
('masks = F.interpolate(masks[:, None].float(), size=size, mode="bilinear",','先缩放连续分数，再取阈值；顺序不能随意交换。\nnearest_binary_union 则先取阈值再用最近邻缩放，它是另一个比较分支，可能产生不同分数。'),
('result = {"overall": metrics(records)}','cIoU 是把全部交集相加后除以全部并集，大区域影响较大；\ngIoU 在本文件中是逐例 IoU 的平均，不是另一种框度量 Generalized IoU。'),
('if set(records) != seen or len(seen) != shard_count:','不仅要循环结束，还要确认结果恰好覆盖所需样本，不能缺少、重复或混入别的运行结果。')]:block(f,a,n)
# 修正阅读核对中发现的两处表述，确保解释跟实际返回值一致。
p=R/'src/xiaomi_lens/train/preflight.py';s=p.read_text().replace('读取枚举的实际值并转成文字，返回统一显示形式，便于记录训练选项。','如果输入是枚举则返回其实际值，否则原样返回，便于判断训练选项。').replace('接收数据集及来源配置，汇总来源、选取数量和长度等信息，返回训练前报告的数据部分。','接收数据集与分组名称，抽查图文及区域数量、来源和缺失情况；返回数据报告，并记录是否只检查了部分样本。');p.write_text(s)
p=R/'eval/run.py';s=p.read_text().replace('为文字与区域两个引擎准备各自需要的模型文件视图，返回路径，供后续引擎加载。','为文字与区域两个引擎准备模型文件视图并写入插件登记信息；不返回值，后续按约定路径加载。');p.write_text(s)
print('Added key-step walkthroughs')
