from pathlib import Path
import ast,json
ROOT=Path('Xiaomi-Lens-0-raw')
FILES={
'infer.py':('2','单张图片的完整使用入口。把图片和问题交给模型，先获得文字，再把回答中的区域画到图片上。','从 main 开始，按 build_inputs → generate_completion → run_forward → visualize_pixel_mask 跟读。'),
 'train.py':('5','训练总入口：读取训练选项、确定数据来源、创建训练器，并启动或继续训练。','先看 main，再回头看恢复训练的检查；真正的一步训练在 train/trainer.py。'),
 'eval/run.py':('6','在 LensBench 上逐张预测、与人工标注比较并汇总分数，也支持分卡执行和比较两种结果缩放方式。','先看 score_masks、summarize，再看 main；最后读并行与 vLLM 部分。这里的 Prec@0.5 按区域重叠计分，不要直接当作文字框坐标的评分。'),
 'examples/train/make_toy_data.py':('4','画简单形状，并生成对应的问题、答案和区域标注，用于理解数据格式与检查训练流程。','先看 draw_image，再看 stage1_row、stage2_rows；这些人造图形不是正式效果评测。'),
 'tools/prepare_base_model.py':('7','检查并可选地统一基础模型的边界框特殊符号名称，不重新训练或改写模型权重。','默认只检查；--in-place 才写入分词器文件。先看 inspect_model，再看 rewrite_model。'),
 'src/xiaomi_lens/constants.py':('2','集中存放特殊符号、画图颜色和默认生成长度，供推理与训练共用。','MASK_TOKEN 是文字中的区域标记；它不是一张图片，模型需要另一步把它解释为区域。'),
 'src/xiaomi_lens/model/config.py':('3','从基础模型的设置中取出文字和图像宽度，构造区域解码器需要的尺寸设置。','先看 get_qwen_vision_config，再看 build_mask_decoder_config；不同宽度必须通过投影对齐。'),
 'src/xiaomi_lens/model/modeling_xiaomi_lens.py':('3','把基础图文模型与区域解码器接起来，是理解算法的核心文件。','先看 forward 的主线，再看 _prepare_multimodal_inputs 和 _decode_pixel_masks。基础 Qwen 的内部层在 transformers 中。'),
 'src/xiaomi_lens/model/pixel_mask_decoder.py':('3','把目标的文字状态与图片各位置的特征结合，输出每个像素属于目标的原始分数。','先看 PixelMaskDecoder.forward，再看构造函数；注意力加速和分块反向传播最后读。'),
 'src/xiaomi_lens/model/losses.py':('5','衡量预测区域与正确区域的差距：既看整体重叠，也看每个像素是否分对。','先看 MaskLoss，再看 DiceLoss、BCEWithLogitsLoss；输入是未变成概率的分数。'),
 'src/xiaomi_lens/model/lm_loss.py':('5','衡量下一个文字单元是否预测正确，同时提供节省显存的等价计算路径。','先理解 shift_labels_flat 的错一位对齐，再看 _causal_lm_loss；加速路径不改变学习目标。'),
 'src/xiaomi_lens/train/config.py':('5','定义训练、数据和模型三个方面可调整的选项及默认值。','与两个训练脚本对照：脚本传入的值可以覆盖这里的默认值。'),
 'src/xiaomi_lens/train/trainer.py':('5','组织模型加载、可训练部分选择、数据读取、每步损失计算、记录和保存。','先看 compute_loss 与 _build_mask_inputs，再看 __init__；训练更新循环继承自外部 Trainer。'),
 'src/xiaomi_lens/train/collator.py':('4','把多条长短不同的样本临时加工并拼成一批；失败样本会记录并跳过。','先看 __call__；补齐文字长度不等于添加真实内容，labels=-100 的位置不计文字损失。'),
 'src/xiaomi_lens/train/sampler.py':('5','决定每一轮如何打乱、分组和分配样本，使各训练进程得到自己的批次。','先看 BatchSamplerShard.__iter__；分组主要减少无用补齐，不是学习图像特征。'),
 'src/xiaomi_lens/train/preflight.py':('5','训练开始前整理数据规模、可训练参数和训练设置，让人先核对任务是否符合预期。','build_preflight_report 是入口；报告是配置检查，不是模型效果证明。'),
 'src/xiaomi_lens/train/data/io.py':('4','读取数据清单，把样本文件和图片位置整理成统一形式。','先看 DatasetEntry 和 load_dataset_config；清单中的相对路径以清单所在目录为起点。'),
 'src/xiaomi_lens/train/data/jsonl_index.py':('4','只记住每条样本在文件中的位置，需要时再读取，避免一次载入全部数据。','先看 scan_jsonl_offsets → JsonlIndexShard.get → JsonlIndexDataset.__getitem__。'),
 'src/xiaomi_lens/train/data/render.py':('4','把不同来源的对话整理成模型认识的格式，并对齐图片占位符。','先看 render_prompt_and_completion；用户的提问和助手的答案在训练中扮演不同角色。'),
 'src/xiaomi_lens/train/data/preprocess.py':('4','把单条图文样本变成数字输入，找出要学习的回答位置，并对齐正确区域。','encode_sample 是主线；多图时每个区域默认归属其前面最近的一张图片，截断时不能只留半张图片的占位符。'),
 'src/xiaomi_lens/train/data/segmentation.py':('4','把压缩存储的区域标注还原成黑白像素图，并缩放到模型处理图片的大小。','先看 decode_coco_rle，再看 segmentation_to_pixel_mask；标注缩放用最近邻，避免产生不存在的中间类别。'),
 'src/xiaomi_lens/train/data/sft_format.py':('4','把框坐标与区域标记拼成训练答案，保证文字中的区域顺序与标注一致。','从 build_completion 跟到 mask_block；坐标采用 0～1000 的相对刻度。'),
 'src/xiaomi_lens/integrations/vllm/client.py':('7','向文字服务请求回答，再把同一图片和生成结果送给区域服务，并保存叠加图。','main 串联两个服务；保留实际生成的数字编号，避免文字重新编码后对应错位。'),
 'src/xiaomi_lens/integrations/vllm/io_processor.py':('7','把外部区域请求转换成模型输入，再把模型输出转换为可传输的图片或区域编码。','pre_process 负责进入，post_process 负责返回；原图大小与模型画布大小是两个概念。'),
 'src/xiaomi_lens/integrations/vllm/vllm_model.py':('7','在 vLLM 的模型中接入相同的区域解码器，收集图片特征并按请求生成区域。','先看 decode_mask_batch，再看收集特征的两个小函数；这里预期每个请求只有一张图片。'),
 'src/xiaomi_lens/integrations/vllm/plugin.py':('7','向 vLLM 登记本项目的自定义模型类，让服务能找到它。','登记不是加载权重；实际加载与运行由服务框架调用。'),
 'src/xiaomi_lens/integrations/vllm/mask_service.py':('7','提供一个使用普通模型加载方式的 HTTP 区域服务，接收图片、问题和已生成文字。','这是独立的备选服务；当前 client.py 默认调用的是 vLLM 的 /pooling，不是这里的 /decode_mask。'),
 'src/xiaomi_lens/integrations/vllm/prepare_served.py':('7','在独立输出目录准备服务所需的模型文件视图，兼容文字服务与区域服务两种加载方式。','先看 prepare；文字服务可能去除区域权重，区域服务必须保留。输出目录不要设成源模型目录。'),
}
for path,stage,desc in [
('src/xiaomi_lens/__init__.py','2','提供软件包入口，只有访问具体名字时才加载对应模块。'),
('src/xiaomi_lens/model/__init__.py','3','对外提供模型、解码器和损失类，并延迟导入较重的依赖。'),
('src/xiaomi_lens/train/__init__.py','5','对外提供训练器和训练设置，按需加载以降低导入成本。'),
('src/xiaomi_lens/train/data/__init__.py','4','汇总数据相关的对外名称，调用时再去相应文件查找实现。'),
('src/xiaomi_lens/integrations/__init__.py','7','标记外部工具适配目录，本文件没有算法计算。'),
('src/xiaomi_lens/integrations/vllm/__init__.py','7','标记 vLLM 适配包，具体加载、服务和调用逻辑在相邻文件。')]:FILES[path]=(stage,desc,'先了解这个目录提供什么，随后去具体实现文件阅读；__init__.py 不是训练起点。')
D={}
def add(s):
 for line in s.strip().splitlines():
  k,v=line.split('|',1);D[k]=v
add('''
parse_bboxes|从完整回答中提取按出现顺序排列的框坐标，返回坐标列表；此处也会读到思考部分的框。
split_answer_content|以第一个 </think> 分开思考与答案，返回答案文字和思考部分的区域标记数，供后面剔除辅助区域。
parse_answer_bboxes|先去掉思考内容，再解析框；返回只属于最终答案的坐标列表。
load_model_and_processor|接收模型位置和计算设备，加载已训练的模型、图片文字处理器，并返回区域特殊符号的数字编号。
build_inputs|接收图片、完整问题和处理器，用模型自带的对话格式生成输入；返回数字输入与原始提示文字。
generate_completion|根据已准备输入逐步生成回答；返回新生成的编号、解码文字和区域标记总数，总数包含思考中的标记。
run_forward|把原问题与刚生成的回答拼在一起再计算一次，读取每个区域标记处的状态，返回所有区域的原始像素分数。
visualize_pixel_mask|接收图片、预测区域及图片网格信息；在模型画布上给区域着色、可选画框，再返回用于保存的叠加图片。
__getattr__|当访问包里尚未导入的名字时查找其所属模块，返回相应对象；未知名字抛出 AttributeError，避免提前加载所有依赖。
get_qwen_vision_config|从整体设置中查找 vision_config 或兼容名称 visual_config；返回图像部分设置，缺失时明确报错。
build_mask_decoder_config|读取文字宽度、注意力头数与图像宽度，返回构造区域解码器的设置字典；默认中间通道 256、像素表示宽度 32。
_get_fused_linear_ce|第一次需要时尝试导入融合版文字评分工具并缓存，返回求平均与求和两个版本；不可用时返回 None。
_get_flash_ce|尝试取得 flash-attn 的交叉熵计算工具并缓存；返回均值和总和版本，失败后由调用方使用普通计算。
_dist_world_size|读取当前参与训练的进程总数；没有初始化多进程训练时返回 1。
shift_labels_flat|将正确答案左移一位并展平，末尾用 -100 补齐；例如 [10,20,30] 变成 [20,30,-100]，因为当前位置预测下一个词。
_use_fused_linear_lm_loss|检查训练模式、设备、选项和加速工具是否满足条件，返回是否可跳过完整词表分数的显式保存。
_token_sum_scale|根据全局有效文字数量与进程数返回缩放系数，让不同进程参与平均后仍按有效文字数计分。
_fused_linear_lm_loss|从隐藏状态和标签中只取要学习的位置，融合文字预测与评分，返回文字损失，减少巨大词表带来的临时存储。
_causal_lm_loss|接收文字预测分数和正确标签，优先使用可用加速计算，否则调用基础模型的评分方法；返回文字损失。
_global_num_masks|接收本进程的有效区域数，汇总所有进程后返回损失归一化分母；最小为 1，避免没有区域时除零。
_compute_chunk|将一小批预测和标注临时转成较高精度，计算重叠差距与逐像素差距；返回两项加权结果，供所有小批求和。
_get_vision_config|转交给统一设置读取函数，返回图像部分设置，避免在模型类中重复处理兼容名称。
_get_spatial_merge_size|读取图像小块合并的边长倍数；返回整数，用于恢复合并前的空间排列。
_flatten_image_features|从视觉输出提取供语言模型使用的合并后特征，统一返回列表；缺失必要特征时报错。
_extract_pre_merge_features|按每张图片的网格尺寸切分视觉特征，再将合并块顺序还原成逐行顺序；返回每张图片的细网格特征列表。
_prepare_multimodal_inputs|把文字编号转换成表示，把图片特征填入图片占位符，并准备空间位置信息；返回混合输入、视觉输出和位置编号。
_decode_pixel_masks|按样本及所属图片收集区域查询，补齐不同大小的网格后统一解码；返回区域分数和有效像素范围，补齐区域不参与评分。
_resolve_attention_backend|读取环境变量指定的注意力计算方式并检查取值；返回选用的方式名称，供各条计算路径共用。
_flash_warn_once|接收警告标识和文字，每种原因只记录一次；不返回计算结果，避免相同回退提示刷屏。
_get_flash_funcs|按需导入并缓存 FlashAttention 函数；返回函数组合或 None，让上层可以回退到可用方式。
_self_attention|根据设置选择图片小块之间的注意力实现；返回更新后的特征，加速不可用时回退到普通实现。
_mha_self_attention|使用 PyTorch 的多头注意力处理图片小块，传入补齐位置标记；返回每个位置融合邻近和远处信息后的特征。
_project_qkv|用已有权重从每个小块特征生成查询、键和值；返回三组按批次、小块、头、头内宽度排列的数组。
_project_output|把各注意力头的结果合回模型宽度，并应用输出投影；返回与后续残差相加兼容的特征。
_sdpa_self_attention|使用 PyTorch 的缩放点积注意力计算小块关系；正确转换补齐标记的真假含义，返回投影后的特征。
_flash_self_attention|检查设备、类型等条件后使用快速注意力；条件不支持时返回 None，通知上层回退。
_flash_varlen_attention|保留所有查询位置，只打包有效的键和值，执行不同有效长度的注意力；返回恢复成规则排列后的特征。
_bilinear_interpolate_4x|接收区域分数，将高宽分别放大四倍；需要反向传播且超过 CUDA 限制时走分块版本，返回放大后的分数。
_remap_legacy_keys|加载旧权重时改写历史参数名称，使它们对应当前层名；修改传入的权重映射，不改变训练出来的数值。
_maybe_checkpoint|接收要执行的模块和输入，训练需要节省显存时通过重算保留必要信息；返回模块结果。
_validate_inputs|检查区域查询和图片特征的维数、批次及网格数量是否匹配；合法时返回批次大小，不匹配则报错。
_fallback_token_id|为整批失败的占位输入寻找可用文字编号，优先补齐符号，再尝试结束或起始符号，最后返回 0。
_sample_tag|从样本中提取来源名称、标签和图文数量，返回简短文字用于错误记录。
_dummy_batch|接收原批次大小，构造不含图片且所有标签均忽略的占位批次；返回字典，让多进程训练保持同步而不学习坏样本。
_pad|接收短序列、填充值和目标长度，在右侧补齐并返回新列表；文字补 0，学习标签补 -100。
parse_path_spec|把冒号分隔的多个路径拆成列表，忽略空项；没有输入时返回空列表。
resolve_dataset_entries|优先从数据清单读取来源，否则解析命令行中的样本路径和图片目录；返回统一的数据来源列表。
resolve_sample_images|将 image 或 images 统一成图片路径列表，并接上图片根目录；返回样本副本，纯文字样本的图片列表为空。
resolve_dataset_config_path|依次检查直接给出的清单路径和包内预设位置，返回存在的文件路径；两者都没有则报错。
_relative_to|接收基准目录与路径；绝对路径原样返回，相对路径拼到基准目录后返回。
load_dataset_config|读取 YAML 清单，选择训练或评估分组并验证字段；把相对路径解析好后返回 DatasetEntry 列表。
display_name|返回显式的数据来源名称；没有名称时使用样本文件名，便于记录时识别来源。
scan_jsonl_offsets|逐行扫描样本文件，跳过空白行并记录每行起点和长度；返回索引，不把全部样本内容留在内存。
apply_sampling_strategy|根据 all、first、end 或 random 规则挑选样本编号；返回选中编号，百分比向上取整，随机选择受 seed 控制。
stride_for_rank|按进程编号隔位领取样本，必要时裁掉不能平分的尾部；返回本进程编号列表，样本少于进程数时保留原列表。
_open|打开只读文件并建立按需读取的内存映射；更新对象状态，已经打开时直接返回。
__getstate__|返回可传递给另一进程的索引与来源信息，不传递已打开的文件句柄。
__setstate__|接收另一进程恢复的状态，重建本进程自己的文件映射；无返回值，避免跨进程复用无效句柄。
get|按指定行号查索引，只解析这一行，再补齐图片路径与来源信息；返回一条样本。
default_modality|读取首条样本来估计该来源是否含图片并缓存；返回布尔值，这是来源级估计，不逐条检查。
open_jsonl_shard|对一个样本文件先建索引，再创建带图片目录与来源名称的读取对象；返回可按行读取的分片。
__getitem__|根据总编号定位分片和行；坏行会按固定备用顺序有限次重试，成功返回样本，全部失败则报错。
column_names|返回数据对象对外提供的字段名称，供训练工具识别样本结构。
lengths|返回每条所选样本的行字节长度，用作分组时的粗略长短估计；它不等于真正的文字单元数。
modalities|返回各样本所属来源的图文类型估计，辅助将纯文字与图文样本分组。
build_jsonl_index_dataset|接收数据来源清单，逐文件建立索引、应用抽样规则并按进程切分；返回按需读取的数据集对象。
_count_image_placeholders|只统计用户文字中的 <image> 次数，返回占位符数量，用于与真实图片数核对。
ensure_user_message|样本只有助手答案时补一条用户消息，并将图片标记放到用户侧；返回整理后的对话，已有用户消息则原样返回。
align_image_placeholders|接收对话和真实图片数，缺标记时向首条用户消息补，多标记时从后向前删；返回整理后的消息列表。
normalize_qwen_message_content|将通用 <image> 占位符换成该模型认识的图片起止标记；返回替换后的文字，非文字内容原样通过。
last_assistant_index|从后往前找最后一条助手消息，返回其位置；没有助手答案就无法建立训练目标，因此报错。
normalize_messages|逐条复制消息并替换图片标记，返回标准化消息列表，不改传入对象。
render_prompt_and_completion|用同一个对话模板分别排出问题前缀与完整对话，再取差得到目标答案；返回两段文字。
load_sample_images|按路径打开并真正读入图片像素，统一成 RGB；返回图片列表，避免以后才触发延迟读取。
encode_sample|接收一条图文标注，依次整理对话、读取图片、转数字、截断、确定回答位置和区域归属；返回单条训练输入，无法处理时返回 None。
_pixels_per_token|根据图片小块边长和合并倍数计算一个图像文字位置覆盖多少源像素；返回面积，默认例子为 (16×2)²=1024。
_truncate_keeping_whole_images|接收文字序列、长度上限和图片网格；把截断点退到完整图片边界，返回保留文字长度和保留图片数。
_assistant_spans|扫描对话起止标记，定位每一轮助手回答的内容区间；返回起止位置列表，让历史回答也参与学习。
_unpack_masks|将按位打包的正确区域还原成像素数组；返回展开结果，是训练器在设备上解包过程的 CPU 参照。
is_coco_rle|检查对象是否具有区域游程编码所需的 size 和 counts，返回真假，不执行像素解码。
split_segmentation_masks|把空值、单个区域或区域列表统一成按答案顺序排列的列表；不接受无法解释的标注类型。
_decode_compressed_rle_counts|读取 COCO 压缩字符串中的分段整数与差分，返回连续黑白段的长度列表，供解码器还原像素。
decode_coco_rle|把连续黑白段长度按列优先还原成高×宽的 0/1 数组；检查大小和计数，不要求安装 pycocotools。
segmentation_to_pixel_mask|接收标注与目标画布大小，先解码，再按最近邻缩放；多个标注取并集，返回画布上的 0/1 区域。
box_token|把四个相对坐标拼成 <bbox>…</bbox> 文字并返回；这里不负责从图片推算坐标。
mask_block|在框文字后加上区域特殊标记，返回一个实例的标准答案片段。
build_completion|按输入框的顺序拼接答案片段，可选择是否带区域标记；返回用于训练的完整助手文字。
_length_grouped_indices|先随机打乱样本编号，再在大组中按长度排序；返回新顺序，以减少同一批次的补齐浪费。
_modality_length_grouped_indices|把图文与纯文字样本分别按长度分组，再混洗完整组并合并余数；返回完整样本顺序。
rank|返回当前进程编号，优先使用显式设置；没有多进程环境时返回 0。
world_size|返回参与分配样本的进程数，优先使用显式设置；单进程默认 1。
set_epoch|接收训练轮数并更新随机种子；让不同轮次的顺序变化，同时在同一设置下可复现。
_format_count|把数量格式化成便于阅读的文字，返回展示字符串，不参与训练计算。
_format_ratio|接收分子和分母，返回可阅读的比例文字，处理分母为零的情况。
_summarize_numeric|汇总一组数值的范围等统计，返回报告所需的摘要，不执行模型预测。
_normalize_enum|读取枚举的实际值并转成文字，返回统一显示形式，便于记录训练选项。
is_eval_enabled|检查训练设置是否启用了评估，返回真假，决定是否需要准备评估数据。
_count_parameters|统计模型全部参数与允许训练的参数，返回数量摘要，用来核对冻结设置。
_build_dataset_summary|接收数据集及来源配置，汇总来源、选取数量和长度等信息，返回训练前报告的数据部分。
_build_model_summary|汇总模型类别、参数数量和可训练部分，返回模型设置说明。
_build_training_summary|汇总进程数、每批样本量、学习设置和保存设置，返回训练计划说明。
build_preflight_report|组合数据、模型和训练三个摘要，返回完整的训练前检查报告，帮助先核对再运行。
_class_name|接收对象并返回类名，用来显示实际采用的模型或处理器类型。
_hook|接收词嵌入的梯度，将不允许更新的词行清零并返回；只让指定新符号继续学习。
_disable_unneeded_gradient_checkpointing|只训练区域分支时，关闭冻结主干上无用的重算连接；调整设置以减少不必要的反向计算。
_set_requires_grad|对模块中满足条件的参数设置是否参与学习；原地修改标志，不返回预测结果。
_apply_param_trainability|先冻结全部参数，再按选项开放指定部分；检查共享词表权重的冲突，防止开放与冻结规则互相抵消。
_mean_init_new_token_embeddings|把新增特殊符号的初始表示设为原词表平均值；原地初始化新增行，并处理输入输出共享权重的情况。
_build_model|接收模型位置或已有模型及训练设置；加载模型并配置两种损失与计算方式，返回可训练模型。
_configure_image_processor|把图片像素上限和下限写入图片处理器，兼容不同设置保存形式；原地更新，不读取具体图片。
_prepare_inputs|保留样本中不规则的区域信息到 compute_loss 阶段处理，返回输入，避免过早用通用搬运逻辑处理它们。
get_train_dataloader|根据数据是否已分片选择正确的批次分配方式，并组合处理器和数据读取进程；返回训练用数据加载器。
_build_mask_inputs|将每条样本的区域位置、所属图片和压缩标注拼成规则数组；返回区域输入字典，记录有效像素以排除补齐部分。
compute_loss|把文字、图片和解包后的正确区域送入模型，取得并记录两种损失；按调用要求返回总损失或总损失与模型输出。
_to_device|把非空数组搬到当前计算设备，返回搬运结果；空值和零元素数组统一返回 None。
prediction_step|评估训练样本时复用相同的区域准备与损失计算，关闭梯度；返回损失和按需要保留的文字预测。
_distributed_metric_means|按固定顺序汇总所有进程记录的文字与区域损失，返回用于日志的平均值，避免只报告某一张卡的情况。
log|接收当前记录，加入分项损失、时间、进度与显存占用，再交给父类输出并清空临时统计。
_format_duration|把秒数转换成时分秒或分秒的可读文字并返回。
_save|把当前模型、处理器和训练设置写到输出目录，形成之后可以继续训练或预测的文件。
_normalize_resume_from_checkpoint|统一恢复训练选项中的真假值和文字写法；返回 None、True 或指定路径，消除命令行歧义。
_cli_arg_passed|检查用户是否显式传入某个选项，兼容连字符、下划线和等号写法；返回真假，供自动恢复规则判断优先级。
_resolve_auto_resume|显式恢复选项优先，否则按自动恢复设置寻找最新 checkpoint；返回恢复目标或 None，并调整跳过数据的设置。
_validate_resume_target|检查指定恢复目录是否具备所需训练状态；不满足时提前报错，避免把只有模型权重的目录误作完整恢复点。
find_model_dirs|接收模型目录或其父目录，查找同时有两个分词器设置文件的目录并返回排序列表。
token_entries|读取模型目录里的 tokenizer.json 和 tokenizer_config.json，返回两个字典供一致性检查。
inspect_model|检查旧框符号或新框符号是否成对、编号一致且标为特殊符号；返回迁移状态和编号，混杂或冲突则报错。
atomic_write|先将新内容写到同目录临时文件并落盘，再替换目标文件；不返回值，降低中断时只写一半的风险。
rewrite_model|只替换分词器中的旧框符号拼写，先检查并备份，再写回；编号和权重不变，已有备份不覆盖。
encode_rle|将 0/1 区域按列优先数连续段，返回未压缩的 COCO 游程编码，方便人读懂与检查。
per_mille_box|从区域像素求外接框并换算到 0～1000 相对坐标，返回四个整数。
grounding_block|把一个区域转成框坐标与区域标记组成的答案片段并返回，用于合成训练样本。
draw_image|按随机种子画互不重叠的简单形状，返回图片和各形状的信息与区域；几何规则直接给出正确答案。
stage1_row|从一张合成图与形状标注构建第一阶段样本，返回用于学习区域分支的图文记录。
stage2_rows|围绕同一张合成图构造指代、多个实例、计数等不同问答，返回第二阶段样本列表。
write_jsonl|把样本逐条写为一行 JSON 并保存；无预测过程，便于之后按行索引读取。
resolve_model|模型位置若已是本地目录就直接返回，否则调用下载工具取得本地目录；学习检查应传已有目录以避免下载。
resolve_dataset|接受数据文件、目录或在线数据名称，返回本地 test.parquet 位置；只有在线名称才触发下载。
predict_hf|依次准备图片、生成回答、选出最终答案区域并解码；返回文字、答案区域数及区域分数，无区域时分数为空。
score_masks|比较同尺寸的预测与标注，返回交集像素数、并集像素数及其比值；两者都空时本实现记为 1。
answer_mask_range|判断生成是否仍停留在思考部分，并核对区域数量；返回最终答案区域的起点和个数。
merge_masks|将每个区域的原始分数先平滑缩放到原图大小，再以概率 0.5 分界并取并集；返回一个布尔区域。
decode_row|从评测记录中打开内嵌图片并解码所有正确区域，取并集；返回图片与对应大小的正确区域。
summarize|对记录整体及不同领域、表达方式、目标数量分别统计，返回分组结果；内部 metrics 负责具体公式。
metrics|接收同组记录，返回数量、IoU 达到 0.5 的比例、总交集/总并集和逐例 IoU 平均；无样本返回空分数。
write_json|把结果先写到临时文件，再替换最终 JSON 文件；减少被中断时留下半份报告的风险。
prepare_output|建立输出或核对续跑设置，读取已完成记录；只允许清理中断留下的最后半行，返回已完成记录字典。
print_metrics|把整体和分组评分转成百分比并打印成表，供终端查看；不改实际保存的数值。
save_record|追加一条预测记录并刷新到磁盘，更新已完成字典和进度显示；用于可靠续跑。
pending_rows|逐条读取评测表，检查重复编号，按分片规则跳过不属于本进程或已完成的记录；依次产生待预测样本。
checkpoint_identity|读取模型设置文件的摘要和权重文件信息，返回用于比较两次运行来源是否一致的身份记录。
validate_options|核对评测参数组合是否有效，不合法时抛错，避免启动后才发现分片或计算方式冲突。
backend_config|从命令行选项整理 vLLM 运行设置并返回，供记录与子进程使用。
_visible_gpu|将 cuda 设备写法解析为实际可见 GPU 编号并返回，用于给子进程分配设备。
_prepare_views|为文字与区域两个引擎准备各自需要的模型文件视图，返回路径，供后续引擎加载。
_engine_options|接收模型路径与运行设置，组装 vLLM 引擎初始化参数并返回。
_pool_masks|将区域请求交给 vLLM 的 pooling 接口执行，返回处理后的区域结果。
_worker|在独立进程中加载文字或区域引擎，循环接收请求并发送结果；遇到异常通过连接回传错误。
decode_pooling_masks|校验服务返回的区域数量与尺寸，解码并合并区域；返回原图大小的布尔预测。
_receive|等待指定子进程的回复，同时检查退出与超时；返回成功结果，失败则抛出明确异常。
_request|更新当前运行阶段并通过进程连接发送请求，返回 _receive 收到的结果。
predict|先请求文字生成并保留原始编号，再请求最终答案区域；返回合并区域和生成记录，未生成区域时返回全空预测。
merge_results|读取各分片记录并核对设置、样本覆盖和分数一致性，再合并生成总报告；不重复模型预测。
run_parallel|为各设备启动评测子进程，等待成功后合并结果；任一子进程失败就停止并报告。
nearest_binary_union|先对原始分数取零阈值并合并，再用最近邻缩放二值区域；返回另一种后处理结果，用于与默认顺序比较。
select_indices|根据限制、数量与随机种子选择要比较的样本编号，返回确定的选取结果。
comparison_summary|汇总两种后处理方式的分数与差异，返回比较报告；不是训练或重新调整模型。
difference|对同一组的两种评分计算差值并返回，让读者看到仅改变缩放顺序带来的变化。
merge_comparison|读取比较任务的分片结果并核对来源与覆盖，写出合并报告。
run_comparison|对同一次模型输出应用两种后处理顺序，保存逐条差异与汇总；用于定位评测规则造成的差别。
_post_json|将字典编码为 JSON 并发送 HTTP POST，请求失败由网络库报错，成功返回解析后的字典。
_image_data_uri|读取图片并转成带图片类型的 Base64 字符串，返回可嵌入文字服务请求的图片内容。
call_vllm|发送图片和问题给文字服务，要求保留特殊符号与真实生成编号；返回回答文字和编号列表。
_load_image|优先读取请求中的本地图片路径，否则解码 Base64 图片；返回 RGB 图片，两者都没有就报错。
_png_b64|将像素数组按指定颜色模式存成内存 PNG，再转成文本；返回用于网络传输的 Base64 字符串。
_encode_rle|把二值区域按列优先压缩成 COCO RLE，并将字节串改成可写入 JSON 的文字；返回 size/counts 字典。
parse_data|检查请求内容必须是字典，合法就返回原对象；不是字典则报错。
pre_process|根据图片与问题构造前缀，再拼接实际生成的编号；返回服务模型输入，并按请求号暂存展示选项。
post_process|读取网格与区域数量，重建像素分数并移除思考区域；按请求缩放、二值化和编码，返回区域与可选叠加图。
make_io_processor|返回输入输出处理器类的完整名称字符串；框架会再实例化该类，这里不是直接返回对象。
get_supported_tasks|返回该区域输出器支持的 plugin 任务集合，供 vLLM 判断能否处理请求。
get_pooling_updates|要求框架在区域处理时保留文字编号；返回任务设置，因为必须知道哪些位置是区域特殊符号。
load_weights|按参数名分开基础模型权重与区域解码器权重，分别加载；返回已加载名称集合。
_reorder_pre_merge|将按合并块排列的图片特征恢复为从上到下、从左到右的顺序；返回按空间位置排列的二维特征。
split_captured_vision|按每个请求的网格切分刚收集的图片特征；返回特征与网格列表，数量或维数不符时返回空占位。
decode_mask_batch|定位各请求的区域符号，把相同网格的请求分组一次解码；返回每请求带网格和数量头部的扁平输出。
_grab_grid|视觉模块开始计算前读取网格参数，保存在模型对象中供区域解码使用；返回 None，不替换原输入。
_grab_pre_merge|在图片小块合并前保存其特征，供区域分支恢复细节；返回 None，不更改主干输入。
register|向 vLLM 的模型注册表登记本项目模型名称与位置；无返回值，之后工作进程可据此加载。
_send|将状态码和字典写成 HTTP 响应，设置长度与内容类型；无返回值。
do_GET|处理健康检查路径 /health，返回服务状态；其余路径返回不存在。
do_POST|只处理 /decode_mask 请求，读取 JSON 并执行区域解码；成功发回结果，异常转成错误响应。
log_message|覆盖服务器默认的访问日志方法，不输出日志，避免重复终端记录。
keep_key|判断某个权重是否应保留给文字服务；区域权重被排除，只有共享词嵌入时才去除重复输出层权重。
_corrected_tokenizer_config|读取分词器设置并修正服务兼容项，返回新字典；不直接覆盖来源文件。
_rewrite_shard|从一份权重分片挑出文字服务需要的参数并写到目标位置；返回保留与删除的参数项数。
_tie_word_embeddings|从模型设置读取输入词表与输出层是否共用权重，返回真假，用于避免错误删除非共享输出层。
prepare|接收源模型和独立输出目录，生成适合服务加载的文件；返回输出目录，文字模式可重写分片，区域模式保留所有权重。
''')
# 同名方法必须按所属类区分，不能仅按函数名套用说明。
add('''
MLP|把输入表示经过两层线性变换与中间非线性变换，转换成所需宽度；用于生成区域查询。
LayerNorm2d|在每个图片位置对通道做归一化，让不同特征数值尺度较稳定，保持高宽不变。
PatchSelfAttention|让图片的小块彼此交流信息的一层模块；没有区域查询到图片的交叉注意力，也不负责放大图片。
_ChunkedBilinearInterpolate4x|提供高宽四倍插值的自定义反向计算；分块避免 CUDA 单次启动规模过大。
PixelMaskDecoder|把文字中的目标表示与图片特征变成区域分数图；每个区域标记对应一张分数图。
DiceLoss|衡量预测区域与正确区域的整体重叠差距；分数越小表示越接近正确区域。
BCEWithLogitsLoss|逐像素检查前景与背景的预测差距，直接接收原始分数，内部才转成概率。
MaskLoss|组合整体重叠与逐像素两种损失，支持有效区域和分块计算。
CausalLMLossMixin|给主模型补充文字评分功能的辅助类；依赖主模型已有的词表输出层与设置。
XiaomiLensOutput|整理一次模型计算的返回值，包括文字分数、区域分数、有效像素和两类训练损失。
XiaomiLensQwen3_5ForConditionalGeneration|继承基础图文模型并接入区域分支，统一文字生成、像素预测与联合训练。
LazyCollator|在取到一批样本后才处理图片与文字，并补齐长度；不在启动时一次处理完整数据集。
DatasetEntry|记录一个数据来源的样本文件、图片目录、名称、标签和抽样规则，不存放图片像素。
JsonlIndex|保存非空样本行的字节起点与长度，以及源文件大小和修改时间。
JsonlIndexShard|管理一份样本文件的只读映射与索引，提供按行读取及关闭操作。
JsonlIndexDataset|把多个样本文件拼成统一编号空间，并为损坏记录提供有限重试。
BatchSamplerShard|负责每个进程每一批取得哪些样本编号，支持打乱与长短分组。
_FreezeNonTrainableEmbedRows|为词嵌入挂接梯度处理，只允许指定新符号行更新。
XiaomiLensTrainer|扩展通用训练器，把图文模型与区域标注处理接入原有训练循环。
XiaomiLensTrainingConfig|保存损失、数据加载、保存方式等训练选项，继承通用训练器的其他选项。
XiaomiLensScriptArguments|保存数据路径、图片大小等脚本级选项，供数据构建阶段读取。
XiaomiLensModelConfig|保存模型位置、计算精度与哪些部分允许训练等模型选项。
VLLMBackend|管理文字与区域两个独立工作进程，串联请求、错误处理和退出清理。
XiaomiLensMaskIOProcessor|在外部请求和 vLLM 的区域模型之间转换输入输出，暂存每个请求的展示选项。
XiaomiLensMaskPooler|从文字状态中取出区域查询，并把对应图像特征交给区域解码器，返回逐请求输出。
XiaomiLensForVLLM|扩展 vLLM 版 Qwen 模型，添加区域解码器与收集图片特征的回调。
Handler|接收 HTTP 请求并调用区域解码函数，组织成功或失败响应。
MLP.forward|接收最后一维为输入宽度的数组，依次做线性变换、GELU 和线性变换，返回目标宽度的表示。
LayerNorm2d.forward|接收 B×C×H×W 特征，按通道计算均值与方差并归一化；返回同尺寸特征。
PatchSelfAttention.forward|接收 B×P×D 小块序列和可选补齐标记，执行归一化、注意力与前馈变换并加回原输入；返回同尺寸序列。
_ChunkedBilinearInterpolate4x.forward|接收区域分数并放大四倍，同时保存输入尺寸与设备信息，供反向传播还原对应关系。
_ChunkedBilinearInterpolate4x.backward|接收放大后的梯度，按批次与通道分块计算输入梯度；返回原始尺寸的梯度。
PixelMaskDecoder.forward|接收 B×K×D 的区域查询和 B×(h·w)×V 的图片特征；输出 B×K×(16h)×(16w) 原始像素分数。
DiceLoss.forward|接收原始分数、0/1 标注和有效范围；内部转概率后计算 1－(2倍重叠+1)/(预测面积+标注面积+1)，返回归一化损失。
BCEWithLogitsLoss.forward|接收原始分数与标签，先按每个区域的有效像素数平均，再按区域数归一化；返回逐像素分类损失。
MaskLoss.forward|核对预测、标注和有效像素形状，按区域分小批计算两种损失并求和；返回一个可反向传播的总分。
XiaomiLensQwen3_5ForConditionalGeneration.forward|接收图文输入及可选正确答案；先计算共享图文状态，再生成文字和区域分数，训练时返回两类加权损失。
XiaomiLensMaskPooler.forward|按每个请求的文字长度拆开状态与编号，配上捕获的图片特征，再返回区域解码结果列表。
LazyCollator.__call__|接收原始样本列表，逐条编码、跳过失败记录并补齐成功样本；返回批次字典，全失败时返回零损失占位批次。
JsonlIndex.__len__|返回索引中非空样本行的数量，供 len(index) 使用。
JsonlIndexShard.__len__|返回这一份样本文件的有效行数，转交给底层索引统计。
JsonlIndexDataset.__len__|返回抽样和分片后当前数据集的样本数量，不一定等于原文件行数总和。
BatchSamplerShard.__len__|根据本进程样本量、批次大小与是否丢尾返回批次数，便于计算训练进度。
JsonlIndexDataset.__iter__|按当前数据集编号逐个产生样本，实际读取与重试都交给 __getitem__。
BatchSamplerShard.__iter__|使用本轮种子生成顺序、领取当前进程份额，再逐批产生编号列表；按设置保留或丢掉最后不满一批的部分。
JsonlIndexShard.close|释放只读映射与文件描述符并清空状态，可重复调用；不修改样本文件。
JsonlIndexShard.__del__|对象销毁时调用 close 释放文件资源，不执行数据写入。
VLLMBackend.close|请求子进程退出并等待，必要时结束其进程组，然后关闭连接；防止评测结束后留下引擎占用显存。
''')
INIT={
'MLP':'接收输入、中间和输出宽度，创建两层线性变换与激活函数。',
'LayerNorm2d':'接收通道数和防止除零的小量，创建每通道可学习的缩放与偏置。',
'PatchSelfAttention':'接收宽度、头数和中间宽度，创建归一化、注意力与前馈层。',
'PixelMaskDecoder':'读取文字和视觉宽度，创建查询变换、小块注意力、降通道和两次放大层。',
'MaskLoss':'保存两种评分权重和分块大小，创建评分模块；默认重叠权重 1、逐像素权重 2。',
'XiaomiLensQwen3_5ForConditionalGeneration':'先构造基础 Qwen 图文模型，再创建区域解码器和区域损失函数。',
'LazyCollator':'保存处理器、区域符号编号和长度像素限制，初始化错误提示计数。',
'JsonlIndexShard':'保存索引与图片目录等来源信息，并打开当前进程的只读文件映射。',
'JsonlIndexDataset':'建立总编号到分片行号的对应表，保存抽样结果，并生成遇到坏行时的备用读取顺序。',
'BatchSamplerShard':'保存批次与进程设置，检查分组参数，计算当前进程负责的样本量。',
'_FreezeNonTrainableEmbedRows':'根据允许学习的词编号构造梯度行掩码，并登记反向传播回调。',
'XiaomiLensTrainer':'加载处理器与模型、添加区域符号、设置可训练部分，再构建训练/评估数据和拼批处理器。',
'VLLMBackend':'准备模型视图并启动文字、区域两个进程，核对两者的区域符号编号一致；失败则清理。',
'XiaomiLensMaskIOProcessor':'保存服务提供的分词器，并创建按请求号记录输出选项的字典。',
'XiaomiLensMaskPooler':'保存所属模型的引用，避免将它再次登记成自己的子模块形成循环。',
'XiaomiLensForVLLM':'先创建基础模型，再创建区域解码器、取得区域符号编号并登记图片特征捕获回调。',
}
MAIN={
'infer.py':'读取图片和选项，加载模型，先生成回答再解码最终答案区域，按设置保存合成图或每区域单图；没有区域就直接结束。',
 'train.py':'解析三类选项、设定随机种子、创建训练器，打印检查报告后启动或恢复训练，最后保存模型。',
 'eval/run.py':'选择单进程、并行或比较模式，逐样本生成并评分，保存记录和分组报告；失败时保存出错阶段。',
 'examples/train/make_toy_data.py':'读取输出位置和样本规模，按固定种子画图并写出两个阶段的训练文件与清单。',
 'tools/prepare_base_model.py':'查找模型目录，先检查全部目标；指定写入时备份并重命名框符号，最后重新检查并打印报告。',
 'src/xiaomi_lens/integrations/vllm/client.py':'读取图片与查询，先调用文字服务再调用区域服务，解码返回的图片并保存。',
 'src/xiaomi_lens/integrations/vllm/mask_service.py':'读取模型与监听地址，加载一次模型，启动服务器等待区域解码请求。',
 'src/xiaomi_lens/integrations/vllm/prepare_served.py':'读取来源与独立输出目录，按文字或区域模式准备服务视图，返回成功退出状态。',
}
OVERRIDE={('src/xiaomi_lens/integrations/vllm/mask_service.py','decode_mask'):'读取图片、问题和已生成文字，使用已加载模型重新计算区域；返回区域 PNG、框和可选叠加图，无答案区域时返回空列表。',('src/xiaomi_lens/integrations/vllm/vllm_model.py','decode_mask'):'把单个请求包装成批次接口需要的列表，再取回单请求结果；保持旧调用方式可用。'}
# 每个函数都必须有经人工编写的说明；不使用自动生成的空泛兜底。
coverage=[]
for p in sorted(ROOT.rglob('*.py')):
 rel=p.relative_to(ROOT).as_posix(); stage,desc,route=FILES[rel]
 source=p.read_text();tree=ast.parse(source);lines=source.splitlines(keepends=True);insert={}
 # 放在版权段之后、原文档字符串之前，保持 __doc__ 与 future imports 不变。
 at=next((i for i,l in enumerate(lines) if l.startswith('"""') or l.startswith('from ') or l.startswith('import ')),len(lines))
 insert.setdefault(at,[]).extend([f'# 【中文导读｜阅读第 {stage} 步】\n',f'# 本文件：{desc}\n',f'# 怎么读：{route}\n','# 完整路线、目录地图和概念解释见项目根目录「学习指南.md」。\n','\n'])
 def walk(node,parent=''):
  for n in ast.iter_child_nodes(node):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    key=f'{parent}.{n.name}' if parent else n.name
    if isinstance(n,ast.ClassDef): note=D[n.name]; label='类的用途'
    elif n.name=='__init__': note=INIT[parent]+' 构造完成后通过对象保存这些状态，不返回预测结果。';label='步骤说明'
    elif n.name=='main': note=MAIN[rel];label='运行入口'
    elif n.name=='parse_args': note='读取命令行提供的文件位置、运行选项和默认值，返回解析后的选项对象；本函数不执行'+('训练。' if rel=='train.py' else '模型预测。');label='步骤说明'
    else:
     note=OVERRIDE.get((rel,n.name),D.get(key,D.get(n.name)))
     if note is None: raise RuntimeError((rel,key))
     label='步骤说明'
    lineno=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
    indent=' '*n.col_offset
    block=[f'{indent}# 【{label}】{note}\n']
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
     args=[a.arg for a in n.args.posonlyargs+n.args.args+n.args.kwonlyargs if a.arg not in ('self','cls')]
     if args:block.append(f'{indent}# 对照参数：'+', '.join(args)+'；下面保留原始英文说明与类型，方便对应调用处。\n')
    insert.setdefault(lineno,[]).extend(block)
    coverage.append({'file':rel,'symbol':key,'explanation':note})
    walk(n,n.name if isinstance(n,ast.ClassDef) else parent)
   else:walk(n,parent)
 walk(tree)
 p.write_text(''.join(''.join(insert.get(i,[]))+line for i,line in enumerate(lines))+''.join(insert.get(len(lines),[])))
Path('.learning-work/coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2))
Path('.learning-work/files.json').write_text(json.dumps(FILES,ensure_ascii=False,indent=2))
print('Annotated',len(FILES),'files;',len(coverage),'classes/functions')
