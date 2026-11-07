import sys,json,hashlib,importlib.util,random,os
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(root),str(root/'src')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['XIAOMI_LENS_MASK_ATTN']='sdpa'
import numpy as np
import torch
from PIL import Image
import infer
from xiaomi_lens.model.pixel_mask_decoder import PixelMaskDecoder
from xiaomi_lens.model.losses import MaskLoss
from xiaomi_lens.train.data.segmentation import decode_coco_rle,segmentation_to_pixel_mask
from xiaomi_lens.train.data.io import load_dataset_config
from xiaomi_lens.train.data.jsonl_index import build_jsonl_index_dataset,apply_sampling_strategy
from xiaomi_lens.train.collator import LazyCollator
from xiaomi_lens.train.data.preprocess import encode_sample

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def digest(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
toy=module('toy',root/'examples/train/make_toy_data.py');ev=module('evaluation',root/'eval/run.py')
results={}
torch.manual_seed(123)
model=PixelMaskDecoder(dict(llm_hidden_size=16,llm_intermediate_size=32,llm_num_heads=4,vision_hidden_size=12,mask_decoder_channels=8,mask_decoder_embed_dim=4))
q=torch.randn(1,2,16,requires_grad=True);v=torch.randn(1,6,12,requires_grad=True)
pred=model(q,v,2,3);assert pred.shape==(1,2,32,48)
target=torch.zeros_like(pred);target[:,:,8:24,10:30]=1
loss=MaskLoss()(pred.reshape(2,1,32,48),target.reshape(2,1,32,48));loss.backward()
assert torch.isfinite(loss) and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0 and v.grad.abs().sum()>0
results['tiny_decoder']={'shape':list(pred.shape),'sha256':digest(pred.detach()),'loss':loss.item(),'q_gradient':digest(q.grad),'image_gradient':digest(v.grad)}
try:model(q,v,3,3)
except ValueError:results['bad_grid_rejected']=True
else:raise AssertionError('bad grid not rejected')
zero=MaskLoss()(pred.detach().reshape(2,1,32,48),target.reshape(2,1,32,48),torch.zeros(2,1,32,48));assert zero==0
results['zero_valid_loss']=zero.item()
mask=np.zeros((7,9),np.uint8);mask[1:6,2:8]=1
assert np.array_equal(decode_coco_rle(toy.encode_rle(mask)),mask)
compressed={'size':[2,2],'counts':'01100'}
assert np.array_equal(decode_coco_rle(compressed),np.array([[1,1],[0,0]],dtype=np.uint8))
results['rle_roundtrip']=True
image,objs=toy.draw_image(random.Random(7),(96,64));results['toy_image']=digest(image)
overlay=infer.visualize_pixel_mask(image,pred.detach(),torch.tensor([[1,2,3]]));overlay.save(out/'tiny_overlay.png');assert overlay.size==(48,32)
results['overlay']=digest(overlay)
heat=infer.visualize_pixel_mask(image,pred.detach(),torch.tensor([[1,2,3]]),threshold=0);results['heat_overlay']=digest(heat)
assert ev.score_masks(mask,mask)['iou']==1
assert ev.score_masks(mask,1-mask)['iou']==0
assert ev.score_masks(np.zeros_like(mask),np.zeros_like(mask))['iou']==1
assert ev.answer_mask_range('<think><|mask|>',1)==(1,0)
assert infer.parse_answer_bboxes('<think><bbox>[0,0,1,1]</bbox></think><bbox>[2,3,4,5]</bbox>')==[[2,3,4,5]]
results['edge_cases']=True
entries=load_dataset_config(str(root/'examples/train/sample/sample.yaml'))
dataset=build_jsonl_index_dataset(entries);assert len(dataset)==2
sample=dataset[0];results['real_sample']={'messages':sample['messages'],'images_count':len(sample['images'])}
checkpoint=Path('/home/yukun80/codes/Xiaomi-Lens-0/checkpoints/Xiaomi-Lens-0')
from transformers import AutoProcessor
processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
mask_id=processor.tokenizer.convert_tokens_to_ids('<|mask|>')
encoded=encode_sample(sample,processor=processor,mask_token_id=mask_id,model_max_length=4096,max_pixels=262144,min_pixels=65536)
assert encoded is not None and len(encoded['mask_positions'])==1
results['encoded']={k: digest(v) for k,v in encoded.items() if isinstance(v,np.ndarray)}
results['packed_targets']=[hashlib.sha256(v).hexdigest() for v in encoded['mask_pixel_masks']]
collator=LazyCollator(processor,mask_token_id=mask_id,model_max_length=4096,max_pixels=262144,min_pixels=65536)
batch=collator([dataset[0],dataset[1]]);assert batch['input_ids'].shape[0]==2
results['batch']={k:digest(v) for k,v in batch.items() if isinstance(v,torch.Tensor)}
bad=dict(sample,images=['/nonexistent/learning-check-image.png'])
dummy=collator([bad]);assert (dummy['labels']==-100).all();results['invalid_image_dummy']=True
text={'messages':[{'role':'user','content':'Hello'},{'role':'assistant','content':'Hi'}],'images':[]}
txt=encode_sample(text,processor=processor,mask_token_id=mask_id,model_max_length=512);assert txt is not None and len(txt['mask_positions'])==0
results['pure_text']=True
(out/'component_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print('COMPONENT_CHECKS_PASS',flush=True)
# 同一真实图片、同一模型、同一生成设置，分别运行原始版与注释版。
del model,pred,loss,q,v,batch
full,processor,mask_id=infer.load_model_and_processor(checkpoint,torch.device('cuda'),attn_implementation='sdpa')
processor.image_processor.size={'shortest_edge':65536,'longest_edge':262144}
image=Image.open(sample['images'][0]).convert('RGB')
query=next(m['content'] for m in sample['messages'] if m['role']=='user').replace('<image>','').strip()
inputs,prompt=infer.build_inputs(processor,image,query,torch.device('cuda'))
torch.manual_seed(123)
ids,completion,count=infer.generate_completion(full,processor,inputs,512,mask_id)
answer,thought=infer.split_answer_content(completion)
r={'query':query,'completion':completion,'ids':ids.cpu().tolist(),'total_masks':count,'thought_masks':thought,'grid':inputs['image_grid_thw'].cpu().tolist()}
if count:
 logits=infer.run_forward(full,inputs,ids,mask_id,torch.device('cuda'));torch.save(logits.cpu(),out/'real_logits.pt')
 r['logits_sha256']=digest(logits.float().cpu());r['shape']=list(logits.shape)
 if count>thought:
  pic=infer.visualize_pixel_mask(image,logits[:,thought:],inputs['image_grid_thw']);pic.save(out/'real_overlay.png');r['overlay_sha256']=digest(pic)
(out/'real_results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print('REAL_CHECK_PASS',count,flush=True)
