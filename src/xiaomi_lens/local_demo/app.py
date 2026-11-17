"""中文Gradio页面：独立样本浏览、自定义推理、上传与批量结果。"""
import json,uuid,os,time
from html import escape
from pathlib import Path
import gradio as gr
from PIL import Image
from .service import ROOT,DATA,OUT,SERVICE,PROMPT,bundle
from .boxes import presentation,view_path
from .geopage import add_page,box_text
from ..train.data.domain import jsonlines,NAMES

def result_notes(result, interactive=False):
 parts=[]
 if 'iou' in result:parts.append(f"区域重合：{result['iou']:.1%}")
 if result.get('consistency_issues'):parts.append('需要核对：'+'；'.join(result['consistency_issues']))
 else:parts.append('未发现规则能识别的文字与区域矛盾。')
 if result.get('truncated'):parts.append('回答达到长度上限，可能尚未说完。')
 if interactive:parts.append('本次交互结果单独保存，不计入固定批量成绩。')
 return '\n'.join(parts)

def result_table(headers, rows):
 """静态只读表避免大量结果在浏览器表格编辑器内反复布局。"""
 head=''.join('<th>'+escape(str(x))+'</th>' for x in headers)
 body=''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)
 return '<div style="max-height:480px;overflow:auto"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'

def make_app():
 rows=jsonlines(DATA/'samples.jsonl'); byid={r['id']:r for r in rows};ids=list(byid)
 recent_results={}
 def cached(signature):
  entry=recent_results.get(signature)
  return entry[1] if entry and time.monotonic()-entry[0]<15 else None
 def remember(signature,value):
  now=time.monotonic()
  for key in list(recent_results):
   if now-recent_results[key][0]>=15:del recent_results[key]
  recent_results[signature]=(now,value)
  return value
 def choose(source):
  choices=[r['id'] for r in rows if source=='全部' or r['source']==source]
  return gr.update(choices=choices,value=choices[0])
 def move(source,key,offset):
  keys=[r['id'] for r in rows if source=='全部' or r['source']==source]
  return keys[(keys.index(key)+offset)%len(keys)]
 def display(key):
  r=byid[key];folder=OUT/'batch'/key
  result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
  vf=OUT/'localization'/key
  if result and not (vf/'boxes.json').exists():presentation(ROOT/r['images'][-1],folder,vf)
  coordinates=box_text(json.loads((vf/'boxes.json').read_text())) if result else ''
  return (str(ROOT/r['images'][-1]),[str(ROOT/p) for p in r['images'][:-1]],str(ROOT/r['mask']),
   str(folder/'overlay.png') if result else None,str(folder/'mask.png') if result else None,
   result['answer'] if result else '尚无批量结果，可点击推理。',result['derived_description'] if result else '',
   result_notes(result) if result else '',
   str(folder/'result.zip') if (folder/'result.zip').exists() else None,str(vf) if result else '',coordinates,str(vf/'localization.zip') if result else None)
 def check_batch():
  status=OUT/'status.json';pidfile=OUT/'batch-process.json'
  if status.exists() and pidfile.exists() and json.loads(status.read_text()).get('state')=='running':
   try:os.kill(json.loads(pidfile.read_text())['pid'],0)
   except ProcessLookupError:return
   raise gr.Error('批量检查仍在运行，请稍后再做交互预测；已完成样本可直接浏览。')
 def infer_sample(key,prompt,request:gr.Request):
  if not prompt.strip():raise gr.Error('请填写问题。')
  signature=('sample',request.session_hash or uuid.uuid4().hex,key,prompt)
  previous=cached(signature)
  if previous:return previous
  check_batch()
  folder=OUT/'interactive'/uuid.uuid4().hex
  result=SERVICE.run(byid[key],folder,prompt)
  vf=folder/'localization';coords=presentation(ROOT/byid[key]['images'][-1],folder,vf)
  return remember(signature,(str(folder/'overlay.png'),str(folder/'mask.png'),result['answer'],result['derived_description'],result_notes(result,True),bundle(folder),str(vf),box_text(coords),str(vf/'localization.zip')))
 def upload(optical,mode,aux1,aux2,prompt,request:gr.Request):
  if not optical:raise gr.Error('请上传光学图片。')
  if not prompt.strip():raise gr.Error('请填写问题。')
  required={'仅光学':0,'光学＋近红外/短波红外':1,'光学＋InSAR＋高程':2}[mode]
  supplied=[aux1,aux2]
  if any(not x for x in supplied[:required]) or any(x for x in supplied[required:]):raise gr.Error('辅助图片数量与所选输入类型不符，请补齐或清空。')
  files=supplied[:required]+[optical];sizes=[]
  for p in files:
   if Path(p).suffix.lower() not in ['.png','.jpg','.jpeg','.webp']:raise gr.Error('仅支持已经制作好的PNG、JPEG或WebP图片。')
   try:
    with Image.open(p) as im:
     if im.mode not in ['RGB','RGBA','L']:raise ValueError('非普通显示图片')
     im.verify()
    with Image.open(p) as im:sizes.append(im.size)
   except Exception:raise gr.Error('图片无法读取，或不是普通显示图片。')
  if len(set(sizes))!=1:raise gr.Error('辅助图片必须与光学图片尺寸相同，并由你确认已对齐。')
  signature=('upload',request.session_hash or uuid.uuid4().hex,optical,mode,aux1,aux2,prompt)
  previous=cached(signature)
  if previous:return previous
  check_batch()
  folder=OUT/'interactive'/uuid.uuid4().hex;folder.mkdir(parents=True)
  copied=[]
  for i,p in enumerate(files):
   dest=folder/f'input_{i}.png'
   with Image.open(p) as im:im.convert('RGB').save(dest)
   copied.append(str(dest))
  result=SERVICE.run({'id':'upload','images':copied},folder,prompt)
  vf=folder/'localization';coords=presentation(copied[-1],folder,vf)
  return remember(signature,(str(folder/'overlay.png'),str(folder/'mask.png'),result['answer'],result['derived_description'],result_notes(result,True),bundle(folder),str(vf),box_text(coords),str(vf/'localization.zip')))
 def batch_results():
  summary=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {}
  records=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
  failures=json.loads((OUT/'failures.json').read_text()) if (OUT/'failures.json').exists() else []
  names={'GDCLD':'高分辨率光学','landslide4sense':'多光谱','multimodal-landslide-dataset':'InSAR与光学多模态','all':'总体'}
  summary_rows=[[names.get(k,k),v['count'],round(v['mean_iou']*100,2) if v['mean_iou'] is not None else None,v['missed_pixels'],v['extra_pixels'],v['text_region_conflicts'],v['truncated'],round(v['mean_seconds'],2) if v['mean_seconds'] is not None else None] for k,v in summary.items()]
  return result_table(['来源','样本数','区域重合（%）','漏画像素','多画像素','文字区域不一致','回答截断','平均耗时（秒）'],summary_rows),result_table(['样本编号','来源','区域重合','回答截断','文字区域冲突'],[[r['id'],r['source'],round(r['iou'],4),'是' if r['truncated'] else '否','是' if r['text_mask_conflict'] else '否'] for r in records]),result_table(['失败样本','原因'],[[r['id'],r['error']] for r in failures]),str(OUT/'all_results.zip') if (OUT/'all_results.zip').exists() else None
 with gr.Blocks(title='滑坡识别与描述') as app:
  gr.Markdown('# 滑坡识别与描述\n使用第10轮最佳微调模型。标准标注仅供对照，模型只接收图片和问题。\n独立测试集只含有滑坡的图片；不能据此判断无滑坡场景的误报。')
  with gr.Tab('样本测试'):
   with gr.Row():
    source=gr.Dropdown(['全部']+NAMES,value='全部',label='数据来源')
    key=gr.Dropdown(ids,value=ids[0],label='样本编号（可输入搜索）',filterable=True)
    prev=gr.Button('上一条');nxt=gr.Button('下一条')
   with gr.Row():
    with gr.Column():
     optical=gr.Image(label='光学图片',interactive=False)
     aux=gr.Gallery(label='辅助图片：近红外/短波红外，或依次InSAR、高程',columns=2)
     prompt=gr.Textbox(value=PROMPT,label='问题',lines=5)
     run=gr.Button('开始推理',variant='primary')
     with gr.Accordion('标准标注（只用于对照）',open=False):truth=gr.Image(label='标准区域',interactive=False)
    with gr.Column():
     overlay=gr.Image(label='预测叠加图',interactive=False);mask=gr.Image(label='预测区域',interactive=False)
     answer=gr.Textbox(label='模型原始回答',lines=5);derived=gr.Textbox(label='按预测区域计算的说明');score=gr.Textbox(label='本次结果');download=gr.File(label='下载回答与图片')
   vf=gr.State('');sm=gr.Checkbox(value=True,label='显示分割区域');sb=gr.Checkbox(value=True,label='显示定位框')
   coords=gr.Textbox(label='定位框原图位置',lines=4);boxdl=gr.File(label='下载定位框与图片')
   for change in [vf,sm,sb]:change.change(view_path,[vf,sm,sb],[overlay])
   outputs=[optical,aux,truth,overlay,mask,answer,derived,score,download,vf,coords,boxdl]
   source.change(choose,[source],[key]);key.change(display,[key],outputs)
   prev.click(lambda s,k:move(s,k,-1),[source,key],[key]);nxt.click(lambda s,k:move(s,k,1),[source,key],[key])
   run.click(infer_sample,[key,prompt],[overlay,mask,answer,derived,score,download,vf,coords,boxdl],concurrency_limit=1,concurrency_id='model',trigger_mode='once')
   app.load(display,[key],outputs)
  with gr.Tab('上传测试'):
   gr.Markdown('TIF请使用旁边的“TIF整图与地理坐标”页面。上传已制作好的显示图片。辅助图须与光学图同区域、同尺寸；程序不猜测原始遥感波段。')
   mode=gr.Radio(['仅光学','光学＋近红外/短波红外','光学＋InSAR＋高程'],value='仅光学',label='输入类型')
   opt=gr.File(elem_id='upload-optical',label='光学图片',file_types=['.png','.jpg','.jpeg','.webp'])
   a1=gr.File(elem_id='upload-aux1',label='辅助图1：近红外/短波红外，或InSAR',file_types=['.png','.jpg','.jpeg','.webp'])
   a2=gr.File(elem_id='upload-aux2',label='辅助图2：高程',file_types=['.png','.jpg','.jpeg','.webp'])
   q=gr.Textbox(value=PROMPT,label='上传图片的问题',lines=4);button=gr.Button('预测上传图片',variant='primary')
   ov=gr.Image(label='上传预测叠加图');ma=gr.Image(label='上传预测区域');an=gr.Textbox(label='上传图片的回答');de=gr.Textbox(label='按上传预测区域计算');un=gr.Textbox(label='上传结果核对提示');dl=gr.File(label='下载上传结果')
   uvf=gr.State('');um=gr.Checkbox(value=True,label='上传显示分割区域');ub=gr.Checkbox(value=True,label='上传显示定位框')
   uc=gr.Textbox(label='上传定位框原图位置');ud=gr.File(label='下载上传定位框')
   for change in [uvf,um,ub]:change.change(view_path,[uvf,um,ub],[ov])
   button.click(upload,[opt,mode,a1,a2,q],[ov,ma,an,de,un,dl,uvf,uc,ud],concurrency_limit=1,concurrency_id='model',trigger_mode='once')
  add_page()
  with gr.Tab('批量结果'):
   refresh=gr.Button('刷新批量结果');summary=gr.HTML(label='分来源与总体成绩')
   table=gr.HTML(label='逐条结果')
   failed=gr.HTML(label='失败样本（空表表示没有失败）');allzip=gr.File(label='下载全部结果')
   refresh.click(batch_results,outputs=[summary,table,failed,allzip]);app.load(batch_results,outputs=[summary,table,failed,allzip])
 return app

def main():
 make_app().queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=7860,share=False,allowed_paths=[str(DATA),str(OUT)],theme=gr.themes.Citrus())
if __name__=='__main__':main()
