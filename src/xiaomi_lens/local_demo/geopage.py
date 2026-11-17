"""TIF整图页面：预览后运行、进度、停止恢复、逐块回答与坐标下载。"""
import json
from pathlib import Path
import gradio as gr
from . import geojob
from .boxes import view_path
from .service import PROMPT


def box_text(record):
    lines=[]
    for b in record['boxes']:
        if not b['valid']:lines.append(f"框{b['number']}：{b.get('error','无效')}");continue
        lines.append(f"框{b['number']} 原图左右上下边界（左、上、右、下）："+', '.join(f'{v:.3f}' for v in b['pixel_xyxy']))
        if 'native_corners_xy' in b:lines.append('地图四角（左上、右上、右下、左下，单位 '+str(b['native_units'])+'）：'+str(b['native_corners_xy']))
        if 'wgs84_corners_lonlat' in b:lines.append('经纬度四角（经度在前）：'+str(b['wgs84_corners_lonlat']))
        if b.get('geographic_status') and b['geographic_status']!='成功':lines.append(b['geographic_status'])
    return '\n'.join(lines+record.get('warnings',[])) or '没有定位框'


def add_page():
    def status_text(job):
        job=geojob.resolve_job(job);s=json.loads((job/'summary.json').read_text());m=json.loads((job/'job.json').read_text())['inputs'][-1]['meta']
        states={'prepared':'等待开始','running':'运行中','stopped':'已停止，可继续','complete':'全部完成','completed_with_failures':'存在失败块，可重试'}
        return f"原图：{m['width']}×{m['height']}；坐标：{m.get('crs_name') or '无地理信息'}\n{states.get(s['state'],s['state'])}：{s['completed']}/{s['total']} 块；失败 {len(s['failed'])} 块\n预测覆盖 {s['pixel_counts'][str(1)]} 像素，未完成 {s['pixel_counts'][str(254)]} 像素。\n灰蓝表示未完成或失败，暗灰表示无效；框数量不等于滑坡数量。\n"+ ('失败详情：'+str(s['failed']) if s['failed'] else '')+'\n'+'\n'.join(s['box_warnings']+list(set(s['geographic_warnings'])))
    def show(job,sm,sb):
        p=geojob.resolve_job(job);st=json.loads((p/'status.json').read_text());choices=list(st['completed'])
        archive=p/'geographic_results.zip'
        return (str(p/'source_preview.png'),view_path(p,sm,sb),str(p/'mask_preview.png'),status_text(p),box_text(json.loads((p/'boxes.json').read_text())),gr.update(choices=choices,value=choices[0] if choices else None),str(archive) if archive.exists() and st['state']!='running' else None)
    def prepare(opt,mode,a1,a2,aligned,prompt,sm,sb):
        required={'仅光学':0,'光学＋近红外/短波红外':1,'光学＋InSAR＋高程':2}[mode]
        aux=[a1,a2]
        if not opt:raise gr.Error('请上传彩色TIF光学图')
        if any(not p for p in aux[:required]) or any(aux[required:]):raise gr.Error('辅助图片数量与输入类型不符')
        try:job=geojob.prepare(aux[:required]+[opt],prompt,aligned)
        except Exception as e:raise gr.Error(str(e))
        return (job,Path(job).name,*show(job,sm,sb))
    def resume(job,prompt,sm,sb):
        try:
            p=geojob.resolve_job(job)
            if json.loads((p/'job.json').read_text())['prompt']!=prompt:raise ValueError('问题改变，请重新准备新任务；继续任务必须使用原问题')
            for _ in geojob.run(job):yield show(job,sm,sb)
        except Exception as e:raise gr.Error(str(e))
    def restore(name,sm,sb):
        try:
            p=geojob.resolve_job(geojob.JOBS/name)
            if Path(name).name!=name:raise ValueError('只填写任务编号')
            return str(p),json.loads((p/'job.json').read_text())['prompt'],*show(p,sm,sb)
        except Exception as e:raise gr.Error(str(e))
    def detail(job,tile):
        if not job or not tile:return None,'',''
        if len(tile)!=6 or not tile.isdigit():raise gr.Error('无效分块编号')
        p=geojob.resolve_job(job)/'tiles'/tile
        r=json.loads((p/'result.json').read_text());b=json.loads((p/'boxes.json').read_text())
        return str(p/'overlay.png') if (p/'overlay.png').exists() else str(p/'mask.png'),r['answer'],box_text(b)
    with gr.Tab('TIF整图与地理坐标'):
        gr.Markdown('已配色的8位RGB/RGBA TIF。先准备预览，再开始扫描。默认512像素分块、重叠128像素；未完成区域不会当作背景。辅助图须事先对齐。')
        job=gr.State('')
        opt=gr.File(label='彩色TIF光学图',file_types=['.tif','.tiff'],elem_id='geo-optical')
        mode=gr.Radio(['仅光学','光学＋近红外/短波红外','光学＋InSAR＋高程'],value='仅光学',label='TIF输入类型')
        a1=gr.File(label='TIF辅助图1',file_types=['.png','.jpg','.jpeg','.webp','.tif','.tiff'])
        a2=gr.File(label='TIF辅助图2',file_types=['.png','.jpg','.jpeg','.webp','.tif','.tiff'])
        aligned=gr.Checkbox(label='我确认辅助图与光学图是同一区域且已经对齐',value=False)
        prompt=gr.Textbox(label='TIF问题',value=PROMPT,lines=3)
        prep=gr.Button('准备图片并预览')
        task=gr.Textbox(label='任务编号（保留此编号可恢复）');restore_btn=gr.Button('读取已有任务')
        with gr.Row():
            start=gr.Button('开始或继续扫描',variant='primary');stop=gr.Button('停止扫描')
        progress=gr.Textbox(label='整图处理进度',lines=5)
        with gr.Row():
            sm=gr.Checkbox(value=True,label='TIF显示分割区域');sb=gr.Checkbox(value=True,label='TIF显示定位框')
        with gr.Row():
            original=gr.Image(label='TIF原图预览',interactive=False,height=500)
            overlay=gr.Image(label='整图预览（坐标对应原图）',interactive=False,height=500)
        with gr.Accordion('查看独立分割图',open=False):
            mask=gr.Image(label='分割预览：白色目标、黑色背景、灰色未完成/无效',interactive=False,height=320)
        coord=gr.Textbox(label='整图框与坐标',lines=8)
        tile=gr.Dropdown(label='查看已完成分块',choices=[])
        ti=gr.Image(label='分块预测图',interactive=False);ta=gr.Textbox(label='分块原始回答');tc=gr.Textbox(label='分块框与坐标')
        download=gr.File(label='下载地理坐标与整图结果')
        outs=[original,overlay,mask,progress,coord,tile,download]
        prep.click(prepare,[opt,mode,a1,a2,aligned,prompt,sm,sb],[job,task,*outs],concurrency_limit=1)
        start.click(resume,[job,prompt,sm,sb],outs,concurrency_id='model',concurrency_limit=1,trigger_mode='once')
        stop.click(lambda j:geojob.request_stop(j),[job],[progress],queue=False)
        restore_btn.click(restore,[task,sm,sb],[job,prompt,*outs])
        for flag in [sm,sb]:flag.change(lambda j,m,b:view_path(j,m,b),[job,sm,sb],[overlay])
        tile.change(detail,[job,tile],[ti,ta,tc])
