"""用真实浏览器检查本地页面；--live在批量结束后执行真实推理和重复点击。"""
import argparse,json,zipfile,time
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/benchmark500/ui-check'
def main():
 p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');args=p.parse_args()
 rows=[json.loads(l) for l in (ROOT/'datasets/landslide_benchmark_500/samples.jsonl').open()];r=rows[0];checks={}
 OUT.mkdir(parents=True,exist_ok=True)
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1440,'height':1100})
  page.goto('http://127.0.0.1:7860',wait_until='domcontentloaded');page.wait_for_timeout(1000)
  page.get_by_role('button',name='下一条',exact=True).click();page.wait_for_timeout(1000);assert page.get_by_role('combobox').nth(1).input_value()==rows[1]['id'];checks['next']=True
  page.get_by_role('button',name='上一条',exact=True).click();page.wait_for_timeout(1000);assert page.get_by_role('combobox').nth(1).input_value()==r['id'];checks['previous']=True
  page.get_by_role('combobox').nth(0).click();page.get_by_role('option',name='landslide4sense',exact=True).click();page.wait_for_timeout(1000);assert page.get_by_role('combobox').nth(1).input_value().startswith('landslide4sense');checks['filter']=True
  page.get_by_role('combobox').nth(0).click();page.get_by_role('option',name='全部',exact=True).click();page.wait_for_timeout(1000)
  answer=page.get_by_role('textbox',name='模型原始回答',exact=True).input_value();assert answer
  with page.expect_download() as dl:page.locator('a[href$="result.zip"]').click()
  dl.value.save_as(str(OUT/'downloaded.zip'))
  with zipfile.ZipFile(OUT/'downloaded.zip') as z:
   assert json.loads(z.read('result.json'))['id']==r['id'];assert 'overlay.png' in z.namelist()
  checks['download']=True
  if args.live:
   joins=[];page.on('request',lambda req: joins.append(req.url) if '/queue/join' in req.url else None)
   before=set((ROOT/'outputs/benchmark500/interactive').glob('*/result.json'))
   old=page.locator('a[href$="result.zip"]').get_attribute('href')
   page.get_by_role('button',name='开始推理',exact=True).evaluate('(b)=>{b.click();b.click();}')
   page.wait_for_function('(old)=>{let x=document.querySelector("a[href$=\\"result.zip\\"]");return x&&x.href!==old}',arg=old,timeout=240000)
   page.wait_for_timeout(2000)
   created=set((ROOT/'outputs/benchmark500/interactive').glob('*/result.json'))-before
   assert len(created)==1,(joins,created)
   assert page.get_by_role('textbox',name='模型原始回答',exact=True).input_value()==answer
   checks['real_inference_and_duplicate_click']=True
   page.screenshot(path=str(OUT/'inference.png'),full_page=True)
  page.get_by_role('tab',name='上传测试',exact=True).click()
  page.locator('#upload-optical input[type=file]').set_input_files(str(ROOT/r['images'][-1]));page.wait_for_timeout(1000)
  page.get_by_label('光学＋近红外/短波红外',exact=True).check()
  page.locator('#upload-aux1 input[type=file]').set_input_files(str(OUT/'small.png'));page.wait_for_timeout(1000)
  page.get_by_role('button',name='预测上传图片',exact=True).click();page.get_by_text('辅助图片必须与光学图片尺寸相同，并由你确认已对齐。',exact=True).wait_for(timeout=10000);checks['wrong_size_rejected']=True
  if args.live:
   # 使用新页面清空上传状态，核验多模态三图实际推理。
   page.reload(wait_until='domcontentloaded');page.get_by_role('tab',name='上传测试',exact=True).click()
   m=next(x for x in rows if x['source']=='multimodal-landslide-dataset')
   page.get_by_label('光学＋InSAR＋高程',exact=True).check()
   for elem,path in zip(['upload-optical','upload-aux1','upload-aux2'],[m['images'][-1],m['images'][0],m['images'][1]]):
    page.locator(f'#{elem} input[type=file]').set_input_files(str(ROOT/path));page.wait_for_timeout(700)
   page.get_by_role('button',name='预测上传图片',exact=True).click()
   page.get_by_role('textbox',name='上传图片的回答',exact=True).wait_for(timeout=120000)
   expect(page.get_by_role('textbox',name='上传图片的回答',exact=True)).not_to_have_value('',timeout=120000)
   assert page.get_by_role('textbox',name='上传图片的回答',exact=True).input_value()
   checks['multimodal_upload_prediction']=True;page.screenshot(path=str(OUT/'upload-success.png'),full_page=True)
  page.get_by_role('tab',name='批量结果',exact=True).evaluate('(b)=>b.click()');page.get_by_role('button',name='刷新批量结果',exact=True).click();page.wait_for_timeout(1000)
  if args.live:
   assert page.locator('a[href$="all_results.zip"]').count()==1
   with page.expect_download() as dl:page.locator('a[href$="all_results.zip"]').click()
   dl.value.save_as(str(OUT/'all-downloaded.zip'))
   with zipfile.ZipFile(OUT/'all-downloaded.zip') as z:
    assert len([n for n in z.namelist() if n.endswith('/result.json')])==500
    assert len(z.namelist())==2004
    assert json.loads(z.read('summary.json'))['all']['count']==500
   checks['all_500_downloaded_and_checked']=True
  page.screenshot(path=str(OUT/'batch.png'),full_page=True);browser.close()
 (OUT/('live-check.json' if args.live else 'browser-check.json')).write_text(json.dumps(checks,indent=2));print(checks)
if __name__=='__main__':main()
