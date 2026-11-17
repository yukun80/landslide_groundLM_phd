"""实际浏览器检查定位开关、TIF准备、停止继续和下载；需要本地服务。"""
import json,zipfile,time
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/geospatial-check';UI=OUT/'browser'

def main():
    UI.mkdir(exist_ok=True);checks={}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1440,'height':1100})
        page.goto('http://127.0.0.1:7860',wait_until='domcontentloaded')
        expect(page.get_by_role('textbox',name='定位框原图位置',exact=True)).not_to_have_value('',timeout=30000)
        page.wait_for_timeout(1500);page.screenshot(path=str(UI/'boxes.png'),full_page=True)
        page.get_by_label('显示定位框',exact=True).uncheck();page.wait_for_timeout(1000)
        assert page.locator('img').evaluate_all('(xs)=>xs.some(x=>x.src.includes("view_1_0.png"))')
        page.get_by_label('显示分割区域',exact=True).uncheck();page.wait_for_timeout(1000)
        assert page.locator('img').evaluate_all('(xs)=>xs.some(x=>x.src.includes("view_0_0.png"))')
        page.get_by_label('显示定位框',exact=True).check();page.get_by_label('显示分割区域',exact=True).check();checks['layer_toggles']=True
        with page.expect_download() as d:page.locator('a[href$="localization.zip"]').click()
        d.value.save_as(str(UI/'localization.zip'))
        with zipfile.ZipFile(UI/'localization.zip') as z:assert json.loads(z.read('boxes.json'))['boxes'][0]['valid']
        checks['box_download']=True
        page.get_by_role('tab',name='TIF整图与地理坐标',exact=True).click()
        # 固定真实输入，浏览器创建新任务，真实运行9块；测试停止和继续。
        source=json.loads((OUT/'checks.json').read_text())['real_tif_9_tiles_stop_resume']['source']
        page.locator('#geo-optical input[type=file]').set_input_files(source);page.wait_for_timeout(1000)
        page.get_by_role('button',name='准备图片并预览',exact=True).click()
        expect(page.get_by_role('textbox',name='任务编号（保留此编号可恢复）',exact=True)).not_to_have_value('',timeout=240000)
        jobid=page.get_by_role('textbox',name='任务编号（保留此编号可恢复）',exact=True).input_value();job=ROOT/'outputs/benchmark500/geospatial'/jobid
        page.screenshot(path=str(UI/'prepared.png'),full_page=True)
        page.get_by_role('button',name='开始或继续扫描',exact=True).click()
        deadline=time.time()+240
        while time.time()<deadline:
            status=json.loads((job/'status.json').read_text())
            if len(status['completed'])>=1:break
            page.wait_for_timeout(500)
        else:raise AssertionError('first tile timed out')
        page.get_by_role('button',name='停止扫描',exact=True).click()
        deadline=time.time()+90
        while time.time()<deadline:
            status=json.loads((job/'status.json').read_text())
            if status['state']=='stopped':break
            page.wait_for_timeout(500)
        else:raise AssertionError('stop timed out')
        done_before=len(status['completed']);assert 0<done_before<9
        page.wait_for_timeout(1500);page.screenshot(path=str(UI/'stopped.png'),full_page=True)
        page.get_by_role('button',name='开始或继续扫描',exact=True).click()
        deadline=time.time()+240
        while time.time()<deadline:
            status=json.loads((job/'status.json').read_text())
            if status['state']=='complete':break
            page.wait_for_timeout(500)
        else:raise AssertionError(status)
        assert len(status['completed'])==9 and not status['failed'];checks['real_tif_prepare_run_stop_resume']=True
        page.wait_for_timeout(2000)
        expect(page.get_by_role('textbox',name='分块原始回答',exact=True)).not_to_have_value('',timeout=30000)
        assert '经纬度' in page.get_by_role('textbox',name='整图框与坐标',exact=True).input_value()
        page.get_by_label('TIF显示定位框',exact=True).uncheck();page.wait_for_timeout(1000)
        page.get_by_label('TIF显示定位框',exact=True).check()
        page.screenshot(path=str(UI/'complete.png'),full_page=True)
        with page.expect_download(timeout=30000) as d:page.locator('a[href$="geographic_results.zip"]').click()
        d.value.save_as(str(UI/'geographic_results.zip'))
        with zipfile.ZipFile(UI/'geographic_results.zip') as z:
            assert 'segmentation.tif' in z.namelist();assert json.loads(z.read('boxes.geojson'))['features'];assert json.loads(z.read('summary.json'))['state']=='complete'
        page.reload(wait_until='domcontentloaded');page.get_by_role('tab',name='TIF整图与地理坐标',exact=True).click()
        page.get_by_role('textbox',name='任务编号（保留此编号可恢复）',exact=True).fill(jobid);page.get_by_role('button',name='读取已有任务',exact=True).click()
        expect(page.get_by_role('textbox',name='整图处理进度',exact=True)).to_have_value(__import__('re').compile('.*全部完成.*',__import__('re').S),timeout=30000)
        checks.update(download_and_restore=True,job=jobid,completed_before_stop=done_before)
        browser.close()
    (UI/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print(checks)
if __name__=='__main__':main()
