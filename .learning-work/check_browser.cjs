const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},permissions:['clipboard-read','clipboard-write']});
 const outputs=[];
 for(const [name,path] of [['original','.learning-work/original'],['annotated','Xiaomi-Lens-0-raw']]){
  const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:8769/${path}/docs/index.html`,{waitUntil:'networkidle'});
  await page.locator('.case-list button').nth(1).click();
  const selected=await page.locator('.case-list button').nth(1).getAttribute('data-gif');
  if(!(await page.locator('#player-img').getAttribute('src')).includes(selected))throw Error('case switch');
  await page.locator('#replay').click();
  const filters=page.locator('.chip[data-filter]');
  for(let i=0;i<await filters.count();i++){
   await filters.nth(i).click();
   const f=await filters.nth(i).getAttribute('data-filter');
   const visible=await page.locator('.case-list button:visible').evaluateAll(bs=>bs.map(b=>b.dataset.group));
   if(!visible.length || (f!=='all'&&visible.some(g=>g!==f)))throw Error('filter '+f);
  }
  const tabs=page.locator('[role="tab"]');
  for(let i=0;i<await tabs.count();i++){
   await tabs.nth(i).click();const id=await tabs.nth(i).getAttribute('aria-controls');
   if(!await page.locator('#'+id).isVisible())throw Error('tab '+id);
  }
  await page.locator('#copy').click();await page.waitForTimeout(150);
  const clip=await page.evaluate(()=>navigator.clipboard.readText());if(!clip.includes('@'))throw Error('clipboard');
  await page.locator('.chip[data-filter="all"]').click();
  await page.locator('.case-list button').first().click();
  await page.evaluate(()=>window.scrollTo(0,0));await page.waitForTimeout(400);
  await page.screenshot({path:`.learning-work/${name}-desktop.png`});
  const body=await page.locator('body').innerText();
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
  await page.screenshot({path:`.learning-work/${name}-mobile.png`});
  outputs.push({name,errors,filterCount:await filters.count(),tabCount:await tabs.count(),overflow,body});
  await page.close();
 }
 if(outputs[0].body!==outputs[1].body)throw Error('visible content changed');
 if(outputs.some(x=>x.errors.length))throw Error(JSON.stringify(outputs.map(x=>x.errors)));
 outputs.forEach(x=>delete x.body);fs.writeFileSync('.learning-work/browser-results.json',JSON.stringify(outputs,null,2));
 console.log('BROWSER_PASS',JSON.stringify(outputs));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
