import {writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const here=fileURLToPath(new URL('.',import.meta.url));
const targets=await (await fetch('http://127.0.0.1:9223/json')).json();
const target=targets.find(t=>t.type==='page');
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0;const pending=new Map(),errors=[],requests=[];
ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const pair=pending.get(m.id);if(pair){pending.delete(m.id);m.error?pair[1](Error(JSON.stringify(m.error))):pair[0](m.result);}}else if(m.method==='Network.requestWillBeSent')requests.push(m.params.request.url);else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text+' '+JSON.stringify(m.params.exceptionDetails.exception));});
const send=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,[resolve,reject]);ws.send(JSON.stringify({id:n,method,params}));});
async function evaluate(expression){if(expression.includes('await '))expression='(async()=>{'+expression+'})()';const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const check=(ok,message)=>{if(!ok)throw Error(message);};
async function shot(name){await evaluate('Promise.all(document.getAnimations().map(a=>a.finished))');await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');const {data}=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(here+'screenshots/'+name+'.png',Buffer.from(data,'base64'));}
const checks=[];
async function assert(expression,message){check(await evaluate(expression),message);checks.push(message);}
async function open(name){await evaluate(`await explorer.openDemo(explorer.state.manifests.findIndex(x=>x.manifest.demo===${JSON.stringify(name)}));`);}
try{
 await send('Runtime.enable');await send('Page.enable');await send('Network.enable');
 await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1150,deviceScaleFactor:1,mobile:false});
 await send('Page.navigate',{url:'http://127.0.0.1:8792/'});
 for(let i=0;i<100;i++){if(await evaluate('document.documentElement.dataset.ready === "true"'))break;await new Promise(r=>setTimeout(r,100));}
 await assert('explorer.state.gallery && document.querySelectorAll(".demoCard").length===explorer.state.manifests.length','Landing has one card per manifest');
 await assert(`Array.from(document.querySelectorAll('.demoCard.unified')).every(c=>c.querySelector('.cardHash code').textContent==='0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d')`,'All unified cards show identical full network weight hash');
 await assert(`document.querySelectorAll('.demoCard.original').length===2 && Array.from(document.querySelectorAll('.demoCard.original')).every(c=>c.textContent.includes('DIFFERENT NETWORK'))`,'Doom and Phase 1 visibly use another network');
 await shot('09-gallery');
 await open('sha256');
 await assert(`explorer.state.record.id==='showcase-sha256' && document.getElementById('detail').textContent.includes('MATCH') && explorer.state.snapshot.verification.passed`,'SHA card loads correct recording, hashlib comparison and exact verified neural sample');
 await shot('10-sha256');
 await evaluate(`document.getElementById('demoFrame').value='12';document.getElementById('demoFrame').dispatchEvent(new Event('change'));`);
 await assert(`explorer.state.snapshot===null && explorer.state.offset===explorer.state.record.application.events[12].offset && document.getElementById('detail').textContent.includes('empty round 13')`,'SHA round control selects an actual generated event without substituting a neural sample');
 await open('life');
 await assert(`document.getElementById('demoBoard').width===480 && explorer.state.snapshot.app.arrays.board.values.length===384`,'Life renders real captured guest RAM');
 await shot('11-life-neural');
 await evaluate(`await explorer.seek(explorer.state.record.application.events[0].offset);document.getElementById('demoAnimate').click();`);
 await new Promise(r=>setTimeout(r,750));
 await assert(`explorer.state.offset===explorer.state.record.application.events[1].offset && explorer.state.snapshot===null`,'Life animation advances recorded RAM frames with trace provenance');
 await evaluate(`document.getElementById('demoAnimate').click()`);
 await open('tictactoe');
 await assert(`document.querySelectorAll('#ticBoard span').length===9 && document.getElementById('detail').textContent.includes('CPU wins') && document.getElementById('detail').textContent.includes('TOKEN POSITIONS IN SCORE')`,'Tic-tac-toe renders board and decoded minimax progress');
 await shot('12-tictactoe');
 for(const name of ['sha256','life','tictactoe','mnist','mnist-scalar']){
  await open(name);
  await evaluate(`explorer.setLevel(2)`);
  await assert(`explorer.state.program.elf_sha256===explorer.state.manifests.find(x=>x.manifest.demo===${JSON.stringify(name)}).manifest.elf_sha256 && document.querySelectorAll('.registers button').length===32`,name+' L2 uses its own ELF and neural register file');
  await evaluate(`explorer.setLevel(3)`);
  await assert(`document.getElementById('detail').textContent.includes('engine_main')`,name+' L3 shows its ELF program');
 }
 await open('mnist');
 await assert(`document.getElementById('digitImage').width===28 && document.getElementById('hiddenHeat').width===32 && document.querySelectorAll('.logitChart>div').length===10`,'MNIST renders exact input, hidden activations and ten generated logits');
 await assert(`document.getElementById('detail').textContent.includes('189,092') && document.getElementById('detail').textContent.includes('785,128')`,'MNIST side-by-side comparison uses both full trace lengths');
 await evaluate(`document.getElementById('detail').scrollTop=220`);await shot('13-mnist-neural');
 await evaluate(`document.getElementById('detail').scrollTop=0`);
 await evaluate(`await explorer.seek(explorer.state.record.application.events[0].offset);`);
 await assert(`explorer.state.snapshot===null && document.getElementById('detail').textContent.includes('decoded from generated trace')`,'MNIST frame history retains trace provenance');
 await shot('14-mnist-trace');
 await open('tiny-qwen35');
 await assert(`explorer.state.record.id==='tiny-qwen35-final' && explorer.state.snapshot.verification.passed && explorer.state.snapshot.offset===35836005`,'Tiny Qwen opens final verified actual GPU state');
 await assert(`document.getElementById('detail').textContent.includes('Qwen3.5 architecture running inside a Qwen3.5-architecture network.') && document.getElementById('detail').textContent.includes('compiled, not pretrained')`,'Qwen headline and outer compiled weight provenance are explicit');
 await assert(`document.querySelectorAll('[data-qwen-layer]').length===8 && [...document.querySelectorAll('[data-qwen-layer]')].filter(b=>b.textContent.includes('DeltaNet')).length===6`,'Hybrid view has six DeltaNet layers and two full-attention layers');
 await assert(`document.querySelector('.story').textContent==='Once upon a time, there' && document.querySelectorAll('.tokenChip').length===6 && document.getElementById('detail').textContent.includes('291/327') && document.getElementById('detail').textContent.includes('not an unseen')`,'Actual BYTE text and token IDs render with inspected-prompt quality limitation');
 await assert(`explorer.state.snapshot.app.arrays['layer-0-state'].values.length===1024 && document.getElementById('qwenValues').textContent.includes('Q20') && document.getElementById('qwenHeat').width===16 && document.getElementById('qwenHeat').height===16`,'DeltaNet recurrent state decoded from actual neural RAM');
 await assert(`document.getElementById('recording').value===explorer.state.record.id`,'Recording selector follows manifest URL loading');
 await shot('17-qwen35-deltanet');
 await evaluate(`document.getElementById('detail').scrollTop=440`);await shot('21-qwen35-final-state');
 await evaluate(`document.getElementById('detail').scrollTop=0`);
 await evaluate(`document.querySelector('[data-qwen-layer="3"]').click();document.getElementById('qwenPosition').value='5';document.getElementById('qwenPosition').dispatchEvent(new Event('change'));`);
 await assert(`document.getElementById('qwenValues').textContent.includes('position 5, 32 KV dimensions') && document.getElementById('qwenValues').textContent.includes('Q10')`,'Full-attention KV position control uses actual Q10 RAM');
 await assert(`document.getElementById('breadcrumb').textContent.includes('0x'+(explorer.state.snapshot.app.arrays['layer-3-keys'].address+5*64).toString(16).padStart(8,'0'))`,'KV breadcrumb points to the selected guest RAM position');
 await shot('18-qwen35-kv');
 await evaluate(`await explorer.seek(explorer.state.record.end-1);`);
 await assert(`explorer.state.snapshot===null && document.getElementById('detail').textContent.includes('State is unavailable') && !document.getElementById('qwenHeat')`,'Qwen missing-frame view never substitutes final neural RAM');
 for(const name of ['mnist','life']){
  await open(name);
  await assert(`explorer.state.record.snapshots.length>1`,name+' loads periodic neural samples');
  await evaluate(`await explorer.seek(explorer.state.record.snapshots[0].offset);`);
  await assert(`explorer.state.snapshot.verification.passed && explorer.state.snapshot.verification.reference_consumed_tokens===explorer.state.offset-1`,name+' periodic snapshot has exact independent verification boundary');
  await shot(name==='mnist'?'19-mnist-periodic':'20-life-periodic');
 }
 await open('stories260K ndot 38-position story');
 await assert(`explorer.state.record.id==='periodic-stories260k' && explorer.state.record.snapshots.length===16 && explorer.state.record.end===1041913`,'Stories opens 16 verified samples covering one complete inner position from boot');
 await evaluate(`await explorer.seek(explorer.state.record.snapshots[3].offset);`);
 await assert(`explorer.state.snapshot.verification.passed && explorer.state.snapshot.offset===262144`,'Stories intermediate snapshot is available at its exact GPU prefix');
 await shot('22-stories-periodic');
 await open('Doom first frame');
 await evaluate(`explorer.setLevel(2)`);
 await assert(`explorer.state.record===null && document.getElementById('detail').textContent.includes('No compatible recording')`,'Original-network manifest never displays another guest CPU recording');
 await open('mnist');
 await evaluate(`const s=explorer;const L=s.details.findIndex(d=>d?.heads.some(h=>h.name==='ram3'));s.state.layer=L;s.state.pick={kind:'head',index:s.details[L].heads.findIndex(h=>h.name==='ram3'),value:0};s.setLevel(1);`);
 await assert(`document.getElementById('headHeat').width===128 && document.getElementById('detail').textContent.includes('read from neural state')`,'Demo checkpoint S slice renders in L1');
 await shot('16-mnist-neural-S');
 await evaluate(`explorer.showGallery();document.getElementById('detail').scrollTop=document.getElementById('detail').scrollHeight`);
 await assert(`[...document.querySelectorAll('.demoCard')].find(c=>c.textContent.includes('tiny-qwen35')).textContent.includes('35,836,005')`,'Tiny Qwen gallery card shows completed neural trace count');
 await shot('23-qwen35-gallery');
 await send('Emulation.setDeviceMetricsOverride',{width:820,height:1100,deviceScaleFactor:1,mobile:false});
 await evaluate(`explorer.showGallery()`);await shot('15-gallery-compact');
 await assert(`document.getElementById('detail').scrollWidth<=document.getElementById('detail').clientWidth+1`,'Gallery fits compact viewport');
 check(errors.length===0,'No browser exceptions: '+errors.join('\n'));
 check(requests.every(u=>u.startsWith('http://127.0.0.1:8792/')||u==='about:blank'),'All requests are local');
 await writeFile(here+'showcase-browser-validation.json',JSON.stringify({passed:true,checks,screenshots:15,errors},null,2)+'\n');
 console.log('Showcase browser checks passed. Fifteen screenshots saved.');
}finally{ws.close();}
