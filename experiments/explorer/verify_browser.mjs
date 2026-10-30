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
try{
 await send('Runtime.enable');await send('Page.enable');await send('Network.enable');await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1040,deviceScaleFactor:1,mobile:false});await send('Page.navigate',{url:'http://127.0.0.1:8792/#0'});
 for(let i=0;i<100;i++){if(await evaluate('document.documentElement.dataset.ready === "true"'))break;await new Promise(r=>setTimeout(r,100));}
 check(await evaluate('!!window.explorer'),'Explorer initialized');
 check(await evaluate('explorer.graph.layers.length===65 && explorer.graph.geometry.ffn_units===68923'),'Graph geometry');
 await shot('01-network');
 await evaluate(`document.getElementById('pointMode').value='residual';document.getElementById('pointMode').dispatchEvent(new Event('change'));`);
 check(await evaluate('explorer.map.points.length===2304*65'),'Every hidden coordinate has a residual map point');
 await evaluate(`document.getElementById('pointMode').value='full';document.getElementById('pointMode').dispatchEvent(new Event('change'));`);
 check(await evaluate('explorer.map.points.length===601096'),'Complete zero-padded stock capacity');
 await evaluate(`document.getElementById('pointMode').value='units';document.getElementById('pointMode').dispatchEvent(new Event('change'));`);
 await evaluate(`document.getElementById('search').value='ram3';document.getElementById('search').dispatchEvent(new Event('input'));`);
 check(await evaluate(`document.getElementById('results').textContent.includes('ram3')`),'Search finds real RAM head');
 await evaluate(`document.querySelector('#results [data-head]').click()`);
 check(await evaluate(`document.getElementById('detail').textContent.includes('striped word rows')`),'RAM semantic axes');
 await evaluate(`await explorer.loadRecording('stories260k-final');await explorer.seek(explorer.state.record.end);`);
 check(await evaluate(`!!document.getElementById('headHeat')`),'Actual captured head heatmap rendered');
 await shot('02-ram-bank');
 await evaluate(`await explorer.loadRecording('stories260k-position-0')`);
 await evaluate(`document.getElementById('search').value='';document.getElementById('search').dispatchEvent(new Event('input'));explorer.setLevel(3);await explorer.seek(50000);`);
 check(await evaluate(`document.getElementById('detail').textContent.includes('matvec')`),'Function heatmap uses ELF symbols');
 await shot('03-program');
 await evaluate(`await explorer.loadRecording('stories260k-final');await explorer.seek(explorer.state.record.end);explorer.setLevel(2);`);
 check(await evaluate(`explorer.state.snapshot.registers['regs.rs1'].length===32`),'Neural registers present');
 check(await evaluate(`document.getElementById('detail').textContent.includes('read from neural state')`),'Neural provenance shown');
 await shot('04-neural-cpu');
 await evaluate(`explorer.setLevel(4);`);
 check(await evaluate(`explorer.state.snapshot.app.arrays.keys.values.length>0`),'KV arrays decoded from RAM');
 await evaluate(`document.getElementById('arrayName').value='keys';document.getElementById('arrayName').dispatchEvent(new Event('change'));`);
 await evaluate(`document.getElementById('detail').scrollTop=350`);
 await shot('05-inner-transformer');
 await evaluate(`document.getElementById('detail').scrollTop=0`);
 await shot('07-story-output');
 await evaluate(`await explorer.seek(explorer.state.record.end-1);`);
 check(await evaluate(`explorer.state.snapshot===null && document.getElementById('detail').textContent.includes('No neural-state snapshot')`),'Scrubbing does not substitute nearby neural sample');
 await send('Emulation.setDeviceMetricsOverride',{width:820,height:1100,deviceScaleFactor:1,mobile:false});await evaluate(`explorer.setLevel(0)`);await shot('06-compact-network');
 await evaluate(`explorer.setLevel(4);explorer.state.manifestIndex=explorer.state.manifests.findIndex(x=>x.manifest.demo==='life');document.getElementById('levels').querySelector('[data-level="4"]').click();`);
 check(await evaluate(`document.getElementById('detail').textContent.includes('actual self-fed GPU BYTE records') && document.getElementById('manifestCanvas').width===24`),'Accepted Life manifest rendered with provenance');
 await shot('08-life-manifest');
 check(requests.every(u=>u.startsWith('http://127.0.0.1:8792/')||u==='about:blank'),'All app requests are local');
 check(errors.length===0,'No uncaught browser errors: '+errors.join('\n'));
 await writeFile(here+'browser-validation.json',JSON.stringify({passed:true,checks:['initialization','graph geometry','head search','semantic RAM axes','ELF function timeline','neural register and RAM sample','inner KV decoder','no sample interpolation','compact viewport','no uncaught exceptions','accepted Life manifest provenance','all asset requests stay on localhost','every hidden coordinate represented','complete zero-padded stock capacity'],screenshots:8,errors},null,2)+'\n');
 console.log('Browser checks passed. Eight screenshots saved.');
}finally{ws.close();}
