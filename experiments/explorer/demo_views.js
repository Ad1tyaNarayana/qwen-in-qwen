import {$,esc,number,hex,badge,heatmap,outputAt} from './data.js';

export const UNIFIED='0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d';
let animation=null;
export function stopDemoAnimation(){clearInterval(animation);animation=null;}
const source=(label)=>badge(label,label==='read from neural state'?'neural':'trace');
const metrics=m=>`<dl class="demoMetrics">${[['Instructions',m.instruction_count],['Trace tokens',m.trace_tokens],['Mismatches',m.mismatches],['GPU minutes',m.generation_seconds==null?null:(m.generation_seconds/60).toFixed(3)],['Stock prefix tokens',m.stock_hf_prefix_tokens_checked]].map(([k,v])=>`<div><dt>${k}</dt><dd>${v==null?'Unavailable':typeof v==='number'?number(v):esc(v)}</dd></div>`).join('')}</dl>`;

export function gallery(manifests,open){
 $('detail').innerHTML=`<div class="eyebrow">PROGRAM GALLERY / VERIFIED EVIDENCE</div><h2 class="galleryTitle">One network. Different programs.</h2><p class="galleryIntro">Follow each guest from its generated tokens into instructions, memory and application state.</p><div class="networkIdentity"><span>SHARED UNIFIED W=4 NETWORK WEIGHT SHA-256</span><code>${UNIFIED}</code><small>CPU network weights. Guest data and ELF binaries differ.</small></div><div class="galleryGrid">${manifests.map(({manifest:m},i)=>`<button class="demoCard ${m.network_weight_sha256===UNIFIED?'unified':'original'}" data-open-demo="${i}"><div class="cardTop"><span>${esc(m.demo)}</span>${badge(m.network_weight_sha256===UNIFIED?'UNIFIED NETWORK':'DIFFERENT NETWORK · ORIGINAL',m.network_weight_sha256===UNIFIED?'neural':'reference')}</div><div class="tiny">${esc(m.status)} · ${esc(m.output_source)}</div><pre>${esc(m.outputs?.slice(0,170))}${m.outputs?.length>170?'…':''}</pre>${metrics(m)}<div class="cardHash"><span>NETWORK WEIGHT SHA-256</span><code>${esc(m.network_weight_sha256)}</code></div><div class="cardAction">Open application &amp; inspect L2–L4 <span>↗</span></div></button>`).join('')}</div>`;
 $('detail').querySelectorAll('[data-open-demo]').forEach(b=>b.onclick=()=>open(+b.dataset.openDemo));
}

function pixels(canvas,values,width,height){
 canvas.width=width;canvas.height=height;
 const ctx=canvas.getContext('2d'),im=ctx.createImageData(width,height);
 values.forEach((v,i)=>{im.data.set([v,v,v,255],i*4);});ctx.putImageData(im,0,0);
}
function gridBoard(canvas,values,width,height){
 canvas.width=width*20;canvas.height=height*20;const ctx=canvas.getContext('2d');
 values.forEach((v,i)=>{const x=i%width*20,y=Math.floor(i/width)*20;ctx.fillStyle=v?'#7ed6d1':'#141f2c';ctx.fillRect(x+1,y+1,18,18);});
}
function logitsHTML(values){
 const max=Math.max(1,...values.map(Math.abs));
 return `<div class="logitChart" role="img" aria-label="Ten signed guest logits in Q12">${values.map((v,i)=>`<div><span>${i}</span><div class="logitTrack"><i style="left:${v<0?50+v/max*50:50}%;width:${Math.abs(v)/max*50}%;background:${v<0?'#c9829d':'#7ed6d1'}"></i></div><code>${v}</code></div>`).join('')}</div>`;
}
export function comparison(manifests){
 const runs=['mnist','mnist-scalar'].map(name=>manifests.find(x=>x.manifest.demo===name)?.manifest);
 if(runs.some(x=>!x))return '';
 const max=Math.max(...runs.map(m=>m.trace_tokens));
 return `<div class="panel"><h3>NDOT4 and scalar · complete generated trace lengths</h3><div class="traceComparison">${runs.map(m=>`<div><span>${m.demo==='mnist'?'NDOT4':'Scalar'}</span><div><i style="width:${m.trace_tokens/max*100}%"></i></div><strong>${number(m.trace_tokens)}</strong></div>`).join('')}</div><p>Published accepted GPU runs. Identical DROM and output. Counts exclude prefill and GO. These bars compare token counts, not elapsed time.</p></div>`;
}

export function demoView(state,{seek,setLevel,header,availability,bindSamples}){
 const app=state.record.application;
 const events=app.events;
 const event=events.findLast(e=>e.offset<=state.offset);
 const neural=state.snapshot?.app;
 const arrays=neural?.arrays??event?.arrays??{};
 const provenance=neural?'read from neural state':'decoded from generated trace';
 const name=app.kind;
 const m=state.manifests.find(x=>x.manifest.elf_sha256===state.program.elf_sha256)?.manifest;
 const output=outputAt(state.record,state.offset);
 let view='';
 if(name==='life'||name==='tictactoe'){
  view=`<div class="grid2"><div class="panel"><h3>${name==='life'?'Guest RAM board · 24 × 16':'Guest RAM board · X human / O CPU'}</h3>${source(provenance)}<p>${neural?'Exact captured RAM at this prefix. '+(name==='life'?'Final RAM contains the unprinted successor generation.':''):esc(event?event.label+' · recorded RAM frame at prefix '+number(event.offset)+'. Later RAM changes are not shown.':'No complete board yet.')}</p>${arrays.board?(name==='life'?'<canvas id="demoBoard" class="lifeBoard"></canvas>':'<div id="ticBoard" class="ticBoard"></div>'):'<p>No board available at this prefix.</p>'}<p>${name==='life'?'Animation advances recorded RAM boards. It does not simulate Life on the host.':'Minimax progress is measured as generated token positions inside the ELF score function. Search scores and a dynamic recursion stack are not reconstructed.'}</p></div><div class="panel"><h3>${name==='life'?'Generation history':'Minimax progress'}</h3><p>${source('decoded from generated trace')}</p><div class="metrics"><div><b>${name==='life'?events.filter(e=>e.offset<=state.offset).length:number(event?.minimax_token_positions??0)}</b><small>${name==='life'?'DISPLAYED GENERATIONS OBSERVED':'TOKEN POSITIONS IN SCORE THROUGH THIS BOARD'}</small></div></div><pre>${esc(output)}</pre></div></div>`;
 }else if(name.startsWith('mnist')){
  view=comparison(state.manifests)+`<div class="grid2"><div class="panel"><h3>Input digit · 28 × 28 RAM halfwords</h3>${source(provenance)}${arrays.input?'<canvas id="digitImage" class="digitImage"></canvas>':'<p>Input RAM has not been captured or decoded yet.</p>'}<h3>Hidden layer · 32 ReLU activations · Q8</h3>${source(provenance)}${arrays.hidden?'<canvas id="hiddenHeat" class="hiddenHeat"></canvas><pre id="hiddenValues"></pre>':''}<p>${neural?'Exact scratch buffers at the selected prefix. Incomplete layers are not replaced by completed values.':'Buffers from the last completed trace event at prefix '+number(event?.offset??0)+'. Later RAM changes are not shown.'}</p></div><div class="panel"><h3>Logits · ten signed Q12 register results</h3>${source('decoded from generated trace')}<p>${event?esc(event.label)+' · completed at prefix '+number(event.offset):'No complete logits yet.'}</p>${event?.logits?logitsHTML(event.logits):''}<p>The guest does not store a logits array in RAM. These values are decoded from generated result commits at the ELF rescale instruction, never recomputed on the host.</p><pre>${esc(output.match(/Label \d+ prediction \d+/g)?.join('\n')??'')}</pre></div></div>`;
 }else if(name==='sha256'){
  const emitted=[...output.matchAll(/SHA256 (empty|abc)\n([a-f0-9]{64})\n/g)];
  view=`<div class="panel"><h3>Guest digest vs Python hashlib reference</h3><table><thead><tr><th>Message</th><th>Generated BYTE digest / host hashlib reference</th><th>Check</th></tr></thead><tbody>${emitted.map(x=>`<tr><td>${x[1]}</td><td><code>${x[2]}</code><br><code class="tiny">${app.reference_digests[x[1]==='empty'?'':x[1]]}</code></td><td>${x[2]===app.reference_digests[x[1]==='empty'?'':x[1]]?'MATCH':'MISMATCH'}</td></tr>`).join('')}</tbody></table><p>Top digest decoded from generated BYTE output. Bottom digest is the independent host hashlib reference.</p></div><div class="grid2"><div class="panel"><h3>Message schedule · 64 guest words</h3>${source(provenance)}<p>${neural?'Retained abc schedule in the hash-bound ELF stack frame after return.':'Schedule RAM reconstructed from generated store commits at the selected round.'}</p><pre>${arrays.schedule?.values.map((v,i)=>`w[${String(i).padStart(2)}] ${hex(v)}`).join('\n')??'No schedule has been decoded yet.'}</pre></div><div class="panel"><h3>Round working state · a through h</h3>${source('decoded from generated trace')}<p>${esc(event?.label??'No completed round yet.')}</p><pre>${event?.working?.map((v,i)=>`${'abcdefgh'[i]} = ${hex(v)}`).join('\n')??''}</pre><p>Round boundary registers come from generated commits. Register names are bound to this exact ELF. Mid-round temporaries are not interpreted as completed rounds.</p></div></div>`;
 }
 $('detail').innerHTML=header('L4 / GUEST APPLICATION',m?.demo??name,'Program and data change. The unified CPU network weights stay the same.',provenance,neural?'neural':'trace')+`<div class="arraySelect"><button id="demoCPU">L2 CPU</button><button id="demoProgram">L3 Program</button><button id="demoAnimate">${animation?'Pause frames':'Animate recorded frames'}</button><select id="demoFrame" aria-label="Recorded application frame"><option value="">Choose an exact recorded event</option>${events.map((e,i)=>`<option value="${i}" ${e.offset===state.offset?'selected':''}>${esc(e.label)} · ${number(e.offset)}</option>`).join('')}</select></div>`+availability()+view+`<div class="panel"><h3>Bindings and provenance</h3><p>${esc(app.scope)}</p><p>Network weight SHA-256 <code>${esc(m?.network_weight_sha256??state.record.binding.network_weight_sha256)}</code></p><p>ELF <code>${esc(state.program.elf_sha256)}</code></p>${m?metrics(m):''}</div>`;
 bindSamples();
 $('demoCPU').onclick=()=>{stopDemoAnimation();setLevel(2);};$('demoProgram').onclick=()=>{stopDemoAnimation();setLevel(3);};
 $('demoFrame').onchange=()=>{stopDemoAnimation();if($('demoFrame').value!=='')seek(events[+$('demoFrame').value].offset);};
 $('demoAnimate').onclick=()=>{if(animation){stopDemoAnimation();$('demoAnimate').textContent='Animate recorded frames';return;}let i=events.findIndex(e=>e.offset>state.offset);if(i<0)i=0;const record=state.record;animation=setInterval(()=>{if(state.record!==record||state.level!==4||i>=events.length){stopDemoAnimation();return;}seek(events[i++].offset);},600);$('demoAnimate').textContent='Pause frames';};
 if($('demoBoard'))gridBoard($('demoBoard'),arrays.board.values,24,16);
 if($('ticBoard'))$('ticBoard').innerHTML=arrays.board.values.map(v=>`<span>${['','X','O'][v]??'?'}</span>`).join('');
 if($('digitImage'))pixels($('digitImage'),arrays.input.values,28,28);
 if($('hiddenHeat')){heatmap($('hiddenHeat'),arrays.hidden.values,1,32);$('hiddenValues').textContent=arrays.hidden.values.join(', ');}
}

export function tinyQwenView(state,{seek,setLevel,header,availability,bindSamples,updateCrumbs}){
 const output=outputAt(state.record,state.offset);
 const pieces=[...output.matchAll(/@(\d+) (\d+) (\d+) (\d+) (\d+)\n>([\s\S]*?)\n(?=@|$)/g)];
 const app=state.snapshot?.app,pack=state.record.pack;
 const m=state.manifests.find(x=>x.manifest.demo==='tiny-qwen35')?.manifest;
 const layers=pack.graph.map((l,i)=>({index:i,kind:l.kind===3?'DeltaNet':'Full attention'}));
 const layer=Math.min(state.innerLayer,7);
 const arrays=Object.entries(app?.arrays??{}).filter(([,a])=>a.layer==null||a.layer===layer);
 if(!arrays.some(([n])=>n===state.array))state.array=arrays.find(([n])=>n.includes('state')||n.includes('keys'))?.[0]??'x';
 $('detail').innerHTML=header('L4 / HYBRID LANGUAGE MODEL','Qwen3.5 architecture running inside a Qwen3.5-architecture network.','The outer network has compiled, not pretrained, weights. The trained inner model executes as an integer program in guest RAM.','decoded from generated trace','trace')+
 `<div class="arraySelect"><button id="qwenCPU">L2 CPU</button><button id="qwenProgram">L3 Program</button></div><div class="panel"><h3>Actual generated tokens and text</h3>${source('decoded from generated trace')}<div class="story">${esc(pieces.map(m=>m[6]).join(''))}</div>${pieces.map(m=>`<span class="tokenChip" title="position ${m[1]}, input ${m[2]}">#${m[3]} ${esc(m[6])}</span>`).join('')}<p>${esc(m?.quality_limitation??'')}</p></div><div class="appLayers">${layers.map(l=>`<button data-qwen-layer="${l.index}" class="${layer===l.index?'active':''}">Layer ${l.index}<small>${l.kind}</small></button>`).join('')}</div>`+availability()+
 `<div class="grid2"><div class="panel"><h3>${layers[layer].kind} guest RAM</h3>${app?`${source('read from neural state')}<div class="arraySelect"><select id="qwenArray" aria-label="Qwen guest array">${arrays.map(([n])=>`<option ${n===state.array?'selected':''}>${n}</option>`).join('')}</select><label>Position / value head <input id="qwenPosition" type="number" min="0" max="63" value="${state.innerPos}"></label></div><canvas id="qwenHeat" class="heatmap"></canvas><pre id="qwenValues"></pre><p>DeltaNet states are Q20 [value head][key][value]. Full-attention KV is Q10 [position][KV dimension]. Convolution history and activations are Q12. Scratch is shared across layers. This final snapshot provides no past activation history.</p>`:'<p>No exact neural snapshot at this prefix. State is unavailable.</p>'}</div><div class="panel"><h3>Raw actual generated BYTE output</h3>${source('decoded from generated trace')}<pre>${esc(output)}</pre><p>Six DeltaNet layers and two full-attention layers. Unwritten KV positions remain the literal captured RAM values.</p></div></div>`;
 bindSamples();$('qwenCPU').onclick=()=>setLevel(2);$('qwenProgram').onclick=()=>setLevel(3);
 $('detail').querySelectorAll('[data-qwen-layer]').forEach(b=>b.onclick=()=>{state.innerLayer=+b.dataset.qwenLayer;tinyQwenView(state,{seek,setLevel,header,availability,bindSamples,updateCrumbs});updateCrumbs();});
 if(!app)return;
 function draw(){
  const a=app.arrays[state.array];let values=a.values,address=a.address,description='whole buffer';
  if(state.array.endsWith('-state')){const head=Math.min(3,state.innerPos);values=values.slice(head*256,(head+1)*256);address+=head*1024;description=`value head ${head}, 16 × 16`;}
  else if(state.array.endsWith('-keys')||state.array.endsWith('-values')){const pos=Math.min(63,state.innerPos);values=values.slice(pos*32,(pos+1)*32);address+=pos*64;description=`position ${pos}, 32 KV dimensions`;}
  const isState=state.array.endsWith('-state'),isKV=state.array.endsWith('-keys')||state.array.endsWith('-values');
  $('qwenPosition').max=isState?3:63;$('qwenPosition').disabled=!isState&&!isKV;
  $('qwenPosition').value=isState?Math.min(3,state.innerPos):state.innerPos;
  const cols=isState?16:Math.min(32,values.length);heatmap($('qwenHeat'),values,Math.ceil(values.length/cols),cols);
  updateCrumbs();
  $('qwenValues').textContent=`${state.array} · ${hex(address)} · ${description}\nQ${a.fractional_bits} · exact prefix ${number(app.offset)}\nFirst values ${values.slice(0,32).join(', ')}`;
 }
 $('qwenArray').onchange=()=>{state.array=$('qwenArray').value;draw();};$('qwenPosition').onchange=()=>{state.innerPos=Math.max(0,Math.min(63,+$('qwenPosition').value||0));draw();};draw();
}
