import {$,esc,hex,number,json,bytes,badge,tokenFamily,outputAt,instructionText,heatmap} from './data.js';
import {NetworkMap} from './network.js';
import {drawGuest} from './decoders.js';
import {gallery,demoView,stopDemoAnimation} from './demo_views.js';
const LEVELS=['Network','Layer / unit','CPU','Program','Application'];
const state={level:0,layer:4,pick:null,offset:0,record:null,trace:null,program:null,snapshot:null,base:'recordings/',innerLayer:0,innerPos:0,array:'x',manifestIndex:null,memory:0x1000100,list:false,page:0,manifests:[]};
let graph,details,map,catalog,playing=false,loadVersion=0;
const cache=new Map();
const roleColor=r=>graph.roles[r]?.[1]??'#aaa';
const roleName=r=>graph.roles[r]?.[0]??r;
const fail=e=>{console.error(e);$('toast').textContent=e.message;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',7000);};
function header(kicker,title,description,source='compiled IR',type='compiled'){
 return `<div class="detailHeader"><div><div class="eyebrow">${esc(kicker)}</div><h2>${esc(title)}</h2><p>${esc(description)}</p></div>${badge(source,type)}</div>`;
}
function statsHTML(s){if(!s)return '<p>Weight statistics unavailable for this unit.</p>';return `<div class="metrics"><div><b>${number(s[1])}</b><small>NONZERO COEFFICIENTS</small></div><div><b>${s[4].toExponential(2)}</b><small>WEIGHT L2</small></div></div><p>Min ${s[2].toExponential(3)} · max ${s[3].toExponential(3)}<br>Exact stock emitter projection rows, including constant ballast splitting. Zeros excluded from head coordinate counts.</p>`;}
function crumbs(){
 const app=state.manifestIndex===null?state.snapshot?.app:null,a=app?.arrays[state.array];
 const arrayAddress=a?a.address+(a.shape?.length===3?(state.innerLayer*a.shape[1]+state.innerPos)*a.shape[2]*(a.format==='b'?1:2):0):null;
 const address=state.level===4?arrayAddress:state.level===2?state.memory:null;
 let linked=null;
 if(address>=0x1000000&&address<0x1010000){const word=(address-0x1000000)>>>2,bank=(word&15)|((word>>>12)<<4);for(const [L,d] of details.entries()){const j=d?.heads.findIndex(h=>h.name===`ram${bank}`);if(j>=0){linked={layer:L,index:j,kind:'head',value:(word>>>4)&1};break;}}}
 const layer=linked?.layer??state.layer,head=linked?details[layer].heads[linked.index]:state.pick?.kind==='head'?details[state.layer]?.heads[state.pick.index]:null;
 const entries=[['Network',0]];if(state.level>=1)entries.push([`Layer ${layer}`,1]);
 if(head&&state.level>=1)entries.push([head.name,1]);
 if(address!==null&&linked)entries.push([hex(address),2]);
 if(state.level>=2)entries.push(['CPU',2]);if(state.level>=3)entries.push([currentFunction()?.name??'Program',3]);
 if(state.level>=4){
  entries.push([state.manifestIndex===null?(state.record?.application?.kind??state.record?.decoder?.title??state.record?.decoder?.kind??'tiny LLM'):state.manifests[state.manifestIndex]?.manifest.demo??'Demo',4]);
  if(a?.shape?.length===3){
   entries.push([`${state.array}[layer ${state.innerLayer}][pos ${state.innerPos}]`,4]);
   const rows=[...outputAt(state.record,state.offset).matchAll(/@(\d+) (\d+) (\d+) (\d+) (\d+)\n>([\s\S]*?)\n(?=@|$)/g)];
   const row=rows.find(m=>+m[1]===state.innerPos),previous=rows.find(m=>+m[1]===state.innerPos-1);
   if(row)entries.push([`input token #${row[2]}${previous&&previous[3]===row[2]?' '+previous[6]:''}`,4]);
  }
 }
 $('breadcrumb').innerHTML=entries.map(([s,l])=>`<button data-level="${l}">${esc(s)}</button>`).join('<span>›</span>');
 $('breadcrumb').querySelectorAll('button').forEach(b=>b.onclick=()=>{if(+b.dataset.level===1&&linked){state.layer=linked.layer;state.pick=linked;}if(+b.dataset.level===2&&address!==null)state.memory=address;setLevel(+b.dataset.level);});
}
function setLevel(level){
 state.gallery=false;stopDemoAnimation();
 state.level=Math.max(0,Math.min(4,level));$('zoom').value=state.level;
 $('levels').querySelectorAll('button').forEach((b,i)=>b.classList.toggle('active',i===state.level));
 const show=state.level===0;
 for(const id of ['network','mapInfo','mapTools','mapCaption'])$(id).style.display=show?'':'none';
 $('detail').style.display=show?'none':'block';$('tooltip').style.display='none';
 if(!show){$('detail').style.animation='none';void $('detail').offsetWidth;$('detail').style.animation='';renderDetail();}
 crumbs();history.replaceState(null,'',`#${state.level}`);
}
function pickPoint(p){if(!p)return;state.layer=p.layer;state.pick=p;setLevel(1);}
function hover(p,x,y){const t=$('tooltip');if(!p){t.style.display='none';return;}t.innerHTML=`<strong>${esc(p.name)}</strong><br>${esc(roleName(p.role))} · Layer ${p.layer}<br>${badge('compiled IR','compiled')}<br>${p.stats?`${number(p.stats[1])} nonzero coefficients · L2 ${p.stats[4].toExponential(2)}`:p.kind==='residual'?'Physical slot with sublayer-specific owners':'Zero-weight stock capacity'}<br><span class="tiny">Click to inspect exact semantics</span>`;t.style.display='block';t.style.left=Math.min(x+15,$('viewport').clientWidth-285)+'px';t.style.top=Math.min(y+10,$('viewport').clientHeight-160)+'px';}
function unitButton(unit,index,L){return `<button class="unitRow" data-unit="${index}" data-layer="${L}"><i class="dot" style="background:${roleColor(unit[1])}"></i><code>${esc(unit[0])}</code><small>L${L} · ${index}</small></button>`;}
function bindUnits(root){root.querySelectorAll('[data-unit]').forEach(b=>b.onclick=()=>pickPoint({kind:'unit',index:+b.dataset.unit,layer:+b.dataset.layer}));}
function search(){
 const q=$('search').value.toLowerCase(),role=$('roleFilter').value,L=$('layerFilter').value;
 Object.assign(map,{query:q,role,layer:L});map.draw();
 if(!state.list&&!q&&!role&&L===''){$('results').innerHTML='';return;}
 let total=0,html='';
 for(const [i,d] of details.entries()){
  if(!d||L!==''&&+L!==i)continue;
  for(const [u,unit] of d.units.entries())if((!q||unit[0].toLowerCase().includes(q))&&(!role||unit[1]===role)){total++;if(total>state.page*100&&total<=(state.page+1)*100)html+=unitButton(unit,u,i);}
  for(const [j,h]of d.heads.entries())if((!q||h.name.toLowerCase().includes(q))&&(!role||h.role===role)){total++;if(total>state.page*100&&total<=(state.page+1)*100)html+=`<button class="unitRow" data-head="${j}" data-layer="${i}"><i class="dot" style="background:${roleColor(h.role)}"></i><code>${esc(h.name)}</code><small>L${i} · HEAD</small></button>`;}
 }
 $('results').innerHTML=`<p>${number(total)} matching units. Page ${state.page+1}.</p><div class="row"><button id="previousPage" ${state.page===0?'disabled':''}>←</button><button id="nextPage" ${(state.page+1)*100>=total?'disabled':''}>→</button></div>`+html;bindUnits($('results'));$('previousPage').onclick=()=>{state.page=Math.max(0,state.page-1);search();};$('nextPage').onclick=()=>{state.page++;search();};
 $('results').querySelectorAll('[data-head]').forEach(b=>b.onclick=()=>pickPoint({kind:'head',index:+b.dataset.head,layer:+b.dataset.layer,value:0}));
}
function expression(terms){return terms.map(([d,w])=>`${w} × ${d===0?'ONE':`slot${d}`}`).join(' + ')||'0';}
function neuralAvailability(){
 if(state.snapshot)return `<div class="notice neuralNotice">${badge('read from neural state','neural')} Exact sample at generated prefix ${number(state.snapshot.offset)}. ${esc(state.snapshot.state_boundary??'Last generated token awaits feedback; S reflects processed feedback tokens.')} ${state.snapshot.verification?.passed?'Independent reference check passed for PC, anchor/count, 65,536 RAM bytes and 96 register values.':''}</div>`;
 const samples=state.record?.snapshots??[];
 return `<div class="notice">No neural-state snapshot at this token offset. Trace-derived PC and output remain available. Registers, RAM and activations are not interpolated.<br>${samples.map(s=>`<button data-offset="${s.offset}">Jump to neural sample ${number(s.offset)}</button>`).join('')}${!samples.length?'<button data-final>Open validated final neural snapshot</button>':''}</div>`;
}
function bindSamples(){
 $('detail').querySelectorAll('[data-offset]').forEach(b=>b.onclick=()=>seek(+b.dataset.offset));
 $('detail').querySelectorAll('[data-final]').forEach(b=>b.onclick=async()=>{await loadRecording('stories260k-final');await seek(state.record.end);});
}
function layerView(){
 const L=state.layer,d=details[L],layer=graph.layers[L],p=state.pick;
 if(layer.zero){$('detail').innerHTML=header('L1 / STOCK ARCHITECTURE',`Layer ${L} · full attention`,layer.note)+`<div class="panel"><h3>All weights are zero</h3><p>8 query heads, 2 KV heads, head dimension 256. The stock HF wrapper appends this layer to count positions. It stores no CPU architectural state.</p><div class="metrics"><div><b>9,216</b><small>ZERO MLP UNITS</small></div><div><b>0</b><small>NONZERO WEIGHTS</small></div></div></div>`;return;}
 if(p?.kind==='layout'){const coordinate=graph.hidden_dimensions[p.index];$('detail').innerHTML=header('L1 / STOCK RESIDUAL LAYOUT',`Hidden coordinate ${p.index}`,coordinate.kind==='ballast'?'Normalization ballast. This carries 2^40 on chip tokens until stock exit cleanup. IR constant ONE is represented across the ballast coordinates.':coordinate.kind==='tau'?'Reserved tau coordinate. It stays zero in this standalone CPU configuration.':'Unused residual padding. No IR value is assigned to this hidden coordinate.')+`<div class="panel"><h3>Emitter layout</h3><p>${graph.ballast} ballast coordinates, one reserved tau coordinate, ${graph.geometry.dims-1} mapped IR slots, and ${graph.hidden-graph.ballast-graph.geometry.dims} unused padding coordinates. The full hidden size is ${graph.hidden}.</p></div>`;return;}
 if(p?.kind==='zero'){ $('detail').innerHTML=header('L1 / UNUSED STOCK CAPACITY','Zero-weight stock unit',`Layer ${L}. This stock capacity is not used by the compiled IR.`)+`<div class="panel"><h3>Known zero weights</h3><p>The stock emitter initializes unused FFN units and head groups to zero. This unit stores no CPU state and implements no compiled arithmetic gate.</p></div>`;return;}
 if(p?.kind==='residual'){residualView(p.index);return;}
 if(p?.kind==='unit'){
  const u=d.units[p.index];if(u[7]?.hidden_coordinate!==undefined){const meta=u[7];$('detail').innerHTML=header('L1 / STOCK EXIT CLEANUP',u[0],`Clears ballast hidden coordinate ${meta.hidden_coordinate} after the chip output is written.`)+`<div class="grid2"><div class="panel"><h3>Exact stock-emitter unit</h3><pre>gate = 128 × ONE\nup[hidden ${meta.hidden_coordinate}] = ${meta.up_weight}\ndown[hidden ${meta.hidden_coordinate}] = -1</pre><p>This emitter-added cleanup is outside the IR residual slot expressions.</p></div><div class="panel">${statsHTML(u[6])}</div></div>`;return;}$('detail').innerHTML=header('L1 / COMPILED SWIGLU UNIT',u[0],`Layer ${L}, unit ${p.index}. ${roleName(u[1])}.`)+`<div class="grid2"><div class="panel"><h3>Exact compiled gate</h3><pre>${esc(`${u[3]} × a × relu(b)\na = ${expression(u[4])}\nb = ${expression(u[5])}`)}</pre><p>Stock form uses silu(128 × b) × a / 128. This is a compiled Boolean or arithmetic gate, not a learned feature label.</p><button id="openDim">Inspect output slot ${u[2]??'ballast'}</button></div><div class="panel"><h3>Emitted weight statistics</h3>${statsHTML(u[6])}</div></div><button id="backLayer">← All layer units</button>`;
  $('openDim').onclick=()=>{if(u[2]!=null){state.pick={kind:'residual',index:u[2],layer:L};renderDetail();}};$('backLayer').onclick=()=>{state.pick=null;renderDetail();};return;
 }
 if(p?.kind==='head'){headView(d.heads[p.index],p.value??0);return;}
 $('detail').innerHTML=header('L1 / COMPILED LAYER',`Layer ${L} · Gated DeltaNet`,'Named recurrent tables and arithmetic gates share the residual stream. Select a head or unit to inspect its exact role.')+`<div class="metrics"><div><b>${d.heads.length}</b><small>ACTIVE KEY HEADS / 16</small></div><div><b>${d.heads.reduce((n,h)=>n+h.values.length,0)}</b><small>LOGICAL VALUE HEADS / 32</small></div><div><b>${number(d.units.length)}</b><small>EMITTED FFN UNITS / 9,216</small></div></div><div class="grid2"><div class="panel"><h3>Recurrent tables</h3><div class="headGrid">${d.heads.map((h,i)=>`<button data-head="${i}" style="border-top:2px solid ${roleColor(h.role)}">${esc(h.name)}<small>${esc(roleName(h.role))}</small></button>`).join('')||'<p>This layer has no active head. All stock head weights are zero.</p>'}</div></div><div class="panel"><h3>Residual lifetime inspector</h3><p>One physical slot can carry several logical values. Ownership is indexed by sublayer.</p><label>Physical slot <input id="dimNumber" class="dimInput" type="number" min="1" max="2112" value="100"></label> <button id="inspectDim">Inspect</button><p>Odd sublayers are attention. Even sublayers are MLP. The ledger records explicit clears before reuse.</p></div></div><div class="panel"><h3>FFN units · first 120 in stock emitter order</h3><div class="scrollList">${d.units.slice(0,120).map((u,i)=>unitButton(u,i,L)).join('')}</div></div>`;
 $('detail').querySelectorAll('[data-head]').forEach(b=>b.onclick=()=>{state.pick={kind:'head',index:+b.dataset.head,layer:L,value:0};renderDetail();crumbs();});bindUnits($('detail'));
 $('inspectDim').onclick=()=>{const n=+$('dimNumber').value;if(n>0&&n<graph.dimensions.length){state.pick={kind:'residual',index:n,layer:L};renderDetail();}};
}
function residualView(dim){
 const owners=graph.dimensions[dim]??[];
 $('detail').innerHTML=header('L1 / RESIDUAL LIFETIMES',`Physical slot ${dim}`,`Hidden coordinate ${graph.physical_to_hidden[dim-1]}. ${owners.length} logical owner${owners.length===1?'':'s'} across the compiled graph.`)+`<div class="panel"><h3>Ownership by sublayer · 0 → 128</h3><div class="lifetimes">${owners.map(o=>`<div class="life"><i style="left:${o.born/128*100}%;width:${Math.max(1,(o.last-o.born)/128*100)}%;background:${roleColor(o.role)}"></i><span>${esc(o.name)} · born ${o.born}, last read ${o.last}, last write ${o.last_write}${o.pinned?' · PINNED':''}</span></div>`).join('')}</div></div><div class="panel"><h3>Explicit handovers</h3><pre>${esc(graph.ledger.handovers.filter(h=>h.slot===dim).map(h=>`logical ${h.old} → ${h.new}; clear at sublayer ${h.clear}; new owner born ${h.born}`).join('\n')||'No reuse handover for this physical slot.')}</pre><p>Compiler liveness. No runtime activation is inferred from this chart.</p></div>`;
}
function headView(h,v){
 const value=h.values[v],slice=state.snapshot?.slices?.[h.name];
 $('detail').innerHTML=header('L1 / RECURRENT STATE TABLE',`${h.name} · value head ${value.value_head}`,`${roleName(h.role)}. Layer ${state.layer}, key head ${h.key_head}. State geometry 128 rows × 128 columns.`)+neuralAvailability()+`<div class="grid2"><div class="panel"><h3>S tensor · ${slice&&v===0?'actual neural sample':'not sampled at this offset'}</h3>${slice&&v===0?'<canvas id="headHeat" class="heatmap"></canvas><div class="heatLegend"><span>column 0 → 127</span><span>row 0 → 127</span></div>':'<div class="empty">No captured S tensor for this head and offset.<br>The structural mapping below remains available.</div>'}<p>${esc(value.axis.rows)}<br>${esc(value.axis.columns)}<br>${esc(value.axis.encoding)}</p>${value.axis.addresses?`<p>Row 0 → <button id="addressJump">${hex(value.axis.addresses[0])}</button><br>Row 127 → ${hex(value.axis.addresses[127])}</p>`:''}</div><div><div class="panel"><h3>Stock emitter coefficients</h3>${statsHTML(value.stats)}</div><div class="panel"><h3>Output routing</h3><pre>${esc(value.output.slice(0,12).map(([col,dim,w])=>`S column ${col} → slot${dim} × ${w}`).join('\n'))}</pre><p>Write gate ${value.beta===null?'always enabled':esc(expression(value.beta))}<br>Decay selector ${esc(expression(value.decay))}</p><div class="headActions">${h.values.map((_,i)=>`<button data-vh="${i}">Value ${i}</button>`).join('')}<button id="openCPU">Follow into CPU →</button></div></div></div></div>`;
 if(slice&&v===0){heatmap($('headHeat'),slice.values.flat(),128,128,{ballast:true});$('headHeat').onmousemove=e=>{const b=e.target.getBoundingClientRect(),col=Math.min(127,Math.floor((e.clientX-b.left)/b.width*128)),row=Math.min(127,Math.floor((e.clientY-b.top)/b.height*128));e.target.title=`row ${row}${value.axis.addresses?` · ${hex(value.axis.addresses[row])}`:''}, column ${col} = ${slice.values[row][col]} · read from neural state`;};}
 $('detail').querySelectorAll('[data-vh]').forEach(b=>b.onclick=()=>{state.pick.value=+b.dataset.vh;renderDetail();});$('openCPU').onclick=()=>setLevel(2);
 if($('addressJump'))$('addressJump').onclick=()=>{state.memory=value.axis.addresses[0];setLevel(2);};bindSamples();
}
function tracePC(){if(!state.trace?.length)return 0;const i=Math.max(0,Math.min(state.record.end-state.record.start-1,state.offset-state.record.start-1));return state.trace[i*2+1];}
function currentFunction(){const pc=tracePC();return state.program?.symbols.find(s=>s.kind===2&&pc>=s.address&&pc<s.address+s.size);}
function tokenRows(){
 if(!state.trace)return '';
 const center=Math.max(0,state.offset-state.record.start-1),rows=[];
 for(let i=Math.max(0,center-8);i<Math.min(state.trace.length/2,center+13);i++){
  const t=state.trace[i*2],[f,p]=tokenFamily(state.record,t);let value='';
  if(['F0','F1','F2'].includes(f)){
   const before=n=>tokenFamily(state.record,state.trace[Math.max(0,i-n)*2])[1];
   const n=f==='F0'?(p|(p&16384?0xffff8000:0)):f==='F1'?(before(1)|(p<<15)|(p&16384?0xc0000000:0)):(before(2)|(before(1)<<15)|(p<<30));value=hex(n);
  }else if(f==='BYTE')value=p<256?JSON.stringify(String.fromCharCode(p)):'special BYTE';else if(f==='J'||f==='R')value=hex(0x40000+4*p);else if(f==='FLT')value=['illegal','ebreak','misaligned load','misaligned store','load out of range','store out of range','bad jump','ecall'][p];
  rows.push(`<tr class="${i===center?'selected':''}"><td><button data-seek="${state.record.start+i+1}">${number(state.record.start+i+1)}</button></td><td>${hex(state.trace[i*2+1])}</td><td>${f} ${p}</td><td>${esc(value)}</td></tr>`);
 }
 return `<table><thead><tr><th>Generated prefix</th><th>Instruction PC</th><th>Token</th><th>Record value</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
}
function cpuView(){
 const s=state.snapshot,pc=s?.pc??tracePC(),inst=state.program?.instructions.find(i=>i[0]===pc);
 const names=['zero','ra','sp','gp','tp','t0','t1','t2','s0','s1','a0','a1','a2','a3','a4','a5','a6','a7','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','t3','t4','t5','t6'];
 $('detail').innerHTML=header('L2 / ARCHITECTURAL MACHINE','The CPU inside the network.','Instruction PC comes from the generated token protocol. Register and memory values appear only when a recurrent-state sample exists.',s?'read from neural state':'decoded from generated trace',s?'neural':'trace')+`<div class="metrics"><div><b>${hex(pc)}</b><small>PC · ${s?'NEURAL S':'GENERATED TRACE'}</small></div><div><b>${esc(inst?.[2]??'—')}</b><small>CURRENT INSTRUCTION · ELF</small></div><div><b>${number(state.record?.bins.find((b,i,a)=>state.offset>=b.offset&&state.offset<(a[i+1]?.offset??Infinity))?.ndot??0)}</b><small>NDOT TOKEN POSITIONS / BIN</small></div></div><div class="panel"><pre>${esc(instructionText(inst))}</pre></div>`+neuralAvailability()+`<div class="grid2"><div><div class="panel"><h3>Register file · three recurrent copies checked</h3>${s?`<div class="registers">${s.registers['regs.rs1'].map((r,i)=>`<button data-register="${r}"><span>x${i} ${names[i]}</span><b>${hex(r)}</b></button>`).join('')}</div>`:'<div class="empty">Registers not sampled at this offset.</div>'}</div><div class="panel"><h3>Guest address map</h3><table><tr><td>Code ROM</td><td>0x00040000</td><td>256 KiB</td></tr><tr><td>RAM</td><td>0x01000000</td><td>64 KiB</td></tr><tr><td>DROM</td><td>0x02000000</td><td>4 MiB</td></tr></table><p>NDOT4 reads four signed DROM bytes and four signed RAM halfwords, accumulates into rd modulo 2³², and advances PC by four.</p><button id="cpuProgram">Follow instruction into program →</button></div><div class="panel"><h3>RAM inspector</h3><div class="arraySelect"><input id="memoryAddress" class="dimInput" style="width:130px" value="${hex(state.memory)}" aria-label="RAM address"><button id="readMemory">Read 128 bytes</button></div><div id="memoryView"></div></div></div><div class="panel"><h3>Generated token stream</h3><div class="codeview">${tokenRows()}</div><p>The final token commits the instruction. V, QS and QD are partial records. R refreshes the PC anchor without executing an instruction.</p></div></div>`;
 bindSamples();$('cpuProgram').onclick=()=>setLevel(3);$('readMemory').onclick=()=>{const a=Number($('memoryAddress').value);if(Number.isInteger(a)&&a>=0x1000000&&a<0x1010000){state.memory=a;renderMemory();}else fail(Error('Choose an address inside the 64 KiB RAM region.'));};renderMemory();
 $('detail').querySelectorAll('[data-register]').forEach(b=>b.onclick=()=>{const a=+b.dataset.register;if(a>=0x1000000&&a<0x1010000){state.memory=a;renderMemory();$('memoryAddress').value=hex(a);}});bindSeek();
}
function symbolAt(address){return state.program?.symbols.find(s=>s.kind===1&&address>=s.address&&address<s.address+s.size);}
function renderMemory(){
 const s=state.snapshot;if(!s){$('memoryView').innerHTML='<p>No neural RAM sample at this offset.</p>';return;}
 const ram=Uint8Array.from(s.ram_hex.match(/../g),h=>parseInt(h,16));let html='';
 const off=state.memory-0x1000000;
 for(let i=off;i<Math.min(65536,off+128);i+=16){const sym=symbolAt(0x1000000+i);html+=`<div class="memrows"><button data-address="${0x1000000+i}">${hex(0x1000000+i)}</button> <code>${Array.from(ram.slice(i,i+16),n=>n.toString(16).padStart(2,'0')).join(' ')}</code>${sym?`<div><button class="symbol" data-app="${esc(sym.name)}">${esc(sym.name)} + ${i+0x1000000-sym.address}</button></div>`:''}</div>`;}
 $('memoryView').innerHTML=html;$('memoryView').querySelectorAll('[data-app]').forEach(b=>b.onclick=()=>{state.array=b.dataset.app;setLevel(4);});
}
function bindSeek(){ $('detail').querySelectorAll('[data-seek]').forEach(b=>b.onclick=()=>seek(+b.dataset.seek));}
function programView(){
 const bins=state.record?.bins??[],totals=new Map();for(const b of bins)for(const [f,n]of b.hot)totals.set(f,(totals.get(f)??0)+n);
 const ranked=[...totals].sort((a,b)=>b[1]-a[1]),max=ranked[0]?.[1]??1,pc=tracePC();const ins=state.program?.instructions??[],center=ins.findIndex(i=>i[0]===pc);
 $('detail').innerHTML=header('L3 / PROGRAM','A transformer executing a transformer.','ELF symbols name the guest functions. Activity counts token positions in each function; these are not retired-instruction counts or GPU-time measurements.','decoded from generated trace','trace')+`<div class="panel"><h3>Phase timeline · each cell is a recorded token bin</h3><div class="phaseGrid">${bins.map(b=>`<button data-seek="${b.offset+1}" title="${esc(`${number(b.offset)} · ${b.hot.map(([f,n])=>`${f} ${n}`).join(', ')}`)}" style="background:${b.ndot?'#d282ac':'#477d83'};opacity:${.3+.7*(b.hot[0]?.[1]??0)/b.tokens}"></button>`).join('')}</div><p>Pink bins include NDOT instructions. Teal bins cover scalar work. Hover a bin for its hottest functions and click to scrub.</p></div><div class="grid2"><div class="panel"><h3>Hot functions · top-five-per-bin aggregation</h3>${ranked.slice(0,14).map(([f,n])=>`<div class="functionBar"><span style="width:140px">${esc(f)}</span><span class="bar" style="width:${n/max*35}%"></span><small>${number(n)}</small></div>`).join('')}<p>Phase names remain compiler symbols. An exact dynamic call stack is not reconstructed from this token-only recording.</p></div><div class="panel"><h3>${esc(currentFunction()?.name??'Current instruction')} · static ELF disassembly</h3><div class="codeview"><table>${ins.slice(Math.max(0,center-7),Math.max(20,center+18)).map(i=>`<tr class="${i[0]===pc?'selected':''}"><td>${hex(i[0])}</td><td>${esc(instructionText(i))}</td></tr>`).join('')}</table></div><button id="openApplication">Inside the guest application →</button></div></div><div class="panel"><h3>Guest RAM objects · ELF symbol table</h3><table>${(state.program?.symbols??[]).filter(s=>s.kind===1&&s.address>=0x1000000&&s.address<0x1010000).map(s=>`<tr><td><button class="symbol" data-memory="${s.address}">${esc(s.name)}</button></td><td>${hex(s.address)}</td><td>${number(s.size)} bytes</td></tr>`).join('')}</table></div>`;
 bindSeek();$('openApplication').onclick=()=>setLevel(4);$('detail').querySelectorAll('[data-memory]').forEach(b=>b.onclick=()=>{state.memory=+b.dataset.memory;setLevel(2);});
}
function applicationView(){
 if(state.record?.application){demoView(state,{seek,setLevel,header,availability:neuralAvailability,bindSamples});return;}
 if(state.manifestIndex!==null){manifestView();return;}
 const output=state.record?outputAt(state.record,state.offset):'',pieces=[...output.matchAll(/@(\d+) (\d+) (\d+) (\d+) (\d+)\n>([\s\S]*?)\n(?=@|$)/g)],story=pieces.map(m=>m[6]).join('');
 const app=state.snapshot?.app,pack=app?.pack??state.record?.pack;
 $('detail').innerHTML=header('L4 / GUEST APPLICATION','The small language model inside.','stories260K is an integer transformer in guest memory. Its layers are software objects, separate from the outer Qwen3.5 circuit.','decoded from generated trace','trace')+`<div class="panel"><h3>Text emitted so far · actual generated BYTE records</h3><div class="story">${esc(story)||'<span class="tiny">No complete decoded text record yet in this window.</span>'}</div>${pieces.map(m=>`<span class="tokenChip" title="position ${m[1]}, input ${m[2]}, logit hash ${m[4]}, clips ${m[5]}">#${m[3]} ${esc(m[6])}</span>`).join('')}<p>${pieces.length} complete output positions in the retained generated output context. Diagnostic bytes and partial lines appear below.</p></div>${pack?`<div class="metrics"><div><b>${pack.layers}</b><small>INNER LAYERS</small></div><div><b>${pack.dim}</b><small>MODEL DIMENSIONS</small></div><div><b>${pack.heads}</b><small>ATTENTION HEADS</small></div><div><b>${pack.kv_bits}</b><small>KV BITS / VALUE</small></div></div><div class="appLayers">${Array.from({length:pack.layers},(_,i)=>`<button data-inner="${i}" class="${i===state.innerLayer?'active':''}">Layer ${i}<small>Q/K/V → attention → FFN</small></button>`).join('')}</div>`:''}`+neuralAvailability()+`<div class="grid2"><div class="panel"><h3>Guest activations and KV cache</h3>${app?`<div class="arraySelect"><select id="arrayName" aria-label="Guest array">${Object.keys(app.arrays).map(n=>`<option ${n===state.array?'selected':''}>${n}</option>`).join('')}</select><label class="tiny">Position <input id="innerPos" class="dimInput" type="number" min="0" max="${pack.context-1}" value="${state.innerPos}"></label></div><canvas id="appHeat" class="heatmap"></canvas><div id="arrayInfo" class="refLine"></div><p>Activations are shared scratch buffers. Their values are the captured sample, not a history of every inner layer. KV cache has explicit [layer][position][dimension] addressing.</p>`:'<div class="empty">No neural RAM observation at this offset.<br>Activations, logits and KV cache are unavailable.</div>'}</div><div class="panel"><h3>Raw guest output</h3><pre>${esc(output)||'No BYTE records observed yet.'}</pre><p>Token IDs and diagnostic logit hashes above come from actual generated output. They are not inferred from a reference run.</p></div></div><div class="panel"><h3>Other application decoders</h3><p>Doom framebuffer strips, Life, MNIST, tic-tac-toe and SHA-256 use explicit manifest RAM ranges. Private pixels load only from the locally mounted recordings directory.</p><button id="refreshManifests">Refresh manifests</button><div id="manifestList"></div></div>`;
 $('detail').querySelectorAll('[data-inner]').forEach(b=>b.onclick=()=>{state.innerLayer=+b.dataset.inner;renderDetail();});bindSamples();
 if(app){if(!app.arrays[state.array])state.array=Object.keys(app.arrays)[0];$('arrayName').value=state.array;$('arrayName').onchange=()=>{state.array=$('arrayName').value;renderArray();crumbs();};$('innerPos').onchange=()=>{state.innerPos=Math.max(0,Math.min(pack.context-1,+$('innerPos').value||0));renderArray();crumbs();};renderArray();}
 renderManifests();$('refreshManifests').onclick=async()=>{try{state.manifests=await json('/api/manifests');renderManifests();}catch(e){fail(e);}};
}
function renderArray(){
 const app=state.snapshot?.app,a=app?.arrays[state.array];if(!a)return;
 let values=a.values,address=a.address,shape=a.shape;
 if(shape.length===3){const [,context,dim]=shape;const start=(state.innerLayer*context+state.innerPos)*dim;values=a.values.slice(start,start+dim);address+=start*(a.format==='b'?1:2);}
 const cols=Math.min(32,values.length),rows=Math.ceil(values.length/cols);heatmap($('appHeat'),values,rows,cols);
 $('arrayInfo').innerHTML=`${badge('read from neural state','neural')}<br>${esc(state.array)}${shape.length===3?`[layer ${state.innerLayer}][pos ${state.innerPos}]`:''} · ${hex(address)} · ${values.length} values · ${a.format} · ${a.fractional_bits===null?'dynamic quantization':`Q${a.fractional_bits}`}<br>First values ${values.slice(0,16).join(', ')}<br>Sample offset ${number(app.offset)}. No per-position interpolation.`;
 $('appHeat').onmousemove=e=>{const b=e.target.getBoundingClientRect(),x=Math.floor((e.clientX-b.left)/b.width*cols),y=Math.floor((e.clientY-b.top)/b.height*rows),i=y*cols+x;e.target.title=`${state.array}[${i}] = ${values[i]??'padding'} · ${hex(address+i*(a.format==='i'?4:a.format==='h'?2:1))}`;};
}
function renderManifests(){
 const list=$('manifestList');if(!list)return;
 list.innerHTML=state.manifests.length?state.manifests.map(({file,manifest},i)=>`<button class="unitRow" data-demo="${i}"><code>${esc(manifest.demo??manifest.title??file)}</code><small>${esc(manifest.status??'manifest')}</small></button>`).join(''):'<p>No showcase manifests have appeared yet. See README for the recording/decoder adapter contract.</p>';
 list.querySelectorAll('[data-demo]').forEach(b=>b.onclick=()=>openDemo(+b.dataset.demo));
}
function manifestView(){
 const {file,manifest:m}=state.manifests[state.manifestIndex];
 const reference=m.output_source?.includes('reference'),source=reference?'reference':'manifest report';
 $('detail').innerHTML=header('L4 / SHOWCASE MANIFEST',m.demo??file,m.network??'External demo manifest',source,reference?'reference':'trace')+`<div class="arraySelect"><button id="backLLM">← Program gallery</button><select id="chooseDemo" aria-label="Showcase demo">${state.manifests.map((x,i)=>`<option value="${i}" ${i===state.manifestIndex?'selected':''}>${esc(x.manifest.demo??x.file)}</option>`).join('')}</select></div><div class="notice">${esc(m.status)}. Source: ${esc(m.output_source)}. ${reference?'These outputs are reference evidence. No neural run is claimed.':'This is a published evidence manifest, not a loaded neural RAM sample.'}</div><div class="grid2"><div class="panel"><h3>Application output</h3><canvas id="manifestCanvas" class="heatmap" style="display:none"></canvas><div id="manifestFrames" class="arraySelect"></div><pre id="manifestText"></pre></div><div class="panel"><h3>Evidence and input bindings</h3><p>ELF ${esc(m.elf_sha256)}<br>Network weight hash ${esc(m.network_weight_sha256)}<br>${esc(m.weight_hash_scope??'')}</p><pre>${esc((m.evidence_paths??[]).join('\n'))}</pre><p>${esc(m.stock_note??'')}<br>${esc(m.timing_scope??'')}</p>${m.explorer_recording?'<button id="loadDemoRecording">Load local neural recording</button>':'<p>This manifest has no explorer_recording path. Waiting for a compatible local recording. No guest RAM address or framebuffer pixels are guessed.</p>'}</div></div>`;
 $('backLLM').onclick=showGallery;$('chooseDemo').onchange=()=>openDemo(+$('chooseDemo').value);$('manifestText').textContent=m.outputs??'No output in manifest.';
 const frames=[];
 if(m.demo==='life')for(const match of (m.outputs??'').matchAll(/Generation (\d+)\n((?:[.#]+\n)+)/g)){const lines=match[2].trim().split('\n');frames.push({label:'Generation '+match[1],rows:lines.length,cols:lines[0].length,values:[...lines.join('')].map(c=>c==='#'?1:0)});}
 if(m.demo==='tictactoe')for(const match of (m.outputs??'').matchAll(/(?:^|\n)([XO.]{3})\n([XO.]{3})\n([XO.]{3})/g))frames.push({label:'Board '+frames.length,rows:3,cols:3,values:[...(match[1]+match[2]+match[3])].map(c=>c==='X'?1:c==='O'?-1:0)});
 if(m.demo?.startsWith('mnist'))for(const match of (m.outputs??'').matchAll(/MNIST test (\d+)\n([\s\S]*?)Label (\d+) prediction (\d+)/g)){const lines=match[2].replace(/\n$/,'').split('\n');if(lines.every(l=>l.length===lines[0].length))frames.push({label:`Test ${match[1]} · label ${match[3]} · prediction ${match[4]}`,rows:lines.length,cols:lines[0].length,values:[...lines.join('')].map(c=>c==='#'?1:c==='+'?.5:c==='.'?.2:0)});}
 if(frames.length){const draw=i=>{const f=frames[i];$('manifestCanvas').style.display='block';heatmap($('manifestCanvas'),f.values,f.rows,f.cols);};$('manifestFrames').innerHTML=frames.map((f,i)=>`<button data-frame="${i}">${esc(f.label)}</button>`).join('');$('manifestFrames').querySelectorAll('button').forEach(b=>b.onclick=()=>draw(+b.dataset.frame));draw(0);}
 if($('loadDemoRecording'))$('loadDemoRecording').onclick=async()=>{try{await loadRecordingURL(new URL(m.explorer_recording,location.origin+'/recordings/').href);state.manifestIndex=null;renderDetail();}catch(e){fail(e);}};
}
function renderGuestDecoder(){
 const decoder=state.record?.decoder;if(!decoder)return false;
 $('detail').innerHTML=header('L4 / APPLICATION DECODER',decoder.title??decoder.kind,'RAM ranges and pixel format come from the recording manifest.','read from neural state','neural')+neuralAvailability()+`<div class="panel"><canvas id="guestCanvas" class="heatmap"></canvas><pre id="guestText"></pre></div>`;
 bindSamples();if(!state.snapshot)return true;
 try{
  const ram=Uint8Array.from(state.snapshot.ram_hex.match(/../g),h=>parseInt(h,16));
  if(decoder.kind==='sha256'){
   const base=decoder.address-0x1000000;if(base<0||base+32>ram.length)throw Error('Digest outside captured RAM');$('guestText').textContent=Array.from(ram.slice(base,base+32),b=>b.toString(16).padStart(2,'0')).join('');$('guestCanvas').style.display='none';
  }else drawGuest($('guestCanvas'),ram,decoder);
 }catch(e){$('guestText').textContent=e.message;}
 return true;
}
function renderDetail(){
 if(state.gallery){gallery(state.manifests,openDemo);return;}
 if(!state.record&&[2,3].includes(state.level)){$('detail').innerHTML=header('HISTORICAL EVIDENCE','No compatible recording loaded.','This manifest uses a different ELF or network. Its L2 and L3 data are unavailable locally.','manifest report','reference');crumbs();return;}
 if(state.level===1)layerView();if(state.level===2)cpuView();if(state.level===3)programView();if(state.level===4&&!renderGuestDecoder())applicationView();crumbs();
}
async function loadRecording(id){const e=catalog.recordings.find(r=>r.id===id);if(!e)throw Error('Recording not present in this catalog');$('recording').value=id;return loadRecordingURL(new URL(e.url,new URL(state.base,location.href)).href);}
async function loadRecordingURL(url){
 const version=++loadVersion;const record=await json(url);if(record.schema!=='qwenputer.recording.v1')throw Error('Unsupported recording schema');const base=new URL('.',url);
 const [raw,program]=await Promise.all([bytes(new URL(record.trace,base).href),json(new URL(record.program,base).href)]);
 if(raw.length%8)throw Error('Trace must contain [token, PC] uint32 pairs');const trace=new Uint32Array(raw.buffer,raw.byteOffset,raw.byteLength/4);
 if(trace.length/2!==record.end-record.start)throw Error('Trace length does not match recording offsets');
 if(version!==loadVersion)return;
 Object.assign(state,{record,trace,program,recordBase:base,snapshot:null,manifestIndex:null,offset:record.start+1});$('scrubber').min=record.start+1;$('scrubber').max=record.end;
 await seek(record.snapshots.some(s=>s.offset===record.end)?record.end:record.start+1);drawTimeline();
}
async function seek(offset){
 if(!state.record)return;state.offset=Math.max(state.record.start+1,Math.min(state.record.end,offset));$('scrubber').value=state.offset;state.snapshot=null;
 const found=state.record.snapshots.find(s=>s.offset===state.offset),selected=state.offset,record=state.record;
 if(found){const url=new URL(found.path,state.recordBase).href;if(!cache.has(url))cache.set(url,json(url));const sample=await cache.get(url);if(selected!==state.offset||record!==state.record)return;state.snapshot=sample;}
 $('timeLabel').textContent=`${number(state.offset)} / ${number(state.record.end)}`;
 $('timelineNote').textContent=`${state.record.provenance}. ${state.snapshot?'Exact neural sample. Last generated token pending feedback.':'No neural snapshot at this offset.'} ${state.record.binding?.reference_used==='comparison only'?'Reference used for comparison only.':''}`;
 if(state.level)renderDetail();drawTimeline();
}
function drawTimeline(){
 if(!state.record)return;const c=$('timelineCanvas'),ctx=c.getContext('2d');c.width=Math.max(1,c.clientWidth*2);c.height=72;const bins=state.record.bins;
 ctx.clearRect(0,0,c.width,c.height);bins.forEach((b,i)=>{const width=c.width/bins.length;ctx.fillStyle=b.ndot?'#be789e':'#548b91';ctx.globalAlpha=.35+.65*(b.hot[0]?.[1]??0)/b.tokens;ctx.fillRect(i*width,72-(b.hot[0]?.[1]??0)/b.tokens*60,width-1,60);});ctx.globalAlpha=1;
 for(const s of state.record.snapshots){const x=(s.offset-state.record.start)/(state.record.end-state.record.start)*c.width;ctx.fillStyle='#7ed6d1';ctx.fillRect(x-2,0,4,8);}
 const x=(state.offset-state.record.start)/(state.record.end-state.record.start)*c.width;ctx.fillStyle='#e9bc68';ctx.fillRect(x-1,0,2,72);
}
async function init(){
 $('levels').innerHTML=LEVELS.map((s,i)=>`<button data-level="${i}"><small>L${i}</small>${s}</button>`).join('');$('levels').querySelectorAll('button').forEach(b=>b.onclick=()=>setLevel(+b.dataset.level));
 graph=await json('data/graph.json.gz');details=await Promise.all(graph.layers.map(l=>l.detail?json(l.detail):null));
 $('counts').innerHTML=`<div><b>65</b><small>layers</small></div><div><b>${number(graph.hidden)}</b><small>hidden coordinates</small></div><div><b>1,247</b><small>value heads</small></div>`;
 const counts={};for(const l of graph.layers)for(const [r,n]of Object.entries(l.roles))counts[r]=(counts[r]??0)+n;
 counts.unused=graph.layers.reduce((n,l)=>n+(l.inactive_ffn_units??0)+(l.inactive_value_heads??0),0);
 $('legend').innerHTML=Object.entries(graph.roles).map(([r,[name,color]])=>`<div class="legendItem" data-role="${r}"><i class="dot" style="background:${color}"></i>${esc(name)}<span>${number(counts[r]??0)}</span></div>`).join('');
 for(const [r,[name]]of Object.entries(graph.roles))$('roleFilter').add(new Option(name,r));for(let i=0;i<65;i++)$('layerFilter').add(new Option(`Layer ${i}`,i));
 map=new NetworkMap($('network'),graph,details,hover,pickPoint);
 $('legend').querySelectorAll('[data-role]').forEach(b=>b.onclick=()=>{$('roleFilter').value=$('roleFilter').value===b.dataset.role?'':b.dataset.role;search();});
 ['search','roleFilter','layerFilter'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',()=>{state.page=0;search();}));$('pointMode').onchange=()=>{map.mode=$('pointMode').value;map.makePoints();};$('resetCamera').onclick=()=>map.reset();
 $('listTab').onclick=()=>{state.list=true;$('listTab').classList.add('active');$('mapTab').classList.remove('active');search();};$('mapTab').onclick=()=>{state.list=false;$('mapTab').classList.add('active');$('listTab').classList.remove('active');search();setLevel(0);};
 $('brand').onclick=e=>{e.preventDefault();showGallery();};$('galleryNav').onclick=showGallery;$('zoom').oninput=()=>{const z=+$('zoom').value,next=Math.round(z);if(next!==state.level)setLevel(next);$('zoom').value=z;if(next===0)map.animateZoom(1+z*1.3);else $('detail').style.transform=`scale(${1+(z-next)*.08})`;};$('zoom').onchange=()=>{$('detail').style.transform='';$('zoom').value=state.level;};
 $('scrubber').oninput=()=>seek(+$('scrubber').value).catch(fail);$('recording').onchange=()=>loadRecording($('recording').value).catch(fail);
 $('timelineCanvas').onclick=e=>{if(!state.record)return;const b=e.target.getBoundingClientRect();seek(Math.round(state.record.start+(e.clientX-b.left)/b.width*(state.record.end-state.record.start))).catch(fail);};
 $('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Ⅱ':'▶';};
 setInterval(()=>{if(playing&&state.record){const next=Math.min(state.record.end,state.offset+Math.max(1,Math.floor((state.record.end-state.record.start)/600)));seek(next).catch(fail);if(next===state.record.end){playing=false;$('play').textContent='▶';}}},100);
 $('localCatalog').onchange=async e=>{try{const value=JSON.parse(await e.target.files[0].text());if(value.schema!=='qwenputer.catalog.v1')throw Error('Expected recording catalog v1');catalog=value;state.base='recordings/';populateCatalog();await loadRecording(catalog.recordings[0].id);}catch(e){fail(e);}};
 catalog=await json('recordings/index.json');populateCatalog();await loadRecording(catalog.recordings.find(r=>r.id==='stories260k-position-0')?.id??catalog.recordings[0].id);
 try{state.manifests=await json('/api/manifests');}catch{state.manifests=[];}
 if(!location.hash||location.hash==='#gallery')showGallery();else setLevel(Math.max(0,Math.min(4,Number(location.hash.slice(1))||0)));window.explorer={state,graph,details,map,seek,setLevel,loadRecording,showGallery,openDemo};document.documentElement.dataset.ready='true';
}
function showGallery(){
 setLevel(0);state.gallery=true;
 for(const id of ['network','mapInfo','mapTools','mapCaption'])$(id).style.display='none';
 $('detail').style.display='block';$('detail').scrollTop=0;
 gallery(state.manifests,openDemo);$('breadcrumb').innerHTML='<span>Program gallery</span>';history.replaceState(null,'','#gallery');
}
async function openDemo(index){
 try{
  stopDemoAnimation();const {manifest:m}=state.manifests[index];
  const entry=catalog.recordings.find(r=>r.id==='showcase-'+m.demo)??catalog.recordings.find(r=>r.elf_sha256===m.elf_sha256)??(m.demo.includes('38-position')?catalog.recordings.find(r=>r.id==='stories260k-final'):null);
  if(m.explorer_recording)await loadRecordingURL(new URL(m.explorer_recording,location.origin+'/recordings/').href);
  else if(entry)await loadRecording(entry.id);
  else{++loadVersion;Object.assign(state,{record:null,trace:null,program:null,snapshot:null,manifestIndex:index});$('recording').value='';$('timeLabel').textContent='Manifest evidence only';$('timelineNote').textContent='No compatible local recording. No data from another ELF is shown.';$('timelineCanvas').getContext('2d').clearRect(0,0,$('timelineCanvas').width,$('timelineCanvas').height);}
  if(state.program&&state.program.elf_sha256!==m.elf_sha256)throw Error('Manifest and recording ELF mismatch');
  if(state.record)state.manifestIndex=null;
  setLevel(4);$('detail').scrollTop=0;
 }catch(e){fail(e);}
}
function populateCatalog(){ $('recording').innerHTML=catalog.recordings.map(r=>`<option value="${esc(r.id)}">${esc(r.title)} · ${number(r.tokens)} tokens</option>`).join('');}
init().catch(e=>{fail(e);$('detail').style.display='block';$('detail').innerHTML=header('LOAD ERROR','Recording or graph could not load.',e.message)+`<p>Run the exporter and recorder commands in README, then reload.</p>`;});
