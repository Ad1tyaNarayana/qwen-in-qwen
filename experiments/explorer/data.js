export const $ = id => document.getElementById(id);
export const esc = x => String(x ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const hex = n => '0x'+(Number(n)>>>0).toString(16).padStart(8,'0');
export const number = n => Number(n).toLocaleString('en-US');
export async function bytes(url){
  const r=await fetch(url);if(!r.ok)throw Error(`${r.status} loading ${url}`);
  if(url.endsWith('.gz'))return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  return new Uint8Array(await r.arrayBuffer());
}
export async function json(url){return JSON.parse(new TextDecoder().decode(await bytes(url)));}
export const badge = (source,type='compiled') => `<span class="badge ${type}">${esc(source)}</span>`;
export function tokenFamily(record,t){
  let base=0;
  for(const [name,size] of record.token_families){if(t<base+size)return [name,t-base];base+=size;}
  return [record.controls[t-base]??'INVALID',0];
}
export function outputAt(record,offset){return new TextDecoder().decode(new Uint8Array([...(record.prefix_byte_events??[]),...record.byte_events].filter(e=>e[0]<=offset).map(e=>e[1])));}
export function instructionText(i){
  if(!i)return 'No instruction at this address';
  const [, , op,rd,rs1,rs2,imm]=i;
  const R=new Set(['ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND','MUL','MULH','MULHSU','MULHU','DIV','DIVU','REM','REMU','SH1ADD','SH2ADD','SH3ADD','NDOT4','NDOT8']);
  if(R.has(op))return `${op.toLowerCase()} x${rd}, x${rs1}, x${rs2}`;
  if(['LB','LH','LW','LBU','LHU'].includes(op))return `${op.toLowerCase()} x${rd}, ${imm}(x${rs1})`;
  if(['SB','SH','SW'].includes(op))return `${op.toLowerCase()} x${rs2}, ${imm}(x${rs1})`;
  if(['BEQ','BNE','BLT','BGE','BLTU','BGEU'].includes(op))return `${op.toLowerCase()} x${rs1}, x${rs2}, ${hex(i[0]+imm)}`;
  if(op==='JAL')return `jal x${rd}, ${hex(i[0]+imm)}`;
  if(op==='JALR')return `jalr x${rd}, ${imm}(x${rs1})`;
  if(['LUI','AUIPC'].includes(op))return `${op.toLowerCase()} x${rd}, ${hex(imm)}`;
  if(op.endsWith('I'))return `${op.toLowerCase()} x${rd}, x${rs1}, ${imm}`;
  return op.toLowerCase();
}
export function heatmap(canvas,values,rows,cols,{ballast=false}={}){
  const ctx=canvas.getContext('2d');canvas.width=cols;canvas.height=rows;
  const image=ctx.createImageData(cols,rows);
  let max=1;
  for(let r=0;r<rows;r++)for(let c=ballast?2:0;c<cols;c++)max=Math.max(max,Math.abs(values[r*cols+c]??0));
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
    const v=values[r*cols+c]??0;const mag=Math.pow(Math.min(1,Math.abs(v)/max),.35);const k=(r*cols+c)*4;
    const color=ballast&&c<2?[107,90,62]:v<0?[207,122,150]:[81,191,181];
    image.data[k]=12+mag*color[0];image.data[k+1]=18+mag*color[1];image.data[k+2]=27+mag*color[2];image.data[k+3]=255;
  }
  ctx.putImageData(image,0,0);canvas.dataset.max=max;
}
