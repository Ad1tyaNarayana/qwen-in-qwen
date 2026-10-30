import {heatmap} from './data.js';
export function drawGuest(canvas,ram,decoder){
  const {kind,address}=decoder;
  if(kind==='doom'&&!['rgb565','rgb','rgba','indexed'].includes(decoder.format))throw Error('Unknown Doom pixel format');const base=address-0x1000000;
  if(!Number.isInteger(base)||base<0)throw Error('Decoder requires an explicit guest RAM address');
  const width=decoder.width??(kind==='tic-tac-toe'?3:0),height=decoder.height??width;
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width*height>1e6)throw Error('Invalid decoder geometry');
  const channels=decoder.format==='rgba'?4:decoder.format==='rgb'?3:decoder.format==='rgb565'?2:1,stride=decoder.stride??width*channels;
  if(base+(height-1)*stride+width*channels>ram.length)throw Error('Decoder span is outside captured RAM');
  const values=[];for(let y=0;y<height;y++)for(let x=0;x<width;x++)values.push(ram[base+y*stride+x]);
  if(['life','mnist','tic-tac-toe'].includes(kind)){heatmap(canvas,values,height,width);return;}
  if(kind==='doom'){
    const ctx=canvas.getContext('2d');canvas.width=width;canvas.height=height;const image=ctx.createImageData(width,height);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const i=(y*width+x)*4,j=base+y*stride+x*channels;
      const packed=ram[j]|ram[j+1]<<8;
      const color=channels===1?decoder.palette?.[ram[j]]:channels===2?[Math.round((packed>>11)*255/31),Math.round((packed>>5&63)*255/63),Math.round((packed&31)*255/31)]:Array.from(ram.slice(j,j+channels));
      if(!color)throw Error('Indexed Doom pixels require the manifest palette. No guessed colors.');
      image.data.set([color[0],color[1],color[2],255],i);
    }ctx.putImageData(image,0,0);return;
  }
  throw Error(`Unsupported RAM decoder ${kind}`);
}
