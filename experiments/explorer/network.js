export class NetworkMap {
  constructor(canvas,graph,details,onHover,onPick){
    Object.assign(this,{canvas,graph,details,onHover,onPick,yaw:-.34,pitch:.68,zoom:1,mode:'units',role:'',layer:'',query:'',points:[],projected:[]});
    this.makePoints();this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas);
    let drag=null,moved=false;
    canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,this.yaw,this.pitch];moved=false;canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{
      if(drag){const dx=e.clientX-drag[0],dy=e.clientY-drag[1];moved ||= Math.abs(dx)+Math.abs(dy)>4;this.yaw=drag[2]+dx*.005;this.pitch=Math.max(.15,Math.min(1.4,drag[3]+dy*.003));this.draw();return;}
      const box=canvas.getBoundingClientRect();const x=e.clientX-box.left,y=e.clientY-box.top;
      this.hover=this.pick(x,y);onHover(this.hover,x,y);this.draw();
    });
    canvas.addEventListener('pointerup',e=>{if(drag&&!moved){const b=canvas.getBoundingClientRect();onPick(this.pick(e.clientX-b.left,e.clientY-b.top));}drag=null;});
    canvas.addEventListener('pointerleave',()=>{if(!drag){this.hover=null;onHover(null);this.draw();}});
    canvas.addEventListener('wheel',e=>{e.preventDefault();const next=Math.max(.5,Math.min(4,this.zoom*Math.exp(-e.deltaY*.001)));this.animateZoom(next);},{passive:false});
    canvas.addEventListener('dblclick',()=>{if(this.hover)onPick(this.hover);});
  }
  animateZoom(target){if(matchMedia('(prefers-reduced-motion: reduce)').matches){this.zoom=target;this.draw();return;}this.target=target;if(this.animating)return;this.animating=true;const step=()=>{this.zoom+=(this.target-this.zoom)*.23;this.draw();if(Math.abs(this.target-this.zoom)>.003)requestAnimationFrame(step);else{this.zoom=this.target;this.animating=false;}};requestAnimationFrame(step);}
  reset(){this.yaw=-.34;this.pitch=.68;this.animateZoom(1);}
  makePoints(){
    this.points=[];
    for(const layer of this.graph.layers){
      const detail=this.details[layer.index];const cols=100;
      if(this.mode==='residual'){
        for(const coordinate of this.graph.hidden_dimensions){
          const d=coordinate.physical_dim,h=coordinate.index;
          const lives=d?this.graph.dimensions[d]:[],life=lives.find(o=>o.born<=layer.index*2+2&&o.last>=layer.index*2+1);
          this.points.push({x:(h%56/55-.5)*245,y:(Math.floor(h/56)/42-.5)*160,z:(layer.index-32)*8,
            layer:layer.index,kind:d?'residual':'layout',index:d??h,name:d?(life?`${life.name} · sublayers ${life.born}..${life.last}`:`slot ${d} · no live owner`):`${coordinate.kind} · hidden coordinate ${h}`,role:d?(life?.role??'reuse'):coordinate.kind==='ballast'?'normalization':'unused'});
        }
      }else{
        const units=detail?.units??[];const count=layer.zero||this.mode==='full'?9216:units.length;
        for(let u=0;u<count;u++){
          const unit=units[u];this.points.push({x:(u%cols/(cols-1)-.5)*245,y:(Math.floor(u/cols)/92-.5)*160,z:(layer.index-32)*8,
            layer:layer.index,kind:layer.zero||!unit?'zero':'unit',index:u,name:unit?.[0]??(layer.zero?'Appended zero MLP unit':'Unused zero MLP unit'),role:unit?.[1]??(layer.zero?'output':'unused'),stats:unit?.[6]});
        }
        for(const h of detail?.heads??[])for(let v=0;v<h.values.length;v++)this.points.push({x:(h.key_head/15-.5)*245,y:100+v*12,z:(layer.index-32)*8,
          layer:layer.index,kind:'head',index:h.key_head,value:v,name:`${h.name} · value ${v}`,role:h.role,stats:h.values[v].stats});
        if(!layer.zero&&this.mode==='full')for(let j=0;j<16;j++)for(let v=0;v<2;v++){if(detail?.heads[j]?.values[v])continue;this.points.push({x:(j/15-.5)*245,y:100+v*12,z:(layer.index-32)*8,layer:layer.index,kind:'zero',name:`Unused zero value head ${j*2+v}`,role:'unused'});}
        if(layer.zero)for(let h=0;h<8;h++)this.points.push({x:(h/7-.5)*245,y:105,z:(layer.index-32)*8,layer:layer.index,kind:'zero',name:`Zero full-attention query head ${h}`,role:'output'});
      }
    }
    this.draw();
  }
  project(x,y,z){
    const a=x*Math.cos(this.yaw)+z*Math.sin(this.yaw),b=-x*Math.sin(this.yaw)+z*Math.cos(this.yaw);
    const yy=y*Math.cos(this.pitch)-b*Math.sin(this.pitch),depth=y*Math.sin(this.pitch)+b*Math.cos(this.pitch);
    const f=900/(900+depth);const scale=this.scale*this.zoom*f;
    return [this.cx+a*scale,this.cy+yy*scale,f];
  }
  draw(){
    const {canvas}=this;if(!canvas.clientWidth||!canvas.clientHeight)return;
    this.w=canvas.clientWidth;this.h=canvas.clientHeight;const dpr=Math.min(2,devicePixelRatio||1);
    if(canvas.width!==Math.round(this.w*dpr)||canvas.height!==Math.round(this.h*dpr)){canvas.width=Math.round(this.w*dpr);canvas.height=Math.round(this.h*dpr);}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,this.w,this.h);
    this.scale=1;this.cx=0;this.cy=0;
    const bounds=[];for(const z of [-256,256])for(const x of [-147,128])for(const y of [-88,125])bounds.push(this.project(x,y,z));
    const xs=bounds.map(p=>p[0]/this.zoom),ys=bounds.map(p=>p[1]/this.zoom);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    this.scale=Math.min((this.w-95)/(maxX-minX),(this.h-120)/(maxY-minY));
    this.cx=this.w/2-(minX+maxX)/2*this.scale*this.zoom;this.cy=100+(this.h-120)/2-(minY+maxY)/2*this.scale*this.zoom;this.projected=[];
    for(const L of this.graph.layers){
      const z=(L.index-32)*8;const corners=[[-128,-86],[128,-86],[128,119],[-128,119]].map(([x,y])=>this.project(x,y,z));
      ctx.beginPath();corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.strokeStyle=L.zero?'#e9bc6838':'#59708918';ctx.lineWidth=.6;ctx.stroke();
      if(L.index%8===0||String(L.index)===this.layer||L.zero){const [x,y]=this.project(-147,-88,z);ctx.fillStyle='#6f8196';ctx.font='8px monospace';ctx.fillText(String(L.index).padStart(2,'0')+(L.zero?' F':' L'),x,y);}
    }
    const groups=new Map();const query=this.query.toLowerCase();
    for(const p of this.points){
      if(this.role&&p.role!==this.role||this.layer!==''&&p.layer!==+this.layer||query&&!p.name.toLowerCase().includes(query))continue;
      const [x,y,f]=this.project(p.x,p.y,p.z);if(x<0||x>this.w||y<0||y>this.h)continue;
      this.projected.push([x,y,p]);const key=p.role+(p.kind==='zero'?'0':'');if(!groups.has(key))groups.set(key,[]);groups.get(key).push([x,y,p.kind==='head'?3:Math.max(.7,1.15*f*this.zoom)]);
    }
    for(const [key,points] of groups){const zero=key.endsWith('0');const role=zero?key.slice(0,-1):key;ctx.fillStyle=this.graph.roles[role]?.[1]??'#888';ctx.globalAlpha=zero?.12:.67;for(const [x,y,s] of points)ctx.fillRect(x-s/2,y-s/2,s,s);}
    ctx.globalAlpha=1;
    if(this.hover){const [x,y]=this.project(this.hover.x,this.hover.y,this.hover.z);ctx.strokeStyle='#fff1cb';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,7,0,2*Math.PI);ctx.stroke();ctx.beginPath();ctx.moveTo(x-13,y);ctx.lineTo(x+13,y);ctx.moveTo(x,y-13);ctx.lineTo(x,y+13);ctx.stroke();}
  }
  pick(x,y){let best=null,dist=64;for(const [px,py,p] of this.projected){const d=(x-px)**2+(y-py)**2;if(d<dist){dist=d;best=p;}}return best;}
}
