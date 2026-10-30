#!/usr/bin/env python3
"""Draw the LLM article figures from published evidence; no experiment runs."""
import json
from html import escape
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
FONT=next(p for p in [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists())


class Figure:
    def __init__(self,title,subtitle,height):
        self.height=height
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">','<rect width="100%" height="100%" fill="#fafafa"/>']
        self.im=Image.new('RGB',(1200,height),'#fafafa');self.draw=ImageDraw.Draw(self.im)
        self.text(45,45,title,28);self.text(45,78,subtitle,18,'#555555')

    def text(self,x,y,value,size=18,color='#222222',anchor='start'):
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')
        self.draw.text((x,y),str(value),font=ImageFont.truetype(str(FONT),size),fill=color,anchor={'start':'ls','middle':'ms','end':'rs'}[anchor])

    def rect(self,x,y,w,h,color):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>');self.draw.rectangle((x,y,x+w,y+h),fill=color)

    def line(self,x1,y1,x2,y2,color='#777777',width=2):
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>');self.draw.line((x1,y1,x2,y2),fill=color,width=width)

    def arrow(self,x1,y1,x2,y2,color='#555555'):
        self.line(x1,y1,x2,y2,color,3)
        if x1==x2:
            d=1 if y2>y1 else -1;self.line(x2-7,y2-d*12,x2,y2,color,3);self.line(x2+7,y2-d*12,x2,y2,color,3)
        else:
            d=1 if x2>x1 else -1;self.line(x2-d*12,y2-7,x2,y2,color,3);self.line(x2-d*12,y2+7,x2,y2,color,3)

    def save(self,name):
        (OUT/f'{name}.svg').write_text('\n'.join(self.svg+['</svg>'])+'\n');self.im.save(OUT/f'{name}.png');print(name)


def read(path):return json.loads((ROOT/path).read_text())


def loops():
    f=Figure('The guest model and the CPU have different recurrence','Conceptual data flow; full workloads use the compact fused executor',650)
    f.rect(45,115,1110,385,'#e7f0f5')
    f.text(70,150,'Compiled Qwen3.5-architecture network',24,'#00577f')
    f.text(70,180,'Outer weights are compiled CPU logic. Recurrent tables hold guest machine state.',18)
    for x,title,detail in [(80,'Fetch and read','Code, registers, ROM, RAM'),(450,'Compute','Integer CPU and NDOT'),(820,'Encode result','CPU protocol token')]:
        f.rect(x,220,300,90,'#ffffff');f.text(x+150,253,title,21,anchor='middle');f.text(x+150,283,detail,17,anchor='middle')
    f.arrow(380,263,445,263);f.arrow(750,263,815,263)
    f.line(970,312,970,355);f.line(970,355,230,355);f.arrow(230,355,230,312)
    f.text(600,385,'Generated token selects the next embedding',19,anchor='middle')
    f.rect(80,410,1040,65,'#d0e7d7');f.text(600,438,'Guest inference program: activations, attention state, logits, next text ID',19,anchor='middle')
    f.text(600,463,'Learned guest weights and tables are data ROM; working state is recurrent RAM.',17,anchor='middle')
    f.rect(45,530,510,70,'#eee9df');f.text(65,559,'Host provides ELF boot state and prompt IDs.',18);f.text(65,585,'No intermediate guest results enter after boot.',18)
    f.rect(590,530,565,70,'#eee9df');f.text(610,559,'Reference traces compare generated values only.',18);f.text(610,585,'Bounded stock prefixes do not establish a full run.',18)
    f.save('two-loops')


def narrowing():
    variants={v['variant']:v for v in read('experiments/neural-accelerator/resources.json')['variants']}
    f=Figure('Narrowing made dense stock storage plausible','Exact parameter counts × BF16 bytes; this is not an allocation or timing chart',530)
    for tick in range(0,36,5):
        x=270+tick*22;f.line(x,120,x,355,'#dddddd',1);f.text(x,384,tick,16,anchor='middle')
    for i,(name,label) in enumerate([('w4-before','First W=4'),('w4-after','Reduced W=4')]):
        v=variants[name];value=v['bf16_parameter_bytes']/2**30;y=160+i*125
        f.text(250,y+27,label,22,anchor='end');f.rect(270,y,value*22,40,['#c77d34','#0072b2'][i])
        f.text(270+value*22+12,y+28,f'{value:.2f} GiB',20)
        f.text(270,y+70,f"Residual dimensions {v['geometry']['dims']:,}; stock hidden width {v['geometry']['hidden']:,}",18)
    f.text(650,424,'Dense BF16 parameter storage (GiB)',20,anchor='middle')
    f.text(45,475,'Recurrent state, allocator overhead, and desktop VRAM use require additional space.',18)
    f.text(45,505,'Source: experiments/neural-accelerator/resources.json',16,'#555555');f.save('narrowing')


def measurement():
    path=ROOT/'experiments/checkpoint-integrity/evidence/controlled-measurement.json'
    if not path.exists():return
    data=json.loads(path.read_text());assert data['status']=='pass'
    f=Figure('One complete stories260K run per binary','Same unified W=4 network, model data, prompt, five positions, and compact fused executor',600)
    runs=data['runs'];max_minutes=max(r['generation_seconds']/60 for r in runs.values());axis_max=(int(max_minutes/10)+1)*10
    for tick in range(0,axis_max+1,10):
        x=240+tick*800/axis_max;f.line(x,120,x,355,'#dddddd',1);f.text(x,385,tick,16,anchor='middle')
    for i,name in enumerate(['scalar','ndot']):
        r=runs[name];y=165+i*125;minutes=r['generation_seconds']/60
        f.text(220,y+28,{'scalar':'Scalar','ndot':'NDOT4'}[name],22,anchor='end');f.rect(240,y,minutes*800/axis_max,40,['#c77d34','#0072b2'][i])
        f.text(240,y+72,f"{r['generation_seconds']:,.2f} s; {r['tokens']:,} trace tokens; {r['minutes_per_position']:.4f} min/position",18)
    f.text(650,425,'Retained generation minutes',20,anchor='middle')
    f.text(45,480,f"Scalar / NDOT4 generation time = {data['scalar_over_ndot_generation_ratio']:.4f}×",23)
    f.text(45,520,'Every token, final RAM, and all register copies match each binary\'s own reference.',18)
    f.text(45,550,'Stock generation covered bounded prefixes only; these full runs use the compact fused executor.',17)
    f.text(45,580,'No repeated-run variability or uninterrupted end-to-end wall-time claim.',17);f.save('controlled-measurement')


if __name__=='__main__':loops();narrowing();measurement()
