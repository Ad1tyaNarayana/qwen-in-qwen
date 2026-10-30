#!/usr/bin/env python3
"""Independent RV32IM+ndot interpreter, written from SPEC.md and upstream spec.py's docstring only.
Does NOT import or derive from the project's reference.py / ref.py / oracle.py. Emits the token stream
(tiny token numbering re-derived here). QS/QD division digit tokens are NOT regenerated: DIV-class
records are emitted as the value record only, so comparisons strip QS/QD on the other side."""
import array, hashlib, json, struct, sys
SIZES=[("V",1<<15),("F0",1<<15),("F1",1<<15),("F2",4),("J",1<<16),("R",1<<16),("BYTE",512),("FLT",8),("HEX",16),("QS",16),("QD",128),("BD",256)]
BASE={};b=0
for n,s in SIZES: BASE[n]=b;b+=s
CTRL=b
CN=["NT","ST0","ST1","SS0","SS1","HALT","GO","SC"]
for i,n in enumerate(CN): BASE[n]=CTRL+i
RAMB,RAMN=0x01000000,65536
DROMB,DROMN=0x02000000,4194304
ROMB,ROMN=0x40000,262144
M=0xffffffff
def sx(v,b): v&=(1<<b)-1; return v-(1<<b) if v>>(b-1) else v
def s32(v): return sx(v,32)
class Fault(Exception):
    def __init__(s,c): s.c=c
class CPU:
    def __init__(s,elf):
        s.ram=bytearray(RAMN); s.drom=bytearray(DROMN); s.rom=bytearray(ROMN)
        raw=open(elf,'rb').read() if isinstance(elf,str) else bytes(elf)
        entry=struct.unpack_from('<I',raw,24)[0]
        phoff=struct.unpack_from('<I',raw,28)[0]; ps,pn=struct.unpack_from('<HH',raw,42)
        for i in range(pn):
            k,off,addr,_,size,msz,_,_=struct.unpack_from('<8I',raw,phoff+i*ps)
            if k!=1 or not size: continue
            d=raw[off:off+size]
            for base,mem,n in ((RAMB,s.ram,RAMN),(DROMB,s.drom,DROMN),(ROMB,s.rom,ROMN)):
                if base<=addr<base+n: mem[addr-base:addr-base+size]=d
        # Boot ABI from spec.py: SP_INIT is the first byte above writable RAM.
        s.x=[0]*32; s.x[2]=RAMB+RAMN; s.pc=entry; s.anchor=(entry-ROMB)//4; s.n=0
        s.halted=False; s.out=bytearray(); s.tokens=array.array('I'); s.steps=0; s.ndots=0; s.fault=None
        s.ins=0
    def val(s,v):
        v&=M; t=s.tokens; sg=s32(v)
        if -(1<<14)<=sg<(1<<14): t.append(BASE["F0"]+(v&0x7fff))
        elif -(1<<29)<=sg<(1<<29): t.extend((BASE["V"]+(v&0x7fff),BASE["F1"]+(v>>15&0x7fff)))
        else: t.extend((BASE["V"]+(v&0x7fff),BASE["V"]+(v>>15&0x7fff),BASE["F2"]+(v>>30)))
    def wr(s,rd,v):
        if rd: s.x[rd]=v&M; s.val(v)
        else: s.tokens.append(BASE["NT"])
    def load(s,a,n,sign):
        if a%n: raise Fault(2)
        if RAMB<=a and a+n<=RAMB+RAMN: m,o=s.ram,a-RAMB
        elif DROMB<=a and a+n<=DROMB+DROMN: m,o=s.drom,a-DROMB
        else: raise Fault(4)
        v=int.from_bytes(m[o:o+n],'little')
        return sx(v,8*n)&M if sign else v
    def store(s,a,n,v):
        if a%n: raise Fault(3)
        if not(RAMB<=a and a+n<=RAMB+RAMN): raise Fault(5)
        o=a-RAMB; s.ram[o:o+n]=(v&((1<<8*n)-1)).to_bytes(n,'little')
        wo=o&~3; word=int.from_bytes(s.ram[wo:wo+4],'little'); bit=(a>>6)&1
        if n==4: s.tokens.append(BASE["ST1" if bit else "ST0"])
        else:
            s.tokens.extend((BASE["V"]+(word&0x7fff),BASE["V"]+(word>>15&0x7fff),BASE["V"]+(word>>30)))
            s.tokens.append(BASE["SS1" if bit else "SS0"])
    def jump(s,t):
        if t%4 or not ROMB<=t<ROMB+ROMN: raise Fault(6)
        s.pc=t; s.anchor=(t-ROMB)//4; s.n=-1; s.tokens.append(BASE["J"]+s.anchor)
    def step(s):
        if s.n==63:
            s.anchor+=63; s.n=0; s.tokens.append(BASE["R"]+s.anchor)
        pc=s.pc
        try:
            s.exec1(pc)
        except Fault as f:
            s.fault=(f.c,pc); s.tokens.append(BASE["FLT"]+f.c)
            s.tokens.extend(BASE["HEX"]+(pc>>4*(7-i)&15) for i in range(8))
            s.tokens.append(BASE["HALT"]); s.halted=True; return
        s.n+=1; s.steps+=1
    def exec1(s,pc):
        if not ROMB<=pc<ROMB+ROMN: raise Fault(6)
        w=int.from_bytes(s.rom[pc-ROMB:pc-ROMB+4],'little'); x=s.x
        op=w&0x7f; rd=w>>7&31; f3=w>>12&7; rs1=w>>15&31; rs2=w>>20&31; f7=w>>25
        a,b=x[rs1],x[rs2]; nxt=pc+4
        immi=sx(w>>20,12); imms=sx((w>>25)<<5|rd,12)
        immb=sx((w>>31)<<12|(w>>7&1)<<11|(w>>25&63)<<5|(w>>8&15)<<1,13)
        immj=sx((w>>31)<<20|(w>>12&255)<<12|(w>>20&1)<<11|(w>>21&1023)<<1,21)
        if op==0x37: s.wr(rd,w&0xfffff000)
        elif op==0x17: s.wr(rd,pc+(w&0xfffff000))
        elif op==0x6f:
            s.jump((pc+immj)&M)
            if rd: x[rd]=nxt
            return
        elif op==0x67:
            s.jump((a+immi)&~1&M)
            if rd: x[rd]=nxt
            return
        elif op==0x63:
            sa,sb=s32(a),s32(b)
            t={0:a==b,1:a!=b,4:sa<sb,5:sa>=sb,6:a<b,7:a>=b}.get(f3)
            if t is None: raise Fault(0)
            if t: return s.jump((pc+immb)&M)
            s.tokens.append(BASE["NT"])
        elif op==0x03:
            ea=(a+immi)&M
            if f3 not in(0,1,2,4,5): raise Fault(0)
            n={0:1,1:2,2:4,4:1,5:2}[f3]; v=s.load(ea,n,f3<2)
            s.wr(rd,v)
        elif op==0x23:
            if f3>2: raise Fault(0)
            s.store((a+imms)&M,1<<f3,b)
        elif op==0x13 or op==0x33:
            imm=op==0x13
            bb=immi&M if imm else b; sh=bb&31
            if op==0x33 and f7==1:
                sa,sb=s32(a),s32(b)
                if f3==0: r=a*b
                elif f3==1: r=(sa*sb)>>32
                elif f3==2: r=(sa*b)>>32
                elif f3==3: r=(a*b)>>32
                elif f3==4:
                    if b==0: r=M
                    elif sa==-(1<<31) and sb==-1: r=a
                    else:
                        q=abs(sa)//abs(sb); r=-q if (sa<0)!=(sb<0) else q
                elif f3==5: r=a//b if b else M
                elif f3==6:
                    if b==0: r=a
                    elif sa==-(1<<31) and sb==-1: r=0
                    else:
                        m=abs(sa)%abs(sb); r=-m if sa<0 else m
                else: r=a%b if b else a
                s.wr(rd,r&M); s.pc=nxt; return
            if op==0x33 and f7==0x10 and f3 in (2,4,6):  # Zba, inherited by the base CPU
                s.wr(rd,(b+(a<<(f3>>1)))&M); s.pc=nxt; return
            sub=(not imm) and f7==0x20
            alt=f7==0x20
            if f3==0: r=a-bb if sub else a+bb
            elif f3==1: r=a<<sh
            elif f3==2: r=int(s32(a)<s32(bb))
            elif f3==3: r=int(a<(bb&M))
            elif f3==4: r=a^bb
            elif f3==5: r=(s32(a)>>sh) if alt else a>>sh
            elif f3==6: r=a|bb
            else: r=a&bb
            s.wr(rd,r&M)
        elif op==0x0f: s.tokens.append(BASE["NT"])
        elif op==0x73:
            if w==0x73:
                if x[17]==1:
                    c=x[10]&511; s.tokens.append(BASE["BYTE"]+c)
                    if c<256: s.out.append(c)
                elif x[17]==93:
                    s.tokens.append(BASE["HALT"]); s.halted=True; s.exit=x[10]
                else: raise Fault(7)
            else: raise Fault(0)
        elif op==0x0b:
            if f7!=0x20 or f3>1: raise Fault(0)
            N=4 if f3==0 else 8
            if N==8: raise Fault(0)  # W=4 constructor rejects N=8
            if a%N or b%(2*N): raise Fault(2)
            if not(DROMB<=a and a+N<=DROMB+DROMN and RAMB<=b and b+2*N<=RAMB+RAMN): raise Fault(4)
            r=x[rd]
            for i in range(N):
                r+=sx(s.drom[a-DROMB+i],8)*sx(int.from_bytes(s.ram[b-RAMB+2*i:b-RAMB+2*i+2],'little'),16)
            s.ndots+=1
            s.wr(rd,r&M)
        else: raise Fault(0)
        s.pc=nxt
    def run(s,limit=50_000_000):
        while not s.halted and s.steps<limit: s.step()
        return s
    def state(s):
        return dict(tokens=len(s.tokens),instructions=s.steps,ndots=s.ndots,regs=s.x,exit=getattr(s,'exit',None),fault=s.fault,
          output=s.out.decode('latin1'),ram_sha256=hashlib.sha256(s.ram).hexdigest(),
          token_sha256=hashlib.sha256(s.tokens.tobytes()).hexdigest())
if __name__=='__main__':
    c=CPU(sys.argv[1]).run(10**9)
    st=c.state(); print(json.dumps(st,indent=1))
    if len(sys.argv)>2:
        open(sys.argv[2],'wb').write(c.tokens.tobytes()); open(sys.argv[2]+'.ram','wb').write(c.ram)
