#!/usr/bin/env python3
"""Compare rvi.py to the project's reference on any ELF (only here is the project reference imported, for comparison)."""
import sys,json,hashlib,struct,functools,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'experiments/audit/scripts'))
import rvi
def project_tokens(elfp,out):
    sys.path.insert(0,str(R/'experiments/neural-accelerator'))
    from prepare import prepare
    p=Path(out)/f'pkg-{time.time_ns()}'
    prepare(p)
    from chipformer.rv import elf,ref,spec
    import importlib.util
    ms=importlib.util.spec_from_file_location('ndot_reference',R/'experiments/neural-accelerator/reference.py')
    nd=importlib.util.module_from_spec(ms);ms.loader.exec_module(nd)
    ref.decode=functools.lru_cache(None)(ref.decode);nd.decode=functools.lru_cache(None)(nd.decode)
    m=nd.Machine.from_program(elf.program(Path(elfp)));m.maximum=4
    t=[]
    while not m.halted: t.extend(m.record())
    return t,m,spec
if __name__=='__main__':
    e=sys.argv[1];o=sys.argv[2];Path(o).mkdir(parents=True,exist_ok=True)
    c=rvi.CPU(e).run()
    pt,m,spec=project_tokens(e,o)
    q0,qn=rvi.BASE['QS'],rvi.BASE['QD']+128
    # identical numbering check
    assert spec.CTRL==rvi.CTRL and spec.tok('HALT')==rvi.BASE['HALT']
    strip=[t for t in pt if not q0<=t<qn]
    n=min(len(strip),len(c.tokens))
    first=next((i for i in range(n) if strip[i]!=c.tokens[i]),None)
    ram=struct.pack(f'<{len(m.ram)}I',*m.ram)
    r=dict(elf=e,rvi_tokens=len(c.tokens),project_tokens=len(pt),project_tokens_no_qs_qd=len(strip),first_diff=first,
      tokens_equal=strip==list(c.tokens),regs_equal=list(m.x)==c.x,ram_equal=ram==bytes(c.ram),output_equal=bytes(m.out)==bytes(c.out),
      instructions=c.steps,proj_instructions=m.steps,ndots=c.ndots,output=c.out.decode('latin1'),fault=c.fault,proj_fault=str(m.fault),
      project_sha256=hashlib.sha256(struct.pack(f'<{len(pt)}I',*pt)).hexdigest())
    print(json.dumps(r,indent=1))
    Path(o,'project.tokens.u32').write_bytes(struct.pack(f'<{len(pt)}I',*pt))
    Path(o,'result.json').write_text(json.dumps(r,indent=1))
