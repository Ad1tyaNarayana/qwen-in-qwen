"""Require single-bit corruptions of actual exported evidence to fail acceptance."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from verify_gpu import HERE,ROOT,verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('actual',type=Path);a=p.parse_args()
    sys.path.insert(0,str(ROOT/'experiments/neural-accelerator'))
    from prepare import prepare
    result={}
    with tempfile.TemporaryDirectory(prefix='verifier-control-',dir=ROOT/'.cache/showcase') as directory:
        directory=Path(directory);prepare(directory/'package')
        from chipformer.rv import spec
        for name,file in [('fused_tail','fused/actual.tokens.u32'),('stock_tail','stock/actual.tokens.u32'),
                          ('ram','fused/actual-ram.bin'),('register_copy','fused/actual-registers.json')]:
            out=directory/name;out.mkdir()
            shutil.copytree(a.actual/'sha256',out/'sha256')
            for path in a.actual.glob('sha256-*.exit.json'):shutil.copyfile(path,out/path.name)
            path=out/'sha256'/file
            if name=='register_copy':
                registers=json.loads(path.read_text());registers['regs.rs2'][1]^=1
                path.write_text(json.dumps(registers))
            else:
                raw=bytearray(path.read_bytes());raw[-1 if 'tail' in name else 0]^=1;path.write_bytes(raw)
            try:verify(out,'sha256',spec)
            except AssertionError:result[name]=dict(one_bit_change=True,rejected=True)
            else:raise AssertionError(f'Corruption was accepted: {name}')
    (HERE/'evidence/verifier-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
