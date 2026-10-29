"""Replay both retained guests and account for the final temporary values."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/neural-accelerator'))
from prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prepare(args.out / 'scratch')
    from chipformer.rv import elf, ref, spec
    import reference
    ref.decode = functools.lru_cache(None)(ref.decode)
    reference.decode = functools.lru_cache(None)(reference.decode)
    retained_dir = ROOT / '.cache/tiny-llm/ndot-verify64'
    retained = json.loads((retained_dir / 'report.json').read_text())
    result = dict(status='running', rule='final-use liveness and accounted value flows', runs=[])
    def save():
        (args.out / 'gate2-liveness.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    for index, (name, folder, machine_class) in enumerate([
        ('original', 'ndot-original64', ref.Machine), ('ndot', 'ndot64', reference.Machine),
    ]):
        directory = ROOT / '.cache/tiny-llm' / folder
        meta = json.loads((directory / 'build.json').read_text())
        machine = machine_class.from_program(elf.program(directory / 'engine.elf'))
        machine.maximum = 4
        sources = {}
        taints = [set() for _ in range(32)]
        instructions = {}
        active = False
        positions = 0
        while not machine.halted:
            pc = machine.pc
            refresh = machine.n == spec.N_MAX
            if not refresh and pc == meta['elf']['symbols']['position_begin']:
                positions += 1
                active = positions == 64
                if active:
                    print(name, 'tracking final position', machine.steps, flush=True)
            if active and not refresh:
                word = machine.rom[(pc-spec.ROM_BASE)//4]
                ins = reference.decode(word, 4)
                instructions[pc] = dict(pc=hex(pc), word=hex(word), instruction=vars(ins))
                reads = {ins.rs1, ins.rs2} - {0}
                if ins.op in ('LUI', 'AUIPC', 'JAL', 'FENCE'):
                    reads = set()
                if ins.op == 'ECALL':
                    reads = {10, 17}
                if ins.op.startswith('NDOT'):
                    reads.add(ins.rd)
                incoming = set().union(*(taints[r] for r in reads)) if reads else set()
                event = dict(step=machine.steps+1, pc=hex(pc), op=ins.op,
                             registers={str(r): machine.x[r] for r in sorted(reads)})
                if ins.op in ref.STORE_OPS:
                    event.update(address=hex((machine.x[ins.rs1]+ins.imm)&0xffffffff),
                                 bytes={'SB':1, 'SH':2, 'SW':4}[ins.op], value=machine.x[ins.rs2])
                    for root in incoming:
                        events = sources[root]['stores']
                        key = (event['pc'], event['address'], event['bytes'], event['value'])
                        events[key] = event
                if ins.op == 'ECALL':
                    for root in incoming:
                        sources[root]['ecalls'].append(event)
                for root in incoming:
                    sources[root]['last_use'] = event
                writes = ins.op in reference.WRITES_RD and ins.rd != 0
                machine.record()
                if writes:
                    if ins.rd in (28, 29, 30, 31):
                        for tags in taints:
                            tags.discard(ins.rd)
                        incoming.discard(ins.rd)
                        sources[ins.rd] = dict(register=ins.rd, abi_name='t'+str(ins.rd-25),
                            last_write=dict(step=machine.steps, pc=hex(pc), value=machine.x[ins.rd]),
                            last_use=None, stores={}, ecalls=[])
                        incoming.add(ins.rd)
                    taints[ins.rd] = incoming
            else:
                machine.record()
        expected = retained['runs'][index]
        ram = struct.pack('<16384I', *machine.ram)
        assert machine.x == expected['registers']
        assert machine.steps == expected['instructions']
        assert ram == (retained_dir / f'{name}-ram.bin').read_bytes()
        assert bytes(machine.out) == (retained_dir / f'{name}-output.bin').read_bytes()
        assert machine.fault is None and machine.exit_code == 0
        for source in sources.values():
            source['stores'] = list(source['stores'].values())
        run = dict(guest=name, elf_sha256=hashlib.sha256((directory/'engine.elf').read_bytes()).hexdigest(),
                   final_sp=hex(machine.x[2]), pc=hex(machine.pc), registers=machine.x,
                   retained_final_state_exact=True, temporaries=list(sources.values()),
                   exit_abi=dict(registers=[10,17], values=[machine.x[10],machine.x[17]],
                                 convention='ECALL a7=93 exits with a0; a7=1 outputs a0 byte'),
                   disassembly=list(instructions.values()))
        result['runs'].append(run)
        save()
    old, new = result['runs']
    differences = []
    old_ram = (retained_dir/'original-ram.bin').read_bytes()
    new_ram = (retained_dir/'ndot-ram.bin').read_bytes()
    for offset in range(0,65536,4):
        if old_ram[offset:offset+4] != new_ram[offset:offset+4]:
            address = spec.RAM_BASE+offset
            differences.append(dict(address=hex(address), below_both_final_sp=all(address+4<=int(r['final_sp'],16) for r in result['runs'])))
    assert all(d['below_both_final_sp'] for d in differences)
    stack_bottom = spec.SP_INIT-8192
    assert old_ram[:stack_bottom-spec.RAM_BASE] == new_ram[:stack_bottom-spec.RAM_BASE]
    for run in result['runs']:
        for source in run['temporaries']:
            assert not source['ecalls'], source
            for store in source['stores']:
                address = int(store['address'],16)
                offset = address-spec.RAM_BASE
                assert 0 <= offset <= 65536-store['bytes'], store
                if address >= stack_bottom:
                    assert address+store['bytes'] <= int(run['final_sp'],16), store
                    store['covered_by'] = 'popped stack below final SP'
                else:
                    assert old_ram[offset:offset+store['bytes']] == new_ram[offset:offset+store['bytes']], store
                    store['covered_by'] = 'byte-exact non-stack RAM'
    result.update(status='pass', stack_differences=differences, non_stack_ram_exact=True,
                  differing_registers=[i for i,(x,y) in enumerate(zip(old['registers'],new['registers'])) if x!=y],
                  final_registers_outside_exit_abi=True, output_bytes_exact=True,
                  flow_method='Register data/address dependencies from each final x28-x31 write through HALT; old root tags removed on a later write; stores retain effective address and value; ECALL dependencies checked. Control flow is the executed binary path, not a claim of universal noninterference.')
    assert result['differing_registers'] == [28,29,30,31]
    save()
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2),flush=True)


if __name__ == '__main__':
    main()
