"""Check actual native/reference outputs against independent problem definitions."""
import functools
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/showcase'


def main():
    reports={}
    for demo in ['sha256','life','tictactoe','mnist','mnist-scalar']:
        out=CACHE/demo
        ref=json.loads((out/'reference.json').read_text())
        build=json.loads((out/'build.json').read_text())
        assert hashlib.sha256((out/'engine.elf').read_bytes()).hexdigest()==ref['elf_sha256']==build['elf_sha256']
        assert hashlib.sha256((out/'drom.bin').read_bytes()).hexdigest()==build['drom_sha256']
        native=subprocess.check_output([str(out/'native')])
        assert native==(out/'reference-output.bin').read_bytes()
        assert hashlib.sha256((out/'reference.tokens.u32').read_bytes()).hexdigest()==ref['sha256']
        reports[demo]=dict(native_reference_equal=True,instructions=ref['instructions'],trace_tokens=ref['tokens'])
    expected='SHA256 empty\n'+hashlib.sha256(b'').hexdigest()+'\nSHA256 abc\n'+hashlib.sha256(b'abc').hexdigest()+'\n'
    assert (CACHE/'sha256/reference-output.bin').read_text()==expected
    board={(2,1),(3,2),(1,3),(2,3),(3,3),(14,9),(15,9),(16,9),(19,5),(20,5),(19,6),(20,6)}
    frames=[]
    for gen in range(5):
        frames.append(f'Generation {gen}\n'+''.join(''.join('#' if (x,y) in board else '.' for x in range(24))+'\n' for y in range(16)))
        counts={(x,y):sum(((x+dx)%24,(y+dy)%16) in board for dx in [-1,0,1] for dy in [-1,0,1] if dx or dy) for y in range(16) for x in range(24)}
        board={p for p,n in counts.items() if n==3 or n==2 and p in board}
    assert (CACHE/'life/reference-output.bin').read_text()==''.join(frames)
    lines=[(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    @functools.lru_cache(None)
    def value(board,turn):
        for a,b,c in lines:
            if board[a] and board[a]==board[b]==board[c]:return 1 if board[a]==2 else -1
        scores=[value(board[:i]+(turn,)+board[i+1:],3-turn) for i in range(9) if not board[i]]
        return (max(scores) if turn==2 else min(scores)) if scores else 0
    board=(0,)*9;checked=0
    for line in (CACHE/'tictactoe/reference-output.bin').read_text().splitlines():
        if len(line.split())==2 and line.split()[0] in ('Human','CPU') and line.split()[1].isdigit():
            player=1 if line.startswith('Human ') else 2
            pos=int(line[-1]);assert board[pos]==0
            new=board[:pos]+(player,)+board[pos+1:]
            if player==2:assert value(new,1)==value(board,2);checked+=1
            board=new
    assert checked>0 and value(board,1)>=0
    reports['tictactoe']['optimal_cpu_moves_checked']=checked
    perfect=CACHE/'tictactoe/check-perfect'
    subprocess.run(['clang','-O2','-DNATIVE','-I'+str(HERE),'-I'+str(CACHE/'tictactoe'),str(HERE/'tictactoe/check_perfect.c'),'-o',str(perfect)],check=True)
    perfect_output=subprocess.check_output([str(perfect)],text=True)
    assert perfect_output=='681 games, 0 human wins\n'
    perfect_report=dict(native_terminal_games=681,human_wins=0,all_human_continuations_explored=True,
        source='experiments/showcase/tictactoe/check_perfect.c',source_sha256=hashlib.sha256((HERE/'tictactoe/check_perfect.c').read_bytes()).hexdigest())
    (HERE/'tictactoe/evidence/perfect-policy.json').write_text(json.dumps(perfect_report,indent=2)+'\n')
    model=CACHE/'mnist-model'
    cmd=['clang','-O2','-fwrapv','-std=c11','-DNATIVE','-DALL_TEST',
         '-DTEST_IMAGES='+json.dumps(str(model/'test-images.bin')),
         '-I'+str(HERE),'-I'+str(model),str(HERE/'mnist/main.c'),'-o',str(model/'all-test')]
    subprocess.run(cmd,check=True)
    predictions=subprocess.check_output([str(model/'all-test')])
    assert predictions==(model/'test-predictions.bin').read_bytes()
    labels=(model/'test-labels.bin').read_bytes()
    training=json.loads((model/'training.json').read_text())
    correct=sum(a==b for a,b in zip(predictions,labels))
    assert len(predictions)==10000 and correct==training['integer_correct']
    assert (CACHE/'mnist/reference-output.bin').read_bytes()==(CACHE/'mnist-scalar/reference-output.bin').read_bytes()
    assert json.loads((CACHE/'mnist/reference.json').read_text())['ndot_instructions']==3*(784*32+32*10)//4
    assert json.loads((CACHE/'mnist-scalar/reference.json').read_text())['ndot_instructions']==0
    reports['mnist'].update(integer_correct=correct,test_examples=10000,float_accuracy=training['float_accuracy'],
        integer_accuracy=correct/10000,native_all_test_predictions_equal=True,
        instruction_reduction_ratio=reports['mnist-scalar']['instructions']/reports['mnist']['instructions'],
        trace_token_reduction_ratio=reports['mnist-scalar']['trace_tokens']/reports['mnist']['trace_tokens'],
        ratio_scope='deterministic guest work counts; not a measured GPU wall-time speedup')
    (HERE/'evidence/local-validation.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
