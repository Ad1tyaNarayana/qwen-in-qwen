"""Render the verified output frames into a local replay, PNG, and animated GIF."""
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=Path("artifacts/compact-game"))
    args = parser.parse_args()
    frames = json.loads((args.directory / "frames.json").read_text())
    report = json.loads((args.directory / "report.json").read_text())
    images = []
    for frame in frames:
        image = Image.new("RGB", (64, 32), "#101b22")
        image.putdata([(166, 247, 207) if frame['bytes'][y * 8 + x // 8] & (128 >> (x & 7)) else (16, 27, 34)
                       for y in range(32) for x in range(64)])
        images.append(image.resize((1024, 512), Image.Resampling.NEAREST))
    images[-1].save(args.directory / "frame.png")
    images[0].save(args.directory / "replay.gif", save_all=True, append_images=images[1:], duration=250, loop=0)
    document = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weight Paddle · Qwenputer</title>
<style>body{background:#0a1219;color:#e7f1f7;font:16px system-ui;max-width:1100px;margin:7vh auto;padding:24px}small{color:#9baebd}h1{font-size:48px;letter-spacing:-2px;margin-bottom:8px}canvas{width:100%;image-rendering:pixelated;background:#101b22;border:1px solid #294453;border-radius:8px;margin-top:28px}button,input{accent-color:#a6f7cf}button{background:#a6f7cf;border:0;border-radius:6px;padding:10px 20px;cursor:pointer}section{display:flex;justify-content:space-between;gap:24px;margin-top:20px;align-items:center}#scrub{flex:1}p{line-height:1.6;max-width:850px}a{color:#a6f7cf}</style>
<small>QWENPUTER / RISC-V / CHIP-8 · VERIFIED PRERECORDED REPLAY</small><h1>Weight Paddle</h1>
<p>An original CHIP-8 paddle game interpreted by C firmware on the compiled RISC-V circuit. Every displayed frame comes from the verified tensor execution.</p>
<canvas width="64" height="32"></canvas><section><button id="play">Pause</button><input id="scrub" type="range" min="0"><span id="status"></span></section>
<p id="evidence"></p><small>This is a prerecorded replay of upstream's compact circuit tensor executor on an Apple M4 Pro CPU. Playback runs at four frames per second. Actual execution time appears above. The game frames have not been generated through the full Qwen language model. A separate stock Qwen3_5ForCausalLM smoke test verifies the compiled RISC-V weights.</small>
<p><a href="report.json">Execution report</a> · <a href="frames.json">Verified frames</a> · <a href="game.json">Original ROM and key schedule</a></p>
<script>
const frames=FRAMES, report=REPORT, canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),scrub=document.querySelector('#scrub'),play=document.querySelector('#play');
let index=0,playing=true;scrub.max=frames.length-1;
function draw(){const f=frames[index];ctx.fillStyle='#101b22';ctx.fillRect(0,0,64,32);ctx.fillStyle='#a6f7cf';for(let y=0;y<32;y++)for(let x=0;x<64;x++)if(f.bytes[y*8+(x>>3)]&(128>>(x&7)))ctx.fillRect(x,y,1,1);scrub.value=index;document.querySelector('#status').textContent=`Frame ${index+1}/${frames.length} · Hits ${f.hits} · Misses ${f.misses}`;}
play.onclick=()=>{playing=!playing;play.textContent=playing?'Pause':'Play'};scrub.oninput=()=>{index=+scrub.value;draw()};setInterval(()=>{if(playing){index=(index+1)%frames.length;draw()}},250);draw();
document.querySelector('#evidence').textContent=`${report.generated_tokens.toLocaleString()} self-generated trace tokens checked. ${report.mismatches} mismatches. ${report.distinct_frames} distinct frames. Actual execution took ${report.elapsed_seconds_one_run.toFixed(1)} seconds in one run. Inputs and firmware are preserved beside the report.`;
</script>'''
    (args.directory / "index.html").write_text(document.replace("FRAMES", json.dumps(frames)).replace("REPORT", json.dumps(report)))
    print(args.directory / "index.html")


if __name__ == "__main__":
    main()
