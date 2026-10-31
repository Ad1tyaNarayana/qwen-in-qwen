// Drives the real Explorer over the Chrome DevTools protocol. Run through capture.sh.
import {mkdir, writeFile} from 'node:fs/promises';

const OUT = process.argv[2];
const SCALE = 2, W = 1440, H = 1040, ORBIT = 150;
const targets = await (await fetch('http://127.0.0.1:9223/json')).json();
const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, {once: true}));
let id = 0;
const pending = new Map(), errors = [];
ws.addEventListener('message', e => {
	const m = JSON.parse(e.data);
	if (m.id) {
		const [ok, bad] = pending.get(m.id) ?? [];
		pending.delete(m.id);
		m.error ? bad?.(Error(JSON.stringify(m.error))) : ok?.(m.result);
	} else if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise((ok, bad) => {
	const n = ++id;
	pending.set(n, [ok, bad]);
	ws.send(JSON.stringify({id: n, method, params}));
});
async function run(expression) {
	if (expression.includes('await ')) expression = `(async()=>{${expression}})()`;
	const r = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
	if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
	return r.result.value;
}
const settle = () => run('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>setTimeout(r,120))))');
async function shot(path) {
	await settle();
	const {data} = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
	await writeFile(`${OUT}/${path}.png`, Buffer.from(data, 'base64'));
	console.log(path);
}
async function open(demo) {
	const index = await run(`explorer.state.manifests.findIndex(x=>x.manifest.demo==='${demo}')`);
	if (index < 0) throw Error(`No manifest for ${demo}`);
	await run(`await explorer.openDemo(${index})`);
	await settle();
}
const role = name => run(`{const f=document.getElementById('roleFilter');f.value=[...f.options].find(o=>o.textContent.startsWith('${name}'))?.value??'';f.dispatchEvent(new Event('change'))}`);

try {
	await mkdir(`${OUT}/orbit`, {recursive: true});
	await mkdir(`${OUT}/orbit-ndot`, {recursive: true});
	await send('Runtime.enable');
	await send('Page.enable');
	await send('Emulation.setEmulatedMedia', {features: [{name: 'prefers-reduced-motion', value: 'reduce'}]});
	await send('Emulation.setDeviceMetricsOverride', {width: W, height: H, deviceScaleFactor: SCALE, mobile: false});
	await send('Page.navigate', {url: 'http://127.0.0.1:8792/#gallery'});
	for (let i = 0; i < 200 && !(await run('document.documentElement.dataset.ready==="true"')); i++) await new Promise(r => setTimeout(r, 100));
	await shot('gallery');
	await run(`document.getElementById('detail').scrollTop=[...document.querySelectorAll('.demoCard')].find(c=>c.textContent.includes('life')).offsetTop-40`);
	await shot('gallery-programs');
	await open('tiny-qwen35');
	await run('explorer.setLevel(0)');
	await settle();
	await shot('l0-network');
	for (const [dir, filter] of [['orbit', ''], ['orbit-ndot', 'NDOT']]) {
		if (process.env.SKIP_ORBIT) break;
		if (filter) await role(filter);
		for (let i = 0; i < ORBIT; i++) {
			await run(`explorer.map.yaw=${-0.34 + (i / ORBIT) * 1.3};explorer.map.draw()`);
			const {data} = await send('Page.captureScreenshot', {format: 'jpeg', quality: 92});
			await writeFile(`${OUT}/${dir}/${String(i).padStart(4, '0')}.jpg`, Buffer.from(data, 'base64'));
		}
		console.log(dir);
	}
	await run('explorer.map.reset();explorer.map.draw()');
	await shot('l0-ndot');
	await role('');
	await run(`{const s=document.getElementById('search');s.value='ndot';s.dispatchEvent(new Event('input'))}`);
	await settle();
	await run(`document.querySelector('#results [data-head],#results button')?.click()`);
	await run('explorer.setLevel(1)');
	await shot('l1-layer');
	await run(`{const s=document.getElementById('search');s.value='';s.dispatchEvent(new Event('input'))}`);
	for (const [level, name] of [[2, 'l2-cpu'], [3, 'l3-program'], [4, 'l4-app']]) {
		await run(`explorer.setLevel(${level})`);
		await shot(name);
	}
	await open('stories260K ndot 38-position story');
	await run('explorer.setLevel(4)');
	await shot('stories');
	for (const demo of ['life', 'mnist', 'sha256', 'tictactoe']) {
		await open(demo);
		await run('explorer.setLevel(4)');
		await run(`document.getElementById('detail').scrollTop=${{life: 300, mnist: 300}[demo] ?? 0}`);
		await shot(demo);
	}
	if (errors.length) throw Error(errors.join('\n'));
} finally {
	ws.close();
}
