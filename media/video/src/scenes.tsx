import React from 'react';
import {AbsoluteFill, interpolate, Sequence, useCurrentFrame, useVideoConfig} from 'remotion';
import {Crop, Fade, Kicker, Rise, speedFactor, StoryText, Title, useVertical} from './parts';
import story from './story.json';
import {C, mono, REGION, serif, STORY_FRAMES} from './theme';

type Scene = React.FC<{len: number}>;
const pad = (v: boolean) => (v ? '150px 80px 320px' : '90px 110px');

export const Hook: Scene = ({len}) => {
	const v = useVertical();
	return (
		<Fade len={len}>
			<AbsoluteFill style={{justifyContent: 'center', padding: pad(v), background: C.bg}}>
				<Rise><Title size={v ? 128 : 124}>A CPU made of<br />LLM layers.</Title></Rise>
				<div style={{height: v ? 50 : 34}} />
				<Rise at={34}><Title size={v ? 128 : 124} color={C.gold}>Running an LLM.</Title></Rise>
			</AbsoluteFill>
		</Fade>
	);
};

const LAYERS = [
	['Qwen3.5-architecture network', 'Fixed compiled weights. None pretrained.', C.gold],
	['RISC-V CPU', 'Registers and RAM in recurrent state.', C.teal],
	['Dot-product accelerator', 'Custom instructions in the same weights.', C.pink],
	['Tiny LLM', 'An ordinary program in memory.', C.text],
] as const;

export const Nest: Scene = ({len}) => {
	const v = useVertical();
	const f = useCurrentFrame();
	const box: React.FC<{i: number}> = ({i}) => {
		const [name, note, color] = LAYERS[i];
		const on = interpolate(f, [i * 34, i * 34 + 16], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
		return (
			<div style={{opacity: on, border: `2px solid ${color}`, borderRadius: 14, padding: v ? '26px 28px' : '20px 26px', background: `${C.panel}cc`}}>
				<div style={{fontFamily: serif, fontSize: v ? 50 : 40, color}}>{name}</div>
				<div style={{fontFamily: mono, fontSize: v ? 25 : 21, color: C.dim, margin: '8px 0 16px'}}>{note}</div>
				{i < 3 && React.createElement(box, {i: i + 1})}
			</div>
		);
	};
	return (
		<Fade len={len}>
			<AbsoluteFill style={{justifyContent: 'center', padding: pad(v), background: C.bg}}>
				<Kicker size={v ? 26 : 22}>Diagram</Kicker>
				<div style={{height: 22}} />
				{React.createElement(box, {i: 0})}
			</AbsoluteFill>
		</Fade>
	);
};

const Footage: React.FC<{dir: string; title: string; note: string; color: string; len: number}> = ({dir, title, note, color, len}) => {
	const v = useVertical();
	const {width, height} = useVideoConfig();
	const f = useCurrentFrame();
	const frame = String(Math.min(149, f)).padStart(4, '0');
	const side = v ? width : Math.round(width * 0.58);
	const tall = v ? 1120 : height;
	return (
		<Fade len={len}>
			<AbsoluteFill style={{background: C.bg, flexDirection: v ? 'column' : 'row-reverse'}}>
				<div style={{position: 'relative', marginTop: v ? 130 : 0}}>
					<Crop src={`${dir}/${frame}.jpg`} region={REGION.network} width={side} height={tall} fadeLeft />
				</div>
				<div style={{flex: 1, padding: v ? '30px 80px' : '0 0 0 110px', display: 'flex', flexDirection: 'column', justifyContent: v ? 'flex-start' : 'center'}}>
					<Kicker size={v ? 25 : 22} color={color}>Real Explorer capture</Kicker>
					<div style={{height: 22}} />
					<Rise><Title size={v ? 74 : 72}>{title}</Title></Rise>
					<div style={{height: 20}} />
					<Rise at={16}><div style={{fontFamily: mono, fontSize: v ? 30 : 27, lineHeight: 1.4, color: C.dim}}>{note}</div></Rise>
				</div>
			</AbsoluteFill>
		</Fade>
	);
};

export const Orbit: Scene = ({len}) => (
	<Footage dir="orbit" len={len} color={C.gold} title="Every weight placed by a compiler." note="65 layers. 2,304 hidden coordinates." />
);

export const OrbitNdot: Scene = ({len}) => (
	<Footage dir="orbit-ndot" len={len} color={C.pink} title="The accelerator is in there too." note="24,262 units. Same network." />
);

const Still: React.FC<{src: string; region: keyof typeof REGION; title: string; len: number}> = ({src, region, title, len}) => {
	const v = useVertical();
	const {width} = useVideoConfig();
	const f = useCurrentFrame();
	const zoom = interpolate(f, [0, len], [1, 1.015]);
	return (
		<Fade len={len} edge={8}>
			<AbsoluteFill style={{background: C.bg, justifyContent: 'center', alignItems: 'center'}}>
				<div style={{width: '100%', padding: v ? '0 80px 34px' : '0 110px 26px', boxSizing: 'border-box'}}>
					<Kicker size={v ? 25 : 22}>Real Explorer capture</Kicker>
					<div style={{height: 14}} />
					<Title size={v ? 80 : 70}>{title}</Title>
				</div>
				<Crop src={src} region={REGION[region]} width={v ? width : 1365} height={v ? 617 : 780} zoom={zoom} />
			</AbsoluteFill>
		</Fade>
	);
};

export const Levels: Scene = ({len}) => {
	const part = Math.floor(len / 3);
	return (
		<>
			<Sequence durationInFrames={part}><Still src="l2-cpu.png" region="cpu" title="Into the CPU." len={part} /></Sequence>
			<Sequence from={part} durationInFrames={part}><Still src="l3-program.png" region="program" title="Into the program." len={part} /></Sequence>
			<Sequence from={part * 2} durationInFrames={len - part * 2}><Still src="l4-app.png" region="app" title="Into the model." len={len - part * 2} /></Sequence>
		</>
	);
};

export const Story: Scene = ({len}) => {
	const v = useVertical();
	return (
		<Fade len={len} edge={10}>
			<AbsoluteFill style={{background: C.bg, justifyContent: 'center', padding: pad(v)}}>
				<Kicker size={v ? 25 : 22}>Recorded output from stories260K</Kicker>
				<div style={{height: 28}} />
				<div style={{minHeight: v ? 880 : 610}}>
					<StoryText frames={STORY_FRAMES} size={v ? 68 : 72} chips={v ? 4 : 6} />
				</div>
				<div style={{fontFamily: mono, fontSize: v ? 27 : 25, lineHeight: 1.45, color: C.teal}}>
					Prerecorded. Sped up about {speedFactor(STORY_FRAMES).toLocaleString('en-US')}×.<br />
					Real speed: {story.minutesPerPosition.toFixed(2)} minutes per position.
				</div>
			</AbsoluteFill>
		</Fade>
	);
};

export const Programs: Scene = ({len}) => {
	const v = useVertical();
	return (
		<Fade len={len}>
			<AbsoluteFill style={{background: C.bg, justifyContent: 'center', padding: pad(v)}}>
				<Rise><Title size={v ? 92 : 80}>One network.<br />Different programs.</Title></Rise>
				<div style={{height: 36}} />
				<div style={{display: 'flex', flexDirection: v ? 'column' : 'row', gap: 40}}>
					<Rise at={10}><Crop src="life.png" region={REGION.life} width={v ? 920 : 540} height={v ? 600 : 430} /></Rise>
					<Rise at={24}><Crop src="mnist.png" region={REGION.mnist} width={v ? 920 : 1120} height={v ? 350 : 430} /></Rise>
				</div>
				<div style={{height: 28}} />
				<div style={{fontFamily: mono, fontSize: v ? 28 : 25, color: C.dim}}>Game of Life. MNIST. Also SHA-256 and tic-tac-toe.</div>
			</AbsoluteFill>
		</Fade>
	);
};

const STATS = [
	[story.traceTokens.toLocaleString('en-US'), 'CPU trace tokens for the story', C.gold],
	[String(story.mismatches), 'mismatches against an independent reference', C.teal],
	['3.01×', 'accelerator gain, from one run each', C.pink],
] as const;

export const Numbers: Scene = ({len}) => {
	const v = useVertical();
	return (
		<Fade len={len}>
			<AbsoluteFill style={{background: C.bg, justifyContent: 'center', padding: pad(v), gap: v ? 66 : 42}}>
				{STATS.map(([value, label, color], i) => (
					<Rise key={value} at={i * 22}>
						<div style={{fontFamily: mono, fontWeight: 700, fontSize: v ? 96 : 88, color}}>{value}</div>
						<div style={{fontFamily: mono, fontSize: v ? 30 : 28, color: C.dim, marginTop: 6}}>{label}</div>
					</Rise>
				))}
			</AbsoluteFill>
		</Fade>
	);
};

export const Limits: Scene = ({len}) => {
	const v = useVertical();
	return (
		<Fade len={len}>
			<AbsoluteFill style={{background: C.bg, justifyContent: 'center', padding: pad(v)}}>
				<Rise><Title size={v ? 96 : 92}>It is gloriously slow.</Title></Rise>
				<div style={{height: 34}} />
				<Rise at={22}>
					<div style={{fontFamily: mono, fontSize: v ? 33 : 31, lineHeight: 1.5, color: C.dim}}>
						About 6 minutes per position on an RTX 5070 Ti.<br /><br />
						Complete runs use a compact executor on the compiled weights. Stock Hugging Face generate checked bounded prefixes.
					</div>
				</Rise>
			</AbsoluteFill>
		</Fade>
	);
};

export const End: Scene = ({len}) => {
	const v = useVertical();
	return (
		<Fade len={len}>
			<AbsoluteFill style={{background: C.bg, justifyContent: 'center', alignItems: 'center', textAlign: 'center', padding: pad(v)}}>
				<Title size={v ? 150 : 170} color={C.gold}>Qwen in Qwen</Title>
				<div style={{height: 30}} />
				<div style={{fontFamily: mono, fontSize: v ? 33 : 36, color: C.text}}>github.com/Ad1tyaNarayana/qwen-in-qwen</div>
				<div style={{height: 26}} />
				<div style={{fontFamily: mono, fontSize: v ? 26 : 26, color: C.dim}}>Built on Qwenputer by Divyansh Agrawal</div>
			</AbsoluteFill>
		</Fade>
	);
};

// README loop: real captures only, ending on the generated text.
export const Hero: React.FC = () => {
	const {width, height} = useVideoConfig();
	const f = useCurrentFrame();
	const caption = (text: string) => (
		<div style={{position: 'absolute', left: 0, right: 0, bottom: 0, padding: '12px 22px', background: `${C.bg}e6`, borderTop: `1px solid ${C.line}`, fontFamily: mono, fontSize: 22, color: C.gold}}>{text}</div>
	);
	const orbit = (dir: string, text: string, start: number) => (
		<AbsoluteFill>
			<Crop src={`${dir}/${String(Math.min(149, (f - start) * 2)).padStart(4, '0')}.jpg`} region={REGION.networkWide} width={width} height={height} />
			{caption(text)}
		</AbsoluteFill>
	);
	const still = (src: string, region: keyof typeof REGION, text: string) => (
		<AbsoluteFill><Crop src={src} region={REGION[region]} width={width} height={height} />{caption(text)}</AbsoluteFill>
	);
	return (
		<AbsoluteFill style={{background: C.bg}}>
			<Sequence durationInFrames={75}>{orbit('orbit', 'L0  The Qwen3.5 network: 65 layers of compiled weights', 0)}</Sequence>
			<Sequence from={75} durationInFrames={75}>{orbit('orbit-ndot', 'L0  The accelerator inside it: 24,262 units', 75)}</Sequence>
			<Sequence from={150} durationInFrames={45}>{still('l2-cpu.png', 'cpu', 'L2  The CPU, read from neural state')}</Sequence>
			<Sequence from={195} durationInFrames={45}>{still('l3-program.png', 'program', 'L3  The program it runs')}</Sequence>
			<Sequence from={240} durationInFrames={75}>{still('l4-app.png', 'app', 'L4  A tiny Qwen3.5 inside: "Once upon a time, there"')}</Sequence>
		</AbsoluteFill>
	);
};

export const StoryCard: React.FC<{frames: number}> = ({frames}) => (
	<AbsoluteFill style={{background: C.bg, padding: '34px 40px'}}>
		<div style={{flex: 1}}><StoryText frames={frames} size={34} chips={8} /></div>
		<div style={{fontFamily: mono, fontSize: 19, color: C.teal}}>
			Recorded stories260K output. Prerecorded, sped up about {speedFactor(frames).toLocaleString('en-US')}×. Real speed: {story.minutesPerPosition.toFixed(2)} minutes per position.
		</div>
	</AbsoluteFill>
);

export const Preview: React.FC = () => (
	<AbsoluteFill style={{background: C.bg, flexDirection: 'row'}}>
		<div style={{width: 640, padding: '0 40px 0 64px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
			<Title size={86} color={C.gold}>Qwen in Qwen</Title>
			<div style={{height: 26}} />
			<div style={{fontFamily: mono, fontSize: 28, lineHeight: 1.42, color: C.text}}>A CPU and accelerator compiled into Qwen3.5 weights, running tiny LLMs.</div>
		</div>
		<Crop src="orbit-ndot/0075.jpg" region={REGION.network} width={640} height={640} fadeLeft />
	</AbsoluteFill>
);

