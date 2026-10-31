import React from 'react';
import {AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import story from './story.json';
import {C, CAPTURE, FPS, mono, Region, serif} from './theme';

export const useVertical = () => {
	const {width, height} = useVideoConfig();
	return height > width;
};

// Fits a capture region into its box. `zoom` drifts slowly so stills do not feel frozen.
export const Crop: React.FC<{src: string; region: Region; width: number; height: number; zoom?: number; fadeLeft?: boolean}> = ({
	src,
	region,
	width,
	height,
	zoom = 1,
	fadeLeft = false,
}) => {
	const scale = Math.max(width / region.w, height / region.h) * zoom;
	const left = width / 2 - (region.x + region.w / 2) * scale;
	const top = height / 2 - (region.y + region.h / 2) * scale;
	return (
		<div
			style={{
				position: 'relative',
				width,
				height,
				overflow: 'hidden',
				background: C.bg,
				// Hides the Explorer heading that sits beside the network map.
				maskImage: fadeLeft ? 'linear-gradient(to right, transparent 13%, black 24%)' : undefined,
			}}
		>
			<Img
				src={staticFile(src)}
				style={{position: 'absolute', left, top, width: CAPTURE.width * scale, height: CAPTURE.height * scale, maxWidth: 'none'}}
			/>
		</div>
	);
};

export const Fade: React.FC<{len: number; children: React.ReactNode; edge?: number}> = ({len, children, edge = 12}) => {
	const f = useCurrentFrame();
	const opacity = interpolate(f, [0, edge, len - edge, len], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
	return <AbsoluteFill style={{opacity}}>{children}</AbsoluteFill>;
};

export const Rise: React.FC<{at?: number; children: React.ReactNode}> = ({at = 0, children}) => {
	const f = useCurrentFrame();
	const t = interpolate(f, [at, at + 14], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
	return <div style={{opacity: t, transform: `translateY(${(1 - t) * 18}px)`}}>{children}</div>;
};

export const Kicker: React.FC<{children: React.ReactNode; size?: number; color?: string}> = ({children, size = 24, color = C.gold}) => (
	<div style={{fontFamily: mono, fontSize: size, letterSpacing: '0.14em', textTransform: 'uppercase', color}}>{children}</div>
);

export const Title: React.FC<{children: React.ReactNode; size: number; color?: string}> = ({children, size, color = C.text}) => (
	<div style={{fontFamily: serif, fontSize: size, lineHeight: 1.08, letterSpacing: '-0.015em', color}}>{children}</div>
);

// The label is computed from the real run time, so it cannot drift from the animation.
export const speedFactor = (frames: number, fps = FPS) => Math.round((story.computeSeconds / (frames / fps)) / 10) * 10;

export const StoryText: React.FC<{frames: number; size: number; chips: number}> = ({frames, size, chips}) => {
	const f = useCurrentFrame();
	const shown = Math.min(story.tokens.length, Math.floor((f / frames) * story.tokens.length) + 1);
	const recent = story.tokens.slice(Math.max(0, shown - chips), shown);
	return (
		<>
			<div style={{fontFamily: serif, fontSize: size, lineHeight: 1.26, color: C.text}}>
				{story.tokens.slice(0, shown).join('')}
				<span style={{color: C.gold}}>▌</span>
			</div>
			<div style={{display: 'flex', flexWrap: 'wrap', gap: size * 0.2, marginTop: size * 0.7}}>
				{recent.map((token, i) => (
					<span
						key={shown - recent.length + i}
						style={{
							fontFamily: mono,
							fontSize: size * 0.42,
							whiteSpace: 'pre',
							color: i === recent.length - 1 ? C.bg : C.gold,
							background: i === recent.length - 1 ? C.gold : 'transparent',
							border: `1px solid ${C.gold}`,
							borderRadius: 4,
							padding: `${size * 0.08}px ${size * 0.2}px`,
						}}
					>
						{JSON.stringify(token).slice(1, -1)}
					</span>
				))}
			</div>
		</>
	);
};
