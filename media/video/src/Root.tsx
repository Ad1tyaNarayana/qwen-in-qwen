import React from 'react';
import {AbsoluteFill, Composition, Sequence, Still} from 'remotion';
import {End, Hero, Hook, Levels, Limits, Nest, Numbers, Orbit, OrbitNdot, Preview, Programs, Story, StoryCard} from './scenes';
import {C, FPS, STORY_FRAMES} from './theme';

const SCENES = [
	[Hook, 90],
	[Nest, 150],
	[Orbit, 120],
	[OrbitNdot, 120],
	[Levels, 120],
	[Story, STORY_FRAMES + 45],
	[Programs, 105],
	[Numbers, 105],
	[Limits, 105],
	[End, 75],
] as const;

const TOTAL = SCENES.reduce((sum, [, len]) => sum + len, 0);
const STORY_CARD = 360;

const Social: React.FC = () => {
	let from = 0;
	return (
		<AbsoluteFill style={{background: C.bg}}>
			{SCENES.map(([Scene, len]) => {
				const start = from;
				from += len;
				return <Sequence key={start} from={start} durationInFrames={len}><Scene len={len} /></Sequence>;
			})}
		</AbsoluteFill>
	);
};

export const Root: React.FC = () => (
	<>
		<Composition id="SocialVertical" component={Social} durationInFrames={TOTAL} fps={FPS} width={1080} height={1920} />
		<Composition id="SocialLandscape" component={Social} durationInFrames={TOTAL} fps={FPS} width={1920} height={1080} />
		<Composition id="Hero" component={Hero} durationInFrames={315} fps={FPS} width={1000} height={570} />
		<Composition id="StoryCard" component={() => <StoryCard frames={STORY_CARD} />} durationInFrames={STORY_CARD + 45} fps={FPS} width={1000} height={380} />
		<Still id="Preview" component={Preview} width={1280} height={640} />
	</>
);
