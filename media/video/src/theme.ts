import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import {loadFont as loadSerif} from '@remotion/google-fonts/Newsreader';

export const serif = loadSerif('normal', {weights: ['400'], subsets: ['latin']}).fontFamily;
export const mono = loadMono('normal', {weights: ['400', '700'], subsets: ['latin']}).fontFamily;

// Matches the Explorer.
export const C = {
	bg: '#0c1119',
	panel: '#111923',
	line: '#26303d',
	text: '#e6e8e8',
	dim: '#8c9aa9',
	gold: '#edbd63',
	teal: '#7fd8cf',
	pink: '#d981b0',
};

export const FPS = 30;

export const CAPTURE = {width: 2880, height: 2080};

export type Region = {x: number; y: number; w: number; h: number};

// Regions of the 2x Explorer captures.
export const REGION = {
	network: {x: 1000, y: 490, w: 1460, h: 1200},
	networkWide: {x: 590, y: 420, w: 2260, h: 1290},
	cpu: {x: 560, y: 250, w: 2320, h: 1325},
	program: {x: 560, y: 250, w: 2320, h: 1325},
	app: {x: 560, y: 250, w: 2320, h: 1325},
	life: {x: 650, y: 320, w: 1150, h: 920},
	mnist: {x: 640, y: 640, w: 2200, h: 830},
} satisfies Record<string, Region>;

// Prerecorded story timing, in frames.
export const STORY_FRAMES = 240;
