import {Config} from '@remotion/cli/config';

// Real Explorer captures from media/capture/capture.sh stay out of git.
Config.setPublicDir('../../.cache/media/capture');
Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(92);
Config.setCodec('h264');
Config.setCrf(18);
