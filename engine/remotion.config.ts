import {Config} from '@remotion/cli/config';

// Paper texture, grain and light leaks are WebGL effects.
Config.setChromiumOpenGlRenderer('angle');
Config.setVideoImageFormat('jpeg');
Config.setPublicDir('public');
