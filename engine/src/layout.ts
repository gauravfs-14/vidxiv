import {useVideoConfig} from 'remotion';

/**
 * Responsive geometry. Size everything in design units times `u` (short side / 1080).
 * `stage` is the content box between the kicker and the captions; portrait keeps clear of the social UI zones.
 */
export type Layout = ReturnType<typeof useLayout>;

export const useLayout = () => {
  const {width: W, height: H} = useVideoConfig();
  const u = Math.min(W, H) / 1080;
  const aspect = W / H;
  const orientation: 'portrait' | 'landscape' | 'square' = aspect < 0.8 ? 'portrait' : aspect > 1.25 ? 'landscape' : 'square';
  const isPortrait = orientation === 'portrait';
  const isLandscape = orientation === 'landscape';

  const caption = isPortrait
    ? {bottom: H * 0.19, width: W - 160 * u, size: 72 * u, maxWords: 4}
    : isLandscape
      ? {bottom: H * 0.075, width: W * 0.72, size: 62 * u, maxWords: 6}
      : {bottom: H * 0.08, width: W - 140 * u, size: 62 * u, maxWords: 5};
  const captionTop = H - caption.bottom - caption.size * 2.4;

  const pad = (isLandscape ? 90 : 60) * u;
  const kickerTop = isPortrait ? H * 0.09 : 56 * u;
  const stageTop = kickerTop + 70 * u;
  const stage = {x: pad, y: stageTop, w: W - pad * 2, h: captionTop - stageTop - 30 * u};

  return {
    W, H, u, aspect, orientation, isPortrait, isLandscape, pad, kickerTop, caption, stage,
    stack: (isLandscape ? 'row' : 'column') as 'row' | 'column',
    fit: (k = 1) => Math.min(stage.w, stage.h) * k,
  };
};

/**
 * Largest font size (px, from `max` down to `min`) at which `text` is expected to fit a w x h box.
 * `cw` is the average glyph width in em (serif ~0.42, sans ~0.5), `lh` the line height. A rough estimate by design.
 */
export const fitFont = (text: string, w: number, h: number, max: number, min: number, cw = 0.5, lh = 1.15) => {
  const len = Math.max(1, (text ?? '').length) * 1.07;
  for (let s = max; s > min; s -= 1) {
    const perLine = Math.max(1, Math.floor(w / (s * cw)));
    if (Math.ceil(len / perLine) * s * lh <= h) return s;
  }
  return min;
};
