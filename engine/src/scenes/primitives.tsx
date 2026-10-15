import React, {useState} from 'react';
import {Img, interpolate, staticFile} from 'remotion';
import {MONO, SANS, SERIF, useTheme} from '../theme';
import {useLayout} from '../layout';
import {useVideo} from '../context';
import {Boil, clamp, useEnter, useSafeId} from '../components';

/** A hand-drawn arrow that draws on. Coordinates are px inside the nearest positioned parent. */
export const HandArrow: React.FC<{x1: number; y1: number; x2: number; y2: number; progress: number; color: string; width?: number; bow?: number; head?: number}> = ({x1, y1, x2, y2, progress, color, width = 5, bow = 0.12, head = 22}) => {
  const {u} = useLayout();
  const id = useSafeId('ha');
  if (progress <= 0) return null;
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  const cx = (x1 + x2) / 2 - ((y2 - y1) / len) * len * bow;
  const cy = (y1 + y2) / 2 + ((x2 - x1) / len) * len * bow;
  const a = Math.atan2(y2 - cy, x2 - cx);
  const h = head * u;
  const hp = interpolate(progress, [0.8, 1], [0, 1], clamp);
  const tip = (da: number) => `${x2 - Math.cos(a + da) * h},${y2 - Math.sin(a + da) * h}`;
  return (
    <svg style={{position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none'}}>
      <defs>
        <Boil id={id} scale={2.2 * u} />
      </defs>
      <g filter={`url(#${id})`} stroke={color} strokeWidth={width * u} strokeLinecap="round" strokeLinejoin="round" fill="none">
        <path d={`M${x1},${y1} Q${cx},${cy} ${x2},${y2}`} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - Math.min(1, progress / 0.9)} />
        <path d={`M${tip(0.5)} L${x2},${y2} L${tip(-0.5)}`} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - hp} />
      </g>
    </svg>
  );
};

/** Arrow between two flow cards: horizontal or vertical, sized by its own box. */
export const FlowArrow: React.FC<{dir: 'h' | 'v'; progress: number; color: string; size: number}> = ({dir, progress, color, size}) => {
  const {u} = useLayout();
  const s = size * u;
  const w = dir === 'h' ? s : 44 * u;
  const h = dir === 'h' ? 44 * u : s;
  return (
    <div style={{position: 'relative', width: w, height: h, flex: 'none'}}>
      {dir === 'h' ? <HandArrow x1={4 * u} y1={h / 2} x2={w - 4 * u} y2={h / 2} progress={progress} color={color} bow={0.14} head={18} /> : <HandArrow x1={w / 2} y1={4 * u} x2={w / 2} y2={h - 4 * u} progress={progress} color={color} bow={0.14} head={18} />}
    </div>
  );
};

export const fitBox = (aw: number, ah: number, maxW: number, maxH: number) => {
  const s = Math.min(maxW / (aw || 1), maxH / (ah || 1));
  return {w: Math.max(1, aw * s), h: Math.max(1, ah * s)};
};

/** Figure asset by id, with the dimensions to lay out against (3:2 for unknown ids). */
export const useFigure = (id?: string | null) => {
  const {figures} = useVideo();
  const f = id ? figures?.[id] : undefined;
  const ok = !!f?.file;
  return {id: id ?? '', asset: ok ? f : undefined, aw: ok && f!.width > 0 ? f!.width : 1200, ah: ok && f!.height > 0 ? f!.height : 800};
};

/**
 * A paper figure on a white card with a soft shadow. `w` x `h` is the image area; the card adds padding.
 * Unknown ids and missing files render a labelled placeholder. `children` draw inside the clip, over the image.
 */
export const FigureCard: React.FC<{id?: string | null; w: number; h: number; pad?: number; wrapStyle?: React.CSSProperties; style?: React.CSSProperties; children?: React.ReactNode}> = ({id, w, h, pad = 16, wrapStyle, style, children}) => {
  const {u} = useLayout();
  const {C} = useTheme();
  const {asset} = useFigure(id);
  const [failed, setFailed] = useState(false);
  const showImg = asset && !failed;
  return (
    <div style={{background: '#FFFFFF', padding: pad * u, borderRadius: 18 * u, border: `${2 * u}px solid ${C.line}`, boxShadow: `0 ${24 * u}px ${60 * u}px ${C.shadow}, 0 ${4 * u}px ${12 * u}px rgba(0,0,0,0.06)`, flex: 'none', ...style}}>
      <div style={{position: 'relative', width: w, height: h, overflow: 'hidden', borderRadius: 6 * u}}>
        <div style={{position: 'absolute', inset: 0, transformOrigin: '0 0', ...wrapStyle}}>
          {showImg ? (
            <Img src={staticFile(asset.file)} maxRetries={0} onError={() => setFailed(true)} style={{width: '100%', height: '100%', display: 'block'}} />
          ) : (
            <Placeholder id={id} />
          )}
          {children}
        </div>
      </div>
    </div>
  );
};

const Placeholder: React.FC<{id?: string | null}> = ({id}) => {
  const {u} = useLayout();
  return (
    <div style={{width: '100%', height: '100%', background: 'repeating-linear-gradient(135deg, #F4EFE6 0 14px, #EBE4D6 14px 28px)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 * u, color: '#6A6359'}}>
      <div style={{fontFamily: SERIF, fontSize: 52 * u, fontStyle: 'italic'}}>Figure unavailable</div>
      <div style={{fontFamily: MONO, fontSize: 24 * u}}>{id || 'no id'}</div>
    </div>
  );
};

/** Pill chip, used for meta lines and links. */
export const Chip: React.FC<{children: React.ReactNode; size?: number; color?: string; mono?: boolean; fill?: string; style?: React.CSSProperties}> = ({children, size = 28, color, mono = true, fill, style}) => {
  const {u} = useLayout();
  const {C} = useTheme();
  return (
    <div style={{display: 'inline-flex', alignItems: 'center', gap: 12 * u, fontFamily: mono ? MONO : SANS, fontWeight: mono ? 500 : 600, fontSize: size * u, color: color ?? C.ink, background: fill ?? C.muted, border: `${2 * u}px solid ${C.line}`, borderRadius: 999, padding: `${size * 0.32 * u}px ${size * 0.8 * u}px`, ...style}}>
      {children}
    </div>
  );
};

/** Fade and rise in at a scene-local frame. */
export const Reveal: React.FC<{at: number; children: React.ReactNode; rise?: number; damping?: number; style?: React.CSSProperties}> = ({at, children, rise = 28, damping = 200, style}) => {
  const p = useEnter(at, damping);
  const {u} = useLayout();
  return <div style={{opacity: Math.min(1, p * 1.4), translate: `0px ${(1 - p) * rise * u}px`, ...style}}>{children}</div>;
};

/** Small round numeral badge. */
export const Badge: React.FC<{n: number | string; color: string; size?: number}> = ({n, color, size = 64}) => {
  const {u} = useLayout();
  const {C} = useTheme();
  return (
    <div style={{width: size * u, height: size * u, borderRadius: '50%', background: color, color: C.dark ? C.paper : '#FFFCF6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: SERIF, fontSize: size * 0.62 * u, lineHeight: 1, flex: 'none', paddingBottom: size * 0.04 * u}}>
      {n}
    </div>
  );
};

/** A hand-drawn wavy rule that draws on. */
export const Squiggle: React.FC<{width: number; color: string; progress: number; stroke?: number}> = ({width, color, progress, stroke = 6}) => {
  const {u} = useLayout();
  return (
    <svg width={width * u} height={width * 0.12 * u} viewBox="0 0 200 24" style={{overflow: 'visible', display: 'block'}}>
      <path d="M3 13 C 30 3, 58 23, 92 11 S 150 3, 197 12" fill="none" stroke={color} strokeWidth={stroke * 200 / width} strokeLinecap="round" pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - progress} />
    </svg>
  );
};
