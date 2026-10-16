import React, {useState} from 'react';
import {Img, staticFile} from 'remotion';
import {useTheme} from '../theme';
import {useLayout} from '../layout';
import {useVideo} from '../context';
import {useEnter, useRaised} from '../components';

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
 * A paper figure as a raised object. `w` x `h` is the image area; the frame adds padding.
 * Unknown ids and missing files render a labelled placeholder. `children` draw inside the clip, over the image.
 */
export const FigureCard: React.FC<{id?: string | null; w: number; h: number; pad?: number; wrapStyle?: React.CSSProperties; style?: React.CSSProperties; children?: React.ReactNode}> = ({id, w, h, pad = 14, wrapStyle, style, children}) => {
  const {u} = useLayout();
  const {r} = useTheme();
  const raised = useRaised();
  const {asset} = useFigure(id);
  const [failed, setFailed] = useState(false);
  const showImg = asset && !failed;
  return (
    <div style={{...raised, padding: pad * u, flex: 'none', boxSizing: 'content-box', ...style}}>
      <div style={{position: 'relative', width: w, height: h, overflow: 'hidden', borderRadius: Math.max(0, r - 8) * u * 0.5}}>
        <div style={{position: 'absolute', inset: 0, transformOrigin: '0 0', ...wrapStyle}}>
          {showImg ? <Img src={staticFile(asset.file)} maxRetries={0} onError={() => setFailed(true)} style={{width: '100%', height: '100%', display: 'block'}} /> : <Placeholder id={id} />}
          {children}
        </div>
      </div>
    </div>
  );
};

const Placeholder: React.FC<{id?: string | null}> = ({id}) => {
  const {u} = useLayout();
  const {sans, mono} = useTheme();
  return (
    <div style={{width: '100%', height: '100%', background: 'repeating-linear-gradient(135deg, #F1F2F4 0 14px, #E7E9EC 14px 28px)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 * u, color: '#5A6270'}}>
      <div style={{fontFamily: sans, fontWeight: 700, fontSize: 44 * u, letterSpacing: '-0.02em'}}>Figure unavailable</div>
      <div style={{fontFamily: mono, fontSize: 24 * u}}>{id || 'no id'}</div>
    </div>
  );
};

/** Fade and rise in at a scene-local frame. */
export const Reveal: React.FC<{at: number; children: React.ReactNode; rise?: number; len?: number; style?: React.CSSProperties}> = ({at, children, rise = 20, len = 16, style}) => {
  const p = useEnter(at, len);
  const {u} = useLayout();
  return <div style={{opacity: p, translate: `0px ${(1 - p) * rise * u}px`, ...style}}>{children}</div>;
};

/** Small monospace label in the soft ink colour. */
export const Mono: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({children, size = 26, color, style}) => {
  const {u} = useLayout();
  const {C, mono} = useTheme();
  return <div style={{fontFamily: mono, fontWeight: 500, fontSize: size * u, color: color ?? C.soft, lineHeight: 1.3, ...style}}>{children}</div>;
};

/** Frames for n tokens that are not tied to speech: a quick even stagger from `at`. */
export const stagger = (at: number, n: number, gap = 3) => Array.from({length: n}, (_, i) => at + i * gap);
