import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {useTheme} from './theme';
import {useLayout} from './layout';
import {clamp, easeOut, prog} from './components';

/* Crisp annotations. Every mark is a straight bar, a clean block or a drawn rectangle: nothing is hand-drawn. */

/** A bar that wipes in from the left under its children. */
export const Underline: React.FC<{progress: number; color: string; thickness?: number; offset?: number; children: React.ReactNode; style?: React.CSSProperties}> = ({progress, color, thickness, offset = 0.04, children, style}) => {
  const {u} = useLayout();
  const t = thickness ?? Math.max(4 * u, 0);
  return (
    <span style={{position: 'relative', display: 'inline-block', ...style}}>
      {children}
      <span style={{position: 'absolute', left: 0, right: 0, bottom: `${-offset}em`, height: t, background: color, transformOrigin: '0 50%', transform: `scaleX(${progress})`}} />
    </span>
  );
};

/** A tinted block behind the words, wiped in from the left. `solid` blocks are used with contrast text. */
export const Marker: React.FC<{progress: number; color: string; children: React.ReactNode; padX?: number; style?: React.CSSProperties}> = ({progress, color, children, padX = 0.12, style}) => {
  const {r} = useTheme();
  const {u} = useLayout();
  return (
    <span style={{position: 'relative', display: 'inline-block', ...style}}>
      <span style={{position: 'absolute', left: `${-padX}em`, right: `${-padX}em`, top: '0.04em', bottom: '-0.02em', background: color, borderRadius: Math.min(r * 0.4 * u, 14 * u), transformOrigin: '0 50%', transform: `scaleX(${progress})`}} />
      <span style={{position: 'relative'}}>{children}</span>
    </span>
  );
};

/** A rectangle outline whose stroke draws around its children. Sized by CSS, so it follows the text whatever the font. */
export const Outline: React.FC<{progress: number; color: string; stroke: number; padX?: number; padY?: number; children: React.ReactNode; style?: React.CSSProperties}> = ({progress, color, stroke, padX = 14, padY = 6, children, style}) => {
  const {r} = useTheme();
  const {u} = useLayout();
  return (
    <span style={{position: 'relative', display: 'inline-block', ...style}}>
      {children}
      {progress > 0 && (
        <svg style={{position: 'absolute', left: -padX, top: -padY, width: `calc(100% + ${2 * padX}px)`, height: `calc(100% + ${2 * padY}px)`, overflow: 'visible', pointerEvents: 'none'}}>
          <rect x={stroke / 2} y={stroke / 2} rx={r * 0.5 * u} fill="none" stroke={color} strokeWidth={stroke} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - progress} style={{width: `calc(100% - ${stroke}px)`, height: `calc(100% - ${stroke}px)`}} />
        </svg>
      )}
    </span>
  );
};

/** A rule that draws along its axis. */
export const Rule: React.FC<{progress: number; length?: number | string; thickness?: number; vertical?: boolean; color?: string; style?: React.CSSProperties}> = ({progress, length = '100%', thickness, vertical, color, style}) => {
  const {u} = useLayout();
  const {C, rule} = useTheme();
  const t = thickness ?? Math.max(1.5, rule * 0.5) * u;
  return (
    <div style={{width: vertical ? t : length, height: vertical ? length : t, background: color ?? C.line, transformOrigin: '0 0', transform: vertical ? `scaleY(${progress})` : `scaleX(${progress})`, flex: 'none', ...style}} />
  );
};

export type WordMark = {kind: 'outline' | 'underline' | 'marker'; color: string; at?: number};

/**
 * Headline whose words rise out of a mask on the frames in `times` (so each word lands as it is spoken).
 * `marks` annotates words by index.
 */
export const Headline: React.FC<{tokens: string[]; times: number[]; size: number; weight?: number; color?: string; align?: 'left' | 'center'; marks?: Record<number, WordMark>; lineHeight?: number; tracking?: number; style?: React.CSSProperties; rise?: number}> = ({tokens, times, size, weight, color, align = 'left', marks = {}, lineHeight, tracking, style, rise = 1}) => {
  const frame = useCurrentFrame();
  const {C, sans, dw, dt, dl, solid, onColor} = useTheme();
  const {u} = useLayout();
  return (
    <div style={{fontFamily: sans, fontWeight: weight ?? dw, fontSize: size, lineHeight: lineHeight ?? dl, letterSpacing: `${tracking ?? dt}em`, color: color ?? C.ink, textAlign: align, textWrap: 'balance', ...style}}>
      {tokens.map((t, i) => {
        const p = interpolate(frame, [times[i] ?? 0, (times[i] ?? 0) + 14], [0, 1], {...clamp, easing: easeOut});
        const m = marks[i];
        const mp = m ? prog(frame, m.at ?? (times[i] ?? 0) + 8, 16) : 0;
        const word = (
          <span style={{display: 'inline-block', clipPath: `inset(${-0.1 * size}px ${-0.06 * size}px ${-0.12 * size}px ${-0.06 * size}px)`}}>
            <span style={{display: 'inline-block', transform: `translateY(${(1 - p) * 112 * rise}%)`, opacity: p > 0 ? 1 : 0, color: m?.kind === 'marker' && solid ? onColor(m.color) : undefined}}>{t}</span>
          </span>
        );
        let node: React.ReactNode = word;
        if (m?.kind === 'outline') node = <Outline progress={mp} color={m.color} stroke={Math.max(4 * u, size * 0.04)} padX={size * 0.03} padY={size * 0.0}>{word}</Outline>;
        if (m?.kind === 'underline') node = <Underline progress={mp} color={m.color} thickness={Math.max(5 * u, size * 0.06)} offset={0.02}>{word}</Underline>;
        if (m?.kind === 'marker') node = <Marker progress={mp} color={solid ? m.color : `${m.color}40`}>{word}</Marker>;
        return (
          <React.Fragment key={i}>
            {node}
            {i < tokens.length - 1 ? ' ' : ''}
          </React.Fragment>
        );
      })}
    </div>
  );
};

/** Straight leader line with a square end that draws from (x1,y1) to (x2,y2), in px of the nearest positioned parent. */
export const Leader: React.FC<{x1: number; y1: number; x2: number; y2: number; progress: number; color: string; width?: number}> = ({x1, y1, x2, y2, progress, color, width = 3}) => {
  const {u} = useLayout();
  if (progress <= 0) return null;
  const e = interpolate(progress, [0, 0.85], [0, 1], clamp);
  const dot = interpolate(progress, [0.8, 1], [0, 1], clamp);
  return (
    <svg style={{position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none'}}>
      <line x1={x1} y1={y1} x2={x1 + (x2 - x1) * e} y2={y1 + (y2 - y1) * e} stroke={color} strokeWidth={width * u} strokeLinecap="butt" />
      <rect x={x2 - 7 * u} y={y2 - 7 * u} width={14 * u} height={14 * u} fill={color} transform={`scale(${dot})`} style={{transformOrigin: `${x2}px ${y2}px`}} />
    </svg>
  );
};
