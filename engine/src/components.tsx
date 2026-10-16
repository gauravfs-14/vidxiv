import React, {useLayoutEffect, useMemo, useRef, useState} from 'react';
import {AbsoluteFill, Easing, interpolate, useCurrentFrame} from 'remotion';
import {EASE_OUT, useTheme} from './theme';
import {useLayout} from './layout';
import {useVideo} from './context';
import {norm} from './timeline';

export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
export const easeOut = Easing.bezier(...EASE_OUT);
/** 0..1 progress between two scene-local frames, eased out. */
export const prog = (frame: number, a: number, len = 16) => interpolate(frame, [a, a + len], [0, 1], {...clamp, easing: easeOut});

/** Eased 0..1 progress starting at scene-local frame `delay`. Ease out, no overshoot. */
export const useEnter = (delay = 0, len = 16) => prog(useCurrentFrame(), delay, len);

/** Flat ground per look with at most one structural element. */
export const Background: React.FC = () => {
  const {C, bg, rgba, look} = useTheme();
  const {W, H, u, isLandscape, pad, caption} = useLayout();
  let layer: React.ReactNode = null;
  if (bg === 'dots') {
    layer = <AbsoluteFill style={{backgroundImage: `radial-gradient(circle, ${rgba(C.ink, 0.13)} ${1.6 * u}px, transparent ${1.8 * u}px)`, backgroundSize: `${48 * u}px ${48 * u}px`, backgroundPosition: `${pad}px ${pad}px`}} />;
  } else if (bg === 'lines') {
    const l = `1px`;
    layer = <AbsoluteFill style={{backgroundImage: `linear-gradient(${rgba('#FFFFFF', 0.035)} ${l}, transparent ${l}), linear-gradient(90deg, ${rgba('#FFFFFF', 0.035)} ${l}, transparent ${l})`, backgroundSize: `${72 * u}px ${72 * u}px`, backgroundPosition: `${pad}px ${pad}px`}} />;
  } else if (bg === 'grid') {
    const cols = isLandscape ? 12 : 6;
    const cw = (W - pad * 2) / cols;
    layer = (
      <AbsoluteFill>
        {Array.from({length: cols + 1}, (_, i) => (
          <div key={i} style={{position: 'absolute', left: pad + i * cw - 0.5, top: 0, width: 1, height: H, background: rgba(C.ink, 0.1)}} />
        ))}
        {Array.from({length: Math.floor(H / (120 * u)) + 1}, (_, i) => (
          <div key={i} style={{position: 'absolute', top: i * 120 * u, left: 0, height: 1, width: W, background: rgba(C.ink, 0.06)}} />
        ))}
      </AbsoluteFill>
    );
  } else if (look === 'cobalt') {
    // A flat darker field under the captions.
    const top = H - caption.bottom - caption.size * 2.4 - 6 * u;
    layer = <div style={{position: 'absolute', left: 0, right: 0, top, bottom: 0, background: C.wash}} />;
  }
  return (
    <AbsoluteFill style={{background: C.ground}}>
      {layer}
    </AbsoluteFill>
  );
};

/** Quick and clean: a short rise-in and a short fade-out. Pairs with the scene overlap. */
export const SceneShell: React.FC<{dur: number; children: React.ReactNode; isLast?: boolean}> = ({dur, children, isLast}) => {
  const frame = useCurrentFrame();
  const {u} = useLayout();
  const inP = prog(frame, 0, 10);
  const outP = isLast ? 0 : interpolate(frame, [dur - 8, dur], [0, 1], {...clamp, easing: Easing.in(Easing.quad)});
  return <AbsoluteFill style={{opacity: inP * (1 - outP), translate: `0px ${(1 - inP) * 16 * u}px`}}>{children}</AbsoluteFill>;
};

/**
 * The content box for a scene, between the kicker and the captions. Content is left aligned in a column as wide as
 * the box; `v` places it vertically. Content that is taller than the box is scaled down to fit.
 */
export const Stage: React.FC<{children: React.ReactNode; v?: 'center' | 'start' | 'end'; style?: React.CSSProperties}> = ({children, v = 'center', style}) => {
  const {stage} = useLayout();
  const ref = useRef<HTMLDivElement>(null);
  const [k, setK] = useState(1);
  const fit = () => {
    const el = ref.current;
    if (!el) return;
    const next = Math.min(1, stage.h / (el.offsetHeight || 1));
    setK((prev) => (Math.abs(next - prev) > 0.005 ? next : prev));
  };
  useLayoutEffect(fit);
  // Text reflows when fonts finish loading; a still has no later frame to notice.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage.h]);
  const justify = v === 'center' ? 'center' : v === 'start' ? 'flex-start' : 'flex-end';
  return (
    <div style={{position: 'absolute', left: stage.x, top: stage.y, width: stage.w, height: stage.h, display: 'flex', flexDirection: 'column', justifyContent: justify}}>
      <div ref={ref} style={{width: stage.w, flex: 'none', transformOrigin: v === 'start' ? '0 0' : v === 'end' ? '0 100%' : '0 50%', scale: String(k), display: 'flex', flexDirection: 'column', ...style}}>
        {children}
      </div>
    </div>
  );
};

/** Quiet context line: small, sentence case, a short rule before it. Rendered only when a scene asks for one. */
export const Kicker: React.FC<{children: React.ReactNode}> = ({children}) => {
  const p = useEnter(4, 14);
  const {u, kickerTop, pad} = useLayout();
  const {C, mono} = useTheme();
  return (
    <div style={{position: 'absolute', top: kickerTop, left: pad, display: 'flex', alignItems: 'center', gap: 16 * u, opacity: p, translate: `0px ${(1 - p) * 8 * u}px`, fontFamily: mono, fontWeight: 500, fontSize: 24 * u, letterSpacing: 0.2 * u, color: C.soft}}>
      <span style={{width: 36 * u * p, height: 2 * u, background: C.soft}} />
      {children}
    </div>
  );
};

/** Raised-object frame (the paper figure): the only boxed thing in a scene. */
export const useRaised = (): React.CSSProperties => {
  const {u} = useLayout();
  const {C, r, raised} = useTheme();
  const base: React.CSSProperties = {background: C.raised, borderRadius: r * u};
  if (raised === 'shadow') return {...base, boxShadow: `0 ${26 * u}px ${60 * u}px rgba(${C.shadow}, 0.28), 0 ${3 * u}px ${10 * u}px rgba(${C.shadow}, 0.14)`};
  if (raised === 'ring') return {...base, boxShadow: `0 0 0 ${2 * u}px rgba(255, 255, 255, 0.14)`};
  return {...base, border: `${3 * u}px solid ${C.ink}`};
};

/* ---------- Captions: chunks, words land when spoken, the key word gets a block or the accent colour ---------- */
type CWord = {w: string; f: number; tint?: string};
type Chunk = {line: number; start: number; end: number; words: CWord[]};

export const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const {caption, u} = useLayout();
  const {lines, fps} = useVideo();
  const {C, sans, dw, solid, color, onColor, r, rgba} = useTheme();
  const chunks = useMemo(() => {
    const out: Chunk[] = [];
    lines.forEach((l, li) => {
      let cur: CWord[] = [];
      const flush = () => {
        if (cur.length) out.push({line: li, start: cur[0].f, end: 0, words: cur});
        cur = [];
      };
      const hl = Object.entries(l.highlight ?? {});
      (l.words ?? []).forEach((w) => {
        const hit = hl.find(([k]) => norm(k) === norm(w.w));
        cur.push({w: w.w, f: l.from + Math.round((w.s ?? 0) * fps), tint: hit ? hit[1] || 'accent' : undefined});
        if (/[,.?!:;…]$/.test(w.w) || cur.length >= caption.maxWords) flush();
      });
      flush();
    });
    out.forEach((c, i) => {
      const next = out[i + 1];
      c.end = next && next.line === c.line ? next.start : lines[c.line].from + lines[c.line].dur + 8;
    });
    return out;
  }, [lines, fps, caption.maxWords]);
  const chunk = chunks.find((c) => frame >= c.start - 2 && frame < c.end);
  if (!chunk) return null;
  const exit = interpolate(frame, [chunk.end - 5, chunk.end], [1, 0], clamp);
  const s = caption.size;
  return (
    <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: caption.bottom}}>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: `${s * 0.1}px ${s * 0.28}px`, width: caption.width, opacity: exit}}>
        {chunk.words.map((w, i) => {
          const p = prog(frame, w.f - 3, 8);
          const text = w.w.replace(/[,:;]$/, '');
          const base: React.CSSProperties = {fontFamily: sans, fontWeight: Math.min(800, dw), fontSize: s, lineHeight: 1.15, letterSpacing: `${-0.02}em`};
          let inner: React.ReactNode = <span style={{...base, color: C.ink}}>{text}</span>;
          if (w.tint) {
            const c = color(w.tint);
            const bp = prog(frame, w.f, 8);
            const block = solid ? c : rgba(c, 0.2);
            inner = (
              <span style={{position: 'relative', display: 'inline-block'}}>
                <span style={{position: 'absolute', left: -s * 0.14, right: -s * 0.14, top: s * 0.02, bottom: -s * 0.02, background: block, borderRadius: Math.min(r * 0.45 * u, 14 * u), transformOrigin: '0 50%', transform: `scaleX(${bp})`}} />
                <span style={{...base, position: 'relative', color: solid ? (bp > 0.5 ? onColor(c) : C.ink) : c}}>{text}</span>
              </span>
            );
          }
          return (
            <span key={`${chunk.start}-${i}`} style={{display: 'inline-block', opacity: 0.12 + 0.88 * p, translate: `0px ${(1 - p) * s * 0.18}px`}}>
              {inner}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/** A thin accent bar along the top edge. */
export const ProgressLine: React.FC = () => {
  const frame = useCurrentFrame();
  const {total} = useVideo();
  const {W, u} = useLayout();
  const {C} = useTheme();
  return <div style={{position: 'absolute', top: 0, left: 0, height: 6 * u, width: interpolate(frame, [0, Math.max(1, total)], [0, W], clamp), background: C.accent}} />;
};
