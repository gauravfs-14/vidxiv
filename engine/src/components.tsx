import React, {useId, useLayoutEffect, useMemo, useRef, useState} from 'react';
import {AbsoluteFill, Easing, Solid, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Highlight} from '@remotion/rough-notation';
import {paper} from '@remotion/effects/paper';
import {noise} from '@remotion/effects/noise';
import {lightLeak} from '@remotion/effects/light-leak';
import {EASE_OUT, HAND, SANS, useTheme} from './theme';
import {useLayout} from './layout';
import {useVideo} from './context';
import {norm} from './timeline';

export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
export const easeOut = Easing.bezier(...EASE_OUT);
/** 0..1 progress between two scene-local frames, eased out. */
export const prog = (frame: number, a: number, len = 16) => interpolate(frame, [a, a + len], [0, 1], {...clamp, easing: easeOut});

/** Calm spring (no bounce by default) starting at `delay` scene-local frames. Lower damping (12 to 18) adds life. */
export const useEnter = (delay = 0, damping = 200) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: {damping, stiffness: 120, mass: 0.9}});
};

export const PaperBackground: React.FC = () => {
  const frame = useCurrentFrame();
  const {W, H} = useLayout();
  const {C} = useTheme();
  return (
    <AbsoluteFill>
      <Solid
        width={W}
        height={H}
        color={C.paper}
        effects={[
          paper({amount: C.dark ? 0.25 : 0.32, colorFront: C.paperFront, colorBack: C.paperBack, contrast: 0.2, roughness: 0.35, fiber: 0.25, crumples: 0.12, folds: 0.15, foldCount: 3, drops: 0.08, scale: 0.8, seed: 14}),
          noise({amount: 0.035, seed: Math.floor(frame / 2)}),
        ]}
      />
      <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 42%, transparent 55%, ${C.dark ? 'rgba(0,0,0,0.35)' : 'rgba(150, 100, 50, 0.10)'} 100%)`}} />
    </AbsoluteFill>
  );
};

/** Hand-animation line boil. Render inside <defs>, reference with filter="url(#id)". */
export const Boil: React.FC<{id: string; scale?: number}> = ({id, scale = 3.2}) => {
  const frame = useCurrentFrame();
  return (
    <filter id={id} x="-10%" y="-10%" width="120%" height="120%">
      <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves={2} seed={Math.floor(frame / 4) % 7} result="n" />
      <feDisplacementMap in="SourceGraphic" in2="n" scale={scale} xChannelSelector="R" yChannelSelector="G" />
    </filter>
  );
};
export const useSafeId = (p = 'b') => `${p}${useId().replace(/[^a-zA-Z0-9]/g, '')}`;

/** Soft rise-in, soft lift-out. Pairs with the scene overlap for crossfades. */
export const SceneShell: React.FC<{dur: number; children: React.ReactNode; isLast?: boolean}> = ({dur, children, isLast}) => {
  const frame = useCurrentFrame();
  const {u} = useLayout();
  const inP = interpolate(frame, [0, 18], [0, 1], {...clamp, easing: easeOut});
  const outP = isLast ? 0 : interpolate(frame, [dur - 14, dur], [0, 1], {...clamp, easing: Easing.in(Easing.cubic)});
  return (
    <AbsoluteFill style={{opacity: inP * (1 - outP), translate: `0px ${((1 - inP) * 70 - outP * 50) * u}px`, scale: String(0.97 + 0.03 * inP), filter: `blur(${((1 - inP) * 6 + outP * 6) * u}px)`}}>
      {children}
    </AbsoluteFill>
  );
};

/** The content box for a scene, between the kicker and the captions. */
export const Stage: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => {
  const {stage} = useLayout();
  const ref = useRef<HTMLDivElement>(null);
  const [k, setK] = useState(1);
  // Safety net: if content is larger than the stage (long text, many items), scale it down to fit.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const next = Math.min(1, stage.h / (el.offsetHeight || 1), stage.w / (el.offsetWidth || 1));
    if (Math.abs(next - k) > 0.005) setK(next);
  });
  return (
    <div style={{position: 'absolute', left: stage.x, top: stage.y, width: stage.w, height: stage.h, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
      <div ref={ref} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 'none', scale: String(k), ...style}}>
        {children}
      </div>
    </div>
  );
};

/** Column in portrait and square, row in landscape. */
export const Split: React.FC<{children: React.ReactNode; gap?: number; style?: React.CSSProperties}> = ({children, gap = 50, style}) => {
  const {stack, u} = useLayout();
  return <div style={{display: 'flex', flexDirection: stack, alignItems: 'center', justifyContent: 'center', gap: gap * u, ...style}}>{children}</div>;
};

export const LightLeakOverlay: React.FC<{seed?: number}> = ({seed = 0}) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const {W, H} = useLayout();
  const {C} = useTheme();
  return (
    <AbsoluteFill style={{mixBlendMode: 'screen', opacity: C.dark ? 0.35 : 0.55}}>
      <Solid width={W} height={H} effects={[lightLeak({seed, hueShift: 0, progress: interpolate(frame, [0, durationInFrames - 1], [0, 1], clamp)})]} />
    </AbsoluteFill>
  );
};

export const Kicker: React.FC<{children: React.ReactNode; color?: string; delay?: number}> = ({children, color, delay = 4}) => {
  const p = useEnter(delay);
  const {u, kickerTop} = useLayout();
  const {C} = useTheme();
  return (
    <div style={{position: 'absolute', top: kickerTop, width: '100%', display: 'flex', justifyContent: 'center', opacity: p, translate: `0px ${(1 - p) * 16 * u}px`}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 14 * u, fontFamily: SANS, fontWeight: 600, fontSize: 28 * u, letterSpacing: 5 * u, color: C.inkSoft, textTransform: 'uppercase'}}>
        <span style={{width: 12 * u, height: 12 * u, borderRadius: 6 * u, background: color ?? C.accent}} />
        {children}
      </div>
    </div>
  );
};

/** Handwritten note. Seasoning, at most one or two per scene. */
export const Note: React.FC<{children: React.ReactNode; delay: number; style?: React.CSSProperties; color?: string; rotate?: number; size?: number}> = ({children, delay, style, color, rotate = -4, size = 50}) => {
  const p = useEnter(delay);
  const {u} = useLayout();
  const {C} = useTheme();
  return (
    <div style={{fontFamily: HAND, fontWeight: 600, fontSize: size * u, color: color ?? C.inkSoft, opacity: p, rotate: `${rotate}deg`, translate: `0px ${(1 - p) * 12 * u}px`, whiteSpace: 'nowrap', ...style}}>
      {children}
    </div>
  );
};

export const useCardStyle = (): React.CSSProperties => {
  const {u} = useLayout();
  const {C} = useTheme();
  return {background: C.card, border: `${2 * u}px solid ${C.line}`, borderRadius: 32 * u, boxShadow: `0 ${30 * u}px ${70 * u}px ${C.shadow}`};
};

/* ---------- Captions: chunks, words land when spoken, key words get a marker swipe ---------- */
type Chunk = {line: number; start: number; end: number; words: {w: string; f: number; tint?: string}[]};

export const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const {caption} = useLayout();
  const {lines, fps} = useVideo();
  const {C, marker} = useTheme();
  const chunks = useMemo(() => {
    const out: Chunk[] = [];
    lines.forEach((l, li) => {
      let cur: Chunk['words'] = [];
      const flush = () => {
        if (cur.length) out.push({line: li, start: cur[0].f, end: 0, words: cur});
        cur = [];
      };
      const hl = Object.entries(l.highlight ?? {});
      (l.words ?? []).forEach((w) => {
        const hit = hl.find(([k]) => norm(k) === norm(w.w));
        cur.push({w: w.w, f: l.from + Math.round((w.s ?? 0) * fps), tint: hit ? marker(hit[1]) : undefined});
        if (/[,.?!:;…]$/.test(w.w) || cur.length >= caption.maxWords) flush();
      });
      flush();
    });
    out.forEach((c, i) => {
      const next = out[i + 1];
      c.end = next && next.line === c.line ? next.start : lines[c.line].from + lines[c.line].dur + 8;
    });
    return out;
  }, [lines, fps, caption.maxWords, marker]);
  const chunk = chunks.find((c) => frame >= c.start - 2 && frame < c.end);
  if (!chunk) return null;
  const exit = interpolate(frame, [chunk.end - 5, chunk.end], [1, 0], clamp);
  const s = caption.size;
  return (
    <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: caption.bottom}}>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: `${s * 0.06}px ${s * 0.28}px`, width: caption.width, opacity: exit}}>
        {chunk.words.map((w, i) => {
          const p = interpolate(frame, [w.f - 3, w.f + 5], [0, 1], {...clamp, easing: easeOut});
          const word = <span style={{fontFamily: SANS, fontWeight: 700, fontSize: s, lineHeight: 1.15, letterSpacing: -s * 0.02, color: C.ink}}>{w.w.replace(/[,:;]$/, '')}</span>;
          return (
            <span key={`${chunk.start}-${i}`} style={{display: 'inline-block', opacity: 0.12 + 0.88 * p, translate: `0px ${(1 - p) * s * 0.2}px`}}>
              {w.tint ? (
                <Highlight color={w.tint} progress={interpolate(frame, [w.f, w.f + 9], [0, 1], clamp)} iterations={1} padding={{left: s * 0.08, right: s * 0.08, top: 2, bottom: 2}} seed={i + 3}>
                  {word}
                </Highlight>
              ) : (
                word
              )}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

export const ProgressLine: React.FC = () => {
  const frame = useCurrentFrame();
  const {total} = useVideo();
  const {W, u} = useLayout();
  const {C} = useTheme();
  return <div style={{position: 'absolute', top: 0, left: 0, height: 8 * u, width: interpolate(frame, [0, Math.max(1, total)], [0, W], clamp), background: C.accent, borderTopRightRadius: 4 * u, borderBottomRightRadius: 4 * u, opacity: 0.85}} />;
};
