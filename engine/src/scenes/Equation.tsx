import React, {useEffect, useLayoutEffect, useMemo, useRef, useState} from 'react';
import {continueRender, delayRender, useCurrentFrame} from 'remotion';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import {MONO, SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useCardStyle, useEnter} from '../components';
import type {SceneOf} from '../types';

const FONTS = ['1em KaTeX_Main', 'italic 1em KaTeX_Math', 'bold 1em KaTeX_Main', '1em KaTeX_Size1', '1em KaTeX_Size2', '1em KaTeX_Size3', '1em KaTeX_AMS', '1em KaTeX_Caligraphic', '1em KaTeX_Script'];

/** LaTeX via KaTeX, scaled down to fit `maxW`. Holds the render until KaTeX fonts have loaded. Falls back to monospace. */
const Formula: React.FC<{tex: string; size: number; maxW: number; color: string}> = ({tex, size, maxW, color}) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {displayMode: false, throwOnError: true, output: 'html', strict: 'ignore'});
    } catch {
      return null;
    }
  }, [tex]);
  const ref = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [handle] = useState(() => delayRender('katex fonts'));
  const measure = () => {
    const el = ref.current;
    if (el && el.scrollWidth > 0) setScale(Math.min(1, maxW / el.scrollWidth));
  };
  useEffect(() => {
    let alive = true;
    const finish = () => {
      if (!alive) return;
      requestAnimationFrame(() => {
        measure();
        continueRender(handle);
      });
    };
    const fonts = (document as Document & {fonts?: FontFaceSet}).fonts;
    if (!html || !fonts) return finish();
    Promise.all(FONTS.map((f) => fonts.load(f).catch(() => null)))
      .then(() => fonts.ready)
      .then(finish, finish);
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [html]);
  useLayoutEffect(measure, [maxW, html, size]);
  const style: React.CSSProperties = {fontSize: size, color, whiteSpace: 'nowrap', display: 'inline-block', transform: `scale(${scale})`, transformOrigin: 'center center', lineHeight: 1.3};
  return (
    <div style={{width: maxW, display: 'flex', justifyContent: 'center', height: size * 2.3 * scale}}>
      {html ? <div ref={ref} style={style} dangerouslySetInnerHTML={{__html: html}} /> : <div ref={ref} style={{...style, fontFamily: MONO}}>{tex}</div>}
    </div>
  );
};

export const Equation: React.FC<{scene: SceneOf<'equation'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape} = useLayout();
  const {C, color} = useTheme();
  const S = useScene();
  const card = useCardStyle();
  const terms = (scene.terms ?? []).slice(0, 4);
  const cues = S.cues(terms.map((t) => t.at));
  const cardW = Math.min(stage.w, 1500 * u);
  const innerW = cardW - 120 * u;
  const p = useEnter(S.start - 4);
  const reveal = prog(frame, S.start + 6, 30);
  const fsize = (isLandscape ? 92 : 80) * u;
  const capSize = scene.caption ? fitFont(scene.caption, innerW, 70 * u, 36 * u, 22 * u, 0.4, 1.2) : 0;
  const row = isLandscape && terms.length > 0;
  const termW = row ? (stage.w - (terms.length - 1) * 26 * u) / terms.length : Math.min(stage.w, 1000 * u);
  return (
    <Stage style={{gap: 32 * u}}>
      <div style={{...card, width: cardW, boxSizing: 'border-box', padding: `${50 * u}px ${60 * u}px ${scene.caption ? 36 * u : 50 * u}px`, opacity: p, translate: `0px ${(1 - p) * 40 * u}px`, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20 * u}}>
        <div style={{clipPath: `inset(-40% ${(1 - reveal) * 100}% -40% -5%)`, opacity: Math.min(1, reveal * 4)}}>
          <Formula tex={scene.formula ?? ''} size={fsize} maxW={innerW} color={C.ink} />
        </div>
        {scene.caption && <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: capSize, color: C.inkSoft, textAlign: 'center', lineHeight: 1.2}}>{scene.caption}</div>}
      </div>
      {terms.length > 0 && (
        <div style={{display: 'flex', flexDirection: row ? 'row' : 'column', gap: (row ? 26 : 16) * u, width: row ? stage.w : undefined}}>
          {terms.map((t, i) => (
            <Term key={i} label={t.label} detail={t.detail} at={cues[i]} c={color(t.tint, i)} w={termW} row={row} n={terms.length} />
          ))}
        </div>
      )}
    </Stage>
  );
};

const Term: React.FC<{label: string; detail?: string | null; at: number; c: string; w: number; row: boolean; n: number}> = ({label, detail, at, c, w, row, n}) => {
  const p = useEnter(at, 20);
  const {u} = useLayout();
  const {C} = useTheme();
  const k = n > 3 ? 0.85 : 1;
  return (
    <div style={{width: w, boxSizing: 'border-box', display: 'flex', alignItems: 'center', gap: 22 * u, padding: `${18 * k * u}px ${28 * u}px`, borderRadius: 18 * u, background: C.card, border: `${2 * u}px solid ${C.line}`, borderLeft: `${8 * u}px solid ${c}`, opacity: Math.min(1, p * 1.5), translate: `0px ${(1 - p) * 26 * u}px`}}>
      <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: fitFont(label ?? '', w * 0.4, 70 * u, 56 * u * k, 28 * u, 0.4, 1), color: c, lineHeight: 1, flex: 'none', maxWidth: '45%'}}>{label}</div>
      {detail && <div style={{fontFamily: SANS, fontWeight: 500, fontSize: Math.min(30 * u * k, fitFont(detail, w * 0.5, 80 * u, 30 * u * k, 18 * u, 0.5, 1.25)), color: C.inkSoft, lineHeight: 1.25}}>{detail}</div>}
    </div>
  );
};
