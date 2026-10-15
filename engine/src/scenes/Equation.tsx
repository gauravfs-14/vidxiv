import React, {useEffect, useLayoutEffect, useMemo, useRef, useState} from 'react';
import {continueRender, delayRender, useCurrentFrame} from 'remotion';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Rule} from '../marks';
import {Reveal} from './primitives';
import type {SceneOf} from '../types';

const FONTS = ['1em KaTeX_Main', 'italic 1em KaTeX_Math', 'bold 1em KaTeX_Main', '1em KaTeX_Size1', '1em KaTeX_Size2', '1em KaTeX_Size3', '1em KaTeX_AMS', '1em KaTeX_Caligraphic', '1em KaTeX_Script'];

/** LaTeX via KaTeX, left aligned and scaled down to fit `maxW`. Holds the render until KaTeX fonts have loaded. Falls back to monospace. */
const Formula: React.FC<{tex: string; size: number; maxW: number; color: string}> = ({tex, size, maxW, color}) => {
  const {mono} = useTheme();
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
  const style: React.CSSProperties = {fontSize: size, color, whiteSpace: 'nowrap', display: 'inline-block', transform: `scale(${scale})`, transformOrigin: 'left center', lineHeight: 1.3};
  return (
    <div style={{width: maxW, display: 'flex', justifyContent: 'flex-start', height: size * 2.3 * scale}}>
      {html ? <div ref={ref} style={style} dangerouslySetInnerHTML={{__html: html}} /> : <div ref={ref} style={{...style, fontFamily: mono}}>{tex}</div>}
    </div>
  );
};

/** The formula set large between rules, then the caption and a ruled list of terms. */
export const Equation: React.FC<{scene: SceneOf<'equation'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, mono, dw, color} = useTheme();
  const S = useScene();
  const terms = (scene.terms ?? []).slice(0, 4);
  const cues = S.cues(terms.map((t) => t.at));
  const reveal = prog(frame, S.start + 6, 30);
  const fsize = (isLandscape ? 100 : isPortrait ? 84 : 76) * u;
  const split = isLandscape && terms.length > 0;
  const capW = split ? stage.w * 0.34 : stage.w;
  const listW = split ? stage.w * 0.6 : stage.w;
  const capSize = scene.caption ? fitFont(scene.caption, capW, 160 * u, (isLandscape ? 52 : 44) * u, 24 * u, 0.5, 1.12) : 0;
  const k = terms.length > 3 ? 0.88 : 1;
  const cap = scene.caption && (
    <Reveal at={S.start + 4} rise={12} style={{width: capW, fontFamily: sans, fontWeight: dw, fontSize: capSize, lineHeight: 1.1, letterSpacing: '-0.03em', color: C.ink}}>
      {scene.caption}
    </Reveal>
  );
  const list = terms.length > 0 && (
    <div style={{width: listW, display: 'flex', flexDirection: 'column'}}>
      {terms.map((t, i) => {
        const c = t.tint ? color(t.tint) : C.ink;
        return (
          <div key={i} style={{position: 'relative', paddingTop: 20 * k * u, paddingBottom: 20 * k * u}}>
            <div style={{position: 'absolute', top: 0, left: 0, width: '100%'}}>
              <Rule progress={prog(frame, cues[i] - 6, 16)} thickness={1.5 * u} color={C.line} />
            </div>
            <Reveal at={cues[i]} rise={12} style={{display: 'flex', alignItems: 'baseline', gap: 28 * u}}>
              <span style={{fontFamily: mono, fontWeight: 600, fontSize: 46 * k * u, color: c, minWidth: 96 * u, lineHeight: 1}}>{t.label}</span>
              <span style={{fontFamily: sans, fontWeight: 500, fontSize: Math.min(34 * k * u, fitFont(t.detail ?? '', listW - 140 * u, 90 * u, 34 * k * u, 20 * u, 0.5, 1.25)), color: C.ink, lineHeight: 1.25}}>{t.detail}</span>
            </Reveal>
          </div>
        );
      })}
    </div>
  );
  return (
    <Stage v='center' style={{gap: (isLandscape ? 36 : 40) * u}}>
      <div style={{display: 'flex', flexDirection: 'column', gap: 26 * u}}>
        <Rule progress={prog(frame, S.start - 2, 24)} thickness={3 * u} color={C.ink} />
        <div style={{clipPath: `inset(-40% ${(1 - reveal) * 100}% -40% -5%)`, opacity: Math.min(1, reveal * 4)}}>
          <Formula tex={scene.formula ?? ''} size={fsize} maxW={stage.w} color={C.ink} />
        </div>
      </div>
      {split ? (
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start'}}>
          {cap}
          {list}
        </div>
      ) : (
        <>
          {cap}
          {list}
        </>
      )}
    </Stage>
  );
};
