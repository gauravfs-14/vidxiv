import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, clamp, paceLen, prog} from '../components';
import {Rule, Underline} from '../marks';
import {Reveal} from './primitives';
import type {SceneOf} from '../types';

/** Split "$1,200.5M" into prefix, number and suffix. Returns null when there is no number to count. */
export const parseValue = (v: string) => {
  const m = /^(\D*?)(\d[\d,]*(?:\.\d+)?|\.\d+)(.*)$/.exec((v ?? '').trim());
  if (!m) return null;
  const num = m[2];
  return {prefix: m[1], suffix: m[3], target: parseFloat(num.replace(/,/g, '')), decimals: (num.split('.')[1] ?? '').length, commas: num.includes(',')};
};
const fmt = (x: number, d: number, commas: boolean) => x.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: commas});

/** One huge number on a colour field, a rule, then the label and the comparison. */
export const Stat: React.FC<{scene: SceneOf<'stat'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, W, H, stage, isLandscape, caption} = useLayout();
  const {C, sans, mono, dw, dt, look, rgba} = useTheme();
  const S = useScene();
  const cmp = scene.compare;
  const parsed = parseValue(scene.value);
  const t0 = S.start + 4;
  const dur = paceLen(40);
  const t = interpolate(frame, [t0, t0 + dur], [0, 1], {...clamp, easing: (x) => 1 - Math.pow(1 - x, 4)});
  const size = fitFont(scene.value ?? '', stage.w * 0.9, stage.h * (isLandscape ? 0.62 : 0.4), (isLandscape ? 420 : 360) * u, 110 * u, 0.56, 0.9);
  const shown = parsed ? fmt(parsed.target * t, parsed.decimals, parsed.commas) : scene.value;
  const full = parsed ? fmt(parsed.target, parsed.decimals, parsed.commas) : scene.value;
  const smallSuffix = !!parsed && parsed.suffix.trim().length > 1;
  const colW = isLandscape ? stage.w * 0.46 : stage.w;
  const labelSize = fitFont(scene.label ?? '', colW, 130 * u, isLandscape ? 56 * u : 54 * u, 28 * u, 0.5, 1.15);
  const cmpAt = cmp ? S.cue(cmp.at, 0.5) : 0;
  const done = t0 + dur;
  const fieldP = prog(frame, S.start - 4, 22);
  // In cobalt the whole scene is a flat accent block (edge to edge, ending above the captions).
  const block = look === 'cobalt';
  const captionTop = H - caption.bottom - caption.size * 2.4;
  const ink = block ? C.onAccent : C.ink;
  const soft = block ? rgba(C.onAccent, 0.72) : C.soft;
  const numColor = block ? C.onAccent : C.accent;
  const ruleColor = block ? C.onAccent : C.ink;

  const affix = (s: string, small: boolean) => <span style={{fontSize: small ? size * 0.4 : size, marginLeft: small ? size * 0.06 : 0, letterSpacing: small ? 0 : undefined}}>{s}</span>;
  const number = (
    <Underline progress={prog(frame, done, 18)} color={numColor} thickness={Math.max(8 * u, size * 0.035)} offset={0.0} style={{alignSelf: 'flex-start'}}>
      <div style={{fontFamily: sans, fontWeight: dw, color: numColor, lineHeight: 0.92, letterSpacing: `${dt}em`, whiteSpace: 'nowrap', display: 'inline-grid', fontSize: size, paddingBottom: size * 0.1, fontVariantNumeric: 'tabular-nums'}}>
        <span style={{gridArea: '1 / 1', visibility: 'hidden'}}>
          {parsed ? affix(parsed.prefix, false) : null}
          {affix(full, false)}
          {parsed ? affix(parsed.suffix, smallSuffix) : null}
        </span>
        <span style={{gridArea: '1 / 1'}}>
          {parsed ? affix(parsed.prefix, false) : null}
          {affix(shown, false)}
          {parsed ? affix(parsed.suffix, smallSuffix) : null}
        </span>
      </div>
    </Underline>
  );

  return (
    <>
      {block && (
        <div style={{position: 'absolute', left: 0, top: 0, width: W, height: captionTop - 6 * u, background: C.accent, transformOrigin: '0 0', transform: `scaleY(${fieldP})`}} />
      )}
      <Stage v='center' style={{gap: (isLandscape ? 44 : 40) * u}}>
        <Reveal at={S.start} rise={30} len={20}>{number}</Reveal>
        <Rule progress={prog(frame, done - 6, 24)} thickness={3 * u} color={ruleColor} />
        <div style={{display: 'flex', flexDirection: isLandscape ? 'row' : 'column', justifyContent: 'space-between', gap: (isLandscape ? 60 : 34) * u}}>
          <Reveal at={S.start + 12} rise={14} style={{width: colW, display: 'flex', flexDirection: 'column', gap: 14 * u}}>
            <div style={{fontFamily: sans, fontWeight: dw, fontSize: labelSize, color: ink, lineHeight: 1.1, letterSpacing: '-0.025em'}}>{scene.label}</div>
            {scene.detail && <div style={{fontFamily: mono, fontWeight: 500, fontSize: Math.min(labelSize * 0.5, 28 * u), color: soft, lineHeight: 1.3}}>{scene.detail}</div>}
          </Reveal>
          {cmp && (
            <Reveal at={cmpAt} rise={14} style={{width: isLandscape ? (stage.w * 5) / 12 : stage.w, display: 'flex', flexDirection: 'column', gap: 8 * u}}>
              <div style={{fontFamily: sans, fontWeight: dw, fontSize: Math.min(size * 0.3, 110 * u), color: soft, lineHeight: 1, letterSpacing: `${dt}em`, fontVariantNumeric: 'tabular-nums'}}>{cmp.detail}</div>
              <div style={{fontFamily: sans, fontWeight: 500, fontSize: Math.min(32 * u, labelSize * 0.6), color: soft, lineHeight: 1.25}}>{cmp.label}</div>
            </Reveal>
          )}
        </div>
      </Stage>
    </>
  );
};
