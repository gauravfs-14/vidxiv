import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {Underline} from '@remotion/rough-notation';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Split, Stage, clamp, useEnter} from '../components';
import type {SceneOf} from '../types';

/** Split "$1,200.5M" into prefix, number and suffix. Returns null when there is no number to count. */
export const parseValue = (v: string) => {
  const m = /^(\D*?)(\d[\d,]*(?:\.\d+)?|\.\d+)(.*)$/.exec((v ?? '').trim());
  if (!m) return null;
  const num = m[2];
  return {prefix: m[1], suffix: m[3], target: parseFloat(num.replace(/,/g, '')), decimals: (num.split('.')[1] ?? '').length, commas: num.includes(',')};
};
const fmt = (x: number, d: number, commas: boolean) => x.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: commas});

export const Stat: React.FC<{scene: SceneOf<'stat'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape} = useLayout();
  const {C} = useTheme();
  const S = useScene();
  const cmp = scene.compare;
  const parsed = parseValue(scene.value);
  const t0 = S.start + 6;
  const dur = 42;
  const t = interpolate(frame, [t0, t0 + dur], [0, 1], {...clamp, easing: (x) => 1 - Math.pow(1 - x, 4)});
  const enter = useEnter(S.start, 14);
  const colW = cmp && isLandscape ? stage.w * 0.55 : stage.w * 0.96;
  const size = fitFont(scene.value ?? '', colW, stage.h * (cmp && !isLandscape ? 0.34 : 0.5), 380 * u, 110 * u, 0.5, 0.95);
  const shown = parsed ? fmt(parsed.target * t, parsed.decimals, parsed.commas) : scene.value;
  const full = parsed ? fmt(parsed.target, parsed.decimals, parsed.commas) : scene.value;
  const smallSuffix = !!parsed && parsed.suffix.trim().length > 1;
  const labelSize = fitFont(scene.label ?? '', colW, 130 * u, 54 * u, 28 * u, 0.5, 1.2);
  const cmpAt = cmp ? S.cue(cmp.at, 0.5) : 0;
  const cp = useEnter(cmpAt, 20);
  const done = t0 + dur;

  const affix = (s: string, small: boolean) => <span style={{fontSize: small ? size * 0.42 : size, marginLeft: small ? size * 0.06 : 0, letterSpacing: small ? 0 : undefined}}>{s}</span>;
  const main = (
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 26 * u, opacity: Math.min(1, enter * 1.5), scale: String(0.92 + 0.08 * enter)}}>
      <Underline color={C.accent} strokeWidth={Math.max(6, size * 0.04)} iterations={2} seed={3} padding={{top: size * 0.0}} progress={interpolate(frame, [done + 2, done + 20], [0, 1], clamp)}>
        <div style={{fontFamily: SERIF, color: C.accent, lineHeight: 0.95, letterSpacing: -size * 0.02, whiteSpace: 'nowrap', display: 'inline-grid', fontSize: size, paddingBottom: size * 0.08}}>
          <span style={{gridArea: '1 / 1', visibility: 'hidden'}}>
            {parsed ? affix(parsed.prefix, false) : null}
            {parsed ? affix(full, false) : affix(full, false)}
            {parsed ? affix(parsed.suffix, smallSuffix) : null}
          </span>
          <span style={{gridArea: '1 / 1'}}>
            {parsed ? affix(parsed.prefix, false) : null}
            {affix(shown, false)}
            {parsed ? affix(parsed.suffix, smallSuffix) : null}
          </span>
        </div>
      </Underline>
      <div style={{fontFamily: SANS, fontWeight: 600, fontSize: labelSize, color: C.ink, textAlign: 'center', width: colW, lineHeight: 1.2, marginTop: 10 * u}}>{scene.label}</div>
      {scene.detail && <div style={{fontFamily: SANS, fontWeight: 500, fontSize: Math.min(labelSize * 0.62, 34 * u), color: C.inkSoft, textAlign: 'center', width: colW * 0.9, lineHeight: 1.3}}>{scene.detail}</div>}
    </div>
  );
  if (!cmp) return <Stage>{main}</Stage>;
  const cSize = Math.min(size * 0.5, fitFont(cmp.detail ?? '', stage.w * 0.3, 200 * u, 190 * u, 60 * u, 0.5, 1));
  return (
    <Stage>
      <Split gap={isLandscape ? 80 : 40}>
        {main}
        <div style={{display: 'flex', flexDirection: isLandscape ? 'row' : 'column', alignItems: 'center', gap: (isLandscape ? 70 : 28) * u, opacity: Math.min(1, cp * 1.4), translate: `0px ${(1 - cp) * 30 * u}px`}}>
          <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: 64 * u, color: C.inkSoft}}>vs</div>
          <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 * u}}>
            <div style={{fontFamily: SERIF, fontSize: cSize, color: C.inkSoft, lineHeight: 1, whiteSpace: 'nowrap'}}>{cmp.detail}</div>
            <div style={{fontFamily: SANS, fontWeight: 600, fontSize: Math.min(36 * u, labelSize * 0.7), color: C.inkSoft, textAlign: 'center', maxWidth: stage.w * (isLandscape ? 0.26 : 0.8)}}>{cmp.label}</div>
          </div>
        </div>
      </Split>
    </Stage>
  );
};
