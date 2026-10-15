import React from 'react';
import {useCurrentFrame} from 'remotion';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useEnter} from '../components';
import {Badge, FlowArrow} from './primitives';
import type {SceneOf} from '../types';

export const Steps: React.FC<{scene: SceneOf<'steps'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, color} = useTheme();
  const S = useScene();
  const steps = (scene.steps ?? []).slice(0, 6);
  const n = steps.length;
  const cues = S.cues(steps.map((s) => s.at));
  const heading = scene.heading;
  const headH = heading ? (isLandscape ? 110 : 120) * u : 0;
  const hp = useEnter(S.start - 4);
  const arrowGap = 60 * u;
  const cardW = isLandscape ? (stage.w - (n - 1) * arrowGap) / Math.max(n, 1) : Math.min(stage.w, isPortrait ? 880 * u : 960 * u);
  const avail = stage.h - headH - 30 * u;
  const cardH = isLandscape ? Math.min(avail, 380 * u) : Math.min(190 * u, (avail - (n - 1) * 56 * u) / Math.max(n, 1));
  const arrowAt = (i: number) => Math.max(cues[i] + 4, Math.min(cues[i] + 10, (cues[i + 1] ?? 1e9) - 14));

  const card = (i: number) => {
    const s = steps[i];
    const c = color(s.tint, i);
    const p = useEnter(cues[i], 18);
    const label = fitFont(s.label ?? '', isLandscape ? cardW - 50 * u : cardW - 170 * u, cardH * 0.4, (isLandscape ? 50 : 44) * u, 20 * u, 0.5, 1.15);
    return (
      <div key={i} style={{width: cardW, height: isLandscape ? undefined : cardH, minHeight: isLandscape ? cardH * 0.8 : undefined, flex: 'none', boxSizing: 'border-box', background: C.card, border: `${2 * u}px solid ${C.line}`, borderTop: `${8 * u}px solid ${c}`, borderRadius: 22 * u, boxShadow: `0 ${16 * u}px ${36 * u}px ${C.shadow}`, padding: `${(isLandscape ? 34 : 24) * u}px ${28 * u}px`, display: 'flex', flexDirection: isLandscape ? 'column' : 'row', alignItems: isLandscape ? 'flex-start' : 'center', justifyContent: isLandscape ? 'flex-start' : 'flex-start', gap: (isLandscape ? 22 : 28) * u, opacity: Math.min(1, p * 1.5), translate: `0px ${(1 - p) * 36 * u}px`, scale: String(0.94 + 0.06 * p)}}>
        <Badge n={i + 1} color={c} size={isLandscape ? 60 : Math.min(72, (cardH / u) * 0.45)} />
        <div style={{display: 'flex', flexDirection: 'column', gap: 8 * u, minWidth: 0}}>
          <div style={{fontFamily: SANS, fontWeight: 700, fontSize: label, color: C.ink, lineHeight: 1.12}}>{s.label}</div>
          {s.detail && <div style={{fontFamily: SANS, fontWeight: 500, fontSize: Math.min(label * 0.68, (isLandscape ? 32 : 28) * u), color: C.inkSoft, lineHeight: 1.25}}>{s.detail}</div>}
        </div>
      </div>
    );
  };

  const flow: React.ReactNode[] = [];
  steps.forEach((_, i) => {
    flow.push(card(i));
    if (i < n - 1) flow.push(<FlowArrow key={`a${i}`} dir={isLandscape ? 'h' : 'v'} progress={prog(frame, arrowAt(i), 14)} color={color(steps[i].tint, i)} size={isLandscape ? 60 : 56} />);
  });

  return (
    <Stage style={{justifyContent: 'center', gap: 30 * u}}>
      {heading && (
        <div style={{fontFamily: SERIF, fontSize: fitFont(heading, stage.w * 0.9, headH, 84 * u, 40 * u, 0.4, 1.05), color: C.ink, textAlign: 'center', opacity: hp, translate: `0px ${(1 - hp) * 24 * u}px`, letterSpacing: -1 * u}}>{heading}</div>
      )}
      <div style={{display: 'flex', flexDirection: isLandscape ? 'row' : 'column', alignItems: 'center', justifyContent: 'center'}}>{flow}</div>
    </Stage>
  );
};
