import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Underline} from '@remotion/rough-notation';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useEnter} from '../components';
import type {SceneOf, Side} from '../types';

const SideCard: React.FC<{side: Side; at: number; pointCues: number[]; color: string; w: number; k: number}> = ({side, at, pointCues, color, w, k}) => {
  const {u} = useLayout();
  const {C} = useTheme();
  const p = useEnter(at, 20);
  const pts = (side.points ?? []).slice(0, 4);
  const title = fitFont(side.title ?? '', w - 90 * u, 90 * u, 68 * u * k, 32 * u, 0.4, 1.05);
  return (
    <div style={{width: w, boxSizing: 'border-box', flex: 1, position: 'relative', background: C.card, border: `${2 * u}px solid ${C.line}`, borderRadius: 26 * u, boxShadow: `0 ${24 * u}px ${56 * u}px ${C.shadow}`, padding: `${34 * k * u}px ${42 * u}px`, display: 'flex', flexDirection: 'column', gap: 22 * k * u, opacity: Math.min(1, p * 1.4), translate: `0px ${(1 - p) * 40 * u}px`, overflow: 'hidden'}}>
      <div style={{position: 'absolute', left: 0, right: 0, top: 0, height: 10 * u, background: color}} />
      <div style={{fontFamily: SERIF, fontSize: title, color, lineHeight: 1.05, letterSpacing: -u, marginTop: 6 * u}}>{side.title}</div>
      <div style={{display: 'flex', flexDirection: 'column', gap: 18 * k * u}}>
        {pts.map((t, i) => (
          <Point key={i} text={t} at={pointCues[i]} color={color} w={w - 84 * u} k={k} />
        ))}
      </div>
    </div>
  );
};

const Point: React.FC<{text: string; at: number; color: string; w: number; k: number}> = ({text, at, color, w, k}) => {
  const p = useEnter(at, 20);
  const {u} = useLayout();
  const {C} = useTheme();
  return (
    <div style={{display: 'flex', alignItems: 'flex-start', gap: 18 * u, opacity: Math.min(1, p * 1.5), translate: `${(1 - p) * -30 * u}px 0px`}}>
      <span style={{width: 14 * u, height: 14 * u, borderRadius: '50%', background: color, marginTop: 14 * k * u, flex: 'none'}} />
      <span style={{fontFamily: SANS, fontWeight: 500, fontSize: fitFont(text ?? '', w - 40 * u, 90 * u, 38 * u * k, 22 * u, 0.5, 1.25), color: C.ink, lineHeight: 1.25}}>{text}</span>
    </div>
  );
};

export const Compare: React.FC<{scene: SceneOf<'compare'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, color} = useTheme();
  const S = useScene();
  const L = scene.left;
  const R = scene.right;
  const nl = Math.min(4, L?.points?.length ?? 0);
  const nr = Math.min(4, R?.points?.length ?? 0);
  const cues = S.cues(new Array(nl + nr).fill(null));
  const lc = cues.slice(0, nl);
  const rc = cues.slice(nl);
  const k = isLandscape ? 1.1 : isPortrait ? 1.15 : 0.85;
  const gap = (isLandscape ? 60 : 36) * u;
  const cardW = isLandscape ? (stage.w - gap) / 2 : Math.min(stage.w, 980 * u);
  const vAt = (cues[cues.length - 1] ?? S.start) + 18;
  const vp = useEnter(vAt, 20);
  const vSize = scene.verdict ? fitFont(scene.verdict, stage.w * 0.9, 100 * u, 62 * u, 30 * u, 0.4, 1.1) : 0;
  return (
    <Stage style={{gap: 34 * u}}>
      <div style={{position: 'relative', display: 'flex', flexDirection: isLandscape ? 'row' : 'column', gap, alignItems: 'stretch', width: isLandscape ? stage.w : cardW}}>
        {L && <SideCard side={L} at={S.start} pointCues={lc} color={color(L.tint, 3)} w={cardW} k={k} />}
        {R && <SideCard side={R} at={(rc[0] ?? S.start + 20) - 12} pointCues={rc} color={color(R.tint, 0)} w={cardW} k={k} />}
        <div style={{position: 'absolute', left: '50%', top: '50%', translate: '-50% -50%', width: 76 * u, height: 76 * u, borderRadius: '50%', background: C.paper, border: `${2 * u}px solid ${C.line}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: SERIF, fontStyle: 'italic', fontSize: 38 * u, color: C.inkSoft, boxShadow: `0 ${6 * u}px ${16 * u}px ${C.shadow}`}}>vs</div>
      </div>
      {scene.verdict && (
        <div style={{opacity: Math.min(1, vp * 1.4), translate: `0px ${(1 - vp) * 20 * u}px`}}>
          <Underline color={C.accent} strokeWidth={6 * u} iterations={2} seed={6} progress={prog(frame, vAt + 10, 18)}>
            <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: vSize, color: C.ink, textAlign: 'center', whiteSpace: 'normal', maxWidth: stage.w * 0.9, padding: `0 ${10 * u}px`}}>{scene.verdict}</div>
          </Underline>
        </div>
      )}
    </Stage>
  );
};
