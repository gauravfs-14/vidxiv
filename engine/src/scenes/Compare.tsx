import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Rule, Underline} from '../marks';
import {Reveal, stagger} from './primitives';
import type {SceneOf, Side} from '../types';

const SideBlock: React.FC<{side: Side; at: number; pointCues: number[]; color: string; w: number; k: number}> = ({side, at, pointCues, color, w, k}) => {
  const {u} = useLayout();
  const {C, sans, dw, r} = useTheme();
  const pts = (side.points ?? []).slice(0, 4);
  const tokens = (side.title ?? '').split(/\s+/).filter(Boolean);
  const title = fitFont(side.title ?? '', w, 100 * u, 70 * u * k, 32 * u, 0.55, 1.0);
  return (
    <div style={{width: w, flex: 'none', display: 'flex', flexDirection: 'column', gap: 26 * k * u}}>
      <Headline tokens={tokens} times={stagger(at, tokens.length)} size={title} color={color} />
      <div style={{display: 'flex', flexDirection: 'column', gap: 22 * k * u}}>
        {pts.map((t, i) => (
          <Reveal key={i} at={pointCues[i]} rise={10} style={{display: 'flex', alignItems: 'baseline', gap: 20 * u}}>
            <span style={{width: 12 * u, height: 12 * u, background: color, flex: 'none', borderRadius: Math.min(r * 0.2 * u, 3 * u), translate: '0px -2px'}} />
            <span style={{fontFamily: sans, fontWeight: 500, fontSize: fitFont(t ?? '', w - 40 * u, 90 * u, 40 * u * k, 22 * u, 0.5, 1.25), color: C.ink, lineHeight: 1.25}}>{t}</span>
          </Reveal>
        ))}
      </div>
    </div>
  );
};

/** A split with a drawn divider: left side and right side, then the verdict as a big underlined statement. */
export const Compare: React.FC<{scene: SceneOf<'compare'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, dw, color} = useTheme();
  const S = useScene();
  const L = scene.left;
  const R = scene.right;
  const nl = Math.min(4, L?.points?.length ?? 0);
  const nr = Math.min(4, R?.points?.length ?? 0);
  const cues = S.cues(new Array(nl + nr).fill(null));
  const lc = cues.slice(0, nl);
  const rc = cues.slice(nl);
  const k = isLandscape ? 1.3 : isPortrait ? 1.1 : 0.8;
  const gap = (isLandscape ? 90 : 40) * u;
  const colW = isLandscape ? (stage.w - gap - 4 * u) / 2 : stage.w;
  const vAt = (cues[cues.length - 1] ?? S.start) + 16;
  const verdict = scene.verdict;
  const vSize = verdict ? fitFont(verdict, stage.w * 0.92, 130 * u, (isLandscape ? 74 : 70) * u, 30 * u, 0.5, 1.05) : 0;
  const vTokens = (verdict ?? '').split(/\s+/).filter(Boolean);
  return (
    <Stage v='center' style={{gap: (isLandscape ? 70 : 44) * u}}>
      <div style={{display: 'flex', flexDirection: isLandscape ? 'row' : 'column', gap, alignItems: 'stretch'}}>
        {L && <SideBlock side={L} at={S.start} pointCues={lc} color={color(L.tint)} w={colW} k={k} />}
        <Rule progress={prog(frame, S.start + 2, 26)} vertical={isLandscape} length="auto" thickness={3 * u} color={C.ink} style={isLandscape ? {alignSelf: 'stretch', height: 'auto'} : {width: '100%'}} />
        {R && <SideBlock side={R} at={(rc[0] ?? S.start + 20) - 12} pointCues={rc} color={color(R.tint)} w={colW} k={k} />}
      </div>
      {verdict && (
        <Reveal at={vAt} rise={14} len={14}>
          <Underline progress={prog(frame, vAt + 8, 18)} color={C.accent} thickness={Math.max(6 * u, vSize * 0.06)} offset={0.04}>
            <span style={{fontFamily: sans, fontWeight: dw, fontSize: vSize, color: C.ink, lineHeight: 1.05, letterSpacing: '-0.035em'}}>{verdict}</span>
          </Underline>
        </Reveal>
      )}
    </Stage>
  );
};
