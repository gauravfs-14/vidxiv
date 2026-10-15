import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Rule, WordMark} from '../marks';
import {Mono, Reveal} from './primitives';
import {norm} from '../timeline';
import type {SceneOf} from '../types';

/** The quote in large sans type beside a thick accent rule. Highlighted words get a block that wipes in on the word. */
export const Quote: React.FC<{scene: SceneOf<'quote'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, color} = useTheme();
  const S = useScene();
  const tokens = (scene.text ?? '').split(/\s+/).filter(Boolean);
  const times = S.align(tokens);
  const bar = 12 * u;
  const w = Math.min(stage.w * (isLandscape ? 0.86 : 1), 1500 * u) - bar - 40 * u;
  const size = fitFont(scene.text ?? '', w, stage.h * (scene.attribution ? 0.72 : 0.86), (isLandscape ? 96 : isPortrait ? 92 : 72) * u, 36 * u, 0.5, 1.08);
  const hl = Object.entries(S.line.highlight ?? {});
  const marks: Record<number, WordMark> = {};
  tokens.forEach((t, i) => {
    const hit = hl.find(([k]) => norm(k) === norm(t));
    if (hit) marks[i] = {kind: 'marker', color: color(hit[1] || 'accent'), at: times[i] + 6};
  });
  if (!Object.keys(marks).length && tokens.length) marks[tokens.length - 1] = {kind: 'marker', color: C.accent, at: (times[tokens.length - 1] ?? 0) + 6};
  const last = times.length ? times[times.length - 1] : S.start;
  return (
    <Stage v='center'>
      <div style={{display: 'flex', gap: 40 * u, alignItems: 'stretch'}}>
        <Rule progress={prog(frame, S.start - 2, 24)} vertical length="auto" thickness={bar} color={C.accent} style={{height: 'auto', alignSelf: 'stretch'}} />
        <div style={{width: w, display: 'flex', flexDirection: 'column', gap: 40 * u}}>
          <Headline tokens={tokens} times={times} size={size} weight={600} marks={marks} lineHeight={1.12} tracking={-0.03} rise={0.6} />
          {scene.attribution && (
            <Reveal at={last + 10} rise={10}>
              <Mono size={30} color={C.soft}>{scene.attribution}</Mono>
            </Reveal>
          )}
        </div>
      </div>
    </Stage>
  );
};
