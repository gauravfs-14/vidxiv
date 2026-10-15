import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Rule} from '../marks';
import {Mono, Reveal} from './primitives';
import type {SceneOf} from '../types';

/** Closing statement, a full-width rule, then the takeaway on the left and the call to action on the right. */
export const Outro: React.FC<{scene: SceneOf<'outro'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, dw, r} = useTheme();
  const S = useScene();
  const w = isLandscape ? stage.w * 0.9 : stage.w;
  const tokens = (scene.headline ?? '').split(/\s+/).filter(Boolean);
  const times = S.align(tokens);
  const hSize = fitFont(scene.headline ?? '', w, stage.h * (isLandscape ? 0.56 : 0.46), (isLandscape ? 190 : isPortrait ? 150 : 120) * u, 48 * u, 0.55, 1.0);
  const row = isLandscape;
  const tW = row ? stage.w * 0.52 : stage.w;
  const tSize = scene.takeaway ? fitFont(scene.takeaway, tW, 150 * u, 40 * u, 24 * u, 0.5, 1.35) : 0;
  const ctaSize = scene.cta ? Math.min(40 * u, fitFont(scene.cta, 520 * u, 70 * u, 40 * u, 24 * u, 0.55, 1.1)) : 0;
  const last = times.length ? times[times.length - 1] : S.start;
  return (
    <Stage v='center' style={{gap: (isLandscape ? 56 : 48) * u}}>
      <Headline tokens={tokens} times={times} size={hSize} style={{maxWidth: w}} />
      <div style={{display: 'flex', flexDirection: 'column', gap: 34 * u}}>
        <Rule progress={prog(frame, last + 2, 26)} thickness={3 * u} color={C.ink} />
        <div style={{display: 'flex', flexDirection: row ? 'row' : 'column', justifyContent: 'space-between', alignItems: row ? 'flex-start' : 'flex-start', gap: 40 * u}}>
          {scene.takeaway && (
            <Reveal at={last + 8} rise={14} style={{width: tW, fontFamily: sans, fontWeight: 500, fontSize: tSize, color: C.soft, lineHeight: 1.35}}>
              {scene.takeaway}
            </Reveal>
          )}
          <div style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 22 * u}}>
            {scene.cta && (
              <Reveal at={last + 18} rise={14}>
                <div style={{background: C.accent, color: C.onAccent, fontFamily: sans, fontWeight: dw, fontSize: ctaSize, letterSpacing: '-0.01em', padding: `${22 * u}px ${44 * u}px`, borderRadius: r * 0.6 * u, whiteSpace: 'nowrap'}}>{scene.cta}</div>
              </Reveal>
            )}
            {scene.link && (
              <Reveal at={last + 26} rise={10}>
                <Mono size={30} color={C.ink}>{scene.link}</Mono>
              </Reveal>
            )}
          </div>
        </div>
      </div>
    </Stage>
  );
};
