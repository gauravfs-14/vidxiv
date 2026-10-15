import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {Underline} from '@remotion/rough-notation';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, clamp, easeOut, prog, useEnter} from '../components';
import {norm} from '../timeline';
import type {SceneOf} from '../types';

export const Quote: React.FC<{scene: SceneOf<'quote'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isPortrait} = useLayout();
  const {C, color} = useTheme();
  const S = useScene();
  const tokens = (scene.text ?? '').split(/\s+/).filter(Boolean);
  const times = S.align(tokens);
  const w = Math.min(stage.w * 0.9, 1400 * u);
  const size = fitFont(scene.text ?? '', w, stage.h * (scene.attribution ? 0.68 : 0.8), (isPortrait ? 112 : 104) * u, 36 * u, 0.4, 1.12);
  const hl = Object.entries(scene.text ? S.line.highlight ?? {} : {});
  const marks = new Map<number, string>();
  tokens.forEach((t, i) => {
    const hit = hl.find(([k]) => norm(k) === norm(t));
    if (hit) marks.set(i, color(hit[1] || 'accent'));
  });
  if (!marks.size) for (let i = Math.max(0, tokens.length - 2); i < tokens.length; i++) marks.set(i, C.accent);
  const last = times.length ? times[times.length - 1] : S.start;
  const mp = useEnter(S.start - 2);
  const ap = useEnter(last + 14);
  return (
    <Stage>
      <div style={{width: w, display: 'flex', flexDirection: 'column', gap: 40 * u}}>
        <div style={{fontFamily: SERIF, fontSize: size * 2.2, lineHeight: 0.6, height: size * 0.75, color: C.accent, opacity: mp * 0.9, translate: `0px ${(1 - mp) * 20 * u}px`, marginLeft: -size * 0.05}}>{'“'}</div>
        <div style={{fontFamily: SERIF, fontSize: size, lineHeight: 1.12, letterSpacing: -size * 0.015, color: C.ink, textWrap: 'pretty'}}>
          {tokens.map((t, i) => {
            const p = interpolate(frame, [times[i], times[i] + 12], [0, 1], {...clamp, easing: easeOut});
            const word = <span style={{display: 'inline-block', opacity: p, translate: `0px ${(1 - p) * size * 0.15}px`}}>{t}</span>;
            const c = marks.get(i);
            return (
              <React.Fragment key={i}>
                {c ? (
                  <Underline color={c} strokeWidth={Math.max(4, size * 0.05)} iterations={2} seed={i + 1} padding={{top: -size * 0.02}} progress={prog(frame, times[i] + 10, 16)}>
                    {word}
                  </Underline>
                ) : (
                  word
                )}{' '}
              </React.Fragment>
            );
          })}
        </div>
        {scene.attribution && (
          <div style={{display: 'flex', alignItems: 'center', gap: 22 * u, opacity: ap, translate: `0px ${(1 - ap) * 16 * u}px`}}>
            <div style={{width: 70 * u * ap, height: 3 * u, background: C.inkSoft}} />
            <div style={{fontFamily: SANS, fontWeight: 600, fontSize: Math.min(34 * u, fitFont(scene.attribution, w * 0.8, 90 * u, 34 * u, 20 * u, 0.55, 1.3)), letterSpacing: 3 * u, textTransform: 'uppercase', color: C.inkSoft}}>{scene.attribution}</div>
          </div>
        )}
      </div>
    </Stage>
  );
};
