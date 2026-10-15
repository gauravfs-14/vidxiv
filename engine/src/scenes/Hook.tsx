import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Circle} from '@remotion/rough-notation';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, clamp, easeOut, useEnter} from '../components';
import {norm} from '../timeline';
import {FigureCard, fitBox, Reveal, useFigure} from './primitives';
import type {SceneOf} from '../types';
import {interpolate} from 'remotion';

export const Hook: React.FC<{scene: SceneOf<'hook'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C} = useTheme();
  const S = useScene();
  const {aw, ah} = useFigure(scene.figure);
  const tokens = (scene.headline ?? '').split(/\s+/).filter(Boolean);
  const times = S.align(tokens);
  const hasFig = !!scene.figure;
  const side = hasFig && isLandscape;
  const textW = side ? stage.w * 0.56 : stage.w * 0.96;
  const textH = (hasFig && !isLandscape ? stage.h * 0.5 : stage.h * 0.8) - (scene.sub ? 90 * u : 0);
  const size = fitFont(scene.headline ?? '', textW, textH, (isPortrait ? 190 : 170) * u, 40 * u, 0.4, 1.02);
  const circleIdx = scene.circle ? tokens.findIndex((t) => norm(t) === norm(scene.circle!) || norm(t).startsWith(norm(scene.circle!))) : -1;
  const last = times.length ? times[times.length - 1] : S.start;
  const figMax = side ? {w: stage.w * 0.34, h: stage.h * 0.82} : {w: stage.w * 0.7, h: stage.h * 0.36};
  const fig = fitBox(aw, ah, figMax.w - 32 * u, figMax.h - 32 * u);
  const figP = useEnter(S.start + 8, 20);

  const headline = (
    <div style={{textAlign: side ? 'left' : 'center', width: textW, fontFamily: SERIF, fontSize: size, lineHeight: 1.02, letterSpacing: -size * 0.02, color: C.ink, textWrap: 'balance'}}>
      {tokens.map((t, i) => {
        const p = interpolate(frame, [times[i], times[i] + 12], [0, 1], {...clamp, easing: easeOut});
        const circled = i === circleIdx;
        const word = (
          <span style={{display: 'inline-block', opacity: p, translate: `0px ${(1 - p) * size * 0.22}px`, filter: `blur(${(1 - p) * 8 * u}px)`, color: circled ? C.accent : C.ink, fontStyle: circled ? 'italic' : 'normal'}}>{t}</span>
        );
        return (
          <React.Fragment key={i}>
            {circled ? (
              <Circle color={C.accent} strokeWidth={Math.max(3, size * 0.035)} iterations={2} seed={5} padding={{left: size * 0.16, right: size * 0.16, top: size * 0.04, bottom: size * 0.0}} progress={interpolate(frame, [times[i] + 10, times[i] + 28], [0, 1], clamp)}>
                {word}
              </Circle>
            ) : (
              word
            )}{' '}
          </React.Fragment>
        );
      })}
    </div>
  );
  const sub = scene.sub ? (
    <Reveal at={last + 8} rise={16} style={{fontFamily: SANS, fontWeight: 500, fontSize: Math.min(42 * u, fitFont(scene.sub, textW, 100 * u, 44 * u, 24 * u, 0.5, 1.3)), color: C.inkSoft, textAlign: side ? 'left' : 'center', width: textW * 0.92, marginTop: 36 * u, lineHeight: 1.3}}>
      {scene.sub}
    </Reveal>
  ) : null;

  return (
    <Stage style={{flexDirection: side ? 'row' : 'column', gap: (side ? 70 : 44) * u, justifyContent: 'center'}}>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: side ? 'flex-start' : 'center'}}>
        {headline}
        {sub}
      </div>
      {hasFig && (
        <div style={{opacity: Math.min(1, figP * 1.5), translate: `0px ${(1 - figP) * 50 * u + Math.sin(frame / 45) * 4 * u}px`, rotate: `${side ? 4 : -3}deg`}}>
          <FigureCard id={scene.figure} w={fig.w} h={fig.h} />
        </div>
      )}
    </Stage>
  );
};
