import React from 'react';
import {useCurrentFrame} from 'remotion';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useCardStyle, useEnter} from '../components';
import {Chip, FigureCard, fitBox, Reveal, Squiggle, useFigure} from './primitives';
import type {SceneOf} from '../types';

export const Title: React.FC<{scene: SceneOf<'title'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C} = useTheme();
  const S = useScene();
  const card = useCardStyle();
  const {aw, ah} = useFigure(scene.figure);
  const hasFig = !!scene.figure;
  const side = hasFig && isLandscape;
  const cardW = side ? stage.w * 0.6 : Math.min(stage.w * (isLandscape ? 0.78 : 1), 1500 * u);
  const innerW = cardW - 150 * u;
  const titleSize = fitFont(scene.title ?? '', innerW, stage.h * (hasFig && !isLandscape ? 0.28 : 0.42), (isPortrait ? 124 : 150) * u, 44 * u, 0.41, 1.05);
  const authorsSize = fitFont(scene.authors ?? '', innerW, 130 * u, 34 * u, 20 * u, 0.5, 1.35);
  const p = useEnter(S.start - 6, 200);
  const fig = fitBox(aw, ah, side ? stage.w * 0.3 : stage.w * 0.6, side ? stage.h * 0.7 : stage.h * 0.24);

  return (
    <Stage style={{flexDirection: side ? 'row' : 'column', gap: 50 * u}}>
      <div style={{...card, width: cardW, padding: `${64 * u}px ${75 * u}px`, borderRadius: 14 * u, opacity: p, translate: `0px ${(1 - p) * 50 * u}px`, rotate: `${(1 - p) * -1.5}deg`, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 34 * u}}>
        {scene.meta && (
          <Reveal at={S.start + 4} rise={10}>
            <Chip size={26} color={C.accent} fill="transparent">{scene.meta}</Chip>
          </Reveal>
        )}
        <div style={{fontFamily: SERIF, fontSize: titleSize, lineHeight: 1.05, letterSpacing: -titleSize * 0.015, color: C.ink}}>{scene.title}</div>
        <Squiggle width={260} color={C.accent} progress={prog(frame, S.start + 14, 22)} />
        {scene.authors && (
          <Reveal at={S.start + 20} rise={14} style={{fontFamily: SANS, fontWeight: 500, fontSize: authorsSize, color: C.inkSoft, lineHeight: 1.35}}>
            {scene.authors}
          </Reveal>
        )}
      </div>
      {hasFig && (
        <Reveal at={S.start + 14} rise={40} style={{rotate: side ? '3deg' : '-2deg'}}>
          <FigureCard id={scene.figure} w={fig.w} h={fig.h} />
        </Reveal>
      )}
    </Stage>
  );
};
