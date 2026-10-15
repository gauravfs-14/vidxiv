import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Rule} from '../marks';
import {FigureCard, fitBox, Mono, Reveal, useFigure} from './primitives';
import type {SceneOf} from '../types';

/** Title card: an accent bar, a very large title, then a rule with authors on the left and the meta line on the right. */
export const Title: React.FC<{scene: SceneOf<'title'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, mono} = useTheme();
  const S = useScene();
  const {aw, ah} = useFigure(scene.figure);
  const hasFig = !!scene.figure;
  const side = hasFig && isLandscape;
  const w = side ? stage.w * 0.58 : stage.w * (isLandscape ? 0.86 : 1);
  const tokens = (scene.title ?? '').split(/\s+/).filter(Boolean);
  const times = S.align(tokens);
  const size = fitFont(scene.title ?? '', w, stage.h * (hasFig && !isLandscape ? 0.34 : 0.56), (isLandscape ? 200 : isPortrait ? 150 : 120) * u, 44 * u, 0.55, 1.0);
  const authorsSize = scene.authors ? fitFont(scene.authors, w * (isLandscape && !side ? 0.62 : 1), 120 * u, 34 * u, 20 * u, 0.5, 1.35) : 0;
  const fig = fitBox(aw, ah, side ? stage.w * 0.34 : stage.w * 0.6, side ? stage.h * 0.7 : stage.h * 0.26);
  const inline = isLandscape && !side;
  return (
    <Stage style={{flexDirection: side ? 'row' : 'column', alignItems: side ? 'center' : 'flex-start', justifyContent: side ? 'space-between' : undefined, gap: 44 * u}}>
      <div style={{width: w, display: 'flex', flexDirection: 'column', gap: 36 * u, flex: 'none'}}>
        <Rule progress={prog(frame, S.start - 2, 18)} length={110 * u} thickness={12 * u} color={C.accent} />
        <Headline tokens={tokens} times={times} size={size} />
        <div style={{display: 'flex', flexDirection: 'column', gap: 26 * u}}>
          <Rule progress={prog(frame, S.start + 14, 26)} thickness={Math.max(2, 2) * u} color={C.ink} />
          <div style={{display: 'flex', flexDirection: inline ? 'row' : 'column', justifyContent: 'space-between', alignItems: inline ? 'flex-start' : 'flex-start', gap: (inline ? 60 : 16) * u}}>
            {scene.authors && (
              <Reveal at={S.start + 20} rise={12} style={{fontFamily: sans, fontWeight: 500, fontSize: authorsSize, color: C.soft, lineHeight: 1.35, maxWidth: inline ? w * 0.66 : w}}>
                {scene.authors}
              </Reveal>
            )}
            {scene.meta && (
              <Reveal at={S.start + 26} rise={12}>
                <Mono size={28} color={C.ink} style={{whiteSpace: 'nowrap', fontFamily: mono}}>{scene.meta}</Mono>
              </Reveal>
            )}
          </div>
        </div>
      </div>
      {hasFig && (
        <Reveal at={S.start + 14} rise={30} style={{alignSelf: side ? 'center' : 'flex-start'}}>
          <FigureCard id={scene.figure} w={fig.w} h={fig.h} />
        </Reveal>
      )}
    </Stage>
  );
};
