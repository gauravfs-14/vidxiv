import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Underline} from '@remotion/rough-notation';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useEnter} from '../components';
import {Chip, Reveal} from './primitives';
import type {SceneOf} from '../types';

export const Outro: React.FC<{scene: SceneOf<'outro'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isPortrait} = useLayout();
  const {C} = useTheme();
  const S = useScene();
  const w = Math.min(stage.w * 0.94, 1500 * u);
  const hSize = fitFont(scene.headline ?? '', w, stage.h * 0.4, (isPortrait ? 150 : 140) * u, 48 * u, 0.4, 1.04);
  const tSize = scene.takeaway ? fitFont(scene.takeaway, w * 0.8, 160 * u, 46 * u, 26 * u, 0.5, 1.35) : 0;
  const cta = scene.cta;
  const hp = useEnter(S.start, 200);
  const cp = useEnter(S.start + 34, 16);
  const pulse = 1 + Math.sin(frame / 14) * 0.012;
  return (
    <Stage style={{gap: 40 * u}}>
      <div style={{opacity: hp, translate: `0px ${(1 - hp) * 40 * u}px`, textAlign: 'center'}}>
        <Underline color={C.accent} strokeWidth={Math.max(5, hSize * 0.045)} iterations={2} seed={8} padding={{top: -hSize * 0.04}} progress={prog(frame, S.start + 22, 22)}>
          <div style={{fontFamily: SERIF, fontSize: hSize, lineHeight: 1.05, letterSpacing: -hSize * 0.02, color: C.ink, maxWidth: w, whiteSpace: 'normal', textWrap: 'balance', paddingBottom: hSize * 0.06}}>{scene.headline}</div>
        </Underline>
      </div>
      {scene.takeaway && (
        <Reveal at={S.start + 16} rise={20} style={{fontFamily: SANS, fontWeight: 500, fontSize: tSize, color: C.inkSoft, textAlign: 'center', width: w * 0.8, lineHeight: 1.35}}>
          {scene.takeaway}
        </Reveal>
      )}
      {cta && (
        <div style={{opacity: Math.min(1, cp * 1.5), translate: `0px ${(1 - cp) * 30 * u}px`, scale: String((0.9 + 0.1 * cp) * pulse)}}>
          <div style={{display: 'inline-flex', alignItems: 'center', gap: 20 * u, background: C.accent, color: C.dark ? C.paper : '#FFFCF6', fontFamily: SANS, fontWeight: 700, fontSize: Math.min(46 * u, fitFont(cta, w * 0.7, 80 * u, 46 * u, 26 * u, 0.55, 1.1)), padding: `${24 * u}px ${52 * u}px`, borderRadius: 999, boxShadow: `0 ${16 * u}px ${36 * u}px ${C.shadow}`}}>
            {cta}
            <svg width={40 * u} height={24 * u} viewBox="0 0 40 24" fill="none" stroke="currentColor" strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round"><path d="M2 12 H36 M26 3 L37 12 L26 21" /></svg>
          </div>
        </div>
      )}
      {scene.link && (
        <Reveal at={S.start + 46} rise={16}>
          <Chip size={30} color={C.ink} fill={C.card} style={{maxWidth: w}}>{scene.link}</Chip>
        </Reveal>
      )}
    </Stage>
  );
};
