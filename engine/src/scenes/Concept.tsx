import React from 'react';
import {useCurrentFrame} from 'remotion';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useEnter} from '../components';
import {Squiggle} from './primitives';
import type {Item, SceneOf} from '../types';

const ItemCard: React.FC<{item: Item; at: number; i: number; k: number; w: number}> = ({item, at, i, k, w}) => {
  const p = useEnter(at, 20);
  const {u, isLandscape, isPortrait} = useLayout();
  const {C, color} = useTheme();
  const c = color(item.tint, i);
  const label = fitFont(item.label ?? '', w - 90 * u, 80 * u, (isPortrait ? 48 : 40) * u * k, 22 * u, 0.5, 1.15);
  return (
    <div style={{width: w, display: 'flex', alignItems: 'stretch', gap: 24 * u, background: C.card, border: `${2 * u}px solid ${C.line}`, borderRadius: 20 * u, boxShadow: `0 ${14 * u}px ${34 * u}px ${C.shadow}`, padding: `${22 * k * u}px ${30 * u}px`, opacity: Math.min(1, p * 1.5), translate: `${isLandscape ? (1 - p) * 40 * u : 0}px ${isLandscape ? 0 : (1 - p) * 30 * u}px`}}>
      <div style={{width: 8 * u, borderRadius: 4 * u, background: c, flex: 'none'}} />
      <div style={{display: 'flex', flexDirection: 'column', gap: 6 * u, justifyContent: 'center', minWidth: 0}}>
        <div style={{fontFamily: SANS, fontWeight: 700, fontSize: label, color: C.ink, lineHeight: 1.15}}>{item.label}</div>
        {item.detail && <div style={{fontFamily: SANS, fontWeight: 500, fontSize: Math.min(label * 0.74, (isPortrait ? 34 : 30) * u * k), color: C.inkSoft, lineHeight: 1.25}}>{item.detail}</div>}
      </div>
    </div>
  );
};

export const Concept: React.FC<{scene: SceneOf<'concept'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C} = useTheme();
  const S = useScene();
  const items = (scene.items ?? []).slice(0, 5);
  const cues = S.cues(items.map((i) => i.at));
  const n = items.length;
  const k = n <= 3 ? 1 : n === 4 ? 0.9 : 0.8;
  const side = isLandscape && n > 0;
  const colW = side ? stage.w * 0.4 : Math.min(stage.w, isPortrait ? 900 * u : 1100 * u);
  const headSize = fitFont(scene.heading ?? '', colW, side ? stage.h * 0.5 : stage.h * 0.2, (isPortrait ? 112 : 110) * u, 44 * u, 0.4, 1.04);
  const bodySize = fitFont(scene.body ?? '', colW, side ? stage.h * 0.4 : stage.h * 0.2, 40 * u, 24 * u, 0.5, 1.35);
  const hp = useEnter(S.start - 4);
  const cardW = side ? stage.w * 0.5 : colW;

  const text = (
    <div style={{width: colW, display: 'flex', flexDirection: 'column', alignItems: side || n === 0 ? 'flex-start' : 'flex-start', gap: 26 * u, opacity: hp, translate: `0px ${(1 - hp) * 30 * u}px`}}>
      <div style={{fontFamily: SERIF, fontSize: n === 0 ? headSize * 1.1 : headSize, lineHeight: 1.04, letterSpacing: -headSize * 0.015, color: C.ink, textWrap: 'balance'}}>{scene.heading}</div>
      <Squiggle width={200} color={C.accent} progress={prog(frame, S.start + 8, 20)} />
      {scene.body && <div style={{fontFamily: SANS, fontWeight: 500, fontSize: n === 0 ? bodySize * 1.1 : bodySize, lineHeight: 1.35, color: C.inkSoft}}>{scene.body}</div>}
    </div>
  );
  return (
    <Stage style={{flexDirection: side ? 'row' : 'column', gap: (side ? 90 : 40) * u, justifyContent: 'center'}}>
      {text}
      {n > 0 && (
        <div style={{display: 'flex', flexDirection: 'column', gap: (n > 3 ? 16 : 24) * u, width: cardW}}>
          {items.map((it, i) => (
            <ItemCard key={i} item={it} at={cues[i]} i={i} k={k} w={cardW} />
          ))}
        </div>
      )}
    </Stage>
  );
};
