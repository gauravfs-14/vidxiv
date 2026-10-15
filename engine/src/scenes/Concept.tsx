import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Rule} from '../marks';
import {Reveal, stagger} from './primitives';
import type {Item, SceneOf} from '../types';

/** Heading and body on one side, a ruled definition list on the other. No boxes. */
export const Concept: React.FC<{scene: SceneOf<'concept'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, dw, color} = useTheme();
  const S = useScene();
  const items = (scene.items ?? []).slice(0, 5);
  const n = items.length;
  const cues = S.cues(items.map((i) => i.at));
  const side = isLandscape && n > 0;
  const tokens = (scene.heading ?? '').split(/\s+/).filter(Boolean);
  const colW = side ? stage.w * 0.4 : stage.w * (n === 0 && isLandscape ? 0.72 : 1);
  const head = fitFont(scene.heading ?? '', colW, side ? stage.h * 0.5 : stage.h * (n ? 0.24 : 0.5), (isLandscape ? 110 : isPortrait ? 118 : 90) * u * (n === 0 ? 1.25 : 1), 44 * u, 0.55, 1.0);
  const body = fitFont(scene.body ?? '', colW, side ? stage.h * 0.4 : stage.h * 0.2, 38 * u, 24 * u, 0.5, 1.35);
  const listW = side ? stage.w * 0.54 : stage.w;
  const k = n <= 3 ? 1 : n === 4 ? 0.9 : 0.8;

  const row = (it: Item, i: number) => {
    const at = cues[i];
    const c = it.tint ? color(it.tint) : C.soft;
    const label = fitFont(it.label ?? '', side ? listW * 0.5 : listW - 60 * u, 90 * u, (isPortrait ? 52 : 44) * u * k, 22 * u, 0.5, 1.1);
    return (
      <div key={i} style={{position: 'relative', paddingTop: (side ? 28 : 26) * k * u, paddingBottom: (side ? 28 : 26) * k * u}}>
        <div style={{position: 'absolute', left: 0, top: 0, width: '100%'}}>
          <Rule progress={prog(frame, at - 6, 18)} color={i === 0 ? C.ink : C.line} thickness={(i === 0 ? 2 : 1.5) * u} />
        </div>
        <Reveal at={at} rise={14} style={{display: 'grid', gridTemplateColumns: side ? '1.05fr 1fr' : '1fr', columnGap: 34 * u, rowGap: 8 * u, alignItems: 'baseline'}}>
          <div style={{display: 'flex', alignItems: 'baseline', gap: 18 * u, minWidth: 0}}>
            <span style={{width: 14 * u, height: 14 * u, background: c, flex: 'none', borderRadius: 3 * u, translate: `0px ${-label * 0.06}px`}} />
            <span style={{fontFamily: sans, fontWeight: dw, fontSize: label, color: C.ink, lineHeight: 1.1, letterSpacing: '-0.02em'}}>{it.label}</span>
          </div>
          {it.detail && <div style={{fontFamily: sans, fontWeight: 500, fontSize: Math.min(label * 0.7, (isPortrait ? 34 : 30) * u * k), color: C.soft, lineHeight: 1.3, paddingLeft: side ? 0 : 32 * u}}>{it.detail}</div>}
        </Reveal>
      </div>
    );
  };

  return (
    <Stage v='center' style={{flexDirection: side ? 'row' : 'column', justifyContent: side ? 'space-between' : undefined, alignItems: side ? 'center' : 'flex-start', gap: (side ? 80 : 44) * u}}>
      <div style={{width: colW, flex: 'none', display: 'flex', flexDirection: 'column', gap: 28 * u}}>
        <Headline tokens={tokens} times={stagger(S.start, tokens.length)} size={head} />
        {scene.body && (
          <Reveal at={S.start + 10} rise={14} style={{fontFamily: sans, fontWeight: 500, fontSize: body, lineHeight: 1.35, color: C.soft, maxWidth: colW * 0.95}}>
            {scene.body}
          </Reveal>
        )}
      </div>
      {n > 0 && (
        <div style={{width: listW, flex: 'none', display: 'flex', flexDirection: 'column'}}>
          {items.map(row)}
          <Rule progress={prog(frame, (cues[n - 1] ?? 0) + 6, 22)} color={C.line} thickness={1.5 * u} />
        </div>
      )}
    </Stage>
  );
};
