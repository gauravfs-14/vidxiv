import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Circle} from '@remotion/rough-notation';
import {SANS, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useEnter} from '../components';
import type {SceneOf} from '../types';

export const Bars: React.FC<{scene: SceneOf<'bars'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape} = useLayout();
  const {C, rgba} = useTheme();
  const S = useScene();
  const bars = (scene.bars ?? []).slice(0, 6);
  const n = bars.length;
  const cues = S.cues(bars.map((b) => b.at));
  const max = Math.max(1e-9, ...bars.map((b) => (Number.isFinite(b.value) ? b.value : 0)));
  const hp = useEnter(S.start - 4);
  const width = Math.min(stage.w, 1500 * u);
  const headH = scene.title ? 120 * u : 0;
  const rowH = Math.min(190 * u, (stage.h - headH - 30 * u) / Math.max(1, n));
  const labelSize = Math.min((isLandscape ? 38 : 46) * u, rowH * (isLandscape ? 0.34 : 0.3));
  const trackH = Math.min((isLandscape ? 58 : 76) * u, rowH * (isLandscape ? 0.5 : 0.36));
  const base = rgba(C.inkSoft, 0.5);
  return (
    <Stage style={{gap: 26 * u}}>
      {scene.title && (
        <div style={{textAlign: 'center', opacity: hp, translate: `0px ${(1 - hp) * 24 * u}px`}}>
          <div style={{fontFamily: SERIF, fontSize: fitFont(scene.title, width, 100 * u, 80 * u, 36 * u, 0.4, 1.05), color: C.ink, letterSpacing: -u}}>{scene.title}</div>
          {scene.unit && <div style={{fontFamily: SANS, fontWeight: 600, fontSize: 26 * u, letterSpacing: 3 * u, textTransform: 'uppercase', color: C.inkSoft, marginTop: 8 * u}}>{scene.unit}</div>}
        </div>
      )}
      <div style={{width, display: 'flex', flexDirection: 'column', gap: Math.max(10 * u, (rowH - (isLandscape ? trackH : trackH + labelSize * 1.3)) * 0.9)}}>
        {bars.map((b, i) => {
          const g = prog(frame, cues[i], 24);
          const v = Math.max(0, Number.isFinite(b.value) ? b.value : 0) / max;
          const col = b.highlight ? C.accent : base;
          const display = b.display ?? String(b.value);
          const label = (
            <div style={{fontFamily: SANS, fontWeight: b.highlight ? 700 : 600, fontSize: labelSize, color: b.highlight ? C.ink : C.inkSoft, lineHeight: 1.1, width: isLandscape ? '24%' : undefined, flex: 'none', opacity: Math.min(1, g * 3)}}>{b.label}</div>
          );
          const track = (
            <div style={{position: 'relative', flex: isLandscape ? 1 : 'none', width: isLandscape ? undefined : '100%', height: trackH, background: C.muted, borderRadius: trackH * 0.28}}>
              <div style={{position: 'absolute', left: 0, top: 0, bottom: 0, width: `${Math.max(0.01, v * g) * 86}%`, background: col, borderRadius: trackH * 0.28}} />
              <div style={{position: 'absolute', left: `calc(${Math.max(0.01, v * g) * 86}% + ${16 * u}px)`, top: 0, bottom: 0, display: 'flex', alignItems: 'center', opacity: Math.min(1, Math.max(0, (g - 0.5) * 2))}}>
                {b.highlight ? (
                  <Circle color={C.accent} strokeWidth={3.5 * u} iterations={1} seed={i + 2} padding={{left: 14 * u, right: 14 * u, top: 4 * u, bottom: 4 * u}} progress={prog(frame, cues[i] + 22, 16)}>
                    <span style={{fontFamily: SANS, fontWeight: 700, fontSize: labelSize * 1.1, color: C.accent, whiteSpace: 'nowrap'}}>{display}</span>
                  </Circle>
                ) : (
                  <span style={{fontFamily: SANS, fontWeight: 700, fontSize: labelSize * 1.1, color: C.inkSoft, whiteSpace: 'nowrap'}}>{display}</span>
                )}
              </div>
            </div>
          );
          return (
            <div key={i} style={{display: 'flex', flexDirection: isLandscape ? 'row' : 'column', alignItems: isLandscape ? 'center' : 'stretch', gap: (isLandscape ? 24 : 10) * u}}>
              {label}
              {track}
            </div>
          );
        })}
      </div>
    </Stage>
  );
};
