import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Outline, Rule} from '../marks';
import {Mono, Reveal, stagger} from './primitives';
import type {SceneOf} from '../types';

/** Flat bars on a baseline rule. No tracks: the length is the data. The highlighted bar takes the accent and an outlined value. */
export const Bars: React.FC<{scene: SceneOf<'bars'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, dw, r, rgba} = useTheme();
  const S = useScene();
  const bars = (scene.bars ?? []).slice(0, 6);
  const n = bars.length;
  const cues = S.cues(bars.map((b) => b.at));
  const max = Math.max(1e-9, ...bars.map((b) => (Number.isFinite(b.value) ? b.value : 0)));
  const tokens = (scene.title ?? '').split(/\s+/).filter(Boolean);
  const head = scene.title ? fitFont(scene.title, stage.w * 0.8, 120 * u, (isLandscape ? 92 : 100) * u, 44 * u, 0.55, 1.0) : 0;
  const headH = scene.title ? head * 1.1 + (scene.unit ? 50 * u : 0) + 40 * u : 0;
  const rowH = Math.min((isPortrait ? 240 : 176) * u, (stage.h - headH) / Math.max(1, n));
  const labelSize = Math.min((isLandscape ? 42 : 50) * u, rowH * (isLandscape ? 0.36 : 0.3));
  const barH = Math.min((isLandscape ? 76 : 90) * u, rowH * (isLandscape ? 0.56 : 0.4));
  const labelW = isLandscape ? stage.w * 0.22 : stage.w;
  const trackW = isLandscape ? stage.w - labelW - 30 * u : stage.w;
  const valueRoom = 190 * u;
  const gap = Math.max(10 * u, (rowH - (isLandscape ? barH : barH + labelSize * 1.2)) * 0.8);
  return (
    <Stage v='center' style={{gap: 34 * u}}>
      {scene.title && (
        <div style={{display: 'flex', flexDirection: 'column', gap: 14 * u}}>
          <Headline tokens={tokens} times={stagger(S.start, tokens.length)} size={head} />
          {scene.unit && (
            <Reveal at={S.start + 8} rise={10}>
              <Mono size={26}>{scene.unit}</Mono>
            </Reveal>
          )}
        </div>
      )}
      <div style={{position: 'relative', width: stage.w, display: 'flex', flexDirection: 'column', gap}}>
        {isLandscape && (
          <div style={{position: 'absolute', left: labelW + 30 * u - 1 * u, top: -10 * u, bottom: -10 * u}}>
            <Rule progress={prog(frame, S.start, 24)} vertical length="100%" thickness={2 * u} color={C.ink} />
          </div>
        )}
        {bars.map((b, i) => {
          const g = prog(frame, cues[i], 24);
          const v = Math.max(0, Number.isFinite(b.value) ? b.value : 0) / max;
          const col = b.highlight ? C.accent : rgba(C.ink, C.ink === '#FFFFFF' ? 0.42 : 0.22);
          const display = b.display ?? String(b.value);
          const w = Math.max(0.01, v) * (trackW - valueRoom) * g;
          const valueEl = (
            <span style={{fontFamily: sans, fontWeight: dw, fontSize: labelSize * 1.15, color: b.highlight ? C.ink : C.soft, whiteSpace: 'nowrap', letterSpacing: '-0.02em', fontVariantNumeric: 'tabular-nums'}}>{display}</span>
          );
          return (
            <div key={i} style={{display: 'flex', flexDirection: isLandscape ? 'row' : 'column', alignItems: isLandscape ? 'center' : 'flex-start', gap: (isLandscape ? 30 : 10) * u}}>
              <div style={{width: labelW, flex: 'none', fontFamily: sans, fontWeight: b.highlight ? dw : 500, fontSize: labelSize, color: b.highlight ? C.ink : C.soft, lineHeight: 1.1, letterSpacing: '-0.01em', opacity: Math.min(1, g * 3)}}>{b.label}</div>
              <div style={{display: 'flex', alignItems: 'center', width: trackW, height: barH}}>
                <div style={{width: w, height: barH, background: col, borderRadius: `0 ${Math.min(r * 0.35 * u, barH / 2)}px ${Math.min(r * 0.35 * u, barH / 2)}px 0`, flex: 'none'}} />
                <div style={{marginLeft: 18 * u, opacity: Math.min(1, Math.max(0, (g - 0.4) * 2.5))}}>
                  {b.highlight ? (
                    <Outline progress={prog(frame, cues[i] + 20, 16)} color={C.accent} stroke={3.5 * u} padX={14 * u} padY={4 * u}>
                      {valueEl}
                    </Outline>
                  ) : (
                    valueEl
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </Stage>
  );
};
