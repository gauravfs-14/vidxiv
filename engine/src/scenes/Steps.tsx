import React from 'react';
import {useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog} from '../components';
import {Headline, Rule} from '../marks';
import {Reveal, stagger} from './primitives';
import type {SceneOf} from '../types';

/** A connected sequence. Landscape: nodes on a horizontal rule with the text alternating above and below. Portrait: a vertical rule. */
export const Steps: React.FC<{scene: SceneOf<'steps'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, dw, r} = useTheme();
  const S = useScene();
  const steps = (scene.steps ?? []).slice(0, 6);
  const n = steps.length;
  const cues = S.cues(steps.map((s) => s.at));
  const tokens = (scene.heading ?? '').split(/\s+/).filter(Boolean);
  const current = cues.reduce((acc, c, i) => (frame >= c ? i : acc), -1);
  const head = scene.heading ? fitFont(scene.heading, stage.w * 0.8, 130 * u, (isLandscape ? 88 : 96) * u, 44 * u, 0.55, 1.0) : 0;
  const node = 24 * u;
  const nodeBox = (i: number) => {
    const p = prog(frame, cues[i] - 2, 10);
    return <div style={{position: 'absolute', width: node, height: node, background: i === current ? C.accent : C.ink, borderRadius: Math.min(r * 0.3 * u, 8 * u), transform: `scale(${p})`}} />;
  };

  let body: React.ReactNode;
  if (isLandscape) {
    const cw = stage.w / Math.max(1, n);
    const tH = 400 * u;
    const mid = tH / 2;
    const textW = Math.min(cw * 2 - 30 * u, 640 * u);
    const label = Math.min(54 * u, fitFont(steps.reduce((a, s) => (s.label.length > a.length ? s.label : a), ''), textW, 70 * u, 54 * u, 0.5, 1.1));
    body = (
      <div style={{position: 'relative', width: stage.w, height: tH, flex: 'none'}}>
        <div style={{position: 'absolute', left: 0, top: mid - 1 * u}}>
          <Rule progress={prog(frame, S.start, 24)} length={Math.max(0, n - 1) * cw} thickness={2 * u} color={C.line} />
        </div>
        {steps.map((s, i) => {
          const x = i * cw;
          const above = i % 2 === 1;
          const seg = i > 0 ? cw * prog(frame, cues[i] - 14, 14) : 0;
          return (
            <React.Fragment key={i}>
              {i > 0 && <div style={{position: 'absolute', left: x - cw, top: mid - 1.5 * u, width: seg, height: 3 * u, background: C.ink}} />}
              <div style={{position: 'absolute', left: x, top: mid - node / 2}}>{nodeBox(i)}</div>
              <div style={{position: 'absolute', left: x + node / 2 - 1 * u, top: above ? mid - 40 * u : mid + node / 2, width: 2 * u, height: 40 * u - node / 2, background: C.line, transformOrigin: above ? '0 100%' : '0 0', transform: `scaleY(${prog(frame, cues[i], 10)})`}} />
              <Reveal at={cues[i] + 2} rise={above ? 12 : -12} style={{position: 'absolute', left: x, width: textW, ...(above ? {bottom: tH - mid + 46 * u} : {top: mid + 46 * u}), display: 'flex', flexDirection: 'column', gap: 8 * u}}>
                <div style={{fontFamily: sans, fontWeight: dw, fontSize: label, color: C.ink, lineHeight: 1.08, letterSpacing: '-0.025em'}}>{s.label}</div>
                {s.detail && <div style={{fontFamily: sans, fontWeight: 500, fontSize: Math.min(label * 0.6, 32 * u), color: C.soft, lineHeight: 1.28, maxWidth: cw * 1.7}}>{s.detail}</div>}
              </Reveal>
            </React.Fragment>
          );
        })}
      </div>
    );
  } else {
    const gap = (isPortrait ? 72 : 26) * u;
    const label = isPortrait ? 70 * u : 44 * u;
    const gut = 80 * u;
    body = (
      <div style={{display: 'flex', flexDirection: 'column', gap, width: stage.w}}>
        {steps.map((s, i) => (
          <div key={i} style={{position: 'relative', paddingLeft: gut}}>
            <div style={{position: 'absolute', left: 0, top: label * 0.28}}>{nodeBox(i)}</div>
            {i < n - 1 && <div style={{position: 'absolute', left: node / 2 - 1.5 * u, top: label * 0.28 + node, bottom: -(gap + label * 0.28), width: 3 * u, background: C.ink, transformOrigin: '0 0', transform: `scaleY(${prog(frame, cues[i + 1] - 14, 14)})`}} />}
            <Reveal at={cues[i] + 2} rise={14} style={{display: 'flex', flexDirection: 'column', gap: 6 * u}}>
              <div style={{fontFamily: sans, fontWeight: dw, fontSize: label, color: C.ink, lineHeight: 1.08, letterSpacing: '-0.025em'}}>{s.label}</div>
              {s.detail && <div style={{fontFamily: sans, fontWeight: 500, fontSize: (isPortrait ? 40 : 28) * u, color: C.soft, lineHeight: 1.28}}>{s.detail}</div>}
            </Reveal>
          </div>
        ))}
      </div>
    );
  }
  return (
    <Stage v='center' style={{gap: (isLandscape ? 56 : 54) * u}}>
      {scene.heading && <Headline tokens={tokens} times={stagger(S.start, tokens.length)} size={head} />}
      {body}
    </Stage>
  );
};
