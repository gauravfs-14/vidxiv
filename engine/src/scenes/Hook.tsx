import React from 'react';
import {AbsoluteFill} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, prog, useEnter} from '../components';
import {Headline, Rule} from '../marks';
import {norm} from '../timeline';
import {FigureCard, fitBox, Reveal, useFigure} from './primitives';
import type {SceneOf} from '../types';
import {useCurrentFrame} from 'remotion';

/** A big question. With a figure, a flat colour field bleeds off the edge and the figure sits on it. */
export const Hook: React.FC<{scene: SceneOf<'hook'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, W, H, stage, pad, isLandscape, isPortrait, caption} = useLayout();
  const {C, look, sans, rgba} = useTheme();
  const S = useScene();
  const {aw, ah} = useFigure(scene.figure);
  const tokens = (scene.headline ?? '').split(/\s+/).filter(Boolean);
  const times = S.align(tokens);
  const hasFig = !!scene.figure;
  const side = hasFig && isLandscape;
  const circleIdx = scene.circle ? tokens.findIndex((t) => norm(t) === norm(scene.circle!) || norm(t).startsWith(norm(scene.circle!))) : -1;
  const last = times.length ? times[times.length - 1] : S.start;

  const panelX = W * 0.56;
  const textW = side ? panelX - pad - 70 * u : stage.w * (isLandscape ? 0.82 : 1);
  const subH = scene.sub ? 130 * u : 0;
  const textH = (hasFig && !isLandscape ? stage.h * 0.5 : stage.h * 0.92) - subH;
  const size = fitFont(scene.headline ?? '', textW, textH, (isLandscape ? 170 : isPortrait ? 150 : 120) * u, 44 * u, 0.55, 1.0);

  // Colour field behind the figure. It ends above the captions so caption contrast is never affected.
  const captionTop = H - caption.bottom - caption.size * 2.4;
  const figPad = 14 * u;
  const field = side
    ? {left: panelX, top: 0, width: W - panelX, height: captionTop - 24 * u}
    : {left: 0, top: stage.y + stage.h - stage.h * 0.4 - 50 * u, width: W, height: stage.h * 0.4 + 50 * u + 14 * u};
  const fig = side ? fitBox(aw, ah, field.width - 2 * 70 * u - 2 * figPad, field.height * 0.62) : fitBox(aw, ah, stage.w - 2 * figPad, stage.h * 0.4 - 2 * figPad);
  const fieldP = prog(frame, S.start - 6, 20);
  const fieldBg = look === 'night' ? C.wash : C.accent;
  const marks = circleIdx >= 0 ? {[circleIdx]: {kind: 'outline' as const, color: C.accent, at: times[circleIdx] + 8}} : {};

  return (
    <>
      {hasFig && (
        <AbsoluteFill>
          <div style={{position: 'absolute', ...field, background: fieldBg, boxShadow: look === 'night' ? `inset 0 0 0 ${2 * u}px ${C.line}` : undefined, transformOrigin: side ? '100% 50%' : '50% 100%', transform: side ? `scaleX(${fieldP})` : `scaleY(${fieldP})`}} />
        </AbsoluteFill>
      )}
      <Stage v={hasFig && !isLandscape ? 'start' : 'center'} style={{alignItems: 'flex-start'}}>
        <div style={{width: textW, flex: 'none', display: 'flex', flexDirection: 'column'}}>
          <Headline tokens={tokens} times={times} size={size} marks={marks} />
          {scene.sub && (
            <Reveal at={last + 6} rise={12} style={{marginTop: 40 * u, display: 'flex', flexDirection: 'column', gap: 22 * u}}>
              <Rule progress={prog(frame, last + 6, 20)} length={120 * u} thickness={Math.max(3, 4) * u} color={C.ink} />
              <div style={{fontFamily: sans, fontWeight: 500, fontSize: Math.min(42 * u, fitFont(scene.sub, textW, 110 * u, 44 * u, 24 * u, 0.5, 1.3)), color: C.soft, lineHeight: 1.3, maxWidth: textW * 0.9}}>{scene.sub}</div>
            </Reveal>
          )}
        </div>
      </Stage>
      {hasFig && (
        <div style={{position: 'absolute', left: field.left + (field.width - fig.w - 2 * figPad) / 2, top: field.top + (field.height - fig.h - 2 * figPad) / 2}}>
          <FigureEnter at={S.start + 10}>
            <FigureCard id={scene.figure} w={fig.w} h={fig.h} pad={14} />
          </FigureEnter>
        </div>
      )}
    </>
  );
};

const FigureEnter: React.FC<{at: number; children: React.ReactNode}> = ({at, children}) => {
  const p = useEnter(at, 20);
  const {u} = useLayout();
  return <div style={{opacity: p, translate: `0px ${(1 - p) * 30 * u}px`}}>{children}</div>;
};
