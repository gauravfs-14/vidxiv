import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {Box} from '@remotion/rough-notation';
import {HAND, SERIF, useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, clamp, prog, useEnter} from '../components';
import {FigureCard, HandArrow, fitBox, useFigure} from './primitives';
import type {SceneOf} from '../types';

const PAD = 16;

export const FigureScene: React.FC<{scene: SceneOf<'figure'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isPortrait} = useLayout();
  const {C, color} = useTheme();
  const S = useScene();
  const {asset, aw, ah} = useFigure(scene.figure);
  const caption = scene.caption ?? asset?.caption ?? '';
  const capH = caption ? (isPortrait ? 130 : 96) * u : 0;
  const dims = fitBox(aw, ah, stage.w * 0.97 - 2 * PAD * u, stage.h - capH - 2 * PAD * u - 20 * u);
  const outerW = dims.w + 2 * PAD * u;
  const outerH = dims.h + 2 * PAD * u;
  const enter = useEnter(S.start - 4, 200);

  // Zoom: a hand box marks the region, then the image scales so the region fills the card.
  const z = scene.zoom;
  const zAt = z ? S.cue(z.at, 0.4) : 0;
  const zp = z ? prog(frame, zAt + 14, 28) : 0;
  const zs = z ? Math.max(1, Math.min(3.4, Math.min(1 / z.w, 1 / z.h) * 0.92)) : 1;
  const tx = z ? Math.min(0, Math.max(dims.w - zs * dims.w, dims.w / 2 - zs * (z.x + z.w / 2) * dims.w)) : 0;
  const ty = z ? Math.min(0, Math.max(dims.h - zs * dims.h, dims.h / 2 - zs * (z.y + z.h / 2) * dims.h)) : 0;
  const sc = 1 + (zs - 1) * zp;
  const mapX = (x: number) => PAD * u + tx * zp + sc * x * dims.w;
  const mapY = (y: number) => PAD * u + ty * zp + sc * y * dims.h;

  const callouts = (scene.callouts ?? []).slice(0, 3);
  const cCues = S.cues(callouts.map((c) => c.at));
  const kb = interpolate(frame, [0, Math.max(1, S.dur)], [1, 1.035], clamp);

  return (
    <Stage style={{gap: 22 * u}}>
      <div style={{opacity: Math.min(1, enter * 1.4), translate: `0px ${(1 - enter) * 40 * u}px`, scale: String(kb), position: 'relative', width: outerW, height: outerH, flex: 'none'}}>
        <FigureCard id={scene.figure} w={dims.w} h={dims.h} pad={PAD} wrapStyle={{transform: `translate(${tx * zp}px, ${ty * zp}px) scale(${sc})`}}>
          {z && (
            <div style={{position: 'absolute', left: `${z.x * 100}%`, top: `${z.y * 100}%`, width: `${z.w * 100}%`, height: `${z.h * 100}%`, opacity: 1 - zp}}>
              <Box color={C.accent} strokeWidth={4 * u / sc} iterations={1} seed={4} progress={interpolate(frame, [zAt, zAt + 14], [0, 1], clamp)}>
                <div style={{width: '100%', height: '100%'}} />
              </Box>
            </div>
          )}
        </FigureCard>
        {callouts.map((c, i) => {
          const size = Math.min(48 * u, fitFont(c.text, outerW * 0.45, 100 * u, 48 * u, 28 * u, 0.36, 1.1));
          const nw = c.text.length * size * 0.38 + 44 * u;
          const nh = size * 1.25 + 14 * u;
          const tX = Math.min(outerW - 14 * u, Math.max(14 * u, mapX(c.x)));
          const tY = Math.min(outerH - 14 * u, Math.max(14 * u, mapY(c.y)));
          const ox = c.x - 0.5;
          const oy = c.y - 0.5;
          const m = Math.hypot(ox, oy) || 1;
          const dist = Math.min(outerW, outerH) * 0.2 + 40 * u;
          const outside = (stage.w - outerW) / 2 > nw + 60 * u;
          const cxIn = Math.min(outerW - nw / 2 - 8 * u, Math.max(nw / 2 + 8 * u, tX + (ox / m) * dist * 1.4));
          const cyIn = Math.min(outerH - nh / 2 - 8 * u, Math.max(nh / 2 + 8 * u, tY + (oy / m) * dist * 0.9 - (Math.abs(oy) < 0.1 ? 60 * u : 0)));
          const cx = outside ? (c.x < 0.5 ? -nw / 2 - 36 * u : outerW + nw / 2 + 36 * u) : cxIn;
          const cy = outside ? Math.min(outerH - nh / 2, Math.max(nh / 2, tY)) : cyIn;
          const dx = tX - cx;
          const dy = tY - cy;
          const d = Math.hypot(dx, dy) || 1;
          const edge = Math.min(nw / 2 / Math.max(Math.abs(dx / d), 1e-3), nh / 2 / Math.max(Math.abs(dy / d), 1e-3)) + 6 * u;
          const col = color('accent');
          const np = useEnter(cCues[i], 18);
          return (
            <React.Fragment key={i}>
              <div style={{position: 'absolute', left: cx - nw / 2, top: cy - nh / 2, width: nw, height: nh, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: HAND, fontWeight: 600, fontSize: size, color: col, background: C.dark ? 'rgba(22,26,34,0.9)' : 'rgba(255,252,246,0.94)', border: `${2 * u}px solid ${C.line}`, borderRadius: 12 * u, whiteSpace: 'nowrap', rotate: `${i % 2 ? 2 : -2}deg`, opacity: np, scale: String(0.9 + 0.1 * np), boxShadow: `0 ${6 * u}px ${16 * u}px ${C.shadow}`}}>
                {c.text}
              </div>
              {d > edge + 24 * u && <HandArrow x1={cx + (dx / d) * edge} y1={cy + (dy / d) * edge} x2={tX - (dx / d) * 8 * u} y2={tY - (dy / d) * 8 * u} progress={prog(frame, cCues[i] + 8, 14)} color={col} width={4.5} bow={0.18} head={18} />}
            </React.Fragment>
          );
        })}
      </div>
      {caption && (
        <div style={{fontFamily: SERIF, fontStyle: 'italic', fontSize: fitFont(caption, stage.w * 0.9, capH - 10 * u, 38 * u, 22 * u, 0.4, 1.2), color: C.inkSoft, textAlign: 'center', width: stage.w * 0.9, lineHeight: 1.2, opacity: enter}}>{caption}</div>
      )}
    </Stage>
  );
};
