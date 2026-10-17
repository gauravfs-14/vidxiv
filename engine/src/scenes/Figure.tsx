import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {useTheme} from '../theme';
import {fitFont, useLayout} from '../layout';
import {useScene} from '../context';
import {Stage, clamp, paceLen, prog, useEnter} from '../components';
import {Leader} from '../marks';
import {FigureCard, fitBox, Mono, useFigure} from './primitives';
import type {SceneOf} from '../types';

const PAD = 14;

export const FigureScene: React.FC<{scene: SceneOf<'figure'>}> = ({scene}) => {
  const frame = useCurrentFrame();
  const {u, stage, isLandscape, isPortrait} = useLayout();
  const {C, sans, dw, r} = useTheme();
  const S = useScene();
  const {asset, aw, ah} = useFigure(scene.figure);
  const caption = scene.caption ?? asset?.caption ?? '';
  const m = /^\s*((?:Fig(?:ure)?\.?|Table)\s*\d+[a-z]?)\s*[:.]\s*(.+)$/i.exec(caption);
  const capLabel = m ? m[1] : '';
  const capText = m ? m[2] : caption;

  const maxW = isLandscape ? stage.w * 0.7 : stage.w;
  const stackCaption = !isLandscape;
  const capH = caption && stackCaption ? (isPortrait ? 150 : 110) * u : 0;
  const dims = fitBox(aw, ah, maxW - 2 * PAD * u, stage.h - capH - 2 * PAD * u - (stackCaption && caption ? 24 * u : 0));
  const outerW = dims.w + 2 * PAD * u;
  const outerH = dims.h + 2 * PAD * u;
  const leftW = isLandscape ? stage.w - outerW - 70 * u : 0;
  const capSize = caption ? fitFont(capText, isLandscape ? leftW : stage.w, isLandscape ? 220 * u : 90 * u, isLandscape ? 52 * u : 36 * u, 22 * u, 0.5, 1.2) : 0;
  const enter = useEnter(S.start - 4, 14);

  // Zoom: a rectangle draws around the region, then the image scales so the region fills the frame.
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
  const capBlockH = caption && isLandscape ? capSize * 1.2 * Math.ceil((capText.length * capSize * 0.5) / Math.max(1, leftW)) + 60 * u : 0;
  const zoomBox = z && (
    <div style={{position: 'absolute', left: `${z.x * 100}%`, top: `${z.y * 100}%`, width: `${z.w * 100}%`, height: `${z.h * 100}%`, opacity: 1 - zp}}>
      <svg width="100%" height="100%" style={{overflow: 'visible', display: 'block'}}>
        <rect x="0" y="0" width="100%" height="100%" rx={r * 0.3 * u} fill="none" stroke={C.accent} strokeWidth={(5 * u) / sc} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - interpolate(frame, [zAt, zAt + paceLen(14)], [0, 1], clamp)} />
      </svg>
    </div>
  );

  return (
    <Stage v='center' style={{flexDirection: isLandscape ? 'row-reverse' : 'column', alignItems: isLandscape ? 'stretch' : 'flex-start', gap: (isLandscape ? 70 : 24) * u}}>
      <div style={{opacity: enter, translate: `0px ${(1 - enter) * 24 * u}px`, position: 'relative', width: outerW, height: outerH, flex: 'none'}}>
        <FigureCard id={scene.figure} w={dims.w} h={dims.h} pad={PAD} wrapStyle={{transform: `translate(${tx * zp}px, ${ty * zp}px) scale(${sc})`}}>
          {zoomBox}
        </FigureCard>
        {callouts.map((c, i) => {
          const size = Math.min(34 * u, fitFont(c.text, outerW * 0.45, 100 * u, 34 * u, 22 * u, 0.5, 1.1));
          const natural = c.text.length * size * 0.52 + 40 * u;
          const nw = Math.min(natural, outerW * 0.6);
          const nh = size * 1.2 * Math.ceil(natural / nw) + 22 * u;
          const tX = Math.min(outerW - 14 * u, Math.max(14 * u, mapX(c.x)));
          const tY = Math.min(outerH - 14 * u, Math.max(14 * u, mapY(c.y)));
          const ox = c.x - 0.5;
          const oy = c.y - 0.5;
          const mag = Math.hypot(ox, oy) || 1;
          const dist = Math.min(outerW, outerH) * 0.2 + 40 * u;
          const outside = isLandscape && c.x < 0.5 && leftW > nw + 90 * u;
          const cxIn = Math.min(outerW - nw / 2 - 8 * u, Math.max(nw / 2 + 8 * u, tX + (ox / mag) * dist * 1.4));
          const cyIn = Math.min(outerH - nh / 2 - 8 * u, Math.max(nh / 2 + 8 * u, tY + (oy / mag) * dist * 0.9 - (Math.abs(oy) < 0.1 ? 60 * u : 0)));
          const cx = outside ? -nw / 2 - 40 * u : cxIn;
          const cy = outside ? Math.min(outerH - capBlockH - nh / 2 - 20 * u, Math.max(nh / 2, tY)) : cyIn;
          const dx = tX - cx;
          const dy = tY - cy;
          const d = Math.hypot(dx, dy) || 1;
          const edge = Math.min(nw / 2 / Math.max(Math.abs(dx / d), 1e-3), nh / 2 / Math.max(Math.abs(dy / d), 1e-3));
          const np = prog(frame, cCues[i], 12);
          return (
            <React.Fragment key={i}>
              <div style={{position: 'absolute', left: cx - nw / 2, top: cy - nh / 2, width: nw, height: nh, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: sans, fontWeight: 600, fontSize: size, color: C.onAccent, background: C.accent, borderRadius: r * 0.4 * u, textAlign: 'center', lineHeight: 1.2, padding: `0 ${14 * u}px`, boxSizing: 'border-box', opacity: np, translate: `0px ${(1 - np) * 10 * u}px`}}>
                {c.text}
              </div>
              {d > edge + 20 * u && <Leader x1={cx + (dx / d) * edge} y1={cy + (dy / d) * edge} x2={tX} y2={tY} progress={prog(frame, cCues[i] + 8, 14)} color={C.accent} width={3} />}
            </React.Fragment>
          );
        })}
      </div>
      {caption && (
        <div style={{flex: isLandscape ? 1 : undefined, display: 'flex', flexDirection: 'column', justifyContent: isLandscape ? 'flex-end' : undefined, gap: 12 * u, opacity: enter, width: isLandscape ? undefined : stage.w, minWidth: 0}}>
          {capLabel && <Mono size={isLandscape ? 28 : 24}>{capLabel}</Mono>}
          <div style={{fontFamily: sans, fontWeight: dw, fontSize: capSize, color: C.ink, lineHeight: 1.15, letterSpacing: '-0.025em', textWrap: 'balance'}}>{capText}</div>
        </div>
      )}
    </Stage>
  );
};
