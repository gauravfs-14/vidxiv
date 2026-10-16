import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';

export type SheetProps = {columns: {label: string; landscape: string; portrait: string}[]};

/** Dev only: lays out stills (one column per look) for docs and review. Rendered by scripts/looks.mjs. */
export const ContactSheet: React.FC<SheetProps> = ({columns}) => (
  <AbsoluteFill style={{background: '#16171A', padding: 48, fontFamily: 'Geist, system-ui, sans-serif', color: '#E8EAED'}}>
    <div style={{display: 'flex', gap: 36, alignItems: 'flex-start'}}>
      {columns.map((c) => (
        <div key={c.label} style={{width: 560, display: 'flex', flexDirection: 'column', gap: 18}}>
          <div style={{fontSize: 26, fontWeight: 600, letterSpacing: '-0.01em'}}>{c.label}</div>
          <Img src={staticFile(c.landscape)} style={{width: 560, height: 315, borderRadius: 8, display: 'block'}} />
          <Img src={staticFile(c.portrait)} style={{width: 560, height: 996, borderRadius: 8, display: 'block'}} />
        </div>
      ))}
    </div>
  </AbsoluteFill>
);
