import React from 'react';
import {Composition} from 'remotion';
import {Video, normalizeProps} from './Video';
import sample from './sample.json';
import type {VideoProps} from './types';
import {ContactSheet} from './ContactSheet';

export const FORMATS = {
  Vertical: {width: 1080, height: 1920},
  Landscape: {width: 1920, height: 1080},
  Square: {width: 1080, height: 1080},
} as const;

const defaults = sample as unknown as VideoProps;

export const RemotionRoot: React.FC = () => (
  <>
    {Object.entries(FORMATS).map(([id, f]) => (
      <Composition
        key={id}
        id={id}
        component={Video}
        width={f.width}
        height={f.height}
        fps={defaults.fps}
        durationInFrames={defaults.total}
        defaultProps={defaults}
        calculateMetadata={({props}) => {
          const p = normalizeProps(props);
          return {durationInFrames: p.total, fps: p.fps};
        }}
      />
    ))}
    <Composition id="ContactSheet" component={ContactSheet} width={2400} height={1450} fps={1} durationInFrames={1} defaultProps={{columns: []}} />
  </>
);
