import React, {useMemo} from 'react';
import {AbsoluteFill, Sequence, interpolate, staticFile} from 'remotion';
import {Audio} from '@remotion/media';
import {ThemeContext, makeTheme, resolveLook} from './theme';
import {Background, Captions, ProgressLine, SceneShell, setPaceSpeed} from './components';
import {EMPTY_PROPS, STANDARD_PACE, SceneContext, VideoContext} from './context';
import {SceneView} from './scenes';
import {sfxPlan} from './sfx';
import type {LineProps, VideoProps} from './types';

/** Fill defaults so partial or malformed props never throw. */
export const normalizeProps = (p: Partial<VideoProps> | undefined): VideoProps => {
  const q = {...EMPTY_PROPS, ...(p ?? {})} as VideoProps;
  const fps = q.fps > 0 ? q.fps : 30;
  const pace = {...STANDARD_PACE, ...(q.pace ?? {})};
  pace.speed = pace.speed > 0 ? Math.min(4, pace.speed) : 1;
  const lines: LineProps[] = (Array.isArray(q.lines) ? q.lines : []).map((l, i) => ({...l, index: i, words: Array.isArray(l.words) ? l.words : [], from: l.from ?? 0, dur: Math.max(1, l.dur ?? 30), file: l.file ?? null}));
  const scenes = lines.map((l, i) => {
    const given = q.scenes?.[i];
    if (given && Number.isFinite(given.from) && given.dur > 0) return given;
    const from = i === 0 ? 0 : l.from - pace.pre;
    const next = lines[i + 1] ? lines[i + 1].from - pace.pre + pace.overlap : (q.total || l.from + l.dur + 90);
    return {from, dur: Math.max(20, next - from)};
  });
  const look = resolveLook(p?.look ?? p?.palette);
  return {...q, look, fps, pace, lines, scenes, figures: q.figures ?? {}, sfx: Array.isArray(q.sfx) ? q.sfx : [], total: q.total > 0 ? q.total : Math.max(60, ...lines.map((l) => l.from + l.dur + 60))};
};

export const Video: React.FC<Partial<VideoProps>> = (raw) => {
  const props = useMemo(() => normalizeProps(raw), [raw]);
  setPaceSpeed(props.pace.speed); // before any scene renders, so every prog() in this frame uses this video's speed
  const theme = useMemo(() => makeTheme(props.look), [props.look]);
  const sfx = useMemo(() => sfxPlan(props), [props]);
  const {lines, scenes, fps, total} = props;

  // Music ducks under speech (0.38 of the nominal level at 0.5) and rises between lines, fading out at the end.
  const musicVolume = (f: number) => {
    const dist = Math.min(1e9, ...lines.map((l) => (f < l.from ? l.from - f : f > l.from + l.dur ? f - l.from - l.dur : 0)));
    const v = props.music?.volume ?? 0.5;
    const duck = interpolate(dist, [0, 12], [v * 0.76, Math.min(1, v * 1.6)], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
    return duck * interpolate(f, [total - 50, total - 2], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  };
  return (
    <VideoContext.Provider value={props}>
      <ThemeContext.Provider value={theme}>
        <AbsoluteFill style={{backgroundColor: theme.C.ground}}>
          <Background />
          {lines.map((line, i) => (
            <Sequence key={i} name={`${i + 1} ${line.scene?.type ?? 'scene'}`} from={scenes[i].from} durationInFrames={scenes[i].dur} premountFor={fps}>
              <SceneContext.Provider value={{line, index: i, from: scenes[i].from, dur: scenes[i].dur, fps}}>
                <SceneShell dur={scenes[i].dur} isLast={i === lines.length - 1}>
                  <SceneView line={line} />
                </SceneShell>
              </SceneContext.Provider>
            </Sequence>
          ))}
          {props.captions && <Captions />}
          {props.progressBar && <ProgressLine />}

          {props.music?.file && <Audio name="Music" src={staticFile(props.music.file)} volume={musicVolume} loop />}
          {lines.map((l, i) =>
            l.file ? (
              <Sequence key={`vo${i}`} name={`VO ${i + 1}`} from={l.from} durationInFrames={l.dur + 6} premountFor={fps}>
                <Audio src={staticFile(l.file)} />
              </Sequence>
            ) : null,
          )}
          {sfx.map((s, i) => (
            <Sequence key={`sfx${i}`} name={`SFX ${s.name}`} from={s.at} durationInFrames={60} premountFor={fps}>
              <Audio src={staticFile(`audio/${s.name}.wav`)} volume={s.vol} />
            </Sequence>
          ))}
        </AbsoluteFill>
      </ThemeContext.Provider>
    </VideoContext.Provider>
  );
};
