import {createContext, useContext, useMemo} from 'react';
import type {LineProps, PaceProps, VideoProps} from './types';
import {alignText, cueFrames, findWord, lineSpan} from './timeline';

export const STANDARD_PACE: PaceProps = {name: 'standard', speed: 1, words: null, pre: 10, overlap: 12};

export const EMPTY_PROPS: VideoProps = {fps: 30, total: 300, title: '', look: 'studio', pace: STANDARD_PACE, captions: true, progressBar: true, music: null, sfx: [], figures: {}, lines: [], scenes: []};

export const VideoContext = createContext<VideoProps>(EMPTY_PROPS);
export const useVideo = () => useContext(VideoContext);

type SceneInfo = {line: LineProps; index: number; from: number; dur: number; fps: number};
export const SceneContext = createContext<SceneInfo | null>(null);

/** The current scene's line plus timing helpers. All frames returned are scene-local. */
export const useScene = () => {
  const s = useContext(SceneContext);
  if (!s) throw new Error('useScene outside a scene');
  return useMemo(() => {
    const {start, end} = lineSpan(s.line, s.from);
    return {
      ...s, start, end,
      cues: (ats: (string | null | undefined)[]) => cueFrames(ats, s.line, s.from, s.fps),
      align: (tokens: string[]) => alignText(tokens, s.line, s.from, s.fps),
      /** Frame where a single word is spoken, or `fallback` fractions of the speech if it is not found. */
      cue: (at: string | null | undefined, fallback = 0.3) => {
        const i = findWord(s.line.words ?? [], at);
        return i >= 0 ? Math.round(start + s.line.words[i].s * s.fps) - 2 : Math.round(start + (end - start) * fallback);
      },
    };
  }, [s]);
};
