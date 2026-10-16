import type {VideoProps} from './types';
import {cueFrames, findWord, lineSpan} from './timeline';

export type Sfx = {name: string; at: number; vol: number};

/**
 * Quiet sound effects derived from each scene's cue times (absolute frames). Only names listed in props.sfx are
 * returned, so a job without a sound file never references it. Keep volumes low: texture, not punchlines.
 */
export const sfxPlan = (props: VideoProps): Sfx[] => {
  const allowed = new Set(props.sfx ?? []);
  const out: Sfx[] = [];
  const add = (name: string, at: number, vol: number) => allowed.has(name) && out.push({name, at: Math.max(0, Math.round(at)), vol});
  props.lines.forEach((line, i) => {
    const sc = props.scenes[i];
    if (!sc || !line.scene) return;
    const s = line.scene;
    const base = sc.from;
    const {start, end} = lineSpan(line, base);
    const cues = (ats: (string | null | undefined)[]) => cueFrames(ats, line, base, props.fps).map((f) => base + f);
    if (i > 0 && s.type !== 'title') add('whoosh', base + 2, 0.15);
    switch (s.type) {
      case 'hook':
        add('pop', base + start + 4, 0.3);
        if (s.circle) add('pencil', base + start + (end - start) * 0.5, 0.3);
        break;
      case 'title':
        add('swell', base + start - 36, 0.3);
        add('ding', base + start + 8, 0.15);
        break;
      case 'concept':
        cues((s.items ?? []).map((x) => x.at)).forEach((f) => add('pop', f, 0.28));
        break;
      case 'steps':
        cues((s.steps ?? []).map((x) => x.at)).forEach((f) => {
          add('tick', f, 0.3);
          add('pencil', f + 10, 0.18);
        });
        break;
      case 'figure': {
        add('page-turn', base + start - 2, 0.25);
        cues((s.callouts ?? []).map((x) => x.at)).forEach((f) => add('pop', f, 0.28));
        if (s.zoom) {
          const k = findWord(line.words, s.zoom.at);
          add('whoosh', base + (k >= 0 ? Math.round(line.from - base + line.words[k].s * props.fps) : start + (end - start) * 0.4) + 12, 0.18);
        }
        break;
      }
      case 'stat':
        add('tick', base + start + 8, 0.3);
        add('ding', base + start + 52, 0.15);
        break;
      case 'bars':
        cues((s.bars ?? []).map((x) => x.at)).forEach((f) => add('tick', f, 0.3));
        break;
      case 'compare': {
        const n = Math.min(4, s.left?.points?.length ?? 0) + Math.min(4, s.right?.points?.length ?? 0);
        cues(new Array(n).fill(null)).forEach((f) => add('pop', f, 0.22));
        break;
      }
      case 'equation':
        add('page-turn', base + start, 0.2);
        add('tick', base + start + 14, 0.3);
        cues((s.terms ?? []).map((x) => x.at)).forEach((f) => add('pop', f, 0.25));
        break;
      case 'quote':
        add('page-turn', base + start, 0.2);
        add('pencil', base + end - 24, 0.25);
        break;
      case 'outro':
        add('pop', base + start + 4, 0.28);
        if (s.cta) add('ding', base + start + 36, 0.2);
        break;
    }
  });
  return out;
};
