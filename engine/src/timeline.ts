import type {LineProps, Word} from './types';

export const norm = (s: string) => (s ?? '').toLowerCase().replace(/[^a-z0-9']/g, '');
const stem = (s: string) => norm(s).replace(/'s?$/, '').replace(/(ing|ed|es|s)$/, '');

/**
 * Index of the spoken word that `at` refers to, or -1. Tries exact, then stemmed, then prefix matches,
 * preferring occurrences at or after `from` (so repeated words resolve to successive items).
 */
export const findWord = (words: Word[], at?: string | null, from = 0): number => {
  const toks = (at ?? '').split(/\s+/).map(norm).filter(Boolean);
  if (!toks.length || !words?.length) return -1;
  const tests: ((w: string, t: string) => boolean)[] = [
    (w, t) => norm(w) === t,
    (w, t) => stem(w) === stem(t) && stem(t).length > 1,
    (w, t) => t.length >= 4 && norm(w).length >= 4 && (norm(w).startsWith(t) || t.startsWith(norm(w))),
  ];
  for (const test of tests) {
    for (const start of [from, 0]) {
      for (let i = start; i < words.length; i++) {
        if (test(words[i].w, toks[0])) return i;
      }
    }
  }
  return -1;
};

/** Scene-local frame where the line's speech starts and ends. */
export const lineSpan = (line: LineProps, sceneFrom: number) => {
  const start = line.from - sceneFrom;
  return {start, end: start + Math.max(1, line.dur)};
};

/**
 * Scene-local frames for items anchored on spoken words. Items whose `at` is missing or not found are spread
 * evenly across the line's speech by position, so a bad anchor never hides an item.
 */
export const cueFrames = (ats: (string | null | undefined)[], line: LineProps, sceneFrom: number, fps: number): number[] => {
  const {start, end} = lineSpan(line, sceneFrom);
  const n = ats.length;
  const span = Math.max(24, (end - start) * 0.78);
  let prev = -1;
  return ats.map((at, i) => {
    const idx = findWord(line.words ?? [], at, prev + 1);
    if (idx >= 0) {
      prev = idx;
      return Math.round(start + (line.words[idx].s ?? 0) * fps) - 2;
    }
    return Math.round(start + 4 + (span * i) / Math.max(1, n));
  });
};

/**
 * Frames for each token of some on-screen text (a headline, a quote), aligned to the narration where tokens match
 * and interpolated between matches. With no matches the tokens spread across the first part of the line.
 */
export const alignText = (tokens: string[], line: LineProps, sceneFrom: number, fps: number): number[] => {
  const {start, end} = lineSpan(line, sceneFrom);
  const words = line.words ?? [];
  const out: (number | null)[] = new Array(tokens.length).fill(null);
  let last = -1;
  tokens.forEach((t, i) => {
    const idx = findWord(words.slice(last + 1), t);
    if (idx >= 0 && t.length > 0) {
      last = last + 1 + idx;
      out[i] = Math.round(start + words[last].s * fps) - 2;
    }
  });
  const anchors = out.map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
  const fallbackEnd = start + (end - start) * 0.6;
  if (!anchors.length) return tokens.map((_, i) => Math.round(start + 4 + ((fallbackEnd - start - 4) * i) / Math.max(1, tokens.length)));
  const res = out.slice() as number[];
  const first = anchors[0];
  for (let i = 0; i < first; i++) res[i] = Math.round(start + 4 + ((res[first] - start - 4) * i) / first);
  for (let a = 0; a < anchors.length; a++) {
    const i0 = anchors[a];
    const i1 = anchors[a + 1];
    const f0 = res[i0];
    const f1 = i1 === undefined ? f0 + 7 * (tokens.length - i0) : res[i1];
    const stop = i1 ?? tokens.length;
    for (let i = i0 + 1; i < stop; i++) res[i] = Math.round(f0 + ((f1 - f0) * (i - i0)) / (stop - i0));
  }
  return res.map((f, i) => Math.max(i ? Math.round(res[i - 1]) : start, Math.round(f)));
};
