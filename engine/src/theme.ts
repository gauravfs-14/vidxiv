import {createContext, useContext} from 'react';
import {loadFont as loadSerif} from '@remotion/google-fonts/InstrumentSerif';
import {loadFont as loadSans} from '@remotion/google-fonts/InterTight';
import {loadFont as loadHand} from '@remotion/google-fonts/Caveat';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export const SERIF = loadSerif('normal', {weights: ['400'], subsets: ['latin']}).fontFamily;
loadSerif('italic', {weights: ['400'], subsets: ['latin']});
export const SANS = loadSans('normal', {weights: ['500', '600', '700'], subsets: ['latin']}).fontFamily;
export const HAND = loadHand('normal', {weights: ['600'], subsets: ['latin']}).fontFamily;
export const MONO = loadMono('normal', {weights: ['400', '500'], subsets: ['latin']}).fontFamily;

const PALETTES = {
  paper: {
    dark: false,
    paper: '#F7F0E3', card: '#FFFCF6', ink: '#211E1A', inkSoft: '#6A6359', line: '#D9CFBF', muted: '#EDE4D6',
    accent: '#D9572B', teal: '#1E7C74', rose: '#C4466F', mustard: '#F2C14E', blue: '#3C62C4',
    paperFront: '#EEDFC4', paperBack: '#FCF7EE', shadow: 'rgba(90, 60, 30, 0.14)',
  },
  ink: {
    dark: true,
    paper: '#161A22', card: '#20252F', ink: '#F1ECE2', inkSoft: '#A39C90', line: '#343B48', muted: '#2A303B',
    accent: '#F07A4A', teal: '#4FB3A9', rose: '#E07399', mustard: '#F2C14E', blue: '#7C9BEA',
    paperFront: '#1B2029', paperBack: '#12151C', shadow: 'rgba(0, 0, 0, 0.35)',
  },
};
export type Palette = (typeof PALETTES)['paper'] | (typeof PALETTES)['ink'];

const rgba = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

const CYCLE = ['accent', 'teal', 'blue', 'rose', 'mustard'] as const;

export const makeTheme = (name?: string) => {
  const C: Palette = PALETTES[name as keyof typeof PALETTES] ?? PALETTES.paper;
  /** Solid colour for a tint name. Unknown or empty tints fall back to the cycle so neighbours differ. */
  const color = (tint?: string | null, i = 0) => {
    const key = (tint || CYCLE[i % CYCLE.length]) as keyof Palette;
    if (key === 'mustard') return C.dark ? C.mustard : '#D9A21B';
    return (typeof C[key] === 'string' && String(C[key]).startsWith('#') ? C[key] : C.accent) as string;
  };
  /** Translucent marker colour for captions and highlights. */
  const marker = (tint?: string | null) => {
    const key = (tint || 'mustard') as 'mustard' | 'accent' | 'teal' | 'rose' | 'blue';
    const hex = ['mustard', 'accent', 'teal', 'rose', 'blue'].includes(key) ? C[key] : C.mustard;
    return rgba(hex, key === 'mustard' ? (C.dark ? 0.45 : 0.65) : C.dark ? 0.45 : 0.32);
  };
  return {C, color, marker, rgba};
};
export type Theme = ReturnType<typeof makeTheme>;

export const ThemeContext = createContext<Theme>(makeTheme('paper'));
export const useTheme = () => useContext(ThemeContext);

export const EASE_OUT = [0.16, 1, 0.3, 1] as const;
