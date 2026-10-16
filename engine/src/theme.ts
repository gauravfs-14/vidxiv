import {createContext, useContext} from 'react';
import {loadFont as loadGeist} from '@remotion/google-fonts/Geist';
import {loadFont as loadGeistMono} from '@remotion/google-fonts/GeistMono';
import {loadFont as loadHanken} from '@remotion/google-fonts/HankenGrotesk';
import {loadFont as loadBricolage} from '@remotion/google-fonts/BricolageGrotesque';
import type {Look} from './types';

export const LOOKS: Look[] = ['studio', 'night', 'swiss', 'cobalt'];
/** Unknown or missing values fall back to studio. Legacy palette names are mapped. */
export const resolveLook = (v?: string | null): Look => {
  if (v === 'paper') return 'studio';
  if (v === 'ink') return 'night';
  return LOOKS.includes(v as Look) ? (v as Look) : 'studio';
};

export type Colors = {
  ground: string; ink: string; soft: string; line: string; wash: string; raised: string;
  accent: string; onAccent: string; teal: string; rose: string; blue: string; mustard: string;
  /** Rgb triplet of the shadow tint, tuned to the ground hue. */
  shadow: string;
};

type Def = {
  dark: boolean;
  C: Colors;
  fonts: () => {sans: string; mono: string};
  /** Display weight, tracking (em) and line height for headlines. */
  dw: number; dt: number; dl: number;
  /** Corner radius in design px, locked for the whole look. */
  r: number;
  /** Rule thickness in design px. */
  rule: number;
  /** Caption and marker highlight: a solid block with contrast text, or a tinted block behind ink text. */
  solid: boolean;
  /** Raised objects (the paper figure): soft tinted shadow, an outer ring, or a hard border. */
  raised: 'shadow' | 'ring' | 'border';
  bg: 'dots' | 'lines' | 'grid' | 'flat';
};

const mono = () => loadGeistMono('normal', {weights: ['400', '500', '600'], subsets: ['latin']}).fontFamily;
const geist = () => loadGeist('normal', {weights: ['400', '500', '600', '700', '800'], subsets: ['latin']}).fontFamily;

const DEFS: Record<Look, Def> = {
  studio: {
    dark: false,
    C: {ground: '#E9ECF0', ink: '#0F1218', soft: '#566070', line: '#CBD1DA', wash: '#DEE2E8', raised: '#FFFFFF', accent: '#2D5BFF', onAccent: '#FFFFFF', teal: '#0C8F86', rose: '#D8456B', blue: '#2193D6', mustard: '#C98900', shadow: '28, 44, 96'},
    fonts: () => ({sans: geist(), mono: mono()}),
    dw: 700, dt: -0.038, dl: 1.02, r: 14, rule: 2, solid: true, raised: 'shadow', bg: 'dots',
  },
  night: {
    dark: true,
    C: {ground: '#131519', ink: '#ECEEF1', soft: '#9AA2AE', line: '#2B2F37', wash: '#1C1F25', raised: '#F2F3F5', accent: '#2FCB8A', onAccent: '#08130E', teal: '#4FB7D3', rose: '#E5708F', blue: '#7B93F0', mustard: '#E2B13C', shadow: '0, 0, 0'},
    fonts: () => ({sans: geist(), mono: mono()}),
    dw: 600, dt: -0.035, dl: 1.04, r: 12, rule: 2, solid: false, raised: 'ring', bg: 'lines',
  },
  swiss: {
    dark: false,
    C: {ground: '#FAFAFA', ink: '#0A0A0A', soft: '#575757', line: '#D6D6D6', wash: '#EEEEEE', raised: '#FFFFFF', accent: '#E5251B', onAccent: '#FFFFFF', teal: '#00766E', rose: '#C0287A', blue: '#1D4ED8', mustard: '#C98500', shadow: '0, 0, 0'},
    fonts: () => ({sans: loadHanken('normal', {weights: ['400', '500', '600', '700', '800', '900'], subsets: ['latin']}).fontFamily, mono: mono()}),
    dw: 800, dt: -0.04, dl: 0.98, r: 0, rule: 4, solid: true, raised: 'border', bg: 'grid',
  },
  cobalt: {
    dark: true,
    C: {ground: '#1F43E6', ink: '#FFFFFF', soft: '#DCE3FF', line: '#5873F2', wash: '#1733C2', raised: '#FFFFFF', accent: '#FFC82E', onAccent: '#101A5E', teal: '#54E3C2', rose: '#FF8FB4', blue: '#A5C8FF', mustard: '#FF9F45', shadow: '6, 16, 100'},
    fonts: () => ({sans: loadBricolage('normal', {weights: ['400', '500', '600', '700', '800'], subsets: ['latin']}).fontFamily, mono: mono()}),
    dw: 800, dt: -0.035, dl: 0.98, r: 22, rule: 4, solid: true, raised: 'shadow', bg: 'flat',
  },
};

export const rgba = (hex: string, a: number) => {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};
const lum = (hex: string) => {
  const n = parseInt(hex.replace('#', ''), 16);
  const f = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * f((n >> 16) & 255) + 0.7152 * f((n >> 8) & 255) + 0.0722 * f(n & 255);
};
const ratio = (a: string, b: string) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

/** Tints that stay meaningful across scenes. An unset tint is plain ink: only meaning earns colour. */
const TINTS = ['accent', 'teal', 'rose', 'blue', 'mustard'] as const;

export const makeTheme = (name?: string | null) => {
  const look = resolveLook(name);
  const d = DEFS[look];
  const C = d.C;
  const f = d.fonts();
  /** Solid colour for a tint name. Empty or unknown tints give ink. */
  const color = (tint?: string | null) => (TINTS.includes(tint as never) ? C[tint as (typeof TINTS)[number]] : C.ink);
  /** Readable text colour on top of a solid block. */
  const dk = look === 'cobalt' ? '#101A5E' : '#0A0A0A';
  const onColor = (bg: string) => (ratio('#FFFFFF', bg) >= ratio(dk, bg) ? '#FFFFFF' : dk);
  return {look, dark: d.dark, C, sans: f.sans, mono: f.mono, dw: d.dw, dt: d.dt, dl: d.dl, r: d.r, rule: d.rule, solid: d.solid, raised: d.raised, bg: d.bg, color, onColor, rgba};
};
export type Theme = ReturnType<typeof makeTheme>;

export const ThemeContext = createContext<Theme | null>(null);
export const useTheme = (): Theme => {
  const t = useContext(ThemeContext);
  if (!t) throw new Error('useTheme outside ThemeContext');
  return t;
};

export const EASE_OUT = [0.16, 1, 0.3, 1] as const;
