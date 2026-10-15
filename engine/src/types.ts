// Props contract: see ../../docs/ENGINE_CONTRACT.md and vidxiv/schema.py. Scene keys are camelCase.
export type Tint = '' | 'accent' | 'teal' | 'rose' | 'blue' | 'mustard';
export type Word = {w: string; s: number; e: number};

export type Item = {label: string; detail?: string | null; at?: string | null; tint?: Tint | null};
export type Zoom = {x: number; y: number; w: number; h: number; at?: string | null};
export type Callout = {text: string; x: number; y: number; at?: string | null};
export type Bar = {label: string; value: number; display?: string | null; highlight?: boolean; at?: string | null};
export type Side = {title: string; points: string[]; tint?: Tint | null};

export type Scene =
  | {type: 'hook'; headline: string; sub?: string | null; circle?: string | null; figure?: string | null}
  | {type: 'title'; title: string; authors?: string | null; meta?: string | null; figure?: string | null}
  | {type: 'concept'; heading: string; body?: string | null; items?: Item[]}
  | {type: 'steps'; heading?: string | null; steps: Item[]}
  | {type: 'figure'; figure: string; caption?: string | null; zoom?: Zoom | null; callouts?: Callout[]}
  | {type: 'stat'; value: string; label: string; detail?: string | null; compare?: Item | null}
  | {type: 'bars'; title?: string | null; unit?: string | null; bars: Bar[]}
  | {type: 'compare'; left: Side; right: Side; verdict?: string | null}
  | {type: 'equation'; formula: string; caption?: string | null; terms?: Item[]}
  | {type: 'quote'; text: string; attribution?: string | null}
  | {type: 'outro'; headline: string; takeaway?: string | null; cta?: string | null; link?: string | null};

export type SceneType = Scene['type'];
export type SceneOf<T extends SceneType> = Extract<Scene, {type: T}>;

export type LineProps = {
  index: number;
  text: string;
  file: string | null;
  from: number;
  dur: number;
  words: Word[];
  scene: Scene;
  highlight?: Record<string, Tint>;
  kicker?: string | null;
  gapAfter?: number | null;
};

export type FigureAsset = {file: string; width: number; height: number; caption?: string | null};

export type VideoProps = {
  fps: number;
  total: number;
  title: string;
  palette: 'paper' | 'ink';
  captions: boolean;
  progressBar: boolean;
  music: {file: string; volume: number} | null;
  sfx: string[];
  figures: Record<string, FigureAsset>;
  lines: LineProps[];
  scenes: {from: number; dur: number}[];
};
