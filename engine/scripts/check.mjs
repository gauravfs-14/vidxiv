// Typecheck, then render stills of the sample in all four looks and in every format, plus a malformed-props job.
// Exits non-zero on any failure.
import {spawnSync} from 'node:child_process';
import {mkdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const fail = (msg) => {
  console.error(`check failed: ${msg}`);
  process.exit(1);
};

const tsc = spawnSync('npx', ['tsc'], {cwd: root, stdio: 'inherit'});
if (tsc.status !== 0) fail('tsc');

const sample = JSON.parse(readFileSync(join(root, 'src/sample.json'), 'utf8'));
const out = join(root, 'out/check');
mkdirSync(out, {recursive: true});
const LOOKS = ['studio', 'night', 'swiss', 'cobalt'];
// Scene types that stress layout most: hero with figure, lists, KaTeX, a big number, a split.
const want = ['hook', 'concept', 'figure', 'equation', 'stat', 'compare'];
const frameOf = (t) => {
  const l = sample.lines.find((x) => x.scene.type === t);
  return l.from + l.dur - 2;
};

const shoot = async (serveUrl, comp, inputProps, frame, name) => {
  const file = join(out, `${name}.png`);
  const c = await selectComposition({serveUrl, id: comp, inputProps});
  await renderStill({composition: c, serveUrl, output: file, frame, inputProps, scale: 0.5, imageFormat: 'png', logLevel: 'error'});
  if (statSync(file).size < 5000) fail(`${file} looks empty`);
};

try {
  const serveUrl = await bundle({entryPoint: join(root, 'src/index.ts'), publicDir: join(root, 'public')});
  for (const look of LOOKS) {
    const props = {...sample, look};
    for (const comp of ['Landscape', 'Vertical']) {
      for (const t of want) await shoot(serveUrl, comp, props, frameOf(t), `${look}-${comp}-${t}`);
    }
    for (const t of ['hook', 'steps']) await shoot(serveUrl, 'Square', props, frameOf(t), `${look}-Square-${t}`);
  }
  // Robustness: unknown look, legacy palette name, empty arrays, missing figure, bad LaTeX, very long text, file null.
  const long = 'A very long headline that keeps going well past what any sensible storyboard would allow on one screen, to see that it shrinks to fit';
  const mk = (scene, i) => ({index: i, text: 'x', file: null, from: 10, dur: 90, words: [], scene, kicker: 'Same', highlight: {}});
  const scenes = [
    {type: 'hook', headline: long, sub: long, figure: 'missing-id'},
    {type: 'concept', heading: long, body: long, items: []},
    {type: 'steps', steps: []},
    {type: 'bars', bars: []},
    {type: 'equation', formula: '\\frac{a', caption: long, terms: []},
    {type: 'figure', figure: 'nope', callouts: [{text: long, x: 0.5, y: 0.5}]},
  ];
  for (const [i, s] of scenes.entries()) {
    const props = {fps: 30, total: 150, look: i % 2 ? 'ink' : 'no-such-look', lines: [mk(s, 0)]};
    await shoot(serveUrl, 'Vertical', props, 80, `robust-${s.type}`);
  }
  await shoot(serveUrl, 'Landscape', {fps: 30, total: 150, palette: 'paper', lines: []}, 20, 'robust-empty');
} catch (e) {
  fail(String(e?.stack ?? e));
}
console.log(`check ok: stills in ${out}`);
