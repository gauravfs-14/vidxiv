// Typecheck, then render a handful of stills of the sample in every format. Exits non-zero on any failure.
import {spawnSync} from 'node:child_process';
import {mkdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const run = (cmd, args) => {
  const r = spawnSync(cmd, args, {cwd: root, stdio: 'inherit'});
  if (r.status !== 0) {
    console.error(`check failed: ${cmd} ${args.join(' ')}`);
    process.exit(r.status || 1);
  }
};

run('npx', ['tsc']);

const sample = JSON.parse(readFileSync(join(root, 'src/sample.json'), 'utf8'));
const out = join(root, 'out/check');
mkdirSync(out, {recursive: true});
// Scene types that stress layout most: figure with zoom, equation (KaTeX), steps, compare.
const want = ['figure', 'equation', 'steps', 'compare'];
const frames = want.map((t) => {
  const l = sample.lines.find((x) => x.scene.type === t);
  return [t, Math.round(l.from + l.dur * 0.9)];
});
for (const comp of ['Vertical', 'Landscape', 'Square']) {
  for (const [t, f] of frames) {
    const file = join(out, `${comp}-${t}.png`);
    run('npx', ['remotion', 'still', 'src/index.ts', comp, file, `--frame=${f}`, '--scale=0.5', '--public-dir=public', '--log=error']);
    if (statSync(file).size < 5000) {
      console.error(`check failed: ${file} looks empty`);
      process.exit(1);
    }
  }
}
console.log(`check ok: stills in ${out}`);
