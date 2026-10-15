// Render the sample in every look and lay stills out as a contact sheet.
//   node scripts/looks.mjs                  contact sheet: one landscape and one portrait still per look -> out/looks/looks.png
//   node scripts/looks.mjs --all            every scene type in every look, both orientations -> out/qa/
//   --scenes=hook,stat   limit to some scene types      --out=path   copy the contact sheet there
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import {copyFileSync, mkdirSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.replace(/^--/, '').split('=')).map(([k, v]) => [k, v ?? true]));
const LOOKS = ['studio', 'night', 'swiss', 'cobalt'];
const sample = JSON.parse(readFileSync(join(root, 'src/sample.json'), 'utf8'));
const sceneFrame = (type) => {
  const l = sample.lines.find((x) => x.scene.type === type);
  return Math.round(l.from + l.dur - 2);
};

const shoot = async (serveUrl, id, look, frame, file, scale = 0.5) => {
  const inputProps = {...sample, look};
  const comp = await selectComposition({serveUrl, id, inputProps});
  await renderStill({composition: comp, serveUrl, output: file, frame, inputProps, scale, imageFormat: 'png', logLevel: 'error'});
};

const main = async () => {
  const serveUrl = await bundle({entryPoint: join(root, 'src/index.ts'), publicDir: join(root, 'public')});
  if (args.all) {
    const out = join(root, 'out/qa');
    mkdirSync(out, {recursive: true});
    const types = args.scenes ? String(args.scenes).split(',') : [...new Set(sample.lines.map((l) => l.scene.type))];
    for (const look of LOOKS) {
      for (const [comp, tag] of [['Landscape', 'L'], ['Vertical', 'P']]) {
        for (const t of types) await shoot(serveUrl, comp, look, sceneFrame(t), join(out, `${look}-${tag}-${t}.png`));
      }
    }
    console.log(`stills in ${out}`);
    return;
  }
  const out = join(root, 'out/looks');
  mkdirSync(out, {recursive: true});
  const [lt, pt] = String(args.pair ?? 'hook,stat').split(',');
  for (const look of LOOKS) {
    await shoot(serveUrl, 'Landscape', look, sceneFrame(lt), join(out, `${look}-L.png`));
    await shoot(serveUrl, 'Vertical', look, sceneFrame(pt), join(out, `${look}-P.png`));
  }
  const sheetUrl = await bundle({entryPoint: join(root, 'src/index.ts'), publicDir: out});
  const inputProps = {columns: LOOKS.map((l) => ({label: l, landscape: `${l}-L.png`, portrait: `${l}-P.png`}))};
  const comp = await selectComposition({serveUrl: sheetUrl, id: 'ContactSheet', inputProps});
  const file = join(out, 'looks.png');
  await renderStill({composition: comp, serveUrl: sheetUrl, output: file, frame: 0, inputProps, imageFormat: 'png', logLevel: 'error'});
  if (typeof args.out === 'string') {
    mkdirSync(dirname(resolve(args.out)), {recursive: true});
    copyFileSync(file, resolve(args.out));
  }
  console.log(`contact sheet: ${args.out ? resolve(args.out) : file}`);
};
main().catch((e) => {
  console.error(e);
  process.exit(1);
});
