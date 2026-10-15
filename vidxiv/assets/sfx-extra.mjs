// Extra SFX for this video (synthesized, royalty-free). Run from the project dir: node sfx-extra.mjs
//   soft  – felt-marker swish        chime – gentle two-note "correct" (used once)
//   thud  – soft stamp / block       blip  – tiny token "type" blip (blip-lo / blip-hi variants)
//   mask  – downward filtered sweep  rise  – short upward riser into a reveal
//   buzz  – muted error buzz         snap  – crisp UI snap for kinetic text
import fs from 'node:fs';
const SR = 48000;
const writeWav = (file, x) => {
  const b = Buffer.alloc(44 + x.length * 4);
  b.write('RIFF', 0); b.writeUInt32LE(36 + x.length * 4, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(2, 22);
  b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(x.length * 4, 40);
  x.forEach((v, i) => { const s = Math.max(-1, Math.min(1, v)) * 32767; b.writeInt16LE(s | 0, 44 + i * 4); b.writeInt16LE(s | 0, 46 + i * 4); });
  fs.writeFileSync(`public/audio/${file}`, b);
};
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
const lp = (x, fc) => { const a = 1 - Math.exp(-2 * Math.PI * fc / SR); let s = 0; return x.map(v => (s += a * (v - s))); };
const hp = (x, fc) => { const l = lp(x, fc); return x.map((v, i) => v - l[i]); };
const echo = (x, d = 0.11, g = 0.28, n = 4) => { const y = Float32Array.from(x); const k = Math.floor(d * SR); for (let r = 1; r <= n; r++) for (let i = r * k; i < y.length; i++) y[i] += x[i - r * k] * g ** r * 0.9; return lp(y, 6000); };
const buf = (d) => new Float32Array(Math.floor(d * SR));

{ const n = Math.floor(0.42 * SR); let x = new Float32Array(n).map(() => rnd());
  const a = lp(x, 900), b = lp(x, 2600);
  x = x.map((_, i) => { const p = i / n; return (b[i] - a[i] * 0.7) * Math.sin(Math.PI * p) ** 2 * 0.55; });
  writeWav('soft.wav', echo(x, 0.07, 0.2, 2)); }

{ const x = buf(1.4);
  [[659.25, 0], [987.77, 0.09]].forEach(([f, t0]) => { const s0 = Math.floor(t0 * SR);
    for (let i = s0; i < x.length; i++) { const t = (i - s0) / SR; const env = Math.min(1, t / 0.006) * Math.exp(-t * 4.2);
      x[i] += (Math.sin(2 * Math.PI * f * t) + 0.18 * Math.sin(2 * Math.PI * f * 2.01 * t) * Math.exp(-t * 9)) * env * 0.26; } });
  writeWav('chime.wav', echo(x, 0.13, 0.25, 3)); }

// thud: pitch-dropping sine body + short paper slap
{ const x = buf(0.5); let ph = 0;
  for (let i = 0; i < x.length; i++) { const t = i / SR; const f = 55 + 95 * Math.exp(-t * 28); ph += 2 * Math.PI * f / SR;
    x[i] = Math.sin(ph) * Math.exp(-t * 11) * 0.85; }
  let s = buf(0.06).map(() => rnd()); s = lp(hp(s, 400), 3500);
  s.forEach((v, i) => { x[i] += v * Math.exp(-i / SR * 70) * 0.5; });
  writeWav('thud.wav', echo(x, 0.06, 0.12, 1)); }

// blips: short soft sine+triangle with tiny pitch flick
for (const [name, f0] of [['blip-lo', 740], ['blip', 988], ['blip-hi', 1318]]) {
  const x = buf(0.09); let ph = 0;
  for (let i = 0; i < x.length; i++) { const t = i / SR; ph += 2 * Math.PI * f0 * (1 + 0.08 * Math.exp(-t * 80)) / SR;
    const tri = 2 / Math.PI * Math.asin(Math.sin(ph)); x[i] = (Math.sin(ph) * 0.6 + tri * 0.25) * Math.min(1, t / 0.002) * Math.exp(-t * 55) * 0.5; }
  writeWav(`${name}.wav`, x); }

// mask: noise through a falling low-pass, plus falling sine
{ const x = buf(0.38); let ph = 0; let s = 0;
  for (let i = 0; i < x.length; i++) { const t = i / SR, p = t / 0.38; const fc = 5000 * Math.exp(-p * 3.2) + 200;
    const a = 1 - Math.exp(-2 * Math.PI * fc / SR); s += a * (rnd() - s); ph += 2 * Math.PI * (900 * Math.exp(-p * 2.2)) / SR;
    const env = Math.min(1, t / 0.01) * (1 - p) ** 1.5; x[i] = (s * 0.7 + Math.sin(ph) * 0.18) * env * 0.6; }
  writeWav('mask.wav', echo(x, 0.05, 0.15, 1)); }

// rise: upward filtered-noise sweep + soft rising sine
{ const D = 0.75, x = buf(D); let ph = 0, s = 0;
  for (let i = 0; i < x.length; i++) { const t = i / SR, p = t / D; const fc = 300 + 5200 * p * p;
    const a = 1 - Math.exp(-2 * Math.PI * fc / SR); s += a * (rnd() - s); ph += 2 * Math.PI * (220 + 660 * p * p) / SR;
    const env = p ** 1.6 * (p > 0.94 ? (1 - p) / 0.06 : 1); x[i] = (s * 0.55 + Math.sin(ph) * 0.14) * env * 0.55; }
  writeWav('rise.wav', x); }

// buzz: muted low square through low-pass, two short pulses
{ const x = buf(0.32);
  for (let i = 0; i < x.length; i++) { const t = i / SR; const on = (t < 0.11) || (t > 0.15 && t < 0.27);
    const sq = Math.sign(Math.sin(2 * Math.PI * 155 * t)); x[i] = on ? sq * 0.32 : 0; }
  const y = lp(lp(x, 1400), 1800); const n = Math.floor(0.004 * SR);
  for (let i = 0; i < y.length; i++) { const t = i / SR; const edge = Math.min(1, (t % 0.15) * SR / n); y[i] *= edge; }
  writeWav('buzz.wav', y); }

// snap: bright click + tiny body
{ const x = buf(0.12); let s = buf(0.12).map(() => rnd()); s = hp(s, 2000);
  for (let i = 0; i < x.length; i++) { const t = i / SR; x[i] = s[i] * Math.exp(-t * 160) * 0.7 + Math.sin(2 * Math.PI * 1900 * t) * Math.exp(-t * 90) * 0.18; }
  writeWav('snap.wav', x); }
console.log('extra sfx written');
