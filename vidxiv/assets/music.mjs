// Royalty-free music bed + soft foley SFX, synthesized locally.
// Usage: node music.mjs <project-dir> [--mood warm|bright|calm|night] [--duration seconds]
//   warm   – lo-fi keys + marimba + brushed drums (default; friendly explainers)
//   bright – a bit quicker and sunnier (product/feature, upbeat topics)
//   calm   – no drums, soft keys only (serious or sensitive topics)
//   night  – minor-key, mellow (pairs with the "ink" dark palette)
// Writes public/audio/music.wav (+ pop, pencil, tick, swell .wav)
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const proj = path.resolve(args[0] ?? '.');
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const mood = opt('mood', 'warm');
let duration = Number(opt('duration', 0));
if (!duration) {
  try {
    const vo = JSON.parse(fs.readFileSync(path.join(proj, 'src/vo.json'), 'utf8'));
    const script = JSON.parse(fs.readFileSync(path.join(proj, 'script.json'), 'utf8'));
    const gaps = script.lines.slice(0, -1).reduce((s, l) => s + (l.gapAfter ?? 14), 0) / 30;
    duration = Math.ceil(vo.reduce((s, l) => s + l.duration, 0) + gaps + (script.tail ?? 90) / 30 + 4);
  } catch { duration = 70; }
}
const OUT = path.join(proj, 'public/audio');
fs.mkdirSync(OUT, {recursive: true});

const SR = 48000;
function writeWav(file, L, R = L) {
  const n = L.length, buf = Buffer.alloc(44 + n * 4);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 4, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22); buf.writeUInt32LE(SR, 24);
  buf.writeUInt32LE(SR * 4, 28); buf.writeUInt16LE(4, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) {
    buf.writeInt16LE(Math.round(Math.tanh(L[i]) * 32767), 44 + i * 4);
    buf.writeInt16LE(Math.round(Math.tanh(R[i]) * 32767), 46 + i * 4);
  }
  fs.writeFileSync(path.join(OUT, file), buf);
}
let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
const mtof = (m) => 440 * 2 ** ((m - 69) / 12);
const lp1 = (x, fc) => { const a = 1 - Math.exp(-2 * Math.PI * fc / SR); let s = 0; return x.map(v => (s += a * (v - s))); };
const hp1 = (x, fc) => { const l = lp1(x, fc); return x.map((v, i) => v - l[i]); };
function reverb(x, mix = 0.25, size = 0.84, damp = 0.35, spread = 23) {
  const combs = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116], aps = [556, 441, 341, 225];
  const run = (off) => {
    const out = new Float32Array(x.length);
    for (const c of combs) { const len = Math.round((c + off) * SR / 44100), b = new Float32Array(len); let idx = 0, f = 0;
      for (let i = 0; i < x.length; i++) { const y = b[idx]; f = y * (1 - damp) + f * damp; b[idx] = x[i] * 0.015 + f * size; out[i] += y; idx = (idx + 1) % len; } }
    for (const a of aps) { const len = Math.round((a + off) * SR / 44100), b = new Float32Array(len); let idx = 0;
      for (let i = 0; i < x.length; i++) { const bb = b[idx]; const y = -out[i] + bb; b[idx] = out[i] + bb * 0.5; out[i] = y; idx = (idx + 1) % len; } }
    return out;
  };
  const wl = run(0), wr = run(spread);
  return [x.map((v, i) => v * (1 - mix) + wl[i] * mix), x.map((v, i) => v * (1 - mix) + wr[i] * mix)];
}

const MOODS = {
  warm:   {bpm: 86,  drums: true,  marimba: true,  prog: [[48, 55, 59, 62, 64], [45, 52, 55, 59, 60], [41, 48, 52, 55, 57], [43, 50, 52, 57, 59]], roots: [36, 33, 29, 31]},
  bright: {bpm: 104, drums: true,  marimba: true,  prog: [[53, 57, 60, 64], [52, 55, 60, 64], [50, 53, 57, 60], [46, 50, 53, 57]], roots: [29, 28, 26, 34]},
  calm:   {bpm: 72,  drums: false, marimba: false, prog: [[48, 55, 59, 64], [45, 52, 55, 60], [41, 48, 52, 57], [43, 50, 55, 59]], roots: [36, 33, 29, 31]},
  night:  {bpm: 80,  drums: true,  marimba: true,  prog: [[45, 52, 55, 59, 60], [41, 48, 52, 55], [38, 45, 48, 52, 53], [40, 47, 50, 52, 56]], roots: [33, 29, 26, 28]},
};
const M = MOODS[mood] ?? MOODS.warm;

{
  const D = duration, N = Math.floor(D * SR), beat = 60 / M.bpm, bar = beat * 4;
  const keys = new Float32Array(N), mar = new Float32Array(N), drums = new Float32Array(N), bass = new Float32Array(N);
  const add = (arr, t0, dur, f, amp, kind) => {
    const s0 = Math.floor(t0 * SR), n = Math.floor(dur * SR), det = 1 + rnd() * 0.0015;
    for (let j = 0; j < n && s0 + j < N; j++) {
      const t = j / SR; let v;
      if (kind === 'ep') {
        const env = Math.min(1, t / 0.015) * Math.exp(-t * 1.1) * (j > n - 2400 ? (n - j) / 2400 : 1);
        v = (Math.sin(2 * Math.PI * f * det * t) + 0.25 * Math.sin(2 * Math.PI * f * 2 * t) * Math.exp(-t * 4) + 0.08 * Math.sin(2 * Math.PI * f * 7.1 * t) * Math.exp(-t * 18)) * env;
        v *= 0.85 + 0.15 * Math.sin(2 * Math.PI * 4.2 * (t0 + t));
      } else if (kind === 'mar') {
        v = (Math.sin(2 * Math.PI * f * t) + 0.3 * Math.sin(2 * Math.PI * f * 3.93 * t) * Math.exp(-t * 30)) * Math.exp(-t * 7) * Math.min(1, t / 0.002);
      } else {
        const env = Math.min(1, t / 0.01) * Math.exp(-t * 1.6) * (j > n - 1200 ? (n - j) / 1200 : 1);
        v = Math.tanh(1.4 * Math.sin(2 * Math.PI * f * t)) * env;
      }
      arr[s0 + j] += v * amp;
    }
  };
  for (let b = 0; b < Math.ceil(D / bar); b++) {
    const t0 = b * bar, ch = M.prog[b % 4];
    ch.forEach((m, k) => add(keys, t0 + k * 0.018, bar * 0.95, mtof(m), 0.07, 'ep'));
    ch.slice(1).forEach((m, k) => add(keys, t0 + beat * 2.5 + k * 0.012, beat * 1.4, mtof(m), 0.035, 'ep'));
    if (b >= 1) { add(bass, t0, beat * 2.6, mtof(M.roots[b % 4]), 0.16, 'bass'); add(bass, t0 + beat * 3, beat * 0.9, mtof(M.roots[b % 4] + 7), 0.11, 'bass'); }
    if (M.marimba && b >= 2) [0, null, 2, 3, null, 2, 4, null].forEach((p, k) => {
      if (p !== null && !(b % 4 === 3 && k > 5)) add(mar, t0 + k * beat / 2, 0.6, mtof(ch[p % ch.length] + 12), 0.05 + (k === 0 ? 0.02 : 0), 'mar');
    });
    if (M.drums && b >= 1) {
      for (const kb of [0, 2.5]) { const s0 = Math.floor((t0 + kb * beat) * SR); for (let j = 0; j < 0.35 * SR && s0 + j < N; j++) { const t = j / SR; drums[s0 + j] += Math.sin(2 * Math.PI * (48 + 70 * Math.exp(-t * 35)) * t) * Math.exp(-t * 9) * 0.32; } }
      for (const sb of [1, 3]) { const s0 = Math.floor((t0 + sb * beat) * SR); for (let j = 0; j < 0.25 * SR && s0 + j < N; j++) { const t = j / SR; drums[s0 + j] += rnd() * Math.exp(-t * 18) * 0.07 + Math.sin(2 * Math.PI * 190 * t) * Math.exp(-t * 30) * 0.04; } }
      for (let h = 0; h < 8; h++) { const s0 = Math.floor((t0 + h * beat / 2 + (h % 2 ? 0.02 : 0)) * SR); for (let j = 0; j < 0.06 * SR && s0 + j < N; j++) { const t = j / SR; drums[s0 + j] += rnd() * Math.exp(-t * 60) * (h % 2 ? 0.03 : 0.018); } }
    }
  }
  const keysF = lp1(keys, 3200), marF = lp1(mar, 5000), dry = new Float32Array(N);
  for (let i = 0; i < N; i++) dry[i] = keysF[i] * (M.drums ? 1 : 1.6) + marF[i] * 0.9;
  const [wl, wr] = reverb(dry, M.drums ? 0.32 : 0.45);
  const [dl, dr] = reverb(hp1(lp1(drums, 7000), 30), 0.12);
  const L = new Float32Array(N), R = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const crackle = Math.random() < 0.00025 ? (Math.random() - 0.5) * 0.25 : 0, hiss = rnd() * 0.0035;
    const t = i / SR, g = Math.min(1, t / 1.2) * Math.min(1, (D - t) / 3);
    L[i] = (wl[i] + bass[i] + dl[i] + crackle + hiss) * g;
    R[i] = (wr[i] + bass[i] + dr[i] + crackle + hiss) * g;
  }
  writeWav('music.wav', lp1(L, 9000), lp1(R, 9000));
}

const arr = (s) => new Float32Array(Math.floor(s * SR));
{ const x = arr(0.16); let ph = 0; for (let i = 0; i < x.length; i++) { const t = i / SR; ph += 2 * Math.PI * (380 + 520 * Math.min(1, t / 0.05)) / SR; x[i] = Math.sin(ph) * Math.exp(-t * 32) * Math.min(1, t / 0.003) * 0.5; }
  const [l, r] = reverb(x, 0.15); writeWav('pop.wav', l, r); }
{ const d = 0.7, x = arr(d); for (let i = 0; i < x.length; i++) { const t = i / SR; x[i] = rnd() * (0.6 + 0.4 * Math.sin(2 * Math.PI * 23 * t + Math.sin(t * 40))) * Math.sin(Math.PI * t / d) ** 0.6; }
  writeWav('pencil.wav', hp1(lp1(x, 5200), 1800).map(v => v * 0.9)); }
{ const x = arr(0.05); for (let i = 0; i < x.length; i++) { const t = i / SR; x[i] = (Math.sin(2 * Math.PI * 1500 * t) * 0.6 + rnd() * 0.2) * Math.exp(-t * 140) * 0.45; } writeWav('tick.wav', x); }
{ const d = 1.6, x = arr(d); for (let i = 0; i < x.length; i++) { const t = i / SR, p = t / d; x[i] = (Math.sin(2 * Math.PI * 261.6 * t) + Math.sin(2 * Math.PI * 329.6 * t) + Math.sin(2 * Math.PI * 392 * t) + rnd() * 0.3) * p ** 2.2 * 0.12; }
  const [l, r] = reverb(lp1(x, 2500), 0.4); writeWav('swell.wav', l, r); }
console.log(`music (${mood}, ${duration}s) + sfx → ${OUT}`);
