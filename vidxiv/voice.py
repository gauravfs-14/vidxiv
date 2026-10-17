"""Narration: text to speech, user recordings and uploads, word-level alignment.

Every line ends up as narration/lines/line-N.wav (48 kHz mono, about -16 LUFS) plus word times, whatever its source:
a local TTS engine, a clip the user recorded in the browser, a file the user uploaded, or a slice cut from one long take.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from . import config, script
from .jobs import Job

SR = 48000
KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
KOKORO_VOICES = [
    ("af_heart", "Heart", "American, female, warm"), ("af_bella", "Bella", "American, female, bright"),
    ("af_nicole", "Nicole", "American, female, soft"), ("af_sarah", "Sarah", "American, female, even"),
    ("am_michael", "Michael", "American, male, warm"), ("am_adam", "Adam", "American, male, deep"),
    ("bf_emma", "Emma", "British, female"), ("bf_isabella", "Isabella", "British, female, light"),
    ("bm_george", "George", "British, male"), ("bm_lewis", "Lewis", "British, male, calm"),
]
_kokoro = None
_whisper = None
_whisper_failed = False


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True, stdin=subprocess.DEVNULL)


# ---------------------------------------------------------------- engines

def kokoro_ready() -> bool:
    return all((config.MODELS / f).exists() and (config.MODELS / f).stat().st_size > 1_000_000 for f in KOKORO_FILES)


def download_kokoro(log=print) -> None:
    for name in KOKORO_FILES:
        dst = config.MODELS / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            continue
        log(f"Downloading {name} (one time)")
        part = dst.with_suffix(dst.suffix + ".part")
        with requests.get(KOKORO_BASE + name, stream=True, timeout=60) as r:
            r.raise_for_status()
            total, got, last = int(r.headers.get("content-length", 0)), 0, -10
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if total and got * 100 // total >= last + 10:
                        last = got * 100 // total
                        log(f"  {name} {last}%")
        part.rename(dst)


def _load_kokoro(log=print):
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        download_kokoro(log)
        _kokoro = Kokoro(str(config.MODELS / KOKORO_FILES[0]), str(config.MODELS / KOKORO_FILES[1]))
    return _kokoro


def mac_voices() -> list[dict]:
    if platform.system() != "Darwin" or not shutil.which("say"):
        return []
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s{2,}(\w\w_\w\w)\s+#", line)
        if m and m.group(2).startswith("en_") and not any(x in m.group(1) for x in ("Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Jester", "Organ", "Superstar", "Trinoids", "Whisper", "Zarvox", "Good News", "Albert")):
            voices.append({"id": m.group(1), "label": m.group(1), "note": m.group(2).replace("_", "-")})
    return voices


def engines() -> dict:
    mac = mac_voices()
    return {
        "kokoro": {"label": "Kokoro (local neural voice)", "ready": kokoro_ready(), "needsDownload": not kokoro_ready(), "size": "about 340 MB, one time",
                   "voices": [{"id": i, "label": l, "note": n} for i, l, n in KOKORO_VOICES]},
        "say": {"label": "macOS system voices", "ready": bool(mac), "needsDownload": False, "voices": mac},
        "espeak": {"label": "espeak-ng", "ready": bool(shutil.which("espeak-ng")), "needsDownload": False,
                   "voices": [{"id": "en-us", "label": "English (US)", "note": "robotic"}] if shutil.which("espeak-ng") else []},
    }


def resolve_engine(engine: str) -> str:
    if engine in ("kokoro", "say", "espeak"):
        return engine
    return "kokoro"  # auto: best quality first; synthesize() falls back if it fails


def synthesize(text: str, engine: str, voice: str, speed: float, out_wav: Path, log=print) -> str:
    """Write raw speech for `text` to out_wav. Returns the engine actually used."""
    spoken = script.clean(text)
    order = [resolve_engine(engine)]
    if engine == "auto" or order[0] == "kokoro":
        order += [e for e in ("say", "espeak") if e not in order]
    last_err = None
    for eng in order:
        try:
            if eng == "kokoro":
                k = _load_kokoro(log)
                v = voice if voice in {x[0] for x in KOKORO_VOICES} else "af_heart"
                audio, sr = k.create(spoken, voice=v, speed=float(speed), lang="en-us")
                sf.write(out_wav, audio, sr)
            elif eng == "say":
                if not mac_voices():
                    raise RuntimeError("no macOS voices")
                names = {x["id"] for x in mac_voices()}
                v = voice if voice in names else ("Samantha" if "Samantha" in names else next(iter(names)))
                tmp = out_wav.with_suffix(".aiff")
                subprocess.run(["say", "-v", v, "-r", str(int(175 * speed)), "-o", str(tmp), spoken], check=True, stdin=subprocess.DEVNULL)
                ffmpeg("-i", str(tmp), str(out_wav))
                tmp.unlink(missing_ok=True)
            elif eng == "espeak":
                if not shutil.which("espeak-ng"):
                    raise RuntimeError("espeak-ng not installed")
                subprocess.run(["espeak-ng", "-v", "en-us", "-s", str(int(165 * speed)), "-w", str(out_wav), spoken], check=True, stdin=subprocess.DEVNULL)
            return eng
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"{eng} voice failed: {e}")
    raise RuntimeError(f"No text-to-speech engine worked. Last error: {last_err}")


# ---------------------------------------------------------------- clip processing

CLEAN_VOICE = ("highpass=f=80,afftdn=nf=-28:tn=1,deesser=i=0.3,equalizer=f=220:t=q:w=1:g=-1.2,"
               "acompressor=threshold=-22dB:ratio=2.4:attack=8:release=140,")
LEVEL = "loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.9:level=disabled"
TRIM = ("silenceremove=start_periods=1:start_threshold=-48dB:start_silence=0.05,areverse,"
        "silenceremove=start_periods=1:start_threshold=-48dB:start_silence=0.05,areverse,")


def process_clip(src: Path, dst: Path, user: bool) -> float:
    """Convert to 48 kHz mono, trim edge silence, even out loudness. Returns duration in seconds."""
    chain = TRIM + ("highpass=f=70,equalizer=f=180:t=q:w=1:g=1.2,equalizer=f=6000:t=q:w=1:g=-1," if not user else CLEAN_VOICE) + LEVEL + ",apad=pad_dur=0.06"
    try:
        ffmpeg("-i", str(src), "-ac", "1", "-ar", str(SR), "-af", chain, str(dst))
    except subprocess.CalledProcessError:
        ffmpeg("-i", str(src), "-ac", "1", "-ar", str(SR), "-af", TRIM + LEVEL, str(dst))  # plain chain if a filter is missing
    return sf.info(dst).duration


# ---------------------------------------------------------------- whisper and alignment

def whisper_ready() -> bool:
    """True once the speech model is on disk (checking does not load it)."""
    d = config.MODELS / "whisper"
    return d.exists() and any(d.rglob("model.bin"))


def whisper():
    global _whisper, _whisper_failed
    if _whisper is None and not _whisper_failed:
        try:
            from faster_whisper import WhisperModel

            _whisper = WhisperModel("base.en", device="cpu", compute_type="int8", download_root=str(config.MODELS / "whisper"))
        except Exception:  # noqa: BLE001
            _whisper_failed = True
    return _whisper


def _pcm16k(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, np.float32)


def transcribe(path: Path, prompt: str | None = None) -> list[tuple[str, float, float]]:
    """(word, start, end) for everything heard in the audio. Empty when whisper is unavailable."""
    m = whisper()
    if m is None:
        return []
    segs, _ = m.transcribe(_pcm16k(path), word_timestamps=True, initial_prompt=prompt, vad_filter=False)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in s.words if w.word.strip()]


def estimate_words(text: str, duration: float) -> list[dict]:
    """Word times spread over the clip by character count. Used when whisper is not available."""
    toks = text.split()
    weights = [len(t) + (4 if re.search(r"[,;:]$", t) else 8 if re.search(r"[.!?]$", t) else 0) + 2 for t in toks]
    total = sum(weights) or 1
    out, t = [], 0.0
    speech = max(0.1, duration - 0.1)
    for tok, w in zip(toks, weights):
        d = speech * w / total
        out.append({"w": tok, "s": round(t, 3), "e": round(t + d * 0.85, 3)})
        t += d
    return out


def align(wav: Path, text: str, duration: float) -> list[dict]:
    """One {w, s, e} per whitespace token of `text`. Matches what whisper heard; fills gaps proportionally."""
    toks = text.split()
    heard = transcribe(wav, prompt=text)
    if not heard:
        return estimate_words(text, duration)
    hits: list[tuple[float, float] | None] = []
    j = 0
    for tok in toks:
        n = norm(tok)
        hit = None
        if n:
            for k in range(j, min(j + 5, len(heard))):
                h = norm(heard[k][0])
                if h == n or (len(n) > 3 and (h.startswith(n[:4]) or n.startswith(h[:4])) and len(h) > 2):
                    hit = k
                    break
        if hit is None:
            hits.append(None)
        else:
            hits.append((heard[hit][1], heard[hit][2]))
            j = hit + 1
    out: list[dict] = [{"w": t, "s": 0.0, "e": 0.0} for t in toks]
    i = 0
    while i < len(toks):
        if hits[i] is not None:
            out[i].update(s=hits[i][0], e=hits[i][1])
            i += 1
            continue
        k = i
        while k < len(toks) and hits[k] is None:
            k += 1
        a = hits[i - 1][1] if i > 0 else 0.0
        b = hits[k][0] if k < len(toks) else max(a, duration - 0.05)
        weights = [len(toks[x]) + 1 for x in range(i, k)]
        tot, t = sum(weights), a
        for x, w in zip(range(i, k), weights):
            d = (b - a) * w / tot
            out[x].update(s=t, e=t + d * 0.9)
            t += d
        i = k
    prev = 0.0
    for o in out:  # monotonic, end after start
        o["s"] = round(max(o["s"], prev), 3)
        o["e"] = round(max(o["e"], o["s"] + 0.04), 3)
        prev = o["s"]
    return out


# ---------------------------------------------------------------- per-line pipeline

def _h(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def user_clip(job: Job, i: int) -> Path | None:
    p = job.dir / "narration" / "user" / f"line-{i}.wav"
    return p if p.exists() else None


def line_wav(job: Job, i: int) -> Path:
    return job.dir / "narration" / "lines" / f"line-{i}.wav"


def text_hash(text: str) -> str:
    return _h(script.clean(text))


def tts_speed(job: Job, cfg: dict) -> float:
    """Voice speed for synthetic clips: the voice setting times the pace's multiplier, kept in a range the models handle well."""
    from . import config

    return round(min(1.35, float(cfg["speed"]) * config.pace(job.load()["brief"])["voice"]), 3)


def is_fresh(job: Job, i: int, text: str) -> bool:
    """True when line i has a clip that matches its text (and, for synthetic clips, the current voice, speed and pace)."""
    state = job.load()
    rec = state["narration"]["clips"].get(str(i), {})
    if rec.get("thash") != text_hash(text) or not line_wav(job, i).exists():
        return False
    if rec.get("source") == "tts":
        cfg = state["narration"]
        return rec.get("ahash") == _h("tts", cfg["engine"], cfg["voice"], tts_speed(job, cfg), script.clean(text))
    return True


def build_line(job: Job, i: int, text: str, cfg: dict, log=print) -> dict:
    """Make sure line i has a processed clip and word times for `text`. Returns the clip record."""
    text = script.clean(text)
    clips = job.load()["narration"]["clips"]
    old = clips.get(str(i), {})
    dst = line_wav(job, i)
    dst.parent.mkdir(parents=True, exist_ok=True)
    user = user_clip(job, i)
    th = text_hash(text)
    if user:
        ah = _h("user", user.stat().st_size, int(user.stat().st_mtime))
        if old.get("source") == "user" and old.get("ahash") == ah and dst.exists():
            dur = old["duration"]
        else:
            dur = process_clip(user, dst, user=True)
            old = {}
        if old.get("source") == "user" and old.get("thash") == th and old.get("words"):
            words = old["words"]
        else:
            words = align(dst, text, dur)
        return {"source": "user", "ahash": ah, "thash": th, "duration": round(dur, 3), "words": words}
    speed = tts_speed(job, cfg)
    ah = _h("tts", cfg["engine"], cfg["voice"], speed, text)
    if old.get("source") == "tts" and old.get("ahash") == ah and dst.exists() and old.get("words"):
        return old
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw.wav"
        used = synthesize(text, cfg["engine"], cfg["voice"], speed, raw, log)
        dur = process_clip(raw, dst, user=False)
    return {"source": "tts", "engine": used, "ahash": ah, "thash": th, "duration": round(dur, 3), "words": align(dst, text, dur)}


def run_narration(job: Job, only: list[int] | None = None) -> None:
    sb = job.storyboard()
    if not sb:
        raise RuntimeError("Write or generate the script first")
    cfg = job.load()["narration"]
    lines = sb["lines"]
    todo = list(range(len(lines))) if only is None else [i for i in only if 0 <= i < len(lines)]
    for n, i in enumerate(todo):
        job.check()
        job.progress("narration", n / max(1, len(todo)), f"Line {i + 1} of {len(lines)}")
        rec = build_line(job, i, lines[i]["text"], cfg, lambda m: job.log("narration", m))

        def put(s, i=i, rec=rec):
            s["narration"]["clips"][str(i)] = rec
        job.update(put)
        job.log("narration", f"line {i + 1}: {rec['source']} {rec['duration']:.1f}s")
    # drop clips for lines that no longer exist
    def trim(s):
        s["narration"]["clips"] = {k: v for k, v in s["narration"]["clips"].items() if int(k) < len(lines)}
    job.update(trim)


# ---------------------------------------------------------------- user audio

def save_user_clip(job: Job, i: int, upload: Path) -> None:
    dst = job.dir / "narration" / "user" / f"line-{i}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg("-i", str(upload), "-ac", "1", "-ar", str(SR), str(dst))
    except subprocess.CalledProcessError as e:
        raise ValueError("That audio file could not be read") from e
    if sf.info(dst).duration < 0.3:
        dst.unlink()
        raise ValueError("That recording is too short")


def clear_user_clip(job: Job, i: int) -> None:
    p = job.dir / "narration" / "user" / f"line-{i}.wav"
    p.unlink(missing_ok=True)
    job.update(lambda s: s["narration"]["clips"].pop(str(i), None))


def _match_lines(heard: list[tuple[str, float, float]], lines: list[str]) -> list[tuple[float, float] | None]:
    """Find each script line in the transcript of one long take, in order. Later repeats win (retakes)."""
    from difflib import SequenceMatcher

    H = [norm(w) for w, _, _ in heard]
    result: list[tuple[float, float] | None] = []
    cursor = 0
    for text in lines:
        L = [norm(t) for t in text.split() if norm(t)]
        n = len(L)
        best = (0.0, -1, -1)
        for s in range(cursor, len(H)):
            for ln in range(max(1, n - 2), n + 3):
                if s + ln > len(H):
                    break
                score = SequenceMatcher(None, H[s:s + ln], L, autojunk=False).ratio()
                if score >= best[0] - 1e-9 and score > 0.55:
                    best = (score, s, s + ln)
        if best[1] < 0:
            result.append(None)
            continue
        result.append((heard[best[1]][1], heard[best[2] - 1][2]))
        cursor = best[2]
    return result


def split_take(job: Job, audio: Path, lines: list[str]) -> dict:
    """Cut one long recording into per-line clips by matching the script against a transcript."""
    wav = job.dir / "narration" / "take.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-i", str(audio), "-ac", "1", "-ar", str(SR), str(wav))
    heard = transcribe(wav)
    if not heard:
        raise RuntimeError("Could not transcribe the recording (the speech model is unavailable). Record or upload line by line instead.")
    spans = _match_lines(heard, lines)
    data, sr = sf.read(wav, dtype="float32")
    matched, missing = [], []
    for i, span in enumerate(spans):
        if span is None:
            missing.append(i)
            continue
        prev_end = max((s[1] for s in spans[:i] if s), default=0.0)
        nxt = next((s[0] for s in spans[i + 1:] if s), len(data) / sr)
        a = max(prev_end + 0.02, span[0] - 0.2)
        b = min(nxt - 0.02, span[1] + 0.3, len(data) / sr)
        dst = job.dir / "narration" / "user" / f"line-{i}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, data[int(a * sr):int(b * sr)], sr)
        matched.append(i)
    return {"matched": matched, "missing": missing}


def split_by_marks(job: Job, audio: Path, marks: list[dict], n_lines: int) -> dict:
    """Cut one take using the moments the speaker pressed 'next line'. marks: [{line, t}] in seconds, in time order.
    A line that was redone has several marks. The last one counts. A line runs until the next mark of any line."""
    wav = job.dir / "narration" / "take.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-i", str(audio), "-ac", "1", "-ar", str(SR), str(wav))
    data, sr = sf.read(wav, dtype="float32")
    total = len(data) / sr
    events = sorted((float(m["t"]), int(m["line"])) for m in marks if isinstance(m, dict) and 0 <= int(m.get("line", -1)) < n_lines)
    matched, missing = [], []
    for i in range(n_lines):
        idx = [k for k, (_, ln) in enumerate(events) if ln == i]
        if not idx:
            missing.append(i)
            continue
        k = idx[-1]
        a = max(0.0, events[k][0] - 0.1)
        b = min(total, (events[k + 1][0] if k + 1 < len(events) else total) + 0.15)
        if b - a < 0.4:
            missing.append(i)
            continue
        dst = job.dir / "narration" / "user" / f"line-{i}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, data[int(a * sr):int(b * sr)], sr)
        matched.append(i)
    return {"matched": matched, "missing": missing}


def transcript_lines(job: Job, audio: Path, max_words: int = 34) -> list[str]:
    """Narration lines from a recording with no script. Cuts the clips too (line-N.wav in narration/user)."""
    wav = job.dir / "narration" / "take.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-i", str(audio), "-ac", "1", "-ar", str(SR), str(wav))
    heard = transcribe(wav)
    if not heard:
        raise RuntimeError("Could not transcribe the recording (the speech model is unavailable). Paste the script text instead.")
    groups: list[list[tuple[str, float, float]]] = []
    cur: list[tuple[str, float, float]] = []
    for k, (w, s, e) in enumerate(heard):
        cur.append((w, s, e))
        gap = heard[k + 1][1] - e if k + 1 < len(heard) else 9
        if (re.search(r"[.!?]$", w) and (len(cur) >= 8 or gap > 0.5)) or len(cur) >= max_words or (gap > 1.1 and len(cur) >= 6):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    data, sr = sf.read(wav, dtype="float32")
    udir = job.dir / "narration" / "user"
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    texts = []
    for i, g in enumerate(groups):
        a = max(0.0, g[0][1] - 0.15)
        b = min(len(data) / sr, g[-1][2] + 0.25)
        if i + 1 < len(groups):
            b = min(b, (g[-1][2] + groups[i + 1][0][1]) / 2 + 0.05)
        sf.write(udir / f"line-{i}.wav", data[int(a * sr):int(b * sr)], sr)
        texts.append(script.clean(" ".join(w for w, _, _ in g)))
    return texts


def preview_voice(engine: str, voice: str, speed: float, text: str | None = None) -> Path:
    text = text or "Attention is all you need. That was the title, and it turned out to be an understatement."
    key = _h(engine, voice, speed, text)
    out = config.CACHE / f"preview-{key}.wav"
    if not out.exists():
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw.wav"
            synthesize(text, engine, voice, speed, raw)
            process_clip(raw, out, user=False)
    return out
