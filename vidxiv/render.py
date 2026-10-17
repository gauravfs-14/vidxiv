"""Turn a storyboard and narration clips into render props, then drive Remotion and master the audio."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import soundfile as sf

from . import config, review, schema, script, voice
from .jobs import Job, slug

REMOTION_SFX = ("whoosh", "whip", "page-turn", "switch", "mouse-click", "ding")
SFX_OWN = ("pop", "pencil", "tick", "swell", "soft", "chime", "thud", "blip", "blip-lo", "blip-hi", "mask", "rise", "buzz", "snap")
SPEECH_WPS = 2.6  # words per second, only for estimates before narration exists


# ---------------------------------------------------------------- setup

def engine_ready() -> bool:
    return (config.ENGINE / "node_modules" / "remotion" / "package.json").exists()


def node_version() -> str | None:
    if not shutil.which("node"):
        return None
    try:
        return subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10).stdout.strip().lstrip("v") or None
    except Exception:  # noqa: BLE001
        return None


def node_ok() -> bool:
    """Node 20 or newer. Older versions run but Remotion does not support them."""
    v = node_version()
    try:
        return bool(v) and int(v.split(".")[0]) >= 20
    except ValueError:
        return False


def node_tools() -> dict:
    return {"node": node_ok(), "npm": bool(shutil.which("npm")), "ffmpeg": bool(shutil.which("ffmpeg")),
            "engine": engine_ready(), "nodeVersion": node_version()}


def ensure_engine(log=print) -> None:
    if engine_ready():
        return
    if not shutil.which("npm"):
        raise RuntimeError("Node.js is required to render video. Install it from https://nodejs.org and try again.")
    log("Installing the video engine (npm install, one time)")
    lock = config.ENGINE / "package-lock.json"
    cmd = ["npm", "ci" if lock.exists() else "install", "--no-audit", "--no-fund"]
    p = subprocess.run(cmd, cwd=config.ENGINE, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RuntimeError("npm install failed: " + (p.stderr or p.stdout)[-600:])
    log("Video engine installed")


def ensure_sfx(log=print) -> Path:
    """Sound effects live in cache/sfx once generated or downloaded."""
    d = config.CACHE / "sfx"
    d.mkdir(exist_ok=True)
    work = config.CACHE / "sfx-build"
    (work / "public" / "audio").mkdir(parents=True, exist_ok=True)
    if not (d / "soft.wav").exists() and shutil.which("node"):
        log("Generating sound effects")
        subprocess.run(["node", str(config.ASSETS / "sfx-extra.mjs")], cwd=work, capture_output=True, stdin=subprocess.DEVNULL)
        for f in (work / "public" / "audio").glob("*.wav"):
            shutil.copy(f, d / f.name)
    for name in REMOTION_SFX:
        if (d / f"{name}.wav").exists():
            continue
        try:
            r = requests.get(f"https://remotion.media/{name}.wav", timeout=20)
            if r.ok and r.content[:4] == b"RIFF":
                (d / f"{name}.wav").write_bytes(r.content)
        except Exception:  # noqa: BLE001
            log(f"Could not fetch {name}.wav, the video will play without it")
    return d


# ---------------------------------------------------------------- props

def _estimate(text: str) -> float:
    return max(1.6, len(text.split()) / SPEECH_WPS + 0.3)


def build_props(job: Job, estimated_ok: bool) -> tuple[dict, dict[int, Path]]:
    """Returns (props, {line index: clip path}). With estimated_ok, lines without clips get estimated timing and no audio."""
    state = job.load()
    sb = job.storyboard()
    if not sb:
        raise RuntimeError("There is no storyboard yet")
    brief = state["brief"]
    clips = state["narration"]["clips"]
    figs = {f["id"]: f for f in job.figures()}
    pace = config.pace(brief)
    lines, cursor, files = [], pace["lead"], {}
    for i, ln in enumerate(sb["lines"]):
        text = ln["text"]
        c = clips.get(str(i))
        fresh = c and c.get("thash") == voice.text_hash(text) and voice.line_wav(job, i).exists()  # pace changes re-synthesize in render()
        if fresh:
            dur_s, words, file = c["duration"], c["words"], f"vo/line{i}.wav"
            files[i] = voice.line_wav(job, i)
        elif estimated_ok:
            dur_s = _estimate(text)
            words, file = voice.estimate_words(text, dur_s), None
        else:
            raise RuntimeError(f"Line {i + 1} has no narration yet")
        dur = math.ceil(dur_s * config.FPS)
        gap = ln.get("gapAfter")
        gap = pace["gap"] if gap is None else (gap if pace["name"] == "standard" else min(gap, pace["gap"] * 2))
        lines.append({"index": i, "text": text, "file": file, "from": cursor, "dur": dur, "words": words,
                      "scene": ln["scene"], "highlight": ln.get("highlight", {}), "kicker": ln.get("kicker"), "gapAfter": gap})
        cursor += dur + gap
    total = lines[-1]["from"] + lines[-1]["dur"] + pace["tail"]
    scenes = []
    for i, ln in enumerate(lines):
        start = 0 if i == 0 else ln["from"] - pace["pre"]
        end = total if i == len(lines) - 1 else lines[i + 1]["from"] - pace["pre"]
        scenes.append({"from": start, "dur": end - start + (0 if i == len(lines) - 1 else pace["overlap"])})
    used = {s["scene"].get("figure") for s in lines}
    figures = {fid: {"file": f"figures/{f['file']}", "width": f["width"], "height": f["height"], "caption": f.get("caption", "")}
               for fid, f in figs.items() if fid in used or len(figs) <= 12}
    music_mood = brief.get("music", "warm")
    props = {
        "fps": config.FPS, "total": total, "title": sb["title"], "look": resolve_looks(brief, sb)[0],
        "captions": bool(brief.get("captions", True)), "progressBar": True,
        "pace": {k: pace[k] for k in ("name", "speed", "words", "pre", "overlap")},
        "music": None, "sfx": [], "figures": figures, "lines": lines, "scenes": scenes,
        "_musicMood": music_mood,
    }
    return props, files


def assemble_public(job: Job, props: dict, files: dict[int, Path], log=print) -> None:
    """Fill <job>/public with this render's assets and finish the props (music, sfx lists)."""
    pub = job.dir / "public"
    for sub in ("vo", "figures", "audio"):
        shutil.rmtree(pub / sub, ignore_errors=True)
        (pub / sub).mkdir(parents=True, exist_ok=True)
    for i, src in files.items():
        shutil.copy(src, pub / "vo" / f"line{i}.wav")
    for f in job.figures():
        shutil.copy(job.dir / "paper" / "figures" / f["file"], pub / "figures" / f["file"])
    sfx_dir = ensure_sfx(log)
    for f in sfx_dir.glob("*.wav"):
        shutil.copy(f, pub / "audio" / f.name)
    mood = props.pop("_musicMood")
    if mood != "none" and shutil.which("node"):
        seconds = math.ceil(props["total"] / config.FPS) + 3
        log(f"Composing {mood} music bed ({seconds}s)")
        p = subprocess.run(["node", str(config.ASSETS / "music.mjs"), str(job.dir), "--mood", mood, "--duration", str(seconds)],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if (pub / "audio" / "music.wav").exists():
            props["music"] = {"file": "audio/music.wav", "volume": 0.5}
        else:
            log("Music generation failed, continuing without it: " + p.stderr[-200:])
    props["sfx"] = sorted(f.stem for f in (pub / "audio").glob("*.wav") if f.stem != "music")


# ---------------------------------------------------------------- remotion

def _npx(job: Job, stage: str, args: list[str], on_line=None, timeout=3600) -> None:
    code = job.run(["npx", "--no-install", "remotion", *args], stage, cwd=config.ENGINE, on_line=on_line, timeout=timeout)
    if code != 0:
        tail = " ".join(job.read_log(stage)["lines"][-4:])
        raise RuntimeError(f"Remotion failed (exit {code}): {tail[-400:]}")


def bundle(job: Job, stage: str, progress: float = 0.05) -> Path:
    out = job.dir / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    job.progress(stage, progress, "Bundling the video engine")
    _npx(job, stage, ["bundle", "src/index.ts", f"--out-dir={out}", f"--public-dir={job.dir / 'public'}"])
    return out


def prepare(job: Job, stage: str, estimated_ok: bool) -> Path:
    ensure_engine(lambda m: job.log(stage, m))
    props, files = build_props(job, estimated_ok)
    assemble_public(job, props, files, lambda m: job.log(stage, m))
    pf = job.dir / "render.json"
    pf.write_text(json.dumps(props))
    return pf


def preview(job: Job) -> None:
    """One still per scene using estimated timing where narration is missing. Shows layout, not pacing."""
    stage = "preview"
    fmt = (job.load()["brief"].get("formats") or ["Vertical"])[0]
    props_file = prepare(job, stage, estimated_ok=True)
    props = json.loads(props_file.read_text())
    b = bundle(job, stage)
    out = job.dir / "previews" / fmt.lower()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    n = len(props["lines"])
    done = [0]

    def one(i: int) -> str:
        ln = props["lines"][i]
        frame = min(props["total"] - 1, ln["from"] + int(ln["dur"] * 0.8))
        target = out / f"scene-{i}.png"
        job.check()
        _npx(job, stage, ["still", str(b), fmt, str(target), f"--props={props_file}", f"--frame={frame}", "--scale=0.4", "--log=error"],
             on_line=lambda s: None, timeout=300)
        done[0] += 1
        job.progress(stage, 0.15 + 0.85 * done[0] / n, f"Scene {done[0]} of {n}")
        return f"previews/{fmt.lower()}/scene-{i}.png"

    with ThreadPoolExecutor(max_workers=3) as ex:
        files = list(ex.map(one, range(n)))
    shutil.rmtree(job.dir / "bundle", ignore_errors=True)
    job.patch(previews=files)


def master(src: Path, dst: Path) -> None:
    """Two-pass loudnorm to -14 LUFS, true peak -1.5 dBFS. Video stream is copied."""
    meas = subprocess.run(["ffmpeg", "-nostats", "-i", str(src), "-vn", "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL).stderr
    try:
        m = json.loads(meas[meas.rindex("{"): meas.rindex("}") + 1])
        af = (f"loudnorm=I=-14:TP=-1.5:LRA=7:linear=true:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
              f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']},alimiter=limit=0.83:level=disabled")
    except (ValueError, KeyError):
        af = "loudnorm=I=-14:TP=-1.5:LRA=7"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "copy", "-af", af, "-ar", "48000", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", str(dst)], check=True, stdin=subprocess.DEVNULL)


def ensure_narrated(job: Job, stage: str) -> None:
    """Narrate lines that have no clip, a clip for older text, or a synthetic clip made at another speed or pace."""
    sb = job.storyboard()
    stale = [i for i, ln in enumerate(sb["lines"]) if not voice.is_fresh(job, i, ln["text"])]
    if stale:
        job.progress(stage, 0.02, f"Narrating {len(stale)} line(s)")
        voice.run_narration(job, stale)


def resolve_looks(brief: dict, sb: dict) -> list[str]:
    """Looks to render. 'auto' means the one the storyboard was written for. Order kept, duplicates dropped."""
    wanted = brief.get("looks") or [brief.get("look", "auto")]
    out: list[str] = []
    for look in wanted:
        look = sb.get("look", "studio") if look == "auto" else look
        if look in schema.LOOKS and look not in out:
            out.append(look)
    return out or [sb.get("look", "studio")]


def render_draft(job: Job, stage: str, lo: float = 0.0, hi: float = 1.0) -> dict:
    """A half-size, low-quality render of the first selected look and format, for review. Not mastered, not kept as an output."""
    state = job.load()
    sb = job.storyboard()
    fmt = ([f for f in state["brief"].get("formats", ["Vertical"]) if f in config.FORMATS] or ["Vertical"])[0]
    look = resolve_looks(state["brief"], sb)[0]
    at = lambda v: lo + (hi - lo) * v  # noqa: E731
    job.progress(stage, at(0.02), "Preparing a draft to review")
    props = json.loads(prepare(job, stage, estimated_ok=False).read_text())
    b = bundle(job, stage, at(0.06))
    out = job.dir / "review" / "draft.mp4"
    out.parent.mkdir(exist_ok=True)
    pf = job.dir / "render-draft.json"
    pf.write_text(json.dumps({**props, "look": look}))

    def on_line(line: str) -> None:
        m = re.search(r"(\d+)/(\d+)", line)
        if m and int(m.group(2)) > 1:
            job.progress(stage, at(0.1 + 0.9 * int(m.group(1)) / int(m.group(2))), f"Rendering a {look} {fmt.lower()} draft: frame {m.group(1)} of {m.group(2)}")
        elif line.strip():
            job.log(stage, line)

    try:
        _npx(job, stage, ["render", str(b), fmt, str(out), f"--props={pf}", "--codec=h264", "--crf=30", "--scale=0.5", "--audio-bitrate=160k", "--log=info"], on_line=on_line)
    finally:
        shutil.rmtree(b, ignore_errors=True)
    return {"file": out, "props": {**props, "look": look}, "format": fmt, "look": look}


def render(job: Job, lo: float = 0.0, hi: float = 1.0) -> None:
    """Render every selected look in every selected format. Earlier renders are kept: each run adds new files.
    lo and hi place this render's progress inside the whole stage, which a review before it may already have used."""
    import time

    stage = "render"
    at = lambda v: lo + (hi - lo) * v  # noqa: E731
    state = job.load()
    formats = [f for f in state["brief"].get("formats", ["Vertical"]) if f in config.FORMATS] or ["Vertical"]
    sb = job.storyboard()
    if not sb:
        raise RuntimeError("There is no storyboard yet")
    looks = resolve_looks(state["brief"], sb)
    ensure_narrated(job, stage)
    props_file = prepare(job, stage, estimated_ok=False)
    props = json.loads(props_file.read_text())
    b = bundle(job, stage, at(0.05))
    out_dir = job.dir / "out"
    out_dir.mkdir(exist_ok=True)
    run = 1 + max((o.get("run", 0) for o in job.load().get("outputs", [])), default=0)
    combos = [(look, fmt) for look in looks for fmt in formats]
    for k, (look, fmt) in enumerate(combos):
        base = 0.1 + 0.88 * k / len(combos)
        span = 0.88 / len(combos)
        label = f"{look} {fmt.lower()}"
        stem = f"{slug(sb['title'])}-{look}-{fmt.lower()}-r{run}"
        pf = job.dir / f"render-{look}.json"
        pf.write_text(json.dumps({**props, "look": look}))
        raw = out_dir / f"raw-{stem}.mp4"
        job.progress(stage, at(base), f"Rendering {label}")

        def on_line(line: str, base=base, span=span, label=label) -> None:
            m = re.search(r"(\d+)/(\d+)", line)
            if m and int(m.group(2)) > 1:
                job.progress(stage, at(base + span * 0.92 * int(m.group(1)) / int(m.group(2))), f"Rendering {label}: frame {m.group(1)} of {m.group(2)}")
            elif line.strip():
                job.log(stage, line)

        _npx(job, stage, ["render", str(b), fmt, str(raw), f"--props={pf}", "--codec=h264", "--crf=22",
                          "--audio-bitrate=320k", "--log=info"], on_line=on_line)
        job.progress(stage, at(base + span * 0.94), f"Mastering {label} audio")
        final = out_dir / f"{stem}.mp4"
        master(raw, final)
        raw.unlink(missing_ok=True)
        poster = final.with_suffix(".jpg")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "2", "-i", str(final), "-frames:v", "1", "-vf", "scale=480:-1", str(poster)],
                       stdin=subprocess.DEVNULL)
        seconds = props["total"] / config.FPS
        qa = [f"{i['kind']}: {i['detail']}" for i in review.check_video(final, seconds)]
        for q in qa:
            job.log(stage, f"Check on {label}: {q}")
        entry = {"id": stem, "run": run, "look": look, "format": fmt, "file": f"out/{final.name}",
                 "poster": f"out/{poster.name}" if poster.exists() else None, "size": final.stat().st_size,
                 "seconds": round(seconds, 1), "created": time.time(), "qa": qa}
        job.update(lambda s, e=entry: s["outputs"].append(e))  # saved as soon as it exists
    shutil.rmtree(b, ignore_errors=True)


def delete_output(job: Job, output_id: str) -> None:
    def drop(s):
        keep = []
        for o in s.get("outputs", []):
            if o.get("id", o["file"]) == output_id:
                for f in (o["file"], o.get("poster")):
                    if f:
                        (job.dir / f).unlink(missing_ok=True)
            else:
                keep.append(o)
        s["outputs"] = keep
    job.update(drop)
