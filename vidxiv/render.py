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

from . import config, script, voice
from .jobs import Job, slug

REMOTION_SFX = ("whoosh", "whip", "page-turn", "switch", "mouse-click", "ding")
SFX_OWN = ("pop", "pencil", "tick", "swell", "soft", "chime", "thud", "blip", "blip-lo", "blip-hi", "mask", "rise", "buzz", "snap")
SPEECH_WPS = 2.6  # words per second, only for estimates before narration exists


# ---------------------------------------------------------------- setup

def engine_ready() -> bool:
    return (config.ENGINE / "node_modules" / "remotion" / "package.json").exists()


def node_tools() -> dict:
    return {"node": bool(shutil.which("node")), "npm": bool(shutil.which("npm")), "ffmpeg": bool(shutil.which("ffmpeg")),
            "engine": engine_ready()}


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
    lines, cursor, files = [], config.LEAD, {}
    for i, ln in enumerate(sb["lines"]):
        text = ln["text"]
        c = clips.get(str(i))
        fresh = c and c.get("thash") == voice.text_hash(text) and voice.line_wav(job, i).exists()
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
        gap = config.DEFAULT_GAP if gap is None else gap
        lines.append({"index": i, "text": text, "file": file, "from": cursor, "dur": dur, "words": words,
                      "scene": ln["scene"], "highlight": ln.get("highlight", {}), "kicker": ln.get("kicker"), "gapAfter": gap})
        cursor += dur + gap
    total = lines[-1]["from"] + lines[-1]["dur"] + config.TAIL
    scenes = []
    for i, ln in enumerate(lines):
        start = 0 if i == 0 else ln["from"] - 10
        end = total if i == len(lines) - 1 else lines[i + 1]["from"] - 10
        scenes.append({"from": start, "dur": end - start + (0 if i == len(lines) - 1 else config.OVERLAP)})
    used = {s["scene"].get("figure") for s in lines}
    figures = {fid: {"file": f"figures/{f['file']}", "width": f["width"], "height": f["height"], "caption": f.get("caption", "")}
               for fid, f in figs.items() if fid in used or len(figs) <= 12}
    music_mood = brief.get("music", "warm")
    props = {
        "fps": config.FPS, "total": total, "title": sb["title"], "look": sb.get("look", "studio") if brief.get("look", "auto") == "auto" else brief["look"],
        "captions": bool(brief.get("captions", True)), "progressBar": True,
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


def bundle(job: Job, stage: str) -> Path:
    out = job.dir / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    job.progress(stage, 0.05, "Bundling the video engine")
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


def render(job: Job) -> None:
    stage = "render"
    state = job.load()
    formats = [f for f in state["brief"].get("formats", ["Vertical"]) if f in config.FORMATS] or ["Vertical"]
    sb = job.storyboard()
    if not sb:
        raise RuntimeError("There is no storyboard yet")
    # narrate anything missing or stale
    stale = [i for i, ln in enumerate(sb["lines"])
             if state["narration"]["clips"].get(str(i), {}).get("thash") != voice.text_hash(ln["text"]) or not voice.line_wav(job, i).exists()]
    if stale:
        job.progress(stage, 0.02, f"Narrating {len(stale)} line(s)")
        voice.run_narration(job, stale)
    props_file = prepare(job, stage, estimated_ok=False)
    props = json.loads(props_file.read_text())
    b = bundle(job, stage)
    out_dir = job.dir / "out"
    out_dir.mkdir(exist_ok=True)
    outputs = []
    for k, fmt in enumerate(formats):
        base = 0.1 + 0.85 * k / len(formats)
        span = 0.85 / len(formats)
        raw = out_dir / f"raw-{fmt.lower()}.mp4"
        name = f"{slug(sb['title'])}-{fmt.lower()}.mp4"
        job.progress(stage, base, f"Rendering {fmt}")

        def on_line(line: str, base=base, span=span, fmt=fmt) -> None:
            m = re.search(r"(\d+)/(\d+)", line)
            if m and int(m.group(2)) > 1:
                job.progress(stage, base + span * 0.92 * int(m.group(1)) / int(m.group(2)), f"Rendering {fmt}: frame {m.group(1)} of {m.group(2)}")
            elif line.strip():
                job.log(stage, line)

        _npx(job, stage, ["render", str(b), fmt, str(raw), f"--props={props_file}", "--codec=h264", "--crf=22",
                          "--audio-bitrate=320k", "--log=info"], on_line=on_line)
        job.progress(stage, base + span * 0.94, f"Mastering {fmt} audio")
        final = out_dir / name
        master(raw, final)
        raw.unlink(missing_ok=True)
        poster = final.with_suffix(".jpg")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "2", "-i", str(final), "-frames:v", "1", "-vf", "scale=480:-1", str(poster)],
                       stdin=subprocess.DEVNULL)
        outputs.append({"format": fmt, "file": f"out/{name}", "poster": f"out/{poster.name}" if poster.exists() else None,
                        "size": final.stat().st_size, "seconds": round(props["total"] / config.FPS, 1), "created": __import__("time").time()})
    shutil.rmtree(b, ignore_errors=True)
    job.patch(outputs=outputs)
