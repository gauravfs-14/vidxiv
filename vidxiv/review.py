"""Check a draft render before the final one: mechanical checks on voice and audio, contact sheets of the frames, and the prompt for the agent that fixes what it sees.

Nothing here renders or edits the storyboard. pipeline.py runs the loop: draft, analyze, ask the agent, apply the fixes, repeat.
An issue is {"kind": voice|audio|video|sfx|music|visual, "line": index or None, "detail": text, "fixable": bool}. `fixable` means the
agent can act on it by editing the storyboard. Everything else is reported to the user.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config, voice
from .jobs import Job

WPS_RANGE = (1.0, 5.8)  # spoken words per second outside this means a clipped, stretched or garbled clip
HEARD_MIN = 0.6  # share of the written words that speech recognition must find in a clip
SHEET_CELLS = 8
LAYOUT = {"portrait": (4, 2, 380), "landscape": (2, 4, 620), "square": (4, 2, 420)}  # columns, rows, cell width per sheet


def issue(kind: str, detail: str, line: int | None = None, fixable: bool = False) -> dict:
    return {"kind": kind, "line": line, "detail": detail, "fixable": fixable}


# ---------------------------------------------------------------- voice

def heard_ratio(text: str, heard: list[str]) -> float:
    """Share of the written words that appear in what was heard, each heard word used once."""
    want = [w for w in (voice.norm(t) for t in text.split()) if w]
    pool = [w for w in (voice.norm(t) for t in heard) if w]
    if not want:
        return 1.0
    hit = 0
    for w in want:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / len(want)


def check_voice(job: Job, log=print) -> list[dict]:
    """Per-line checks on the narration clips. Recorded lines are checked too, but only synthetic ones are fixable (by rewording)."""
    sb = job.storyboard() or {"lines": []}
    clips = job.load()["narration"]["clips"]
    listen = voice.whisper_ready()
    if not listen:
        log("Speech recognition is not installed, so the voice is checked for timing and level only")
    out: list[dict] = []
    for i, ln in enumerate(sb["lines"]):
        rec = clips.get(str(i))
        wav = voice.line_wav(job, i)
        if not rec or not wav.exists():
            out.append(issue("voice", "has no narration clip", i))
            continue
        fix = rec.get("source") == "tts"
        words = len(ln["text"].split())
        dur = float(rec.get("duration") or 0)
        wps = words / dur if dur else 0
        if wps and not WPS_RANGE[0] <= wps <= WPS_RANGE[1] and words > 3:
            out.append(issue("voice", f"speaks {wps:.1f} words per second ({words} words in {dur:.1f}s), which sounds {'rushed' if wps > WPS_RANGE[1] else 'dragged or cut off'}", i, fix))
        try:
            data, _ = sf.read(wav, always_2d=False)
            data = data if data.ndim == 1 else data.mean(axis=1)
            peak, rms = float(np.abs(data).max()), float(np.sqrt(np.mean(data**2)))
            if peak >= 0.999:
                out.append(issue("voice", "the clip is clipped (distorted at peak level)", i, fix))
            if rms < 0.004:
                out.append(issue("voice", "the clip is almost silent", i, fix))
        except Exception as e:  # noqa: BLE001
            out.append(issue("voice", f"the clip cannot be read ({e})", i, fix))
            continue
        if listen and words > 2:
            heard = [w for w, _, _ in voice.transcribe(wav)]
            ratio = heard_ratio(ln["text"], heard)
            if ratio < HEARD_MIN:
                out.append(issue("voice", f"speech recognition heard only {ratio:.0%} of the words. It heard: \"{' '.join(heard)[:160]}\"", i, fix))
    return out


# ---------------------------------------------------------------- audio and video

def _probe(path: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {}


def check_video(path: Path, expected_seconds: float) -> list[dict]:
    """File-level checks on a rendered MP4: length, audio track, black frames, near-silence and long silent gaps."""
    info = _probe(path)
    out: list[dict] = []
    try:
        seconds = float(info["format"]["duration"])
    except (KeyError, ValueError):
        return [issue("video", "the file cannot be read")]
    if abs(seconds - expected_seconds) > 0.8:
        out.append(issue("video", f"is {seconds:.1f}s long but the timeline is {expected_seconds:.1f}s"))
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        return out + [issue("audio", "has no audio track")]
    p = subprocess.run(["ffmpeg", "-nostats", "-v", "info", "-i", str(path), "-vf", "blackdetect=d=0.35:pix_th=0.03",
                        "-af", "volumedetect,silencedetect=noise=-50dB:d=2", "-f", "null", "-"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    log = p.stderr
    for m in re.finditer(r"black_start:([\d.]+) black_end:([\d.]+)", log):
        out.append(issue("video", f"has an empty black frame from {float(m.group(1)):.1f}s to {float(m.group(2)):.1f}s"))
    mean = re.search(r"mean_volume: (-?[\d.]+) dB", log)
    if mean and float(mean.group(1)) < -45:
        out.append(issue("audio", f"is almost silent (mean level {mean.group(1)} dB)"))
    for m in re.finditer(r"silence_start: ([\d.]+)", log):
        end = re.search(r"silence_end: ([\d.]+)", log[m.end():])
        a = float(m.group(1))
        b = float(end.group(1)) if end else seconds
        if b - a >= 2 and a > 0.5 and b < seconds - 1:
            out.append(issue("audio", f"is silent from {a:.1f}s to {b:.1f}s"))
    return out[:8]


def check_assets(props: dict, brief: dict) -> list[dict]:
    """Things the render depends on that can fail quietly: no sound effects, or the music bed did not get made."""
    out: list[dict] = []
    if not props.get("sfx"):
        out.append(issue("sfx", "no sound effect files are available, so the video plays without them (check the network and node, then render again)"))
    if brief.get("music", "warm") != "none" and not props.get("music"):
        out.append(issue("music", "the music bed could not be made, so the video plays without it"))
    return out


# ---------------------------------------------------------------- contact sheets

def _frame(video: Path, t: float, dst: Path, width: int) -> bool:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)],
                   stdin=subprocess.DEVNULL)
    return dst.exists() and dst.stat().st_size > 0


def contact_sheets(video: Path, props: dict, orientation: str, out_dir: Path) -> tuple[list[Path], list[str]]:
    """Tile frames from the draft into a few images. Each line gets an early frame (reveals starting) and a late one (everything shown).
    Returns the sheet paths and one legend entry per sheet that says which cell is which."""
    import shutil

    cols, rows, width = LAYOUT[orientation]
    shutil.rmtree(out_dir, ignore_errors=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True)
    fps = props["fps"]
    lines = props["lines"]
    late_only = len(lines) * 2 > SHEET_CELLS * 9  # very long videos: one frame per line keeps the review affordable
    cells: list[tuple[Path, str]] = []
    for ln in lines:
        stops = [("late", 0.92)] if late_only else [("early", 0.3), ("late", 0.92)]
        for tag, at in stops:
            dst = cells_dir / f"c{len(cells):04d}.jpg"
            if _frame(video, (ln["from"] + ln["dur"] * at) / fps, dst, width):
                cells.append((dst, f"line {ln['index'] + 1} {tag} ({ln['scene'].get('type', 'scene')})"))
    sheets, legend = [], []
    for s in range(0, len(cells), cols * rows):
        part = cells[s: s + cols * rows]
        for k, (src, _) in enumerate(part):
            shutil.copy(src, cells_dir / f"s{s // (cols * rows)}-{k:02d}.jpg")
        dst = out_dir / f"sheet-{s // (cols * rows) + 1}.jpg"
        p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-i", str(cells_dir / f"s{s // (cols * rows)}-%02d.jpg"), "-frames:v", "1",
                            "-vf", f"tile={cols}x{rows}:padding=8:margin=8:color=0x808080", "-q:v", "4", str(dst)], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if dst.exists():
            sheets.append(dst)
            legend.append(f"{dst.name} ({cols} columns, {rows} rows, read left to right, top to bottom): " + "; ".join(label for _, label in part))
    shutil.rmtree(cells_dir, ignore_errors=True)
    return sheets, legend


# ---------------------------------------------------------------- the agent

REVIEW_PROMPT = """You are the quality reviewer for a narrated explainer video. This is review round {round} of at most {rounds}.
A draft of the video was rendered at half size from ./storyboard.json. Your job is to find what is broken in it and fix what you can, by editing ./storyboard.json.

Look at the contact sheets (images). Each cell is a frame from the draft, labelled in the legend below. An "early" frame is just after the line starts, a "late" frame is near its end,
when everything the scene reveals should be on screen. Open every sheet with the Read tool before you decide anything.

Contact sheets:
{legend}

Automatic checks found these problems (the voice, audio and file were measured, not guessed):
{issues}

Look for, in this order:
1. Broken elements: text cut off or running outside its box or the frame, text overlapping other text or the captions, a "Figure unavailable" placeholder, a figure crop or zoom that shows the wrong part, callouts pointing at nothing, an equation drawn as plain monospace text, a scene that is empty or nearly empty in the late frame.
2. Broken animation: items that should be visible in the late frame but are missing (their `at` word is wrong or too late in the line), a scene that looks identical in the early and late frame when it should be building up, the same layout repeated line after line.
3. Readability: on-screen text too long or too small to read at phone size, low contrast.
4. Voice: lines listed above as voice problems. A synthetic voice mangles numbers, symbols, unusual names and long sentences.

How to fix:
- Edit ./storyboard.json in place. Change scene fields: shorten on-screen text, choose a better scene type, fix a figure id, zoom or callout position, correct `at` anchors, adjust `highlight`, `gapAfter` or `kicker`.
- Do not change the number or order of lines. Do not change any line's `text`{text_rule}
- Keep the narration facts as they are. Never invent numbers.
- After editing, run ./validate.sh storyboard.json until it prints OK.
- Fix only what is actually wrong. If the draft looks good, change nothing. Do not restyle a working scene.
- Problems you cannot fix by editing the storyboard (a silent music bed, a recorded voice that is garbled, a missing sound effect) go in "remaining".

Finally write ./review/verdict.json, a JSON object:
{{"satisfied": true or false, "summary": "one or two plain sentences on what you found and changed", "remaining": ["problems you could not fix"]}}
Set "satisfied" to true only when you looked at every sheet and found nothing left that is worth fixing. If you changed the storyboard this round, set it to false so the new version gets checked.
"""


def text_rule(editable: list[int]) -> str:
    if not editable:
        return "."
    return (f", except lines {', '.join(str(i + 1) for i in editable)}, whose voice problems you may fix by rewording that line with the same meaning "
            "(spell out numbers and symbols, shorten, avoid unusual words) and the same facts.")


def build_prompt(round_no: int, rounds: int, legend: list[str], issues: list[dict], editable: list[int]) -> str:
    def row(it: dict) -> str:
        where = f"line {it['line'] + 1}" if it["line"] is not None else "the video"
        return f"- [{it['kind']}] {where}: {it['detail']}" + (" (you may reword this line)" if it["line"] in editable else "")
    return REVIEW_PROMPT.format(round=round_no, rounds=rounds, legend="\n".join(f"- {x}" for x in legend) or "- (no sheets could be made)",
                                issues="\n".join(row(i) for i in issues) or "- none", text_rule=text_rule(editable))


def read_verdict(job: Job) -> dict:
    f = job.dir / "review" / "verdict.json"
    try:
        v = json.loads(f.read_text())
    except (OSError, ValueError):
        return {"satisfied": False, "summary": "", "remaining": []}
    return {"satisfied": bool(v.get("satisfied")), "summary": str(v.get("summary") or "")[:600],
            "remaining": [str(x)[:200] for x in (v.get("remaining") or [])[:8]] if isinstance(v.get("remaining"), list) else []}


def orientation(fmt: str) -> str:
    w, h = config.FORMATS[fmt]
    return "portrait" if w < h else "landscape" if w > h else "square"
