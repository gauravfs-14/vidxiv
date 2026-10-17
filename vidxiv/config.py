"""Paths and constants. Everything vidxiv writes lives under HOME (default: ./workspace)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("VIDXIV_HOME", ROOT / "workspace")).expanduser().resolve()
JOBS = HOME / "jobs"
MODELS = HOME / "models"
CACHE = HOME / "cache"
ENGINE = ROOT / "engine"
SKILLS = ROOT / "skills"
ASSETS = Path(__file__).resolve().parent / "assets"
WEB = Path(__file__).resolve().parent / "web"

FPS = 30

FORMATS = {"Vertical": (1080, 1920), "Landscape": (1920, 1080), "Square": (1080, 1080)}
PRESET_SECONDS = {"short": 45, "medium": 90, "long": 180}
MIN_SECONDS, MAX_SECONDS = 10, 600

# Pace sets timing in frames (30 fps), animation speed, caption chunking, voice rate and how long a narration line runs.
#   lead: silence before the first line   gap: between lines   tail: after the last line
#   pre: a scene starts this many frames before its line      overlap: frames a scene overlaps the next
#   speed: multiplier on every animation length               words: max caption words per chunk (None keeps the layout's)
#   voice: multiplier on the synthetic voice speed            line_seconds: spoken seconds per line, sets the line count
PACES = {
    "standard": {"label": "Standard", "lead": 10, "gap": 14, "tail": 75, "pre": 10, "overlap": 12, "speed": 1.0, "words": None, "voice": 1.0, "line_seconds": 5.2,
                 "guide": "Standard pace for an academic audience. Lines of 8 to 26 words, one or two sentences."},
    "fast": {"label": "Fast", "lead": 5, "gap": 6, "tail": 45, "pre": 6, "overlap": 8, "speed": 1.45, "words": 4, "voice": 1.08, "line_seconds": 3.8,
             "guide": "Fast pace, tighter cuts. Lines of 6 to 16 words, a single sentence each, a new visual on every line. "
                      "Cut setup and hedging, keep the number and the idea. Prefer stat, bars, steps and figure scenes over long concept text."},
    "reel": {"label": "Reel", "lead": 2, "gap": 1, "tail": 24, "pre": 4, "overlap": 5, "speed": 2.0, "words": 3, "voice": 1.18, "line_seconds": 2.6,
             "guide": "Reel or TikTok pace: relentless and short. Every line is one punchy sentence of 4 to 11 words, about 2 to 3 seconds. "
                      "The first line must be the hook, a number or a surprising claim, spoken in the first two seconds. "
                      "Change the visual on every line, use big stat, bars, quick steps and a zoomed figure, and keep on-screen text to a few words. "
                      "Skip the title scene unless it is one line, skip equation and quote scenes unless the paper is only about that, "
                      "keep one limit as a single line, and end on a one-line takeaway. Cut everything that is not the single point."},
}
DEFAULT_PACE = "standard"
REVIEW_MAX = 8  # hard cap on review rounds, which is also what "until satisfied" means


def pace(brief: dict) -> dict:
    """Timing settings for the brief's pace. Unknown names fall back to standard."""
    name = brief.get("pace") if brief.get("pace") in PACES else DEFAULT_PACE
    return {"name": name, **PACES[name]}


def review_rounds(brief: dict) -> int:
    try:
        return max(0, min(REVIEW_MAX, int(brief.get("reviewRounds") or 0)))
    except (TypeError, ValueError):
        return 0


DETAILS = {
    "overview": "Overview. One result and one mechanism, in plain words for a general audience. Skip equations, ablations and implementation details.",
    "balanced": "Balanced. The problem, the idea, how it works, the main result with its numbers, and one limit. Define terms the first time they appear.",
    "deep": "In depth. For a technical audience. Cover the method step by step, the key equation if there is one, the main results and what the ablations show, how it compares with the strongest baseline, and the limits and assumptions. Use more figures, bars and steps scenes.",
}


def target(brief: dict) -> tuple[int, str, str]:
    """(seconds, line count text, detail text) for a brief. Old `length` presets still work."""
    try:
        seconds = int(brief.get("seconds") or PRESET_SECONDS.get(brief.get("length", "short"), 45))
    except (TypeError, ValueError):
        seconds = 45
    seconds = max(MIN_SECONDS, min(MAX_SECONDS, seconds))
    n = max(3, round(seconds / pace(brief)["line_seconds"]))
    detail = DETAILS.get(brief.get("detail", "balanced"), DETAILS["balanced"])
    return seconds, f"{max(3, n - 1)} to {n + 1}", detail


for d in (JOBS, MODELS, CACHE):
    d.mkdir(parents=True, exist_ok=True)
