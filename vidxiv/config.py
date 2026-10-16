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
LEAD = 10  # frames of silence before the first line
DEFAULT_GAP = 14  # frames between lines
TAIL = 75  # frames after the last line
OVERLAP = 12  # frames a scene overlaps the next one

FORMATS = {"Vertical": (1080, 1920), "Landscape": (1920, 1080), "Square": (1080, 1080)}
PRESET_SECONDS = {"short": 45, "medium": 90, "long": 180}
MIN_SECONDS, MAX_SECONDS = 20, 600
SECONDS_PER_LINE = 5.2
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
    n = max(3, round(seconds / SECONDS_PER_LINE))
    detail = DETAILS.get(brief.get("detail", "balanced"), DETAILS["balanced"])
    return seconds, f"{max(3, n - 1)} to {n + 1}", detail


for d in (JOBS, MODELS, CACHE):
    d.mkdir(parents=True, exist_ok=True)
