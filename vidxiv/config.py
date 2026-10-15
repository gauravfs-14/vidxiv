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
# Spoken length per brief choice: (target seconds, number of narration lines)
LENGTHS = {
    "short": (45, "8 to 10"),
    "medium": (90, "14 to 18"),
    "long": (180, "26 to 32"),
}

for d in (JOBS, MODELS, CACHE):
    d.mkdir(parents=True, exist_ok=True)
