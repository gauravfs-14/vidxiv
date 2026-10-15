"""Script helpers: unslop lint for narration, splitting user text into lines, and a storyboard built without an agent."""
from __future__ import annotations

import re

from . import schema

# Words and phrases from skills/unslop/SKILL.md that read as machine-written.
SLOP_WORDS = [
    "delve", "tapestry", "testament", "pivotal", "crucial", "intricate", "interplay", "underscore", "showcase",
    "vibrant", "groundbreaking", "game-changer", "game changer", "revolutionize", "revolutionary", "unlock", "leverage",
    "utilize", "facilitate", "landscape", "paradigm", "cutting-edge", "state-of-the-art", "seamless", "robust",
    "harness", "foster", "enhance", "realm", "ever-evolving", "navigate",
]
SLOP_PHRASES = [
    r"let'?s dive", r"it'?s important to note", r"it is important to note", r"in today'?s", r"in conclusion",
    r"imagine a world", r"not just [^.,;]{1,40}, but", r"not only [^.,;]{1,40} but", r"here'?s the thing",
    r"in order to", r"a testament to", r"stands as", r"serves as", r"plays a (?:crucial|key|vital) role",
    r"the future (?:looks|is) bright", r"changes everything", r"on a whole new level",
]


def lint(text: str) -> list[str]:
    """Problems that make narration read as AI-written. Empty means clean."""
    out = []
    if re.search(r"[—–]", text):
        out.append("dash used as punctuation, use a period or comma")
    if re.search(r"[‘’“”]", text):
        out.append("curly quotes")
    low = text.lower()
    for w in SLOP_WORDS:
        if re.search(rf"\b{re.escape(w)}\w*\b", low):
            out.append(f"AI-sounding word: {w}")
    for ph in SLOP_PHRASES:
        m = re.search(ph, low)
        if m:
            out.append(f"AI-sounding phrase: {m.group(0)}")
    if text.count(";") > 0:
        out.append("semicolon, split into two sentences")
    if re.search(r"\b\w+ly\b.*\b\w+ly\b.*\b\w+ly\b", low):
        out.append("many adverbs")
    return out[:6]


def clean(text: str) -> str:
    """Mechanical fixes only: dashes, curly quotes, stray markup, whitespace."""
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"\s*--+\s*", ", ", t)
    t = t.replace("*", "")
    return re.sub(r"\s+", " ", t).strip()


def lint_storyboard(sb: dict) -> dict[int, list[str]]:
    return {i: p for i, line in enumerate(sb.get("lines", [])) if (p := lint(line.get("text", "")))}


def split_script(text: str, max_chars: int = 230, min_chars: int = 60) -> list[str]:
    """Split pasted narration into lines of one or two sentences. Blank lines in the input always split."""
    chunks = [c.strip() for c in re.split(r"\n\s*\n|\n", text.strip()) if c.strip()]
    lines: list[str] = []
    for chunk in chunks:
        chunk = clean(chunk)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", chunk)
        cur = ""
        for s in sentences:
            if cur and len(cur) + len(s) + 1 > max_chars:
                lines.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
            if len(cur) >= min_chars and re.search(r"[.!?]$", cur) and len(chunks) > 1:
                lines.append(cur)
                cur = ""
        if cur:
            if lines and len(cur) < min_chars and len(lines[-1]) + len(cur) < max_chars and len(chunks) == 1:
                lines[-1] = f"{lines[-1]} {cur}"
            else:
                lines.append(cur)
    return lines[:60]


# ---------------------------------------------------------------- no-agent storyboard

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if len(s.strip()) > 25]


def _short(s: str, n: int = 90) -> str:
    s = s.strip().rstrip(".")
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0]


def fallback_storyboard(paper: dict, figures: list[dict], length: str = "short", palette: str = "paper") -> dict:
    """A plain but valid storyboard from the abstract and figure captions. Used when no agent CLI is installed."""
    title = paper.get("title") or "Untitled paper"
    authors = paper.get("authors") or []
    meta = paper.get("url", "").replace("https://", "")
    sents = _sentences(paper.get("abstract", "")) or ["This paper does not include a readable abstract."]
    n_fig = {"short": 2, "medium": 4, "long": 6}.get(length, 2)
    lines: list[dict] = []

    def add(text, scene, kicker=None):
        lines.append({"text": clean(text), "scene": scene, "kicker": kicker})

    add(f"{_short(title, 120)}.", {"type": "hook", "headline": _short(title, 80)})
    add(f"This video walks through the paper by {authors[0] + ' and colleagues' if len(authors) > 1 else authors[0] if authors else 'its authors'}.",
        {"type": "title", "title": _short(title, 150), "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "") or None, "meta": meta or None})
    body = sents[: {"short": 4, "medium": 6, "long": 9}.get(length, 4)]
    for i, s in enumerate(body):
        add(s, {"type": "concept", "heading": "The paper in brief" if i == 0 else _short(s, 60), "body": None, "items": []}, "Abstract")
    for fig in figures[:n_fig]:
        cap = fig.get("caption") or ""
        text = _sentences(cap)[0] if _sentences(cap) else f"{fig.get('label', 'This figure')} shows part of the method."
        add(text, {"type": "figure", "figure": fig["id"], "caption": _short(fig.get("label", ""), 40) or None}, "Figure")
    closing = "The full paper is on arXiv, linked below." if "arxiv" in meta else "That is the paper in brief. The full text has the details."
    add(closing, {"type": "outro", "headline": "Read the paper", "takeaway": _short(title, 100), "cta": "Full text on arXiv" if "arxiv" in meta else None, "link": meta or None})
    return {"title": title, "source": meta or None, "palette": palette, "lines": lines}


def assign_scenes(texts: list[str], paper: dict, figures: list[dict], palette: str = "paper") -> dict:
    """Storyboard for user-written lines when no agent is available: opening hook, figures spread through, closing outro."""
    n = len(texts)
    fig_slots = {}
    if figures and n > 3:
        step = max(1, (n - 2) // (min(len(figures), max(1, n // 3)) + 1))
        for k, fig in enumerate(figures[: max(1, n // 3)]):
            pos = 1 + step * (k + 1)
            if pos < n - 1:
                fig_slots[pos] = fig
    lines = []
    for i, t in enumerate(texts):
        if i == 0:
            sc = {"type": "hook", "headline": _short(t, 80)}
        elif i == n - 1 and n > 2:
            sc = {"type": "outro", "headline": "Read the paper", "takeaway": _short(t, 100), "link": (paper.get("url") or "").replace("https://", "") or None}
        elif i in fig_slots:
            sc = {"type": "figure", "figure": fig_slots[i]["id"], "caption": fig_slots[i].get("label")}
        else:
            sc = {"type": "concept", "heading": _short(t, 60), "body": None, "items": []}
        lines.append({"text": t, "scene": sc})
    return {"title": paper.get("title") or "Untitled", "source": (paper.get("url") or "").replace("https://", "") or None, "palette": palette, "lines": lines}


def prune(obj):
    """Drop None values so the storyboard stays tidy on disk."""
    if isinstance(obj, dict):
        return {k: prune(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [prune(v) for v in obj]
    return obj


def normalized(sb: dict, figure_ids: set[str]) -> tuple[dict, list[str]]:
    """Validate a storyboard dict. Returns (clean dict in camelCase, problems)."""
    sb = prune(sb)
    problems = schema.check(sb, figure_ids)
    if problems:
        return sb, problems
    model = schema.parse(sb)
    return model.model_dump(by_alias=True, exclude_none=True), []
