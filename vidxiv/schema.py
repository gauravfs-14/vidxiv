"""Storyboard schema: the single contract between the script agent, the editor UI and the Remotion engine.

The JSON Schema generated from these models is shipped to the agents (skills/paper-video/storyboard.schema.json)
and is what `vidxiv validate` checks. The engine renders only these scene types, so a malformed agent
output is caught before any rendering starts.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator
from pydantic.alias_generators import to_camel

TINTS = ("", "accent", "teal", "rose", "blue", "mustard")
Tint = Literal["", "accent", "teal", "rose", "blue", "mustard"]


class Base(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class Item(Base):
    """A labelled thing that appears on screen. `at` is a word from the line's narration; the item shows when that word is spoken.
    Without `at`, items are spread evenly across the line."""
    label: str = Field(max_length=70)
    detail: str | None = Field(default=None, max_length=140)
    at: str | None = None
    tint: Tint | None = None


class Zoom(Base):
    """Region of a figure as fractions of its width and height (0 to 1)."""
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    at: str | None = None


class Callout(Base):
    text: str = Field(max_length=60)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    at: str | None = None


class Bar(Base):
    label: str = Field(max_length=40)
    value: float
    display: str | None = Field(default=None, max_length=16)
    highlight: bool = False
    at: str | None = None


class Side(Base):
    title: str = Field(max_length=50)
    points: list[str] = Field(min_length=1, max_length=4)
    tint: Tint | None = None


class Hook(Base):
    type: Literal["hook"] = "hook"
    headline: str = Field(max_length=90)
    sub: str | None = Field(default=None, max_length=120)
    circle: str | None = None  # a word in the headline to circle by hand
    figure: str | None = None  # optional figure id shown as a backdrop card


class Title(Base):
    type: Literal["title"] = "title"
    title: str = Field(max_length=160)
    authors: str | None = Field(default=None, max_length=160)
    meta: str | None = Field(default=None, max_length=100)  # e.g. "arXiv:2401.06015 / 2024"
    figure: str | None = None


class Concept(Base):
    type: Literal["concept"] = "concept"
    heading: str = Field(max_length=80)
    body: str | None = Field(default=None, max_length=220)
    items: list[Item] = Field(default_factory=list, max_length=5)


class Steps(Base):
    type: Literal["steps"] = "steps"
    heading: str | None = Field(default=None, max_length=80)
    steps: list[Item] = Field(min_length=2, max_length=6)


class Figure(Base):
    type: Literal["figure"] = "figure"
    figure: str  # id from figures.json
    caption: str | None = Field(default=None, max_length=140)
    zoom: Zoom | None = None
    callouts: list[Callout] = Field(default_factory=list, max_length=3)


class Stat(Base):
    type: Literal["stat"] = "stat"
    value: str = Field(max_length=14)  # "93.2%", "4x", "1.5B". The numeric part counts up.
    label: str = Field(max_length=80)
    detail: str | None = Field(default=None, max_length=120)
    compare: Item | None = None  # a second number to set against it: label = what it is, detail = its value text


class Bars(Base):
    type: Literal["bars"] = "bars"
    title: str | None = Field(default=None, max_length=80)
    unit: str | None = Field(default=None, max_length=20)
    bars: list[Bar] = Field(min_length=2, max_length=6)


class Compare(Base):
    type: Literal["compare"] = "compare"
    left: Side
    right: Side
    verdict: str | None = Field(default=None, max_length=100)


class Equation(Base):
    type: Literal["equation"] = "equation"
    formula: str = Field(max_length=220)  # LaTeX, no $ delimiters
    caption: str | None = Field(default=None, max_length=140)
    terms: list[Item] = Field(default_factory=list, max_length=4)


class Quote(Base):
    type: Literal["quote"] = "quote"
    text: str = Field(max_length=240)
    attribution: str | None = Field(default=None, max_length=100)


class Outro(Base):
    type: Literal["outro"] = "outro"
    headline: str = Field(max_length=100)
    takeaway: str | None = Field(default=None, max_length=160)
    cta: str | None = Field(default=None, max_length=80)
    link: str | None = Field(default=None, max_length=100)


Scene = Annotated[
    Union[Hook, Title, Concept, Steps, Figure, Stat, Bars, Compare, Equation, Quote, Outro],
    Field(discriminator="type"),
]
SCENE_TYPES = ["hook", "title", "concept", "steps", "figure", "stat", "bars", "compare", "equation", "quote", "outro"]


class Line(Base):
    text: str = Field(min_length=2, max_length=420)  # narration, spoken as written
    scene: Scene
    highlight: dict[str, Tint] = Field(default_factory=dict)  # spoken word -> caption marker tint
    kicker: str | None = Field(default=None, max_length=32)  # small chapter label above the scene
    gap_after: int | None = Field(default=None, ge=0, le=90)  # frames of silence after the line

    @field_validator("highlight")
    @classmethod
    def _few(cls, v: dict) -> dict:
        if len(v) > 5:
            raise ValueError("at most 5 highlighted words per line")
        return v


LOOKS = ("studio", "night", "swiss", "cobalt")
LEGACY_LOOKS = {"paper": "studio", "ink": "night"}


class Storyboard(Base):
    title: str = Field(max_length=160)
    source: str | None = Field(default=None, max_length=160)  # "arXiv:2401.06015" or a filename
    look: Literal["studio", "night", "swiss", "cobalt"] = "studio"
    lines: list[Line] = Field(min_length=1, max_length=120)

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, v):
        if isinstance(v, dict) and "palette" in v:  # storyboards written before looks existed
            v = dict(v)
            v["look"] = LEGACY_LOOKS.get(v.pop("palette"), "studio")
        return v


_adapter = TypeAdapter(Storyboard)


def parse(data: dict | str) -> Storyboard:
    return Storyboard.model_validate_json(data) if isinstance(data, str) else Storyboard.model_validate(data)


def check(data: dict, figure_ids: set[str] | None = None) -> list[str]:
    """Human-readable problems with a storyboard dict, empty when it is valid."""
    try:
        sb = Storyboard.model_validate(data)
    except ValidationError as e:
        out = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            out.append(f"{loc}: {err['msg']}")
        return out[:25]
    problems = []
    for i, line in enumerate(sb.lines):
        words = {_norm(w) for w in line.text.split()}
        for field in _refs(line.scene):
            fig = field
            if figure_ids is not None and fig not in figure_ids:
                problems.append(f"lines.{i}.scene: unknown figure id '{fig}'. Available: {', '.join(sorted(figure_ids)) or 'none'}")
        for at in _ats(line.scene):
            if _norm(at) not in words:
                problems.append(f"lines.{i}: at='{at}' is not a word in the narration text")
        for w in line.highlight:
            if _norm(w) not in words:
                problems.append(f"lines.{i}: highlight word '{w}' is not in the narration text")
    return problems[:25]


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def _refs(scene) -> list[str]:
    fig = getattr(scene, "figure", None)
    return [fig] if isinstance(fig, str) and fig else []


def _ats(scene) -> list[str]:
    out = []
    for name in ("items", "steps", "terms", "callouts", "bars"):
        for it in getattr(scene, name, None) or []:
            if getattr(it, "at", None):
                out.append(it.at)
    z = getattr(scene, "zoom", None)
    if z is not None and z.at:
        out.append(z.at)
    return out


def _walk(data, loc):
    """Follow a pydantic error location into data, skipping union tags that are not keys."""
    parent, key, cur = None, None, data
    for part in loc:
        if isinstance(cur, dict) and part in cur:
            parent, key, cur = cur, part, cur[part]
        elif isinstance(cur, list) and isinstance(part, int) and part < len(cur):
            parent, key, cur = cur, part, cur[part]
    return parent, key, cur


def repair(data: dict, rounds: int = 4) -> dict:
    """Fix what is safe to fix mechanically: over-long strings and lists are cut, unknown `at` words and
    highlight words are dropped. Anything else is left for check() to report."""
    for _ in range(rounds):
        try:
            Storyboard.model_validate(data)
            break
        except ValidationError as e:
            changed = False
            for err in e.errors():
                parent, key, cur = _walk(data, err["loc"])
                if parent is None:
                    continue
                if err["type"] == "string_too_long" and isinstance(cur, str):
                    n = err["ctx"]["max_length"]
                    cut = cur[: n - 1].rsplit(" ", 1)[0] if " " in cur[:n] else cur[: n - 1]
                    parent[key] = cut.rstrip(" ,;:")
                    changed = True
                elif err["type"] == "too_long" and isinstance(cur, list):
                    del cur[err["ctx"]["max_length"]:]
                    changed = True
                elif err["type"] == "literal_error" and key == "tint":
                    parent[key] = ""
                    changed = True
            if not changed:
                break
    for line in data.get("lines", []) if isinstance(data, dict) else []:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            continue
        words = {_norm(w) for w in line["text"].split()}
        if isinstance(line.get("highlight"), dict):
            line["highlight"] = {w: t for w, t in line["highlight"].items() if _norm(w) in words}
        _strip_ats(line.get("scene"), words)
    return data


def _strip_ats(scene, words) -> None:
    if not isinstance(scene, dict):
        return
    for name in ("items", "steps", "terms", "callouts", "bars"):
        for it in scene.get(name) or []:
            if isinstance(it, dict) and it.get("at") and _norm(it["at"]) not in words:
                it["at"] = None
    z = scene.get("zoom")
    if isinstance(z, dict) and z.get("at") and _norm(z["at"]) not in words:
        z["at"] = None
    c = scene.get("compare")
    if isinstance(c, dict) and c.get("at") and _norm(c["at"]) not in words:
        c["at"] = None


def json_schema() -> dict:
    return Storyboard.model_json_schema(by_alias=True)


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=2))
