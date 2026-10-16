import json
from pathlib import Path

import numpy as np
import pytest

from vidxiv import paper, schema, script, voice

EXAMPLE = Path(__file__).resolve().parent.parent / "skills/paper-video/references/example-storyboard.json"


def test_example_storyboard_is_valid_and_clean():
    data = json.loads(EXAMPLE.read_text())
    assert schema.check(data, {"fig-01"}) == []
    assert script.lint_storyboard(data) == {}


def test_unknown_figure_and_anchor_are_reported():
    data = json.loads(EXAMPLE.read_text())
    data["lines"][3]["scene"]["figure"] = "fig-99"
    data["lines"][4]["scene"]["steps"][0]["at"] = "nonsense"
    problems = schema.check(data, {"fig-01"})
    assert any("fig-99" in p for p in problems)
    assert any("nonsense" in p for p in problems)


def test_repair_cuts_long_strings_and_drops_bad_anchors():
    data = {"title": "t", "lines": [{"text": "Hello there world", "scene": {"type": "concept", "heading": "x " * 100,
            "items": [{"label": "a", "at": "missing"}] * 8}, "highlight": {"zzz": "teal", "world": "blue"}}]}
    schema.repair(data)
    assert schema.check(data) == []
    assert data["lines"][0]["highlight"] == {"world": "blue"}


def test_lint_flags_ai_tells_and_clean_fixes_mechanics():
    assert script.lint("Let's delve into this pivotal moment — it is a testament to the landscape.")
    assert script.lint("The model reads the whole sentence at once.") == []
    assert script.clean("one — two “quoted”") == 'one, two "quoted"'


def test_split_script_keeps_paragraphs_and_limits_length():
    text = "First idea here, a full sentence that is long enough to stand alone.\n\nSecond idea. It has two short sentences."
    lines = script.split_script(text)
    assert len(lines) == 2 and all(len(l) < 240 for l in lines)


def test_fallback_storyboards_are_valid():
    info = {"title": "A paper", "authors": ["A B", "C D"], "abstract": "We study a thing in depth. The results show it works well. We also test it elsewhere. Limits remain.", "url": "https://arxiv.org/abs/1"}
    figs = [{"id": "fig-01", "label": "Figure 1", "caption": "The setup of the experiment."}]
    sb = script.fallback_storyboard(info, figs)
    assert schema.check(sb, {"fig-01"}) == []
    sb2 = script.assign_scenes(["One sentence of narration here.", "Another sentence to talk through.", "A third one for the middle.", "Last one wraps up."], info, figs)
    assert schema.check(sb2, {"fig-01"}) == []


@pytest.mark.parametrize("s,expected", [
    ("1706.03762", "1706.03762"), ("arXiv:1706.03762v5", "1706.03762"), ("https://arxiv.org/abs/1706.03762v2", "1706.03762"),
    ("https://arxiv.org/pdf/2401.06015.pdf", "2401.06015"), ("hep-th/9901001", "hep-th/9901001"), ("not an id", None),
])
def test_parse_arxiv_id(s, expected):
    assert paper.parse_arxiv_id(s) == expected


def test_estimate_words_matches_tokens():
    w = voice.estimate_words("Hello there, big world.", 2.0)
    assert [x["w"] for x in w] == ["Hello", "there,", "big", "world."]
    assert all(a["s"] <= b["s"] for a, b in zip(w, w[1:]))


def test_match_lines_finds_last_retake():
    words = "the cat sat the cat sat on the mat then dogs bark loud".split()
    heard = [(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate(words)]
    spans = voice._match_lines(heard, ["the cat sat on the mat", "then dogs bark loud"])
    assert spans[0] is not None and spans[0][0] == pytest.approx(3 * 0.4)  # second attempt, not the false start
    assert spans[1] is not None and spans[1][0] > spans[0][1]


def test_add_regions_crops_and_reports_bad_boxes(tmp_path):
    import pymupdf

    (tmp_path / "paper").mkdir()
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.draw_rect(pymupdf.Rect(100, 200, 500, 400), color=(0, 0, 1), fill=(0.8, 0.9, 1))
    doc.save(tmp_path / "paper" / "paper.pdf")
    (tmp_path / "paper" / "ai-figures.json").write_text(json.dumps([
        {"id": "x-01", "label": "Table 1", "kind": "table", "page": 1, "bbox": [0.1, 0.2, 0.9, 0.55], "caption": "A box"},
        {"id": "x-02", "page": 3, "bbox": [0.1, 0.2, 0.9, 0.5]},
        {"id": "x-03", "page": 1, "bbox": [0.5, 0.2, 0.4, 0.5]},
        {"id": "bad", "page": 1, "bbox": [0.1, 0.1, 0.9, 0.5]},
    ]))
    added, problems = paper.add_regions(tmp_path)
    assert [f["id"] for f in added] == ["x-01"]
    assert (tmp_path / "paper" / "figures" / "x-01.png").exists()
    assert len(problems) == 3
    assert json.loads((tmp_path / "paper" / "figures.json").read_text())[0]["kind"] == "table"


def test_split_by_marks_uses_last_attempt_and_reports_missing(tmp_path):
    import types

    import numpy as np
    import soundfile as sf

    sr = 48000
    sf.write(tmp_path / "take.wav", np.zeros(sr * 10, dtype="float32"), sr)
    job = types.SimpleNamespace(dir=tmp_path)
    marks = [{"line": 0, "t": 0.0}, {"line": 1, "t": 3.0}, {"line": 1, "t": 5.0}, {"line": 2, "t": 8.0}]
    res = voice.split_by_marks(job, tmp_path / "take.wav", marks, 4)
    assert res["matched"] == [0, 1, 2] and res["missing"] == [3]
    d1 = sf.info(tmp_path / "narration" / "user" / "line-1.wav").duration
    assert 3.0 < d1 < 3.4  # the redo (5s to 8s), not the first attempt (3s to 5s)
