"""Command line: serve the app, validate a storyboard, check the environment, or make a video headless."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__


def cmd_serve(a) -> int:
    import uvicorn

    url = f"http://{a.host}:{a.port}"
    if not a.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"vidxiv {__version__} running at {url}  (Ctrl+C to stop)")
    uvicorn.run("vidxiv.server:app", host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_validate(a) -> int:
    """Used by the script agent: repairs small mistakes in place, then reports what is left."""
    from . import schema, script

    path = Path(a.file)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"PROBLEM: cannot read {path}: {e}")
        return 1
    from . import paper

    added, region_problems = paper.add_regions(path.parent.resolve())
    for f in added:
        print(f"Region {f['id']} cropped to paper/figures/{f['file']} ({f['width']}x{f['height']}). Open it to check the crop.")
    fig_file = Path(a.figures)
    figs = {f["id"] for f in json.loads(fig_file.read_text())} if fig_file.exists() else None
    for ln in data.get("lines", []) if isinstance(data, dict) else []:
        if isinstance(ln, dict) and isinstance(ln.get("text"), str):
            ln["text"] = script.clean(ln["text"])
    schema.repair(data)
    problems = region_problems + schema.check(script.prune(data), figs)
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"- {p}")
        return 1
    path.write_text(json.dumps(schema.parse(script.prune(data)).model_dump(by_alias=True, exclude_none=True), indent=2))
    warns = script.lint_storyboard(json.loads(path.read_text()))
    for i, w in warns.items():
        print(f"WARN line {i + 1}: " + "; ".join(w))
    n = len(data["lines"])
    print(f"OK: {n} lines" + (f", {len(warns)} with style warnings (rewrite them in plain words)" if warns else ""))
    return 0


def cmd_page(a) -> int:
    """Used by the script agent: render one PDF page to an image it can open."""
    from . import paper

    try:
        out = paper.render_page(Path(a.dir).resolve(), a.number)
    except (ValueError, OSError) as e:
        print(f"PROBLEM: {e}")
        return 1
    print(f"Saved {out.relative_to(Path(a.dir).resolve())}. Open it to look at the page.")
    return 0


def cmd_doctor(a) -> int:
    from . import agents, render, voice

    ag = agents.detect()
    tools = render.node_tools()
    eng = voice.engines()
    rows = [
        ("Claude Code CLI", ag["claude"]["found"], ag["claude"]["version"] or "not found"),
        ("Codex CLI", ag["codex"]["found"], ag["codex"]["version"] or "not found"),
        ("ffmpeg", tools["ffmpeg"], ""),
        ("Node.js", tools["node"], ""),
        ("Video engine installed", tools["engine"], "run: vidxiv setup" if not tools["engine"] else ""),
        ("Kokoro voice model", eng["kokoro"]["ready"], "downloads on first use" if not eng["kokoro"]["ready"] else ""),
        ("macOS voices", eng["say"]["ready"], ""),
    ]
    for name, ok, note in rows:
        print(f"{'ok ' if ok else '-- '} {name:26} {note}")
    return 0 if tools["ffmpeg"] and tools["node"] else 1


def cmd_setup(a) -> int:
    from . import render, voice

    render.ensure_engine(print)
    if a.voice:
        voice.download_kokoro(print)
    render.ensure_sfx(print)
    return 0


def cmd_install_skills(a) -> int:
    """Copy the bundled skills into the global skill folders so the CLIs can use them in any project."""
    import shutil

    from . import config

    targets = [Path.home() / ".claude" / "skills", Path.home() / ".agents" / "skills"]
    for name in ("paper-video", "video-taste", "unslop"):
        for t in targets:
            dest = t / name
            if dest.exists() and not a.force:
                print(f"skip {dest} (exists, use --force to replace)")
                continue
            shutil.rmtree(dest, ignore_errors=True)
            t.mkdir(parents=True, exist_ok=True)
            shutil.copytree(config.SKILLS / name, dest)
            print(f"installed {dest}")
    return 0


def cmd_make(a) -> int:
    """Headless: paper in, MP4 out."""
    from . import agents, config, jobs, paper, pipeline, voice

    if a.model:
        chosen = agents.pick(a.agent)
        if chosen is None or not agents.valid_model(chosen, a.model):
            offered = ", ".join(m["id"] for m in agents.models(chosen) if m["id"]) if chosen else "none"
            print(f"Model '{a.model}' is not offered by {chosen or 'any installed agent'}. Available: {offered}", file=sys.stderr)
            return 2
    jobs.recover_interrupted()
    if a.seconds > 300:
        print(f"Note: a {a.seconds // 60} minute video makes the agent read the whole paper and write about {round(a.seconds / config.pace({"pace": a.pace})["line_seconds"])} lines. This uses a lot of tokens.", file=sys.stderr)
    if Path(a.source).exists():
        job = jobs.create({"type": "pdf", "filename": Path(a.source).name}, Path(a.source).stem)
        pipeline.start_ingest(job, Path(a.source).resolve())
    else:
        aid = paper.parse_arxiv_id(a.source)
        if not aid:
            print("Source must be an arXiv id, an arXiv link, or a path to a PDF", file=sys.stderr)
            return 2
        job = jobs.create({"type": "arxiv", "id": aid, "url": f"https://arxiv.org/abs/{aid}"}, f"arXiv:{aid}")
        pipeline.start_ingest(job)
    job.update(lambda s: s["brief"].update(seconds=a.seconds, detail=a.detail, formats=a.format, look=a.look[0], looks=a.look, captions=not a.no_captions, music=a.music, instructions=a.notes or "", pace=a.pace, reviewRounds=a.review))
    if a.voice:
        job.update(lambda s: s["narration"].update(voice=a.voice))

    def wait(stage: str) -> None:
        seen = 0
        while True:
            time.sleep(1)
            lg = job.read_log(stage, seen)
            for line in lg["lines"]:
                print(f"[{stage}] {line}")
            seen = lg["next"]
            st = job.load()["stages"][stage]
            if st["state"] in ("done", "error", "idle"):
                if st["state"] == "error":
                    print(f"{stage} failed: {st.get('error')}", file=sys.stderr)
                    raise SystemExit(1)
                return

    wait("ingest")
    text = Path(a.script).read_text() if a.script else ""
    pipeline.start_script(job, a.agent, a.model or "", "provided" if text else "generate", text)
    wait("script")
    if a.audio:
        import shutil

        dest = job.dir / "narration" / f"take-upload{Path(a.audio).suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.audio, dest)
        pipeline.start_split(job, dest)
        wait("narration")
    pipeline.start_render(job)
    wait("render")
    for o in job.load()["outputs"]:
        print(f"Done: {job.dir / o['file']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vidxiv", description="Turn a paper into a narrated explainer video.")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web app (default)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8420)
    s.add_argument("--no-open", action="store_true")
    s.set_defaults(fn=cmd_serve)
    v = sub.add_parser("validate", help="check a storyboard.json (used by the script agent)")
    v.add_argument("file")
    v.add_argument("--figures", default="paper/figures.json")
    v.set_defaults(fn=cmd_validate)
    pg = sub.add_parser("page", help="render one PDF page to an image (used by the script agent)")
    pg.add_argument("number", type=int)
    pg.add_argument("--dir", default=".")
    pg.set_defaults(fn=cmd_page)
    d = sub.add_parser("doctor", help="check what is installed")
    d.set_defaults(fn=cmd_doctor)
    st = sub.add_parser("setup", help="install the video engine and sound effects")
    st.add_argument("--voice", action="store_true", help="also download the Kokoro voice model")
    st.set_defaults(fn=cmd_setup)
    k = sub.add_parser("install-skills", help="copy the paper-video, video-taste and unslop skills into ~/.claude/skills and ~/.agents/skills")
    k.add_argument("--force", action="store_true")
    k.set_defaults(fn=cmd_install_skills)
    m = sub.add_parser("make", help="make a video without the web app")
    m.add_argument("source", help="arXiv id or link, or a PDF path")
    m.add_argument("--agent", default="auto", choices=["auto", "claude", "codex", "none"])
    m.add_argument("--model", default="")
    m.add_argument("--seconds", type=int, default=45, help="target narration length, 10 to 600")
    m.add_argument("--pace", default="standard", choices=["standard", "fast", "reel"], help="fast and reel give shorter lines, tighter cuts and quicker animation")
    m.add_argument("--review", type=int, default=0, metavar="ROUNDS", help="let the agent review a draft and fix problems, up to this many rounds (0 to 8, 0 is off, 8 is until satisfied)")
    m.add_argument("--detail", default="balanced", choices=["overview", "balanced", "deep"])
    m.add_argument("--format", action="append", default=None, choices=["Vertical", "Landscape", "Square"])
    m.add_argument("--look", action="append", default=None, help="repeat to render several looks", choices=["auto", "studio", "night", "swiss", "cobalt"])
    m.add_argument("--music", default="warm", choices=["warm", "bright", "calm", "night", "none"])
    m.add_argument("--voice", default="")
    m.add_argument("--script", help="text file with your own narration")
    m.add_argument("--audio", help="audio file of you reading the script")
    m.add_argument("--notes", help="extra guidance for the scriptwriter")
    m.add_argument("--no-captions", action="store_true")
    m.set_defaults(fn=cmd_make)
    a = p.parse_args(argv)
    if not a.cmd:
        a = p.parse_args(["serve"])
    if a.cmd == "make":
        a.format = a.format or ["Vertical"]
        a.look = a.look or ["auto"]
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
