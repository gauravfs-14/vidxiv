"""Stage workers. Each runs inside jobs.background() so progress and errors land in job.json."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import agents, jobs, paper, render, schema, script, voice
from .jobs import Job


def start_ingest(job: Job, upload: Path | None = None) -> None:
    def work():
        log = lambda m: (job.log("ingest", m), job.progress("ingest", 0.5, m))  # noqa: E731
        job.progress("ingest", 0.1, "Fetching the paper")
        state = job.load()
        info = paper.ingest(job.dir, state["source"], upload, log)
        figures = job.figures()
        job.patch(title=info["title"] or state["title"], paper={k: info[k] for k in ("title", "authors", "abstract", "url", "published", "pages", "sections")} | {"figureCount": len(figures)})
    jobs.background(job, "ingest", work)


def start_script(job: Job, agent: str, model: str, mode: str, text: str = "") -> None:
    """mode: generate (agent writes narration and scenes), provided (user text, agent picks scenes), transcript (from recorded audio)."""
    def work():
        sb = None
        fixed: list[str] | None = None
        paper_info = job.paper() or {}
        figures = job.figures()
        brief = job.load()["brief"]
        if mode == "transcript":
            take = next((p for p in (job.dir / "narration").glob("take-upload.*")), None)
            if not take:
                raise RuntimeError("Upload the recording first")
            job.progress("script", 0.05, "Transcribing the recording")
            fixed = voice.transcript_lines(job, take)
            job.log("script", f"Transcribed {len(fixed)} lines")
        else:  # new lines invalidate old recordings
            shutil.rmtree(job.dir / "narration" / "user", ignore_errors=True)
            job.update(lambda s: s["narration"].update(clips={}))
        if mode == "provided":
            fixed = script.split_script(text)
            if not fixed:
                raise RuntimeError("The script text is empty")
            job.log("script", f"Using your script: {len(fixed)} lines")
        chosen = agents.pick(agent)
        if fixed is not None:
            job.patch(narrationText=fixed)
        if chosen is None:
            if agent not in ("auto", "none"):
                raise RuntimeError(f"The {agent} CLI was not found on this machine")
            job.log("script", "No agent CLI in use, building scenes with simple rules")
            if fixed is None:
                sb = script.fallback_storyboard(paper_info, figures, brief.get("length", "short"), brief.get("palette", "paper"))
            else:
                sb = script.assign_scenes(fixed, paper_info, figures, brief.get("palette", "paper"))
            sb, problems = agents.finalize(sb, {f["id"] for f in figures}, fixed)
            if problems:
                raise RuntimeError("; ".join(problems[:3]))
            job.storyboard_path.write_text(json.dumps(sb, indent=2))
        else:
            sb = agents.generate_storyboard(job, chosen, model, fixed)
        sb["palette"] = brief.get("palette", sb.get("palette", "paper"))
        job.storyboard_path.write_text(json.dumps(sb, indent=2))
        job.patch(title=sb["title"] or job.load()["title"])
        job.progress("script", 1, f"{len(sb['lines'])} lines written")
    jobs.background(job, "script", work)


def start_narration(job: Job, only: list[int] | None = None) -> None:
    jobs.background(job, "narration", lambda: voice.run_narration(job, only))


def start_split(job: Job, audio: Path) -> None:
    """Cut one uploaded or recorded take into line clips using the current script, then align every line."""
    def work():
        sb = job.storyboard()
        if not sb:
            raise RuntimeError("Write the script first, then upload the recording")
        job.progress("narration", 0.1, "Transcribing and matching your take to the script")
        res = voice.split_take(job, audio, [ln["text"] for ln in sb["lines"]])
        job.log("narration", f"Matched {len(res['matched'])} lines" + (f", missing: {[m + 1 for m in res['missing']]}" if res["missing"] else ""))
        voice.run_narration(job)
        if res["missing"]:
            job.set_stage("narration", message=f"Lines {', '.join(str(m + 1) for m in res['missing'])} were not found in the recording and use the synthetic voice")
    jobs.background(job, "narration", work)


def start_preview(job: Job) -> None:
    jobs.background(job, "preview", lambda: render.preview(job))


def start_render(job: Job) -> None:
    jobs.background(job, "render", lambda: render.render(job))


def save_storyboard(job: Job, data: dict) -> tuple[dict | None, list[str]]:
    """Validate and store an edited storyboard. Returns (saved, problems)."""
    fig_ids = {f["id"] for f in job.figures()}
    data = script.prune(data)
    for ln in data.get("lines", []):
        if isinstance(ln, dict) and isinstance(ln.get("text"), str):
            ln["text"] = script.clean(ln["text"])
    schema.repair(data)
    problems = schema.check(data, fig_ids)
    if problems:
        return None, problems
    clean = schema.parse(data).model_dump(by_alias=True, exclude_none=True)
    job.storyboard_path.write_text(json.dumps(clean, indent=2))
    job.patch(title=clean["title"])
    return clean, []
