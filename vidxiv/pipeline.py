"""Stage workers. Each runs inside jobs.background() so progress and errors land in job.json."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import agents, config, jobs, paper, render, review, schema, script, voice
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
        archive_storyboard(job)
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
                sb = script.fallback_storyboard(paper_info, figures, config.target(brief)[0], brief.get("look", "auto"))
            else:
                sb = script.assign_scenes(fixed, paper_info, figures, brief.get("look", "auto"))
            sb, problems = agents.finalize(sb, {f["id"] for f in figures}, fixed)
            if problems:
                raise RuntimeError("; ".join(problems[:3]))
            job.storyboard_path.write_text(json.dumps(sb, indent=2))
        else:
            sb = agents.generate_storyboard(job, chosen, model, fixed)
        if brief.get("look", "auto") != "auto":
            sb["look"] = brief["look"]
        job.storyboard_path.write_text(json.dumps(sb, indent=2))
        job.patch(title=sb["title"] or job.load()["title"])
        job.progress("script", 1, f"{len(sb['lines'])} lines written")
    jobs.background(job, "script", work)


def start_narration(job: Job, only: list[int] | None = None) -> None:
    jobs.background(job, "narration", lambda: voice.run_narration(job, only))


def start_split(job: Job, audio: Path, marks: list[dict] | None = None) -> None:
    """Cut one uploaded or recorded take into line clips using the current script, then align every line."""
    def work():
        sb = job.storyboard()
        if not sb:
            raise RuntimeError("Write the script first, then upload the recording")
        job.progress("narration", 0.1, "Transcribing and matching your take to the script")
        if marks:
            job.log("narration", "Cutting your take at the moments you pressed next line")
            res = voice.split_by_marks(job, audio, marks, len(sb["lines"]))
        else:
            res = voice.split_take(job, audio, [ln["text"] for ln in sb["lines"]])
        job.log("narration", f"Matched {len(res['matched'])} lines" + (f", missing: {[m + 1 for m in res['missing']]}" if res["missing"] else ""))
        voice.run_narration(job)
        if res["missing"]:
            job.set_stage("narration", message=f"Lines {', '.join(str(m + 1) for m in res['missing'])} were not found in the recording and use the synthetic voice")
    jobs.background(job, "narration", work)


def start_preview(job: Job) -> None:
    jobs.background(job, "preview", lambda: render.preview(job))


def start_render(job: Job) -> None:
    jobs.background(job, "render", lambda: _render_all(job))


REVIEW_SHARE = 0.55  # of the render stage's progress bar, when a review runs before the final render


def _render_all(job: Job) -> None:
    """Optionally review a draft and fix what the agent finds, then render the final videos from the fixed storyboard."""
    brief = job.load()["brief"]
    rounds = config.review_rounds(brief)
    share = 0.0
    if not rounds:
        job.patch(review=None)
    elif (agent := agents.pick(brief.get("agent", "auto"))) is None:
        job.log("render", "Review skipped: no Claude Code or Codex CLI is available to look at the draft")
        job.patch(review={"status": "skipped", "reason": "No Claude Code or Codex CLI was available to review the video.", "maxRounds": rounds, "rounds": []})
    else:
        _review_loop(job, agent, brief.get("model", ""), rounds)
        share = REVIEW_SHARE
    job.check()
    render.render(job, share, 1.0)


def _review_loop(job: Job, agent: str, model: str, rounds: int) -> None:
    """Draft, measure, show the agent the frames, apply its fixes, and go again until it is satisfied, nothing changes, or the rounds run out.
    A failed review never blocks the render: the video is rendered from the storyboard as it stands."""
    rec = {"agent": agent, "model": model, "maxRounds": rounds, "status": "running", "rounds": [], "started": time.time()}
    job.patch(review=rec)
    per = REVIEW_SHARE / rounds
    for n in range(1, rounds + 1):
        job.check()
        lo = per * (n - 1)
        job.log("render", f"Review round {n} of {rounds}")
        entry = {"round": n, "issues": [], "changed": 0, "summary": "", "satisfied": False, "remaining": []}
        rec["rounds"].append(entry)
        try:
            render.ensure_narrated(job, "render")
            issues = review.check_voice(job, lambda m: job.log("render", m))
            draft = render.render_draft(job, "render", lo, lo + per * 0.6)
            job.progress("render", lo + per * 0.62, f"Review {n} of {rounds}: checking the draft")
            brief = job.load()["brief"]
            issues += review.check_assets(draft["props"], brief)
            issues += review.check_video(draft["file"], draft["props"]["total"] / config.FPS)
            out = job.dir / "review" / f"round-{n}"
            sheets, legend = review.contact_sheets(draft["file"], draft["props"], review.orientation(draft["format"]), out)
            entry["issues"] = issues
            (out / "report.json").write_text(json.dumps({"issues": issues, "sheets": legend}, indent=2))
            for it in issues:
                job.log("render", f"Check: {it['kind']}, {('line ' + str(it['line'] + 1)) if it['line'] is not None else 'video'} {it['detail']}")
            job.patch(review=rec)
            editable = sorted({it["line"] for it in issues if it["kind"] == "voice" and it["fixable"] and it["line"] is not None})
            agents.prepare_workspace(job, brief, None)
            (job.dir / "review" / "verdict.json").unlink(missing_ok=True)
            job.progress("render", lo + per * 0.7, f"Review {n} of {rounds}: {agent} is looking at the draft")
            changed = _agent_pass(job, agent, model, review.build_prompt(n, rounds, legend, issues, editable), editable, tuple(sheets))
        except jobs.Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            job.log("render", f"Review round {n} stopped: {e}")
            entry["error"] = str(e)[:400]
            rec["status"] = "failed"
            break
        verdict = review.read_verdict(job)
        entry.update(changed=changed, summary=verdict["summary"], remaining=verdict["remaining"], satisfied=verdict["satisfied"] and changed == 0)
        job.log("render", f"Review round {n}: {changed} line(s) changed. {verdict['summary']}")
        if entry["satisfied"]:
            rec["status"] = "satisfied"
        elif changed == 0:
            rec["status"] = "stalled"  # nothing it could fix, and not content: another round would see the same draft
        elif n == rounds:
            rec["status"] = "limit"
        job.patch(review=rec)
        if rec["status"] != "running":
            break
    if rec["status"] == "running":
        rec["status"] = "limit"
    rec["finished"] = time.time()
    (job.dir / "review" / "draft.mp4").unlink(missing_ok=True)
    job.patch(review=rec)


def _agent_pass(job: Job, agent: str, model: str, prompt: str, editable: list[int], sheets: tuple[Path, ...]) -> int:
    """Run the reviewing agent on storyboard.json. Returns how many lines it changed.
    Its edits are held to the same rules as any storyboard (validation), the line count and order, and the narration text except for
    lines it was allowed to reword. If they cannot be made valid the original is restored and this raises."""
    before = job.storyboard()
    original = job.storyboard_path.read_text()
    fig_ids = {f["id"] for f in job.figures()}
    problems: list[str] = []
    data: dict = {}
    try:
        for _ in range(2):
            agents.run_agent(job, agent, model, prompt if not problems else agents.build_prompt({}, False, problems), images=sheets)
            try:
                data = json.loads(job.storyboard_path.read_text())
            except (OSError, ValueError) as e:
                problems = [f"storyboard.json is not valid JSON: {e}"]
                continue
            data, problems = _constrain(data, before, editable, fig_ids)
            if not problems:
                break
            job.log("render", "Review edits did not validate:\n- " + "\n- ".join(problems))
    finally:
        job.storyboard_path.write_text(original)  # the real save happens below, through the normal path
    if problems:
        raise RuntimeError("The reviewer's edits did not validate: " + "; ".join(problems[:3]))
    changed = sum(1 for a, b in zip(before["lines"], data["lines"]) if a != b)
    if changed:
        archive_storyboard(job)
        saved, bad = save_storyboard(job, data)
        if bad:
            raise RuntimeError("The reviewer's edits did not validate: " + "; ".join(bad[:3]))
    return changed


def _constrain(data: dict, before: dict, editable: list[int], fig_ids: set[str]) -> tuple[dict, list[str]]:
    lines = data.get("lines")
    if not isinstance(lines, list) or len(lines) != len(before["lines"]):
        return data, [f"storyboard.json must keep exactly {len(before['lines'])} lines in the same order, it has {len(lines) if isinstance(lines, list) else 'no'} lines"]
    for i, (ln, old) in enumerate(zip(lines, before["lines"])):
        if isinstance(ln, dict) and i not in editable:
            ln["text"] = old["text"]
    return agents.finalize(data, fig_ids)


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
    _remap_clips(job, [ln["text"] for ln in (job.storyboard() or {"lines": []})["lines"]], [ln["text"] for ln in clean["lines"]])
    job.storyboard_path.write_text(json.dumps(clean, indent=2))
    job.patch(title=clean["title"])
    return clean, []


def _remap_clips(job: Job, old: list[str], new: list[str]) -> None:
    """Recordings follow their text when lines are reordered, added or deleted. Edited lines keep their slot."""
    mapping: dict[int, int] = {}  # new index -> old index
    used: set[int] = set()
    for j, t in enumerate(new):
        i = next((i for i, o in enumerate(old) if o == t and i not in used), None)
        if i is not None:
            mapping[j] = i
            used.add(i)
    for j in range(min(len(new), len(old))):
        if j not in mapping and j not in used:
            mapping[j] = j
            used.add(j)
    if all(j == i for j, i in mapping.items()) and len(new) >= len(old):
        return
    udir = job.dir / "narration" / "user"
    stash = udir / "_stash"
    stash.mkdir(parents=True, exist_ok=True)
    for i in range(len(old)):
        f = udir / f"line-{i}.wav"
        if f.exists():
            f.rename(stash / f.name)
    for j, i in mapping.items():
        f = stash / f"line-{i}.wav"
        if f.exists():
            f.rename(udir / f"line-{j}.wav")
    shutil.rmtree(stash, ignore_errors=True)

    def clips(s):
        s["narration"]["clips"] = {str(j): s["narration"]["clips"][str(i)] for j, i in mapping.items() if str(i) in s["narration"]["clips"]}
    job.update(clips)


def archive_storyboard(job: Job) -> None:
    """Keep the current script as a version before it is replaced."""
    import time

    sb = job.storyboard()
    if not sb:
        return
    d = job.dir / "storyboards"
    d.mkdir(exist_ok=True)
    (d / f"sb-{int(time.time())}.json").write_text(json.dumps(sb, indent=2))


def storyboard_versions(job: Job) -> list[dict]:
    out = []
    for f in sorted((job.dir / "storyboards").glob("sb-*.json"), reverse=True):
        try:
            sb = json.loads(f.read_text())
        except ValueError:
            continue
        out.append({"id": f.stem, "created": int(f.stem[3:]), "title": sb.get("title", ""), "lines": len(sb.get("lines", [])),
                    "preview": sb["lines"][0]["text"][:90] if sb.get("lines") else ""})
    return out


def restore_storyboard(job: Job, version_id: str) -> None:
    f = job.dir / "storyboards" / f"{version_id}.json"
    if not f.exists() or not version_id.startswith("sb-"):
        raise ValueError("No such script version")
    data = json.loads(f.read_text())
    archive_storyboard(job)
    saved, problems = save_storyboard(job, data)
    if problems:
        raise ValueError("That version no longer validates: " + "; ".join(problems[:2]))
