"""Local web API and static UI. Binds to localhost only."""
from __future__ import annotations

import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, agents, config, install, jobs, paper, pipeline, render, schema, script, voice
from .jobs import Job

@asynccontextmanager
async def lifespan(_):
    jobs.recover_interrupted()
    yield


app = FastAPI(title="vidxiv", version=__version__, lifespan=lifespan)
SETUP: dict[str, dict] = {}
FILE_DIRS = ("paper/figures", "narration/lines", "narration/user", "out", "previews")


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.exception_handler(RuntimeError)
async def _runtime_error(_, exc: RuntimeError):
    return JSONResponse({"error": str(exc)}, status_code=409)


def get_job(job_id: str) -> Job:
    if not job_id.replace("-", "").isalnum():
        raise HTTPException(404, "No such job")
    j = Job(job_id)
    if not j.exists():
        raise HTTPException(404, "No such job")
    return j


# ---------------------------------------------------------------- environment

@app.get("/api/status")
def status():
    ag, tools = agents.detect(), render.node_tools()
    missing = [n for n in ("claude", "codex") if not ag[n]["found"]] + [n for n in ("ffmpeg", "node") if not tools[n]]
    return {"version": __version__, "platform": {"system": install.system(), "manager": install.manager()}, "install": install.hints(missing),
            "agents": ag, "tools": tools, "voices": {k: {"ready": v["ready"], "label": v["label"]} for k, v in voice.engines().items()},
            "whisper": voice.whisper_ready(), "setup": SETUP}


@app.get("/api/voices")
def voices():
    return voice.engines()


class PreviewBody(BaseModel):
    engine: str = "auto"
    voice: str = "af_heart"
    speed: float = 0.95
    text: str | None = None


@app.post("/api/voices/preview")
def voice_preview(body: PreviewBody):
    try:
        path = voice.preview_voice(body.engine, body.voice, body.speed, body.text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e)) from e
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/setup/{component}")
def setup(component: str):
    if component not in ("engine", "kokoro"):
        raise HTTPException(404, "Unknown component")
    if SETUP.get(component, {}).get("state") == "running":
        return SETUP[component]
    SETUP[component] = {"state": "running", "message": "Starting"}

    def work():
        def log(m):
            SETUP[component]["message"] = m
        try:
            render.ensure_engine(log) if component == "engine" else voice.download_kokoro(log)
            SETUP[component] = {"state": "done", "message": "Ready"}
        except Exception as e:  # noqa: BLE001
            SETUP[component] = {"state": "error", "message": str(e)[:300]}

    threading.Thread(target=work, daemon=True).start()
    return SETUP[component]


# ---------------------------------------------------------------- jobs

def view(job: Job) -> dict:
    s = job.load()
    sb = job.storyboard()
    texts = {str(i): voice.text_hash(ln["text"]) for i, ln in enumerate(sb["lines"])} if sb else {}
    clips = {}
    for k, c in s["narration"]["clips"].items():
        clips[k] = {"source": c["source"], "duration": c["duration"], "engine": c.get("engine"),
                    "stale": texts.get(k) != c.get("thash"), "url": f"/files/{job.id}/narration/lines/line-{k}.wav?v={c.get('ahash', '')}"}
    for i in range(len(sb["lines"]) if sb else 0):
        u = voice.user_clip(job, i)
        if u:
            clips.setdefault(str(i), {})["userUrl"] = f"/files/{job.id}/narration/user/line-{i}.wav?v={int(u.stat().st_mtime)}"
    s["narration"]["clips"] = clips
    s["storyboard"] = sb
    s["lint"] = script.lint_storyboard(sb) if sb else {}
    s["figures"] = [{**f, "url": f"/files/{job.id}/paper/figures/{f['file']}"} for f in job.figures()]
    for o in s.get("outputs", []):  # renders made before versions existed
        o.setdefault("id", o["file"])
        o.setdefault("run", 0)
        o.setdefault("look", "studio")
    s["outputs"] = [{**o, "url": f"/files/{job.id}/{o['file']}", "posterUrl": f"/files/{job.id}/{o['poster']}" if o.get("poster") else None} for o in s.get("outputs", [])]
    s["previews"] = [f"/files/{job.id}/{p}" for p in s.get("previews", [])]
    s["scriptVersions"] = pipeline.storyboard_versions(job)
    s["hasPdf"] = (job.dir / "paper" / "paper.pdf").exists()
    return s


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.post("/api/jobs")
async def create_job(arxiv: str = Form(""), file: UploadFile | None = File(None)):
    if file is not None and file.filename:
        title = Path(file.filename).stem
        job = jobs.create({"type": "pdf", "filename": file.filename}, title)
        up = job.path("paper", "upload.pdf")
        with open(up, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not up.read_bytes()[:5].startswith(b"%PDF"):
            jobs.delete(job.id)
            raise ValueError("That file is not a PDF")
        pipeline.start_ingest(job, up)
    elif arxiv.strip():
        aid = paper.parse_arxiv_id(arxiv)
        if not aid:
            raise ValueError("That does not look like an arXiv id or link. Try 1706.03762 or https://arxiv.org/abs/1706.03762")
        job = jobs.create({"type": "arxiv", "id": aid, "url": f"https://arxiv.org/abs/{aid}"}, f"arXiv:{aid}")
        pipeline.start_ingest(job)
    else:
        raise ValueError("Enter an arXiv id or upload a PDF")
    return view(job)


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    return view(get_job(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    get_job(job_id)
    jobs.delete(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/log/{stage}")
def read_log(job_id: str, stage: str, since: int = 0):
    if stage not in jobs.STAGES:
        raise HTTPException(404, "Unknown stage")
    return get_job(job_id).read_log(stage, since)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    get_job(job_id).cancel()
    return {"ok": True}


@app.put("/api/jobs/{job_id}/brief")
def update_brief(job_id: str, body: dict):
    job = get_job(job_id)
    allowed = set(jobs.default_brief())
    job.update(lambda s: s["brief"].update({k: v for k, v in body.items() if k in allowed}))
    return view(job)


class NarrationConfig(BaseModel):
    engine: str = "auto"
    voice: str = "af_heart"
    speed: float = 0.95


@app.put("/api/jobs/{job_id}/narration/config")
def narration_config(job_id: str, body: NarrationConfig):
    job = get_job(job_id)
    job.update(lambda s: s["narration"].update(engine=body.engine, voice=body.voice, speed=max(0.7, min(1.2, body.speed))))
    return view(job)


# ---------------------------------------------------------------- script

class ScriptBody(BaseModel):
    agent: str = "auto"
    model: str = ""
    mode: str = "generate"  # generate | provided | transcript
    text: str = ""


@app.post("/api/jobs/{job_id}/script")
def make_script(job_id: str, body: ScriptBody):
    job = get_job(job_id)
    if job.load()["stages"]["ingest"]["state"] != "done":
        raise RuntimeError("The paper has not finished loading")
    if body.mode not in ("generate", "provided", "transcript"):
        raise ValueError("Unknown script mode")
    chosen = agents.pick(body.agent)
    if body.model and (chosen is None or not agents.valid_model(chosen, body.model)):
        raise ValueError("That model is not offered by the selected agent. Pick one from the list.")
    job.update(lambda s: s["brief"].update(agent=body.agent, model=body.model, scriptMode=body.mode, providedText=body.text))
    pipeline.start_script(job, body.agent, body.model, body.mode, body.text)
    return view(job)


@app.post("/api/jobs/{job_id}/storyboard/restore/{version_id}")
def restore_script(job_id: str, version_id: str):
    job = get_job(job_id)
    pipeline.restore_storyboard(job, version_id)
    return view(job)


@app.put("/api/jobs/{job_id}/storyboard")
def put_storyboard(job_id: str, body: dict):
    job = get_job(job_id)
    saved, problems = pipeline.save_storyboard(job, body)
    if problems:
        return JSONResponse({"error": "The storyboard has problems", "problems": problems}, status_code=422)
    return view(job)


# ---------------------------------------------------------------- narration

class SynthBody(BaseModel):
    lines: list[int] | None = None


@app.post("/api/jobs/{job_id}/narration")
def synthesize(job_id: str, body: SynthBody | None = None):
    job = get_job(job_id)
    pipeline.start_narration(job, body.lines if body else None)
    return view(job)


async def _save_upload(file: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


@app.post("/api/jobs/{job_id}/narration/line/{index}")
async def upload_line(job_id: str, index: int, file: UploadFile = File(...)):
    """A recording or uploaded file for one line. Replaces the synthetic voice for that line."""
    job = get_job(job_id)
    sb = job.storyboard()
    if not sb or not 0 <= index < len(sb["lines"]):
        raise ValueError("No such line")
    tmp = await _save_upload(file, job.dir / "narration" / "tmp" / f"upload-{index}{Path(file.filename or 'a.webm').suffix or '.webm'}")
    try:
        voice.save_user_clip(job, index, tmp)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    rec = voice.build_line(job, index, sb["lines"][index]["text"], job.load()["narration"])
    job.update(lambda s: s["narration"]["clips"].__setitem__(str(index), rec))
    return view(job)


@app.delete("/api/jobs/{job_id}/narration/line/{index}")
def delete_line(job_id: str, index: int):
    job = get_job(job_id)
    voice.clear_user_clip(job, index)
    return view(job)


@app.post("/api/jobs/{job_id}/narration/take")
async def upload_take(job_id: str, file: UploadFile = File(...), mode: str = Form("split")):
    """One long recording. split: cut it into lines using the current script. transcribe: use it to create the script."""
    job = get_job(job_id)
    ext = Path(file.filename or "take.webm").suffix or ".webm"
    ndir = job.dir / "narration"
    for old in ndir.glob("take-upload.*"):
        old.unlink()
    dest = await _save_upload(file, ndir / f"take-upload{ext}")
    if mode == "split":
        pipeline.start_split(job, dest)
    elif mode != "transcribe":
        raise ValueError("mode must be split or transcribe")
    return view(job)


# ---------------------------------------------------------------- video

@app.post("/api/jobs/{job_id}/preview")
def make_preview(job_id: str):
    job = get_job(job_id)
    if not job.storyboard():
        raise RuntimeError("Write the script first")
    pipeline.start_preview(job)
    return view(job)


class RenderBody(BaseModel):
    formats: list[str] | None = None
    captions: bool | None = None
    look: str | None = None
    looks: list[str] | None = None
    music: str | None = None


@app.post("/api/jobs/{job_id}/render")
def make_render(job_id: str, body: RenderBody | None = None):
    job = get_job(job_id)
    if not job.storyboard():
        raise RuntimeError("Write the script first")
    if body:
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        if "look" in patch and "looks" not in patch:
            patch["looks"] = [patch.pop("look")]
        patch.pop("look", None)
        if "looks" in patch:
            bad = [x for x in patch["looks"] if x != "auto" and x not in schema.LOOKS]
            if bad or not patch["looks"]:
                raise ValueError("Choose at least one look: auto, " + ", ".join(schema.LOOKS))
        if patch:
            job.update(lambda s: s["brief"].update(patch))
    pipeline.start_render(job)
    return view(job)


@app.delete("/api/jobs/{job_id}/outputs/{output_id}")
def delete_output(job_id: str, output_id: str):
    job = get_job(job_id)
    render.delete_output(job, output_id)
    return view(job)


@app.get("/files/{job_id}/{path:path}")
def files(job_id: str, path: str, download: bool = False):
    job = get_job(job_id)
    target = (job.dir / path).resolve()
    if not any(str(target).startswith(str((job.dir / d).resolve()) + "/") for d in FILE_DIRS) or not target.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(target, headers={"Cache-Control": "no-cache"}, filename=target.name if download else None)


class FreshFiles(StaticFiles):
    """The UI is plain files that change between versions, so always revalidate instead of trusting a cached copy."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/", FreshFiles(directory=config.WEB, html=True), name="web")
