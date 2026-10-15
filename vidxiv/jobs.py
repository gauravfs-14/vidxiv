"""Job store. A job is a directory with a job.json state file; stages run in background threads."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from . import config

STAGES = ("ingest", "script", "narration", "preview", "render")
_locks: dict[str, threading.RLock] = {}
_cancel: dict[str, threading.Event] = {}
_procs: dict[str, set[subprocess.Popen]] = {}
_guard = threading.Lock()


class Cancelled(Exception):
    pass


def _lock(job_id: str) -> threading.RLock:
    with _guard:
        return _locks.setdefault(job_id, threading.RLock())


def slug(s: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n] or "video"


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.dir = config.JOBS / job_id
        self.file = self.dir / "job.json"

    # ---- state ----
    def exists(self) -> bool:
        return self.file.exists()

    def load(self) -> dict:
        with _lock(self.id):
            return json.loads(self.file.read_text())

    def save(self, state: dict) -> None:
        with _lock(self.id):
            state["updated"] = time.time()
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, self.file)

    def update(self, fn: Callable[[dict], None]) -> dict:
        with _lock(self.id):
            s = self.load()
            fn(s)
            self.save(s)
            return s

    def patch(self, **kv) -> dict:
        return self.update(lambda s: s.update(kv))

    # ---- paths ----
    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def storyboard_path(self) -> Path:
        return self.dir / "storyboard.json"

    def storyboard(self) -> dict | None:
        return json.loads(self.storyboard_path.read_text()) if self.storyboard_path.exists() else None

    def paper(self) -> dict | None:
        p = self.dir / "paper" / "paper.json"
        return json.loads(p.read_text()) if p.exists() else None

    def figures(self) -> list[dict]:
        p = self.dir / "paper" / "figures.json"
        return json.loads(p.read_text()) if p.exists() else []

    # ---- logging ----
    def log(self, stage: str, msg: str) -> None:
        with _lock(self.id), open(self.path("logs", f"{stage}.log"), "a") as f:
            f.write(msg.rstrip("\n") + "\n")

    def read_log(self, stage: str, since: int = 0) -> dict:
        p = self.dir / "logs" / f"{stage}.log"
        lines = p.read_text().splitlines() if p.exists() else []
        return {"lines": lines[since:], "next": len(lines)}

    def set_stage(self, stage: str, **kv) -> None:
        def f(s):
            s["stages"].setdefault(stage, {}).update(kv)
        self.update(f)

    def progress(self, stage: str, value: float, message: str | None = None) -> None:
        kv = {"progress": round(max(0.0, min(1.0, value)), 3)}
        if message is not None:
            kv["message"] = message
        self.set_stage(stage, **kv)

    # ---- cancellation and subprocesses ----
    def cancelled(self) -> bool:
        return _cancel.setdefault(self.id, threading.Event()).is_set()

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled()

    def run(self, cmd: list[str], stage: str, cwd: Path | None = None, env: dict | None = None,
            on_line: Callable[[str], None] | None = None, timeout: float | None = None) -> int:
        """Run a subprocess, stream its output into the stage log, and kill it on cancel or timeout."""
        self.check()
        p = subprocess.Popen(cmd, cwd=cwd or self.dir, env={**os.environ, **(env or {})}, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True)
        with _guard:
            _procs.setdefault(self.id, set()).add(p)
        timer = None
        if timeout:
            timer = threading.Timer(timeout, lambda: _kill(p))
            timer.start()
        try:
            assert p.stdout
            for raw in p.stdout:
                line = raw.rstrip("\n")
                if on_line:
                    on_line(line)
                else:
                    self.log(stage, line)
            p.wait()
        finally:
            if timer:
                timer.cancel()
            with _guard:
                _procs.get(self.id, set()).discard(p)
        self.check()
        if timeout and p.returncode in (-9, -15):
            raise TimeoutError(f"{cmd[0]} timed out after {int(timeout)}s")
        return p.returncode

    def cancel(self) -> None:
        _cancel.setdefault(self.id, threading.Event()).set()
        with _guard:
            for p in list(_procs.get(self.id, ())):
                _kill(p)


def _kill(p: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(p.pid), 15)
    except (ProcessLookupError, PermissionError):
        pass


def create(source: dict, title: str = "Untitled") -> Job:
    job_id = time.strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    job = Job(job_id)
    job.dir.mkdir(parents=True)
    state = {
        "id": job_id, "created": time.time(), "title": title, "source": source,
        "brief": default_brief(),
        "narration": {"engine": "auto", "voice": "af_heart", "speed": 0.95, "clips": {}},
        "stages": {s: {"state": "idle", "progress": 0, "message": ""} for s in STAGES},
        "outputs": [], "previews": [],
    }
    job.save(state)
    return job


def default_brief() -> dict:
    return {
        "length": "short", "formats": ["Vertical"], "palette": "paper", "captions": True, "music": "warm",
        "agent": "auto", "model": "", "instructions": "", "scriptMode": "generate", "providedText": "",
    }


def list_jobs() -> list[dict]:
    out = []
    for d in sorted(config.JOBS.iterdir(), reverse=True):
        j = Job(d.name)
        if j.exists():
            try:
                s = j.load()
            except Exception:
                continue
            out.append({"id": s["id"], "title": s["title"], "created": s["created"], "updated": s.get("updated"),
                        "source": s["source"], "stages": {k: v["state"] for k, v in s["stages"].items()},
                        "outputs": s.get("outputs", [])})
    return out


def delete(job_id: str) -> None:
    j = Job(job_id)
    j.cancel()
    shutil.rmtree(j.dir, ignore_errors=True)


@contextmanager
def stage(job: Job, name: str):
    """Mark a stage running, then done or error. Re-raises nothing: errors are stored in the job."""
    cur = job.load()["stages"].get(name, {})
    if cur.get("state") == "running":
        raise RuntimeError(f"{name} is already running")
    _cancel.setdefault(job.id, threading.Event()).clear()
    job.set_stage(name, state="running", progress=0, message="Starting", error=None, started=time.time(), finished=None, pid=os.getpid())
    try:
        yield
        job.set_stage(name, state="done", progress=1, message="Done", finished=time.time())
    except Cancelled:
        job.log(name, "Cancelled.")
        job.set_stage(name, state="idle", progress=0, message="Cancelled", finished=time.time())
    except Exception as e:  # noqa: BLE001
        job.log(name, traceback.format_exc())
        job.set_stage(name, state="error", message=str(e)[:400], error=str(e)[:2000], finished=time.time())


def background(job: Job, name: str, fn: Callable[[], None]) -> None:
    """Run fn as stage `name` on a daemon thread. Raises if that stage is already running."""
    for other, st in job.load()["stages"].items():
        if st.get("state") == "running":
            raise RuntimeError(f"{other.capitalize()} is still running. Wait for it to finish or cancel it first.")
    job.set_stage(name, state="running", progress=0, message="Queued", error=None)

    def target():
        # stage() would refuse because we just set running; reset to idle first so it takes ownership
        job.set_stage(name, state="idle")
        job.path("logs", f"{name}.log").write_text("")
        with stage(job, name):
            fn()

    threading.Thread(target=target, daemon=True, name=f"{job.id}:{name}").start()


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_interrupted() -> None:
    """After a restart, stages left 'running' by a process that no longer exists were killed with it."""
    for d in config.JOBS.iterdir():
        j = Job(d.name)
        if not j.exists():
            continue
        try:
            def fix(s):
                for k, v in s["stages"].items():
                    if v.get("state") == "running" and not _alive(v.get("pid")):
                        v.update(state="error", message="Interrupted by a restart", error="Interrupted by a restart")
            j.update(fix)
        except Exception:
            pass
