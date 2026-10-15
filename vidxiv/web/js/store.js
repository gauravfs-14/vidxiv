import { api } from "./api.js";

export const isRunning = (job) => !!job && Object.values(job.stages).some((s) => s.state === "running");

/**
 * Holds the latest job view and keeps it fresh. Polls every second while anything runs (and for two
 * seconds after a start call), otherwise every five seconds while the tab is visible.
 */
export class JobStore {
  constructor(id) {
    this.id = id;
    this.job = null;
    this.json = "";
    this.fastUntil = 0;
    this.timer = null;
    this.stopped = false;
    this.inflight = false;
    this.subs = new Set();
    this.onVis = () => { if (!document.hidden) this.poll(); };
    document.addEventListener("visibilitychange", this.onVis);
  }

  subscribe(fn) { this.subs.add(fn); if (this.job) fn(this.job, null); return () => this.subs.delete(fn); }

  start() { return this.poll(); }

  stop() {
    this.stopped = true;
    clearTimeout(this.timer);
    document.removeEventListener("visibilitychange", this.onVis);
  }

  /** Call after any request that starts work, so the UI catches the first progress quickly. */
  bump() {
    this.fastUntil = Date.now() + 2000;
    this.poll();
  }

  /** Feed a job view that came back from a POST/PUT so the UI updates without waiting for a poll. */
  apply(view) {
    if (this.stopped || !view || view.id !== this.id) return;
    if (this.job && view.updated < this.job.updated) return;
    const json = JSON.stringify(view);
    if (json === this.json) return;
    const prev = this.job;
    this.job = view;
    this.json = json;
    this.subs.forEach((fn) => fn(view, prev));
  }

  async poll() {
    if (this.stopped || this.inflight) return;
    clearTimeout(this.timer);
    this.inflight = true;
    let delay = 5000;
    try {
      this.apply(await api.job(this.id));
      this.error = null;
    } catch (e) {
      this.error = e;
      if (e.status === 404) { this.subs.forEach((fn) => fn(null, this.job, e)); this.stopped = true; return; }
      delay = 3000;
    } finally {
      this.inflight = false;
    }
    if (this.stopped) return;
    if (isRunning(this.job) || Date.now() < this.fastUntil) delay = Math.min(delay, 1000);
    else if (document.hidden) return; // resumes on visibilitychange
    this.timer = setTimeout(() => this.poll(), delay);
  }
}
