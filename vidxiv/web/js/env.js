import { api } from "./api.js";

/** Shared view of GET /api/status. Polls faster while an install is running. */
export const env = {
  data: null,
  subs: new Set(),
  timer: null,

  subscribe(fn) { this.subs.add(fn); if (this.data) fn(this.data); return () => this.subs.delete(fn); },

  async refresh() {
    clearTimeout(this.timer);
    try {
      this.data = await api.status();
      this.subs.forEach((fn) => fn(this.data));
    } catch { /* offline banner covers it */ }
    const installing = this.data && Object.values(this.data.setup || {}).some((s) => s.state === "running");
    if (installing || this.subs.size === 0) {
      if (installing) this.timer = setTimeout(() => this.refresh(), 1500);
    }
  },

  async setup(what) {
    const r = await api.setup(what);
    this.data = { ...(this.data || {}), setup: { ...(this.data?.setup || {}), [what]: r } };
    this.subs.forEach((fn) => fn(this.data));
    this.timer = setTimeout(() => this.refresh(), 1000);
    return r;
  },

  agentFound(name) { return !!this.data?.agents?.[name]?.found; },
  get engineReady() { return !!this.data?.tools?.engine; },
  get ffmpeg() { return !!this.data?.tools?.ffmpeg; },
};
