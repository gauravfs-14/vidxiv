import { h, icon, replace, nextId } from "./dom.js";
import { api } from "./api.js";
import { getMode, setMode } from "./theme.js";

// ---------------------------------------------------------------- toasts

let toastHost;
export function toast(message, kind = "info", ms = 6000) {
  toastHost ||= document.getElementById("toasts");
  const t = h(`div.toast.${kind}`, { role: kind === "error" ? "alert" : "status" },
    h("span", message),
    h("button.icon-btn", { "aria-label": "Dismiss", onclick: () => t.remove() }, icon("x", 14)));
  toastHost.append(t);
  if (ms) setTimeout(() => t.remove(), ms);
  return t;
}

export const errorMessage = (e) => (e && e.message) || "Something went wrong. Try again.";

// ---------------------------------------------------------------- dialogs

function dialog(cls, content) {
  const d = h(`dialog.${cls.trim().split(/\s+/).join(".")}`, content);
  d.addEventListener("close", () => d.remove());
  d.addEventListener("click", (e) => { if (e.target === d && d.dataset.light !== undefined) d.close(); });
  document.body.append(d);
  return d;
}

export function confirmDialog({ title, body, confirm = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    let answer = false;
    const d = dialog("modal small", h("form", { method: "dialog", onsubmit: () => { answer = true; } },
      h("h2.modal-title", title),
      body && h("p.muted", body),
      h("div.modal-actions",
        h("button.btn.ghost", { type: "button", onclick: () => d.close() }, "Cancel"),
        h(`button.btn.${danger ? "danger" : "primary"}`, { type: "submit" }, confirm))));
    d.addEventListener("close", () => resolve(answer));
    d.showModal();
  });
}

/** Image viewer with caption and arrow key navigation. items: [{url, title, caption}] */
export function lightbox(items, start = 0) {
  let i = start;
  const img = h("img", { alt: "" });
  const title = h("p.lb-title");
  const caption = h("p.lb-caption");
  const count = h("span.mono.muted");
  const show = (n) => {
    i = (n + items.length) % items.length;
    const it = items[i];
    img.src = it.url;
    img.alt = it.title || "";
    replace(title, it.title || "");
    replace(caption, it.caption || "");
    replace(count, items.length > 1 ? `${i + 1} of ${items.length}` : "");
  };
  const d = dialog("lightbox", h("div.lb",
    h("div.lb-stage", img),
    h("div.lb-bar",
      h("div.lb-text", title, caption),
      h("div.lb-nav", count,
        items.length > 1 && h("button.icon-btn", { "aria-label": "Previous", onclick: () => show(i - 1) }, icon("arrow-left")),
        items.length > 1 && h("button.icon-btn", { "aria-label": "Next", onclick: () => show(i + 1) }, icon("caret-right")),
        h("button.icon-btn", { "aria-label": "Close", onclick: () => d.close() }, icon("x"))))));
  d.dataset.light = "";
  d.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") show(i - 1);
    if (e.key === "ArrowRight") show(i + 1);
  });
  show(i);
  d.showModal();
}

export function modal(cls, content, { light = false } = {}) {
  const d = dialog(`modal ${cls}`, content);
  if (light) d.dataset.light = "";
  d.showModal();
  return d;
}

// ---------------------------------------------------------------- popover

/** Opens a floating panel anchored under `anchor`. Closes on Escape or outside click. Returns close(). */
export function popover(anchor, content) {
  const pop = h("div.popover", { role: "dialog" }, content);
  document.body.append(pop);
  const place = () => {
    const r = anchor.getBoundingClientRect();
    const w = pop.offsetWidth;
    pop.style.top = Math.min(window.innerHeight - pop.offsetHeight - 8, r.bottom + 6) + "px";
    pop.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left)) + "px";
  };
  place();
  const close = () => {
    pop.remove();
    document.removeEventListener("pointerdown", away, true);
    document.removeEventListener("keydown", key, true);
    window.removeEventListener("resize", place);
    anchor.focus();
  };
  const away = (e) => { if (!pop.contains(e.target) && !anchor.contains(e.target)) close(); };
  const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  document.addEventListener("pointerdown", away, true);
  document.addEventListener("keydown", key, true);
  window.addEventListener("resize", place);
  return close;
}

// ---------------------------------------------------------------- status pieces

export function progressBar() {
  const fill = h("div.bar-fill");
  const el = h("div.bar", { role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100 }, fill);
  el.set = (value, label) => {
    const pct = Math.round((value || 0) * 100);
    el.setAttribute("aria-valuenow", pct);
    if (label) el.setAttribute("aria-label", label);
    fill.style.transform = `scaleX(${Math.max(0.02, Math.min(1, value || 0))})`;
  };
  el.set(0);
  return el;
}

export const STATE_LABEL = { idle: "Not started", running: "Running", done: "Done", error: "Failed" };

/** Small status glyph. Carries real state only: idle ring, spinner, check, warning. */
export function statusMark(kind) {
  const ic = kind === "running" ? icon("circle-notch", 14) : kind === "done" ? icon("check", 14) : kind === "error" ? icon("warning-circle", 14) : null;
  return h(`span.mark.${kind}`, ic);
}

/** Inline status text. kind: idle | running | done | error. */
export function chip(kind, text, progress) {
  const t = text ?? STATE_LABEL[kind];
  return h(`span.chip.${kind}`, kind !== "idle" && statusMark(kind), t, kind === "running" && progress != null ? ` ${Math.round(progress * 100)}%` : "");
}

/** What to do when a list or pane has nothing in it yet. Left aligned, says how to fill it. */
export function emptyState(title, body, ...actions) {
  return h("div.empty", h("p.empty-title", title), body && h("p.empty-body", body), actions.length > 0 && h("div.empty-actions", ...actions));
}

/** Placeholder block for loading content. Pass a width like "60%". */
export const skel = (w = "100%", cls = "") => h(`span.sk${cls ? "." + cls : ""}`, { style: `width:${w}` });

/** System, light and dark switch. */
export function themeControl() {
  const opts = [["system", "desktop", "System theme"], ["light", "sun", "Light theme"], ["dark", "moon", "Dark theme"]];
  const root = h("div.theme-seg", { role: "group", "aria-label": "Theme" });
  const paint = () => {
    const m = getMode();
    replace(root, opts.map(([v, ic, label]) => h("button", { type: "button", "aria-label": label, title: label, "aria-pressed": String(m === v), onclick: () => { setMode(v); paint(); } }, icon(ic, 15))));
  };
  paint();
  return root;
}

/** Live activity feed backed by GET /log/<stage>?since=N. */
export class LogFeed {
  constructor(jobId, stage, { max = 400 } = {}) {
    this.jobId = jobId;
    this.stage = stage;
    this.since = 0;
    this.max = max;
    this.busy = false;
    this.timer = null;
    this.el = h("div.log", { role: "log", tabIndex: 0, "aria-label": `${stage} activity` }, h("p.log-empty", "No activity yet."));
    this.lines = [];
  }

  async poll() {
    if (this.busy) return;
    this.busy = true;
    try {
      const r = await api.log(this.jobId, this.stage, this.since);
      if (r.next < this.since) { this.lines = []; this.since = 0; return this.poll2(); }
      this.add(r.lines, r.next);
    } catch { /* the main poll reports connection problems */ } finally { this.busy = false; }
  }

  async poll2() {
    this.busy = false;
    return this.poll();
  }

  add(lines, next) {
    this.since = next;
    if (!lines.length) return;
    const stick = this.el.scrollTop + this.el.clientHeight >= this.el.scrollHeight - 24;
    this.lines.push(...lines);
    if (this.lines.length > this.max) this.lines = this.lines.slice(-this.max);
    this.render(stick);
  }

  render(stick = true) {
    replace(this.el, this.lines.length ? this.lines.map((l) => h("div.log-line", l || " ")) : h("p.log-empty", "No activity yet."));
    if (stick) this.el.scrollTop = this.el.scrollHeight;
  }

  /** Call from the job update. Polls every second while running, once otherwise. */
  sync(running, key) {
    if (key !== this.key) { this.key = key; this.since = 0; this.lines = []; this.render(false); this.poll(); }
    if (running && !this.timer) {
      this.poll();
      this.timer = setInterval(() => this.poll(), 1000);
    } else if (!running && this.timer) {
      clearInterval(this.timer);
      this.timer = null;
      this.poll();
    }
  }

  destroy() { clearInterval(this.timer); this.timer = null; }
}

// ---------------------------------------------------------------- form helpers

export function field(label, control, hint, { id = nextId("f"), cls = "" } = {}) {
  control.id ||= id;
  return h(`div.field${cls ? "." + cls : ""}`,
    h("label", { for: control.id }, label),
    control,
    hint && h("p.hint", hint));
}

export function segmented(name, options, value, onChange, label) {
  const root = h("div.seg", { role: "radiogroup", "aria-label": label });
  const inputs = options.map((o) => {
    const input = h("input", { type: "radio", name, value: o.value, checked: o.value === value, disabled: !!o.disabled, onchange: () => onChange(o.value) });
    root.append(h("label.seg-opt", { title: o.title }, input, h("span", o.label, o.sub && h("small", o.sub))));
    return input;
  });
  root.set = (v) => inputs.forEach((i) => { i.checked = i.value === v; });
  return root;
}

export function downloadBlob(blob, name) {
  const a = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  a.remove();
}
