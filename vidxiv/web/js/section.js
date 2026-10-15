import { h, icon, replace, nextId } from "./dom.js";

const key = (jobId) => `vidxiv-open-${jobId}`;
const readOpen = (jobId) => { try { return JSON.parse(sessionStorage.getItem(key(jobId))) || null; } catch { return null; } };

/** Collapsible numbered section. Open state survives re-renders and reloads of the same job. */
export function createSection({ jobId, n, title }) {
  const bodyId = nextId("sec");
  const chipSlot = h("span.sec-chip");
  const summary = h("span.sec-summary");
  const caret = icon("caret-down", 18);
  const btn = h("button.sec-head", { type: "button", "aria-expanded": "false", "aria-controls": bodyId },
    h("span.sec-num", String(n)),
    h("span.sec-title", h("span.sec-name", title), summary),
    chipSlot, h("span.sec-caret", caret));
  const inner = h("div.sec-inner");
  const body = h("div.sec-body", { id: bodyId, role: "region", "aria-label": title, inert: true }, inner);
  const el = h("section.sec", { id: `sec-${n}`, dataset: { n } }, h("h2.sec-h", btn), body);
  const openFns = [];
  let open = false;

  const sec = {
    el, body: inner, n,
    setChip(node) { replace(chipSlot, node); },
    setSummary(t) { summary.textContent = t || ""; },
    isOpen: () => open,
    onOpen(fn) { openFns.push(fn); },
    setOpen(v, persist = true) {
      if (v === open) return;
      open = v;
      el.classList.toggle("open", v);
      btn.setAttribute("aria-expanded", String(v));
      body.inert = !v;
      if (persist) {
        const st = readOpen(jobId) || {};
        st[n] = v;
        sessionStorage.setItem(key(jobId), JSON.stringify(st));
      }
      if (v) requestAnimationFrame(() => openFns.forEach((f) => f()));
    },
    /** Stored preference, or undefined when the user has not touched this section. */
    stored: () => readOpen(jobId)?.[n],
  };
  btn.addEventListener("click", () => sec.setOpen(!open));
  return sec;
}
