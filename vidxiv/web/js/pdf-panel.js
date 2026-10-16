import { h, icon } from "./dom.js";
import { createPdfView } from "./pdf-view.js";

const KEY = "vidxiv.pdf";
const MIN = 340;
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };

/**
 * Docked PDF viewer. pdf.js (vendored, Apache-2.0) does the rendering, so it works offline and in browsers
 * that have no PDF viewer of their own. The panel is a third column on wide screens and a drawer on narrow ones.
 */
export function createPdfPanel({ jobId, app }) {
  const url = `/files/${encodeURIComponent(jobId)}/paper/paper.pdf`;
  const saved = load();
  let width = Math.max(MIN, saved.w || 520);
  let isOpen = false;
  let view = null;

  const close = h("button.icon-btn", { type: "button", "aria-label": "Close PDF", onclick: () => toggle(false) }, icon("x", 15));
  const tab = h("a.btn.small.ghost", { href: url, target: "_blank", rel: "noopener noreferrer" }, icon("arrow-square-out", 14), "New tab");
  const body = h("div.pdf-body");
  const grip = h("div.pdf-grip", { role: "separator", "aria-orientation": "vertical", "aria-label": "Resize the PDF panel", tabindex: 0 });
  const el = h("aside.pdf-panel", { hidden: true, "aria-label": "Paper PDF" },
    grip, h("header.pdf-head", h("h2", "Paper PDF"), h("div.pdf-head-actions", tab, close)), body);

  const clampW = (w) => Math.max(MIN, Math.min(w, Math.round(window.innerWidth * 0.62)));
  const paintWidth = () => app.style.setProperty("--pdf-w", `${clampW(width)}px`);
  paintWidth();

  grip.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    grip.setPointerCapture(e.pointerId);
    app.classList.add("pdf-resizing");
    const move = (ev) => { width = clampW(window.innerWidth - ev.clientX - 16); paintWidth(); };
    const up = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      app.classList.remove("pdf-resizing");
      localStorage.setItem(KEY, JSON.stringify({ ...load(), w: width }));
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
  });
  grip.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    width = clampW(width + (e.key === "ArrowLeft" ? 40 : -40));
    paintWidth();
    localStorage.setItem(KEY, JSON.stringify({ ...load(), w: width }));
  });
  window.addEventListener("resize", paintWidth);

  function toggle(open = !isOpen, page) {
    isOpen = open;
    el.hidden = !open;
    app.classList.toggle("pdf-open", open);
    localStorage.setItem(KEY, JSON.stringify({ ...load(), open }));
    if (open) show(page);
    btns.forEach((b) => { b.setAttribute("aria-pressed", String(open)); });
  }

  /** Load the file on first use. A page number jumps the viewer to that page. */
  function show(page) {
    if (!view) {
      view = createPdfView(url);
      body.append(view.el);
      if (page) setTimeout(() => view.goto(page, false), 700);
    } else if (page) view.goto(page);
  }

  const btns = [];
  /** A button that toggles the panel, for the sidebar. */
  function button() {
    const b = h("button.btn.pdf-toggle", { type: "button", "aria-pressed": "false", onclick: () => toggle() }, icon("file-pdf", 15), "View PDF");
    btns.push(b);
    return b;
  }

  if (saved.open && window.innerWidth >= 1180) toggle(true);
  return { el, button, open: (page) => toggle(true, page), toggle, destroy() { window.removeEventListener("resize", paintWidth); view?.destroy(); } };
}
