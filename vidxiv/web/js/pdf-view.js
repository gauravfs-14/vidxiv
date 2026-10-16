import * as pdfjs from "../vendor/pdfjs/pdf.min.mjs";
import { h, icon, replace } from "./dom.js";

pdfjs.GlobalWorkerOptions.workerSrc = "/vendor/pdfjs/pdf.worker.min.mjs";

const ZOOMS = [0.5, 0.65, 0.8, 1, 1.25, 1.5, 2, 3];

/**
 * PDF reader built on pdf.js, so it works in any browser, including ones with no built-in PDF viewer.
 * Pages render lazily as they scroll into view. Text is selectable.
 */
export function createPdfView(url) {
  let doc = null;
  let sizes = []; // [w, h] of each page at scale 1
  let zoom = 1; // relative to fit width
  let fit = 1;
  let current = 1;
  let token = 0;
  const slots = [];
  const rendered = new Map();

  const pageInput = h("input.pdf-page-input", { type: "number", min: 1, value: 1, "aria-label": "Page number", onchange: () => goto(+pageInput.value) });
  const total = h("span.pdf-total", "");
  const zoomOut = h("button.icon-btn", { type: "button", "aria-label": "Zoom out", onclick: () => step(-1) }, icon("minus", 15));
  const zoomIn = h("button.icon-btn", { type: "button", "aria-label": "Zoom in", onclick: () => step(1) }, icon("plus", 15));
  const zoomOut2 = zoomOut;
  const level = h("button.pdf-level", { type: "button", "aria-label": "Fit width", onclick: () => { zoom = 1; layout(); } }, "100%");
  const bar = h("div.pdf-bar",
    h("button.icon-btn", { type: "button", "aria-label": "Previous page", onclick: () => goto(current - 1) }, icon("caret-up", 15)),
    h("button.icon-btn", { type: "button", "aria-label": "Next page", onclick: () => goto(current + 1) }, icon("caret-down", 15)),
    h("span.pdf-pager", pageInput, h("span", "of"), total),
    h("span.pdf-zoom", zoomOut, level, zoomIn));
  const scroller = h("div.pdf-scroll", { tabindex: 0, "aria-label": "PDF pages" });
  const status = h("p.pdf-status", "Loading the PDF");
  const el = h("div.pdf-view", bar, scroller, status);

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const i = +e.target.dataset.i;
      if (e.isIntersecting) draw(i);
    }
    // the page that covers the middle of the viewport is the current one
    const mid = scroller.getBoundingClientRect().top + scroller.clientHeight / 2;
    const hit = slots.findIndex((s) => { const r = s.getBoundingClientRect(); return r.top <= mid && r.bottom >= mid; });
    if (hit >= 0 && hit + 1 !== current) { current = hit + 1; if (document.activeElement !== pageInput) pageInput.value = current; }
  }, { root: scroller, rootMargin: "300px 0px", threshold: [0, 0.01] });

  async function load() {
    try {
      doc = await pdfjs.getDocument({ url }).promise;
      for (let i = 1; i <= doc.numPages; i++) {
        const v = (await doc.getPage(i)).getViewport({ scale: 1 });
        sizes.push([v.width, v.height]);
      }
      total.textContent = String(doc.numPages);
      pageInput.max = doc.numPages;
      status.hidden = true;
      build();
    } catch (e) {
      replace(status, "The PDF could not be shown. ", h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, "Open it in a new tab"), ".");
    }
  }

  function build() {
    io.disconnect();
    rendered.clear();
    token++;
    slots.length = 0;
    const wide = Math.max(...sizes.map((s) => s[0]));
    fit = Math.max(0.3, (scroller.clientWidth - 24) / wide);
    replace(scroller, sizes.map((s, i) => {
      const slot = h("div.pdf-slot", { dataset: { i }, style: `width:${Math.floor(s[0] * fit * zoom)}px;height:${Math.floor(s[1] * fit * zoom)}px` });
      slots.push(slot);
      io.observe(slot);
      return slot;
    }));
    level.textContent = `${Math.round(zoom * 100)}%`;
  }

  function layout() {
    const keep = current;
    build();
    goto(keep, false);
  }

  function step(dir) {
    const i = ZOOMS.findIndex((z) => z >= zoom - 1e-6);
    zoom = ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, (i < 0 ? ZOOMS.length - 1 : i) + dir))];
    layout();
  }

  async function draw(i) {
    if (!doc || rendered.get(i) === token) return;
    rendered.set(i, token);
    const my = token;
    const page = await doc.getPage(i + 1);
    if (my !== token) return;
    const scale = fit * zoom;
    const vp = page.getViewport({ scale });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const canvas = h("canvas");
    canvas.width = Math.floor(vp.width * dpr);
    canvas.height = Math.floor(vp.height * dpr);
    canvas.style.width = `${Math.floor(vp.width)}px`;
    canvas.style.height = `${Math.floor(vp.height)}px`;
    await page.render({ canvasContext: canvas.getContext("2d"), viewport: vp, transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null }).promise;
    if (my !== token) return;
    const text = h("div.textLayer");
    text.style.setProperty("--scale-factor", String(scale));
    text.style.setProperty("--user-unit", "1");
    slots[i].replaceChildren(canvas, text);
    try {
      await new pdfjs.TextLayer({ textContentSource: page.streamTextContent(), container: text, viewport: vp }).render();
    } catch { /* text selection is optional */ }
  }

  function goto(n, smooth = true) {
    if (!slots.length) return;
    current = Math.max(1, Math.min(slots.length, Math.round(n) || 1));
    pageInput.value = current;
    scroller.scrollTo({ top: slots[current - 1].offsetTop - 8, behavior: smooth ? "smooth" : "auto" });
  }

  let rt;
  const ro = new ResizeObserver(() => {
    if (!doc || !scroller.clientWidth) return;
    clearTimeout(rt);
    rt = setTimeout(() => {
      const w = Math.max(...sizes.map((s) => s[0]));
      if (Math.abs((scroller.clientWidth - 24) / w - fit) > 0.02) layout();
    }, 150);
  });
  ro.observe(scroller);
  load();
  return { el, goto, destroy() { io.disconnect(); ro.disconnect(); doc?.destroy(); } };
}
