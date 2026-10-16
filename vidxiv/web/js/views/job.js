import { h, icon, replace } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { JobStore } from "../store.js";
import { createPaper } from "./paper.js";
import { createScript } from "./script.js";
import { createNarration } from "./narration.js";
import { createVideo } from "./video.js";
import { confirmDialog, toast, errorMessage, chip, statusMark, themeControl, skel } from "../ui.js";
import { navigate, jobHref, replaceHash, STAGE_KEYS } from "../router.js";
import { createPdfPanel } from "../pdf-panel.js";

const STAGES = [
  { key: "paper", name: "Paper", sub: "Source text and figures", make: createPaper },
  { key: "script", name: "Script", sub: "One line of narration per scene. Edits save as you type.", make: createScript },
  { key: "narration", name: "Narration", sub: "Synthetic voice or your own recordings, line by line.", make: createNarration },
  { key: "video", name: "Video", sub: "Pick formats and a look, then render.", make: createVideo },
];

/** Project workspace: a raised stage card beside the content, centered with it, and one stage at a time. */
export function mountJob(root, id, initialStage) {
  const store = new JobStore(id);
  let stage = initialStage;
  let chosen = !!initialStage; // false until the URL or the user picks a stage

  // ---------------------------------------------------------------- rail
  const title = h("h1.rail-title", skel("80%"));
  const meta = h("p.rail-meta");
  const del = h("button.btn.ghost.small.danger-text", { type: "button", onclick: async () => {
    const ok = await confirmDialog({ title: "Delete this project?", body: "The paper, script, narration and videos are removed from this computer.", confirm: "Delete project", danger: true });
    if (!ok) return;
    try { await api.deleteJob(id); navigate("#/"); } catch (e) { toast(errorMessage(e), "error"); }
  } }, icon("trash", 14), "Delete project");

  const hosts = STAGES.map((def, i) => {
    const mark = h("span.step-mark", statusMark("idle"));
    const sub = h("span.step-sub", "");
    const bar = h("span.step-bar", { hidden: true }, h("span"));
    const link = h("a.step", { href: jobHref(id, def.key) }, mark, h("span.step-text", h("span.step-name", def.name), sub), bar);
    const chipSlot = h("span.pane-chip");
    const actions = h("div.pane-actions");
    const body = h("div.pane-body");
    const el = h("section.pane", { id: `pane-${def.key}`, hidden: true, "aria-labelledby": `pane-${def.key}-h` },
      h("header.pane-head",
        h("div.pane-titles", h("h2.pane-title", { id: `pane-${def.key}-h` }, def.name), h("p.pane-sub", def.sub)),
        h("div.pane-side", chipSlot, actions)),
      body);
    const showFns = [];
    const host = {
      key: def.key, el, body, actions, link,
      onShow: (fn) => showFns.push(fn),
      shown: () => showFns.forEach((f) => f()),
      /** kind: idle | running | done | error. text is the short status shown in the rail and header. */
      setStatus(kind, text, progress) {
        replace(mark, statusMark(kind));
        link.dataset.state = kind;
        sub.textContent = text || "";
        replace(chipSlot, chip(kind, text, progress));
        bar.hidden = !(kind === "running" && progress != null);
        bar.firstChild.style.transform = `scaleX(${Math.max(0.03, Math.min(1, progress || 0))})`;
      },
      go: (key) => { replaceHash(jobHref(id, key)); setStage(key); },
      openPdf: (page) => pdf.open(page),
    };
    host.setStatus("idle", "");
    return host;
  });
  const part = hosts.map((host, i) => ({ host, ...STAGES[i].make({ store, sec: host }) }));

  // Previous and next at the bottom of every stage, so the way forward does not depend on spotting the sidebar.
  const READY = [
    (j) => j.stages.ingest.state === "done" || "Wait for the paper to finish loading.",
    (j) => !!j.storyboard || "Write the script to continue.",
    () => true,
    () => true,
  ];
  const navs = hosts.map((host, i) => {
    const prev = i > 0 && h("button.btn", { type: "button", onclick: () => host.go(STAGES[i - 1].key) }, icon("arrow-left", 15), `Back to ${STAGES[i - 1].name.toLowerCase()}`);
    const next = i < STAGES.length - 1 && h("button.btn.primary", { type: "button", onclick: () => host.go(STAGES[i + 1].key) }, `Next: ${STAGES[i + 1].name.toLowerCase()}`, icon("arrow-right", 15));
    const hint = h("p.nav-hint", { "aria-live": "polite" });
    host.el.append(h("footer.pane-nav", h("div.nav-prev", prev), hint, h("div.nav-next", next)));
    return { prev, next, hint };
  });
  function paintNav(job) {
    navs.forEach((n, i) => {
      if (!n.next) return;
      const r = READY[i](job);
      n.next.disabled = r !== true;
      n.hint.textContent = r === true ? "" : r;
    });
  }

  const appEl = h("div.app");
  const pdf = createPdfPanel({ jobId: id, app: appEl });
  const pdfBtn = pdf.button();
  const rail = h("aside.rail", { "aria-label": "Project" },
    h("a.brand", { href: "#/", "aria-label": "vidxiv, all projects" }, h("span.brand-mark", { "aria-hidden": "true" }), "vidxiv"),
    h("a.back", { href: "#/" }, icon("arrow-left", 14), "All projects"),
    h("div.rail-project", title, meta),
    h("nav", { "aria-label": "Stages" }, h("ol.steps", hosts.map((x) => h("li", x.link)))),
    h("div.rail-tools", pdfBtn),
    h("div.rail-foot", del, themeControl()));

  const loading = h("div.pane-loading", { "aria-busy": "true", "aria-label": "Loading project" },
    skel("30%", "tall"), skel("55%"), h("div.sk-block"), h("div.sk-block"));
  const missing = h("div.pane-missing", { hidden: true },
    h("h2.pane-title", "Project not found"),
    h("p.muted", "It may have been deleted. Go back to your projects."),
    h("a.btn", { href: "#/" }, "All projects"));
  const main = h("div.workspace", loading, ...hosts.map((x) => x.el), missing);
  appEl.append(rail, main, pdf.el);
  root.append(appEl);

  function setStage(key) {
    if (!STAGE_KEYS.includes(key)) return;
    const changed = key !== stage;
    stage = key;
    chosen = true;
    hosts.forEach((x) => {
      const on = x.key === key;
      x.el.hidden = !on || !store.job;
      if (on) x.link.setAttribute("aria-current", "page"); else x.link.removeAttribute("aria-current");
    });
    if (changed) window.scrollTo(0, 0);
    hosts.find((x) => x.key === key)?.shown();
  }

  let first = true;
  const unsub = store.subscribe((job, prev) => {
    if (!job) {
      rail.querySelector(".rail-project").hidden = true;
      rail.querySelector("nav").hidden = true;
      pdfBtn.hidden = true;
      del.hidden = true;
      hosts.forEach((x) => { x.el.hidden = true; });
      loading.hidden = true; missing.hidden = false;
      return;
    }
    pdfBtn.hidden = !job.hasPdf;
    document.title = `${job.title} - vidxiv`;
    replace(title, job.title || "Untitled");
    replace(meta, job.source.type === "arxiv" ? `arXiv:${job.source.id}` : job.source.filename || "PDF upload");
    part.forEach((p) => p.update(job, prev));
    paintNav(job);

    if (first) {
      first = false;
      loading.hidden = true;
      if (!chosen) {
        const clips = job.narration.clips || {};
        const n = job.storyboard?.lines.length || 0;
        const narrated = n > 0 && Object.keys(clips).filter((k) => clips[k].source && !clips[k].stale).length === n;
        const current = job.stages.ingest.state !== "done" ? "paper" : !job.storyboard ? "script" : !narrated ? "narration" : "video";
        stage = null;
        replaceHash(jobHref(id, current));
        setStage(current);
      } else setStage(stage);
    } else if (prev && prev.stages.ingest.state === "running" && job.stages.ingest.state === "done" && stage === "paper") {
      hosts[0].go("script");
    }
  });
  store.start();
  env.refresh();

  return {
    setStage,
    unmount() { unsub(); store.stop(); pdf.destroy(); part.forEach((p) => p.destroy?.()); document.title = "vidxiv"; },
  };
}
