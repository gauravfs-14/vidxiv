import { h, icon, replace, debounce, fmtDuration, fmtSize, timeAgo } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { progressBar, LogFeed, field, toast, errorMessage, emptyState, confirmDialog } from "../ui.js";
import { installEngineHint } from "./script.js";

const FORMATS = [
  { value: "Vertical", ratio: "9 / 16", name: "Vertical", sub: "9:16 for Shorts and Reels" },
  { value: "Landscape", ratio: "16 / 9", name: "Landscape", sub: "16:9 for YouTube" },
  { value: "Square", ratio: "1 / 1", name: "Square", sub: "1:1 for feeds" },
];
const LOOKS = [
  { value: "auto", name: "Auto", suits: "The script agent picks the look that suits the paper." },
  { value: "studio", name: "Studio", suits: "General, machine learning and software." },
  { value: "night", name: "Night", suits: "Systems, security and dense results." },
  { value: "swiss", name: "Swiss", suits: "Math, physics and theory." },
  { value: "cobalt", name: "Cobalt", suits: "Short vertical videos." },
];
const MUSIC = [["warm", "Warm"], ["bright", "Bright"], ["calm", "Calm"], ["night", "Night"], ["none", "No music"]];

/** CSS-drawn preview of a look: ground, headline, secondary line and accent. */
function swatch(look) {
  if (look === "auto") return h("span.swatch.auto", { "aria-hidden": "true" }, h("i.s-studio"), h("i.s-night"), h("i.s-swiss"), h("i.s-cobalt"));
  return h(`span.swatch.${look}`, { "aria-hidden": "true" }, h("b", "Aa"), h("u"), h("i"));
}

export function createVideo({ store, sec }) {
  const cfg = { formats: ["Vertical"], looks: ["auto"], captions: true, music: "warm" };
  let init = false, dirty = false;

  const save = debounce(async () => {
    try { await api.putBrief(store.id, { ...cfg, look: cfg.looks[0] || "auto" }); dirty = false; } catch (e) { toast(`Settings were not saved. ${errorMessage(e)}`, "error"); }
  }, 500);
  const changed = () => { dirty = true; save(); paint(); };

  // formats
  const fmtInputs = FORMATS.map((f) => {
    const input = h("input", { type: "checkbox", value: f.value, onchange: () => {
      cfg.formats = fmtInputs.filter((i) => i.checked).map((i) => i.value);
      changed();
    } });
    input.el = h("label.opt.fmt", input,
      h("span.glyph-wrap", h("span.glyph", { style: `aspect-ratio:${f.ratio}` })),
      h("span.opt-text", h("span.opt-name", f.name), h("span.opt-sub", f.sub)),
      h("span.opt-check", icon("check", 14)));
    return input;
  });
  const fmtGroup = h("fieldset.group", h("legend", "Formats"), h("div.opt-list", fmtInputs.map((i) => i.el)));
  const fmtErr = h("p.field-error", { role: "alert", hidden: true }, "Choose at least one format to render.");

  // look
  const lookInputs = LOOKS.map((l) => {
    const input = h("input", { type: "checkbox", value: l.value, onchange: () => {
      cfg.looks = lookInputs.filter((i) => i.checked).map((i) => i.value);
      changed();
    } });
    input.el = h("label.opt.look", input, swatch(l.value),
      h("span.opt-text", h("span.opt-name", l.name), h("span.opt-sub", l.suits)),
      h("span.opt-check", icon("check", 14)));
    return input;
  });
  const lookGroup = h("fieldset.group", h("legend", "Looks"), h("p.hint.group-hint", "Pick several to get one video per look from the same narration."), h("div.opt-list", lookInputs.map((i) => i.el)));
  const lookErr = h("p.field-error", { role: "alert", hidden: true }, "Choose at least one look to render.");

  // captions, music
  const captions = h("input", { type: "checkbox", role: "switch", onchange: () => { cfg.captions = captions.checked; changed(); } });
  const capLabel = h("label.switch", captions, h("span.track", h("span.thumb")), h("span", "Captions"));
  const music = h("select", { onchange: () => { cfg.music = music.value; changed(); } }, MUSIC.map(([v, l]) => h("option", { value: v }, l)));

  // run
  const bar = progressBar();
  const msg = h("p.progress-msg", { "aria-live": "polite" });
  const btn = h("button.btn.primary.lg", { type: "button", onclick: () => render() }, icon("film-strip", 16), "Render video");
  const why = h("p.hint");
  const count = h("p.hint.render-count");
  const cancel = h("button.btn.small", { type: "button", onclick: async () => { try { await api.cancel(store.id); store.bump(); } catch (e) { toast(errorMessage(e), "error"); } } }, icon("x", 14), "Cancel");
  const progress = h("div.progress-row", { hidden: true }, bar, cancel);
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const feed = new LogFeed(store.id, "render");
  const log = h("details.disclosure", { hidden: true }, h("summary", "Render log"), feed.el);
  const outputs = h("div.outputs");
  const hintSlot = h("div.hint-slot");
  const empty = h("div", emptyState("No video yet", "Choose formats and looks, then render. Lines without narration are narrated first."));

  async function render() {
    save.cancel();
    btn.disabled = true;
    try {
      store.apply(await api.render(store.id, { ...cfg }));
      dirty = false;
      store.bump();
    } catch (e) { toast(errorMessage(e), "error"); }
    paint();
  }

  sec.body.append(h("div.video-grid",
    h("div.video-form",
      fmtGroup, fmtErr, lookGroup, lookErr,
      h("div.opt-row", capLabel, field("Music", music)),
      h("div.render-row", btn, count, why, hintSlot)),
    h("div.video-results", progress, msg, errBox, log, empty, outputs)));

  let lastJob = null;
  function paint() { if (lastJob) update(lastJob); }
  let outSig = "";

  function update(job) {
    lastJob = job;
    if (!init || (!dirty && JSON.stringify(cfg) !== JSON.stringify(pick(job.brief)))) {
      init = true;
      Object.assign(cfg, pick(job.brief));
    }
    fmtInputs.forEach((i) => (i.checked = cfg.formats.includes(i.value)));
    lookInputs.forEach((i) => (i.checked = cfg.looks.includes(i.value)));
    captions.checked = cfg.captions;
    if (document.activeElement !== music) music.value = cfg.music;

    const st = job.stages.render;
    const running = st.state === "running";
    const has = !!job.storyboard;
    const done = job.outputs.length > 0 && !running;
    sec.setStatus(running ? "running" : st.state === "error" ? "error" : done ? "done" : "idle",
      running ? "Rendering" : st.state === "error" ? "Failed" : done ? `${job.outputs.length} video${job.outputs.length === 1 ? "" : "s"}` : has ? "Ready" : "Waiting", st.progress);

    const noFormat = cfg.formats.length === 0 || cfg.looks.length === 0;
    fmtErr.hidden = cfg.formats.length > 0;
    lookErr.hidden = cfg.looks.length > 0;
    const total = cfg.formats.length * cfg.looks.length;
    btn.lastChild.textContent = job.outputs.length ? "Render again" : "Render video";
    count.textContent = total ? `${cfg.looks.length} look${cfg.looks.length === 1 ? "" : "s"} x ${cfg.formats.length} format${cfg.formats.length === 1 ? "" : "s"} = ${total} video${total === 1 ? "" : "s"}. ${job.outputs.length ? "Earlier renders are kept." : ""}` : "";
    const reasons = [];
    if (!has) reasons.push("Write the script first.");
    else if (env.data && !env.engineReady) reasons.push("The video engine is not installed.");
    if (env.data && !env.ffmpeg) reasons.push("ffmpeg is missing.");
    btn.disabled = running || noFormat || reasons.length > 0;
    why.textContent = reasons.join(" ");
    replace(hintSlot, has ? installEngineHint() : null);
    progress.hidden = !running;
    bar.set(st.progress, "Rendering video");
    msg.textContent = running ? st.message : "";
    errBox.hidden = st.state !== "error";
    if (st.state === "error") replace(errBox, h("p", h("strong", "The video was not rendered. "), st.error || st.message, " Open the render log for details, fix the cause, then render again."));
    log.hidden = st.state === "idle" && !feed.lines.length;
    feed.sync(running, st.started);
    empty.hidden = job.outputs.length > 0 || running || st.state === "error";

    const sig = JSON.stringify(job.outputs);
    if (sig !== outSig) {
      outSig = sig;
      const runs = new Map();
      for (const o of job.outputs) (runs.get(o.run) || runs.set(o.run, []).get(o.run)).push(o);
      const ordered = [...runs.entries()].sort((x, y) => y[0] - x[0]);
      replace(outputs, ordered.map(([run, list], idx) => h("section.run",
        h("header.run-head", h("h3", run === 0 ? "Earlier render" : `Render ${run}`), h("span.run-meta", [timeAgo(Math.max(...list.map((o) => o.created || 0))), `${list.length} video${list.length === 1 ? "" : "s"}`].filter(Boolean).join(", ")), idx === 0 && ordered.length > 1 && h("span.run-tag", "Latest")),
        h("div.run-grid", list.map(card)))));
    }
  }

  function card(o) {
    const fmt = FORMATS.find((f) => f.value === o.format);
    const look = LOOKS.find((l) => l.value === o.look);
    const v = `?v=${o.created || ""}`;
    const video = h("video", { controls: true, preload: "metadata", poster: o.posterUrl ? o.posterUrl + v : undefined, src: o.url + v, "aria-label": `${look?.name || o.look} ${o.format} video`, style: fmt ? `aspect-ratio:${fmt.ratio}` : "" });
    return h("article.out", { dataset: { fmt: o.format } },
      h("div.out-media", video),
      h("div.out-info",
        h("h4.out-name", `${look?.name || o.look}, ${o.format}`),
        h("p.out-meta", [fmtDuration(o.seconds), fmtSize(o.size)].filter(Boolean).join(", ")),
        h("div.out-actions",
          h("a.btn.small", { href: `${o.url}?download=1`, download: "" }, icon("download-simple", 14), "Download"),
          h("button.btn.small.ghost", { type: "button", "aria-label": "Delete video", onclick: async () => {
            if (!(await confirmDialog({ title: "Delete this video?", body: "The file is removed from this project. Other renders are not affected.", confirm: "Delete video", danger: true }))) return;
            try { store.apply(await api.deleteOutput(store.id, o.id)); } catch (e) { toast(errorMessage(e), "error"); }
          } }, icon("trash", 14)))));
  }

  const unsubEnv = env.subscribe(() => paint());
  const pick = (b = {}) => ({ formats: b.formats || ["Vertical"], looks: (b.looks && b.looks.length ? b.looks : [b.look || "auto"]).filter((x) => LOOKS.some((l) => l.value === x)), captions: b.captions !== false, music: b.music || "warm" });
  return { update, destroy() { feed.destroy(); save.cancel(); unsubEnv(); } };
}
