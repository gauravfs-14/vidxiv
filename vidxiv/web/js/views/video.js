import { h, icon, replace, debounce, fmtDuration, fmtSize } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { progressBar, LogFeed, field, toast, errorMessage, emptyState } from "../ui.js";
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
  const cfg = { formats: ["Vertical"], look: "auto", captions: true, music: "warm" };
  let init = false, dirty = false;

  const save = debounce(async () => {
    try { await api.putBrief(store.id, { ...cfg }); dirty = false; } catch (e) { toast(`Settings were not saved. ${errorMessage(e)}`, "error"); }
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
    const input = h("input", { type: "radio", name: "look", value: l.value, onchange: () => { cfg.look = l.value; changed(); } });
    input.el = h("label.opt.look", input, swatch(l.value),
      h("span.opt-text", h("span.opt-name", l.name), h("span.opt-sub", l.suits)),
      h("span.opt-check", icon("check", 14)));
    return input;
  });
  const lookGroup = h("fieldset.group", h("legend", "Look"), h("div.opt-list", lookInputs.map((i) => i.el)));

  // captions, music
  const captions = h("input", { type: "checkbox", role: "switch", onchange: () => { cfg.captions = captions.checked; changed(); } });
  const capLabel = h("label.switch", captions, h("span.track", h("span.thumb")), h("span", "Captions"));
  const music = h("select", { onchange: () => { cfg.music = music.value; changed(); } }, MUSIC.map(([v, l]) => h("option", { value: v }, l)));

  // run
  const bar = progressBar();
  const msg = h("p.progress-msg", { "aria-live": "polite" });
  const btn = h("button.btn.primary.lg", { type: "button", onclick: () => render() }, icon("film-strip", 16), "Render video");
  const why = h("p.hint");
  const cancel = h("button.btn.small", { type: "button", onclick: async () => { try { await api.cancel(store.id); store.bump(); } catch (e) { toast(errorMessage(e), "error"); } } }, icon("x", 14), "Cancel");
  const progress = h("div.progress-row", { hidden: true }, bar, cancel);
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const feed = new LogFeed(store.id, "render");
  const log = h("details.disclosure", { hidden: true }, h("summary", "Render log"), feed.el);
  const outputs = h("div.outputs");
  const hintSlot = h("div.hint-slot");
  const empty = h("div", emptyState("No video yet", "Choose formats and a look, then render. Lines without narration are narrated first."));

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
      fmtGroup, fmtErr, lookGroup,
      h("div.opt-row", capLabel, field("Music", music)),
      h("div.render-row", btn, why, hintSlot)),
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
    lookInputs.forEach((i) => (i.checked = i.value === cfg.look));
    captions.checked = cfg.captions;
    if (document.activeElement !== music) music.value = cfg.music;

    const st = job.stages.render;
    const running = st.state === "running";
    const has = !!job.storyboard;
    const done = job.outputs.length > 0 && !running;
    sec.setStatus(running ? "running" : st.state === "error" ? "error" : done ? "done" : "idle",
      running ? "Rendering" : st.state === "error" ? "Failed" : done ? `${job.outputs.length} video${job.outputs.length === 1 ? "" : "s"}` : has ? "Ready" : "Waiting", st.progress);

    const noFormat = cfg.formats.length === 0;
    fmtErr.hidden = !noFormat;
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
      replace(outputs, job.outputs.map((o) => {
        const fmt = FORMATS.find((f) => f.value === o.format);
        const video = h("video", { controls: true, preload: "metadata", poster: o.posterUrl ? `${o.posterUrl}?v=${o.created || ""}` : undefined, src: `${o.url}?v=${o.created || ""}`, "aria-label": `${o.format} video`, style: fmt ? `aspect-ratio:${fmt.ratio}` : "" });
        return h("article.out", { dataset: { fmt: o.format } },
          h("div.out-media", video),
          h("div.out-info",
            h("h3.out-name", o.format),
            h("p.out-meta", [fmt?.sub, fmtDuration(o.seconds), fmtSize(o.size)].filter(Boolean).join(", ")),
            h("a.btn.small", { href: `${o.url}?download=1`, download: "" }, icon("download-simple", 14), "Download")));
      }));
    }
  }

  const unsubEnv = env.subscribe(() => paint());
  const pick = (b = {}) => ({ formats: b.formats || ["Vertical"], look: LOOKS.some((l) => l.value === b.look) ? b.look : "auto", captions: b.captions !== false, music: b.music || "warm" });
  return { update, destroy() { feed.destroy(); save.cancel(); unsubEnv(); } };
}
