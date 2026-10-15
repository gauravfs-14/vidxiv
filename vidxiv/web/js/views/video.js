import { h, icon, replace, debounce, fmtDuration, fmtSize } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { chip, progressBar, LogFeed, field, toast, errorMessage } from "../ui.js";
import { installEngineHint } from "./script.js";

const FORMATS = [
  { value: "Vertical", ratio: "9 / 16", name: "Vertical", sub: "9:16, Shorts and Reels" },
  { value: "Landscape", ratio: "16 / 9", name: "Landscape", sub: "16:9, YouTube" },
  { value: "Square", ratio: "1 / 1", name: "Square", sub: "1:1, feeds" },
];
const PALETTES = [
  { value: "paper", name: "Paper", sub: "Warm and light" },
  { value: "ink", name: "Ink", sub: "Dark slate" },
];
const MUSIC = [["warm", "Warm"], ["bright", "Bright"], ["calm", "Calm"], ["night", "Night"], ["none", "No music"]];

export function createVideo({ store, sec }) {
  const cfg = { formats: ["Vertical"], palette: "paper", captions: true, music: "warm" };
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
    input.el = h("label.fmt", input,
      h("span.glyph-wrap", h("span.glyph", { style: `aspect-ratio:${f.ratio}` })),
      h("span.fmt-text", h("span.fmt-name", f.name), h("span.fmt-sub", f.sub)));
    return input;
  });
  const fmtGroup = h("fieldset.group", h("legend", "Formats"), h("div.fmt-row", fmtInputs.map((i) => i.el)));
  const fmtErr = h("p.field-error", { role: "alert", hidden: true }, "Choose at least one format to render.");

  // palette
  const palInputs = PALETTES.map((p) => {
    const input = h("input", { type: "radio", name: "palette", value: p.value, onchange: () => { cfg.palette = p.value; changed(); } });
    input.el = h("label.pal", input, h(`span.swatch.${p.value}`, h("i"), h("b")), h("span.fmt-text", h("span.fmt-name", p.name), h("span.fmt-sub", p.sub)));
    return input;
  });
  const palGroup = h("fieldset.group", h("legend", "Palette"), h("div.pal-row", palInputs.map((i) => i.el)));

  // captions, music
  const captions = h("input", { type: "checkbox", role: "switch", onchange: () => { cfg.captions = captions.checked; changed(); } });
  const capLabel = h("label.switch", captions, h("span.track", h("span.thumb")), h("span", "Captions"));
  const music = h("select", { onchange: () => { cfg.music = music.value; changed(); } }, MUSIC.map(([v, l]) => h("option", { value: v }, l)));

  // run
  const bar = progressBar();
  const msg = h("p.small.muted", { "aria-live": "polite" });
  const btn = h("button.btn.primary.lg", { type: "button", onclick: () => render() }, icon("film-strip", 18), "Render video");
  const why = h("p.hint");
  const cancel = h("button.btn.small", { type: "button", onclick: async () => { try { await api.cancel(store.id); store.bump(); } catch (e) { toast(errorMessage(e), "error"); } } }, icon("x", 14), "Cancel");
  const progress = h("div.progress-row", { hidden: true }, bar, cancel);
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const feed = new LogFeed(store.id, "render");
  const log = h("details.disclosure", { hidden: true }, h("summary", "Render log"), feed.el);
  const outputs = h("div.outputs");
  const hintSlot = h("div.hint-slot");

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

  sec.body.append(
    h("div.video-opts",
      h("div.opt-formats", fmtGroup, fmtErr),
      palGroup,
      h("div.opt-row", capLabel, field("Music", music))),
    h("div.render-row", btn, h("div.render-notes", h("p.hint", "Lines without narration, or with changed text, are narrated automatically before rendering."), why, hintSlot)),
    progress, msg, errBox, log, outputs);

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
    palInputs.forEach((i) => (i.checked = i.value === cfg.palette));
    captions.checked = cfg.captions;
    if (document.activeElement !== music) music.value = cfg.music;

    const st = job.stages.render;
    const running = st.state === "running";
    const has = !!job.storyboard;
    const done = job.outputs.length > 0 && st.state !== "running";
    sec.setChip(chip(running ? "running" : st.state === "error" ? "error" : done ? "done" : "idle", done && !running ? `${job.outputs.length} video${job.outputs.length === 1 ? "" : "s"}` : undefined, st.progress));
    sec.setSummary(done ? job.outputs.map((o) => o.format).join(", ") : "");

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
            h("p.small.muted", [fmt?.sub, fmtDuration(o.seconds), fmtSize(o.size)].filter(Boolean).join(", ")),
            h("a.btn", { href: `${o.url}?download=1`, download: "" }, icon("download-simple", 16), "Download")));
      }));
    }
  }

  const unsubEnv = env.subscribe(() => paint());
  const pick = (b = {}) => ({ formats: b.formats || ["Vertical"], palette: b.palette || "paper", captions: b.captions !== false, music: b.music || "warm" });
  return { update, destroy() { feed.destroy(); save.cancel(); unsubEnv(); } };
}
