import { h, icon, replace, debounce, fmtDuration } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { progressBar, LogFeed, field, toast, errorMessage, emptyState } from "../ui.js";
import { openStudio } from "../studio.js";

export function createNarration({ store, sec }) {
  const cfg = { engine: "auto", voice: "af_heart", speed: 0.95 };
  let cfgInit = false, cfgDirty = false, voices = null;

  // ---------------------------------------------------------------- audio (one shared player)
  const audio = new Audio();
  let playing = -1;
  audio.addEventListener("ended", () => setPlaying(-1));
  audio.addEventListener("pause", () => { if (audio.ended || audio.paused) setPlaying(-1); });
  const rowsByIndex = [];
  function setPlaying(i) {
    playing = i;
    rowsByIndex.forEach((r, k) => r && paintPlay(r, k));
  }
  function paintPlay(r, k) {
    const on = playing === k;
    replace(r.play, icon(on ? "pause" : "play", 15));
    r.play.setAttribute("aria-label", `${on ? "Pause" : "Play"} line ${k + 1}`);
    r.play.setAttribute("aria-pressed", String(on));
    r.el.classList.toggle("playing", on);
  }
  function toggle(i, url) {
    if (playing === i) { audio.pause(); return; }
    audio.src = url;
    audio.play().then(() => setPlaying(i)).catch((e) => { setPlaying(-1); toast(`The clip could not be played. ${e.message || ""}`, "error"); });
  }

  // ---------------------------------------------------------------- voice settings
  const engineSel = h("select", { onchange: () => { cfg.engine = engineSel.value; paintVoices(); changed(); } });
  const voiceSel = h("select", { onchange: () => { cfg.voice = voiceSel.value; changed(); } });
  const speedOut = h("output.mono", "0.95x");
  const speed = h("input#speed", { type: "range", min: 0.8, max: 1.1, step: 0.01, value: cfg.speed, oninput: () => { cfg.speed = +speed.value; speedOut.textContent = cfg.speed.toFixed(2) + "x"; changed(); } });
  const saveState = h("span.save-state", { role: "status", "aria-live": "polite" });
  const engineHint = h("p.hint");
  const sampleErr = h("p.field-error", { role: "alert", hidden: true });
  const sampleBtn = h("button.btn", { type: "button", onclick: async () => {
    sampleBtn.disabled = true;
    replace(sampleBtn, icon("circle-notch", 15), "Generating sample");
    sampleErr.hidden = true;
    try {
      const blob = await api.voicePreview({ engine: cfg.engine, voice: cfg.voice, speed: cfg.speed });
      audio.src = URL.createObjectURL(blob);
      setPlaying(-1);
      await audio.play();
    } catch (e) {
      sampleErr.textContent = `The sample could not be made. ${errorMessage(e)} Try another engine.`;
      sampleErr.hidden = false;
    } finally {
      replace(sampleBtn, icon("speaker-high", 15), "Hear a sample");
      sampleBtn.disabled = false;
    }
  } }, icon("speaker-high", 15), "Hear a sample");

  const persist = debounce(async () => {
    saveState.textContent = "Saving";
    try {
      store.apply(await api.putNarrationConfig(store.id, { ...cfg }));
      cfgDirty = false;
      saveState.textContent = "Saved";
    } catch (e) {
      saveState.textContent = `Not saved. ${errorMessage(e)}`;
    }
  }, 600);
  function changed() { cfgDirty = true; saveState.textContent = "Unsaved changes"; persist(); }

  function paintEngines() {
    if (!voices) return;
    const opts = [h("option", { value: "auto" }, "Auto (best available)")];
    for (const [k, e] of Object.entries(voices)) {
      const usable = e.ready || e.needsDownload;
      const note = e.ready ? "" : e.needsDownload ? ` (needs download, ${e.size || "one time"})` : " (not available)";
      opts.push(h("option", { value: k, disabled: !usable }, e.label + note));
    }
    replace(engineSel, opts);
    engineSel.value = cfg.engine;
    paintVoices();
  }
  function paintVoices() {
    if (!voices) return;
    const eng = cfg.engine === "auto" ? voices.kokoro : voices[cfg.engine];
    const list = eng?.voices || [];
    replace(voiceSel, list.length ? list.map((v) => h("option", { value: v.id }, v.note ? `${v.label}, ${v.note}` : v.label)) : h("option", { value: "" }, "No voices available"));
    if (list.some((v) => v.id === cfg.voice)) voiceSel.value = cfg.voice;
    else if (list.length) { cfg.voice = list[0].id; voiceSel.value = cfg.voice; }
    voiceSel.disabled = !list.length;
    const needs = (cfg.engine === "kokoro" || cfg.engine === "auto") && voices.kokoro?.needsDownload;
    replace(engineHint,
      cfg.engine === "auto" ? "Auto tries Kokoro first, then system voices. " : "",
      needs && ["Kokoro is not downloaded yet. ", h("button.link-btn", { type: "button", onclick: () => env.setup("kokoro").then(() => toast("Downloading the voice model. Progress shows on the home page.", "info")).catch((e) => toast(errorMessage(e), "error")) }, "Download voice model"), ". "],
      "Voice changes apply when you generate narration again.");
  }
  api.voices().then((v) => { voices = v; paintEngines(); }).catch(() => {});

  const settings = h("div.voice-bar",
    field("Engine", engineSel), field("Voice", voiceSel),
    h("div.field.speed", h("label", { for: "speed" }, "Speed ", speedOut), speed),
    h("div.voice-actions", sampleBtn, saveState));

  // ---------------------------------------------------------------- generate and takes
  const bar = progressBar();
  const msg = h("p.progress-msg", { "aria-live": "polite" });
  const genBtn = h("button.btn.primary", { type: "button", onclick: () => start() }, icon("waveform", 15), "Generate narration");
  const cancel = h("button.btn.small", { type: "button", onclick: async () => { try { await api.cancel(store.id); store.bump(); } catch (e) { toast(errorMessage(e), "error"); } } }, icon("x", 14), "Cancel");
  const tally = h("span.tally");
  const progress = h("div.progress-row", { hidden: true }, bar, cancel);
  const feed = new LogFeed(store.id, "narration");
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const takeInput = h("input", { type: "file", accept: "audio/*,video/webm,.m4a,.mp3,.wav,.webm", hidden: true, "aria-label": "Choose one full recording", onchange: () => { if (takeInput.files[0]) sendTake(takeInput.files[0]); takeInput.value = ""; } });
  const takeBtn = h("button.btn", { type: "button", onclick: () => takeInput.click() }, icon("upload-simple", 15), "Upload full recording");
  const hasMine = (i) => { const c = store.job?.narration.clips?.[i]; return !!(c && (c.source === "user" || c.userUrl)); };
  function studio(mode, start = 0) {
    const sb = store.job?.storyboard;
    if (!sb?.lines.length) return;
    return openStudio({
      lines: sb.lines.map((l) => l.text), start, mode, has: hasMine,
      saveLine: async (i, f) => { store.apply(await api.uploadLine(store.id, i, f)); store.bump(); },
      saveTake: async (f, marks) => { store.apply(await api.uploadTake(store.id, f, "split", marks)); store.bump(); },
    });
  }
  const firstOpen = () => { const n = store.job?.storyboard?.lines.length || 0; for (let i = 0; i < n; i++) if (!hasMine(i)) return i; return 0; };
  const takeRec = h("button.btn", { type: "button", onclick: () => studio("all") }, icon("microphone", 15), "Record all at once");
  const lineRec = h("button.btn", { type: "button", onclick: () => studio("lines", firstOpen()) }, icon("microphone", 15), "Record line by line");

  async function call(fn, okMsg) {
    try {
      store.apply(await fn());
      store.bump();
      if (okMsg) toast(okMsg, "info", 3000);
    } catch (e) { toast(errorMessage(e), "error"); }
  }
  const start = (lines) => call(() => api.synthesize(store.id, lines));
  const sendTake = (f) => call(() => api.uploadTake(store.id, f, "split"));

  // ---------------------------------------------------------------- lines table
  const fileIn = h("input", { type: "file", accept: "audio/*,video/webm,.m4a,.mp3,.wav,.webm", hidden: true, "aria-label": "Choose an audio file for this line" });
  let fileTarget = -1;
  fileIn.onchange = () => { const f = fileIn.files[0]; fileIn.value = ""; if (f) lineUpload(fileTarget, f); };
  const lineUpload = (i, f) => withBusy(i, () => api.uploadLine(store.id, i, f));
  async function withBusy(i, fn) {
    const r = rowsByIndex[i];
    r?.el.classList.add("busy");
    r?.all.forEach((b) => (b.disabled = true));
    try { store.apply(await fn()); store.bump(); } catch (e) { toast(`Line ${i + 1}: ${errorMessage(e)}`, "error"); }
    r?.el.classList.remove("busy");
    if (r) r.sig = "";
    if (store.job) update(store.job);
  }

  const list = h("ol.nlines");
  const head = h("div.nhead", { "aria-hidden": "true" }, h("span", "Line"), h("span"), h("span", "Narration"), h("span", "Source"), h("span.r", "Length"), h("span"));
  const empty = h("div", { hidden: true }, emptyState("No lines to narrate yet", "Write the script first. Each line gets its own clip here.", h("a.btn", { href: `#/job/${store.id}/script`, onclick: (e) => { e.preventDefault(); sec.go("script"); } }, "Go to script")));

  function buildRow(i) {
    const r = { sig: "" };
    r.idx = h("span.n");
    r.text = h("p.ntext");
    r.play = h("button.icon-btn.play", { type: "button", onclick: () => r.url && toggle(i, r.url) });
    r.dur = h("span.dur");
    r.source = h("span.source");
    r.acts = [
      h("button.btn.small.ghost", { type: "button", onclick: async () => {
        studio("lines", i);
      } }, icon("microphone", 14), "Record"),
      h("button.btn.small.ghost", { type: "button", onclick: () => { fileTarget = i; fileIn.click(); } }, icon("upload-simple", 14), "Upload"),
    ];
    r.gen = h("button.btn.small.ghost", { type: "button", onclick: () => start([i]) }, icon("arrow-clockwise", 14), "Generate");
    r.synth = h("button.btn.small.ghost", { type: "button", onclick: () => withBusy(i, () => api.deleteLine(store.id, i)) }, icon("speaker-high", 14), "Use synthetic voice");
    r.all = [...r.acts, r.gen, r.synth];
    r.el = h("li.nrow", r.idx, r.play, r.text, r.source, r.dur, h("div.nacts", ...r.all));
    paintPlay(r, i);
    return r;
  }

  function updateRow(r, i, line, clip, busyAll) {
    const stale = !!clip?.stale;
    const user = clip?.source === "user";
    const sig = JSON.stringify([line.text, clip && [clip.source, clip.duration, clip.url, stale, clip.userUrl], busyAll]);
    if (r.sig === sig) return;
    r.sig = sig;
    r.idx.textContent = String(i + 1).padStart(2, "0");
    r.text.textContent = line.text;
    r.url = clip?.url || null;
    r.play.disabled = !r.url;
    r.dur.textContent = clip ? fmtDuration(clip.duration) : "";
    replace(r.source,
      !clip?.source && h("span.badge.none", "Not generated"),
      clip?.source && h(`span.badge.${user ? "user" : "synth"}`, user ? "Your recording" : "Synthetic"),
      stale && h("span.badge.stale", { title: "The text changed after this clip was made" }, "Stale"));
    r.acts.forEach((b) => (b.disabled = busyAll));
    r.gen.hidden = user || (!!clip?.source && !stale);
    r.gen.disabled = busyAll;
    r.synth.hidden = !(user || clip?.userUrl);
    r.synth.disabled = busyAll;
    paintPlay(r, i);
  }

  sec.body.append(
    h("div.voice-section", h("h3.sub-title", "Voice"), settings, engineHint, sampleErr),
    h("div.nar-gen",
      h("div.gen-actions", genBtn, tally),
      progress, msg, errBox,
      h("details.disclosure", h("summary", "Activity log"), feed.el)),
    h("div.nar-lines", head, list, empty, fileIn),
    h("div.nar-takes",
      h("h3.sub-title", "Record the whole script"),
      h("p.hint", "Read the lines in order. vidxiv matches the take to the script and cuts it into lines. Lines it cannot find keep the synthetic voice."),
      h("div.take-actions", lineRec, takeRec, takeBtn, takeInput)));

  function update(job) {
    const st = job.stages.narration;
    const sb = job.storyboard;
    const lines = sb ? sb.lines : [];
    const clips = job.narration.clips || {};
    const running = st.state === "running";
    const have = (i) => !!clips[i]?.source;
    const ready = lines.filter((_, i) => have(i) && !clips[i].stale).length;
    const stale = lines.filter((_, i) => clips[i]?.stale).length;
    const complete = lines.length > 0 && ready === lines.length;

    const kind = running ? "running" : st.state === "error" ? "error" : complete ? "done" : "idle";
    sec.setStatus(kind, running ? "Generating" : st.state === "error" ? "Failed" : lines.length ? `${ready} of ${lines.length} lines` : "Waiting", st.progress);
    sec.el.classList.toggle("is-busy", running);

    if (!cfgDirty && !settings.contains(document.activeElement)) {
      const n = job.narration;
      if (!cfgInit || n.engine !== cfg.engine || n.voice !== cfg.voice || n.speed !== cfg.speed) {
        cfgInit = true;
        Object.assign(cfg, { engine: n.engine, voice: n.voice, speed: n.speed });
        speed.value = cfg.speed;
        speedOut.textContent = (+cfg.speed).toFixed(2) + "x";
        paintEngines();
      }
    }

    const noLines = !lines.length;
    genBtn.disabled = running || noLines;
    takeBtn.disabled = takeRec.disabled = lineRec.disabled = running || noLines;
    tally.textContent = noLines ? "" : stale ? `${ready} of ${lines.length} ready, ${stale} stale` : `${ready} of ${lines.length} ready`;
    progress.hidden = !running;
    bar.set(st.progress, "Generating narration");
    msg.textContent = running || (st.state === "done" && st.message && st.message !== "Done") ? st.message : "";
    feed.sync(running, st.started);
    errBox.hidden = st.state !== "error";
    if (st.state === "error") replace(errBox, h("p", h("strong", "Narration stopped. "), st.error || st.message, " Generate narration again to continue. Finished lines are kept."));
    if (env.data && !env.ffmpeg) { errBox.hidden = false; replace(errBox, h("p", h("strong", "ffmpeg is missing. "), "Narration needs it. Install it with brew install ffmpeg, then restart vidxiv.")); }

    empty.hidden = !noLines;
    head.hidden = noLines;
    while (rowsByIndex.length > lines.length) rowsByIndex.pop().el.remove();
    lines.forEach((line, i) => {
      if (!rowsByIndex[i]) { rowsByIndex[i] = buildRow(i); list.append(rowsByIndex[i].el); }
      updateRow(rowsByIndex[i], i, line, clips[i], running);
    });
  }

  return { update, destroy() { feed.destroy(); audio.pause(); persist.cancel(); } };
}
