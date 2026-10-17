import { h, icon, replace, autosize, fmtSize, timeAgo } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { installHelp, progressBar, LogFeed, segmented, field as fieldEl, toast, errorMessage, confirmDialog, lightbox, skel, emptyState } from "../ui.js";
import { recordClip } from "../recorder.js";
import { StoryboardEditor } from "./storyboard.js";
import { SCENES } from "../scenes.js";
import { PACES, paceOf } from "../pace.js";

const MODES = [
  { value: "generate", label: "Write with an agent" },
  { value: "provided", label: "I have a script" },
  { value: "transcript", label: "I have a recording" },
];
const PRESETS = [
  { value: "short", label: "Short", seconds: 45 },
  { value: "medium", label: "Medium", seconds: 90 },
  { value: "long", label: "Long", seconds: 180 },
];
const DETAILS = [
  { value: "overview", label: "Overview", help: "One result and one mechanism. No equations." },
  { value: "balanced", label: "Balanced", help: "Problem, idea, method, main result and one limit." },
  { value: "deep", label: "Deep", help: "Step by step method, key equation, ablations, baselines and limits. For technical viewers." },
];
const MIN_SECONDS = 10, MAX_SECONDS = 600, WARN_SECONDS = 300, STRONG_SECONDS = 480;

/** 90 becomes "1 min 30 s". */
export function fmtSeconds(s) {
  const m = Math.floor(s / 60), r = s % 60;
  return m === 0 ? `${r} s` : r === 0 ? `${m} min` : `${m} min ${r} s`;
}
/** 90 becomes "1:30". */
const clock = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
const presetFor = (s) => PRESETS.find((p) => p.seconds === s);

export function installEngineHint() {
  if (!env.data || env.engineReady) return null;
  const node = env.data?.tools?.node;
  return h("div.engine-hint", h("p.hint", node ? "The video engine is not installed. " : "Node.js 20 or newer is not installed, so the video engine cannot be set up. ",
    node && h("button.link-btn", { type: "button", onclick: () => env.setup("engine").catch((e) => toast(errorMessage(e), "error")) }, "Install video engine")),
    !node && installHelp("node", env.data));
}

export function createScript({ store, sec }) {
  const state = { mode: "generate", seconds: 45, custom: false, detail: "balanced", pace: "standard", agent: "auto", model: "", instructions: "", text: "", file: null, init: false, armed: false };
  const editor = new StoryboardEditor({ store });
  sec.onShow(() => editor.refit());

  // ---------------------------------------------------------------- generator form
  const modeSeg = segmented("script-mode", MODES, state.mode, (v) => { state.mode = v; state.armed = false; paintMode(); }, "How the script is written");
  modeSeg.classList.add("seg-fit");

  // length: three named presets plus a custom slider, stored as brief.seconds
  const durSeg = segmented("duration", [...PRESETS.map((p) => ({ value: p.value, label: p.label, sub: fmtSeconds(p.seconds) })), { value: "custom", label: "Custom" }],
    "short", (v) => {
      state.armed = false;
      if (v === "custom") state.custom = true;
      else { state.custom = false; state.seconds = PRESETS.find((p) => p.value === v).seconds; }
      paintDuration();
    }, "Video length");
  const slider = h("input", { type: "range", min: MIN_SECONDS, max: MAX_SECONDS, step: 5, "aria-label": "Custom length", "aria-valuetext": "", oninput: () => { state.seconds = +slider.value; state.armed = false; paintDuration(); } });
  const sliderOut = h("output.mono.slider-out");
  const customRow = h("div.custom-row", { hidden: true }, slider, sliderOut);
  const lineHint = h("p.hint");
  const lengthWarn = h("p.notice.warn", { hidden: true, role: "status" });

  const detailSeg = segmented("detail", DETAILS, state.detail, (v) => { state.detail = v; paintDuration(); }, "How much detail");
  const detailHelp = h("p.hint");

  const paceSeg = segmented("pace", PACES.map((p) => ({ value: p.value, label: p.label, sub: p.sub })), state.pace, (v) => { state.pace = v; paintDuration(); }, "Pace");
  const paceHelp = h("p.hint");

  const notes = h("textarea", { rows: 2, placeholder: "Focus on the method. Skip the related work.", oninput: () => { state.instructions = notes.value; autosize(notes); } });
  const agentSel = h("select", { onchange: () => { state.agent = agentSel.value; state.armed = false; paintAgent(); paintDuration(); } });
  const agentHint = h("p.hint");
  const model = h("select", { onchange: () => { state.model = model.value; paintModels(); } });
  const modelHint = h("p.hint");
  const text = h("textarea.script-text", { rows: 8, placeholder: "Paste or type the narration. Blank lines start a new line of narration.", oninput: () => { state.text = text.value; autosize(text); } });
  const file = { name: h("span.file-name"), clear: h("button.link-btn", { type: "button", hidden: true }, "Remove") };
  const audioIn = h("input", { type: "file", accept: "audio/*,video/webm,.m4a,.mp3,.wav,.webm", hidden: true, "aria-label": "Choose a recording", onchange: () => setFile(audioIn.files[0]) });
  const setFile = (f) => {
    state.file = f || null;
    file.name.textContent = f ? `${f.name}, ${fmtSize(f.size)}` : "No recording chosen.";
    file.clear.hidden = !f;
    audioIn.value = "";
  };
  file.clear.onclick = () => setFile(null);
  const whisperNote = h("p.hint");

  const paneAgent = h("div.pane-fields",
    h("div.field", h("span.label-like", "Length"), durSeg, customRow, lineHint, lengthWarn),
    h("div.field", h("span.label-like", "Detail"), detailSeg, detailHelp),
    h("div.field", h("span.label-like", "Pace"), paceSeg, paceHelp),
    fieldEl("Notes for the agent", notes, "Optional. What to focus on or leave out."));
  const paneText = h("div.pane-fields", fieldEl("Narration script", text, "The agent only picks scenes. Long text is split into lines of one or two sentences."));
  const paneRec = h("div.pane-fields",
    h("div.field", h("span.label-like", "Recording"),
      h("div.rec-row",
        h("button.btn", { type: "button", onclick: async () => {
          const f = await recordClip({ title: "Record the narration", help: "Speak the whole narration. It is transcribed and cut into lines, then you can edit the text." });
          if (f) setFile(f);
        } }, icon("microphone", 15), "Record"),
        h("button.btn", { type: "button", onclick: () => audioIn.click() }, icon("upload-simple", 15), "Upload audio"),
        file.name, file.clear),
      whisperNote, audioIn));
  const modelField = fieldEl("Model", model, modelHint);
  const agentRow = h("div.pane-fields", fieldEl("Scene writer", agentSel, agentHint), modelField);

  const runBtn = h("button.btn.primary.lg", { type: "button", onclick: () => run() });
  const runHint = h("p.hint");
  const runErr = h("p.field-error", { role: "alert", hidden: true });
  const genTitle = h("h3.gen-title");
  const genSub = h("p.gen-sub");
  const form = h("div.gen-form", modeSeg, paneAgent, paneText, paneRec, agentRow, h("div.run-row", runBtn, runHint), runErr);
  const gen = h("div.generator", { hidden: true }, h("div.gen-head", genTitle, genSub), form);
  const toggleGen = h("button.btn", { type: "button", "aria-expanded": "false", onclick: () => { genOpen = !genOpen; update(store.job); if (genOpen) modeSeg.querySelector("input:checked")?.focus(); } }, "Write a new script");
  sec.actions.append(toggleGen);
  let genOpen = false;

  const linesNeeded = () => Math.max(1, Math.round(state.seconds / paceOf(state.pace).lineSeconds));
  const usesAgent = () => state.mode === "generate" && state.agent !== "none";

  function paintDuration() {
    const preset = state.custom ? null : presetFor(state.seconds);
    durSeg.set(state.custom || !preset ? "custom" : preset.value);
    state.custom = state.custom || !preset;
    customRow.hidden = !state.custom;
    slider.value = state.seconds;
    sliderOut.textContent = clock(state.seconds);
    slider.setAttribute("aria-valuetext", fmtSeconds(state.seconds));
    lineHint.textContent = `${fmtSeconds(state.seconds)} is about ${linesNeeded()} narration lines.`;
    detailSeg.set(state.detail);
    detailHelp.textContent = DETAILS.find((d) => d.value === state.detail)?.help || "";
    paceSeg.set(state.pace);
    paceHelp.textContent = paceOf(state.pace).help + " The same setting controls animation speed and pauses when you render.";
    const n = linesNeeded();
    const warn = usesAgent() && state.seconds > WARN_SECONDS;
    const strong = usesAgent() && state.seconds > STRONG_SECONDS;
    lengthWarn.hidden = !warn;
    lengthWarn.classList.toggle("strong", strong);
    lengthWarn.textContent = strong
      ? `A ${Math.round(state.seconds / 60)} minute script is a heavy job. The agent reads the whole paper, writes about ${n} lines and often needs several attempts, which costs a lot of tokens and time. Three minutes or less is usually enough.`
      : warn ? `Long videos use a lot of agent tokens and time. The agent reads the whole paper and writes about ${n} lines, and may need several attempts. Consider 3 minutes or less.` : "";
    paintRun();
  }

  function paintRun() {
    const strong = usesAgent() && state.seconds > STRONG_SECONDS;
    if (!strong) state.armed = false;
    const base = { generate: "Write script", provided: "Use my script", transcript: "Transcribe recording" }[state.mode];
    runBtn.textContent = strong && state.armed ? `Write a ${Math.round(state.seconds / 60)} minute script anyway` : base;
    runBtn.classList.toggle("caution", strong && state.armed);
  }

  function paintMode() {
    modeSeg.set(state.mode);
    paneAgent.hidden = state.mode !== "generate";
    paneText.hidden = state.mode !== "provided";
    paneRec.hidden = state.mode !== "transcript";
    agentRow.hidden = state.mode === "provided";
    whisperNote.textContent = env.data && !env.data.whisper ? "The speech model downloads the first time you transcribe, which can take a minute." : "The recording is transcribed on this computer.";
    autosize(text); autosize(notes);
    paintDuration();
  }

  function paintAgent() {
    const claude = env.agentFound("claude"), codex = env.agentFound("codex");
    const opts = [
      ["auto", "Auto" + (claude || codex ? "" : " (no agent found, uses simple rules)"), false],
      ["claude", "Claude Code" + (claude ? "" : " (not found)"), !claude && !!env.data],
      ["codex", "Codex" + (codex ? "" : " (not found)"), !codex && !!env.data],
      ["none", "Simple rules, no agent", false],
    ];
    const cur = agentSel.value || state.agent;
    replace(agentSel, opts.map(([v, l, dis]) => h("option", { value: v, disabled: dis }, l)));
    agentSel.value = opts.find((o) => o[0] === cur && !o[2]) ? cur : "auto";
    state.agent = agentSel.value;
    agentHint.textContent = {
      auto: "Uses Claude Code if it is installed, then Codex, then simple rules.",
      claude: "Reads the paper and writes the script with Claude Code.",
      codex: "Reads the paper and writes the script with Codex.",
      none: "Builds scenes from the abstract and figures with fixed rules. Fast, but plainer.",
    }[state.agent];
    paintModels();
  }

  // The list comes from the installed CLI, so a model that does not exist cannot be chosen.
  function paintModels() {
    const resolved = state.agent === "auto" ? (env.agentFound("claude") ? "claude" : env.agentFound("codex") ? "codex" : null) : state.agent;
    modelField.hidden = !resolved || resolved === "none";
    if (modelField.hidden) { state.model = ""; return; }
    const list = env.agentModels(resolved);
    replace(model, list.map((m) => h("option", { value: m.id }, m.label)));
    if (!list.some((m) => m.id === state.model)) state.model = "";
    model.value = state.model;
    model.disabled = list.length < 2;
    const cur = list.find((m) => m.id === state.model);
    modelHint.textContent = list.length < 2
      ? "This CLI did not report any other models, so it uses its own default."
      : [state.agent === "auto" ? `Models from ${resolved === "claude" ? "Claude Code" : "Codex"}. ` : "", cur?.note || ""].join("").trim();
  }

  async function run() {
    const job = store.job;
    runErr.hidden = true;
    if (state.mode === "provided" && !state.text.trim()) return fail("Paste your script first.");
    if (state.mode === "transcript" && !state.file) return fail("Record or upload an audio file first.");
    if (usesAgent() && state.seconds > STRONG_SECONDS && !state.armed) {
      state.armed = true;
      paintRun();
      return;
    }
    if (job.storyboard) {
      const ok = await confirmDialog({ title: "Write a new script?", body: "The current script is kept as a version you can restore. Recordings you made for its lines are removed.", confirm: "Replace script", danger: true });
      if (!ok) return;
    }
    runBtn.disabled = true;
    try {
      await api.putBrief(store.id, { seconds: state.seconds, detail: state.detail, pace: state.pace, instructions: state.instructions });
      if (state.mode === "transcript") await api.uploadTake(store.id, state.file, "transcribe");
      const view = await api.makeScript(store.id, { agent: state.agent, model: state.model, mode: state.mode, text: state.text });
      store.apply(view);
      store.bump();
      genOpen = false;
      state.armed = false;
    } catch (e) {
      fail(errorMessage(e));
    } finally { runBtn.disabled = false; paintRun(); update(store.job); }
  }
  const fail = (m) => { runErr.textContent = m; runErr.hidden = false; };

  // ---------------------------------------------------------------- activity
  const bar = progressBar();
  const msg = h("p.progress-msg", { "aria-live": "polite" });
  const cancel = h("button.btn.small", { type: "button", onclick: async () => {
    cancel.disabled = true;
    try { await api.cancel(store.id); store.bump(); } catch (e) { toast(errorMessage(e), "error"); } finally { cancel.disabled = false; }
  } }, icon("x", 14), "Cancel");
  const feed = new LogFeed(store.id, "script");
  // Shaped like the editor rows it will turn into.
  const shape = h("div.line-skeleton", { "aria-hidden": "true" }, [88, 72, 94, 60, 80].map((w) => h("div.sk-row", skel("52px", "tag"), skel(w + "%"))));
  const activity = h("div.activity", { hidden: true }, h("div.progress-row", bar, cancel), msg, shape);
  const log = h("details.disclosure", { hidden: true }, h("summary", "Activity log"), feed.el);
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const blocked = h("div", { hidden: true }, emptyState("The paper is still loading", "The script is written from the paper text, so it needs to finish first."));

  // ---------------------------------------------------------------- preview filmstrip
  const pvBar = progressBar();
  const pvMsg = h("span.progress-msg", { "aria-live": "polite" });
  const pvBtn = h("button.btn.small", { type: "button", onclick: async () => {
    pvBtn.disabled = true;
    try { await editor.flush(); store.apply(await api.preview(store.id)); store.bump(); } catch (e) { toast(errorMessage(e), "error"); }
    update(store.job);
  } }, icon("film-strip", 14), "Preview scenes");
  const pvErr = h("p.field-error", { role: "alert", hidden: true });
  const pvHint = h("div.hint-slot");
  const strip = h("ul.filmstrip", { "aria-label": "Scene previews" });
  const pvEmpty = h("p.strip-empty", "No previews yet. Render a still of every scene to check layouts before you narrate.");
  const pvProgress = h("div.progress-row", { hidden: true }, pvBar, pvMsg);
  const preview = h("div.preview",
    h("div.preview-head", h("h3.sub-title", "Scene previews"), pvBtn),
    pvHint, pvProgress, pvErr, pvEmpty, strip);
  editor.slot.append(preview);
  let stripSig = "";

  const versionList = h("ul.versions");
  const versions = h("details.disclosure.versions-box", { hidden: true }, h("summary", "Earlier scripts"),
    h("p.hint", "Writing a new script keeps the old one here. Restoring swaps it in and keeps the current one."), versionList);
  let verSig = "";
  async function restore(v) {
    if (!(await confirmDialog({ title: "Restore this script?", body: "It becomes the current script. The current one is kept as a version. Recordings for lines that change are removed.", confirm: "Restore script" }))) return;
    try { store.apply(await api.restoreScript(store.id, v.id)); store.bump(); } catch (e) { toast(errorMessage(e), "error"); }
  }
  function paintVersions(list) {
    const sig = JSON.stringify(list);
    if (sig === verSig) return;
    verSig = sig;
    versions.hidden = !list.length;
    versions.querySelector("summary").textContent = `Earlier scripts (${list.length})`;
    replace(versionList, list.map((v) => h("li",
      h("div.ver-text", h("span.ver-title", `${v.lines} lines, ${timeAgo(v.created)}`), h("span.ver-prev", v.preview)),
      h("button.btn.small.ghost", { type: "button", onclick: () => restore(v) }, "Restore"))));
  }
  sec.body.append(gen, blocked, activity, log, errBox, editor.el, versions);
  paintMode();
  paintAgent();
  setFile(null);
  const unsubEnv = env.subscribe(() => { paintAgent(); paintMode(); if (store.job) update(store.job); });

  function update(job) {
    if (!job) return;
    const st = job.stages.script;
    const ing = job.stages.ingest.state;
    const running = st.state === "running";
    const has = !!job.storyboard;
    if (!state.init) {
      state.init = true;
      const b = job.brief || {};
      state.mode = b.scriptMode || "generate";
      state.seconds = Number.isFinite(b.seconds) ? Math.max(MIN_SECONDS, Math.min(MAX_SECONDS, Math.round(b.seconds))) : 45;
      state.custom = !presetFor(state.seconds);
      state.detail = DETAILS.some((d) => d.value === b.detail) ? b.detail : "balanced";
      state.pace = paceOf(b.pace).value;
      state.instructions = b.instructions || "";
      state.text = b.providedText || "";
      state.model = b.model || "";
      agentSel.value = b.agent || "auto";
      state.agent = agentSel.value;
      notes.value = state.instructions; text.value = state.text;
      paintAgent(); paintMode();
    }
    const kind = running ? "running" : st.state === "error" ? "error" : has ? "done" : "idle";
    sec.setStatus(kind, running ? "Writing" : kind === "done" ? `${job.storyboard.lines.length} lines` : kind === "error" ? "Failed" : ing === "done" ? "Not written" : "Waiting", st.progress);

    // generator: the main content until a script exists, then behind the header button
    const ready = ing === "done";
    const showGen = !running && (!has || genOpen) && ready;
    gen.hidden = !showGen;
    blocked.hidden = ready || has || running;
    toggleGen.hidden = !has || running;
    toggleGen.textContent = genOpen ? "Close" : "Write a new script";
    toggleGen.setAttribute("aria-expanded", String(genOpen));
    genTitle.textContent = has ? "Replace the script" : "Write the script";
    genSub.textContent = has ? "The current script is kept under Earlier scripts." : "Pick how the narration is written. You can edit every line afterwards.";
    runBtn.disabled = !ready || running;
    runHint.textContent = !ready ? "The paper has to finish loading first." : "";

    activity.hidden = !running;
    bar.set(st.progress, "Writing the script");
    msg.textContent = st.message || "Starting";
    log.hidden = !(running || st.state === "error");
    feed.sync(running, st.started);
    errBox.hidden = st.state !== "error";
    if (st.state === "error") replace(errBox, h("p", h("strong", "The script was not written. "), st.error || st.message, " Try another agent, or choose simple rules."));

    editor.sync(job, { busy: running });
    paintVersions(job.scriptVersions || []);

    // preview
    const pv = job.stages.preview;
    const pvRun = pv.state === "running";
    pvBtn.disabled = !env.engineReady || pvRun;
    replace(pvHint, installEngineHint());
    pvProgress.hidden = !pvRun;
    pvBar.set(pv.progress, "Rendering previews");
    pvMsg.textContent = pvRun ? pv.message : "";
    pvErr.hidden = pv.state !== "error";
    pvErr.textContent = pv.state === "error" ? `Previews failed. ${pv.error || pv.message} Check that the video engine is installed, then try again.` : "";
    const sig = JSON.stringify([job.previews, pv.finished]);
    if (sig !== stripSig) {
      stripSig = sig;
      const items = job.previews.map((u, i) => ({ url: `${u}?v=${Math.round(pv.finished || 0)}`, title: `Scene ${i + 1}`, caption: job.storyboard?.lines[i]?.text }));
      replace(strip, items.map((it, i) => {
        const type = job.storyboard?.lines[i]?.scene?.type;
        return h("li", h("button.shot", { type: "button", onclick: () => lightbox(items, i), "aria-label": `Open preview of scene ${i + 1}` },
          h("img", { src: it.url, alt: "" }), h("span.shot-label", h("span.mono", String(i + 1).padStart(2, "0")), SCENES[type]?.label || "")));
      }));
      strip.hidden = !items.length;
      pvEmpty.hidden = items.length > 0;
    }
  }

  return { update, destroy() { feed.destroy(); unsubEnv(); }, editor };
}
