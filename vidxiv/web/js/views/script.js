import { h, icon, replace, autosize, fmtSize } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { chip, progressBar, LogFeed, segmented, field as fieldEl, toast, errorMessage, confirmDialog, lightbox } from "../ui.js";
import { recordClip } from "../recorder.js";
import { StoryboardEditor } from "./storyboard.js";
import { SCENES } from "../scenes.js";

const LENGTHS = [
  { value: "short", label: "Short", sub: "about 45 s" },
  { value: "medium", label: "Medium", sub: "about 90 s" },
  { value: "long", label: "Long", sub: "about 3 min" },
];
const MODES = [
  { value: "generate", label: "Write with an AI agent" },
  { value: "provided", label: "I have a script" },
  { value: "transcript", label: "I have a recording" },
];

export function installEngineHint() {
  if (!env.data || env.engineReady) return null;
  const node = env.data?.tools?.node;
  return h("p.hint.engine-hint", node ? "The video engine is not installed. " : "Node.js is not installed, so the video engine cannot be set up. ",
    node && h("button.link-btn", { type: "button", onclick: () => env.setup("engine").catch((e) => toast(errorMessage(e), "error")) }, "Install video engine"));
}

export function createScript({ store, sec }) {
  const state = { mode: "generate", length: "short", agent: "auto", model: "", instructions: "", text: "", file: null, init: false };
  const editor = new StoryboardEditor({ store });
  sec.onOpen(() => editor.refit());

  // ---------------------------------------------------------------- generator form
  const modeSeg = segmented("script-mode", MODES, state.mode, (v) => { state.mode = v; paintMode(); }, "How the script is written");
  modeSeg.classList.add("seg-wide");

  const lengthSeg = segmented("length", LENGTHS, state.length, (v) => { state.length = v; }, "Video length");
  const notes = h("textarea", { rows: 2, placeholder: "Focus on the method. Skip the related work.", oninput: () => { state.instructions = notes.value; autosize(notes); } });
  const agentSel = h("select", { onchange: () => { state.agent = agentSel.value; paintAgent(); } });
  const agentHint = h("p.hint");
  const model = h("input", { type: "text", placeholder: "Leave empty for the agent default", autocomplete: "off", spellcheck: false, oninput: () => { state.model = model.value.trim(); } });
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

  const paneAgent = h("div.pane", fieldEl("Length", lengthSeg), fieldEl("Notes for the agent", notes, "Optional. What to focus on or leave out."));
  const paneText = h("div.pane", fieldEl("Narration script", text, "The agent only picks scenes. Long text is split into lines of one or two sentences."));
  const paneRec = h("div.pane",
    h("div.field", h("span.label-like", "Recording"),
      h("div.rec-row",
        h("button.btn", { type: "button", onclick: async () => {
          const f = await recordClip({ title: "Record the narration", help: "Speak the whole narration. It is transcribed and cut into lines, then you can edit the text." });
          if (f) setFile(f);
        } }, icon("microphone"), "Record"),
        h("button.btn", { type: "button", onclick: () => audioIn.click() }, icon("upload-simple"), "Upload audio"),
        file.name, file.clear),
      whisperNote, audioIn));
  const adv = h("details.disclosure.adv", h("summary", "Advanced"), fieldEl("Model", model, "Passed to the agent as its model name. Optional."));
  const agentRow = h("div.pane", fieldEl("Scene writer", agentSel, agentHint), adv);

  const runBtn = h("button.btn.primary.lg", { type: "button", onclick: () => run() });
  const runHint = h("p.hint");
  const runErr = h("p.field-error", { role: "alert", hidden: true });
  const form = h("div.gen-form", modeSeg, paneAgent, paneText, paneRec, agentRow, h("div.run-row", runBtn, runHint), runErr);
  const gen = h("details.generator", h("summary", h("span", "Write a new script"), h("span.muted.small.gen-sub", "Replaces the current lines")), form);

  function paintMode() {
    modeSeg.set(state.mode);
    paneAgent.hidden = state.mode !== "generate";
    paneText.hidden = state.mode !== "provided";
    paneRec.hidden = state.mode !== "transcript";
    runBtn.textContent = { generate: "Write script", provided: "Use my script", transcript: "Transcribe recording" }[state.mode];
    whisperNote.textContent = env.data && !env.data.whisper ? "The speech model downloads the first time you transcribe, which can take a minute." : "The recording is transcribed on this computer.";
    autosize(text); autosize(notes);
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
  }

  async function run() {
    const job = store.job;
    runErr.hidden = true;
    if (state.mode === "provided" && !state.text.trim()) return fail("Paste your script first.");
    if (state.mode === "transcript" && !state.file) return fail("Record or upload an audio file first.");
    if (job.storyboard) {
      const ok = await confirmDialog({ title: "Write a new script?", body: "This replaces the current lines and removes any recordings you made for them.", confirm: "Replace script", danger: true });
      if (!ok) return;
    }
    runBtn.disabled = true;
    try {
      await api.putBrief(store.id, { length: state.length, instructions: state.instructions });
      if (state.mode === "transcript") await api.uploadTake(store.id, state.file, "transcribe");
      const view = await api.makeScript(store.id, { agent: state.agent, model: state.model, mode: state.mode, text: state.text });
      store.apply(view);
      store.bump();
    } catch (e) {
      fail(errorMessage(e));
    } finally { runBtn.disabled = false; update(store.job); }
  }
  const fail = (m) => { runErr.textContent = m; runErr.hidden = false; };

  // ---------------------------------------------------------------- activity
  const bar = progressBar();
  const msg = h("p.small.muted.act-msg", { "aria-live": "polite" });
  const cancel = h("button.btn.small", { type: "button", onclick: async () => {
    cancel.disabled = true;
    try { await api.cancel(store.id); store.bump(); } catch (e) { toast(errorMessage(e), "error"); } finally { cancel.disabled = false; }
  } }, icon("x", 14), "Cancel");
  const feed = new LogFeed(store.id, "script");
  const activity = h("div.activity", { hidden: true }, h("div.progress-row", bar, cancel), msg);
  const log = h("details.disclosure", { hidden: true }, h("summary", "Activity log"), feed.el);
  const errBox = h("div.notice.error", { role: "alert", hidden: true });

  // ---------------------------------------------------------------- preview
  const pvBar = progressBar();
  const pvMsg = h("p.small.muted", { "aria-live": "polite" });
  const pvBtn = h("button.btn", { type: "button", onclick: async () => {
    pvBtn.disabled = true;
    try { await editor.flush(); store.apply(await api.preview(store.id)); store.bump(); } catch (e) { toast(errorMessage(e), "error"); }
    update(store.job);
  } }, icon("film-strip"), "Preview scenes");
  const pvErr = h("p.field-error", { role: "alert", hidden: true });
  const pvHint = h("div.hint-slot");
  const strip = h("ul.filmstrip", { "aria-label": "Scene previews" });
  const preview = h("div.preview", { hidden: true },
    h("div.preview-head", h("div", h("h3.sub-title", "Scene previews"), h("p.hint", "Stills of each scene with estimated timing. Pacing in the final video follows the narration.")), pvBtn),
    pvHint, h("div.progress-row", { hidden: true }, pvBar), pvMsg, pvErr, strip);
  const pvProgress = preview.querySelector(".progress-row");
  let stripSig = "";

  sec.body.append(gen, activity, log, errBox, editor.el, preview);
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
      state.length = b.length || "short";
      state.instructions = b.instructions || "";
      state.text = b.providedText || "";
      state.model = b.model || "";
      agentSel.value = b.agent || "auto";
      state.agent = agentSel.value;
      notes.value = state.instructions; text.value = state.text; model.value = state.model;
      lengthSeg.set(state.length);
      if (state.model) adv.open = true;
      paintAgent(); paintMode();
      gen.open = !has;
    }
    // chip and summary
    const kind = running ? "running" : st.state === "error" ? "error" : has ? "done" : "idle";
    sec.setChip(chip(kind, kind === "done" ? `${job.storyboard.lines.length} lines` : undefined, st.progress));
    sec.setSummary(has ? job.storyboard.title : "");

    // run controls
    const ready = ing === "done";
    runBtn.disabled = !ready || running;
    runHint.textContent = !ready ? "The paper has to finish loading first." : running ? "The script is being written." : "";
    form.classList.toggle("busy", running);
    gen.querySelector(".gen-sub").textContent = has ? "Replaces the current lines" : "";
    activity.hidden = !running;
    bar.set(st.progress, "Writing the script");
    msg.textContent = st.message || "Starting";
    log.hidden = st.state === "idle" && !feed.lines.length;
    feed.sync(running, st.started);
    errBox.hidden = st.state !== "error";
    if (st.state === "error") replace(errBox, h("p", h("strong", "The script was not written. "), st.error || st.message, " Try another agent, or choose simple rules."));

    editor.sync(job, { busy: running });

    // preview
    preview.hidden = !has || running;
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
    }
  }

  return { update, destroy() { feed.destroy(); unsubEnv(); }, editor };
}
