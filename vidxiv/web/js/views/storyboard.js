import { h, icon, replace, autosize, debounce } from "../dom.js";
import { api } from "../api.js";
import { popover, errorMessage, toast } from "../ui.js";
import { SCENES, SCENE_TYPES, getPath, setPath, defaultScene } from "../scenes.js";

const SAVE_DELAY = 900;
let rowUid = 0;

/**
 * Storyboard editor. Owns a local draft; server state is adopted only when the draft is clean,
 * so a poll can never overwrite text the user is typing.
 */
export class StoryboardEditor {
  constructor({ store }) {
    this.store = store;
    this.rows = [];
    this.meta = { title: "" }; // everything on the storyboard except `lines`, so unknown top-level keys survive a save
    this.serverJson = "";
    this.pending = null;
    this.dirty = false;
    this.saving = false;
    this.version = 0;
    this.figures = [];
    this.lint = {};
    this.problems = new Map();
    this.hold = false; // true while the script stage runs

    this.status = h("span.save-state", { role: "status", "aria-live": "polite" });
    this.titleInput = h("input", { type: "text", maxLength: 160, "aria-label": "Video title", oninput: () => { this.meta.title = this.titleInput.value; this.touch(); } });
    this.general = h("div.notice.error", { role: "alert", hidden: true });
    this.list = h("ol.lines");
    this.empty = h("p.empty", { hidden: true }, "The script has no lines.");
    this.save = debounce(() => this.persist(), SAVE_DELAY);
    this.tools = h("div.editor-tools");
    this.slot = h("div.editor-slot");
    this.count = h("span.editor-count");
    this.addEnd = h("button.btn.ghost.small.add-end", { type: "button", onclick: () => this.addBelow(this.rows[this.rows.length - 1]) }, icon("plus", 14), "Add line");
    this.el = h("div.editor", { hidden: true },
      h("div.editor-bar",
        h("label.editor-title", h("span.sr-only", "Video title"), this.titleInput),
        this.status, this.tools),
      this.general, this.slot,
      h("div.lines-head", { "aria-hidden": "true" }, h("span", "Line"), h("span", "Scene"), h("span", "Narration")),
      this.list, this.empty, this.addEnd);
    this.el.addEventListener("focusout", () => setTimeout(() => this.adoptPending(), 0));
    window.addEventListener("beforeunload", (e) => { if (this.dirty) { e.preventDefault(); e.returnValue = ""; } });
  }

  get ctx() { return { figures: this.figures, title: this.meta.title }; }

  // ---------------------------------------------------------------- server state

  sync(job, { busy }) {
    this.figures = job.figures || [];
    this.lint = job.lint || {};
    this.hold = busy;
    const sb = job.storyboard;
    this.el.hidden = !sb || busy;
    if (!sb) { this.serverJson = ""; this.rows = []; replace(this.list); return; }
    const json = JSON.stringify(sb);
    if (json !== this.serverJson) {
      this.pending = { sb, json };
      this.adoptPending();
    }
    this.renderNotes();
  }

  adoptPending() {
    if (!this.pending) return;
    if (this.dirty || this.saving) return;
    if (this.rows.length && this.el.contains(document.activeElement)) return;
    const { sb, json } = this.pending;
    this.pending = null;
    this.serverJson = json;
    const { lines: _lines, ...meta } = structuredClone(sb);
    this.meta = meta;
    if (document.activeElement !== this.titleInput) this.titleInput.value = sb.title || "";
    this.problems.clear();
    this.rows = sb.lines.map((line) => ({ id: ++rowUid, line: structuredClone(line) }));
    this.rows.forEach((r) => this.buildRow(r));
    this.layout();
    this.setStatus("saved");
  }

  refit() { this.rows.forEach((r) => autosize(r.refs.text)); }

  // ---------------------------------------------------------------- saving

  setStatus(kind, text) {
    this.statusKind = kind;
    const map = { saved: ["Saved", "check"], saving: ["Saving", "circle-notch"], dirty: ["Unsaved changes", null], error: [text || "Not saved", "warning-circle"], wait: [text, "info"] };
    const [label, ic] = map[kind];
    this.status.className = `save-state ${kind}`;
    replace(this.status, ic && icon(ic, 14), label, kind === "error" && !text?.includes("notes") && h("button.link-btn", { type: "button", onclick: () => this.persist() }, "Retry"));
  }

  touch() {
    this.version++;
    this.dirty = true;
    if (this.statusKind !== "error") this.setStatus("dirty");
    this.save();
  }

  payload() {
    return { ...this.meta, lines: this.rows.map((r) => r.line) };
  }

  async persist() {
    this.save.cancel();
    if (this.saving) { this.again = true; return; }
    if (!this.dirty) return;
    const bad = this.rows.findIndex((r) => r.line.text.trim().length < 2);
    if (bad >= 0) { this.setStatus("wait", `Line ${bad + 1} needs narration text before it can be saved`); return; }
    if (!this.meta.title.trim()) { this.setStatus("wait", "Add a video title to save"); return; }
    this.saving = true;
    const v = this.version;
    this.setStatus("saving");
    try {
      const view = await api.putStoryboard(this.store.id, this.payload());
      this.serverJson = JSON.stringify(view.storyboard);
      this.pending = null;
      this.problems.clear();
      this.store.apply(view);
      if (this.version === v) { this.dirty = false; this.setStatus("saved"); } else { this.save(); }
    } catch (e) {
      if (e.status === 422) {
        this.setProblems(e.problems);
        this.setStatus("error", "Not saved. See the notes below");
      } else {
        this.setStatus("error", `Not saved. ${errorMessage(e)}`);
        setTimeout(() => { if (this.dirty && this.statusKind === "error" && !this.saving) this.persist(); }, 5000);
      }
    } finally {
      this.saving = false;
      if (this.again) { this.again = false; if (this.dirty && this.statusKind !== "error") this.persist(); }
      this.adoptPending();
    }
  }

  flush() { return this.dirty ? this.persist() : Promise.resolve(); }

  setProblems(list) {
    this.problems.clear();
    const general = [];
    for (const p of list || []) {
      const m = p.match(/^lines\.(\d+)[.:]?\s*(.*)$/);
      if (m) {
        const i = +m[1];
        this.problems.set(i, [...(this.problems.get(i) || []), m[2] || p]);
      } else general.push(p);
    }
    replace(this.general, general.length ? [h("p", h("strong", "The script has problems. "), "Fix these and it saves again."), h("ul", general.map((g) => h("li", g)))] : null);
    this.general.hidden = !general.length;
    this.renderNotes();
  }

  // ---------------------------------------------------------------- rows

  layout() {
    replace(this.list, this.rows.map((r) => r.el));
    this.empty.hidden = this.rows.length > 0;
    this.addEnd.hidden = this.rows.length === 0;
    this.count.textContent = `${this.rows.length} line${this.rows.length === 1 ? "" : "s"}`;
    this.renumber();
    requestAnimationFrame(() => this.refit());
  }

  renumber() {
    const n = this.rows.length;
    this.rows.forEach((r, i) => {
      const k = i + 1;
      r.refs.idx.textContent = String(k).padStart(2, "0");
      r.refs.type.setAttribute("aria-label", `Scene type for line ${k}`);
      r.refs.text.setAttribute("aria-label", `Narration for line ${k}`);
      r.refs.up.disabled = i === 0;
      r.refs.down.disabled = i === n - 1;
      r.refs.del.disabled = n === 1;
      r.refs.up.setAttribute("aria-label", `Move line ${k} up`);
      r.refs.down.setAttribute("aria-label", `Move line ${k} down`);
      r.refs.add.setAttribute("aria-label", `Add a line below line ${k}`);
      r.refs.more.setAttribute("aria-label", `${r.refs.detail.hidden ? "Show" : "Hide"} scene fields for line ${k}`);
      r.refs.del.setAttribute("aria-label", `Delete line ${k}`);
    });
    this.renderNotes();
  }

  renderNotes() {
    this.rows.forEach((r, i) => {
      const lint = this.lint[i] || [];
      const probs = this.problems.get(i) || [];
      const key = JSON.stringify([lint, probs]);
      if (r.notesKey === key) return;
      r.notesKey = key;
      replace(r.refs.notes,
        probs.map((p) => h("p.note.err", icon("warning-circle", 14), p)),
        lint.map((l) => h("p.note.warn", icon("info", 14), l)));
      r.refs.notes.hidden = !lint.length && !probs.length;
    });
  }

  buildRow(r) {
    const refs = {};
    r.refs = refs;
    refs.idx = h("span.line-idx");
    refs.type = h("select.type-select", { onchange: () => this.changeType(r, refs.type.value) },
      SCENE_TYPES.map((t) => h("option", { value: t, disabled: t === "figure" && !this.figures.length, title: t === "figure" && !this.figures.length ? "This paper has no extracted figures" : undefined }, SCENES[t].label)));
    const act = (name, ic, fn) => (refs[name] = h("button.icon-btn", { type: "button", onclick: fn }, icon(ic, 15)));
    refs.text = h("textarea.narration", { rows: 1, maxLength: 420, spellcheck: true, value: r.line.text, oninput: () => {
      r.line.text = refs.text.value;
      autosize(refs.text);
      refs.count.textContent = refs.text.value.length > 340 ? `${refs.text.value.length} of 420` : "";
      this.touch();
    } });
    refs.count = h("span.count");
    refs.notes = h("div.line-notes", { hidden: true });
    refs.fields = h("div.scene-fields");
    refs.peek = h("p.line-peek");
    refs.json = h("textarea.json", { rows: 6, spellcheck: false, "aria-label": "Line JSON", oninput: () => this.jsonInput(r) });
    refs.jsonErr = h("p.note.err", { hidden: true });
    refs.more = h("button.icon-btn.more", { type: "button", "aria-expanded": "false", onclick: () => this.toggle(r) }, icon("caret-down", 15));
    refs.detail = h("div.line-detail", { id: `ld-${r.id}`, hidden: true },
      refs.fields,
      h("details.json-edit", { ontoggle: (e) => { if (e.target.open) autosize(refs.json); } },
        h("summary", "Edit as JSON"),
        h("p.hint", "Scene, highlight words, kicker and gapAfter for this line. The narration is edited above."),
        refs.json, refs.jsonErr));
    refs.more.setAttribute("aria-controls", `ld-${r.id}`);
    r.el = h("li.line", { dataset: { id: r.id } },
      refs.idx,
      h("span.type-wrap", { dataset: { tint: SCENES[r.line.scene.type]?.tint || "ink" } }, refs.type, icon("caret-down", 11)),
      h("div.line-main", refs.text, refs.count, refs.peek, refs.notes),
      h("div.line-actions",
        act("up", "arrow-up", () => this.move(r, -1)),
        act("down", "arrow-down", () => this.move(r, 1)),
        act("add", "plus", () => this.addBelow(r)),
        act("del", "trash", () => this.remove(r)),
        refs.more),
      refs.detail);
    refs.type.value = SCENES[r.line.scene.type] ? r.line.scene.type : "concept";
    this.buildFields(r);
    this.writeJson(r);
  }

  toggle(r, open = r.refs.detail.hidden) {
    r.refs.detail.hidden = !open;
    r.refs.more.setAttribute("aria-expanded", String(open));
    r.el.classList.toggle("open", open);
    r.refs.more.setAttribute("aria-label", `${open ? "Hide" : "Show"} scene fields for line ${this.rows.indexOf(r) + 1}`);
    if (open) r.refs.fields.querySelectorAll("textarea").forEach(autosize);
  }

  /** One muted line under the narration with the main on-screen text of the scene. */
  peek(r) {
    const scene = r.line.scene;
    const cfg = SCENES[scene.type];
    const first = cfg?.fields.map((f) => getPath(scene, f.path)).find((v) => typeof v === "string" && v.trim());
    r.refs.peek.textContent = first ? `On screen: ${first}` : cfg?.figure && scene.figure ? "On screen: figure" : "";
    r.refs.peek.hidden = !r.refs.peek.textContent;
  }

  buildFields(r) {
    const scene = r.line.scene;
    const cfg = SCENES[scene.type];
    const nodes = [];
    if (!cfg) { replace(r.refs.fields, h("p.hint", `Unknown scene type "${scene.type}". Edit it as JSON.`)); return; }
    const inputs = cfg.fields.map((f) => {
      const long = f.max > 110 && !f.mono;
      const el = h(long ? "textarea" : "input", { type: long ? undefined : "text", rows: long ? 1 : undefined, maxLength: f.max, value: getPath(scene, f.path) ?? "", spellcheck: !f.mono, class: f.mono ? "mono" : "",
        "aria-label": `${f.label} for line ${this.rows.indexOf(r) + 1}`,
        oninput: () => {
          setPath(scene, f.path, el.value, f.required);
          if (long) autosize(el);
          this.writeJson(r);
          this.peek(r);
          this.touch();
        } });
      return h("label.sf", { style: `grid-column: span ${f.span || 6}` }, h("span", f.label), el);
    });
    nodes.push(h("div.sf-grid", inputs));
    if (cfg.figure) nodes.push(this.figureControl(r));
    const note = cfg.note?.(scene);
    if (note) nodes.push(h("p.hint.scene-note", note));
    replace(r.refs.fields, nodes);
    this.peek(r);
    requestAnimationFrame(() => r.refs.fields.querySelectorAll("textarea").forEach(autosize));
  }

  figureControl(r) {
    const scene = r.line.scene;
    const required = scene.type === "figure";
    const btn = h("button.fig-pick", { type: "button", "aria-haspopup": "dialog" });
    const paint = () => {
      const f = this.figures.find((x) => x.id === scene.figure);
      replace(btn,
        f ? h("img", { src: f.url, alt: "" }) : h("span.fig-none", icon("image", 18)),
        h("span.fig-pick-text", h("span.fig-pick-label", f ? f.label : scene.figure ? `Unknown figure ${scene.figure}` : required ? "Choose a figure" : "No figure"),
          f && h("span.fig-pick-cap", f.caption)),
        icon("caret-down", 12));
    };
    btn.onclick = () => {
      const close = popover(btn, h("div.fig-grid", { role: "listbox", "aria-label": "Figures" },
        !required && h("button.fig-opt", { type: "button", role: "option", "aria-selected": String(!scene.figure), onclick: () => { delete scene.figure; pick(); } }, h("span.fig-none", icon("x", 18)), h("span", "No figure")),
        this.figures.map((f) => h("button.fig-opt", { type: "button", role: "option", "aria-selected": String(f.id === scene.figure), title: f.caption, onclick: () => { scene.figure = f.id; pick(); } },
          h("img", { src: f.url, alt: "" }), h("span", f.label)))));
      const pick = () => { close(); paint(); this.writeJson(r); this.touch(); };
    };
    paint();
    return h("div.sf-fig", h("span.sf-label", "Figure"), btn);
  }

  changeType(r, type) {
    const old = r.line.scene;
    r.line.scene = defaultScene(type, old, r.line.text, this.ctx);
    r.el.querySelector(".type-wrap").dataset.tint = SCENES[type].tint;
    this.toggle(r, true);
    this.buildFields(r);
    this.writeJson(r);
    this.touch();
  }

  writeJson(r) {
    if (document.activeElement === r.refs.json) return;
    const { text, ...rest } = r.line;
    r.refs.json.value = JSON.stringify(rest, null, 2);
    r.refs.jsonErr.hidden = true;
    autosize(r.refs.json);
  }

  jsonInput(r) {
    let parsed;
    try {
      parsed = JSON.parse(r.refs.json.value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || typeof parsed.scene !== "object" || !parsed.scene?.type) throw new Error('It needs a "scene" object with a "type".');
    } catch (e) {
      r.refs.jsonErr.textContent = `Not valid JSON yet. ${e.message}`;
      r.refs.jsonErr.hidden = false;
      return;
    }
    r.refs.jsonErr.hidden = true;
    const text = r.line.text;
    r.line = { ...parsed, text };
    if (SCENES[parsed.scene.type]) {
      r.refs.type.value = parsed.scene.type;
      r.el.querySelector(".type-wrap").dataset.tint = SCENES[parsed.scene.type].tint;
    }
    this.buildFields(r);
    autosize(r.refs.json);
    this.touch();
  }

  // ---------------------------------------------------------------- operations

  warnRecordings() {
    const clips = this.store.job?.narration?.clips || {};
    if (Object.values(clips).some((c) => c.source === "user" || c.userUrl)) {
      toast("Recordings are tied to line numbers. Check the Narration stage after moving or deleting lines.", "info", 8000);
    }
  }

  move(r, d) {
    const i = this.rows.indexOf(r);
    const j = i + d;
    if (j < 0 || j >= this.rows.length) return;
    [this.rows[i], this.rows[j]] = [this.rows[j], this.rows[i]];
    this.layout();
    r.refs[d < 0 ? "up" : "down"].focus();
    if (r.refs[d < 0 ? "up" : "down"].disabled) r.refs.text.focus();
    this.warnRecordings();
    this.touch();
  }

  remove(r) {
    if (this.rows.length === 1) return;
    const i = this.rows.indexOf(r);
    this.rows.splice(i, 1);
    this.layout();
    (this.rows[Math.min(i, this.rows.length - 1)].refs.text).focus();
    this.warnRecordings();
    this.touch();
  }

  addBelow(r) {
    if (this.rows.length >= 60) { toast("A video can have at most 60 lines.", "error"); return; }
    const nr = { id: ++rowUid, line: { text: "", scene: { type: "concept", heading: "" } } };
    this.buildRow(nr);
    this.rows.splice(this.rows.indexOf(r) + 1, 0, nr);
    this.layout();
    nr.refs.text.focus();
    this.warnRecordings();
    this.touch();
  }
}
