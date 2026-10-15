import { h, icon, replace, timeAgo } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { jobHref, navigate } from "../router.js";
import { toast, errorMessage, confirmDialog, emptyState, skel, themeControl } from "../ui.js";

const STAGE_LABELS = [["ingest", "Paper"], ["script", "Script"], ["narration", "Narration"], ["render", "Video"]];

/** Short status for a project row: what is running, what failed, or what comes next. */
function projectStatus(j) {
  const st = j.stages;
  const failed = STAGE_LABELS.find(([k]) => st[k] === "error");
  if (failed) return { kind: "error", text: `${failed[1]} failed` };
  const running = STAGE_LABELS.find(([k]) => st[k] === "running");
  if (running) return { kind: "running", text: `${running[1]} running` };
  const n = j.outputs?.length || 0;
  if (n) return { kind: "done", text: `${n} video${n === 1 ? "" : "s"}` };
  const next = STAGE_LABELS.find(([k]) => st[k] !== "done");
  return { kind: "idle", text: next ? `Next: ${next[1].toLowerCase()}` : "Ready" };
}

export function mountHome(root) {
  let alive = true;
  let timer;

  // ---------------------------------------------------------------- input
  const arxiv = h("input#arxiv", { type: "text", name: "arxiv", autocomplete: "off", spellcheck: false, placeholder: "1706.03762", "aria-describedby": "arxiv-help arxiv-err" });
  const arxivErr = h("p#arxiv-err.field-error", { role: "alert", hidden: true });
  const loadBtn = h("button.btn.primary.lg", { type: "submit" }, "Load paper");
  const fileInput = h("input", { type: "file", accept: "application/pdf,.pdf", hidden: true, "aria-label": "Choose a PDF file", onchange: () => fileInput.files[0] && submit({ file: fileInput.files[0] }) });

  const showErr = (msg) => { arxivErr.hidden = !msg; arxivErr.textContent = msg || ""; arxiv.setAttribute("aria-invalid", msg ? "true" : "false"); };
  let busy = false;
  async function submit({ file }) {
    if (busy) return;
    const text = arxiv.value.trim();
    if (!file && !text) { showErr("Enter an arXiv id or link, or drop a PDF."); arxiv.focus(); return; }
    showErr("");
    busy = true;
    loadBtn.disabled = true;
    loadBtn.textContent = file ? "Uploading" : "Loading";
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      else form.append("arxiv", text);
      const job = await api.createJob(form);
      navigate(jobHref(job.id));
    } catch (e) {
      showErr(errorMessage(e));
    } finally {
      busy = false;
      loadBtn.disabled = false;
      loadBtn.textContent = "Load paper";
      fileInput.value = "";
    }
  }

  const form = h("form.arxiv-form", { novalidate: true, onsubmit: (e) => { e.preventDefault(); submit({}); } },
    h("label", { for: "arxiv" }, "arXiv id or link"),
    h("div.input-row", arxiv, loadBtn),
    arxivErr,
    h("p#arxiv-help.pdf-line", "No arXiv id? ",
      h("button.link-btn", { type: "button", onclick: () => fileInput.click() }, "Choose a PDF"),
      " or drop one anywhere on this page."),
    fileInput);

  // Drop a PDF anywhere on the page.
  let depth = 0;
  const home = h("div.home");
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  home.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; home.classList.add("dropping"); });
  home.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  home.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) home.classList.remove("dropping"); });
  home.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    home.classList.remove("dropping");
    const f = [...(e.dataTransfer?.files || [])].find((x) => /pdf$/i.test(x.name) || x.type === "application/pdf");
    if (f) submit({ file: f });
    else showErr("That is not a PDF. Drop a .pdf file, or enter an arXiv id.");
  });

  // ---------------------------------------------------------------- environment
  const envRow = h("div.env-row", { "aria-busy": "true" }, skel("70%"));
  const envNotes = h("ul.env-notes");

  function renderEnv(s) {
    const setup = s.setup || {};
    const setupAction = (key, label, disabled) => {
      const st = setup[key];
      if (st?.state === "running") return h("span.env-progress", { role: "status" }, icon("circle-notch", 13), st.message || "Working");
      return h("button.link-btn", { type: "button", disabled, onclick: async (e) => {
        e.currentTarget.disabled = true;
        try { await env.setup(key); } catch (err) { toast(errorMessage(err), "error"); e.currentTarget.disabled = false; }
      } }, label);
    };
    const a = s.agents || {}, t = s.tools || {};
    const items = [
      { ok: a.claude?.found, name: "Claude Code", note: "Not found. Script writing can use Codex or simple rules." },
      { ok: a.codex?.found, name: "Codex", note: "Not found. Script writing can use Claude Code or simple rules." },
      { ok: t.ffmpeg, name: "ffmpeg", note: "Not found. Narration and video need it. Install it with brew install ffmpeg." },
      { ok: t.node, name: "Node.js", note: "Not found. The video engine needs it. Install it from nodejs.org." },
      { ok: t.engine, name: "Video engine", note: "Not installed. Scene previews and video rendering are off.", action: !t.engine && setupAction("engine", "Install", !t.node) },
      { ok: s.voices?.kokoro?.ready, name: "Kokoro voice", note: "Not downloaded, about 340 MB. Narration uses system voices until it is.", action: !s.voices?.kokoro?.ready && setupAction("kokoro", "Download", false) },
      { ok: s.whisper, name: "Speech recognition", note: "Downloads the first time you split or transcribe a recording.", soft: true },
    ];
    envRow.removeAttribute("aria-busy");
    replace(envRow, h("span.env-label", "Environment"),
      items.map((it) => h(`span.env-item.${it.ok ? "ok" : it.soft ? "soft" : "warn"}`,
        icon(it.ok ? "check" : "warning-circle", 13), it.name, !it.ok && it.action)));
    replace(envNotes, items.filter((it) => !it.ok && !it.soft).map((it) => h("li", h("strong", it.name + ". "), it.note)));
    const setupErr = Object.entries(setup).find(([, v]) => v.state === "error");
    if (setupErr) envNotes.append(h("li.err", { role: "alert" }, `${setupErr[1].message} Try again, or check your network.`));
  }
  const unsub = env.subscribe(renderEnv);
  env.refresh();

  // ---------------------------------------------------------------- recent
  const recentList = h("ul.recent-list", { "aria-busy": "true" },
    [70, 55, 62].map((w) => h("li.recent-row", h("div.recent-main", skel(w + "%"), skel("34%")), skel("64px"))));
  const recentCount = h("span.recent-count");
  let lastRecent = "";

  async function loadRecent() {
    try {
      const list = await api.listJobs();
      const json = JSON.stringify(list);
      if (json !== lastRecent) {
        lastRecent = json;
        renderRecent(list);
      }
      const running = list.some((j) => Object.values(j.stages).includes("running"));
      schedule(running ? 1000 : 5000);
    } catch (e) {
      if (!lastRecent) replace(recentList, h("li.recent-error", { role: "alert" }, errorMessage(e), " ", h("button.link-btn", { type: "button", onclick: loadRecent }, "Try again")));
      recentList.removeAttribute("aria-busy");
      schedule(3000);
    }
  }
  const schedule = (ms) => { clearTimeout(timer); if (alive && !document.hidden) timer = setTimeout(loadRecent, ms); };
  const vis = () => { if (!document.hidden) loadRecent(); };
  document.addEventListener("visibilitychange", vis);

  const segs = (st) => h("span.segs", { role: "img", "aria-label": STAGE_LABELS.map(([k, n]) => `${n} ${st[k] === "idle" ? "not started" : st[k]}`).join(", ") },
    STAGE_LABELS.map(([k]) => h(`i.${st[k]}`)));

  function renderRecent(list) {
    recentList.removeAttribute("aria-busy");
    recentCount.textContent = list.length ? String(list.length) : "";
    if (!list.length) {
      replace(recentList, h("li", emptyState("No projects yet", "Enter an arXiv id or drop a PDF and your first project shows up here.")));
      return;
    }
    replace(recentList, list.map((j) => {
      const src = j.source.type === "arxiv" ? `arXiv:${j.source.id}` : j.source.filename || "PDF";
      const s = projectStatus(j);
      return h("li.recent-row",
        h("div.recent-main",
          h("a.recent-title", { href: jobHref(j.id) }, j.title || "Untitled"),
          h("span.recent-src", src, " ", h("span.recent-when", timeAgo(j.updated || j.created)))),
        h("div.recent-state", segs(j.stages), h(`span.recent-status.${s.kind}`, s.text)),
        h("button.icon-btn.row-del", { "aria-label": `Delete project ${j.title || j.id}`, title: "Delete", onclick: async () => {
          const ok = await confirmDialog({ title: "Delete this project?", body: "The paper, script, narration and videos for this project are removed from this computer.", confirm: "Delete project", danger: true });
          if (!ok) return;
          try { await api.deleteJob(j.id); loadRecent(); } catch (e) { toast(errorMessage(e), "error"); }
        } }, icon("trash", 15)));
    }));
  }
  loadRecent();

  home.append(
    h("header.home-top", h("a.brand", { href: "#/", "aria-label": "vidxiv" }, h("span.brand-mark", { "aria-hidden": "true" }), "vidxiv"), themeControl()),
    h("div.home-grid",
      h("section.start", { "aria-labelledby": "start-h" },
        h("h1#start-h.home-title", "Turn a paper into a video"),
        h("p.lead", "It writes the script, narrates it and renders a short explainer video."),
        form,
        h("div.env", envRow, envNotes)),
      h("section.recent", { "aria-labelledby": "recent-h" },
        h("div.recent-head", h("h2#recent-h.sub-title", "Recent projects"), recentCount),
        recentList)),
    h("div.drop-hint", { "aria-hidden": "true" }, "Drop the PDF to load it"));
  root.append(home);

  return () => { alive = false; clearTimeout(timer); unsub(); document.removeEventListener("visibilitychange", vis); };
}
