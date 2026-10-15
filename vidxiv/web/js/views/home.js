import { h, icon, replace, timeAgo } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { jobHref, navigate } from "../router.js";
import { toast, errorMessage, confirmDialog } from "../ui.js";

export function mountHome(root) {
  let alive = true;
  let timer;

  // ---------------------------------------------------------------- input
  const arxiv = h("input#arxiv", { type: "text", name: "arxiv", autocomplete: "off", spellcheck: false, placeholder: "1706.03762 or arxiv.org/abs/1706.03762" });
  const arxivErr = h("p.field-error", { role: "alert", hidden: true });
  const loadBtn = h("button.btn.primary", { type: "submit" }, "Load paper");
  const fileInput = h("input", { type: "file", accept: "application/pdf,.pdf", hidden: true, "aria-label": "Choose a PDF file", onchange: () => fileInput.files[0] && submit({ file: fileInput.files[0] }) });
  const dropMsg = h("span.drop-main", "Drop a PDF here");
  const drop = h("div.drop", { tabIndex: 0, role: "button", "aria-label": "Choose a PDF file to upload" },
    icon("file-pdf", 28), h("div", dropMsg, h("span.drop-sub", "or ", h("span.link", "choose a file"), ". Papers up to a few hundred pages work.")));
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = [...(e.dataTransfer?.files || [])].find((x) => /pdf$/i.test(x.name) || x.type === "application/pdf");
    if (f) submit({ file: f });
    else showErr("That is not a PDF. Drop a .pdf file, or enter an arXiv id.");
  });

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
    }
  }

  const form = h("form.arxiv-form", { novalidate: true, onsubmit: (e) => { e.preventDefault(); submit({}); } },
    h("label", { for: "arxiv" }, "arXiv id or link"),
    h("div.input-row", arxiv, loadBtn),
    arxivErr);

  // ---------------------------------------------------------------- environment
  const envList = h("ul.env-list");
  const envPanel = h("section.panel.env", { "aria-labelledby": "env-h" },
    h("h2#env-h.panel-title", "Environment"),
    h("p.muted.small", "Missing pieces switch off only the steps that need them."),
    envList);

  function envRow({ ok, name, detail, note, action, level = "warn" }) {
    return h("li.env-row" + (ok ? ".ok" : "." + level),
      h("span.env-mark", icon(ok ? "check" : "warning-circle", 15)),
      h("div.env-body",
        h("div.env-name", name, ok ? h("span.env-detail", detail) : h("span.env-state", "Missing")),
        !ok && h("p.env-note", note)),
      action);
  }

  function renderEnv(s) {
    const setup = s.setup || {};
    const setupAction = (key, label, disabled) => {
      const st = setup[key];
      if (st?.state === "running") return h("span.env-progress", { role: "status" }, icon("circle-notch", 14), st.message || "Working");
      return h("div.env-act",
        h("button.btn.small", { type: "button", disabled, onclick: async (e) => {
          e.currentTarget.disabled = true;
          try { await env.setup(key); } catch (err) { toast(errorMessage(err), "error"); e.currentTarget.disabled = false; }
        } }, label),
        st?.state === "error" && h("p.field-error", { role: "alert" }, `${st.message} Try again, or check your network.`));
    };
    const a = s.agents || {}, t = s.tools || {};
    const rows = [
      envRow({ ok: a.claude?.found, name: "Claude Code CLI", detail: a.claude?.version || "Found", note: "Not found. Script writing can use Codex or simple rules." }),
      envRow({ ok: a.codex?.found, name: "Codex CLI", detail: a.codex?.version || "Found", note: "Not found. Script writing can use Claude Code or simple rules." }),
      envRow({ ok: t.ffmpeg, name: "ffmpeg", detail: "Found", note: "Not found. Narration and video need it. Install it with brew install ffmpeg." }),
      envRow({ ok: t.node, name: "Node.js", detail: "Found", note: "Not found. The video engine needs it. Install it from nodejs.org." }),
      envRow({ ok: t.engine, name: "Video engine", detail: "Installed", note: "Not installed. Scene previews and video rendering are off.", action: !t.engine && setupAction("engine", "Install video engine", !t.node) }),
      envRow({ ok: s.voices?.kokoro?.ready, name: "Kokoro voice model", detail: "Downloaded", note: "Not downloaded, about 340 MB. Narration uses system voices until it is.", action: !s.voices?.kokoro?.ready && setupAction("kokoro", "Download voice model", false) }),
      envRow({ ok: s.whisper, name: "Speech recognition", detail: "Downloaded", note: "Downloads the first time you split or transcribe a recording.", level: "info" }),
    ];
    replace(envList, rows);
  }
  const unsub = env.subscribe(renderEnv);
  env.refresh();

  // ---------------------------------------------------------------- recent
  const recentList = h("ul.recent-list", { "aria-busy": "true" },
    [1, 2, 3].map(() => h("li.recent-row.skeleton", h("span.sk.w60"), h("span.sk.w30"))));
  const recentEmpty = h("p.empty", { hidden: true }, "No projects yet. Load a paper above to start one.");
  let lastRecent = "";

  const stageTrack = (st) => h("span.track", { "aria-label": "Stage status" },
    [["Paper", st.ingest], ["Script", st.script], ["Narration", st.narration], ["Video", st.render]].map(([n, s]) =>
      h(`span.step.${s}`, { title: `${n}: ${s === "idle" ? "not started" : s}` }, n)));

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
      if (!lastRecent) replace(recentList, h("li.recent-error", errorMessage(e)));
      recentList.removeAttribute("aria-busy");
      schedule(3000);
    }
  }
  const schedule = (ms) => { clearTimeout(timer); if (alive && !document.hidden) timer = setTimeout(loadRecent, ms); };
  const vis = () => { if (!document.hidden) loadRecent(); };
  document.addEventListener("visibilitychange", vis);

  function renderRecent(list) {
    recentList.removeAttribute("aria-busy");
    recentEmpty.hidden = list.length > 0;
    replace(recentList, list.map((j) => {
      const src = j.source.type === "arxiv" ? `arXiv:${j.source.id}` : j.source.filename || "PDF";
      return h("li.recent-row",
        h("div.recent-main",
          h("a.recent-title", { href: jobHref(j.id) }, j.title || "Untitled"),
          h("span.mono.muted.small", src)),
        h("span.recent-when.muted.small", timeAgo(j.updated || j.created)),
        stageTrack(j.stages),
        j.outputs?.length ? h("span.small.muted.recent-out", `${j.outputs.length} video${j.outputs.length === 1 ? "" : "s"}`) : h("span.recent-out"),
        h("button.icon-btn", { "aria-label": `Delete project ${j.title || j.id}`, title: "Delete", onclick: async () => {
          const ok = await confirmDialog({ title: "Delete this project?", body: "The paper, script, narration and videos for this project are removed from this computer.", confirm: "Delete project", danger: true });
          if (!ok) return;
          try { await api.deleteJob(j.id); loadRecent(); } catch (e) { toast(errorMessage(e), "error"); }
        } }, icon("trash")));
    }));
  }
  loadRecent();

  root.append(h("div.home",
    h("section.start",
      h("h1.display", "Paper to video"),
      h("p.lead", "Load an arXiv paper or a PDF. vidxiv writes the script, narrates it and renders a short explainer video."),
      form,
      h("p.or", h("span", "or")),
      drop, fileInput),
    envPanel,
    h("section.recent", { "aria-labelledby": "recent-h" },
      h("h2#recent-h.section-title", "Recent projects"),
      recentList, recentEmpty)));

  return () => { alive = false; clearTimeout(timer); unsub(); document.removeEventListener("visibilitychange", vis); };
}
