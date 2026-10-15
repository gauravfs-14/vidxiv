import { h, icon, replace } from "../dom.js";
import { api } from "../api.js";
import { env } from "../env.js";
import { JobStore } from "../store.js";
import { createSection } from "../section.js";
import { createPaper } from "./paper.js";
import { createScript } from "./script.js";
import { createNarration } from "./narration.js";
import { createVideo } from "./video.js";
import { confirmDialog, toast, errorMessage } from "../ui.js";
import { navigate } from "../router.js";

export function mountJob(root, id) {
  const store = new JobStore(id);
  const title = h("h1.job-title.display", "Loading");
  const meta = h("p.job-meta.muted");
  const del = h("button.btn.ghost.small", { type: "button", onclick: async () => {
    const ok = await confirmDialog({ title: "Delete this project?", body: "The paper, script, narration and videos are removed from this computer.", confirm: "Delete project", danger: true });
    if (!ok) return;
    try { await api.deleteJob(id); navigate("#/"); } catch (e) { toast(errorMessage(e), "error"); }
  } }, icon("trash", 14), "Delete project");
  const head = h("header.job-head", h("a.back", { href: "#/" }, icon("arrow-left", 14), "Projects"), h("div.job-title-row", title, del), meta);

  const defs = [["Paper", createPaper], ["Script", createScript], ["Narration", createNarration], ["Video", createVideo]];
  const parts = defs.map(([name, make], i) => {
    const sec = createSection({ jobId: id, n: i + 1, title: name });
    sec.setChip(h("span.chip.idle", "Loading"));
    return { sec, ...make({ store, sec }) };
  });
  const flow = h("div.flow", parts.map((p) => p.sec.el));
  const missing = h("div.empty-page", { hidden: true },
    h("h2.display", "Project not found"), h("p.muted", "It may have been deleted. Go back to your projects."), h("a.btn", { href: "#/" }, "All projects"));
  root.append(h("div.job", head, flow, missing));

  let first = true;
  const unsub = store.subscribe((job, prev, err) => {
    if (!job) { head.hidden = flow.hidden = true; missing.hidden = false; return; }
    document.title = `${job.title} - vidxiv`;
    replace(title, job.title || "Untitled");
    replace(meta, job.source.type === "arxiv" ? `arXiv:${job.source.id}` : job.source.filename || "PDF upload");
    parts.forEach((p) => p.update(job, prev));

    if (first) {
      first = false;
      const clips = job.narration.clips || {};
      const n = job.storyboard?.lines.length || 0;
      const narrated = n > 0 && Object.keys(clips).filter((k) => clips[k].source && !clips[k].stale).length === n;
      const current = job.stages.ingest.state !== "done" ? 1 : !job.storyboard ? 2 : !narrated ? 3 : 4;
      parts.forEach((p, i) => { const s = p.sec.stored(); p.sec.setOpen(s ?? i + 1 === current, false); });
    } else if (prev && prev.stages.ingest.state === "running" && job.stages.ingest.state === "done") {
      parts[1].sec.setOpen(true, false);
    }
  });
  store.start();
  env.refresh();

  return () => { unsub(); store.stop(); parts.forEach((p) => p.destroy?.()); document.title = "vidxiv"; };
}
