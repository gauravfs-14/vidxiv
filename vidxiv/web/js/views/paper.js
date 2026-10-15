import { h, icon, replace } from "../dom.js";
import { api } from "../api.js";
import { chip, progressBar, LogFeed, lightbox, toast, errorMessage, confirmDialog } from "../ui.js";
import { navigate, jobHref } from "../router.js";

export function createPaper({ store, sec }) {
  const bar = progressBar();
  const msg = h("p.small.muted", { "aria-live": "polite" });
  const feed = new LogFeed(store.id, "ingest");
  const running = h("div.stack", h("div.progress-row", bar, msg), h("details.disclosure", h("summary", "Activity"), feed.el));
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const detail = h("div.paper-detail", { hidden: true });
  sec.body.append(running, errBox, detail);

  let sig = "";

  function renderDetail(job) {
    const p = job.paper;
    const figs = job.figures || [];
    const abstract = h("p.abstract.clamped", p.abstract || "No abstract was found.");
    const more = h("button.link-btn", { type: "button", "aria-expanded": "false", onclick: () => {
      const open = abstract.classList.toggle("clamped") === false;
      more.textContent = open ? "Show less" : "Show more";
      more.setAttribute("aria-expanded", String(open));
    } }, "Show more");
    const meta = [
      p.published && h("span", p.published),
      p.pages && h("span", `${p.pages} pages`),
      figs.length ? h("span", `${figs.length} figure${figs.length === 1 ? "" : "s"} extracted`) : h("span", "No figures extracted"),
    ].filter(Boolean);
    const authors = (p.authors || []);
    const authorText = authors.length > 6 ? `${authors.slice(0, 6).join(", ")} and ${authors.length - 6} more` : authors.join(", ");
    const items = figs.map((f) => ({ url: f.url, title: f.label, caption: f.caption }));
    replace(detail,
      h("h3.paper-title", p.title || job.title),
      authorText && h("p.authors", authorText),
      h("p.meta.small.muted", meta.map((m, i) => [i ? h("span.sep", "/") : null, m]),
        p.url && h("a.link.ext", { href: p.url, target: "_blank", rel: "noopener noreferrer" }, "Open source ", icon("arrow-square-out", 12))),
      abstract, p.abstract && p.abstract.length > 330 && more,
      figs.length > 0 && h("div.figs", { role: "list", "aria-label": "Extracted figures" },
        figs.map((f, i) => h("button.fig", { type: "button", role: "listitem", onclick: () => lightbox(items, i), "aria-label": `Open ${f.label}: ${f.caption}` },
          h("span.fig-img", h("img", { src: f.url, alt: "", loading: "lazy" })),
          h("span.fig-label", f.label)))));
    requestAnimationFrame(() => {
      if (abstract.scrollHeight <= abstract.clientHeight + 2) { abstract.classList.remove("clamped"); more.remove(); }
    });
  }

  async function startOver(job) {
    const src = job.source;
    const ok = await confirmDialog({ title: "Start over?", body: src.type === "arxiv" ? "This deletes the project and loads the paper again." : "This deletes the project. Upload the PDF again from the home page.", confirm: "Start over" });
    if (!ok) return;
    try {
      if (src.type === "arxiv") {
        const f = new FormData();
        f.append("arxiv", src.id);
        const nj = await api.createJob(f);
        await api.deleteJob(job.id);
        navigate(jobHref(nj.id));
      } else {
        await api.deleteJob(job.id);
        navigate("#/");
      }
    } catch (e) { toast(errorMessage(e), "error"); }
  }

  return {
    update(job) {
      const st = job.stages.ingest;
      const kind = st.state === "idle" && job.paper ? "done" : st.state;
      sec.setChip(chip(kind, kind === "idle" ? "Waiting" : undefined, st.progress));
      sec.setSummary(job.paper ? job.paper.title : st.state === "running" ? st.message : "");
      const isRun = st.state === "running";
      running.hidden = !(isRun || (st.state === "idle" && !job.paper));
      bar.set(st.progress, "Loading the paper");
      msg.textContent = st.message || "Starting";
      feed.sync(isRun, st.started);
      errBox.hidden = st.state !== "error";
      if (st.state === "error") {
        replace(errBox, h("p", h("strong", "The paper could not be loaded. "), st.error || st.message || "Unknown error.", " Check the arXiv id or the PDF, then start over."),
          h("button.btn", { type: "button", onclick: () => startOver(job) }, icon("arrow-clockwise"), "Start over"));
      }
      const show = st.state === "done" && job.paper;
      detail.hidden = !show;
      if (show) {
        const s = JSON.stringify([job.paper, job.figures.map((f) => f.id)]);
        if (s !== sig) { sig = s; renderDetail(job); }
      }
    },
    destroy() { feed.destroy(); },
  };
}
