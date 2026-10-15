import { h, icon, replace } from "../dom.js";
import { api } from "../api.js";
import { progressBar, LogFeed, lightbox, toast, errorMessage, confirmDialog, skel } from "../ui.js";
import { navigate, jobHref } from "../router.js";

export function createPaper({ store, sec }) {
  const bar = progressBar();
  const msg = h("p.progress-msg", { "aria-live": "polite" });
  const feed = new LogFeed(store.id, "ingest");
  // Shaped like the loaded paper: title, authors, abstract lines, figure thumbnails.
  const loadingShape = h("div.paper-skeleton", { "aria-hidden": "true" },
    skel("62%", "tall"), skel("38%"), h("div.sk-lines", skel("100%"), skel("96%"), skel("100%"), skel("72%")),
    h("div.figs", [1, 2, 3, 4].map(() => h("div.sk-fig"))));
  const running = h("div.stack", h("div.progress-row", bar), msg, loadingShape, h("details.disclosure", h("summary", "Activity log"), feed.el));
  const errBox = h("div.notice.error", { role: "alert", hidden: true });
  const detail = h("div.paper-detail", { hidden: true });
  sec.body.append(running, errBox, detail);

  let sig = "";
  let fit = null;

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
      p.published,
      p.pages && `${p.pages} pages`,
      figs.length ? `${figs.length} figure${figs.length === 1 ? "" : "s"}` : "No figures extracted",
    ].filter(Boolean);
    const authors = p.authors || [];
    const authorText = authors.length > 6 ? `${authors.slice(0, 6).join(", ")} and ${authors.length - 6} more` : authors.join(", ");
    const items = figs.map((f) => ({ url: f.url, title: f.label, caption: f.caption }));
    replace(detail,
      h("h3.paper-title", p.title || job.title),
      authorText && h("p.authors", authorText),
      h("p.meta", meta.map((m, i) => [i ? h("span.sep", "/") : null, m]),
        p.url && h("a.link.ext", { href: p.url, target: "_blank", rel: "noopener noreferrer" }, "Open source", icon("arrow-square-out", 12))),
      h("div.abstract-wrap", abstract, p.abstract && p.abstract.length > 330 && more),
      h("div.paper-figs",
        h("h3.sub-title", "Figures"),
        figs.length > 0
          ? h("ul.figs", { "aria-label": "Extracted figures" },
            figs.map((f, i) => h("li", h("button.fig", { type: "button", onclick: () => lightbox(items, i), "aria-label": `Open ${f.label}: ${f.caption}` },
              h("span.fig-img", h("img", { src: f.url, alt: "", loading: "lazy" })),
              h("span.fig-label", f.label)))))
          : h("p.muted", "No figures were extracted. Scenes will use text only.")),
      h("div.paper-next", h("a.btn.primary", { href: jobHref(store.id, "script"), onclick: (e) => { e.preventDefault(); sec.go("script"); } }, "Continue to script", icon("arrow-right", 14))));
    fit = () => {
      if (!abstract.offsetParent) return; // hidden pane: measure when it is shown
      if (abstract.classList.contains("clamped") && abstract.scrollHeight <= abstract.clientHeight + 2) { abstract.classList.remove("clamped"); more.remove(); }
      fit = null;
    };
    requestAnimationFrame(() => fit?.());
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

  sec.onShow(() => requestAnimationFrame(() => fit?.()));

  return {
    update(job) {
      const st = job.stages.ingest;
      const kind = st.state === "idle" && job.paper ? "done" : st.state;
      const status = kind === "done" ? "Loaded" : kind === "running" ? "Loading" : kind === "error" ? "Failed" : "Waiting";
      sec.setStatus(kind, status, st.progress);
      const isRun = st.state === "running";
      running.hidden = !(isRun || (st.state === "idle" && !job.paper));
      bar.set(st.progress, "Loading the paper");
      msg.textContent = st.message || "Starting";
      feed.sync(isRun, st.started);
      errBox.hidden = st.state !== "error";
      if (st.state === "error") {
        replace(errBox, h("p", h("strong", "The paper could not be loaded. "), st.error || st.message || "Unknown error.", " Check the arXiv id or the PDF, then start over."),
          h("button.btn", { type: "button", onclick: () => startOver(job) }, icon("arrow-clockwise", 14), "Start over"));
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
