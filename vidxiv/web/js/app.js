import { $, $$, autosize, debounce } from "./dom.js";
import { onConnection } from "./api.js";
import { startRouter } from "./router.js";
import { initTheme } from "./theme.js";
import { initTips } from "./tips.js";
import { mountHome } from "./views/home.js";
import { mountJob } from "./views/job.js";

initTheme();
initTips();

// Text areas size themselves to their content, so they need to re-fit when the width changes.
window.addEventListener("resize", debounce(() => $$("textarea").forEach(autosize), 120));

const view = $("#view");
const banner = $("#offline");
let current = null; // { id, unmount, setStage } for a job, or { unmount } for home

onConnection((ok) => { banner.hidden = ok; });

startRouter((route) => {
  // Moving between stages of the same project keeps the page, so nothing reloads or flickers.
  if (route.name === "job" && current?.id === route.id) {
    if (route.stage) current.setStage(route.stage);
    return;
  }
  current?.unmount();
  view.replaceChildren();
  window.scrollTo(0, 0);
  document.title = "vidxiv";
  if (route.name === "job") {
    const job = mountJob(view, route.id, route.stage);
    current = { id: route.id, ...job };
  } else {
    current = { unmount: mountHome(view) };
  }
  view.focus({ preventScroll: true });
});
