import { h, icon, $ } from "./dom.js";
import { onConnection } from "./api.js";
import { startRouter } from "./router.js";
import { initTheme, toggleTheme, currentTheme } from "./theme.js";
import { mountHome } from "./views/home.js";
import { mountJob } from "./views/job.js";

initTheme();

const view = $("#view");
const banner = $("#offline");
let unmount = null;

// theme toggle
const themeBtn = $("#theme-toggle");
const paintTheme = () => {
  const dark = currentTheme() === "dark";
  themeBtn.replaceChildren(icon(dark ? "sun" : "moon", 18));
  themeBtn.setAttribute("aria-label", dark ? "Switch to the paper theme" : "Switch to the ink theme");
  themeBtn.title = themeBtn.getAttribute("aria-label");
};
themeBtn.addEventListener("click", () => { toggleTheme(); paintTheme(); });
paintTheme();

// server connection banner
onConnection((ok) => { banner.hidden = ok; });

startRouter((route) => {
  unmount?.();
  view.replaceChildren();
  window.scrollTo(0, 0);
  document.title = "vidxiv";
  unmount = route.name === "job" ? mountJob(view, route.id) : mountHome(view);
  view.focus({ preventScroll: true });
});
