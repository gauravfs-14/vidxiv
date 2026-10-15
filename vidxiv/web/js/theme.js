const KEY = "vidxiv-theme";
const mq = window.matchMedia("(prefers-color-scheme: dark)");

/** "system", "light" or "dark": what the user chose. */
export function getMode() {
  const v = localStorage.getItem(KEY);
  return v === "light" || v === "dark" ? v : "system";
}

export function currentTheme() {
  const m = getMode();
  return m === "system" ? (mq.matches ? "dark" : "light") : m;
}

function apply() {
  const t = currentTheme();
  document.documentElement.dataset.theme = t;
  document.documentElement.style.colorScheme = t;
  window.dispatchEvent(new Event("vidxiv-theme"));
}

export function setMode(mode) {
  if (mode === "system") localStorage.removeItem(KEY);
  else localStorage.setItem(KEY, mode);
  apply();
}

export function initTheme() {
  apply();
  mq.addEventListener("change", () => { if (getMode() === "system") apply(); });
}
