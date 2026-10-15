const KEY = "vidxiv-theme";
const mq = window.matchMedia("(prefers-color-scheme: dark)");

export function currentTheme() {
  return document.documentElement.dataset.theme || (mq.matches ? "dark" : "light");
}

export function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  document.documentElement.style.colorScheme = t;
}

export function initTheme() {
  const saved = localStorage.getItem(KEY);
  applyTheme(saved || (mq.matches ? "dark" : "light"));
  mq.addEventListener("change", () => { if (!localStorage.getItem(KEY)) applyTheme(mq.matches ? "dark" : "light"); });
}

export function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  localStorage.setItem(KEY, next);
  applyTheme(next);
  return next;
}
