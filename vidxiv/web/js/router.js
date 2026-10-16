/** Hash routes: #/ and #/job/<id>[/<stage>]. */
export const STAGE_KEYS = ["paper", "script", "narration", "video"];

export function parseRoute() {
  const m = location.hash.match(/^#\/job\/([\w-]+)(?:\/(\w+))?/);
  if (!m) return { name: "home" };
  return { name: "job", id: m[1], stage: STAGE_KEYS.includes(m[2]) ? m[2] : null };
}

export function startRouter(onRoute) {
  const go = () => onRoute(parseRoute());
  window.addEventListener("hashchange", go);
  go();
}

export const navigate = (hash) => { location.hash = hash; };
export const jobHref = (id, stage) => `#/job/${id}${stage ? "/" + stage : ""}`;
/** Change the URL without adding a history entry or firing a route change. */
export const replaceHash = (hash) => history.replaceState(null, "", hash);
