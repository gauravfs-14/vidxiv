/** Hash routes: #/ and #/job/<id>. */
export function parseRoute() {
  const m = location.hash.match(/^#\/job\/([\w-]+)/);
  return m ? { name: "job", id: m[1] } : { name: "home" };
}

export function startRouter(onRoute) {
  const go = () => onRoute(parseRoute());
  window.addEventListener("hashchange", go);
  go();
}

export const navigate = (hash) => { location.hash = hash; };
export const jobHref = (id) => `#/job/${id}`;
