/** Thin fetch wrapper. Every failure becomes an ApiError with a message fit to show the user. */
export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body || {};
    this.problems = this.body.problems || [];
  }
}

const listeners = new Set();
let offline = false;
export function onConnection(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function setOffline(v) {
  if (v === offline) return;
  offline = v;
  listeners.forEach((fn) => fn(!v));
}

async function request(method, url, { json, form, signal } = {}) {
  const init = { method, signal, headers: {} };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (form) {
    init.body = form;
  }
  let res;
  try {
    res = await fetch(url, init);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    setOffline(true);
    throw new ApiError("Cannot reach the vidxiv server. Check that it is still running.", 0);
  }
  setOffline(false);
  const type = res.headers.get("content-type") || "";
  if (!res.ok) {
    let body = null;
    try { body = type.includes("json") ? await res.json() : null; } catch { /* not json */ }
    const detail = body && (body.error || (typeof body.detail === "string" ? body.detail : null));
    throw new ApiError(detail || `The server returned an error (${res.status}).`, res.status, body);
  }
  if (type.includes("json")) return res.json();
  return res;
}

const j = (id) => `/api/jobs/${encodeURIComponent(id)}`;

export const api = {
  status: () => request("GET", "/api/status"),
  voices: () => request("GET", "/api/voices"),
  setup: (what) => request("POST", `/api/setup/${what}`),
  listJobs: () => request("GET", "/api/jobs"),
  createJob: (form) => request("POST", "/api/jobs", { form }),
  job: (id, signal) => request("GET", j(id), { signal }),
  deleteJob: (id) => request("DELETE", j(id)),
  log: (id, stage, since) => request("GET", `${j(id)}/log/${stage}?since=${since}`),
  cancel: (id) => request("POST", `${j(id)}/cancel`),
  putBrief: (id, brief) => request("PUT", `${j(id)}/brief`, { json: brief }),
  makeScript: (id, body) => request("POST", `${j(id)}/script`, { json: body }),
  putStoryboard: (id, sb) => request("PUT", `${j(id)}/storyboard`, { json: sb }),
  putNarrationConfig: (id, cfg) => request("PUT", `${j(id)}/narration/config`, { json: cfg }),
  synthesize: (id, lines) => request("POST", `${j(id)}/narration`, { json: lines ? { lines } : {} }),
  uploadLine: (id, i, file) => {
    const f = new FormData();
    f.append("file", file, file.name || "take.webm");
    return request("POST", `${j(id)}/narration/line/${i}`, { form: f });
  },
  deleteLine: (id, i) => request("DELETE", `${j(id)}/narration/line/${i}`),
  uploadTake: (id, file, mode) => {
    const f = new FormData();
    f.append("file", file, file.name || "take.webm");
    f.append("mode", mode);
    return request("POST", `${j(id)}/narration/take`, { form: f });
  },
  preview: (id) => request("POST", `${j(id)}/preview`),
  render: (id, body) => request("POST", `${j(id)}/render`, { json: body }),
  voicePreview: async (body) => {
    const res = await request("POST", "/api/voices/preview", { json: body });
    return res.blob();
  },
};
