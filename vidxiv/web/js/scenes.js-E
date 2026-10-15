/**
 * Scene types as the editor sees them. Mirrors vidxiv/schema.py: `fields` are the one to three
 * values worth editing inline, everything else goes through the JSON editor.
 * `path` may be dotted (compare.left.title). Empty optional strings are removed from the data.
 */
export const SCENES = {
  hook: { label: "Hook", tint: "accent", figure: true, fields: [
    { path: "headline", label: "Headline", max: 90, required: true },
    { path: "sub", label: "Second line", max: 120 },
  ] },
  title: { label: "Title", tint: "ink", figure: true, fields: [
    { path: "title", label: "Paper title", max: 160, required: true },
    { path: "authors", label: "Authors", max: 160 },
    { path: "meta", label: "Meta", max: 100 },
  ] },
  concept: { label: "Concept", tint: "blue", fields: [
    { path: "heading", label: "Heading", max: 80, required: true },
    { path: "body", label: "Body", max: 220 },
  ], note: (s) => countNote(s.items, "card") },
  steps: { label: "Steps", tint: "teal", fields: [
    { path: "heading", label: "Heading", max: 80 },
  ], note: (s) => (s.steps?.length ? "Steps: " + s.steps.map((x) => x.label).join(", ") : "") },
  figure: { label: "Figure", tint: "mustard", figure: true, fields: [
    { path: "caption", label: "Caption", max: 140 },
  ] },
  stat: { label: "Stat", tint: "accent", fields: [
    { path: "value", label: "Value", max: 14, required: true, span: 2 },
    { path: "label", label: "Label", max: 80, required: true, span: 4 },
    { path: "detail", label: "Detail", max: 120 },
  ] },
  bars: { label: "Bars", tint: "teal", fields: [
    { path: "title", label: "Title", max: 80, span: 4 },
    { path: "unit", label: "Unit", max: 20, span: 2 },
  ], note: (s) => (s.bars?.length ? s.bars.map((b) => `${b.label} ${b.value}`).join(", ") : "") },
  compare: { label: "Compare", tint: "rose", fields: [
    { path: "left.title", label: "Left title", max: 50, required: true, span: 3 },
    { path: "right.title", label: "Right title", max: 50, required: true, span: 3 },
    { path: "verdict", label: "Verdict", max: 100 },
  ] },
  equation: { label: "Equation", tint: "blue", fields: [
    { path: "formula", label: "LaTeX", max: 220, required: true, mono: true },
    { path: "caption", label: "Caption", max: 140 },
  ] },
  quote: { label: "Quote", tint: "mustard", fields: [
    { path: "text", label: "Quote", max: 240, required: true },
    { path: "attribution", label: "Attribution", max: 100 },
  ] },
  outro: { label: "Outro", tint: "accent", fields: [
    { path: "headline", label: "Headline", max: 100, required: true },
    { path: "takeaway", label: "Takeaway", max: 160 },
    { path: "cta", label: "Call to action", max: 80 },
  ] },
};

export const SCENE_TYPES = Object.keys(SCENES);

function countNote(list, noun) {
  return list?.length ? `${list.length} ${noun}${list.length === 1 ? "" : "s"}, edit as JSON` : "";
}

export function getPath(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
}

export function setPath(obj, path, value, keepEmpty = false) {
  const keys = path.split(".");
  const last = keys.pop();
  let o = obj;
  for (const k of keys) o = o[k] ??= {};
  if ((value === "" || value == null) && !keepEmpty) delete o[last];
  else o[last] = value;
}

const clip = (s, n) => {
  s = (s || "").trim();
  if (s.length <= n) return s;
  const cut = s.slice(0, n - 1);
  return (cut.includes(" ") ? cut.slice(0, cut.lastIndexOf(" ")) : cut).replace(/[ ,;:.]+$/, "");
};

const firstText = (scene) => {
  for (const k of ["headline", "heading", "title", "text", "label", "formula", "caption"]) {
    if (typeof scene[k] === "string" && scene[k]) return scene[k];
  }
  return "";
};

/** A valid scene of `type` seeded from the old scene and the line's narration. */
export function defaultScene(type, oldScene, lineText, ctx) {
  const seed = firstText(oldScene || {}) || lineText || "";
  const words = clip(seed, 60);
  switch (type) {
    case "hook": return { type, headline: clip(seed, 90) };
    case "title": return { type, title: clip(ctx.title || seed, 160) };
    case "concept": return { type, heading: words };
    case "steps": return { type, heading: words, steps: [{ label: "First step" }, { label: "Second step" }] };
    case "figure": return { type, figure: ctx.figures[0]?.id || "" };
    case "stat": return { type, value: "0", label: words };
    case "bars": return { type, title: words, bars: [{ label: "Baseline", value: 1 }, { label: "This paper", value: 2, highlight: true }] };
    case "compare": return { type, left: { title: "Before", points: ["Point"] }, right: { title: "After", points: ["Point"] } };
    case "equation": return { type, formula: "x" };
    case "quote": return { type, text: clip(lineText || seed, 240) };
    case "outro": return { type, headline: clip(seed, 100) };
    default: return { type };
  }
}
