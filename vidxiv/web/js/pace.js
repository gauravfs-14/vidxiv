/** Pace and review options shared by the script and video steps. Mirrors vidxiv/config.py (PACES, REVIEW_MAX). */
export const PACES = [
  { value: "standard", label: "Standard", sub: "Academic", lineSeconds: 5.2, help: "Full sentences and a calm rhythm. Suits an academic audience." },
  { value: "fast", label: "Fast", sub: "Tighter cuts", lineSeconds: 3.8, help: "Shorter lines, quicker animation, a new visual almost every line." },
  { value: "reel", label: "Reel", sub: "TikTok, Reels", lineSeconds: 2.6, help: "Punchy one-sentence lines, very quick animation, almost no pauses, three-word captions. Best vertical and under a minute." },
];
export const paceOf = (v) => PACES.find((p) => p.value === v) || PACES[0];

export const REVIEW_MAX = 8;
export const REVIEW = [
  { value: 0, label: "Off" },
  { value: 1, label: "1 round" },
  { value: 2, label: "2 rounds" },
  { value: 3, label: "3 rounds" },
  { value: 5, label: "5 rounds" },
  { value: REVIEW_MAX, label: "Until satisfied", sub: "up to 8" },
];
export const reviewHelp = (n, agent) => n === 0
  ? "The video is rendered once and not checked."
  : `Before the final render, ${agent === "none" ? "an agent" : agent === "auto" ? "your agent" : agent} looks at a draft and fixes broken voice, animation, sound effects and layout, ${n === REVIEW_MAX ? "until it is satisfied or 8 rounds have run" : `up to ${n} round${n === 1 ? "" : "s"}`}. Each round is a draft render and one agent run, and uses your agent's tokens.`;
